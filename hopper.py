import pyhopper
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch.nn.functional as F
import os
import json
import copy
from sklearn.model_selection import KFold
from lib.AlmondDataset import AlmondDataset
from lib.transforms import Standardize as StandardizeTransform


# Function to read scaling parameters from a JSON file
def read_scaling_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


# Function to impute NaNs and nulls with 0
def impute(sample):
    sample.fillna(0, inplace=True)
    return sample


# Function to transform data by applying standardization and imputation
def transform(sample, scaling_params):
    sample = StandardizeTransform(scaling_params)(sample)
    sample = impute(sample)
    return sample


# Function to initialize data objects
def get_data_objects(sub_dir, scaling_params, batch_size):
    transform_fn = lambda sample: transform(sample, scaling_params)
    almonds_train = AlmondDataset(os.path.join('data', sub_dir),
                                  is_train=True, 
                                  transform=transform_fn)
    almonds_test = AlmondDataset(os.path.join('data', sub_dir),
                                 is_train=False,
                                 transform=transform_fn)
    return almonds_train, almonds_test


# Define the SimpleNet model
class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes, hl_sizes=[128, 128, 128, 128], use_bn=True, dropout_rate=0.0):
        super(SimpleNet, self).__init__()
        assert len(hl_sizes) > 0, "hl_sizes must have at least one hidden layer size"
        self.layers = nn.ModuleList()
        self.use_bn = use_bn
        self.use_dropout = dropout_rate > 0.0
        
        prev_size = input_size
        for hl_size in hl_sizes:
            self.layers.append(nn.Linear(prev_size, hl_size))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(hl_size))
            self.layers.append(nn.ReLU(inplace=True))
            if self.use_dropout:
                self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hl_size
        
        self.out = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Training function for one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# Function to initialize model, criterion, and optimizer
def initialize_model(input_size, num_classes, optimizer_config, hl_sizes, use_bn, dropout_rate, device):
    model = SimpleNet(input_size, num_classes, hl_sizes, use_bn, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_config['name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=optimizer_config['lr'],
                              momentum=optimizer_config['momentum'],
                              weight_decay=optimizer_config['weight_decay'])
    elif optimizer_config['name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=optimizer_config['lr'],
                               betas=(optimizer_config['beta1'], optimizer_config['beta2']),
                               weight_decay=optimizer_config['weight_decay'])
    return model, criterion, optimizer


# Function to run K-Fold Cross-Validation with Early Stopping
def run_k_fold_cv(almonds_train, num_epochs, batch_size, optimizer_config, hl_sizes, use_bn, dropout_rate, k_folds, device, patience=15):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_accuracies = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(almonds_train)):
        train_subset = Subset(almonds_train, train_ids)
        val_subset = Subset(almonds_train, val_ids)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        model, criterion, optimizer = initialize_model(almonds_train.num_features(), almonds_train.num_classes(),
                                                       optimizer_config, hl_sizes, use_bn, dropout_rate, device)
        best_val_accuracy = 0.0
        epochs_no_improve = 0  # To track early stopping

        for epoch in range(num_epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_accuracy = evaluate(model, val_loader, device)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0  # Reset counter if validation improves
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break
        
        fold_accuracies.append(best_val_accuracy)
        if best_val_accuracy < 75.0:
            break  # Early stopping if validation accuracy is less than 75.0

    avg_val_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    return avg_val_accuracy


# Objective function for PyHopper
def objective(params):
    # Load datasets
    scaling_params_path = 'data/basic/scaling_params.json'
    scaling_params = read_scaling_params(scaling_params_path)
    almonds_train, almonds_test = get_data_objects(sub_dir='basic', scaling_params=scaling_params, batch_size=params['batch_size'])
    
    # Set device based on batch size
    device = torch.device('cpu') if params['batch_size'] < 128 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run K-fold cross-validation with the provided network architecture parameters
    avg_val_accuracy = run_k_fold_cv(almonds_train, params['epochs'], params['batch_size'], params['optimizer'],
                                     params['hl_sizes'], params['use_bn'], params['dropout_rate'],
                                     k_folds=5, device=device)
    
    return avg_val_accuracy


# Main function using PyHopper
def main():
    # Define the PyHopper search space with network architecture parameters
    search = pyhopper.Search({
        'epochs': pyhopper.choice(200),
        'batch_size': pyhopper.choice(64, 128),
        'optimizer': pyhopper.choice(
            {'name': 'SGD', 'lr': pyhopper.float(0.001, 0.1), 'momentum': pyhopper.float(0.09, 0.9), 'nesterov': pyhopper.choice(True, False), 'weight_decay': pyhopper.choice(0.0001, 0.0005, 0.001, 0.01, is_ordinal=True)},
            {'name': 'Adam', 'lr': pyhopper.float(0.001, 0.1), 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': pyhopper.choice(0.0001, 0.0005, 0.001, 0.01, is_ordinal=True)}
        ),
        'hl_sizes': pyhopper.choice(
            [16, 16],  # Very small architecture
            [256], 
            [64, 64],  # Small architecture
            [128, 128, 128],  # Medium architecture
            [128, 128, 128, 128],  # Medium architecture
            [256, 256, 256, 256]  # Large architecture
        ),
        'use_bn': pyhopper.choice(True, False),  # Whether to use BatchNorm
        'dropout_rate': pyhopper.choice(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, is_ordinal=True)  # Dropout rate
    })
    
    # Run the PyHopper search
    search.run(objective, direction="maximize", runtime="1h", n_jobs=2)

    print(type(search.history))
    print(list(search.history))
    print(list(search.history.fs))

    history = {
        'params': list(search.history),
        'fs': list(search.history.fs),
    }


    import numpy as np
    # save the history to a file
    with open('history.json', 'w') as f:
        json.dump(history, f, indent=4)

    # print the top 5 configurations
    # find the indices of the best configurations
    np_fs = np.array(search.history.fs)
    ind = np.argpartition(np_fs, -5)[-5:]
    print(ind)

    for idx in ind:
        print(list(search.history)[idx], list(search.history.fs)[idx])
        print()


if __name__ == '__main__':
    main()

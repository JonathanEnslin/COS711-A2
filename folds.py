import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch.nn.functional as F

import os
import json
import time

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


# Define the SimpleNet model
class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()
    return running_loss / len(train_loader)


# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
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


# Function to initialize model, criterion, and optimizer
def initialize_model(input_size, num_classes, learning_rate, device):
    model = SimpleNet(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    return model, criterion, optimizer


# Function to run K-Fold Cross-Validation
def run_k_fold_cv(almonds_train, num_epochs, batch_size, learning_rate, k_folds, device):
    best_accuracy = 0.0
    best_model_state = None
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(almonds_train)):
        print(f'FOLD {fold+1}/{k_folds}')
        
        # Create training and validation subsets
        train_subset = Subset(almonds_train, train_ids)
        val_subset = Subset(almonds_train, val_ids)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize model, optimizer, and loss function
        model, criterion, optimizer = initialize_model(almonds_train.num_features(), almonds_train.num_classes(), learning_rate, device)
        print(f'Number of parameters in the model: {model.get_num_params()}')

        # Train the model
        for epoch in range(num_epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_accuracy = evaluate(model, val_loader, device)
            print(f'FOLD {fold+1}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Save the best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict()

    return best_model_state, best_accuracy


# Function to save and evaluate the best model on the test set
def evaluate_best_model(model, best_model_state, test_loader, device):
    model.load_state_dict(best_model_state)
    test_accuracy = evaluate(model, test_loader, device)
    print(f'Test Accuracy with best model: {test_accuracy:.2f}%')
    return test_accuracy


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    k_folds = 5
    scaling_params_path = 'data/basic/scaling_params.json'

    # Load scaling parameters
    scaling_params = read_scaling_params(scaling_params_path)

    # Set device (use GPU if available and batch size >= 128)
    device = torch.device('cuda' if torch.cuda.is_available() and batch_size >= 128 else 'cpu')
    print(f'Using device: {device}')

    # Get datasets
    almonds_train, almonds_test = get_data_objects(sub_dir='basic', scaling_params=scaling_params, batch_size=batch_size)

    # Run K-Fold Cross-Validation
    best_model_state, best_accuracy = run_k_fold_cv(almonds_train, num_epochs, batch_size, learning_rate, k_folds, device)

    # Evaluate best model on the test set
    test_loader = DataLoader(almonds_test, batch_size=batch_size, shuffle=False, num_workers=0)
    evaluate_best_model(SimpleNet(almonds_train.num_features(), almonds_train.num_classes()).to(device), best_model_state, test_loader, device)

    print(f'Best Validation Accuracy: {best_accuracy:.2f}%')


if __name__ == '__main__':
    main()

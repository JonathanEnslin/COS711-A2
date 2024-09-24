import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch import nn
import torch.nn.functional as F

import os
import json
import time
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


# Define the SimpleNet model with more configurability
class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes, hl_sizes=[256, 256, 256, 256], 
                 use_bn=True, dropout_rate=0.0):
        """
        input_size: int, number of input features
        num_classes: int, number of output classes
        hl_sizes: list of ints, number of neurons in each hidden layer
        use_bn: bool, whether to use Batch Normalization
        dropout_rate: float, dropout rate (0.0 means no dropout)
        """
        super(SimpleNet, self).__init__()
        
        # Check that hl_sizes has at least one hidden layer size
        assert len(hl_sizes) > 0, "hl_sizes must have at least one hidden layer size"
        
        # Create layers dynamically based on the hl_sizes list
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
        
        # Output layer
        self.out = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        # Pass through all hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        x = self.out(x)
        return x

    def get_num_params(self):
        """ Returns the total number of trainable parameters """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Training function for one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # print(f'Batch {batch_idx+1}/{len(train_loader)}')
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
def initialize_model(input_size, num_classes, optimizer_config, device):
    model = SimpleNet(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Loss function

    # Initialize optimizer based on the current configuration
    if 'SGD' in optimizer_config:
        optimizer = optim.SGD(model.parameters(), lr=optimizer_config['SGD']['lr'], 
                              momentum=optimizer_config['SGD']['momentum'], 
                              weight_decay=optimizer_config['SGD']['weight_decay'])
    elif 'Adam' in optimizer_config:
        optimizer = optim.Adam(model.parameters(), lr=optimizer_config['Adam']['lr'],
                               betas=(optimizer_config['Adam']['beta1'], optimizer_config['Adam']['beta2']),
                               weight_decay=optimizer_config['Adam']['weight_decay'])
    elif 'Rprop' in optimizer_config:
        optimizer = optim.Rprop(model.parameters(), lr=optimizer_config['Rprop']['lr'], 
                                etas=optimizer_config['Rprop']['etas'], 
                                step_sizes=optimizer_config['Rprop']['step_sizes'])
    
    return model, criterion, optimizer


# Function to run K-Fold Cross-Validation with Early Stopping
def run_k_fold_cv(almonds_train, num_epochs, batch_size, optimizer_config, k_folds, device, patience=15, best_so_far=None):
    best_model_state = None
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    fold_accuracies = []  # Store validation accuracies for each fold
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(almonds_train)):
        print(f'FOLD {fold+1}/{k_folds}')
        
        # Create training and validation subsets
        train_subset = Subset(almonds_train, train_ids)
        val_subset = Subset(almonds_train, val_ids)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize model, optimizer, and loss function
        model, criterion, optimizer = initialize_model(almonds_train.num_features(), almonds_train.num_classes(), optimizer_config, device)
        print(f'Number of parameters in the model: {model.get_num_params()}')

        best_val_accuracy = 0.0
        epochs_no_improve = 0  # Tracks how many epochs have passed without improvement

        for epoch in range(num_epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_accuracy = evaluate(model, val_loader, device)
            print(f'FOLD {fold+1}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            # Early stopping condition
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0  # Reset the counter if validation improves
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} due to no improvement in {patience} epochs.')
                break

        fold_accuracies.append(best_val_accuracy)

        if best_so_far is not None and (best_val_accuracy < best_so_far - 5 or best_val_accuracy < 75.0):
            print(f'Pruning run due to poor performance')
            break

    # Calculate the average validation accuracy across all folds
    avg_val_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    
    return avg_val_accuracy


# Function to save and evaluate the best model on the test set
def evaluate_best_model(model, best_model_state, test_loader, device):
    model.load_state_dict(best_model_state)
    test_accuracy = evaluate(model, test_loader, device)
    print(f'Test Accuracy with best model: {test_accuracy:.2f}%')
    return test_accuracy


# Search space functions
def recursive_build_search_space(option_space, search_space, current_combination={}):
    if len(option_space.keys()) == 0:
        search_space.append(current_combination)
        return

    opt_name = list(option_space.keys())[0]
    trimmed_option_space = copy.deepcopy(option_space)
    del trimmed_option_space[opt_name]

    current_option_space = option_space[opt_name]

    if isinstance(current_option_space, dict):
        for choice in current_option_space.keys():
            choice_search_space = []
            choice_option_space = current_option_space[choice]
            recursive_build_search_space(choice_option_space, choice_search_space)
            for choice_opt_val in choice_search_space:
                combination = copy.deepcopy(current_combination)
                combination[opt_name] = {choice: choice_opt_val}
                recursive_build_search_space(trimmed_option_space, search_space, combination)
    else:
        for opt_val in current_option_space:
            combination = copy.deepcopy(current_combination)
            combination[opt_name] = opt_val
            recursive_build_search_space(trimmed_option_space, search_space, combination)


def build_search_space(option_space):
    search_space = []
    recursive_build_search_space(option_space=option_space, search_space=search_space)
    return search_space


# Main function to run the search and K-fold CV
def main():
    # Search space
    search_space = {
        'epochs': [200],
        'batch_size': [512, 256],
        'optimizer': {
            # 'SGD': {
            #     'lr': [0.001, 0.01, 0.1],
            #     'momentum': [0.09, 0.9],
            #     'weight_decay': [0.001],
            # },
            # 'Adam': {
            #     'lr': [0.001, 0.01, 0.1],
            #     'beta1': [0.9],
            #     'beta2': [0.999],
            #     'weight_decay': [0.001],
            # },
            'Rprop': {
                'lr': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'etas': [(0.5, 1.2), (0.2, 1.5), (0.1, 1.7), (0.9, 1.1)],
                'step_sizes': [(1e-06, 0.1), (1e-05, 0.1), (1e-04, 0.1), (1e-03, 0.1)],
            },
        },
    }

    search_space = build_search_space(search_space)

    scaling_params_path = 'data/basic/scaling_params.json'
    scaling_params = read_scaling_params(scaling_params_path)



    # Loop over search space combinations
    best_combination = None
    best_avg_val_accuracy = 0.0
    for config_idx, config in enumerate(search_space):
        print(f"Running configuration: {config}")
        print(f"{config_idx+1}/{len(search_space)}")

        # Extract parameters from the config
        num_epochs = config['epochs']
        batch_size = config['batch_size']
        optimizer_config = config['optimizer']

        # Get datasets
        almonds_train, almonds_test = get_data_objects(sub_dir='basic', scaling_params=scaling_params, batch_size=batch_size)

        # Set device
        device = torch.device('cpu') if batch_size < 128 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        # Run K-fold CV for the current configuration
        avg_val_accuracy = run_k_fold_cv(almonds_train, num_epochs, batch_size, optimizer_config, k_folds=5, device=device, best_so_far=best_avg_val_accuracy)

        # Track the best combination based on average validation accuracy
        if avg_val_accuracy > best_avg_val_accuracy:
            best_avg_val_accuracy = avg_val_accuracy
            best_combination = config

        print(f"Current Best Avg Val Accuracy: {best_avg_val_accuracy:.2f}% with configuration: {best_combination}")

    # Final output of the best configuration
    print(f"Best configuration found: {best_combination} with Avg Val Accuracy: {best_avg_val_accuracy:.2f}%")

    # Optional: Evaluate on the test set using best configuration if desired


if __name__ == '__main__':
    main()

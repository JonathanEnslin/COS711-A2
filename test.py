import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import os
import json
import time

from sklearn.model_selection import KFold

from lib.AlmondDataset import AlmondDataset
from lib.transforms import Standardize as StandardizeTransform
from lib.transforms import ScaleTo_0p1_0p9, ScaleTo_neg0p9_0p9


scaling_params = json.load(open(r'data/transformed/scaling_params.json', 'r'))
def impute(sample):
    # impute NaNs and nulls with 0s
    sample.fillna(0, inplace=True)
    return sample

def transform(sample):
    # sample = ScaleTo_neg0p9_0p9(scaling_params)(sample)
    # sample = ScaleTo_0p1_0p9(scaling_params)(sample)
    sample = StandardizeTransform(scaling_params)(sample)
    sample = impute(sample)
    return sample

# Define the SimpleNet model with more configurability
class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes, hl_sizes=[128, 128, 128, 128], 
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
            self.layers.append(nn.ReLU())
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


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """Evaluate the model on the test set."""
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

def get_data_objects(sub_dir='transformed', batch_size=64):
    almonds_train = AlmondDataset(os.path.join('data', sub_dir),
                        is_train=True, 
                        transform=transform)

    almonds_test = AlmondDataset(os.path.join('data', sub_dir),
                            is_train=False,
                            transform=transform)
    
        # Create data loaders for training and testing datasets
    train_loader = DataLoader(almonds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(almonds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return almonds_train, almonds_test, train_loader, test_loader


def main():
    # device = torch.device('cpu')

    # Hyperparameters
    batch_size = 512
    learning_rate = 0.010793033310635212
    num_epochs = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if batch_size >= 128 else torch.device('cpu')
    print(f'Using device: {device}')

    almonds_train, almonds_test, train_loader, test_loader = get_data_objects(sub_dir='basic', batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = SimpleNet(almonds_train.num_features(), almonds_train.num_classes(), hl_sizes=[64, 64, 64, 64], use_bn=True, dropout_rate=0.0)
    model = model.to(device)
    print(f'Number of parameters in the model: {model.get_num_params()}')
    
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, nesterov=False, momentum=0.8047999511195054, weight_decay=0.0001)  # SGD optimizer
    # optimizer = optim.Rprop(model.parameters(), lr=0.01)
    # Initialize the scheduler to decrease the learning rate by gamma=0.5 after epoch 160
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160], gamma=0.1)

    # Evaluate initial performance before training
    initial_accuracy = evaluate(model, test_loader, device)
    print(f'Initial accuracy on test set: {initial_accuracy:.2f}%')

    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)

        # Step the scheduler
        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    end_time = time.time()

    print('Training finished. Time taken:', f"{(end_time - start_time):.2f} seconds")

if __name__ == '__main__':
    main()
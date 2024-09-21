import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import os
import json
import time

from sklearn.model_selection import KFold

from lib.AlmondDataset import AlmondDataset
from lib.transforms import Standardize as StandardizeTransform

scaling_params = json.load(open(r'data/basic/scaling_params.json', 'r'))
def impute(sample):
    # impute NaNs and nulls with 0s
    sample.fillna(0, inplace=True)
    return sample

def transform(sample):
    sample = StandardizeTransform(scaling_params)(sample)
    sample = impute(sample)
    return sample

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

def get_data_objects(sub_dir='basic', batch_size=64):
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
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if batch_size >= 128 else torch.device('cpu')
    print(f'Using device: {device}')

    almonds_train, almonds_test, train_loader, test_loader = get_data_objects(sub_dir='basic', batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = SimpleNet(almonds_train.num_features(), almonds_train.num_classes())
    model = model.to(device)
    print(f'Number of parameters in the model: {model.get_num_params()}')
    
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Evaluate initial performance before training
    initial_accuracy = evaluate(model, test_loader, device)
    print(f'Initial accuracy on test set: {initial_accuracy:.2f}%')

    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    end_time = time.time()

    print('Training finished. Time taken:', f"{(end_time - start_time):.2f} seconds")

if __name__ == '__main__':
    main()
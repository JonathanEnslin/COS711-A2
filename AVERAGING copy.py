import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Define a small network with BatchNorm and Dropout layers
class SimpleNetWithBNDropout(nn.Module):
    def __init__(self):
        super(SimpleNetWithBNDropout, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.bn1 = nn.BatchNorm1d(2)  # BatchNorm layer
        self.dropout = nn.Dropout(0.5)  # Dropout layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply BatchNorm
        x = self.dropout(x)  # Apply Dropout
        return x

# Mock dataset (2D input and 2D target)
input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

# Loss function
loss_fn = nn.MSELoss()

# Initialize the model
model = SimpleNetWithBNDropout()

# Set model to training mode
model.train()

# Two different optimizers: Adam and SGD
optimizer1 = optim.Adam(model.parameters(), lr=0.01)
optimizer2 = optim.SGD(model.parameters(), lr=0.01)
# optimizer1, optimizer2 = optimizer2, optimizer1

# Helper function to print tensors as NumPy arrays
def print_numpy_weights(model, msg):
    print(f"{msg}:")
    for name, param in model.named_parameters():
        print(f"  {name}:")
        print(param.detach().cpu().numpy())
    print("=" * 45)  # Add a separator for better readability

# Helper function to print all model parameters, including BatchNorm stats (running_mean, running_var)
def print_numpy_weights(model, msg):
    print(f"{msg}:")
    for name, param in model.state_dict().items():  # Iterate over state_dict, which includes all parameters
        print(f"==>{name}:")
        print(param.detach().cpu().numpy())
    print("=" * 45)  # Add a separator for better readability

def average_optimizer_updates(model, optimizer1, optimizer2, loss_fn, input_data, target):
    print_numpy_weights(model, "Original weights")
    print('-'*45)
    # Step 1: Forward pass and compute loss
    output = model(input_data)
    loss = loss_fn(output, target)

    # Step 2: Save the preupdated weights
    pre_update_weights = copy.deepcopy(model.state_dict())
    print_numpy_weights(model, "Weights after forward before any update")
    print('-'*45)
    # Step 3: Compute gradients (once)
    optimizer1.zero_grad()  # Clear gradients
    optimizer2.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagate to compute gradients
    
    print_numpy_weights(model, "Weights after backward before any update")
    print('-'*45)
    # Step 4: Apply the first optimizer's update
    optimizer1.step()  # Apply gradient update (Adam)
    print_numpy_weights(model, "Weights after Adam update")
    
    # Step 5: Calculate and print delta for Adam update
    delta_adam = {}
    for key in pre_update_weights.keys():
        delta_adam[key] = model.state_dict()[key] - pre_update_weights[key]
        print(f"  Delta {key} (Adam):")
        print(delta_adam[key].detach().cpu().numpy())
    print("=" * 45)  # Add separator after Adam update
    
    # Store the updated weights after optimizer1 update
    weights_after_optimizer1 = copy.deepcopy(model.state_dict())
    
    # Step 6: Revert the model back to its original state
    model.load_state_dict(pre_update_weights)
    print_numpy_weights(model, "Reverted to original weights")
    
    # Step 7: Apply the second optimizer's update using the same gradients
    optimizer2.step()  # Apply gradient update (SGD)
    print_numpy_weights(model, "Weights after SGD update")
    
    # Step 8: Calculate and print delta for SGD update
    delta_sgd = {}
    for key in pre_update_weights.keys():
        delta_sgd[key] = model.state_dict()[key] - pre_update_weights[key]
        print(f"  Delta {key} (SGD):")
        print(delta_sgd[key].detach().cpu().numpy())
    print("=" * 45)  # Add separator after SGD update
    
    # Step 9: Average the weights of optimizer1 and optimizer2
    weights_after_optimizer2 = model.state_dict()
    
    final_weights = {}
    for key in pre_update_weights.keys():
        final_weights[key] = (weights_after_optimizer1[key] + weights_after_optimizer2[key]) / 2.0
        print(f"  Averaged {key}:")
        print(final_weights[key].detach().cpu().numpy())
    
    # Step 10: Load the averaged weights back into the model
    model.load_state_dict(final_weights)
    print_numpy_weights(model, "Final averaged weights")

# Example: Run the function
average_optimizer_updates(model, optimizer1, optimizer2, loss_fn, input_data, target)

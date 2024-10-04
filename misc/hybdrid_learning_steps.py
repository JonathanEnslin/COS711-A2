
# Training function for one epoch with multiple optimizers
def train_one_epoch_multi_optimizers(model, train_loader, criterion, optimizers, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Zero gradients for all optimizers
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Backpropagate
        loss.backward()

        # Apply optimizers' updates and average weights
        pre_update_weights = copy.deepcopy(model.state_dict())
        optimizer_updates = []

        for optimizer in optimizers:
            optimizer.step()
            optimizer_updates.append(copy.deepcopy(model.state_dict()))
            model.load_state_dict(pre_update_weights)  # Revert to original state after each update

        # Average the updates across all optimizers
        final_weights = {key: sum([optimizer_updates[i][key] for i in range(len(optimizers))]) / len(optimizers) for key in pre_update_weights.keys()}

        # Load the averaged weights back into the model
        model.load_state_dict(final_weights)

        running_loss += loss.item()

    return running_loss / len(train_loader)


def train_one_epoch_multi_optimizers(model, train_loader, criterion, optimizers, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Zero gradients for all optimizers
        for optimizer in optimizers:
            optimizer.zero_grad()

        # Backpropagate
        loss.backward()

        # Store pre-update parameters
        pre_update_params = [param.data.clone() for param in model.parameters()]

        # Apply each optimizer's updates
        optimizer_updates = []
        for optimizer in optimizers:
            # Apply the optimizer's update
            optimizer.step()
            # Store the updated parameters
            optimizer_updates.append([param.data.clone() for param in model.parameters()])

            # Revert model parameters to pre-update state
            for param, pre_update in zip(model.parameters(), pre_update_params):
                param.data.copy_(pre_update)

        # Average the updates across all optimizers
        for idx, param in enumerate(model.parameters()):
            avg_update = sum([optimizer_updates[i][idx] for i in range(len(optimizers))]) / len(optimizers)
            param.data.copy_(avg_update)

        running_loss += loss.item()

    return running_loss / len(train_loader)
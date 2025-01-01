import torch
import torch.nn as nn
from torch.optim import SGD

def train_model(model, train_loader, device, learning_rate=1e-2, training_epochs=3, weight_decay=5*1e-4, momentum=0.9):
    """
    Train the model.

    Args:
        model: The PyTorch model to be trained.
        train_loader: DataLoader for training data.
        device: The device to use for training (e.g., "cuda" or "cpu").
        learning_rate: Learning rate for the optimizer.
        training_epochs: Number of training epochs.
        weight_decay: L2 regularization coefficient.
        momentum: Momentum factor for SGD.
    """
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    total_batch = len(train_loader)
    print('Learning started. It takes some time.')

    for epoch in range(training_epochs):
        avg_cost = 0
        model.train()

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)

            # Forward and backward propagation
            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print(f'[Epoch: {epoch + 1:>4}] cost = {avg_cost:.9f}')

    print('Learning Finished!')

def save_model(model, path="./model.pth"):
    """
    Save the trained model to a file.

    Args:
        model: The PyTorch model to be saved.
        path: Path to save the model file.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

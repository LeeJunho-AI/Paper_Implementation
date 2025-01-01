import torch

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model and calculate accuracy.

    Args:
        model: The trained PyTorch model to evaluate.
        test_dataset: Dataset for testing.
        device: Device to use for evaluation (e.g., "cuda" or "cpu").
        batch_size: Batch size for testing.

    Returns:
        accuracy: The overall accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for X_test, Y_test in test_loader:
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            # Get predictions
            prediction = model(X_test)
            correct_prediction = torch.argmax(prediction, 1) == Y_test

            # Update correct and total counts
            correct += correct_prediction.sum().item()
            total += Y_test.size(0)

    # Calculate accuracy
    accuracy = correct / total
    print('Accuracy:', accuracy)
    return accuracy

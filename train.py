import torch
import sys
import pandas as pd

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch=None, total_epochs=None):
    """
    Trains the model for one epoch on the training dataset.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Dataloader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        criterion (loss function): Loss function (e.g. CrossEntropyLoss).
        device (torch.device): Device to run the training on (CPU or GPU).
        epoch (int, optional): Current epoch number (for printing).
        total_epochs (int, optional): Total number of epochs (for printing).

    Returns:
        float: Training accuracy for this epoch.
    """
    model.train()  # Set model to training mode
    running_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Clear previous gradients
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()               # Backpropagation
        optimizer.step()              # Update model weights

        # Compute number of correct predictions in batch
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Optional: print progress per batch
        if epoch is not None and total_epochs is not None:
            progress_str = f"\rEpoch {epoch+1}/{total_epochs} | Batch {batch_idx}/{len(train_loader)}"
            sys.stdout.write(progress_str)
            sys.stdout.flush()

    print()  # Print newline after progress
    return running_correct / total_samples  # Return accuracy for this epoch


def evaluate(model, data_loader, criterion, device):
    """
    Evaluates the model on the given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): Dataloader for validation or test set.
        criterion (loss function): Loss function (not used in current implementation).
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Accuracy on the dataset.
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total  # Return accuracy


def get_best_hyperparameters(filepath, hyperparam_keys, metric='test_acc'):
    """
    Loads a CSV or Excel file with trial results and returns the best hyperparameter config.

    Args:
        filepath (str): Path to CSV or Excel file containing trial results.
        hyperparam_keys (list): List of hyperparameter keys to extract.
        metric (str): Column name to maximize (e.g., 'test_acc' or 'val_acc').

    Returns:
        dict: Dictionary of best hyperparameter values.
    """
    # Load DataFrame depending on file extension
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx.")

    # Ensure the target metric exists
    if metric not in df.columns:
        raise ValueError(f"'{metric}' not found in file. Available columns: {list(df.columns)}")

    # Select the row with the highest metric value
    best_row = df.sort_values(by=metric, ascending=False).iloc[0]

    # Extract relevant hyperparameters into a dictionary
    return {key: best_row[key] for key in hyperparam_keys if key in best_row}

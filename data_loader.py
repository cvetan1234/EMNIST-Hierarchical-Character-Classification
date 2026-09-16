import torch
import numpy as np
from torch.utils.data import Subset, random_split, DataLoader

class RemappedDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that remaps EMNIST class labels to a contiguous 0–35 range.

    Expected input: a dataset with labels from EMNIST ByClass format.
    It remaps:
    - Digits 0–9  → 0–9
    - Uppercase A–M (original labels 10–22) → 10–22 → 10–22
    - Lowercase a–m (original labels 36–48) → 23–35
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

        # Define a mapping from original EMNIST label to new 0–35 label
        self.mapping_table = {orig: new for new, orig in enumerate(
            list(range(10)) + list(range(10, 23)) + list(range(36, 49))
        )}

    def __getitem__(self, idx):
        """
        Returns:
            Tuple[image tensor, remapped label]
        """
        img, label = self.base_dataset[idx]
        return img, self.mapping_table[label.item()]

    def __len__(self):
        """
        Returns:
            Total number of samples in the dataset.
        """
        return len(self.base_dataset)


class TM2Dataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that converts EMNIST class labels into three superclasses:
    0 → digit (0–9)
    1 → uppercase letter (A–Z, 10–35)
    2 → lowercase letter (a–z, 36–61)

    Use this for training the TM2 classifier.
    """
    def __init__(self, base_dataset):
        self.data = base_dataset

    def __getitem__(self, idx):
        """
        Returns:
            Tuple[image tensor, superclass label]
        """
        x, y = self.data[idx]
        y = y.item()

        if y < 10:
            return x, 0  # Digit
        elif 10 <= y < 36:
            return x, 1  # Uppercase letter
        elif 36 <= y < 62:
            return x, 2  # Lowercase letter
        else:
            raise ValueError(f"Label out of range: {y}")

    def __len__(self):
        """
        Returns:
            Total number of samples in the dataset.
        """
        return len(self.data)


def load_emnist_datasets(train_path, test_path, seed=42):
    """
    Loads EMNIST training and test datasets from .pt files.

    Args:
        train_path (str): Path to training dataset file.
        test_path (str): Path to test dataset file.
        seed (int): Unused (reserved for future consistency).

    Returns:
        Tuple[train_dataset, test_dataset]
    """
    train_dataset = torch.load(train_path, weights_only=False)
    test_dataset = torch.load(test_path, weights_only=False)
    return train_dataset, test_dataset


def get_train_val_split(full_dataset, train_fraction=1.0, val_split=0.2, seed=42):
    """
    Randomly splits a dataset into training and validation sets.

    Args:
        full_dataset (Dataset): The full dataset to split.
        train_fraction (float): Fraction of full dataset to use (e.g., 0.1 for quick test).
        val_split (float): Fraction of the selected data to use as validation.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[train_dataset, val_dataset]
    """
    # Select a fraction of the dataset
    use_len = int(len(full_dataset) * train_fraction)

    # Reproducible random subset of indices
    selected_indices = np.random.RandomState(seed).choice(len(full_dataset), use_len, replace=False)
    reduced_dataset = Subset(full_dataset, selected_indices)

    # Split into train and validation
    train_size = int((1 - val_split) * len(reduced_dataset))
    val_size = len(reduced_dataset) - train_size

    train_dataset, val_dataset = random_split(
        reduced_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, val_dataset


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Wraps datasets into PyTorch DataLoaders.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        batch_size (int): Batch size for loading.

    Returns:
        Tuple[train_loader, val_loader, test_loader]
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

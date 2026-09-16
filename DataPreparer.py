import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, TensorDataset
import torchvision.transforms.functional as TF
import random

class DataPreparer:
    """
    Prepares a balanced and optionally label-remapped version of the EMNIST ByClass dataset
    containing only 36 target classes: digits (0–9), uppercase (A–M), lowercase (a–m).

    Includes data augmentation for undersampled classes.
    """

    def __init__(self, samples_per_class_train=5000, samples_per_class_test=1000, seed=42):
        """
        Initializes the preparer.

        Args:
            samples_per_class_train (int): Desired number of samples per class in the training set.
            samples_per_class_test (int): Desired number of samples per class in the test set.
            seed (int): Random seed for reproducibility.
        """
        self.samples_per_class_train = samples_per_class_train
        self.samples_per_class_test = samples_per_class_test
        self.seed = seed

        # Define class index ranges from EMNIST ByClass: digits, uppercase A–M, lowercase a–m
        self.DIGITS = list(range(10))         # 0–9
        self.UPPERCASE = list(range(10, 23))  # A–M
        self.LOWERCASE = list(range(36, 49))  # a–m
        self.TARGET_CLASSES = self.DIGITS + self.UPPERCASE + self.LOWERCASE

    def _get_balanced_subset(self, dataset, samples_per_class):
        """
        Selects a balanced subset from the given dataset.

        Args:
            dataset (Dataset): Original EMNIST dataset.
            samples_per_class (int): Number of samples per target class to select.

        Returns:
            Subset of the dataset with approximately equal class representation.
        """
        random.seed(self.seed)
        labels = dataset.targets.numpy()
        selected_indices = []

        for cls in self.TARGET_CLASSES:
            # Find indices for current class
            class_indices = (labels == cls).nonzero()[0]
            # Select up to `samples_per_class` examples from the class
            selected = random.sample(list(class_indices), min(samples_per_class, len(class_indices)))
            selected_indices.extend(selected)

        return Subset(dataset, selected_indices)

    def _fast_affine(self, tensor_img):
        """
        Applies a fast affine transformation (rotation, translation, scaling) to the image.

        Args:
            tensor_img (Tensor): 1x28x28 grayscale image tensor.

        Returns:
            Transformed image tensor.
        """
        angle = random.uniform(-10, 10)  # random rotation
        translate = [
            random.uniform(-0.1, 0.1) * tensor_img.shape[1],  # x translation
            random.uniform(-0.1, 0.1) * tensor_img.shape[2]   # y translation
        ]
        scale = random.uniform(0.9, 1.1)  # random scaling
        return TF.affine(tensor_img, angle=angle, translate=translate, scale=scale, shear=0)

    def _augment_class(self, dataset, label, needed):
        """
        Augments `needed` number of samples for a specific class.

        Args:
            dataset (List): List of (image, label) tuples.
            label (int): Label for which to generate more samples.
            needed (int): Number of samples to generate.

        Returns:
            List of (augmented_image, label) tuples.
        """
        samples = [(img, lbl) for img, lbl in dataset if lbl == label]
        if not samples:
            print(f"Skipping class {label} — no samples found.")
            return []

        # Generate new samples using random augmentation
        return [(self._fast_affine(random.choice(samples)[0]), label) for _ in range(needed)]

    def _convert_subset(self, subset):
        """
        Converts a PyTorch Subset to a list of (image, label) pairs.

        Args:
            subset (Subset): Subset of a dataset.

        Returns:
            List[(Tensor, int)]
        """
        return list(subset)

    def _remap_labels(self, data_list):
        """
        Creates a mapping from original labels to range [0, 35].

        Args:
            data_list: List of (image, original_label) pairs.

        Returns:
            Dict[int, int]: Mapping table from original label → new label.
        """
        remapped_classes = self.TARGET_CLASSES
        return {orig: new for new, orig in enumerate(remapped_classes)}

    def prepare(self, remap_labels=True):
        """
        Prepares balanced, optionally remapped training and test datasets from EMNIST.

        Args:
            remap_labels (bool): If True, remap the class labels to [0, 35].

        Returns:
            Tuple[TensorDataset, TensorDataset]: Final (train_dataset, test_dataset)
        """
        # Load EMNIST datasets (ByClass split), automatically download if not found
        transform = transforms.ToTensor()
        train_raw = torchvision.datasets.EMNIST(
            root='./data', split='byclass', train=True, download=True, transform=transform
        )
        test_raw = torchvision.datasets.EMNIST(
            root='./data', split='byclass', train=False, download=True, transform=transform
        )

        # Select balanced subset for each set
        train_subset = self._get_balanced_subset(train_raw, self.samples_per_class_train)
        test_subset = self._get_balanced_subset(test_raw, self.samples_per_class_test)

        train_list = self._convert_subset(train_subset)
        test_list = self._convert_subset(test_subset)

        # Augment training data if some classes have too few samples
        print("Augmenting train data...")
        for label in self.TARGET_CLASSES:
            count = sum(1 for _, lbl in train_list if lbl == label)
            if count < self.samples_per_class_train:
                train_list.extend(self._augment_class(train_list, label, self.samples_per_class_train - count))

        # Augment test data similarly (not always common, but here for balance)
        print("Augmenting test data...")
        for label in self.TARGET_CLASSES:
            count = sum(1 for _, lbl in test_list if lbl == label)
            if count < self.samples_per_class_test:
                test_list.extend(self._augment_class(test_list, label, self.samples_per_class_test - count))

        print("Finalizing dataset...")

        # Optional label remapping from original EMNIST labels to 0–35
        if remap_labels:
            print("Remapping labels...")
            class_mapping = self._remap_labels(train_list)
            train_labels = torch.tensor([class_mapping[x[1]] for x in train_list])
            test_labels = torch.tensor([class_mapping[x[1]] for x in test_list])
        else:
            train_labels = torch.tensor([x[1] for x in train_list])
            test_labels = torch.tensor([x[1] for x in test_list])

        # Stack all image tensors into a single tensor (N x 1 x 28 x 28)
        train_images = torch.stack([x[0] for x in train_list])
        test_images = torch.stack([x[0] for x in test_list])

        # Create final TensorDatasets
        final_train_dataset = TensorDataset(train_images, train_labels)
        final_test_dataset = TensorDataset(test_images, test_labels)

        print(f"Final datasets prepared: {len(final_train_dataset)} train samples, {len(final_test_dataset)} test samples")

        return final_train_dataset, final_test_dataset

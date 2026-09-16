from torchvision import models
import torch
import torch.nn as nn
import os

class TM2Classifier(nn.Module):
    """
    A custom CNN classifier for TM2 (superclass prediction: digit, uppercase, lowercase).

    Architecture is defined by:
    - num_blocks: number of convolutional blocks
    - layers_per_block: conv layers per block
    - base_channels: number of channels in the first block
    - use_skip: optional skip-connection support (not yet functional)
    """

    def __init__(self, dropout=0.3, hidden_dim=128, num_blocks=2, layers_per_block=2, base_channels=32, use_skip=False):
        super().__init__()
        self.use_skip = use_skip

        layers = []
        in_channels = 1  # EMNIST images are grayscale

        # Build convolutional blocks
        for block in range(num_blocks):
            block_layers = []
            out_channels = base_channels * (2 ** block)  # Double channels per block

            for layer in range(layers_per_block):
                conv_in = in_channels if layer == 0 else out_channels
                block_layers.append(nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1))
                block_layers.append(nn.ReLU())

            block_seq = nn.Sequential(*block_layers)

            # Placeholder for skip logic (optional)
            if self.use_skip and in_channels == out_channels:
                block_seq = nn.Sequential(
                    nn.Sequential(*block_layers),
                    nn.Identity()
                )

            layers.append(block_seq)
            layers.append(nn.MaxPool2d(2))  # Downsampling layer

            in_channels = out_channels  # Update for next block

        self.cnn = nn.Sequential(*layers)

        # Dynamically compute the size of the flattened CNN output
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)  # Dummy image
            out = self.cnn(dummy)
            flat_dim = out.view(1, -1).shape[1]  # Flattened feature size

        # Fully connected classifier head
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # Output 3 superclasses
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


class CombinedClassifier(nn.Module):
    """
    Combines TM1 (36-class classifier) and TM2 (3-class superclass classifier).

    During forward pass:
    - TM1 predicts all 36 fine-grained classes.
    - TM2 predicts superclass (digit, uppercase, lowercase).
    - TM2 output is used as a mask to weight TM1 outputs.
    """

    def __init__(self, config_tm1, config_tm2,
                 tm1_weights_path="best_model_tm1.pth",
                 tm2_weights_path="best_model_tm2.pth"):
        super().__init__()

        # Build TM1 model (e.g., ResNet18 adapted for grayscale)
        self.tm1 = build_tm1_model(config_tm1)
        if os.path.exists(tm1_weights_path):
            print(f"Loading TM1 weights from {tm1_weights_path}")
            self.tm1.load_state_dict(torch.load(tm1_weights_path, map_location="cpu"))

        # Build TM2 model (custom CNN)
        self.tm2 = TM2Classifier(
            dropout=config_tm2.get("dropout", 0.3),
            hidden_dim=config_tm2["hidden_dim"],
            num_blocks=config_tm2["num_blocks"],
            layers_per_block=config_tm2["layers_per_block"],
            base_channels=config_tm2["base_channels"],
            use_skip=config_tm2["use_skip"]
        )

        if os.path.exists(tm2_weights_path):
            print(f"Loading TM2 weights from {tm2_weights_path}")
            try:
                self.tm2.load_state_dict(torch.load(tm2_weights_path, map_location="cpu"), strict=False)
            except RuntimeError as e:
                print(f"TM2 weights partially loaded:\n{e}")

    def forward(self, x):
        """
        Returns TM1 predictions masked by TM2 predictions.

        TM2 output is converted into a mask that zeroes out irrelevant class groups
        for each sample.
        """
        logits_tm1 = self.tm1(x)  # Shape: [batch_size, 36]
        logits_tm2 = self.tm2(x)  # Shape: [batch_size, 3]

        # Build fixed mask mapping classes to their superclasses
        mask = torch.tensor([
            [1, 0, 0] if i < 10 else [0, 1, 0] if i < 23 else [0, 0, 1]
            for i in range(36)
        ], device=logits_tm1.device, dtype=torch.float32)  # Shape: [36, 3]

        # Weight TM1 outputs based on TM2 superclass probabilities
        weights = torch.matmul(logits_tm2, mask.T)  # Shape: [batch, 36]
        return logits_tm1 * weights


def build_tm1_model(config):
    """
    Builds a ResNet18 model adapted for 1-channel EMNIST input and 36 output classes.

    Args:
        config (dict): Contains 'dropout' parameter for final layer.

    Returns:
        torch.nn.Module: Adapted ResNet18 model.
    """
    model = models.resnet18(pretrained=True)

    # Convert first conv layer from 3-channel to 1-channel input
    weight = model.conv1.weight.data  # Shape: [64, 3, 7, 7]
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = weight.mean(dim=1, keepdim=True)  # Average across RGB

    # Replace the final classification head
    model.fc = nn.Sequential(
        nn.Dropout(config["dropout"]),
        nn.Linear(model.fc.in_features, 36)  # Output layer for 36 classes
    )

    return model


def build_tm2_model(config):
    """
    Utility wrapper to build a TM2Classifier from a config dictionary.

    Args:
        config (dict): Must contain keys like 'dropout', 'hidden_dim', 'num_blocks', etc.

    Returns:
        TM2Classifier
    """
    return TM2Classifier(
        dropout=config.get("dropout", 0.3),
        hidden_dim=config.get("hidden_dim", 128),
        num_blocks=config.get("num_blocks", 2),
        layers_per_block=config.get("layers_per_block", 2),
        base_channels=config.get("base_channels", 32),
        use_skip=config.get("use_skip", False)
    )

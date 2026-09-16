import torch
import subprocess
import sys
import os

def install_requirements():
    """
    Installs dependencies listed in requirements.txt using pip.
    """
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print("requirements.txt not found.")
        return

    print(f"Installing packages from {req_file}...\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
    print("\nAll requirements installed.")


install_requirements()

# Local project imports
from data_loader import get_dataloaders, TM2Dataset, RemappedDataset
from models import build_tm1_model, build_tm2_model, CombinedClassifier
from evaluation import compute_confusion_matrix_and_stats, plot_confusion_matrix
from DataPreparer import DataPreparer
from hyperparameter_optimization import run_hyperparameter_optimization
from train import get_best_hyperparameters

if __name__ == "__main__":
    # === Setup ===
    # install_requirements()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # === Dataset Preparation ===
    if os.path.exists("emnist_train_balanced.pt") and os.path.exists("emnist_test_balanced.pt"):
        print("Loading shared dataset...")
        train_dataset_raw = torch.load("emnist_train_balanced.pt", weights_only=False)
        test_dataset_raw = torch.load("emnist_test_balanced.pt", weights_only=False)
    else:
        print("Preparing balanced dataset...")
        preparer = DataPreparer()
        train_dataset_raw, test_dataset_raw = preparer.prepare(remap_labels=False)
        torch.save(train_dataset_raw, "emnist_train_balanced.pt")
        torch.save(test_dataset_raw, "emnist_test_balanced.pt")

    # Wrap datasets for TM1 (fine-grained 36-class) and TM2 (3-class superclass)
    train_dataset_tm1 = RemappedDataset(train_dataset_raw)
    test_dataset_tm1 = RemappedDataset(test_dataset_raw)
    train_dataset_tm2 = TM2Dataset(train_dataset_raw)
    test_dataset_tm2 = TM2Dataset(test_dataset_raw)

    # === TM1 Hyperparameter Optimization ===
    print("\nRunning TM1 Hyperparameter Optimization...")
    run_hyperparameter_optimization(
        model_builder_fn=build_tm1_model,
        dataset_loader_fn=lambda cfg: (train_dataset_tm1, test_dataset_tm1),
        config_space_fn=lambda trial: {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "SGD"]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128])
        },
        csv_path="hyperparameters_tm1.csv",
        model_path="best_model_tm1.pth",
        timeout=5*60*60,
        n_epochs=30,
        train_fraction=1
    )

    # === TM2 Hyperparameter Optimization ===
    print("\nRunning TM2 Hyperparameter Optimization...")
    run_hyperparameter_optimization(
        model_builder_fn=build_tm2_model,
        dataset_loader_fn=lambda cfg: (train_dataset_tm2, test_dataset_tm2),
        config_space_fn=lambda trial: {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "SGD"]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_blocks": trial.suggest_int("num_blocks", 1, 4),
            "layers_per_block": trial.suggest_int("layers_per_block", 1, 3),
            "base_channels": trial.suggest_categorical("base_channels", [16, 32, 64]),
            "use_skip": trial.suggest_categorical("use_skip", [False, True]),
            "num_layers": trial.suggest_int("num_layers", 1, 3)
        },
        csv_path="hyperparameters_tm2.csv",
        model_path="best_model_tm2.pth",
        timeout=5*60*60,
        n_epochs=30,
        train_fraction=1
    )

    # === Combined Classifier Optimization ===
    print("\nRunning Combined Model Hyperparameter Optimization...")

    best_tm2_architecture = get_best_hyperparameters("hyperparameters_tm2.csv", [
        "hidden_dim", "num_blocks", "layers_per_block", "base_channels", "use_skip", "num_layers"
    ])

    run_hyperparameter_optimization(
        model_builder_fn=lambda cfg: CombinedClassifier(
            config_tm1={
                "lr": cfg["lr"],
                "dropout": cfg["dropout_tm1"],
                "optimizer": cfg["optimizer"],
                "weight_decay": cfg["weight_decay"],
                "batch_size": cfg["batch_size"]
            },
            config_tm2={
                "dropout": cfg["dropout_tm2"],
                **best_tm2_architecture
            }
        ),
        dataset_loader_fn=lambda cfg: (train_dataset_tm1, test_dataset_tm1),
        config_space_fn=lambda trial: {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout_tm1": trial.suggest_float("dropout_tm1", 0.1, 0.5),
            "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "SGD"]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "dropout_tm2": trial.suggest_float("dropout_tm2", 0.1, 0.5)
        },
        csv_path="hyperparameters_combined.csv",
        model_path="best_model_combined.pth",
        timeout=5*60*60,
        n_epochs=30,
        train_fraction=1
    )

    # === Evaluation and Confusion Matrices ===

    print("\nVerwechselungsmatrix für TM1:")
    _, _, test_loader_tm1 = get_dataloaders(train_dataset_tm1, train_dataset_tm1, test_dataset_tm1, 64)
    cm_tm1, labels_tm1 = compute_confusion_matrix_and_stats(
        lambda: build_tm1_model(get_best_hyperparameters("hyperparameters_tm1.csv", ["dropout"])),
        "best_model_tm1.pth",
        test_loader_tm1
    )
    plot_confusion_matrix(cm_tm1, labels_tm1, title="TM1 Verwechselungsmatrix")

    print("\nVerwechselungsmatrix für TM2:")
    _, _, test_loader_tm2 = get_dataloaders(train_dataset_tm2, train_dataset_tm2, test_dataset_tm2, 64)
    cm_tm2, labels_tm2 = compute_confusion_matrix_and_stats(
        lambda: build_tm2_model(get_best_hyperparameters(
            "hyperparameters_tm2.csv",
            ["dropout", "hidden_dim", "num_blocks", "layers_per_block", "base_channels", "use_skip", "num_layers"]
        )),
        "best_model_tm2.pth",
        test_loader_tm2,
        class_labels=["Digit", "Upper", "Lower"]
    )
    plot_confusion_matrix(cm_tm2, ["Digit", "Upper", "Lower"], title="TM2 Verwechselungsmatrix")

    print("\nVerwechselungsmatrix für Combined Model:")
    _, _, test_loader_combined = get_dataloaders(train_dataset_tm1, train_dataset_tm1, test_dataset_tm1, 64)
    best_combined = get_best_hyperparameters("hyperparameters_combined.csv", [
        "dropout_tm1", "optimizer", "weight_decay", "lr", "batch_size", "dropout_tm2"
    ])
    cm_combined, labels_combined = compute_confusion_matrix_and_stats(
        lambda: CombinedClassifier(
            config_tm1={
                "lr": best_combined["lr"],
                "dropout": best_combined["dropout_tm1"],
                "optimizer": best_combined["optimizer"],
                "weight_decay": best_combined["weight_decay"],
                "batch_size": best_combined["batch_size"]
            },
            config_tm2={
                "dropout": best_combined["dropout_tm2"],
                **best_tm2_architecture
            }
        ),
        "best_model_combined.pth",
        test_loader_combined
    )
    plot_confusion_matrix(cm_combined, labels_combined, title="Combined Verwechselungsmatrix")

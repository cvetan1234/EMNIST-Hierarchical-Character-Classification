import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pandas as pd
import time
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from train import train_one_epoch, evaluate
from data_loader import get_train_val_split, get_dataloaders


def run_hyperparameter_optimization(
    model_builder_fn,
    dataset_loader_fn,
    config_space_fn,
    csv_path,
    model_path=None,
    enqueue_trials=None,
    fixed_config={},
    timeout=3600,
    n_epochs=5,
    train_fraction=1.0,
    early_stop=5
):
    """
    Runs hyperparameter optimization using Optuna for a given model and dataset.

    Args:
        model_builder_fn (function): Builds the model given a config dictionary.
        dataset_loader_fn (function): Loads the dataset (train/test) given a config.
        config_space_fn (function): Defines the search space and returns a config for a trial.
        csv_path (str): Where to save trial results as CSV.
        model_path (str, optional): Path to save the best model weights.
        enqueue_trials (list, optional): List of manually specified configs to try first.
        fixed_config (dict, optional): Dict of fixed parameters to inject into all configs.
        timeout (int): Max duration for the entire optimization (in seconds).
        n_epochs (int): Max epochs per trial.
        train_fraction (float): Fraction of the training data to use.
        early_stop (int): Number of stagnant epochs before stopping early.
    """
    best_test_acc = -1  # Tracks the highest test accuracy across all trials
    best_model_state_dict = None  # Stores best model weights
    global_start_time = time.time()
    results = []  # List to store results for all trials

    def objective(trial):
        """
        Objective function for each Optuna trial.
        Trains a model, evaluates it, and returns the best validation accuracy.
        """
        nonlocal best_test_acc, best_model_state_dict

        # Enforce global timeout
        if time.time() - global_start_time > timeout:
            print("Global timeout reached.")
            raise optuna.TrialPruned()

        # Sample hyperparameters and merge with fixed config
        config = config_space_fn(trial)
        config.update(fixed_config)

        # Build model and prepare device
        model = model_builder_fn(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load datasets
        full_dataset, test_dataset = dataset_loader_fn(config)
        train_dataset, val_dataset = get_train_val_split(full_dataset, train_fraction=train_fraction)
        train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, config["batch_size"])

        # Set optimizer based on config
        optimizer_cls = optim.SGD if config["optimizer"] == "SGD" else optim.AdamW
        optimizer = optimizer_cls(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        criterion = nn.CrossEntropyLoss()

        # Early stopping variables
        best_val = 0
        stagnant_epochs = 0

        val_accs = []
        test_accs = []

        # Training loop
        for epoch in range(n_epochs):
            # Enforce global timeout mid-training
            if time.time() - global_start_time > timeout:
                raise optuna.TrialPruned()

            # Train and evaluate
            train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, n_epochs)
            val_acc = evaluate(model, val_loader, criterion, device)
            test_acc = evaluate(model, test_loader, criterion, device)

            val_accs.append(round(val_acc, 4))
            test_accs.append(round(test_acc, 4))

            # Report to Optuna
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Update best val accuracy
            if val_acc > best_val:
                best_val = val_acc
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1

            if stagnant_epochs >= early_stop:
                break

        final_test_acc = test_accs[-1]
        print(f"Final Test Accuracy: {final_test_acc:.4f}")

        # Save best model if new best test accuracy is found
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_model_state_dict = model.state_dict()
            if model_path:
                torch.save(best_model_state_dict, model_path)
                print(f"Best model saved to {model_path}")

        # Record trial results
        results.append({
            "trial": trial.number,
            "test_acc": round(final_test_acc, 4),
            "val_acc": round(best_val, 4),
            "val_acc_per_epoch": str(val_accs),
            "test_acc_per_epoch": str(test_accs),
            **config
        })

        return best_val  # Optuna will maximize this

    def save_callback(study, trial):
        """
        Saves intermediate results after each trial to CSV.
        """
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"Trial {trial.number} saved to {csv_path}")

    # Create Optuna study with TPE sampler and successive halving pruner
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
        pruner=SuccessiveHalvingPruner()
    )

    # Queue user-specified trials (useful for manual grid search or warm start)
    if enqueue_trials:
        for trial_params in enqueue_trials:
            study.enqueue_trial({**trial_params, **fixed_config})

    # Start the optimization loop
    study.optimize(objective, timeout=timeout, callbacks=[save_callback])

    # Save final results to CSV
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Final results saved to: {csv_path}")

    # Save best model weights again just to be sure
    if best_model_state_dict and model_path:
        torch.save(best_model_state_dict, model_path)
        print(f"Final best model weights saved to: {model_path}")

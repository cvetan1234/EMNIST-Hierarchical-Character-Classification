# EMNIST Hierarchical Character Classification

## Overview

This project explores handwritten character recognition using deep learning on the **EMNIST ByClass** dataset. It compares three related classification approaches:

- **TM1 – Fine-grained classification:** directly classifies handwritten characters into 36 target classes.
- **TM2 – Superclass classification:** classifies each image into one of three broader categories: digit, uppercase letter, or lowercase letter.
- **Combined classifier:** combines information from the fine-grained and superclass approaches.

The project also includes balanced dataset preparation, data augmentation, hyperparameter optimization with Optuna, model training, per-class evaluation, and confusion-matrix visualization.

This project was originally developed as part of the **Machine Learning 2** module at Ostbayerische Technische Hochschule Amberg-Weiden (OTH Amberg-Weiden).

## Classification Tasks

### TM1 – Fine-Grained Character Classification

TM1 distinguishes between **36 individual character classes**:

- Digits: `0–9`
- Uppercase letters: `A–M`
- Lowercase letters: `a–m`

The selected EMNIST labels are remapped to a continuous range of 36 output classes for training and evaluation.

### TM2 – Superclass Classification

TM2 simplifies the problem to three broader classes:

- `Digit`
- `Upper`
- `Lower`

This model focuses on distinguishing the general type of handwritten character rather than the exact character.

### Combined Classifier

The combined approach incorporates the TM1 and TM2 classification components into a single classification pipeline. Its hyperparameters are optimized separately and its performance is evaluated on the same fine-grained character task.

## Dataset Preparation

The project uses the **EMNIST ByClass** dataset through `torchvision`.

Only the following classes are selected:

```text
Digits:       0–9
Uppercase:    A–M
Lowercase:    a–m
```

The dataset preparation script:

- downloads EMNIST automatically when it is not already available;
- selects the required character classes;
- creates balanced training and test datasets;
- applies affine augmentation when additional samples are required;
- uses random rotation, translation, and scaling for augmentation;
- stores the prepared datasets locally for reuse.

By default, the preparation targets **5,000 training samples** and **1,000 test samples per class**.

## Hyperparameter Optimization

The project uses **Optuna** to optimize model and training hyperparameters.

Depending on the model, the search includes parameters such as:

- learning rate;
- dropout;
- optimizer (`AdamW` or `SGD`);
- weight decay;
- batch size;
- hidden-layer dimensions;
- number of convolutional blocks and layers;
- base channel count;
- skip connections.

Optimization results are stored in CSV files:

```text
hyperparameters_tm1.csv
hyperparameters_tm2.csv
hyperparameters_combined.csv
```

The best model weights produced during execution are stored as `.pth` files.

## Evaluation

The trained models are evaluated using confusion matrices and per-class accuracy.

The project produces separate confusion matrices for:

- TM1;
- TM2;
- the combined classifier.

For the fine-grained classifiers, the confusion matrix contains the 36 individual character classes. For TM2, it contains the three superclass labels.

## Technologies

- Python
- PyTorch
- Torchvision
- Optuna
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

```text
EMNIST-Hierarchical-Character-Classification/
├── DataPreparer.py
├── data_loader.py
├── evaluation.py
├── hyperparameter_optimization.py
├── main.py
├── models.py
├── train.py
├── requirements.txt
├── hyperparameters_tm1.csv
├── hyperparameters_tm2.csv
├── hyperparameters_combined.csv
├── tm1_verwechselungsmatrix.png
├── tm2_verwechselungsmatrix.png
├── combined_verwechselungsmatrix.png
└── Bericht.pdf
```

## How to Run

### 1. Download or clone the repository

Open a terminal or command prompt and navigate to the project directory:

```bash
cd path/to/EMNIST-Hierarchical-Character-Classification
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

The required packages are:

```text
torch
torchvision
optuna
matplotlib
seaborn
pandas
numpy
scikit-learn
```

### 4. Run the project

```bash
python main.py
```

`main.py` automatically detects whether CUDA is available and otherwise runs on the CPU.

If the prepared EMNIST datasets are not already present, the program downloads and prepares the dataset automatically. The processed datasets are then saved locally as:

```text
emnist_train_balanced.pt
emnist_test_balanced.pt
```

The main pipeline subsequently performs hyperparameter optimization for TM1, TM2, and the combined classifier, trains/saves the best models, and generates evaluation results.

> **Note:** The full pipeline is computationally intensive. In the supplied configuration, each of the three Optuna optimization stages can run for up to five hours and trains candidate models for up to 30 epochs. A CUDA-capable GPU is therefore recommended for practical experimentation, although the code can also use the CPU.

## Generated Files

Running the complete pipeline can generate additional files such as:

```text
data/
emnist_train_balanced.pt
emnist_test_balanced.pt
best_model_tm1.pth
best_model_tm2.pth
best_model_combined.pth
```

These files do not need to be included in the repository if the goal is to keep the GitHub project lightweight, because the EMNIST data can be downloaded and prepared again and the models can be retrained.

## About

The project demonstrates a complete deep-learning workflow for handwritten character recognition, from dataset preparation and balancing to model optimization, training, and evaluation.

A central focus is the comparison between direct fine-grained character recognition, broader superclass recognition, and a combined classification strategy.

## Author

**Tsvetan Stanchev**  
Ostbayerische Technische Hochschule Amberg-Weiden (OTH Amberg-Weiden)

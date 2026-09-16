import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix_and_stats(model_class, weights_path, dataloader, class_labels=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if class_labels is None:
        class_labels = [str(i) for i in range(10)] + list("ABCDEFGHIJKLM") + list("abcdefghijklm")
    label_indices = list(range(len(class_labels)))

    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)

    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_labels):
        correct = cm[i, i]
        total = cm[i].sum()
        acc = (correct / total) * 100 if total > 0 else 0.0
        print(f" - {class_name:>10s}: {acc:6.2f}% ({correct}/{total})")

    return cm, class_labels

def plot_confusion_matrix(cm, class_labels, title="Verwechselungsmatrix"):
    plt.figure(figsize=(14, 12))

    # Normalize by row to get per-class error percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_matrix = np.where(row_sums != 0, cm / row_sums, 0)

    # Create annotation labels: empty except for red errors
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and error_matrix[i, j] > 0.01:  # error > 1%
                annotations[i, j] = f"{cm[i, j]}\n({error_matrix[i, j]*100:.1f}%)"
            elif i == j:
                annotations[i, j] = str(cm[i, j])
            else:
                annotations[i, j] = ""

    # Create mask for red highlighting
    mask_red = np.zeros_like(cm, dtype=bool)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and error_matrix[i, j] > 0.01:
                mask_red[i, j] = True

    # Plot heatmap
    ax = sns.heatmap(cm, fmt='s', cmap='Blues',
                     xticklabels=class_labels,
                     yticklabels=class_labels,
                     annot=annotations, annot_kws={"fontsize": 8}, cbar=False)

    # Add red boxes manually for high-error cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if mask_red[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=1.5))

    plt.xlabel("Vorhergesagte Klasse", fontsize=12)
    plt.ylabel("Wahre Klasse", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    filename = f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"\n✅ Confusion matrix saved as '{filename}'")


"""Saved figures for training curves and the test confusion matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    epochs = list(range(1, len(history) + 1))
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_auc = [float("nan") if row["val_roc_auc"] is None else row["val_roc_auc"] for row in history]
    phases = [row["phase"] for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="Training loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation loss")
    axes[0].set_xlabel("Epoch (across phases)")
    axes[0].set_ylabel("Class-weighted loss")
    axes[0].set_title("Training curves")
    axes[0].legend()
    axes[1].plot(epochs, val_auc, marker="o", color="tab:green")
    axes[1].set_xlabel("Epoch (across phases)")
    axes[1].set_ylabel("Validation ROC-AUC")
    axes[1].set_title("Validation ranking")
    axes[1].set_ylim(0.0, 1.0)
    phase_note = ", ".join(f"{index}:{phase}" for index, phase in enumerate(phases, start=1))
    figure.suptitle(phase_note, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix: list[list[int]], path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Normal", "COVID"])
    axis.set_yticks([0, 1], ["Normal", "COVID"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(matrix[row][column])), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

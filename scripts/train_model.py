import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import INFO, PneumoniaMNIST
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = MODEL_DIR / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
NORMALIZE_MEAN = WEIGHTS.transforms().mean
NORMALIZE_STD = WEIGHTS.transforms().std

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)


def build_model() -> torch.nn.Module:
    model = mobilenet_v3_small(weights=WEIGHTS)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    return model.to(DEVICE)


def load_dataloaders():
    info = INFO["pneumoniamnist"]
    print("Dataset description:", info["description"])

    root = str(BASE_DIR / "data")
    train_dataset = PneumoniaMNIST(split="train", transform=train_transform, download=True, root=root)
    val_dataset = PneumoniaMNIST(split="val", transform=eval_transform, download=True, root=root)
    test_dataset = PneumoniaMNIST(split="test", transform=eval_transform, download=True, root=root)
    baseline_dataset = PneumoniaMNIST(split="train", download=True, root=root)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return baseline_dataset, train_loader, val_loader, test_loader


def collect_epoch_metrics(model: torch.nn.Module, loader: DataLoader, criterion: nn.Module) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = (probabilities >= 0.4).long()

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels)
    predictions_np = np.asarray(all_predictions)
    probabilities_np = np.asarray(all_probabilities)

    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(labels_np, predictions_np),
        "precision": precision_score(labels_np, predictions_np, zero_division=0),
        "recall": recall_score(labels_np, predictions_np, zero_division=0),
        "f1": f1_score(labels_np, predictions_np, zero_division=0),
        "roc_auc": roc_auc_score(labels_np, probabilities_np),
        "labels": labels_np.tolist(),
        "predictions": predictions_np.tolist(),
        "probabilities": probabilities_np.tolist(),
    }


def compute_training_baseline(dataset: PneumoniaMNIST) -> dict:
    means = []
    stds = []
    histograms = []

    for image in dataset.imgs:
        gray = image.astype(np.float32) / 255.0
        means.append(float(np.mean(gray)))
        stds.append(float(np.std(gray)))
        hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
        histograms.append(hist)

    histogram_mean = np.mean(np.stack(histograms), axis=0).tolist()

    return {
        "pixel_mean_mean": float(np.mean(means)),
        "pixel_mean_std": float(np.std(means) + 1e-8),
        "pixel_std_mean": float(np.mean(stds)),
        "pixel_std_std": float(np.std(stds) + 1e-8),
        "histogram_bins": 32,
        "histogram_mean": histogram_mean,
        "drift_threshold": 1.2,
        "class_names": ["Normal", "Pneumonia"],
        "image_size": IMAGE_SIZE,
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
    }


def save_history_plots(history: list[dict]) -> None:
    epochs = [item["epoch"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train Loss", marker="o")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="Val Loss", marker="o")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [item["train_accuracy"] for item in history], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, [item["val_accuracy"] for item in history], label="Val Accuracy", marker="o")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, [item["val_f1"] for item in history], label="Val F1", marker="o")
    axes[2].plot(epochs, [item["val_precision"] for item in history], label="Val Precision", marker="o")
    axes[2].plot(epochs, [item["val_recall"] for item in history], label="Val Recall", marker="o")
    axes[2].set_title("Validation Precision / Recall / F1")
    axes[2].legend()

    axes[3].plot(epochs, [item["val_roc_auc"] for item in history], label="Val ROC-AUC", marker="o")
    axes[3].set_title("Validation ROC-AUC")
    axes[3].legend()

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(labels: np.ndarray, predictions: np.ndarray) -> None:
    matrix = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Normal", "Pneumonia"])
    ax.set_yticklabels(["Normal", "Pneumonia"])
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_plot(labels: np.ndarray, probabilities: np.ndarray, roc_auc: float) -> None:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_plot(labels: np.ndarray, probabilities: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def train():
    baseline_dataset, train_loader, val_loader, test_loader = load_dataloaders()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_predictions = []

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            train_labels.extend(labels.detach().cpu().numpy().tolist())
            train_predictions.extend(predictions.detach().cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        val_metrics = collect_epoch_metrics(model, val_loader, criterion)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_roc_auc": round(val_metrics["roc_auc"], 6),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict()

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = collect_epoch_metrics(model, test_loader, criterion)

    test_labels = np.asarray(test_metrics["labels"])
    test_predictions = np.asarray(test_metrics["predictions"])
    test_probabilities = np.asarray(test_metrics["probabilities"])
    confusion = confusion_matrix(test_labels, test_predictions)
    tn, fp, fn, tp = confusion.ravel()
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)

    torch.save(model.state_dict(), MODEL_DIR / "pneumo_model.pth")

    summary = {
        "best_val_f1": round(best_val_f1, 6),
        "test_loss": round(test_metrics["loss"], 6),
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "test_precision": round(test_metrics["precision"], 6),
        "test_recall": round(test_metrics["recall"], 6),
        "test_f1": round(test_metrics["f1"], 6),
        "test_roc_auc": round(test_metrics["roc_auc"], 6),
        "test_specificity": round(specificity, 6),
        "test_sensitivity": round(sensitivity, 6),
        "confusion_matrix": confusion.tolist(),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device": str(DEVICE),
    }

    baseline = compute_training_baseline(baseline_dataset)

    (MODEL_DIR / "training_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (MODEL_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (MODEL_DIR / "baseline_stats.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    save_history_plots(history)
    save_confusion_matrix_plot(test_labels, test_predictions)
    save_roc_plot(test_labels, test_probabilities, test_metrics["roc_auc"])
    save_precision_recall_plot(test_labels, test_probabilities)

    print(json.dumps(summary, indent=2))
    print(f"Saved PyTorch checkpoint to {MODEL_DIR / 'pneumo_model.pth'}")
    print(f"Saved summary metrics to {MODEL_DIR / 'training_metrics.json'}")
    print(f"Saved epoch history to {MODEL_DIR / 'training_history.json'}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    train()

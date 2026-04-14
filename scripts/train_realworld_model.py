import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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
from torchvision import datasets, transforms
from torchvision.models import DenseNet121_Weights, densenet121


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "realworld"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a higher-resolution DenseNet121 baseline for future real-world chest X-ray scaling. "
            "This script expects an ImageFolder-style dataset with train/val/test splits."
        )
    )
    parser.add_argument(
        "--data-dir",
        "--dataset-root",
        dest="data_dir",
        type=Path,
        required=True,
        help="Dataset root containing train/ val/ test/ folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for checkpoints and reports.")
    parser.add_argument("--image-size", type=int, default=320, help="Square resize target for training and evaluation.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=1,
        help="Freeze DenseNet features for the first N epochs to stabilize warmup training.",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=1000,
        help="Maximum number of training images used to estimate drift baseline statistics.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    weights = DenseNet121_Weights.DEFAULT
    normalize = transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)

    train_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            normalize,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, eval_transform, weights.transforms().mean, weights.transforms().std


def load_datasets(data_dir: Path, image_size: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    missing = [str(path) for path in [train_dir, val_dir, test_dir] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Expected ImageFolder splits at train/ val/ test/. Missing: " + ", ".join(missing)
        )

    train_transform, eval_transform, normalize_mean, normalize_std = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    baseline_dataset = datasets.ImageFolder(train_dir)

    if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
        raise ValueError("train/ val/ and test/ splits must share the same class folder names.")

    return train_dataset, val_dataset, test_dataset, baseline_dataset, normalize_mean, normalize_std


def build_dataloaders(args, train_dataset, val_dataset, test_dataset):
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def build_model(num_classes: int):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(DEVICE)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    targets = np.asarray(dataset.targets)
    class_counts = np.bincount(targets, minlength=len(dataset.classes)).astype(np.float32)
    weights = np.sum(class_counts) / np.maximum(class_counts, 1.0)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def collect_epoch_metrics(model: nn.Module, loader: DataLoader, criterion: nn.Module, num_classes: int):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels)
    predictions_np = np.asarray(all_predictions)
    probabilities_np = np.asarray(all_probabilities)
    average = "binary" if num_classes == 2 else "weighted"

    metrics = {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy_score(labels_np, predictions_np),
        "precision": precision_score(labels_np, predictions_np, average=average, zero_division=0),
        "recall": recall_score(labels_np, predictions_np, average=average, zero_division=0),
        "f1": f1_score(labels_np, predictions_np, average=average, zero_division=0),
        "labels": labels_np.tolist(),
        "predictions": predictions_np.tolist(),
        "probabilities": probabilities_np.tolist(),
    }

    if num_classes == 2:
        positive_probs = probabilities_np[:, 1]
        confusion = confusion_matrix(labels_np, predictions_np)
        tn, fp, fn, tp = confusion.ravel()
        metrics["roc_auc"] = roc_auc_score(labels_np, positive_probs)
        metrics["specificity"] = tn / max(tn + fp, 1)
        metrics["sensitivity"] = tp / max(tp + fn, 1)
    else:
        metrics["roc_auc"] = roc_auc_score(labels_np, probabilities_np, multi_class="ovr", average="weighted")
        metrics["specificity"] = None
        metrics["sensitivity"] = None

    return metrics


def compute_training_baseline(dataset: datasets.ImageFolder, baseline_samples: int, image_size: int):
    sample_count = min(baseline_samples, len(dataset.samples))
    means = []
    stds = []
    histograms = []

    for path, _ in dataset.samples[:sample_count]:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L").resize((image_size, image_size)), dtype=np.float32) / 255.0
        means.append(float(np.mean(gray)))
        stds.append(float(np.std(gray)))
        hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
        histograms.append(hist)

    histogram_mean = np.mean(np.stack(histograms), axis=0).tolist()
    return {
        "sample_count": sample_count,
        "pixel_mean_mean": float(np.mean(means)),
        "pixel_mean_std": float(np.std(means) + 1e-8),
        "pixel_std_mean": float(np.mean(stds)),
        "pixel_std_std": float(np.std(stds) + 1e-8),
        "histogram_bins": 32,
        "histogram_mean": histogram_mean,
        "drift_threshold": 1.5,
        "image_size": image_size,
    }


def save_history_plots(history: list[dict], plots_dir: Path) -> None:
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
    fig.savefig(plots_dir / "training_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(labels: np.ndarray, predictions: np.ndarray, class_names: list[str], plots_dir: Path) -> None:
    matrix = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_yticklabels(class_names, rotation=0)
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_plot(labels: np.ndarray, probabilities: np.ndarray, roc_auc: float, class_names: list[str], plots_dir: Path) -> None:
    if probabilities.shape[1] != 2:
        return

    fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
    positive_class = class_names[1] if len(class_names) > 1 else "Positive"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{positive_class} ROC-AUC = {roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_plot(labels: np.ndarray, probabilities: np.ndarray, plots_dir: Path) -> None:
    if probabilities.shape[1] != 2:
        return

    precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def train():
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, baseline_dataset, normalize_mean, normalize_std = load_datasets(
        args.data_dir,
        args.image_size,
    )
    class_names = train_dataset.classes
    num_classes = len(class_names)
    train_loader, val_loader, test_loader = build_dataloaders(args, train_dataset, val_dataset, test_dataset)

    model = build_model(num_classes)
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = []
    best_val_f1 = -1.0
    best_state = None

    print(f"Training on {DEVICE} with classes: {class_names}")
    print(f"Loaded dataset from {args.data_dir}")
    print(f"Train/Val/Test sizes: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")

    for epoch in range(args.epochs):
        model.train()
        set_backbone_trainable(model, trainable=epoch >= args.freeze_backbone_epochs)

        running_loss = 0.0
        train_labels = []
        train_predictions = []

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            train_labels.extend(labels.detach().cpu().numpy().tolist())
            train_predictions.extend(predictions.detach().cpu().numpy().tolist())

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        val_metrics = collect_epoch_metrics(model, val_loader, criterion, num_classes)

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
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "normalize_mean": normalize_mean,
                "normalize_std": normalize_std,
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = collect_epoch_metrics(model, test_loader, criterion, num_classes)

    test_labels = np.asarray(test_metrics["labels"])
    test_predictions = np.asarray(test_metrics["predictions"])
    test_probabilities = np.asarray(test_metrics["probabilities"])
    confusion = confusion_matrix(test_labels, test_predictions)

    checkpoint_path = output_dir / "realworld_densenet121.pth"
    torch.save(best_state, checkpoint_path)

    summary = {
        "dataset_root": str(args.data_dir),
        "classes": class_names,
        "num_classes": num_classes,
        "backbone": "DenseNet121",
        "best_val_f1": round(best_val_f1, 6),
        "test_loss": round(test_metrics["loss"], 6),
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "test_precision": round(test_metrics["precision"], 6),
        "test_recall": round(test_metrics["recall"], 6),
        "test_f1": round(test_metrics["f1"], 6),
        "test_roc_auc": round(test_metrics["roc_auc"], 6),
        "test_specificity": None if test_metrics["specificity"] is None else round(test_metrics["specificity"], 6),
        "test_sensitivity": None if test_metrics["sensitivity"] is None else round(test_metrics["sensitivity"], 6),
        "confusion_matrix": confusion.tolist(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "image_size": args.image_size,
        "device": str(DEVICE),
    }

    baseline = compute_training_baseline(baseline_dataset, args.baseline_samples, args.image_size)
    baseline["class_names"] = class_names
    baseline["normalize_mean"] = normalize_mean
    baseline["normalize_std"] = normalize_std

    metrics_path = output_dir / "training_metrics.json"
    history_path = output_dir / "training_history.json"
    baseline_path = output_dir / "baseline_stats.json"

    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    save_history_plots(history, plots_dir)
    save_confusion_matrix_plot(test_labels, test_predictions, class_names, plots_dir)
    save_roc_plot(test_labels, test_probabilities, test_metrics["roc_auc"], class_names, plots_dir)
    save_precision_recall_plot(test_labels, test_probabilities, plots_dir)

    print(json.dumps(summary, indent=2))
    print(f"Saved real-world checkpoint to {checkpoint_path}")
    print(f"Saved summary metrics to {metrics_path}")
    print(f"Saved epoch history to {history_path}")
    print(f"Saved baseline stats to {baseline_path}")
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    train()

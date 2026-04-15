"""
Train MobileNetV3-small on ChestMNIST (14-class multi-label).
Outputs:
  models/chestmnist_mobilenetv3/
    mobilenetv3_chestmnist.pth      – PyTorch checkpoint (Model A)
    mobilenetv3_chestmnist.onnx     – ONNX export       (Model B)
    training_metrics.json
    baseline_stats.json             – pixel stats for drift detection

Usage (quick, shared-server-safe):
  python3 scripts/train_chestmnist.py --epochs 5 --batch-size 32
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.onnx
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

try:
    import medmnist
    from medmnist import ChestMNIST, INFO
except ImportError as exc:
    raise SystemExit("medmnist not installed — run: pip install medmnist") from exc

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "models" / "chestmnist_mobilenetv3"

CHESTMNIST_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = 14


def get_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return train_tf, val_tf


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_chestmnist(split: str, transform, download: bool, size: int = 224,
                    max_samples: int | None = None):
    ds = ChestMNIST(split=split, transform=transform, download=download, size=size, as_rgb=True)
    if max_samples and len(ds) > max_samples:
        indices = list(range(max_samples))
        from torch.utils.data import Subset
        ds = Subset(ds, indices)
    return ds


def compute_baseline_stats(loader: DataLoader) -> dict:
    """Compute pixel mean/std from training set for drift detection."""
    pixels = []
    for images, _ in loader:
        pixels.append(images.numpy())
        if len(pixels) >= 20:  # Sample first 20 batches
            break
    arr = np.concatenate(pixels, axis=0)  # (N, C, H, W)
    flat = arr.reshape(arr.shape[0], -1)
    return {
        "pixel_mean": float(flat.mean()),
        "pixel_std": float(flat.std()),
        "channel_means": arr.mean(axis=(0, 2, 3)).tolist(),
        "channel_stds": arr.std(axis=(0, 2, 3)).tolist(),
        "n_samples": int(arr.shape[0]),
    }


def tune_thresholds(model: nn.Module, loader: DataLoader, device: torch.device) -> list[float]:
    """Best-F1 threshold per class on validation set."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy().astype(float))

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    thresholds = []
    for i in range(NUM_CLASSES):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.10, 0.90, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            f = f1_score(labels[:, i], preds, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(round(float(best_t), 3))
    return thresholds


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             thresholds: list[float]) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy().astype(float))

    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)

    thr_arr = np.array(thresholds)
    preds = (probs >= thr_arr).astype(int)

    per_class_auroc, per_class_auprc, per_class_f1 = {}, {}, {}
    for i, cls in enumerate(CHESTMNIST_CLASSES):
        if labels[:, i].sum() > 0:
            per_class_auroc[cls] = round(float(roc_auc_score(labels[:, i], probs[:, i])), 4)
            per_class_auprc[cls] = round(float(average_precision_score(labels[:, i], probs[:, i])), 4)
        else:
            per_class_auroc[cls] = None
            per_class_auprc[cls] = None
        per_class_f1[cls] = round(float(f1_score(labels[:, i], preds[:, i], zero_division=0)), 4)

    macro_auroc_vals = [v for v in per_class_auroc.values() if v is not None]
    macro_auprc_vals = [v for v in per_class_auprc.values() if v is not None]

    return {
        "per_class_auroc": per_class_auroc,
        "per_class_auprc": per_class_auprc,
        "per_class_f1": per_class_f1,
        "test_macro_roc_auc": round(float(np.mean(macro_auroc_vals)), 4) if macro_auroc_vals else None,
        "test_macro_auprc": round(float(np.mean(macro_auprc_vals)), 4) if macro_auprc_vals else None,
        "test_micro_f1": round(float(f1_score(labels, preds, average="micro", zero_division=0)), 4),
        "test_macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
    }


def export_onnx(model: nn.Module, output_path: Path, image_size: int, device: torch.device):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    onnx.checker.check_model(str(output_path))
    print(f"  ONNX saved → {output_path}")


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-small on ChestMNIST (14-class multi-label)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    train_tf, val_tf = get_transforms(args.image_size)
    download = not args.no_download

    print("Loading ChestMNIST…")
    train_ds = load_chestmnist("train", train_tf, download, args.image_size, args.max_train_samples)
    val_ds   = load_chestmnist("val",   val_tf,   download, args.image_size, args.max_val_samples)
    test_ds  = load_chestmnist("test",  val_tf,   download, args.image_size, args.max_test_samples)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=args.workers)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=args.workers)

    # Baseline pixel stats for drift detection
    print("Computing baseline stats…")
    baseline_stats = compute_baseline_stats(train_loader)
    (args.output_dir / "baseline_stats.json").write_text(
        json.dumps(baseline_stats, indent=2), encoding="utf-8"
    )
    print(f"  Mean={baseline_stats['pixel_mean']:.4f}  Std={baseline_stats['pixel_std']:.4f}")

    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
    )

    best_val_loss = float("inf")
    history = []

    print(f"\nTraining {args.epochs} epochs…")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Quick val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.float().to(device)
                val_loss += criterion(model(imgs), lbls).item()
        val_loss /= max(len(val_loader), 1)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{args.epochs} — train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "mobilenetv3_chestmnist.pth")
            print("    ✓ checkpoint saved")

    # Reload best checkpoint
    model.load_state_dict(torch.load(args.output_dir / "mobilenetv3_chestmnist.pth", map_location=device))

    # Threshold tuning on val set
    print("\nTuning thresholds on validation set…")
    thresholds = tune_thresholds(model, val_loader, device)
    print(f"  Thresholds: {thresholds}")

    # Final evaluation on test set
    print("\nEvaluating on test set…")
    test_metrics = evaluate(model, test_loader, device, thresholds)

    # Save training_metrics.json
    training_metrics = {
        "architecture": "MobileNetV3-small",
        "dataset": "ChestMNIST",
        "class_names": CHESTMNIST_CLASSES,
        "num_classes": NUM_CLASSES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "image_size": args.image_size,
        "thresholds": thresholds,
        "multi_label": True,
        "best_val_loss": round(best_val_loss, 4),
        "history": history,
        **test_metrics,
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(training_metrics, indent=2), encoding="utf-8"
    )
    print(f"\n  Macro AUROC : {test_metrics['test_macro_roc_auc']}")
    print(f"  Macro AUPRC : {test_metrics['test_macro_auprc']}")
    print(f"  Micro  F1   : {test_metrics['test_micro_f1']}")

    # ONNX Export
    onnx_path = args.output_dir / "mobilenetv3_chestmnist.onnx"
    print(f"\nExporting ONNX → {onnx_path}")
    export_onnx(model, onnx_path, args.image_size, device)

    # ONNX export report (for backend resolver)
    onnx_report = {
        "base_onnx": str(onnx_path.name),
        "optimized_onnx": str(onnx_path.name),
        "serving_onnx": str(onnx_path.name),
        "input_shape": [1, 3, args.image_size, args.image_size],
    }
    (args.output_dir / "onnx_export_report.json").write_text(
        json.dumps(onnx_report, indent=2), encoding="utf-8"
    )

    print(f"\n✅ All artifacts saved to {args.output_dir}")
    print("   Model A (PyTorch) : mobilenetv3_chestmnist.pth")
    print("   Model B (ONNX)    : mobilenetv3_chestmnist.onnx")


if __name__ == "__main__":
    main()

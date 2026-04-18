"""
PneumoOps — Hugging Face Model Hub Upload Script
=================================================
Uploads both model artifacts to the HF Model Hub for versioning.
Run this once after training, and again whenever you retrain.

Usage:
    huggingface-cli login   # one-time login
    python scripts/upload_to_hf.py

Environment variables (override defaults):
    HF_MODEL_REPO   your-username/pneumoops-chestmnist
    HF_TOKEN        your write token (if not logged in via CLI)
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "chestmnist_mobilenetv3"

HF_TOKEN = os.getenv("HF_TOKEN")  # optional if already logged in via CLI
MODEL_REPO = os.getenv("HF_MODEL_REPO", "")  # e.g. "your-username/pneumoops-chestmnist"

if not MODEL_REPO:
    print("\n⚠️  Please set HF_MODEL_REPO environment variable, e.g.:")
    print('   export HF_MODEL_REPO="your-hf-username/pneumoops-chestmnist"')
    raise SystemExit(1)

# ─── Files to upload ──────────────────────────────────────────────────────────
ARTIFACTS = [
    ("mobilenetv3_chestmnist.pth",    "Model A — Baseline PyTorch checkpoint"),
    ("mobilenetv3_chestmnist.onnx",   "Model B — Optimized ONNX artifact"),
    ("training_metrics.json",          "Training + evaluation metrics (AUROC, AUPRC, F1)"),
    ("baseline_stats.json",            "Pixel distribution stats for drift monitoring"),
    ("onnx_export_report.json",        "ONNX export configuration"),
]

# ─── Model Card ───────────────────────────────────────────────────────────────
MODEL_CARD = """---
license: mit
tags:
  - medical
  - image-classification
  - chest-xray
  - mlops
  - onnx
  - pytorch
  - mobilenetv3
datasets:
  - medmnist/chestmnist
metrics:
  - roc_auc
---

# PneumoOps — ChestMNIST MobileNetV3-small

This repository contains **two model versions** for the PneumoOps MLOps pipeline:
- **Model A (Baseline):** `mobilenetv3_chestmnist.pth` — Standard PyTorch checkpoint
- **Model B (Optimized):** `mobilenetv3_chestmnist.onnx` — ONNX-exported for faster inference

Both models are identical in architecture (MobileNetV3-small) and weights.
The ONNX version is used for inference time optimization in A/B testing.

## Dataset
**ChestMNIST** — 14-class multi-label chest X-ray classification  
78,468 training images, 224×224 pixels, grayscale (converted to 3-channel).

## Classes (14)
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

## Performance (Test Set)
| Metric | Score |
|--------|-------|
| Macro AUROC | **0.808** |
| Macro AUPRC | 0.210 |
| Micro F1 | 0.343 |

## Usage in PneumoOps
These artifacts are loaded by the FastAPI backend and selected via a weighted A/B router:
- 60% of requests → PyTorch model
- 40% of requests → ONNX model

The backend also computes a drift score using `baseline_stats.json` to detect
out-of-distribution inputs in real time.
"""

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    api = HfApi(token=HF_TOKEN)

    print(f"\n📦 Creating/verifying model repository: {MODEL_REPO}")
    create_repo(
        repo_id=MODEL_REPO,
        repo_type="model",
        exist_ok=True,
        token=HF_TOKEN,
    )

    # Write model card
    card_path = MODEL_DIR / "README.md"
    card_path.write_text(MODEL_CARD, encoding="utf-8")
    print("   Model card written.")

    # Upload model card first
    print(f"\n⬆️  Uploading artifacts to https://huggingface.co/{MODEL_REPO}")
    upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Add model card",
        token=HF_TOKEN,
    )

    # Upload all artifacts
    for filename, description in ARTIFACTS:
        local_path = MODEL_DIR / filename
        if not local_path.exists():
            print(f"   ⚠️  Skipping {filename} — file not found")
            continue
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"   Uploading {filename} ({size_mb:.1f} MB) — {description} ...")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"Upload {filename}",
            token=HF_TOKEN,
        )

    print(f"\n✅ All artifacts uploaded!")
    print(f"   View at: https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    main()

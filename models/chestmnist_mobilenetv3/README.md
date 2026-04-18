---
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

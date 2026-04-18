---
title: PneumoOps
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: MLOps A/B testing & drift monitoring
---

# 🫁 PneumoOps

**Continuous MLOps Pipeline with A/B Testing & Data Drift Monitoring for 14-Class Thoracic Disease Detection**

[![CI](https://github.com/Prakhar54-byte/PneumoOps/actions/workflows/deploy.yml/badge.svg)](https://github.com/Prakhar54-byte/PneumoOps/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#)

---

## What This Is

PneumoOps is a production-style MLOps system for **multi-label chest X-ray classification**. It demonstrates real-world deployment challenges:

- **A/B Testing** — every inference request is randomly routed to either *Model A (PyTorch)* or *Model B (ONNX)*, letting you measure real-world latency differences between serving backends.
- **Data Drift Monitoring** — statistical pixel-distribution comparison (KS-test) against the training baseline. When distribution shifts, the system flags `DRIFT_DETECTED` — the trigger for automated retraining in production.
- **Prometheus Observability** — request counters, latency histograms, per-disease prediction rates, and drift alert counters are all scraped at `/metrics`.
- **Dockerized Deployment** — the entire stack runs in containers, deployable to Hugging Face Spaces via a single `git push`.

---

## Architecture

```
         Train (ChestMNIST + MobileNetV3-small)
                    │
          ┌─────────┴──────────┐
    Model A (.pth)       Model B (.onnx)
    PyTorch serving      ONNX Runtime serving
          └─────────┬──────────┘
                    │
        ┌───────────┴───────────┐
        │  FastAPI Backend       │
        │  ├─ A/B Router (60/40) │
        │  ├─ Drift Monitor (KS) │
        │  ├─ Prometheus /metrics│
        │  └─ /health /history   │
        └───────────┬───────────┘
                    │
          ┌─────────┴──────────┐
          │   Gradio UI         │
          │  Top-3 predictions  │
          │  Model arm used     │
          │  Latency (ms)       │
          │  Drift alert badge  │
          └─────────────────────┘
```

---

## Dataset & Model

| Property | Value |
|---|---|
| Dataset | [ChestMNIST](https://medmnist.com/) — 14-class multi-label chest X-ray |
| Classes | Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia |
| Model A | MobileNetV3-small (PyTorch `.pth`) |
| Model B | MobileNetV3-small (ONNX Runtime `.onnx`) |
| Training | 5 epochs, AdamW, BCEWithLogitsLoss, per-class threshold tuning |
| Macro AUROC | 0.686 (5-epoch, 5k samples — improves with full dataset) |

---

## Project Structure

```
pneumo_ops/
├── backend/
│   └── main.py              # FastAPI: A/B routing, drift monitor, Prometheus
├── frontend/
│   └── app.py               # Gradio UI: top-3 chart, drift badge, latency
├── scripts/
│   └── train_chestmnist.py  # Training: ChestMNIST → MobileNetV3 → ONNX export
├── models/
│   └── chestmnist_mobilenetv3/
│       ├── mobilenetv3_chestmnist.pth    # Model A (PyTorch)
│       ├── mobilenetv3_chestmnist.onnx   # Model B (ONNX)
│       ├── training_metrics.json
│       └── baseline_stats.json           # Pixel stats for drift reference
├── model_utils.py           # CalibratedModel + temperature scaling util
├── Dockerfile               # Single-container build
├── docker-compose.yml       # backend + frontend services
├── requirements.txt
└── .github/workflows/
    └── deploy.yml           # CI (lint/import check) + HF Spaces deploy
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Prakhar54-byte/PneumoOps
cd pneumo_ops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Quick run (5k samples, ~1 min on GPU)
python3 scripts/train_chestmnist.py --epochs 5 --batch-size 32 --max-train-samples 5000

# Full dataset
python3 scripts/train_chestmnist.py --epochs 15 --batch-size 64
```

Outputs saved to `models/chestmnist_mobilenetv3/`:
- `mobilenetv3_chestmnist.pth` — PyTorch checkpoint
- `mobilenetv3_chestmnist.onnx` — ONNX export
- `training_metrics.json` — AUROC, AUPRC, F1, thresholds
- `baseline_stats.json` — pixel reference for drift detection

### 3. Run the backend

```bash
PNEUMOOPS_PROFILE=chestmnist python3 -m uvicorn backend.main:app --port 7860
```

Key endpoints:

| Endpoint | Description |
|---|---|
| `POST /predict` | Run inference (A/B routed) |
| `GET /health` | System status + model metadata |
| `GET /metrics` | Prometheus scrape endpoint |
| `GET /history` | Last 20 requests |
| `GET /metrics/class-rates` | Per-class prediction rates |
| `GET /metrics/calibration` | AUROC / AUPRC / Brier per class |

### 4. Run the UI

```bash
BACKEND_PREDICT_URL=http://127.0.0.1:7860/predict python3 frontend/app.py
# Open http://localhost:7861
```

### 5. Docker (full stack)

```bash
docker compose up --build
# Backend → http://localhost:7860
# Frontend → http://localhost:7861
```

---

## Deployment — Hugging Face Spaces

### Manual push

```bash
# Add HF remote
git remote add space https://huggingface.co/spaces/Prakhar54-byte/PneumoOps

# Push (Spaces will build the Docker image automatically)
git push space main
```

### Automated (GitHub Actions)

Set these repository secrets on GitHub:

| Secret | Description |
|---|---|
| `HF_TOKEN` | Hugging Face access token (write permission) |
| `HF_SPACE_REPO` | e.g. `your-username/pneumoops` |
| `HF_MODEL_REPO` | *(optional)* e.g. `your-username/pneumoops-models` |

Every push to `main` triggers CI checks then deploys to your Space automatically.

---

## Real-World MLOps Challenges Addressed

| Challenge | Solution |
|---|---|
| Model degradation over time | Drift Monitor (KS-test on pixel distribution) |
| Serving latency variance | A/B routing between PyTorch and ONNX, latency tracked per arm |
| Class imbalance (rare diseases) | Per-class threshold tuning on val set + AUPRC tracking |
| Missed diagnoses | Per-class recall monitored at `/metrics/class-rates` |
| Production observability | Prometheus metrics — latency histograms, per-disease counters, drift alerts |
| Automated retraining signals | `DRIFT_DETECTED` flag logged + exposed via Prometheus counter |

---

## Libraries

`PyTorch` · `ONNX Runtime` · `FastAPI` · `Gradio` · `Docker` · `Hugging Face Hub/Spaces` · `scikit-learn` · `Prometheus` · `MedMNIST` · `SciPy`

---

## License

MIT

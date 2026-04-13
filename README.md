---
title: PneumoOps
sdk: docker
app_port: 7860
pinned: false
---

# PneumoOps

PneumoOps is a continuous MLOps mini-project for chest X-ray classification built around the assignment requirements:

- `PneumoniaMNIST` training with `MobileNetV3-small`
- baseline PyTorch checkpoint plus optimized ONNX deployment target
- FastAPI inference endpoint with 50/50 A/B routing
- latency tracking and simulated data drift monitoring
- Gradio frontend mounted into the same Dockerized app
- GitHub Actions sync to a Hugging Face Docker Space

## Current measured results

These are the latest model metrics currently saved in `models/training_metrics.json`:

- Test Accuracy: `0.9022`
- Test Precision: `0.9318`
- Test Recall: `0.9103`
- Test F1: `0.9209`
- Test ROC-AUC: `0.9607`
- Specificity: `0.8889`

The current ONNX export report shows that dynamic quantization was not supported by the runtime, so the project serves the optimized ONNX fallback. Present the comparison as:

`Baseline PyTorch vs optimized ONNX Runtime`

## Run order

Use Python `3.10` to `3.13` locally. `onnxruntime` does not currently support Python `3.14`, so if you are using `uv`, recreate the environment with Python `3.11`:

`uv python install 3.11`
`rm -rf venv`
`uv venv --python 3.11 venv`
`source venv/bin/activate`
`uv pip install -r requirements.txt`

1. Train the baseline model:
   `python scripts/train_model.py`
2. Export and quantize the ONNX model:
   `python scripts/convert_to_onnx.py`
3. Benchmark PyTorch vs ONNX latency:
   `python scripts/benchmark_models.py`
4. Smoke test the app:
   `python scripts/smoke_test_app.py`
5. Start the local app:
   `python -m uvicorn backend.main:app --host 0.0.0.0 --port 7860`
6. Open:
   `http://127.0.0.1:7860`

## Two-person split

- Person 1: training, metrics, plots, ONNX export, benchmark evidence
- Person 2: app integration, A/B routing, Docker, GitHub Actions, Hugging Face deployment, demo screenshots

Detailed runbooks are in [docs/TRAINING_AND_HANDOFF.md](/home/prakhar/Downloads/ml_dl_ops_pro/pneumo_ops/docs/TRAINING_AND_HANDOFF.md:1) and [docs/DEPLOYMENT.md](/home/prakhar/Downloads/ml_dl_ops_pro/pneumo_ops/docs/DEPLOYMENT.md:1).

## Files you should mention in the viva/demo

- `scripts/train_model.py`: training plus baseline drift statistics generation
- `scripts/convert_to_onnx.py`: ONNX export plus optimized fallback handling
- `scripts/benchmark_models.py`: PyTorch vs ONNX latency comparison
- `scripts/smoke_test_app.py`: local API/interface validation
- `backend/main.py`: FastAPI API, A/B router, latency tracking, drift scoring
- `frontend/app.py`: upload UI and metrics display
- `.github/workflows/deploy.yml`: CI plus Hugging Face Space sync
- `Dockerfile`: single-container deployment for Hugging Face Spaces

## Recommended demo story

Train one model once, export it into two serving formats, and compare them in production:

- Model A: standard PyTorch checkpoint
- Model B: optimized ONNX Runtime model

That gives you a fair A/B test focused on inference optimization rather than architecture differences.

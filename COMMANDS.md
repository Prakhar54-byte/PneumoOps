# PneumoOps: Operations & Deployment Runbook

This document contains all the essential commands for training, monitoring, deploying, and testing the PneumoOps pipeline.

---

## 1. Local Training & Monitoring

Because the model is training in the background using `nohup` (No Hangup), **you can safely close your laptop or disconnect from the server.** The training process is detached from your local session and will continue running strictly on the server until it finishes 15 epochs. 

**To check the live training progress at any time:**
```bash
tail -f training.log
```
*(Press `Ctrl+C` to exit the live view. The training will continue running behind the scenes.)*

**If you need to manually stop the background training:**
```bash
pkill -f train_chestmnist.py
```

---

## 2. Docker Deployment (With Cybersecurity Patches)

The `Dockerfile` has been hardened to drop root privileges (`appuser`) and automatically fetch the newest OS security patches upon building.

**To build the secure container image:**
```bash
docker compose build
```

**To start the full stack (FastAPI Backend + Gradio Frontend) securely in the background:**
```bash
docker compose up -d
```

**To read the live Docker logs (useful for verifying the FastAPI startup):**
```bash
docker compose logs -f
```

*(Note: Because of GitHub Actions, pushing to the `master` branch will automatically run this build and deploy the containers to your Hugging Face space!)*

---

## 3. Testing the Live Deployment 

Once the Docker containers are running (locally or on Hugging Face), you can probe them to test both Data Science metrics and DevOps/Platform performance.

### A. Testing API Health & Model Statistics
Check what profiles and models the backend is serving, and view the embedded Brier scores.
```bash
curl -s http://127.0.0.1:7860/health | jq
```

### B. Triggering a Prediction (A/B Test + Drift Monitor)
Pass a Chest X-ray image to the pipeline to see the A/B test router in action. The response will explicitly return the `drift_status` ("NORMAL" or "DRIFT_DETECTED") depending on statistical distribution checks.
```bash
curl -X POST -F "file=@Screenshot_or_Xray.png" http://127.0.0.1:7860/predict
```

### C. Scraping Platform DevOps Metrics
Standard enterprise observability. Pull the raw Prometheus text logs to visualize live deployment performance.
```bash
curl -s http://127.0.0.1:7860/metrics | grep "pneumoops"
```
You should actively look for:
* `pneumoops_inference_latency_ms`: To compare PyTorch baseline speed vs ONNX optimizations.
* `pneumoops_disease_predictions_total`: To see which of the 14 multi-label diseases are most frequently diagnosed.

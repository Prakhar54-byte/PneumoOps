# 👩‍💻 Person B — Standalone Task Guide
## PneumoOps: Backend Verification, Model Registry & Monitoring

> **This guide is self-contained.** You only need the GitHub repository link and the HF Token that Person A will share with you privately. You do NOT need access to Person A's machine.

---

## 🔑 What You Need From Person A (Ask Before Starting)

| Item | How to Get It |
|---|---|
| GitHub repository URL | `https://github.com/Prakhar54-byte/PneumoOps` |
| Hugging Face Access Token | Person A will share privately (WhatsApp/DM) — do NOT share publicly |
| HF Model Repo name | `Prakhar54-byte/pneumoops-chestmnist` (or whatever Person A created) |

---

## ⚙️ Step 0: One-Time Environment Setup

Open a terminal on your machine and run these commands:

```bash
# 1. Clone the GitHub repository
git clone https://github.com/Prakhar54-byte/PneumoOps.git
cd PneumoOps

# 2. Create a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install all dependencies (CPU-only torch is fine for testing)
pip install -r requirements.txt

# 4. Login to Hugging Face using the token Person A gave you
pip install huggingface_hub
huggingface-cli login
# → Paste the token when prompted. Press Enter.
```

---

## 📦 Task B1: Download the Model Files From HF Hub

> The `.pth` and `.onnx` model files are too large for Git. They live on Hugging Face Model Hub.

```bash
# Set the model repo (ask Person A for the exact name)
export HF_MODEL_REPO="Prakhar54-byte/pneumoops-chestmnist"

# Download all model artifacts to the correct local folder
python3 - <<'PY'
from huggingface_hub import snapshot_download
import shutil, os

local_dir = snapshot_download(
    repo_id=os.environ["HF_MODEL_REPO"],
    repo_type="model",
    local_dir="models/chestmnist_mobilenetv3",
    ignore_patterns=["*.md"],
)
print(f"✅ Downloaded to: {local_dir}")
PY
```

After this runs, confirm the files exist:
```bash
ls models/chestmnist_mobilenetv3/
# Expected:  mobilenetv3_chestmnist.pth  mobilenetv3_chestmnist.onnx
#            training_metrics.json  baseline_stats.json  onnx_export_report.json
```

---

## 🐳 Task B2: Start the App Locally With Docker

```bash
# Make sure Docker Desktop is installed and running first
docker compose up --build -d app

# Wait ~30 seconds for startup, then check health
curl http://127.0.0.1:7860/health
```

Expected output (both models should show `true`):
```json
{
  "status": "ok",
  "pytorch_model_loaded": true,
  "onnx_model_loaded": true,
  "class_count": 14
}
```

---

## 🧪 Task B3: Run the Automated Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Expected output should show all PASSED:
# tests/test_api.py::test_health_check PASSED
# tests/test_api.py::test_metrics_endpoint PASSED
# tests/test_api.py::test_predict_with_synthetic_image PASSED
# tests/test_api.py::test_drift_on_non_xray PASSED
```


---

## 🖥️ Task B4: Verify the Gradio UI

1. Open your browser and go to: **`http://localhost:7860/ui`**
2. Upload any image (a chest X-ray from Google Images is fine for testing)
3. Click **"Run Screening"**
4. Verify that ALL five output fields are displayed:
   - ✅ Top-3 Predictions bar chart
   - ✅ Model Used (shows "Baseline PyTorch" OR "Optimized ONNX")
   - ✅ Inference Latency (e.g., "12.3 ms")
   - ✅ All Findings Detected
   - ✅ Data Drift Alert (green "NORMAL" for a real X-ray)
5. Upload a non-X-ray (e.g., a photo of a dog or landscape)
   - ✅ You should see a red **"⚠️ DRIFT DETECTED"** badge

> **Take a screenshot of both tests (normal + drift)** — these are needed for the poster and final report.

---

## 📊 Task B5: Production Monitoring

These endpoints let you observe how the model is performing without looking at individual images:

```bash
# Recent request history (last 20 predictions)
curl http://127.0.0.1:7860/history | python3 -m json.tool

# Per-class prediction rates
# (which diseases are predicted most often)
curl http://127.0.0.1:7860/metrics/class-rates | python3 -m json.tool

# AUROC / AUPRC per disease class (from training)
curl http://127.0.0.1:7860/metrics/calibration | python3 -m json.tool

# Prometheus metrics (raw counters/histograms for monitoring tools)
curl http://127.0.0.1:7860/metrics | grep pneumoops
```

> The Prometheus `/metrics` endpoint can be connected to **Grafana** in a real production setup for dashboards. For this project, reading the raw text output is sufficient.

---

## ✍️ Task B6: Documentation (Poster & Report)

- [ ] Copy the text from `POSTER_CONTENT.md` into the Canva poster template
- [ ] Read `ETHICS.md` — you may need to explain the ethical considerations in your presentation
- [ ] Update the `README.md` with your name in the contributors section:
  ```bash
  # In the README, find "Contributors" and add your name
  git add README.md
  git commit -m "docs: add contributors section"
  git push
  ```

---

## 🚨 Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Install Docker Desktop from docker.com |
| `models/*.pth not found` | Run Task B1 again — the download may have failed |
| `curl: (7) Failed to connect` | The Docker container isn't running. Run `docker compose up -d app` |
| `pytest: command not found` | Run `pip install pytest httpx` inside your virtual environment |
| Test `test_predict` fails | Check `docker logs pneumo_ops-app-1` for error messages |

---

## 📞 Contact

If you are stuck, message Person A with:
1. The exact error message (copy-paste it)
2. Which Task step you are on
3. The output of `docker logs pneumo_ops-app-1`

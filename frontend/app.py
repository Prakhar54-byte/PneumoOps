"""
PneumoOps — Simple Assignment Demo UI (Task 1 Spec)
====================================================
Matches the exact Gradio UI requirements from ASSIGNMENT_TARGET.md:
  1. Image upload
  2. Top-3 predictions — bar chart
  3. Model Used — "Baseline PyTorch" or "Optimized ONNX"
  4. Inference Latency (ms)
  5. Drift Alert — "Normal" or "Drift Detected" (color badge)

Start independently of the main dashboard:
  python3 frontend/app_simple.py
"""

import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
import numpy as np
import requests
from PIL import Image

# When running embedded inside FastAPI (HF Spaces), the backend is on the same process.
# When running standalone via docker-compose, override with BACKEND_PREDICT_URL env var.
BACKEND_URL = os.getenv("BACKEND_PREDICT_URL", "http://127.0.0.1:7860/predict")

CHESTMNIST_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------

def call_backend(image: Image.Image, api_url: str) -> dict:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post(
        api_url,
        files={"file": ("xray.png", buf.getvalue(), "image/png")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIER_COLORS = {
    "Pneumonia": "#dc2626",
    "Pneumothorax": "#dc2626",
    "Mass": "#dc2626",
    "Nodule": "#dc2626",
    "Effusion": "#f59e0b",
    "Cardiomegaly": "#f59e0b",
    "Consolidation": "#f59e0b",
}

def _bar_color(label: str) -> str:
    return _TIER_COLORS.get(label, "#6366f1")


def build_top3_chart(top_predictions: list[dict]) -> object:
    """Horizontal bar chart of top-3 predicted pathologies."""
    top3 = sorted(top_predictions, key=lambda x: x["confidence"], reverse=True)[:3]
    if not top3:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.set_title("No pathologies detected above threshold")
        ax.axis("off")
        return fig

    labels = [p["label"] for p in top3]
    values = [p["confidence"] / 100.0 for p in top3]
    colors = [_bar_color(l) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence Score", fontsize=11)
    ax.set_title("Top-3 Predicted Pathologies", fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.25, zorder=0)

    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#dc2626", label="Critical"),
        mpatches.Patch(color="#f59e0b", label="Significant"),
        mpatches.Patch(color="#6366f1", label="Standard"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f1f5f9")
    plt.tight_layout()
    return fig


def drift_badge(status: str) -> str:
    if status == "DRIFT_DETECTED":
        return (
            "<div style='padding:0.8rem 1.2rem;border-radius:12px;font-size:1rem;"
            "font-weight:700;background:#fef3f2;color:#b42318;border:2px solid #fca5a5;'>"
            "⚠️ DRIFT DETECTED — Out-of-distribution input. Consider retraining.</div>"
        )
    return (
        "<div style='padding:0.8rem 1.2rem;border-radius:12px;font-size:1rem;"
        "font-weight:700;background:#ecfdf5;color:#065f46;border:2px solid #6ee7b7;'>"
        "✅ NORMAL — Input distribution matches training baseline.</div>"
    )


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------

def predict(image: Image.Image, api_url: str):
    if image is None:
        raise gr.Error("Please upload a chest X-ray image.")

    try:
        payload = call_backend(image, api_url)
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise gr.Error(f"Backend error: {detail}") from exc
    except requests.RequestException as exc:
        raise gr.Error(f"Cannot reach backend at {api_url} — is the FastAPI server running?") from exc

    top_preds   = payload.get("top_predictions", [])
    model_key   = payload.get("selected_arm", payload.get("selected_model", "onnx"))
    latency_ms  = payload.get("latency_delta_ms") or payload.get("request_latency_ms", "n/a")
    drift_status = payload.get("drift", {}).get("drift_alert", "UNKNOWN")

    model_label = "Optimized ONNX" if "onnx" in str(model_key).lower() else "Baseline PyTorch"

    chart = build_top3_chart(top_preds)
    findings_list = ", ".join(payload.get("predicted_labels", [])) or "No findings above threshold"

    return (
        chart,
        findings_list,
        model_label,
        f"{latency_ms} ms",
        drift_badge(drift_status),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #2563eb 100%);
    padding: 1.5rem 2rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 0.4rem; font-size: 2rem; letter-spacing: -1px; }
.hero p  { margin: 0; opacity: 0.85; font-size: 0.95rem; line-height: 1.6; }
.ethics-banner {
    background: #fff7ed;
    border: 2px solid #fb923c;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #7c2d12;
}
.ethics-banner strong { color: #c2410c; }
.privacy-note {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #1e3a5f;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CSS,
    title="PneumoOps — A/B Testing MLOps Pipeline",
) as demo:

    gr.HTML("""
        <div class="ethics-banner">
          <strong>⚠️ IMPORTANT DISCLAIMER — READ BEFORE USE</strong><br/>
          This tool is for <strong>educational and research purposes ONLY</strong> and is <strong>NOT a medical device</strong>.<br/>
          Predictions <strong>must not be used</strong> for clinical diagnosis, patient care, or any medical decision-making.<br/>
          Always consult a qualified healthcare professional. Model performance may vary across demographics and scan quality.
        </div>
        <div class="privacy-note">
          🔐 <strong>Data Privacy:</strong> Uploaded images are processed in-memory only. No personally identifiable information is stored.
          Do <u>not</u> upload images containing patient names, IDs, or other identifying metadata.
        </div>
    """)

    gr.HTML("""
        <div class="hero">
          <h1>🫁 PneumoOps</h1>
          <p>
            <strong>MLOps A/B Testing Pipeline</strong> for 14-class thoracic disease screening.<br/>
            Every request is randomly routed to <strong>Model A (Baseline PyTorch)</strong> or
            <strong>Model B (Optimized ONNX)</strong>, enabling real-world latency benchmarking.<br/>
            A built-in <strong>Data Drift Monitor</strong> flags out-of-distribution inputs —
            simulating automated retraining triggers in production.
          </p>
        </div>
    """)

    with gr.Accordion("⚙️ Backend Settings", open=False):
        api_url = gr.Textbox(label="Backend URL", value=BACKEND_URL)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Chest X-Ray", height=380)

    submit_btn = gr.Button("🔬 Run Screening", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("## Results")

    with gr.Row():
        model_used   = gr.Textbox(label="Model Used (A/B Arm)", interactive=False, scale=1)
        latency_out  = gr.Textbox(label="Inference Latency",     interactive=False, scale=1)

    findings_out = gr.Textbox(label="All Findings Detected", interactive=False)
    drift_out    = gr.HTML(label="Data Drift Alert")
    top3_chart   = gr.Plot(label="Top-3 Predicted Pathologies")

    gr.Markdown("""
    ---
    **14 Classes:** Atelectasis · Cardiomegaly · Effusion · Infiltration · Mass · Nodule ·
    Pneumonia · Pneumothorax · Consolidation · Edema · Emphysema · Fibrosis ·
    Pleural Thickening · Hernia

    🔴 Critical · 🟠 Significant · 🟣 Standard

    ---
    > ⚖️ **Responsible AI Notice:** This system *assists in prediction*, it does not confirm diagnoses.
    > Results are statistical estimates and must be reviewed by a qualified clinician.
    > See [ETHICS.md](https://github.com/Prakhar54-byte/PneumoOps/blob/master/ETHICS.md) for our full Responsible AI policy.
    """)

    submit_btn.click(
        fn=predict,
        inputs=[image_input, api_url],
        outputs=[top3_chart, findings_out, model_used, latency_out, drift_out],
    )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)


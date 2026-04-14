import io
import os

import gradio as gr
import matplotlib.pyplot as plt
import requests
from PIL import Image


DEFAULT_API_URL = os.getenv("BACKEND_PREDICT_URL", "http://127.0.0.1:7860/predict")


def _base_api_url(api_url: str) -> str:
    cleaned = api_url.rstrip("/")
    return cleaned[: -len("/predict")] if cleaned.endswith("/predict") else cleaned


def _health_url(api_url: str) -> str:
    return f"{_base_api_url(api_url)}/health"


def _post_image(image: Image.Image, api_url: str):
    if image is None:
        raise gr.Error("Please upload a chest X-ray before submitting.")

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)

    response = requests.post(
        api_url,
        files={"file": ("upload.png", buffer.getvalue(), "image/png")},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _get_json(url: str):
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def _build_predictions_table(predictions: list[dict]):
    return [[item["label"], f"{item['confidence']}%", f"{item['threshold']}%"] for item in predictions]


def _build_latency_plot(history: list[dict]):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    if not history:
        ax.set_title("Last 20 Requests")
        ax.set_xlabel("Request")
        ax.set_ylabel("Latency (ms)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    x_values = list(range(1, len(history) + 1))
    latencies = [item["latency_ms"] for item in history]
    colors = ["#2563eb" if "PyTorch" in item["model"] else "#d97706" for item in history]

    ax.plot(x_values, latencies, color="#64748b", alpha=0.5, linewidth=1.5)
    ax.scatter(x_values, latencies, c=colors, s=80)
    ax.set_title("Last 20 Request Latencies")
    ax.set_xlabel("Request")
    ax.set_ylabel("Latency (ms)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _badge(text: str, status: str):
    color_map = {
        "good": ("#ecfdf3", "#027a48"),
        "warn": ("#fffaeb", "#b54708"),
        "bad": ("#fef3f2", "#b42318"),
        "info": ("#eff6ff", "#1d4ed8"),
    }
    background, color = color_map[status]
    return (
        "<div style='padding: 0.75rem 1rem; border-radius: 12px; font-weight: 700; "
        f"background: {background}; color: {color};'>{text}</div>"
    )


def _render_status(payload: dict):
    drift_status = payload["drift"]["drift_alert"]
    drift_badge = _badge(
        f"{drift_status} | KS p-value: {payload['drift']['ks_pvalue']} | Score: {payload['drift']['drift_score']}",
        "bad" if drift_status == "DRIFT_DETECTED" else "good",
    )

    confidence_badge = _badge(
        "Low confidence — review recommended" if payload["low_confidence"] else "Confidence above review threshold",
        "warn" if payload["low_confidence"] else "good",
    )

    warning_text = "No warnings" if not payload["warning_flags"] else " | ".join(payload["warning_flags"])
    warning_badge = _badge(warning_text, "warn" if payload["warning_flags"] else "info")
    return drift_badge, confidence_badge, warning_badge


def _ops_summary(payload: dict):
    predicted = ", ".join(payload["predicted_labels"])
    return (
        "### Screening Summary\n"
        f"- Predicted findings: `{predicted}`\n"
        f"- Selected model: `{payload['selected_model']}` (arm `{payload['selected_arm']}`)\n"
        f"- Confidence: `{payload['confidence']}%`\n"
        f"- PyTorch latency: `{payload['pytorch_latency_ms']} ms`\n"
        f"- ONNX latency: `{payload['onnx_latency_ms']} ms`\n"
        f"- Latency delta (ONNX - PyTorch): `{payload['latency_delta_ms']} ms`\n"
        f"- Total request latency: `{payload['request_latency_ms']} ms`\n"
        f"- Recommendation: {payload['recommendation']}"
    )


def _input_summary(payload: dict):
    summary = payload["input_summary"]
    return (
        "### Input Validation Summary\n"
        f"- Size: `{summary['width']} x {summary['height']}`\n"
        f"- Aspect ratio: `{summary['aspect_ratio']}`\n"
        f"- Mean intensity: `{summary['pixel_mean']}`\n"
        f"- Pixel std: `{summary['pixel_std']}`\n"
        f"- Channel delta: `{summary['channel_delta']}`"
    )


def predict_image(image: Image.Image, api_url: str):
    try:
        payload = _post_image(image, api_url)
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise gr.Error(f"Backend request failed: {detail}") from exc
    except requests.RequestException as exc:
        raise gr.Error(f"Could not reach the FastAPI backend at {api_url}.") from exc

    drift_badge, confidence_badge, warning_badge = _render_status(payload)
    latency_plot = _build_latency_plot(payload.get("recent_history", []))
    return (
        ", ".join(payload["predicted_labels"]),
        f"{payload['confidence']}%",
        payload["selected_model"],
        f"{payload['latency_delta_ms']} ms",
        drift_badge,
        confidence_badge,
        warning_badge,
        _ops_summary(payload),
        _input_summary(payload),
        _build_predictions_table(payload["top_predictions"]),
        latency_plot,
    )


def refresh_health(api_url: str):
    try:
        payload = _get_json(_health_url(api_url))
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise gr.Error(f"Backend request failed: {detail}") from exc
    except requests.RequestException as exc:
        raise gr.Error(f"Could not reach the FastAPI backend at {_health_url(api_url)}.") from exc

    health_md = (
        "### System Health\n"
        f"- Status: `{payload['status']}`\n"
        f"- Profile: `{payload['profile']}`\n"
        f"- Active ONNX artifact: `{payload['active_onnx_model']}`\n"
        f"- PyTorch loaded: `{payload['pytorch_model_loaded']}`\n"
        f"- ONNX loaded: `{payload['onnx_model_loaded']}`\n"
        f"- Multi-label mode: `{payload['multi_label']}`\n"
        f"- Class count: `{payload['class_count']}`"
    )

    metrics = payload.get("training_metrics", {})
    metrics_md = (
        "### Training Snapshot\n"
        f"- Architecture: `{metrics.get('architecture', 'n/a')}`\n"
        f"- Best val micro-F1: `{metrics.get('best_val_micro_f1', 'n/a')}`\n"
        f"- Test micro-F1: `{metrics.get('test_micro_f1', metrics.get('test_f1', 'n/a'))}`\n"
        f"- Test macro-F1: `{metrics.get('test_macro_f1', 'n/a')}`\n"
        f"- Test macro ROC-AUC: `{metrics.get('test_macro_roc_auc', metrics.get('test_roc_auc', 'n/a'))}`"
    )

    recent_plot = _build_latency_plot(payload.get("recent_requests", []))
    return health_md, metrics_md, recent_plot


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css="""
        .hero {
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            color: white;
            border-radius: 20px;
            padding: 1.2rem 1.25rem;
            margin-bottom: 1rem;
        }
        """,
    ) as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>PneumoOps</h1>
              <p>Production-style MLOps demo for chest X-ray classification with multi-label screening,
              weighted A/B routing, drift monitoring, dual-model latency benchmarking, and deployable ops telemetry.</p>
            </div>
            """
        )
        gr.Markdown(
            """
            This interface is built to feel more like an ops console than a toy classifier. Every request surfaces
            model routing, latency comparison, drift status, and a confidence review signal alongside the findings.
            """
        )

        with gr.Accordion("Advanced Settings", open=False):
            api_url = gr.Textbox(label="Backend Predict URL", value=DEFAULT_API_URL)

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Chest X-ray", height=360)

        with gr.Row():
            submit_btn = gr.Button("Run Screening", variant="primary")
            refresh_btn = gr.Button("Refresh Health")

        with gr.Tabs():
            with gr.Tab("Screening Result"):
                with gr.Row():
                    findings = gr.Textbox(label="Predicted Pathologies", interactive=False)
                    confidence = gr.Textbox(label="Top Confidence", interactive=False)
                    model_used = gr.Textbox(label="Model Used", interactive=False)
                    latency_delta = gr.Textbox(label="Latency Delta", interactive=False)

                drift_badge = gr.HTML(label="Drift")
                confidence_badge = gr.HTML(label="Confidence Review")
                warning_badge = gr.HTML(label="Warnings")
                ops_summary = gr.Markdown()
                input_summary = gr.Markdown()
                predictions_table = gr.Dataframe(
                    headers=["Label", "Confidence", "Threshold"],
                    datatype=["str", "str", "str"],
                    label="Top Pathology Scores",
                    interactive=False,
                )

            with gr.Tab("Latency Monitor"):
                latency_plot = gr.Plot(label="Last 20 Requests")

            with gr.Tab("System Health"):
                health_summary = gr.Markdown()
                metrics_summary = gr.Markdown()
                health_plot = gr.Plot(label="Service Latency Trend")

        submit_btn.click(
            fn=predict_image,
            inputs=[image_input, api_url],
            outputs=[
                findings,
                confidence,
                model_used,
                latency_delta,
                drift_badge,
                confidence_badge,
                warning_badge,
                ops_summary,
                input_summary,
                predictions_table,
                latency_plot,
            ],
        )
        refresh_btn.click(
            fn=refresh_health,
            inputs=[api_url],
            outputs=[health_summary, metrics_summary, health_plot],
        )
        demo.load(
            fn=refresh_health,
            inputs=[api_url],
            outputs=[health_summary, metrics_summary, health_plot],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

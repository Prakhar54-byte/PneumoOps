import base64
import io
import os
import requests
import gradio as gr
from PIL import Image

PORT = os.getenv("PORT", "7860")
BACKEND_URL = os.getenv("BACKEND_PREDICT_URL", f"http://127.0.0.1:{PORT}/predict")

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

def render_top_metrics(payload: dict) -> tuple[str, str]:
    model_arm = payload.get("selected_model", "Unknown")
    latency_ms = payload.get("latency_ms", -1)
    
    latency_html = f'<div class="metric-value">{latency_ms} ms</div>'
    if latency_ms < 0:
        latency_html = f'''
        <div class="metric-value" style="color: #ef4444;">{latency_ms} ms</div>
        <div style="background: #452424; border-left: 4px solid #ef4444; padding: 8px 12px; border-radius: 4px; margin-top: 8px;">
            <div style="color:#fca5a5;font-weight:600;font-size:0.85rem;">Bug: negative latency</div>
            <div style="color:#fecaca;font-size:0.8rem;">Timing value is invalid — likely a clock sync issue.<br>Fix the timer before showing this to patients.</div>
        </div>
        '''
        
    arm_html = f'''
    <div class="custom-card">
        <div class="metric-title">MODEL (A/B ARM)</div>
        <div class="metric-value">{model_arm}</div>
    </div>
    '''
    lat_html = f'''
    <div class="custom-card">
        <div class="metric-title">INFERENCE LATENCY</div>
        {latency_html}
    </div>
    '''
    return arm_html, lat_html

def render_quality(payload: dict) -> str:
    drift = payload.get("drift", {})
    alert = drift.get("drift_alert", "NORMAL")
    if alert == "DRIFT_DETECTED":
        return '''
        <div class="custom-card" style="border-left: 4px solid #ef4444;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="metric-title">INPUT IMAGE QUALITY CHECK</div>
                <div style="background:#452424; color:#fca5a5; padding:4px 10px; border-radius:12px; font-size:0.75rem;">⚠ Out of Distribution</div>
            </div>
            <div style="font-size:1.1rem; font-weight:500; margin-top:4px;">Distribution differs from training baseline</div>
            <div style="background:#452424; padding:12px; border-radius:6px; margin-top:12px;">
                <div style="color:#fbbf24; font-weight:600; font-size:0.9rem; margin-bottom:4px;">What this means for the patient</div>
                <div style="color:#d1d5db; font-size:0.85rem;">The model has never seen images like this. Predictions may be unreliable. Proceed with extreme caution.</div>
            </div>
        </div>
        '''
    return '''
    <div class="custom-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="metric-title">INPUT IMAGE QUALITY CHECK</div>
            <div style="background:#064e3b; color:#34d399; padding:4px 10px; border-radius:12px; font-size:0.75rem;">✓ Acceptable</div>
        </div>
        <div style="font-size:1.1rem; font-weight:500; margin-top:4px; margin-bottom:12px;">Distribution matches training baseline</div>
        <div style="background:#453015; padding:12px; border-radius:6px; margin-top:8px; border-left: 4px solid #f59e0b;">
            <div style="color:#fbbf24; font-weight:600; font-size:0.9rem; margin-bottom:4px;">What this means for the patient</div>
            <div style="color:#d1d5db; font-size:0.85rem; line-height:1.4;">This <i>does not</i> mean your lungs are normal. It only means the uploaded X-ray image is technically readable by the model — the contrast, resolution, and framing look similar to images it was trained on.</div>
        </div>
    </div>
    '''

_TIER_COLORS = {
    "Pneumonia": "#ef4444", "Pneumothorax": "#ef4444", "Mass": "#ef4444", "Nodule": "#ef4444",
    "Effusion": "#f59e0b", "Cardiomegaly": "#f59e0b", "Consolidation": "#f59e0b",
}
_TIER_DESC = {
    "Pneumonia": "Infection inflaming air sacs, potentially filling with fluid.",
    "Pneumothorax": "Collapsed lung — air leaked into the space between lung and chest wall.",
    "Mass": "Larger opacity (>3cm). Not actionable at this score.",
    "Nodule": "A small round opacity — may be benign or require follow-up.",
    "Effusion": "Fluid build-up in the pleural space.",
    "Cardiomegaly": "Enlarged heart — could indicate underlying heart conditions.",
    "Consolidation": "Region of normally compressible lung tissue filled with liquid.",
    "Infiltration": "Fluid or tissue density where there should be air — could suggest infection, inflammation, or early pneumonia.",
}

def render_top3(payload: dict) -> str:
    top3 = payload.get("top_predictions", [])
    inconclusive = payload.get("inconclusive_scan", False)
    
    html = '<div class="custom-card">'
    html += '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">'
    html += '<div class="metric-title">TOP-3 PREDICTED PATHOLOGIES</div>'
    html += '<div style="font-size:0.8rem; color:#9ca3af;"><span style="color:#ef4444;">●</span> Critical &nbsp;&nbsp; <span style="color:#f59e0b;">●</span> Significant &nbsp;&nbsp; <span style="color:#3b82f6;">●</span> Standard</div>'
    html += '</div>'
    
    for p in top3:
        lbl = p["label"]
        conf = p["confidence"]
        col = _TIER_COLORS.get(lbl, "#3b82f6")
        desc = _TIER_DESC.get(lbl, "A generic chest anomaly flagged by the model.")
        
        conf_text = f"Confidence is very low ({conf}%)." if conf < 10 else f"Confidence: {conf}%."
        flagged = '<span style="background:#1e3a8a; color:#93c5fd; padding:4px 10px; border-radius:12px; font-size:0.75rem; margin-left:8px;">Flagged finding</span>' if p["detected"] else ''
        
        html += f'''
        <div style="background:#333333; padding:16px; border-radius:8px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="font-size:1.1rem; font-weight:600;"><span style="color:{col}; font-size:1.2rem;">●</span> {lbl} {flagged}</div>
                <div style="font-weight:600; font-size:1.2rem;">{conf}%</div>
            </div>
            <div style="width:100%; background:#4b5563; height:8px; border-radius:4px; overflow:hidden; margin-bottom:12px;">
                <div style="width:{conf}%; background:{col}; height:100%;"></div>
            </div>
            <div style="font-size:0.85rem; color:#d1d5db;">{desc} <strong>{conf_text}</strong></div>
        </div>
        '''
        
    if inconclusive:
        html += '''
        <div style="background:#452424; border:1px solid #7f1d1d; padding:16px; border-radius:8px; margin-top:16px;">
            <div style="color:#fca5a5; font-weight:600; margin-bottom:4px;">All confidence scores are below 10% — interpret with extreme caution</div>
            <div style="color:#fecaca; font-size:0.9rem; line-height:1.4;">Scores this low indicate the model is uncertain. A well-calibrated model should ideally show >50% before a finding is clinically noteworthy. These results should <i>not</i> drive any clinical decision without radiologist review.</div>
        </div>
        '''
    html += '</div>'
    return html

def render_14_classes(payload: dict) -> str:
    all_preds = payload.get("all_predictions", [])
    
    html = '<div class="custom-card">'
    html += '<div class="metric-title" style="margin-bottom:12px;">ALL 14 SCREENED CONDITIONS</div>'
    html += '<div style="display:flex; flex-wrap:wrap; gap:8px;">'
    
    for p in all_preds:
        lbl = p["label"]
        conf = p["confidence"]
        det = p["detected"]
        
        if det:
            bg, color = "#78350f", "#fcd34d" # Yellow highlight
        else:
            bg, color = "#374151", "#9ca3af" # Gray
            
        text = f"{lbl} {conf}%" if det else lbl
        html += f'<div style="background:{bg}; color:{color}; padding:6px 14px; border-radius:16px; font-size:0.85rem; font-weight:500; border:1px solid #4b5563;">{text}</div>'
        
    html += '</div>'
    html += '<div style="font-size:0.8rem; color:#6b7280; margin-top:16px;">Grey chips = below detection threshold. Not shown does not mean absent.</div>'
    html += '</div>'
    return html

def render_calibration_info() -> str:
    """Provides historical benchmark context as requested in the redesign proposal."""
    return '''
    <div class="custom-card">
        <div class="metric-title">MODEL CALIBRATION & BENCHMARKING</div>
        <div style="font-size:0.88rem; color:#d1d5db; line-height:1.6;">
            Model calibrated on NIH ChestX-ray14. Historical AUC benchmarks:
            <ul style="margin-top:8px; padding-left:18px; margin-bottom:8px;">
                <li><b style="color:#34d399;">Cardiomegaly:</b> ~0.81 AUC (Highly reliable)</li>
                <li><b style="color:#fbbf24;">Infiltration:</b> ~0.70 AUC (Historically lower reliability)</li>
                <li><b style="color:#fbbf24;">Pneumonia:</b> ~0.68 AUC</li>
            </ul>
            <i>Note for Clinicians:</i> Predictions for "Infiltration" and "Pneumonia" carry higher uncertainty due to lower historical AUC. Radiologist verification is mandatory.
        </div>
    </div>
    '''

def predict(image: Image.Image, api_url: str):
    if image is None:
        raise gr.Error("Please upload a chest X-ray image.")
    try:
        payload = call_backend(image, api_url)
    except Exception as exc:
        raise gr.Error(f"Backend error: {str(exc)}") from exc

    arm_html, lat_html = render_top_metrics(payload)
    qc_html = render_quality(payload)
    top3_html = render_top3(payload)
    classes_html = render_14_classes(payload)
    calib_html = render_calibration_info()
    
    cam_b64 = payload.get("cam_b64")
    if cam_b64:
        img_data = base64.b64decode(cam_b64)
        out_image = Image.open(io.BytesIO(img_data))
    else:
        out_image = image
        
    return arm_html, lat_html, qc_html, top3_html, classes_html, calib_html, out_image

CSS = """
body, .gradio-container {
    background-color: #1e1e1e !important;
    color: #e5e5e5 !important;
}
.custom-card {
    background: #2b2b2b;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #3f3f46;
    margin-bottom: 16px;
}
.metric-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9ca3af;
    margin-bottom: 6px;
    font-weight: 600;
}
.metric-value {
    font-size: 1.25rem;
    font-weight: 500;
    color: #f4f4f5;
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=CSS, title="PneumoOps Redesigned") as demo:
    with gr.Accordion("⚙️ Backend Settings", open=False):
        api_url = gr.Textbox(label="Backend URL", value=BACKEND_URL)
        
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Chest X-Ray")
            submit_btn = gr.Button("🔬 Run Screening", variant="primary", size="lg")
            
            gr.HTML('<div style="margin-top:20px; font-weight:600; color:#e5e5e5; font-size:1.1rem;">Explainability (Grad-CAM)</div>')
            gr.HTML("<div style='color:#9ca3af; font-size:0.85rem; margin-bottom:8px;'>Heatmap identifying the regions that drove the model's top prediction.</div>")
            analyzed_img_out = gr.Image(label="Analyzed X-Ray", interactive=False)
            
        with gr.Column(scale=2):
            with gr.Row():
                arm_out = gr.HTML()
                lat_out = gr.HTML()
                
            qc_out = gr.HTML()
            top3_out = gr.HTML()
            classes_out = gr.HTML()
            calib_out = gr.HTML()
            
            gr.HTML('''
            <div class="custom-card">
                <div style="font-weight:600; font-size:0.95rem; margin-bottom:4px; display:flex; align-items:center; gap:8px;">
                    ⚖️ Responsible AI notice
                </div>
                <div style="color:#9ca3af; font-size:0.9rem; line-height:1.5;">
                    This system <i>assists in prediction</i>, it does not confirm diagnoses. Results are statistical estimates and must be reviewed by a qualified radiologist or clinician before any medical decision is made. Do not use these results to self-diagnose.
                </div>
            </div>
            ''')

    submit_btn.click(
        fn=predict,
        inputs=[image_input, api_url],
        outputs=[arm_out, lat_out, qc_out, top3_out, classes_out, calib_out, analyzed_img_out],
    )

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)

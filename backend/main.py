import asyncio
import io
import json
import logging
import os
import random
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from scipy.stats import ks_2samp
import matplotlib.cm as cm
import base64

# Heavy ML deps are optional — not installed on the Railway proxy node.
try:
    import torch
    import onnxruntime as ort
    from torchvision import transforms
    from torchvision.models import mobilenet_v3_small, efficientnet_b0
    from model_utils import CalibratedModel

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parents[1]
LEGACY_MODEL_DIR = BASE_DIR / "models"
REALWORLD_MODEL_DIR = BASE_DIR / "models" / "realworld_efficientnet_b0"
CHESTMNIST_MODEL_DIR = BASE_DIR / "models" / "chestmnist_mobilenetv3"

PROFILE = os.getenv("PNEUMOOPS_PROFILE", "chestmnist").lower()
_DEFAULT_MODEL_DIR = {
    "realworld": REALWORLD_MODEL_DIR,
    "chestmnist": CHESTMNIST_MODEL_DIR,
}.get(PROFILE, LEGACY_MODEL_DIR)
MODEL_DIR = Path(os.getenv("PNEUMOOPS_MODEL_DIR", str(_DEFAULT_MODEL_DIR)))
REQUEST_LOG_HISTORY = deque(maxlen=20)

API_KEY = os.getenv("PNEUMOOPS_API_KEY")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("PNEUMOOPS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
TRAFFIC_WEIGHTS = {"pytorch": 60, "onnx": 40}
# When set, all model inference is forwarded to this HF Spaces URL instead of local models.
# Example: https://prakhar54-byte-pneumoops.hf.space
HF_SPACES_URL = os.getenv("HF_SPACES_URL", "").rstrip("/")
COLLECT_DATA = os.getenv("PNEUMOOPS_COLLECT_DATA", "true").lower() == "true"
COLLECT_DIR = BASE_DIR / "data" / "collected_images"
LOW_CONFIDENCE_THRESHOLD = float(
    os.getenv("PNEUMOOPS_LOW_CONFIDENCE_THRESHOLD", "0.60")
)
MIN_UPLOAD_EDGE = int(os.getenv("PNEUMOOPS_MIN_UPLOAD_EDGE", "96"))
MAX_CHANNEL_DELTA = float(os.getenv("PNEUMOOPS_MAX_CHANNEL_DELTA", "0.08"))
MIN_ASPECT_RATIO = float(os.getenv("PNEUMOOPS_MIN_ASPECT_RATIO", "0.6"))
MAX_ASPECT_RATIO = float(os.getenv("PNEUMOOPS_MAX_ASPECT_RATIO", "1.6"))

REQUEST_COUNTER = Counter(
    "pneumoops_requests_total",
    "Total inference requests served by PneumoOps.",
    ["model", "status"],
)
LATENCY_HISTOGRAM = Histogram(
    "pneumoops_inference_latency_ms",
    "Inference latency per model in milliseconds.",
    ["model"],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000),
)
DRIFT_COUNTER = Counter(
    "pneumoops_drift_alerts_total",
    "Number of drift alerts emitted by PneumoOps.",
    ["status"],
)
DISEASE_PREDICTION_COUNTER = Counter(
    "pneumoops_disease_predictions_total",
    "Per-disease prediction counts for production monitoring.",
    ["disease", "model"],
)

logger = logging.getLogger("pneumoops")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def resolve_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path | None, fallback: dict | None = None) -> dict:
    fallback_dict = {} if fallback is None else fallback
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON file {path}, using fallback.")
            return fallback_dict
    return fallback_dict


def resolve_runtime_paths(model_dir: Path) -> dict[str, Path | None]:
    checkpoint_path = resolve_path(
        [
            model_dir / "mobilenetv3_chestmnist.pth",  # chestmnist profile
            model_dir / "realworld_efficientnet_b0.pth",
            model_dir / "pneumo_model.pth",
        ]
    )
    onnx_report_path = resolve_path(
        [
            model_dir / "onnx_export_report.json",
            LEGACY_MODEL_DIR / "onnx_export_report.json",
        ]
    )
    onnx_report = load_json(onnx_report_path)

    base_onnx = onnx_report.get("base_onnx")
    optimized_onnx = onnx_report.get("optimized_onnx")
    serving_onnx = onnx_report.get("serving_onnx")

    onnx_candidates = []
    if serving_onnx:
        onnx_candidates.append(model_dir / serving_onnx)
    if optimized_onnx:
        onnx_candidates.append(model_dir / optimized_onnx)
    if base_onnx:
        onnx_candidates.append(model_dir / base_onnx)
    onnx_candidates.extend(
        [
            # chestmnist profile
            model_dir / "mobilenetv3_chestmnist.onnx",
            # realworld profile
            model_dir / "realworld_efficientnet_b0_quantized.onnx",
            model_dir / "realworld_efficientnet_b0_optimized.onnx",
            model_dir / "realworld_efficientnet_b0.onnx",
            model_dir / "pneumo_model_quantized.onnx",
            model_dir / "pneumo_model_optimized.onnx",
            model_dir / "pneumo_model.onnx",
        ]
    )

    return {
        "checkpoint": checkpoint_path,
        "onnx": resolve_path(onnx_candidates),
        "training_metrics": resolve_path(
            [
                model_dir / "training_metrics.json",
                LEGACY_MODEL_DIR / "training_metrics.json",
            ]
        ),
        "baseline_stats": resolve_path(
            [
                model_dir / "baseline_stats.json",
                LEGACY_MODEL_DIR / "baseline_stats.json",
            ]
        ),
        "onnx_export_report": onnx_report_path,
    }


def download_models_if_missing() -> None:
    """Download model weights from HF Hub at startup if not present locally.
    Only runs on the HF Spaces inference node (HF_SPACES_URL not set)."""
    if HF_SPACES_URL:
        return  # proxy node — no local models needed
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    hf_model_repo = os.getenv("HF_MODEL_REPO", "Prakhar54-byte/pneumoops-chestmnist")
    hf_token = os.getenv("HF_TOKEN")
    files = ["mobilenetv3_chestmnist.pth", "mobilenetv3_chestmnist.onnx"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in files:
        dest = MODEL_DIR / filename
        if dest.exists():
            continue
        try:
            logger.info(f"Downloading {filename} from {hf_model_repo}...")
            hf_hub_download(
                repo_id=hf_model_repo,
                filename=filename,
                local_dir=str(MODEL_DIR),
                token=hf_token,
            )
            logger.info(f"Downloaded {filename} successfully.")
        except Exception as exc:
            logger.warning(f"Could not download {filename}: {exc}")


download_models_if_missing()
RUNTIME_PATHS = resolve_runtime_paths(MODEL_DIR)


def load_checkpoint_metadata(checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None or not checkpoint_path.exists():
        # Proxy/backend-only deployments have no model weights but do have
        # training_metrics.json — use it so class names and thresholds are correct.
        tm_path = MODEL_DIR / "training_metrics.json"
        tm = load_json(tm_path)
        if tm.get("class_names"):
            class_names = tm["class_names"]
            n = len(class_names)
            return {
                "architecture": tm.get("architecture", "mobilenet_v3_small"),
                "class_names": class_names,
                "image_size": int(tm.get("image_size", 224)),
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
                "thresholds": tm.get("thresholds", [0.5] * n),
                "logit_temperature": 1.0,
                "multi_label": bool(tm.get("multi_label", True)),
            }
        return {
            "architecture": "mobilenet_v3_small",
            "class_names": ["Normal", "Pneumonia"],
            "image_size": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "thresholds": [0.5, 0.5],
            "logit_temperature": 1.0,
            "multi_label": False,
        }

    # If torch is not available, fall back to loading metadata from training_metrics.json
    if not _TORCH_AVAILABLE:
        tm_path = MODEL_DIR / "training_metrics.json"
        tm = load_json(tm_path)
        class_names = tm.get("class_names", ["Normal", "Pneumonia"])
        n = len(class_names)
        return {
            "architecture": tm.get("architecture", "mobilenet_v3_small"),
            "class_names": class_names,
            "image_size": int(tm.get("image_size", 224)),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "thresholds": tm.get("thresholds", [0.5] * n),
            "logit_temperature": 1.0,
            "multi_label": bool(tm.get("multi_label", True)),
        }

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return {
            "state_dict": payload["model_state_dict"],
            "architecture": payload.get("architecture", "efficientnet_b0"),
            "class_names": payload.get("class_names", ["No Finding"]),
            "image_size": int(payload.get("image_size", 320)),
            "normalize_mean": payload.get("normalize_mean", [0.485, 0.456, 0.406]),
            "normalize_std": payload.get("normalize_std", [0.229, 0.224, 0.225]),
            "thresholds": payload.get(
                "thresholds", [0.5] * len(payload.get("class_names", ["No Finding"]))
            ),
            "logit_temperature": float(payload.get("logit_temperature", 1.0)),
            "multi_label": bool(payload.get("multi_label", True)),
        }

    # Plain state_dict (e.g. from train_chestmnist.py) — pull metadata from training_metrics.json
    tm_path = MODEL_DIR / "training_metrics.json"
    tm = load_json(tm_path)
    class_names = tm.get("class_names", ["Normal", "Pneumonia"])
    n = len(class_names)
    return {
        "state_dict": payload,
        "architecture": tm.get("architecture", "mobilenet_v3_small"),
        "class_names": class_names,
        "image_size": int(tm.get("image_size", 224)),
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "thresholds": tm.get("thresholds", [0.5] * n),
        "logit_temperature": 1.0,
        "multi_label": bool(tm.get("multi_label", True)),
    }


MODEL_METADATA = load_checkpoint_metadata(RUNTIME_PATHS["checkpoint"])
CLASS_NAMES = MODEL_METADATA["class_names"]
MULTI_LABEL = bool(MODEL_METADATA["multi_label"])
IMAGE_SIZE = int(MODEL_METADATA["image_size"])
THRESHOLDS = np.asarray(MODEL_METADATA["thresholds"], dtype=np.float32)
LOGIT_TEMPERATURE = float(MODEL_METADATA.get("logit_temperature", 1.0))

TRANSFORM = (
    transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MODEL_METADATA["normalize_mean"],
                std=MODEL_METADATA["normalize_std"],
            ),
        ]
    )
    if _TORCH_AVAILABLE
    else None
)

BASELINE_STATS = load_json(
    RUNTIME_PATHS["baseline_stats"],
    fallback={
        "pixel_mean_mean": 0.5,
        "pixel_mean_std": 0.1,
        "pixel_std_mean": 0.2,
        "pixel_std_std": 0.05,
        "histogram_bins": 32,
        "histogram_mean": [1.0 / 32.0] * 32,
        "pixel_reference_sample": [],
        "drift_threshold": 1.2,
        "drift_ks_pvalue_threshold": 0.05,
        "class_names": CLASS_NAMES,
    },
)


def build_model() -> Any:
    checkpoint_path = RUNTIME_PATHS["checkpoint"]
    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    architecture = MODEL_METADATA["architecture"]
    num_outputs = len(CLASS_NAMES)
    if architecture == "efficientnet_b0":
        base_model = efficientnet_b0(weights=None)
        base_model.classifier[1] = torch.nn.Linear(
            base_model.classifier[1].in_features, num_outputs
        )
    else:
        base_model = mobilenet_v3_small(weights=None)
        base_model.classifier[3] = torch.nn.Linear(
            base_model.classifier[3].in_features, num_outputs
        )

    base_model.load_state_dict(MODEL_METADATA["state_dict"])
    model = CalibratedModel(base_model, LOGIT_TEMPERATURE)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def build_onnx_session():
    onnx_path = RUNTIME_PATHS["onnx"]
    if onnx_path is None or not onnx_path.exists():
        return None, None
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session, onnx_path.name


DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _TORCH_AVAILABLE
    else None
)
PYTORCH_MODEL = build_model() if _TORCH_AVAILABLE else None
ONNX_SESSION, ACTIVE_ONNX_MODEL_NAME = (
    build_onnx_session() if _TORCH_AVAILABLE else (None, None)
)

app = FastAPI(
    title="PneumoOps API",
    description="Production-style MLOps pipeline for medical image classification with A/B routing and drift monitoring.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if API_KEY and request.url.path not in {"/health", "/metrics", "/"}:
        provided = request.headers.get("x-api-key")
        if provided != API_KEY:
            return Response(content="Unauthorized", status_code=401)
    return await call_next(request)


def load_image_from_upload(upload: UploadFile) -> Image.Image:
    try:
        content = upload.file.read()
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        ) from exc


def summarize_image(image: Image.Image) -> dict[str, Any]:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    channel_delta = (
        float(
            np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1]))
            + np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2]))
            + np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 2]))
        )
        / 3.0
    )
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / max(height, 1), 4),
        "pixel_mean": round(float(np.mean(gray)), 6),
        "pixel_std": round(float(np.std(gray)), 6),
        "pixel_min": round(float(np.min(gray)), 6),
        "pixel_max": round(float(np.max(gray)), 6),
        "channel_delta": round(channel_delta, 6),
    }


def validate_image(image: Image.Image, summary: dict[str, Any]) -> None:
    if min(summary["width"], summary["height"]) < MIN_UPLOAD_EDGE:
        raise HTTPException(
            status_code=400,
            detail=f"Input is too small for robust X-ray screening. Minimum edge is {MIN_UPLOAD_EDGE}px.",
        )
    if not (MIN_ASPECT_RATIO <= summary["aspect_ratio"] <= MAX_ASPECT_RATIO):
        raise HTTPException(
            status_code=400,
            detail="Input aspect ratio is outside the expected chest X-ray range.",
        )
    if summary["channel_delta"] > MAX_CHANNEL_DELTA:
        raise HTTPException(
            status_code=400,
            detail="Input appears to be a color image instead of a grayscale-style radiograph.",
        )


def calculate_drift(image: Image.Image) -> dict[str, Any]:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    image_mean = float(np.mean(gray))
    image_std = float(np.std(gray))
    hist_bins = int(BASELINE_STATS.get("histogram_bins", 32))
    hist, _ = np.histogram(gray, bins=hist_bins, range=(0.0, 1.0), density=True)
    baseline_hist = np.asarray(
        BASELINE_STATS.get("histogram_mean", [1.0 / hist_bins] * hist_bins),
        dtype=np.float32,
    )

    mean_z = abs(image_mean - BASELINE_STATS.get("pixel_mean_mean", 0.5)) / max(
        BASELINE_STATS.get("pixel_mean_std", 0.1), 1e-6
    )
    std_z = abs(image_std - BASELINE_STATS.get("pixel_std_mean", 0.2)) / max(
        BASELINE_STATS.get("pixel_std_std", 0.05), 1e-6
    )
    hist_distance = float(np.mean(np.abs(hist - baseline_hist)))
    drift_score = round(0.35 * mean_z + 0.35 * std_z + 0.30 * hist_distance, 6)

    reference_sample = np.asarray(
        BASELINE_STATS.get("pixel_reference_sample", []), dtype=np.float32
    )
    incoming_sample = gray.reshape(-1)
    if reference_sample.size > 0:
        sample_size = min(len(incoming_sample), len(reference_sample), 4096)
        incoming_idx = np.random.choice(
            len(incoming_sample), size=sample_size, replace=False
        )
        reference_idx = np.random.choice(
            len(reference_sample), size=sample_size, replace=False
        )
        _, ks_pvalue = ks_2samp(
            incoming_sample[incoming_idx], reference_sample[reference_idx]
        )
        ks_pvalue = float(ks_pvalue)
    else:
        ks_pvalue = 1.0

    drift_detected = ks_pvalue < float(
        BASELINE_STATS.get("drift_ks_pvalue_threshold", 0.05)
    ) or drift_score > float(BASELINE_STATS.get("drift_threshold", 1.2))
    return {
        "drift_alert": "DRIFT_DETECTED" if drift_detected else "NORMAL",
        "drift_score": drift_score,
        "ks_pvalue": round(ks_pvalue, 6),
        "mean_z": round(float(mean_z), 6),
        "std_z": round(float(std_z), 6),
        "histogram_distance": round(hist_distance, 6),
    }


def postprocess_probabilities(probabilities: np.ndarray) -> dict[str, Any]:
    probabilities = probabilities.astype(np.float32)
    if MULTI_LABEL:
        predicted_indices = [
            index
            for index, value in enumerate(probabilities)
            if value >= THRESHOLDS[index]
        ]
        if not predicted_indices:
            predicted_indices = [int(np.argmax(probabilities))]
        predicted_labels = [CLASS_NAMES[index] for index in predicted_indices]
    else:
        predicted_index = int(np.argmax(probabilities))
        predicted_indices = [predicted_index]
        predicted_labels = [CLASS_NAMES[predicted_index]]

    all_predictions = [
        {
            "label": CLASS_NAMES[index],
            "confidence": round(float(probabilities[index]) * 100, 2),
            "threshold": round(float(THRESHOLDS[index]) * 100, 2)
            if index < len(THRESHOLDS)
            else 50.0,
            "detected": index in predicted_indices,
        }
        for index in range(len(CLASS_NAMES))
    ]
    sorted_pairs = sorted(
        all_predictions, key=lambda item: item["confidence"], reverse=True
    )
    top_confidence = sorted_pairs[0]["confidence"] if sorted_pairs else 0.0

    return {
        "predicted_labels": predicted_labels,
        "all_predictions": all_predictions, "summary": "Clinical Review Recommended: No findings reached the 30% threshold, but observation is advised for top results.",
        "top_predictions": sorted_pairs[: min(5, len(sorted_pairs))],
        "max_confidence": top_confidence,
        "low_confidence": top_confidence < (LOW_CONFIDENCE_THRESHOLD * 100.0),
        "inconclusive_scan": top_confidence < 10.0,
    }


class CAMHook:
    """Hook to capture gradients and activations for Grad-CAM."""

    def __init__(self, module):
        self.hook_f = module.register_forward_hook(self.hook_fn_fwd)
        self.hook_b = module.register_full_backward_hook(self.hook_fn_bwd)
        self.features = None
        self.gradients = None

    def hook_fn_fwd(self, module, input, output):
        self.features = output.detach()

    def hook_fn_bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.hook_f.remove()
        self.hook_b.remove()

def generate_cam_overlay(image: Image.Image, cam_tensor: torch.Tensor) -> str:
    try:
        cam = cam_tensor.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-12)
        w, h = image.size
        cam_img = Image.fromarray(np.uint8(255 * cam)).resize((w, h), Image.Resampling.LANCZOS)
        colormap = cm.get_cmap('jet')(np.array(cam_img) / 255.0)[:, :, :3]
        heatmap = np.uint8(255 * colormap)
        overlay = np.uint8(0.5 * np.array(image.convert('RGB')) + 0.5 * heatmap)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')


def run_pytorch_inference(image: Image.Image) -> dict[str, Any]:
    if PYTORCH_MODEL is None:
        raise RuntimeError("PyTorch checkpoint is missing.")

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    target_layer = None
    if hasattr(PYTORCH_MODEL.base_model, "features"):
        features = PYTORCH_MODEL.base_model.features
        def find_last_conv(m):
            last_conv = None
            for name, module in m.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            return last_conv
        target_layer = find_last_conv(features)
        if target_layer is None:
            target_layer = features[-1]
            logger.info(f"Using features[-1] as fallback target layer: {type(target_layer).__name__}")
        else:
            logger.info(f"Found deep Conv2d target layer: {type(target_layer).__name__}")
    else:
        logger.warning("No features attribute found on model for CAM")

    cam_hook = None
    if target_layer is not None:
        try:
            cam_hook = CAMHook(target_layer)
            logger.info(f"CAM hook registered on {type(target_layer).__name__}")
        except Exception as e:
            logger.error(f"Failed to register CAM hook: {e}")
    else:
        logger.warning("No target layer found for CAM")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.set_grad_enabled(True):
        logits = PYTORCH_MODEL(tensor)
        probs_tensor = (torch.sigmoid(logits).squeeze(0) if MULTI_LABEL else torch.softmax(logits, dim=1).squeeze(0))
        probabilities = probs_tensor.detach().cpu().numpy()
        class_idx = int(np.argmax(probabilities))
        
        cam_b64 = None
        if cam_hook:
            try:
                logger.info(f"Computing CAM for class index {class_idx}")
                PYTORCH_MODEL.zero_grad()
                logits[0, class_idx].backward(retain_graph=True)
                if cam_hook.features is not None and cam_hook.gradients is not None:
                    weights = torch.mean(cam_hook.gradients, dim=[0, 2, 3], keepdim=True)
                    cam_tensor = torch.sum(weights * cam_hook.features, dim=1).squeeze()
                    cam_b64 = generate_cam_overlay(image, cam_tensor)
                    logger.info("CAM overlay generated successfully")
                else:
                    logger.warning(f"CAM missing data: features={cam_hook.features is not None}, gradients={cam_hook.gradients is not None}")
            except Exception as e:
                logger.error(f"CAM computation failed: {e}")
            finally:
                cam_hook.remove()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "model_key": "pytorch",
        "model_used": "Baseline PyTorch",
        "latency_ms": latency_ms,
        "probabilities": probabilities.tolist(),
        "cam_b64": cam_b64,
        **postprocess_probabilities(probabilities),
    }


def run_onnx_inference(image: Image.Image) -> dict[str, Any]:
    if ONNX_SESSION is None:
        raise RuntimeError("ONNX artifact is missing.")

    tensor = TRANSFORM(image).unsqueeze(0).numpy().astype(np.float32)
    input_name = ONNX_SESSION.get_inputs()[0].name
    start = time.perf_counter()
    outputs = ONNX_SESSION.run(None, {input_name: tensor})
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logits = torch.from_numpy(outputs[0]).squeeze(0)
    probabilities = (
        torch.sigmoid(logits).numpy()
        if MULTI_LABEL
        else torch.softmax(logits, dim=0).numpy()
    )
    return {
        "model_key": "onnx",
        "model_used": "Optimized ONNX",
        "latency_ms": latency_ms,
        "probabilities": probabilities.tolist(),
        **postprocess_probabilities(probabilities),
    }


async def _run_local_inference(image: Image.Image) -> dict[str, Any]:
    """Run both models in-process — used when model weights are available locally."""

    async def safe_call(model_name: str, fn):
        try:
            result = await asyncio.to_thread(fn, image)
            LATENCY_HISTOGRAM.labels(model=model_name).observe(result["latency_ms"])
            return result
        except Exception as exc:
            return {"model_key": model_name, "error": str(exc)}

    pytorch_result, onnx_result = await asyncio.gather(
        safe_call("pytorch", run_pytorch_inference),
        safe_call("onnx", run_onnx_inference),
    )
    return {"pytorch": pytorch_result, "onnx": onnx_result}


async def _call_hf_infer(image: Image.Image) -> dict[str, Any]:
    """Forward dual-model inference to the HF Spaces node via /infer."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            f"{HF_SPACES_URL}/infer",
            files={"file": ("xray.png", buf.read(), "image/png")},
        )
        resp.raise_for_status()
    data = resp.json()

    result: dict[str, Any] = {}
    for key in ("pytorch", "onnx"):
        raw = data.get(key) or {}
        if "error" in raw:
            result[key] = {"model_key": key, "error": raw["error"]}
        else:
            probs = np.asarray(raw.get("probabilities", []), dtype=np.float32)
            latency = float(raw.get("latency_ms") or 0.0)
            LATENCY_HISTOGRAM.labels(model=key).observe(latency)
            result[key] = {
                "model_key": key,
                "model_used": "Baseline PyTorch"
                if key == "pytorch"
                else "Optimized ONNX",
                "latency_ms": latency,
                "probabilities": probs.tolist(),
                **postprocess_probabilities(probs),
            }
    return result


async def benchmark_both_models(image: Image.Image) -> dict[str, Any]:
    """Route dual-model inference to HF Spaces (proxy mode) or local models."""
    if HF_SPACES_URL:
        return await _call_hf_infer(image)
    return await _run_local_inference(image)


def build_recommendation(
    selected_result: dict[str, Any],
    drift_result: dict[str, Any],
    dual_results: dict[str, Any],
) -> str:
    if drift_result["drift_alert"] == "DRIFT_DETECTED":
        return "Input distribution differs from the stored training baseline. Manual review is recommended before trusting this result."
    if selected_result["low_confidence"]:
        return "Prediction confidence is below the review threshold. Treat this as low confidence and escalate for human review."

    other_key = "onnx" if selected_result["model_key"] == "pytorch" else "pytorch"
    other_result = dual_results.get(other_key, {})
    if (
        other_result.get("predicted_labels")
        and other_result["predicted_labels"] != selected_result["predicted_labels"]
    ):
        return "The two serving paths disagree on the predicted findings. Use this as a monitoring alert and fall back to manual review."

    return "Use this output as a triage aid only. PneumoOps monitors latency and drift, but it is not a clinical decision-maker."


def append_history(entry: dict[str, Any]) -> None:
    REQUEST_LOG_HISTORY.append(entry)


def emit_structured_log(payload: dict[str, Any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=True))


def save_prediction_data(image: Image.Image, payload: dict[str, Any]) -> None:
    """Save anonymized image and prediction data for model fine-tuning."""
    if not COLLECT_DATA:
        return
    try:
        COLLECT_DIR.mkdir(parents=True, exist_ok=True)
        record_id = str(uuid.uuid4())

        # Save image
        img_path = COLLECT_DIR / f"{record_id}.png"
        image.save(img_path, format="PNG")

        # Save prediction payload
        json_path = COLLECT_DIR / f"{record_id}.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save collection data: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "deployment_mode": "proxy" if HF_SPACES_URL else "local",
        "hf_spaces_url": HF_SPACES_URL or None,
        "profile": PROFILE,
        "model_dir": str(MODEL_DIR),
        "pytorch_model_loaded": PYTORCH_MODEL is not None,
        "onnx_model_loaded": ONNX_SESSION is not None,
        "active_onnx_model": ACTIVE_ONNX_MODEL_NAME,
        "class_count": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "multi_label": MULTI_LABEL,
        "logit_temperature": LOGIT_TEMPERATURE,
        "training_metrics": load_json(RUNTIME_PATHS["training_metrics"]),
        "onnx_export_report": load_json(RUNTIME_PATHS["onnx_export_report"]),
        "recent_requests": list(REQUEST_LOG_HISTORY),
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/history")
def history():
    return {"recent_requests": list(REQUEST_LOG_HISTORY)}


@app.get("/metrics/class-rates")
def class_prediction_rates():
    """Return per-class prediction rates from the rolling request history."""
    history_list = list(REQUEST_LOG_HISTORY)
    total = max(len(history_list), 1)
    rates: dict[str, float] = {label: 0.0 for label in CLASS_NAMES}
    avg_confidence: dict[str, list] = {label: [] for label in CLASS_NAMES}

    for entry in history_list:
        per_class = entry.get("per_class_predictions", {})
        per_class_probs = entry.get("per_class_probabilities", {})
        for label in CLASS_NAMES:
            if per_class.get(label, False):
                rates[label] = rates[label] + 1
            if label in per_class_probs:
                avg_confidence[label].append(per_class_probs[label])

    return {
        "window_size": total,
        "per_class_prediction_rate": {
            label: round(count / total, 4) for label, count in rates.items()
        },
        "per_class_avg_confidence": {
            label: round(float(sum(vals) / len(vals)), 4) if vals else None
            for label, vals in avg_confidence.items()
        },
        "drift_rate": round(
            sum(1 for e in history_list if e.get("drift_alert") == "DRIFT_DETECTED")
            / total,
            4,
        ),
    }


@app.get("/metrics/calibration")
def calibration_summary():
    """Return per-class calibration (Brier scores) from training metrics."""
    tm = load_json(RUNTIME_PATHS["training_metrics"])
    return {
        "test_macro_brier": tm.get("test_macro_brier"),
        "test_macro_auprc": tm.get("test_macro_auprc"),
        "test_macro_roc_auc": tm.get("test_macro_roc_auc"),
        "per_class_brier": tm.get("per_class_brier", {}),
        "per_class_auprc": tm.get("per_class_auprc", {}),
        "per_class_roc_auc": tm.get("per_class_roc_auc", {}),
        "per_class_recall": tm.get("per_class_recall", {}),
        "threshold_details": tm.get("threshold_details", {}),
        "class_names": tm.get("class_names", CLASS_NAMES),
    }


@app.post("/predict")
async def predict(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    image = load_image_from_upload(file)
    input_summary = summarize_image(image)
    validate_image(image, input_summary)

    start = time.perf_counter()
    dual_results = await benchmark_both_models(image)
    selected_key = random.choices(
        ["pytorch", "onnx"],
        weights=[TRAFFIC_WEIGHTS["pytorch"], TRAFFIC_WEIGHTS["onnx"]],
        k=1,
    )[0]
    selected_result = dual_results[selected_key]
    warning_flags = []

    if "error" in selected_result:
        fallback_result = dual_results["pytorch"]
        if "error" in fallback_result:
            REQUEST_COUNTER.labels(model=selected_key, status="failure").inc()
            raise HTTPException(
                status_code=503,
                detail=f"Both inference backends failed: {dual_results}",
            )
        selected_result = fallback_result
        warning_flags.append(f"{selected_key.upper()} failed, fallback to PyTorch.")

    # Increment per-class disease prediction counters
    for label in selected_result["predicted_labels"]:
        DISEASE_PREDICTION_COUNTER.labels(
            disease=label, model=selected_result["model_key"]
        ).inc()

    drift_result = calculate_drift(image)
    DRIFT_COUNTER.labels(status=drift_result["drift_alert"]).inc()
    REQUEST_COUNTER.labels(model=selected_result["model_key"], status="success").inc()

    pytorch_latency = (
        dual_results["pytorch"].get("latency_ms")
        if "error" not in dual_results["pytorch"]
        else None
    )
    onnx_latency = (
        dual_results["onnx"].get("latency_ms")
        if "error" not in dual_results["onnx"]
        else None
    )
    latency_delta = None
    if pytorch_latency is not None and onnx_latency is not None:
        latency_delta = round(float(onnx_latency) - float(pytorch_latency), 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    response_payload = {
        "timestamp": timestamp,
        "selected_model": selected_result["model_used"],
        "selected_arm": "A" if selected_result["model_key"] == "pytorch" else "B",
        "weighted_traffic_split": TRAFFIC_WEIGHTS,
        "predicted_labels": selected_result["predicted_labels"],
        "all_predictions": selected_result["all_predictions"],
        "top_predictions": selected_result["top_predictions"],
        "confidence": selected_result["max_confidence"],
        "low_confidence": selected_result["low_confidence"],
        "inconclusive_scan": selected_result["inconclusive_scan"],
        "confidence_review_threshold": LOW_CONFIDENCE_THRESHOLD * 100.0,
        "pytorch_latency_ms": pytorch_latency,
        "onnx_latency_ms": onnx_latency,
        "latency_delta_ms": latency_delta,
        "latency_ms": selected_result["latency_ms"],
        "drift": drift_result,
        "input_summary": input_summary,
        "warning_flags": warning_flags,
        "recommendation": build_recommendation(
            selected_result, drift_result, dual_results
        ),
        "active_onnx_model": ACTIVE_ONNX_MODEL_NAME,
        "recent_history": list(REQUEST_LOG_HISTORY),
        "cam_b64": dual_results["pytorch"].get("cam_b64"),
    }

    history_entry = {
        "timestamp": timestamp,
        "model": selected_result["model_used"],
        "latency_ms": selected_result["latency_ms"],
        "drift_alert": drift_result["drift_alert"],
        "drift_score": drift_result["drift_score"],
        "confidence": selected_result["max_confidence"],
        "labels": selected_result["predicted_labels"],
        "per_class_probabilities": {
            CLASS_NAMES[i]: round(float(selected_result["probabilities"][i]), 4)
            for i in range(len(CLASS_NAMES))
            if i < len(selected_result.get("probabilities", []))
        },
        "per_class_predictions": {
            label: (label in selected_result["predicted_labels"])
            for label in CLASS_NAMES
        },
    }
    append_history(history_entry)
    response_payload["recent_history"] = list(REQUEST_LOG_HISTORY)

    emit_structured_log(
        {
            "event": "predict",
            "timestamp": timestamp,
            "model_used": selected_result["model_used"],
            "latency_ms": selected_result["latency_ms"],
            "confidence": selected_result["max_confidence"],
            "drift_status": drift_result["drift_alert"],
            "pathologies": selected_result["predicted_labels"],
            "client": request.client.host if request.client else None,
        }
    )

    response_payload["request_latency_ms"] = round(
        (time.perf_counter() - start) * 1000, 2
    )

    # Trigger background save for fine-tuning
    background_tasks.add_task(save_prediction_data, image, response_payload)

    return response_payload


@app.post("/infer")
async def infer_raw(file: UploadFile = File(...)):
    """Raw dual-model inference for remote backend nodes.
    Returns probabilities from both PyTorch and ONNX arms with no side-effects
    (no metrics, no drift, no history). Only available on the HF Spaces inference node
    (i.e. when HF_SPACES_URL is not set)."""
    if HF_SPACES_URL:
        raise HTTPException(
            status_code=501,
            detail="This node is a proxy backend; /infer is only served by the inference node.",
        )
    image = load_image_from_upload(file)
    dual = await _run_local_inference(image)
    out: dict[str, Any] = {}
    for key in ("pytorch", "onnx"):
        r = dual.get(key, {})
        out[key] = (
            {
                "probabilities": r.get("probabilities", []),
                "latency_ms": r.get("latency_ms"),
            }
            if "error" not in r
            else {"error": r["error"]}
        )
    return {
        **out,
        "class_names": CLASS_NAMES,
        "thresholds": THRESHOLDS.tolist(),
        "multi_label": MULTI_LABEL,
    }


# ─── Mount Gradio UI into FastAPI (single-port for HF Spaces) ───────────────
# This allows the entire app (API + UI) to run on one port (7860).
# - FastAPI REST endpoints remain at /predict, /health, /metrics, etc.
# - Gradio UI is served at / (root)

try:
    # Dynamically add frontend dir to path so app.py can be imported
    _root_dir = str(BASE_DIR)
    if _root_dir not in sys.path:
        sys.path.insert(0, _root_dir)

    import gradio as gr
    from frontend.app import demo as gradio_demo  # the gr.Blocks() object

    from fastapi.responses import RedirectResponse

    @app.get("/")
    def redirect_to_ui():
        return RedirectResponse(url="/ui")

    # Mount Gradio at /ui; FastAPI routes take priority because they're
    # registered first via @app.get / @app.post decorators.
    app = gr.mount_gradio_app(app, gradio_demo, path="/ui")
    logger.info("Gradio UI mounted at /ui — full app on single port.")

except Exception as _e:
    logger.warning(f"Gradio mount skipped ({_e}). API-only mode active.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


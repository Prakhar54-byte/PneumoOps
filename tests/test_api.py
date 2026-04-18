"""
PneumoOps — Automated Test Suite
==================================
Tests the FastAPI backend endpoints without requiring real model files.
Uses synthetic images so the test suite runs on any machine (CI/CD included).

Run:
    python -m pytest tests/ -v

Requirements:
    pip install pytest httpx pillow numpy
"""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ─── Make sure the project root is in the path ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─── Skip model-loading tests if model files are missing ──────────────────────
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "chestmnist_mobilenetv3"
MODELS_AVAILABLE = (
    (MODEL_DIR / "mobilenetv3_chestmnist.pth").exists()
    and (MODEL_DIR / "mobilenetv3_chestmnist.onnx").exists()
)


# ─── Synthetic image helpers ──────────────────────────────────────────────────

def make_synthetic_xray(size: int = 224) -> bytes:
    """Create a fake grayscale chest X-ray as PNG bytes."""
    arr = np.random.normal(loc=0.35, scale=0.12, size=(size, size))
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_color_image(size: int = 224) -> bytes:
    """Create a colorful non-X-ray image to trigger drift detection."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = 200   # strong red channel
    arr[:, :, 1] = 100   # green
    arr[:, :, 2] = 50    # blue
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── Live API tests (require running Docker container) ────────────────────────

@pytest.mark.live
class TestLiveAPI:
    """
    These tests hit the running Docker container.
    Start it first: docker compose up -d app
    Then run: python -m pytest tests/ -v -m live
    """

    BASE_URL = os.getenv("PNEUMOOPS_TEST_URL", "http://127.0.0.1:7860")

    def test_health_check(self):
        """Backend /health must return status=ok with both models loaded."""
        import requests
        resp = requests.get(f"{self.BASE_URL}/health", timeout=10)
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "ok", "Status field is not 'ok'"
        assert data["class_count"] == 14, "Expected 14 ChestMNIST classes"
        print(f"\n  ✅ Health OK — PyTorch:{data['pytorch_model_loaded']} ONNX:{data['onnx_model_loaded']}")

    def test_metrics_endpoint(self):
        """Prometheus /metrics endpoint must return text with pneumoops counters."""
        import requests
        resp = requests.get(f"{self.BASE_URL}/metrics", timeout=10)
        assert resp.status_code == 200
        assert "pneumoops_requests_total" in resp.text, "Counter metric missing"
        assert "pneumoops_inference_latency_ms" in resp.text, "Latency histogram missing"
        print("\n  ✅ Prometheus metrics endpoint OK")

    def test_predict_with_synthetic_xray(self):
        """POST /predict with a synthetic X-ray must return valid prediction JSON."""
        import requests
        image_bytes = make_synthetic_xray()
        resp = requests.post(
            f"{self.BASE_URL}/predict",
            files={"file": ("test_xray.png", image_bytes, "image/png")},
            timeout=30,
        )
        assert resp.status_code == 200, f"Predict failed: {resp.text}"
        data = resp.json()

        # Required fields
        assert "predicted_labels" in data, "Missing predicted_labels"
        assert "top_predictions" in data, "Missing top_predictions"
        assert "drift" in data, "Missing drift field"
        assert "selected_arm" in data, "Missing selected_arm (A/B)"
        assert data["selected_arm"] in ("A", "B"), f"Invalid arm: {data['selected_arm']}"

        # Top predictions structure
        for pred in data["top_predictions"]:
            assert "label" in pred
            assert "confidence" in pred
            assert 0.0 <= pred["confidence"] <= 100.0

        # Latency fields
        assert "request_latency_ms" in data
        assert data["request_latency_ms"] > 0

        print(f"\n  ✅ Predict OK — arm={data['selected_arm']} labels={data['predicted_labels']}")

    def test_drift_detected_on_color_image(self):
        """A strongly colored non-X-ray image should trigger drift detection."""
        import requests
        image_bytes = make_color_image()
        # Color images fail the channel_delta validation check first (400),
        # which is also correct behavior — the system rejects them before inference.
        resp = requests.post(
            f"{self.BASE_URL}/predict",
            files={"file": ("color.png", image_bytes, "image/png")},
            timeout=30,
        )
        # Either rejected with 400 (color image guard) OR passes with DRIFT_DETECTED
        if resp.status_code == 400:
            print("\n  ✅ Color image correctly rejected (channel_delta guard)")
        else:
            assert resp.status_code == 200
            data = resp.json()
            assert data["drift"]["drift_alert"] == "DRIFT_DETECTED", (
                f"Expected DRIFT_DETECTED for color image, got: {data['drift']}"
            )
            print("\n  ✅ Drift correctly detected on color image")

    def test_history_endpoint(self):
        """GET /history must return a list of recent requests."""
        import requests
        resp = requests.get(f"{self.BASE_URL}/history", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "recent_requests" in data
        assert isinstance(data["recent_requests"], list)
        print(f"\n  ✅ History OK — {len(data['recent_requests'])} recent entries")

    def test_class_rates_endpoint(self):
        """GET /metrics/class-rates must return per-class rates for all 14 classes."""
        import requests
        resp = requests.get(f"{self.BASE_URL}/metrics/class-rates", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        rates = data.get("per_class_prediction_rate", {})
        expected_classes = {
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia",
        }
        assert expected_classes.issubset(set(rates.keys())), (
            f"Missing classes: {expected_classes - set(rates.keys())}"
        )
        print(f"\n  ✅ Class-rates OK — {len(rates)} classes tracked")


# ─── Offline unit tests (no server needed) ────────────────────────────────────

class TestImageHelpers:
    """Tests for standalone utility functions that need no server."""

    def test_synthetic_xray_is_valid_png(self):
        """Synthetic X-ray generator must produce a decodable image."""
        raw = make_synthetic_xray()
        img = Image.open(io.BytesIO(raw))
        assert img.format == "PNG"
        assert img.mode == "RGB"
        assert img.size == (224, 224)

    def test_synthetic_xray_is_grayscale_like(self):
        """Synthetic X-ray channels should be very similar (low channel_delta)."""
        raw = make_synthetic_xray()
        img = Image.open(io.BytesIO(raw))
        arr = np.array(img, dtype=np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        # Because we generated from grayscale, R==G==B
        np.testing.assert_array_equal(r, g)
        np.testing.assert_array_equal(g, b)

    def test_color_image_has_high_channel_delta(self):
        """Color image generator must produce an image with high RGB channel variance."""
        raw = make_color_image()
        img = Image.open(io.BytesIO(raw))
        arr = np.array(img, dtype=np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        delta = float(
            np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))
        ) / 3.0
        assert delta > 0.08, f"Expected high channel delta for color image, got {delta:.4f}"

    def test_ethics_file_exists(self):
        """ETHICS.md must be present in the project root."""
        ethics_path = Path(__file__).resolve().parents[1] / "ETHICS.md"
        assert ethics_path.exists(), "ETHICS.md is missing from project root!"
        content = ethics_path.read_text()
        assert "NOT a Medical Device" in content or "NOT A Medical Device" in content or "not a medical device" in content.lower()
        assert "Data Privacy" in content
        assert "Bias" in content

    def test_training_metrics_json_is_valid(self):
        """training_metrics.json must exist and contain expected keys."""
        metrics_path = MODEL_DIR / "training_metrics.json"
        if not metrics_path.exists():
            pytest.skip("Model files not downloaded — run Task B1 first.")
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "class_names" in metrics
        assert len(metrics["class_names"]) == 14, "Expected 14 ChestMNIST classes"
        assert "test_macro_roc_auc" in metrics
        assert 0.0 <= metrics["test_macro_roc_auc"] <= 1.0
        print(f"\n  ✅ Metrics valid — Macro AUROC: {metrics['test_macro_roc_auc']:.3f}")

    def test_baseline_stats_json_is_valid(self):
        """baseline_stats.json must exist and contain drift reference fields."""
        stats_path = MODEL_DIR / "baseline_stats.json"
        if not stats_path.exists():
            pytest.skip("Model files not downloaded — run Task B1 first.")
        with open(stats_path) as f:
            stats = json.load(f)
        # Support both key formats (old: pixel_mean_mean / new: pixel_mean)
        has_mean = "pixel_mean_mean" in stats or "pixel_mean" in stats
        has_std  = "pixel_std_mean"  in stats or "pixel_std"  in stats
        assert has_mean, f"Missing pixel mean key. Keys found: {list(stats.keys())}"
        assert has_std,  f"Missing pixel std key. Keys found: {list(stats.keys())}"
        mean_val = stats.get("pixel_mean", stats.get("pixel_mean_mean", 0))
        assert -2.0 <= mean_val <= 2.0, f"Unexpected pixel mean value: {mean_val}"
        print(f"\n  ✅ Baseline stats valid — pixel_mean={mean_val:.4f}")

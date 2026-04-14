import asyncio
import json
from pathlib import Path
import sys

from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.main import (
    ACTIVE_ONNX_MODEL_NAME,
    BASELINE_STATS,
    MODEL_DIR,
    PYTORCH_MODEL,
    benchmark_both_models,
    calculate_drift,
    health,
    summarize_image,
)


def create_test_image():
    return Image.new("RGB", (512, 512), color=(128, 128, 128))


async def main():
    image = create_test_image()
    health_payload = health()
    summary = summarize_image(image)
    drift_payload = calculate_drift(image)
    benchmark_payload = await benchmark_both_models(image)

    report = {
        "health": health_payload,
        "input_summary": summary,
        "drift": drift_payload,
        "benchmark": benchmark_payload,
        "artifacts_present": {
            "model_dir": str(MODEL_DIR),
            "pytorch_loaded": PYTORCH_MODEL is not None,
            "onnx_active_model": ACTIVE_ONNX_MODEL_NAME,
            "baseline_stats_loaded": bool(BASELINE_STATS),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

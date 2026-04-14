import csv
import json
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from medmnist import PneumoniaMNIST
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = MODEL_DIR / "plots"

PYTORCH_MODEL_PATH = MODEL_DIR / "pneumo_model.pth"
QUANTIZED_ONNX_PATH = MODEL_DIR / "pneumo_model_quantized.onnx"
OPTIMIZED_ONNX_PATH = MODEL_DIR / "pneumo_model_optimized.onnx"
BENCHMARK_JSON_PATH = MODEL_DIR / "benchmark_report.json"
BENCHMARK_CSV_PATH = MODEL_DIR / "benchmark_samples.csv"
BENCHMARK_PLOT_PATH = PLOTS_DIR / "latency_comparison.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
SAMPLE_COUNT = 100

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=WEIGHTS.transforms().mean, std=WEIGHTS.transforms().std),
    ]
)


def load_test_images():
    dataset = PneumoniaMNIST(split="test", download=True, root=str(BASE_DIR / "data"))
    count = min(SAMPLE_COUNT, len(dataset.imgs))
    return [Image.fromarray(dataset.imgs[idx].squeeze()) for idx in range(count)]


def load_pytorch_model():
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_onnx_session():
    model_path = QUANTIZED_ONNX_PATH if QUANTIZED_ONNX_PATH.exists() else OPTIMIZED_ONNX_PATH
    if not model_path.exists():
        raise FileNotFoundError("No ONNX serving artifact found for benchmarking.")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session, model_path.name


def time_pytorch(model, images):
    latencies = []
    with torch.no_grad():
        for image in images:
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            start = time.perf_counter()
            logits = model(tensor)
            torch.softmax(logits, dim=1)
            latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def time_onnx(session, images):
    latencies = []
    input_name = session.get_inputs()[0].name
    for image in images:
        tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
        start = time.perf_counter()
        session.run(None, {input_name: tensor})
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def summarize(latencies):
    latencies_np = np.asarray(latencies, dtype=np.float32)
    return {
        "mean_ms": round(float(np.mean(latencies_np)), 3),
        "median_ms": round(float(np.median(latencies_np)), 3),
        "p95_ms": round(float(np.percentile(latencies_np, 95)), 3),
        "min_ms": round(float(np.min(latencies_np)), 3),
        "max_ms": round(float(np.max(latencies_np)), 3),
    }


def save_plot(pytorch_latencies, onnx_latencies):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([pytorch_latencies, onnx_latencies], tick_labels=["PyTorch", "ONNX"])
    ax.set_title("Inference Latency Comparison")
    ax.set_ylabel("Latency (ms)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(BENCHMARK_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_csv(pytorch_latencies, onnx_latencies):
    with open(BENCHMARK_CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "pytorch_latency_ms", "onnx_latency_ms"])
        for idx, (p_latency, o_latency) in enumerate(zip(pytorch_latencies, onnx_latencies)):
            writer.writerow([idx, round(p_latency, 4), round(o_latency, 4)])


def main():
    if not PYTORCH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing PyTorch model at {PYTORCH_MODEL_PATH}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    images = load_test_images()
    pytorch_model = load_pytorch_model()
    onnx_session, onnx_model_name = load_onnx_session()

    pytorch_latencies = time_pytorch(pytorch_model, images)
    onnx_latencies = time_onnx(onnx_session, images)

    report = {
        "sample_count": len(images),
        "device": str(DEVICE),
        "onnx_model_name": onnx_model_name,
        "pytorch": summarize(pytorch_latencies),
        "onnx": summarize(onnx_latencies),
        "speedup_ratio_mean": round(
            summarize(pytorch_latencies)["mean_ms"] / max(summarize(onnx_latencies)["mean_ms"], 1e-6),
            3,
        ),
    }

    BENCHMARK_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_csv(pytorch_latencies, onnx_latencies)
    save_plot(pytorch_latencies, onnx_latencies)

    print(json.dumps(report, indent=2))
    print(f"Saved benchmark report to {BENCHMARK_JSON_PATH}")
    print(f"Saved per-sample latencies to {BENCHMARK_CSV_PATH}")
    print(f"Saved latency plot to {BENCHMARK_PLOT_PATH}")


if __name__ == "__main__":
    main()

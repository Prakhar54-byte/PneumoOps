import argparse
import json
import shutil
from pathlib import Path

import onnx
import torch
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
from torchvision.models import efficientnet_b0, mobilenet_v3_small


BASE_DIR = Path(__file__).resolve().parents[1]
LEGACY_MODEL_DIR = BASE_DIR / "models"
REALWORLD_MODEL_DIR = BASE_DIR / "models" / "realworld_efficientnet_b0"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a PneumoOps PyTorch checkpoint to ONNX, generate an optimized ORT graph, "
            "and attempt dynamic INT8 quantization with graceful fallback."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the PyTorch checkpoint. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where ONNX artifacts will be written. Defaults to the checkpoint directory.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default=None,
        help="Base name for exported files. Defaults to the checkpoint stem.",
    )
    return parser.parse_args()


def auto_detect_checkpoint() -> Path:
    candidates = [
        REALWORLD_MODEL_DIR / "realworld_efficientnet_b0.pth",
        LEGACY_MODEL_DIR / "pneumo_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not auto-detect a checkpoint. Pass --checkpoint explicitly.")


def load_checkpoint(checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata = payload
        state_dict = payload["model_state_dict"]
    else:
        metadata = {
            "architecture": "mobilenet_v3_small",
            "class_names": ["Normal", "Pneumonia"],
            "image_size": 224,
            "multi_label": False,
        }
        state_dict = payload
    return state_dict, metadata


def build_model(metadata: dict) -> torch.nn.Module:
    architecture = metadata.get("architecture", "mobilenet_v3_small")
    class_names = metadata.get("class_names", ["Normal", "Pneumonia"])
    num_outputs = len(class_names)

    if architecture == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_outputs)
        return model

    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_outputs)
    return model


def validate_onnx_model(model_path: Path, dummy_input: torch.Tensor) -> None:
    session = InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    session.run(None, {session.get_inputs()[0].name: dummy_input.numpy()})


def export_base_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: Path, multi_label: bool) -> None:
    output_name = "logits"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=[output_name],
        dynamic_axes={"input": {0: "batch_size"}, output_name: {0: "batch_size"}},
        opset_version=13,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    validate_onnx_model(output_path, dummy_input)


def export_graph_optimized_onnx(base_path: Path, optimized_path: Path, dummy_input: torch.Tensor) -> None:
    session_options = SessionOptions()
    session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = str(optimized_path)
    session = InferenceSession(str(base_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    session.run(None, {session.get_inputs()[0].name: dummy_input.numpy()})
    validate_onnx_model(optimized_path, dummy_input)


def try_dynamic_quantization(base_path: Path, optimized_path: Path, quantized_path: Path, dummy_input: torch.Tensor):
    try:
        quantize_dynamic(
            model_input=str(base_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        validate_onnx_model(quantized_path, dummy_input)
        return True, "Dynamic INT8 quantization validated successfully."
    except Exception as exc:
        if quantized_path.exists():
            quantized_path.unlink()
        shutil.copy2(optimized_path, quantized_path)
        validate_onnx_model(quantized_path, dummy_input)
        return False, f"Dynamic quantization failed on this runtime, so the optimized FP32 ONNX model was used instead: {exc}"


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or auto_detect_checkpoint()
    output_dir = args.output_dir or checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict, metadata = load_checkpoint(checkpoint_path)
    model = build_model(metadata)
    model.load_state_dict(state_dict)
    model.eval()

    artifact_prefix = args.artifact_prefix or checkpoint_path.stem
    base_path = output_dir / f"{artifact_prefix}.onnx"
    optimized_path = output_dir / f"{artifact_prefix}_optimized.onnx"
    quantized_path = output_dir / f"{artifact_prefix}_quantized.onnx"
    report_path = output_dir / "onnx_export_report.json"

    image_size = int(metadata.get("image_size", 224))
    dummy_input = torch.randn(1, 3, image_size, image_size)

    export_base_onnx(model, dummy_input, base_path, bool(metadata.get("multi_label", False)))
    export_graph_optimized_onnx(base_path, optimized_path, dummy_input)
    quantized_ok, quantized_message = try_dynamic_quantization(base_path, optimized_path, quantized_path, dummy_input)

    report = {
        "checkpoint": str(checkpoint_path),
        "architecture": metadata.get("architecture", "mobilenet_v3_small"),
        "class_names": metadata.get("class_names", ["Normal", "Pneumonia"]),
        "multi_label": bool(metadata.get("multi_label", False)),
        "image_size": image_size,
        "base_onnx": base_path.name,
        "optimized_onnx": optimized_path.name,
        "serving_onnx": quantized_path.name,
        "dynamic_quantization_worked": quantized_ok,
        "notes": quantized_message,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved ONNX model to {base_path}")
    print(f"Saved graph-optimized ONNX model to {optimized_path}")
    print(f"Saved serving ONNX model to {quantized_path}")
    print(quantized_message)
    print(f"Saved ONNX export report to {report_path}")


if __name__ == "__main__":
    main()

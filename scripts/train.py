import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model_utils import apply_temperature_to_logits

DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "realworld_efficientnet_b0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_HF_CONFIGS = {
    "alkzar90/NIH-Chest-X-ray-dataset": "image-classification",
}
DEFAULT_TRUST_REMOTE_CODE = {
    "alkzar90/NIH-Chest-X-ray-dataset",
}

NIH14_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

CRITICAL_RECALL_CLASSES = {"pneumonia", "pneumothorax", "mass", "nodule"}
SIGNIFICANT_RECALL_CLASSES = {"effusion", "cardiomegaly", "consolidation"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the real-world PneumoOps baseline on a Hugging Face or ImageFolder chest X-ray dataset. "
            "This path is designed for multi-label datasets such as ChestX-ray14 or CheXpert."
        )
    )
    parser.add_argument(
        "--dataset-source",
        choices=["hf", "imagefolder"],
        default="hf",
        help="Use a Hugging Face dataset repo or a local imagefolder dataset.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="alkzar90/NIH-Chest-X-ray-dataset",
        help="Hugging Face dataset repo id to train from.",
    )
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset configuration.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow datasets with custom loading code to run without an interactive prompt.",
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=BASE_DIR / "data" / "hf_cache", help="HF cache directory.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "realworld_cxr", help="Local imagefolder root.")
    parser.add_argument("--train-split", default="train", help="Training split name.")
    parser.add_argument("--val-split", default="validation", help="Validation split name.")
    parser.add_argument("--test-split", default="test", help="Test split name.")
    parser.add_argument("--image-column", default=None, help="Image column name. Auto-detected when omitted.")
    parser.add_argument(
        "--label-columns",
        default=None,
        help="Comma-separated label columns for multi-label datasets. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--finding-column",
        default=None,
        help="Single string column containing pipe-separated pathology names, e.g. 'Finding Labels'.",
    )
    parser.add_argument(
        "--uncertain-policy",
        choices=["zero", "one"],
        default="zero",
        help="How to map CheXpert-style uncertain labels (-1).",
    )
    parser.add_argument("--image-size", type=int, default=320, help="Square resize target.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--disable-temperature-scaling",
        action="store_true",
        help="Skip post-training temperature scaling on the validation split.",
    )
    parser.add_argument(
        "--critical-recall-target",
        type=float,
        default=0.72,
        help="Minimum recall target for critical findings during threshold tuning.",
    )
    parser.add_argument(
        "--significant-recall-target",
        type=float,
        default=0.65,
        help="Minimum recall target for significant findings during threshold tuning.",
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="How many batches each DataLoader worker preloads ahead of time.",
    )
    parser.add_argument(
        "--disable-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers between epochs.",
    )
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1, help="Warmup epochs with frozen backbone.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print batch progress every N training/eval steps.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from a saved training-state checkpoint.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for faster experiments.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap for faster experiments.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for faster experiments.")
    parser.add_argument("--baseline-images", type=int, default=256, help="Max images used to estimate drift baseline.")
    parser.add_argument(
        "--baseline-pixels-per-image",
        type=int,
        default=256,
        help="How many pixels to sample per image for the KS-test reference baseline.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def build_transforms(image_size: int):
    weights = EfficientNet_B0_Weights.DEFAULT
    normalize = transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    train_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform, weights.transforms().mean, weights.transforms().std


def resolve_split_name(dataset_dict, requested: str, fallbacks: list[str]) -> str:
    candidates = [requested, *fallbacks]
    for name in candidates:
        if name in dataset_dict:
            return name
    available = ", ".join(dataset_dict.keys())
    raise KeyError(f"Could not find split '{requested}'. Available splits: {available}")


def derive_validation_split(train_split, seed: int, val_fraction: float = 0.1):
    if len(train_split) < 2:
        raise ValueError("Need at least 2 training examples to derive a validation split.")

    val_size = max(1, int(round(len(train_split) * val_fraction)))
    val_size = min(val_size, len(train_split) - 1)
    split = train_split.train_test_split(test_size=val_size, seed=seed)
    return split["train"], split["test"]


def load_raw_splits(args):
    if args.dataset_source == "hf":
        hf_config = args.hf_config or DEFAULT_HF_CONFIGS.get(args.hf_dataset)
        trust_remote_code = args.trust_remote_code or args.hf_dataset in DEFAULT_TRUST_REMOTE_CODE
        dataset_dict = load_dataset(
            args.hf_dataset,
            hf_config,
            cache_dir=str(args.hf_cache_dir),
            trust_remote_code=trust_remote_code,
        )
    else:
        dataset_dict = load_dataset("imagefolder", data_dir=str(args.data_dir))

    train_name = resolve_split_name(dataset_dict, args.train_split, [])
    test_name = resolve_split_name(dataset_dict, args.test_split, ["test"])

    train_split = dataset_dict[train_name]
    test_split = dataset_dict[test_name]
    try:
        val_name = resolve_split_name(dataset_dict, args.val_split, ["valid", "val", "validation"])
        val_split = dataset_dict[val_name]
    except KeyError:
        if args.val_split not in {"validation", "valid", "val"}:
            raise
        train_split, val_split = derive_validation_split(train_split, args.seed)

    if args.max_train_samples:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))
    if args.max_val_samples:
        val_split = val_split.select(range(min(args.max_val_samples, len(val_split))))
    if args.max_test_samples:
        test_split = test_split.select(range(min(args.max_test_samples, len(test_split))))

    return train_split, val_split, test_split


def detect_image_column(split, explicit_column: str | None) -> str:
    if explicit_column:
        return explicit_column

    for column_name, feature in split.features.items():
        if isinstance(feature, HFImage):
            return column_name

    for candidate in ["image", "img", "xray", "pixel_values"]:
        if candidate in split.column_names:
            return candidate

    raise ValueError("Could not auto-detect the image column. Pass --image-column explicitly.")


def normalize_label_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def determine_threshold_policy(class_name: str, critical_recall_target: float, significant_recall_target: float) -> dict:
    normalized_name = normalize_label_name(class_name)
    if normalized_name in CRITICAL_RECALL_CLASSES:
        return {
            "tier": "critical",
            "strategy": "recall_floor",
            "target_recall": critical_recall_target,
        }
    if normalized_name in SIGNIFICANT_RECALL_CLASSES:
        return {
            "tier": "significant",
            "strategy": "recall_floor",
            "target_recall": significant_recall_target,
        }
    return {
        "tier": "standard",
        "strategy": "best_f1",
        "target_recall": None,
    }


def parse_label_columns(split, explicit_columns: str | None, finding_column: str | None):
    if finding_column:
        return {
            "mode": "finding_column",
            "finding_column": finding_column,
            "class_names": NIH14_LABELS,
        }

    if explicit_columns:
        class_names = [item.strip() for item in explicit_columns.split(",") if item.strip()]
        return {
            "mode": "multicolumn",
            "label_columns": class_names,
            "class_names": class_names,
        }

    column_names = list(split.column_names)
    features = split.features

    finding_candidates = [
        "Finding Labels",
        "finding_labels",
        "finding_labels_text",
        "labels_text",
    ]
    for candidate in finding_candidates:
        if candidate in column_names:
            return {
                "mode": "finding_column",
                "finding_column": candidate,
                "class_names": NIH14_LABELS,
            }

    normalized_to_original = {normalize_label_name(name): name for name in column_names}
    for known_set in [CHEXPERT_LABELS, NIH14_LABELS]:
        matched = [normalized_to_original.get(normalize_label_name(label)) for label in known_set]
        matched = [item for item in matched if item is not None]
        if len(matched) >= max(4, len(known_set) // 2):
            return {
                "mode": "multicolumn",
                "label_columns": matched,
                "class_names": matched,
            }

    for column_name, feature in features.items():
        class_feature = getattr(feature, "feature", None)
        class_names = getattr(class_feature, "names", None)
        if not class_names:
            continue

        no_finding_index = None
        if normalize_label_name(class_names[0]) == "nofinding":
            no_finding_index = 0
            class_names = class_names[1:]

        if class_names:
            return {
                "mode": "sequence_classlabel",
                "label_column": column_name,
                "class_names": class_names,
                "no_finding_index": no_finding_index,
            }

    sample = split[0]
    inferred = []
    for column_name in column_names:
        feature = features[column_name]
        if isinstance(feature, HFImage):
            continue
        value = sample[column_name]
        if isinstance(value, (int, float, np.integer, np.floating)) and column_name.lower() not in {
            "age",
            "patientid",
            "study_id",
        }:
            inferred.append(column_name)

    if inferred:
        return {
            "mode": "multicolumn",
            "label_columns": inferred,
            "class_names": inferred,
        }

    raise ValueError("Could not auto-detect label columns. Pass --label-columns or --finding-column.")


def vectorize_labels(row: dict, label_spec: dict, uncertain_policy: str) -> np.ndarray:
    if label_spec["mode"] == "finding_column":
        findings = row.get(label_spec["finding_column"], "") or ""
        if findings == "No Finding":
            active = set()
        else:
            active = {item.strip() for item in findings.split("|") if item.strip()}
        return np.asarray([1.0 if label in active else 0.0 for label in label_spec["class_names"]], dtype=np.float32)

    if label_spec["mode"] == "sequence_classlabel":
        raw_values = row.get(label_spec["label_column"], []) or []
        active_indices = set()
        for value in raw_values:
            if isinstance(value, str):
                try:
                    value = label_spec["class_names"].index(value)
                except ValueError:
                    continue
            value = int(value)
            if value == label_spec.get("no_finding_index"):
                continue
            adjusted = value
            if label_spec.get("no_finding_index") is not None and value > label_spec["no_finding_index"]:
                adjusted -= 1
            if 0 <= adjusted < len(label_spec["class_names"]):
                active_indices.add(adjusted)
        return np.asarray(
            [1.0 if class_index in active_indices else 0.0 for class_index in range(len(label_spec["class_names"]))],
            dtype=np.float32,
        )

    labels = []
    for column_name in label_spec["label_columns"]:
        value = row.get(column_name, 0)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = 0
        if value == -1:
            value = 1 if uncertain_policy == "one" else 0
        labels.append(float(value))
    return np.asarray(labels, dtype=np.float32)


def get_label_columns(label_spec: dict) -> list[str]:
    if label_spec["mode"] == "finding_column":
        return [label_spec["finding_column"]]
    if label_spec["mode"] == "sequence_classlabel":
        return [label_spec["label_column"]]
    return list(label_spec["label_columns"])


class HFDatasetWrapper(Dataset):
    def __init__(self, split, image_column: str, label_spec: dict, transform, uncertain_policy: str):
        self.split = split
        self.image_column = image_column
        self.label_spec = label_spec
        self.transform = transform
        self.uncertain_policy = uncertain_policy
        label_split = split.select_columns(get_label_columns(label_spec))
        self.targets = np.stack([vectorize_labels(row, label_spec, uncertain_policy) for row in label_split], axis=0)
        self.class_names = label_spec["class_names"]

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index: int):
        row = self.split[index]
        image = row[self.image_column]
        if not isinstance(image, Image.Image):
            if isinstance(image, dict) and image.get("path"):
                image = Image.open(image["path"])
            else:
                raise TypeError("Dataset image column did not decode into a PIL image.")
        tensor = self.transform(image)
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return tensor, target


def compute_sample_weights(targets: np.ndarray) -> np.ndarray:
    positive_freq = np.clip(targets.mean(axis=0), 1e-4, 1.0)
    inverse_freq = 1.0 / positive_freq
    positive_weight = (targets * inverse_freq).sum(axis=1)
    negative_weight = np.where(targets.sum(axis=1) == 0, 1.0, 0.0)
    weights = positive_weight + negative_weight
    return weights / np.mean(weights)


def compute_pos_weight(targets: np.ndarray) -> torch.Tensor:
    positives = np.clip(targets.sum(axis=0), 1.0, None)
    negatives = np.clip(len(targets) - positives, 1.0, None)
    return torch.tensor(negatives / positives, dtype=torch.float32, device=DEVICE)


def build_model(num_labels: int) -> torch.nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_labels)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    return model.to(DEVICE)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def fit_temperature(logits_np: np.ndarray, labels_np: np.ndarray, initial_temperature: float = 1.5, max_iter: int = 50) -> float:
    logits = torch.tensor(logits_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.float32)
    log_temperature = nn.Parameter(torch.log(torch.tensor([initial_temperature], dtype=torch.float32)))
    optimizer = optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.binary_cross_entropy_with_logits(apply_temperature_to_logits(logits, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    calibrated_temperature = float(torch.exp(log_temperature).clamp(min=1e-3, max=10.0).item())
    return calibrated_temperature


def compute_eval_loss(logits_np: np.ndarray, labels_np: np.ndarray, pos_weight: torch.Tensor, temperature: float) -> float:
    logits = torch.tensor(logits_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.float32)
    calibrated_logits = apply_temperature_to_logits(logits, temperature)
    loss = F.binary_cross_entropy_with_logits(calibrated_logits, labels, pos_weight=pos_weight.detach().cpu())
    return float(loss.item())


def select_threshold_with_policy(
    class_labels: np.ndarray,
    class_probabilities: np.ndarray,
    class_name: str,
    critical_recall_target: float,
    significant_recall_target: float,
):
    candidate_thresholds = np.arange(0.1, 0.91, 0.05)
    policy = determine_threshold_policy(class_name, critical_recall_target, significant_recall_target)
    candidates = []
    for threshold in candidate_thresholds:
        predictions = (class_probabilities >= threshold).astype(int)
        candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(class_labels, predictions, zero_division=0)),
                "recall": float(recall_score(class_labels, predictions, zero_division=0)),
                "f1": float(f1_score(class_labels, predictions, zero_division=0)),
            }
        )

    if policy["strategy"] == "recall_floor":
        eligible = [candidate for candidate in candidates if candidate["recall"] >= policy["target_recall"]]
        if eligible:
            selected = max(eligible, key=lambda item: (item["threshold"], item["precision"], item["f1"]))
        else:
            selected = max(candidates, key=lambda item: (item["recall"], item["f1"], -item["threshold"]))
    else:
        selected = max(candidates, key=lambda item: (item["f1"], item["precision"], item["recall"], item["threshold"]))

    return {
        "class_name": class_name,
        "tier": policy["tier"],
        "strategy": policy["strategy"],
        "target_recall": policy["target_recall"],
        "selected_threshold": round(selected["threshold"], 6),
        "selected_precision": round(selected["precision"], 6),
        "selected_recall": round(selected["recall"], 6),
        "selected_f1": round(selected["f1"], 6),
    }


def tune_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    critical_recall_target: float,
    significant_recall_target: float,
):
    thresholds = []
    threshold_details = {}
    for class_index, class_name in enumerate(class_names):
        class_labels = labels[:, class_index]
        if class_labels.max() == class_labels.min():
            thresholds.append(0.5)
            threshold_details[class_name] = {
                "class_name": class_name,
                "tier": "constant",
                "strategy": "default",
                "target_recall": None,
                "selected_threshold": 0.5,
                "selected_precision": None,
                "selected_recall": None,
                "selected_f1": None,
            }
            continue

        selected = select_threshold_with_policy(
            class_labels,
            probabilities[:, class_index],
            class_name,
            critical_recall_target=critical_recall_target,
            significant_recall_target=significant_recall_target,
        )
        thresholds.append(selected["selected_threshold"])
        threshold_details[class_name] = selected
    return np.asarray(thresholds, dtype=np.float32), threshold_details


def summarize_multilabel_metrics(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray, class_names: list[str]):
    predictions = (probabilities >= thresholds[None, :]).astype(int)
    micro_precision = precision_score(labels, predictions, average="micro", zero_division=0)
    micro_recall = recall_score(labels, predictions, average="micro", zero_division=0)
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    sample_f1 = f1_score(labels, predictions, average="samples", zero_division=0)

    per_class_auc = {}
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    auc_values = []
    for class_index, class_name in enumerate(class_names):
        class_labels = labels[:, class_index]
        class_predictions = predictions[:, class_index]
        per_class_precision[class_name] = round(precision_score(class_labels, class_predictions, zero_division=0), 6)
        per_class_recall[class_name] = round(recall_score(class_labels, class_predictions, zero_division=0), 6)
        per_class_f1[class_name] = round(f1_score(class_labels, class_predictions, zero_division=0), 6)
        if class_labels.max() == class_labels.min():
            per_class_auc[class_name] = None
            continue
        auc = roc_auc_score(class_labels, probabilities[:, class_index])
        per_class_auc[class_name] = round(float(auc), 6)
        auc_values.append(float(auc))

    return {
        "micro_precision": round(float(micro_precision), 6),
        "micro_recall": round(float(micro_recall), 6),
        "micro_f1": round(float(micro_f1), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "macro_f1": round(float(macro_f1), 6),
        "sample_f1": round(float(sample_f1), 6),
        "macro_roc_auc": round(float(np.mean(auc_values)), 6) if auc_values else None,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_roc_auc": per_class_auc,
        "predictions": predictions,
    }


def build_dataloader(dataset: Dataset, batch_size: int, workers: int, pin_memory: bool, prefetch_factor: int, persistent_workers: bool, shuffle: bool = False, sampler=None) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def prepare_images(images: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        images = images.to(memory_format=torch.channels_last)
    return images


def build_training_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_f1: float,
    best_state: dict | None,
    history: list[dict],
    class_names: list[str],
    label_spec: dict,
    args,
    normalize_mean,
    normalize_std,
    logit_temperature: float,
) -> dict:
    return {
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_state": copy.deepcopy(best_state),
        "history": copy.deepcopy(history),
        "class_names": class_names,
        "label_spec": label_spec,
        "image_size": args.image_size,
        "normalize_mean": normalize_mean,
        "normalize_std": normalize_std,
        "logit_temperature": logit_temperature,
        "args": vars(args),
    }


def collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    log_every: int = 0,
    split_name: str = "eval",
):
    model.eval()
    all_labels = []
    all_logits = []
    all_probabilities = []
    start_time = time.perf_counter()

    with torch.no_grad():
        for step, (images, labels) in enumerate(loader, start=1):
            images = prepare_images(images)
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
                elapsed = time.perf_counter() - start_time
                samples_done = min(step * loader.batch_size, len(loader.dataset))
                rate = samples_done / max(elapsed, 1e-6)
                print(
                    f"[{split_name}] step {step}/{len(loader)} | "
                    f"samples={samples_done}/{len(loader.dataset)} | "
                    f"{rate:.1f} samples/s"
                )

    return {
        "labels_np": np.concatenate(all_labels, axis=0),
        "logits_np": np.concatenate(all_logits, axis=0),
        "probabilities_np": np.concatenate(all_probabilities, axis=0),
    }


def evaluate_predictions(
    labels_np: np.ndarray,
    logits_np: np.ndarray,
    class_names: list[str],
    pos_weight: torch.Tensor,
    critical_recall_target: float,
    significant_recall_target: float,
    thresholds: np.ndarray | None = None,
    logit_temperature: float = 1.0,
):
    calibrated_probabilities = torch.sigmoid(
        apply_temperature_to_logits(torch.tensor(logits_np, dtype=torch.float32), logit_temperature)
    ).numpy()
    if thresholds is None:
        threshold_values, threshold_details = tune_thresholds(
            labels_np,
            calibrated_probabilities,
            class_names=class_names,
            critical_recall_target=critical_recall_target,
            significant_recall_target=significant_recall_target,
        )
    else:
        threshold_values = thresholds
        threshold_details = {}

    metric_bundle = summarize_multilabel_metrics(labels_np, calibrated_probabilities, threshold_values, class_names)
    metric_bundle["loss"] = round(compute_eval_loss(logits_np, labels_np, pos_weight, logit_temperature), 6)
    metric_bundle["thresholds"] = threshold_values.tolist()
    metric_bundle["threshold_details"] = threshold_details
    metric_bundle["labels"] = labels_np.tolist()
    metric_bundle["probabilities"] = calibrated_probabilities.tolist()
    metric_bundle["logit_temperature"] = round(float(logit_temperature), 6)
    return metric_bundle


def save_training_plots(history: list[dict], output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", label="Train Loss")
    axes[0].plot(epochs, [item["val_loss"] for item in history], marker="o", label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [item["train_micro_f1"] for item in history], marker="o", label="Train Micro-F1")
    axes[1].plot(epochs, [item["val_micro_f1"] for item in history], marker="o", label="Val Micro-F1")
    axes[1].set_title("Micro-F1")
    axes[1].legend()

    axes[2].plot(epochs, [item["val_macro_f1"] for item in history], marker="o", label="Val Macro-F1")
    axes[2].plot(epochs, [item["val_macro_roc_auc"] or 0.0 for item in history], marker="o", label="Val Macro ROC-AUC")
    axes[2].set_title("Macro Metrics")
    axes[2].legend()

    axes[3].plot(epochs, [item["val_micro_precision"] for item in history], marker="o", label="Val Precision")
    axes[3].plot(epochs, [item["val_micro_recall"] for item in history], marker="o", label="Val Recall")
    axes[3].set_title("Precision / Recall")
    axes[3].legend()

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "training_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_class_summary_plots(class_names: list[str], targets: np.ndarray, metrics: dict, thresholds: list[float], output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    prevalence = targets.mean(axis=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=class_names, y=prevalence, ax=ax, color="#2563eb")
    ax.set_title("Training Label Prevalence")
    ax.set_ylabel("Positive Rate")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plots_dir / "label_prevalence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    per_class_f1 = [metrics["per_class_f1"].get(label, 0.0) for label in class_names]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=class_names, y=per_class_f1, ax=ax, color="#059669")
    ax.set_title("Per-Class F1")
    ax.set_ylabel("F1")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plots_dir / "per_class_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=class_names, y=thresholds, ax=ax, color="#d97706")
    ax.set_title("Validation-Tuned Thresholds")
    ax.set_ylabel("Threshold")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plots_dir / "thresholds.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_baseline_stats(split, image_column: str, image_size: int, max_images: int, pixels_per_image: int):
    sampled_rows = split.select(range(min(max_images, len(split))))
    means = []
    stds = []
    histograms = []
    pixel_reference_sample = []

    for row in sampled_rows:
        image = row[image_column]
        if not isinstance(image, Image.Image):
            if isinstance(image, dict) and image.get("path"):
                image = Image.open(image["path"])
            else:
                continue

        gray = np.asarray(image.convert("L").resize((image_size, image_size)), dtype=np.float32) / 255.0
        means.append(float(np.mean(gray)))
        stds.append(float(np.std(gray)))
        hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
        histograms.append(hist)

        flattened = gray.reshape(-1)
        sample_size = min(pixels_per_image, flattened.shape[0])
        selected = np.random.choice(flattened, size=sample_size, replace=False)
        pixel_reference_sample.extend([round(float(value), 6) for value in selected])

    return {
        "pixel_mean_mean": round(float(np.mean(means)), 6),
        "pixel_mean_std": round(float(np.std(means) + 1e-8), 6),
        "pixel_std_mean": round(float(np.mean(stds)), 6),
        "pixel_std_std": round(float(np.std(stds) + 1e-8), 6),
        "histogram_bins": 32,
        "histogram_mean": np.mean(np.stack(histograms), axis=0).round(6).tolist(),
        "pixel_reference_sample": pixel_reference_sample,
        "drift_threshold": 1.4,
        "drift_ks_pvalue_threshold": 0.05,
    }


def train():
    args = parse_args()
    set_seed(args.seed)

    train_transform, eval_transform, normalize_mean, normalize_std = build_transforms(args.image_size)
    print("Loading dataset splits...")
    train_split, val_split, test_split = load_raw_splits(args)
    image_column = detect_image_column(train_split, args.image_column)
    label_spec = parse_label_columns(train_split, args.label_columns, args.finding_column)

    print("Preparing label targets without decoding images...")
    train_dataset = HFDatasetWrapper(train_split, image_column, label_spec, train_transform, args.uncertain_policy)
    val_dataset = HFDatasetWrapper(val_split, image_column, label_spec, eval_transform, args.uncertain_policy)
    test_dataset = HFDatasetWrapper(test_split, image_column, label_spec, eval_transform, args.uncertain_policy)

    class_names = train_dataset.class_names
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_weights = compute_sample_weights(train_dataset.targets)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = args.workers > 0 and not args.disable_persistent_workers
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=persistent_workers,
    )

    model = build_model(num_labels=len(class_names))
    pos_weight = compute_pos_weight(train_dataset.targets)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = []
    best_val_f1 = -1.0
    best_state = None
    start_epoch = 0
    last_checkpoint_path = output_dir / "realworld_efficientnet_b0_last.pth"
    current_temperature = 1.0

    if args.resume_from is not None:
        print(f"Resuming training from {args.resume_from}")
        resume_state = torch.load(args.resume_from, map_location=DEVICE)
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scaler_state = resume_state.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_state.get("epoch", 0))
        best_val_f1 = float(resume_state.get("best_val_f1", -1.0))
        best_state = resume_state.get("best_state")
        history = resume_state.get("history", [])
        current_temperature = float(resume_state.get("logit_temperature", 1.0))
        print(f"Resume state loaded at epoch {start_epoch} with best_val_micro_f1={best_val_f1:.4f}")

    torch.save(
        build_training_state(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=start_epoch,
            best_val_f1=best_val_f1,
            best_state=best_state,
            history=history,
            class_names=class_names,
            label_spec=label_spec,
            args=args,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            logit_temperature=current_temperature,
        ),
        last_checkpoint_path,
    )

    print(f"Training real-world PneumoOps path on {DEVICE}")
    print(f"Dataset source: {args.dataset_source}")
    if args.dataset_source == "hf":
        print(f"Hugging Face dataset: {args.hf_dataset}")
    print(f"Class count: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Train/Val/Test sizes: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    print(f"Temperature scaling enabled: {not args.disable_temperature_scaling}")
    print(
        f"DataLoader config: batch_size={args.batch_size}, workers={args.workers}, "
        f"prefetch_factor={args.prefetch_factor if args.workers > 0 else 'n/a'}, "
        f"persistent_workers={persistent_workers}, pin_memory={pin_memory}"
    )
    print(f"Steps per epoch: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            set_backbone_trainable(model, trainable=epoch >= args.freeze_backbone_epochs)

            running_loss = 0.0
            train_labels = []
            train_probabilities = []
            epoch_start = time.perf_counter()

            for step, (images, labels) in enumerate(train_loader, start=1):
                images = prepare_images(images)
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                train_labels.append(labels.detach().cpu().numpy())
                train_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

                if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                    elapsed = time.perf_counter() - epoch_start
                    samples_done = min(step * train_loader.batch_size, len(train_loader.dataset))
                    rate = samples_done / max(elapsed, 1e-6)
                    print(
                        f"[train] epoch {epoch + 1}/{args.epochs} step {step}/{len(train_loader)} | "
                        f"samples={samples_done}/{len(train_loader.dataset)} | "
                        f"loss={loss.item():.4f} | {rate:.1f} samples/s"
                    )

            train_labels_np = np.concatenate(train_labels, axis=0)
            train_probabilities_np = np.concatenate(train_probabilities, axis=0)
            train_thresholds = np.full(train_labels_np.shape[1], 0.5, dtype=np.float32)
            train_metrics = summarize_multilabel_metrics(train_labels_np, train_probabilities_np, train_thresholds, class_names)
            train_loss = running_loss / max(len(train_loader.dataset), 1)

            val_outputs = collect_outputs(
                model,
                val_loader,
                log_every=args.log_every,
                split_name="val",
            )
            val_temperature = (
                fit_temperature(val_outputs["logits_np"], val_outputs["labels_np"])
                if not args.disable_temperature_scaling
                else 1.0
            )
            val_metrics = evaluate_predictions(
                val_outputs["labels_np"],
                val_outputs["logits_np"],
                class_names=class_names,
                pos_weight=pos_weight,
                critical_recall_target=args.critical_recall_target,
                significant_recall_target=args.significant_recall_target,
                thresholds=None,
                logit_temperature=val_temperature,
            )
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(float(train_loss), 6),
                "train_micro_f1": train_metrics["micro_f1"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_micro_precision": val_metrics["micro_precision"],
                "val_micro_recall": val_metrics["micro_recall"],
                "val_micro_f1": val_metrics["micro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_roc_auc": val_metrics["macro_roc_auc"],
                "val_logit_temperature": val_metrics["logit_temperature"],
            }
            history.append(epoch_record)

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | train_micro_f1={train_metrics['micro_f1']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | val_micro_f1={val_metrics['micro_f1']:.4f} | "
                f"val_macro_auc={val_metrics['macro_roc_auc']} | temp={val_metrics['logit_temperature']:.3f}"
            )

            if val_metrics["micro_f1"] > best_val_f1:
                best_val_f1 = float(val_metrics["micro_f1"])
                current_temperature = float(val_metrics["logit_temperature"])
                best_state = {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "architecture": "efficientnet_b0",
                    "class_names": class_names,
                    "thresholds": val_metrics["thresholds"],
                    "threshold_details": val_metrics["threshold_details"],
                    "image_size": args.image_size,
                    "normalize_mean": normalize_mean,
                    "normalize_std": normalize_std,
                    "logit_temperature": current_temperature,
                    "multi_label": True,
                    "label_spec": label_spec,
                }
                torch.save(best_state, output_dir / "realworld_efficientnet_b0_best.pth")

            torch.save(
                build_training_state(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch + 1,
                    best_val_f1=best_val_f1,
                    best_state=best_state,
                    history=history,
                    class_names=class_names,
                    label_spec=label_spec,
                    args=args,
                    normalize_mean=normalize_mean,
                    normalize_std=normalize_std,
                    logit_temperature=current_temperature,
                ),
                last_checkpoint_path,
            )
    except KeyboardInterrupt:
        torch.save(
            build_training_state(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=len(history),
                best_val_f1=best_val_f1,
                best_state=best_state,
                history=history,
                class_names=class_names,
                label_spec=label_spec,
                args=args,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
                logit_temperature=current_temperature,
            ),
            last_checkpoint_path,
        )
        print(f"Training interrupted. Resume from {last_checkpoint_path}")
        raise

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    checkpoint_path = output_dir / "realworld_efficientnet_b0.pth"
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state["model_state_dict"])

    test_outputs = collect_outputs(
        model,
        test_loader,
        log_every=args.log_every,
        split_name="test",
    )
    test_metrics = evaluate_predictions(
        test_outputs["labels_np"],
        test_outputs["logits_np"],
        class_names=class_names,
        pos_weight=pos_weight,
        critical_recall_target=args.critical_recall_target,
        significant_recall_target=args.significant_recall_target,
        thresholds=np.asarray(best_state["thresholds"], dtype=np.float32),
        logit_temperature=float(best_state.get("logit_temperature", 1.0)),
    )

    baseline_stats = compute_baseline_stats(
        train_split,
        image_column=image_column,
        image_size=args.image_size,
        max_images=args.baseline_images,
        pixels_per_image=args.baseline_pixels_per_image,
    )
    baseline_stats["class_names"] = class_names
    baseline_stats["image_size"] = args.image_size
    baseline_stats["normalize_mean"] = normalize_mean
    baseline_stats["normalize_std"] = normalize_std

    metrics_summary = {
        "dataset_source": args.dataset_source,
        "hf_dataset": args.hf_dataset if args.dataset_source == "hf" else None,
        "architecture": "efficientnet_b0",
        "multi_label": True,
        "class_names": class_names,
        "best_val_micro_f1": round(best_val_f1, 6),
        "logit_temperature": round(float(best_state.get("logit_temperature", 1.0)), 6),
        "test_loss": test_metrics["loss"],
        "test_micro_precision": test_metrics["micro_precision"],
        "test_micro_recall": test_metrics["micro_recall"],
        "test_micro_f1": test_metrics["micro_f1"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_sample_f1": test_metrics["sample_f1"],
        "test_macro_roc_auc": test_metrics["macro_roc_auc"],
        "per_class_precision": test_metrics["per_class_precision"],
        "per_class_recall": test_metrics["per_class_recall"],
        "per_class_f1": test_metrics["per_class_f1"],
        "per_class_roc_auc": test_metrics["per_class_roc_auc"],
        "thresholds": best_state["thresholds"],
        "threshold_details": best_state.get("threshold_details", {}),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "critical_recall_target": args.critical_recall_target,
        "significant_recall_target": args.significant_recall_target,
        "image_size": args.image_size,
        "device": str(DEVICE),
    }

    (output_dir / "training_metrics.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "baseline_stats.json").write_text(json.dumps(baseline_stats, indent=2), encoding="utf-8")

    save_training_plots(history, output_dir)
    save_class_summary_plots(class_names, train_dataset.targets, test_metrics, best_state["thresholds"], output_dir)

    print(json.dumps(metrics_summary, indent=2))
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {output_dir / 'training_metrics.json'}")
    print(f"Saved history to {output_dir / 'training_history.json'}")
    print(f"Saved baseline stats to {output_dir / 'baseline_stats.json'}")
    print(f"Saved plots to {output_dir / 'plots'}")


if __name__ == "__main__":
    train()

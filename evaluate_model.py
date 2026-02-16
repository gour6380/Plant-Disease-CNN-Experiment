from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import Config
from dataset import PlantVillageDataset, get_or_compute_normalization_stats
from models import AdvancedCNN, BaselineCNN
from transform import build_transforms

def _list_class_names(train_folder: Path) -> list[str]:
    """
	List non-hidden class directory names under a training folder.

	Args:
		train_folder: Directory whose immediate subdirectories represent classes.

	Returns:
		Sorted list of class folder names.
	"""
    return sorted([p.name for p in train_folder.iterdir() if p.is_dir() and not p.name.startswith(".")])

def _collect_samples(split_folder: Path, class_to_idx: dict[str, int], allowed_exts: set[str]) -> list[tuple[Path, int]]:
    """
	Collect (path, class_index) samples for a split folder.

	Iterates classes in `class_to_idx` order, scans each class subdirectory under
	`split_folder`, filters by `allowed_exts`, and returns labeled file paths.

	Args:
		split_folder: Root directory for a dataset split (e.g., train/val/test).
		class_to_idx: Mapping of class name to integer index.
		allowed_exts: Set of allowed lowercase file extensions (e.g., {".jpg"}).

	Returns:
		List of (image_path, class_index) tuples.
	"""
    samples: list[tuple[Path, int]] = []
    for class_name in sorted(class_to_idx):
        class_idx = class_to_idx[class_name]
        class_dir = split_folder / class_name
        files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts])
        samples.extend((p, class_idx) for p in files)
    return samples

def _loader_kwargs(cfg: Any) -> dict[str, Any]:
    """
	Build DataLoader keyword arguments from config.

	Always sets batch size, worker count, and disables pinning. If workers are
	enabled, adds `persistent_workers` and `prefetch_factor`.

	Args:
		cfg: Config-like object providing batch_size, num_workers, and worker options.

	Returns:
		Dictionary of kwargs suitable for constructing a PyTorch DataLoader.
	"""
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "pin_memory": False,
    }
    if int(cfg.num_workers) > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return kwargs

def build_shared_eval_loaders(cfg: Any) -> dict[str, Any]:
    """
	Build evaluation DataLoaders that share class mapping and normalization stats.

	Validates split folders, derives `class_to_idx` from train subdirectories,
	collects labeled samples for train/val/test (filtered by `cfg.allowed_exts`),
	computes/loads normalization stats from the training samples, builds train/eval
	transforms, and returns non-shuffled loaders plus split metadata.

	Args:
		cfg: Config-like object providing split folder paths, allowed_exts,
			image size, and DataLoader settings.

	Returns:
		Dictionary containing DataLoaders for {"train","val","test"} and metadata:
		class_names, class_to_idx, normalization_mean/std, and split sizes.

	Raises:
		FileNotFoundError: If any split folder does not exist.
		NotADirectoryError: If any split folder is not a directory.
		ValueError: If no classes are found in train, or any split has no samples
			after extension filtering.
	"""
    train_folder = Path(cfg.train_folder)
    val_folder = Path(cfg.val_folder)
    test_folder = Path(cfg.test_folder)

    for split_folder, split_name in ((train_folder, "train"), (val_folder, "val"), (test_folder, "test")):
        if not split_folder.exists():
            raise FileNotFoundError(f"{split_name} folder does not exist: {split_folder}")
        if not split_folder.is_dir():
            raise NotADirectoryError(f"{split_name} folder is not a directory: {split_folder}")

    class_names = _list_class_names(train_folder)
    if not class_names:
        raise ValueError(f"No class folders found in train split: {train_folder}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    allowed_exts = {ext.lower() for ext in tuple(cfg.allowed_exts)}

    train_samples = _collect_samples(train_folder, class_to_idx, allowed_exts)
    val_samples = _collect_samples(val_folder, class_to_idx, allowed_exts)
    test_samples = _collect_samples(test_folder, class_to_idx, allowed_exts)
    if not train_samples or not val_samples or not test_samples:
        raise ValueError("One or more splits have no images after extension filtering.")

    mean, std = get_or_compute_normalization_stats(cfg=cfg, train_samples=train_samples)
    train_transform, eval_transform = build_transforms(image_height=int(cfg.image_height), image_width=int(cfg.image_width), mean=mean, std=std)

    train_ds = PlantVillageDataset(train_samples, transform=train_transform)
    val_ds = PlantVillageDataset(val_samples, transform=eval_transform)
    test_ds = PlantVillageDataset(test_samples, transform=eval_transform)

    kwargs = _loader_kwargs(cfg)
    return {
        "train": DataLoader(train_ds, shuffle=False, drop_last=False, **kwargs),
        "val": DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs),
        "test": DataLoader(test_ds, shuffle=False, drop_last=False, **kwargs),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "normalization_mean": mean,
        "normalization_std": std,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }

def _build_model(model_name: str, cfg: Any, num_classes: int, device: torch.device) -> nn.Module:
    """
	Instantiate a model by name and move it to the target device.

	Supports "baseline" (with fixed 256x256 input constraint and output-dim check)
	and "advanced" (configurable dropout and class count).

	Args:
		model_name: Model selector ("baseline" or "advanced").
		cfg: Config-like object providing image size and optional advanced dropout.
		num_classes: Number of output classes.
		device: Target `torch.device`.

	Returns:
		Initialized `nn.Module` moved to `device`.

	Raises:
		ValueError: If `model_name` is unsupported, image size is incompatible for
			baseline, or output dimension does not match `num_classes`.
	"""
    name = model_name.strip().lower()
    if name == "baseline":
        if int(cfg.image_width) != 256 or int(cfg.image_height) != 256:
            raise ValueError(
                "BaselineCNN expects 256x256 inputs due to fixed classifier dimensions. "
                f"Got {cfg.image_width}x{cfg.image_height}."
            )
        model = BaselineCNN(input_features=3)
        out_dim = int(model.classifier[-1].out_features)
        if out_dim != num_classes:
            raise ValueError(f"Baseline model output dim {out_dim} does not match classes {num_classes}.")
        return model.to(device)

    if name == "advanced":
        return AdvancedCNN(
            num_classes=num_classes,
            in_channels=3,
            dropout=float(getattr(cfg, "advanced_dropout", 0.3)),
        ).to(device)

    raise ValueError(f"Unsupported model_name: {model_name}. Use baseline or advanced.")

def _validate_class_mapping(
    model_name: str,
    ckpt_class_to_idx: Any,
    dataset_class_to_idx: dict[str, int],
) -> dict[str, int]:
    """
	Validate that a checkpoint's class mapping matches the dataset mapping.

	Ensures `ckpt_class_to_idx` is a dict, normalizes keys/values to (str -> int),
	and raises if it differs from `dataset_class_to_idx`.

	Args:
		model_name: Model identifier used for error context.
		ckpt_class_to_idx: Class mapping loaded from a checkpoint.
		dataset_class_to_idx: Expected class mapping from the dataset.

	Returns:
		The normalized checkpoint mapping (str -> int).

	Raises:
		ValueError: If the checkpoint mapping is missing/invalid or does not match
			the dataset mapping.
	"""
    if not isinstance(ckpt_class_to_idx, dict):
        raise ValueError(f"[{model_name}] checkpoint missing valid class_to_idx mapping.")
    normalized = {str(k): int(v) for k, v in ckpt_class_to_idx.items()}
    if normalized != dataset_class_to_idx:
        ckpt_items = sorted(normalized.items(), key=lambda x: x[1])
        ds_items = sorted(dataset_class_to_idx.items(), key=lambda x: x[1])
        raise ValueError(
            f"[{model_name}] class mapping mismatch between checkpoint and dataset. "
            f"checkpoint_first={ckpt_items[:5]} dataset_first={ds_items[:5]}"
        )
    return normalized

@torch.no_grad()
def _evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    model_name: str,
    num_classes: int,
    collect_confusion: bool = False,
) -> tuple[float, float, torch.Tensor | None]:
    """
	Evaluate a model on a single dataset split and optionally compute confusion.

	Runs inference in eval mode, accumulates dataset-averaged loss/accuracy, and
	optionally builds a `num_classes x num_classes` confusion matrix.

	Args:
		model: Model to evaluate.
		loader: DataLoader yielding (images, labels).
		criterion: Loss function.
		device: Device to run computation on.
		split_name: Split label (e.g., "val", "test") for progress text.
		model_name: Model identifier used for progress/error context.
		num_classes: Number of classes for metrics/confusion sizing.
		collect_confusion: If True, return a confusion matrix.

	Returns:
		(split_loss, split_acc, confusion) where confusion is a tensor or None.

	Raises:
		ValueError: If computed loss/accuracy are non-finite.
	"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64) if collect_confusion else None

    progress = tqdm(loader, desc=f"{model_name}-{split_name}", leave=True)
    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        batch_loss = float(loss.item())
        total_seen += batch_size
        total_loss += batch_loss * batch_size
        total_correct += (preds == labels).sum().item()

        if confusion is not None:
            idx = labels * num_classes + preds
            binc = torch.bincount(idx, minlength=num_classes * num_classes)
            confusion += binc.reshape(num_classes, num_classes).cpu()

        progress.set_postfix(
            batch_loss=f"{batch_loss:.4f}",
            loss=f"{(total_loss / total_seen):.4f}",
            acc=f"{(total_correct / total_seen):.4f}"
        )

    split_loss = total_loss / total_seen
    split_acc = total_correct / total_seen
    if not (math.isfinite(split_loss) and math.isfinite(split_acc)):
        raise ValueError(f"[{model_name}] non-finite metrics on split={split_name}: loss={split_loss}, acc={split_acc}")
    return split_loss, split_acc, confusion

def evaluate_model_from_checkpoint(
    model_name: str,
    ckpt_path: Path,
    cfg: Any,
    loaders: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
	Evaluate a saved checkpoint on train/val/test splits and return metrics.

	Loads `ckpt_path`, verifies the checkpoint model name and class mapping,
	rebuilds the model, restores weights, and computes loss/accuracy for each
	split. Also computes and returns a test confusion matrix.

	Args:
		model_name: Expected model identifier ("baseline" or "advanced").
		ckpt_path: Path to the checkpoint file to load.
		cfg: Config-like object used to instantiate the model.
		loaders: Shared-eval loaders/metadata dict (expects train/val/test loaders
			and `class_to_idx`).
		device: Device to run evaluation on.

	Returns:
		Dictionary containing checkpoint metadata, per-split loss/accuracy, and
		the test confusion matrix.

	Raises:
		ValueError: If checkpoint model name or class mapping does not match.
		RuntimeError: If confusion matrix computation fails.
	"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_model_name = str(checkpoint.get("model_name", model_name)).strip().lower()
    if ckpt_model_name != model_name:
        raise ValueError(
            f"[{model_name}] checkpoint model_name={ckpt_model_name} does not match expected model_name={model_name}"
        )

    class_to_idx = _validate_class_mapping(model_name, checkpoint.get("class_to_idx"), loaders["class_to_idx"])
    num_classes = len(class_to_idx)

    model = _build_model(model_name=model_name, cfg=cfg, num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc, _ = _evaluate_split(
        model=model,
        loader=loaders["train"],
        criterion=criterion,
        device=device,
        split_name="train",
        model_name=model_name,
        num_classes=num_classes,
        collect_confusion=False,
    )
    val_loss, val_acc, _ = _evaluate_split(
        model=model,
        loader=loaders["val"],
        criterion=criterion,
        device=device,
        split_name="val",
        model_name=model_name,
        num_classes=num_classes,
        collect_confusion=False,
    )
    test_loss, test_acc, test_cm = _evaluate_split(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        split_name="test",
        model_name=model_name,
        num_classes=num_classes,
        collect_confusion=True,
    )
    if test_cm is None:
        raise RuntimeError(f"[{model_name}] confusion matrix computation failed.")

    return {
        "checkpoint_path": str(ckpt_path),
        "best_epoch_in_ckpt": int(checkpoint.get("epoch", -1)),
        "class_to_idx": class_to_idx,
        "split_metrics": {
            "train": {"loss": train_loss, "acc": train_acc},
            "val": {"loss": val_loss, "acc": val_acc},
            "test": {"loss": test_loss, "acc": test_acc},
        },
        "test_confusion_matrix": test_cm,
    }

def print_model_summary(model_name: str, result: dict[str, Any], metrics_json: dict[str, Any] | None) -> None:
    """
	Print a concise evaluation summary for a model.

	Displays checkpoint path, best epoch from checkpoint, optional best epoch
	from `metrics.json`, and formatted loss/accuracy for train/val/test.

	Args:
		model_name: Model identifier label for the summary header.
		result: Evaluation result dict (expects split_metrics, checkpoint_path,
			best_epoch_in_ckpt).
		metrics_json: Optional parsed metrics.json dict for reporting best_epoch.

	Returns:
		None
	"""
    split = result["split_metrics"]
    best_epoch_json = metrics_json.get("best_epoch") if isinstance(metrics_json, dict) else None

    print("")
    print(f"[summary:{model_name}]")
    print(f"  checkpoint: {result['checkpoint_path']}")
    print(f"  best_epoch (from checkpoint): {result['best_epoch_in_ckpt']}")
    if best_epoch_json is not None:
        print(f"  best_epoch (from metrics.json): {best_epoch_json}")
    print(
        f"  train: loss={split['train']['loss']:.4f} acc={split['train']['acc']:.4f} | "
        f"val: loss={split['val']['loss']:.4f} acc={split['val']['acc']:.4f} | "
        f"test: loss={split['test']['loss']:.4f} acc={split['test']['acc']:.4f}"
    )

def head_to_head_summary(
    baseline: dict[str, Any],
    advanced: dict[str, Any],
) -> dict[str, Any]:
    """
	Print and return a head-to-head test comparison between two model results.

	Computes test accuracy/loss deltas (advanced - baseline), relative accuracy
	gain vs baseline, declares the winner by test accuracy, prints a summary,
	and returns a dict of the computed comparison metrics.

	Args:
		baseline: Baseline evaluation result dict (expects split_metrics->test).
		advanced: Advanced evaluation result dict (expects split_metrics->test).

	Returns:
		Dictionary with test accuracy delta, relative gain percentage, test loss
		delta, and the winner on test accuracy.
	"""
    base_test_acc = baseline["split_metrics"]["test"]["acc"]
    adv_test_acc = advanced["split_metrics"]["test"]["acc"]
    base_test_loss = baseline["split_metrics"]["test"]["loss"]
    adv_test_loss = advanced["split_metrics"]["test"]["loss"]

    acc_delta = adv_test_acc - base_test_acc
    loss_delta = adv_test_loss - base_test_loss
    rel_gain_pct = (acc_delta / base_test_acc * 100.0) if base_test_acc > 0 else float("inf")
    winner = "advanced" if adv_test_acc >= base_test_acc else "baseline"

    print("")
    print("[head-to-head:test]")
    print(f"  baseline_test_acc={base_test_acc:.6f} advanced_test_acc={adv_test_acc:.6f}")
    print(f"  baseline_test_loss={base_test_loss:.6f} advanced_test_loss={adv_test_loss:.6f}")
    print(f"  accuracy_delta(advanced-baseline)={acc_delta:+.6f}")
    print(f"  relative_accuracy_gain={rel_gain_pct:+.2f}%")
    print(f"  loss_delta(advanced-baseline)={loss_delta:+.6f}")
    if winner == "advanced":
        print(f"  verdict: advanced outperforms baseline on test accuracy by {acc_delta:.4f} points")
    else:
        print(f"  verdict: baseline outperforms advanced on test accuracy by {abs(acc_delta):.4f} points")

    return {
        "test_accuracy_delta_advanced_minus_baseline": acc_delta,
        "test_accuracy_relative_gain_pct_vs_baseline": rel_gain_pct,
        "test_loss_delta_advanced_minus_baseline": loss_delta,
        "winner_on_test_accuracy": winner,
    }

def plot_split_metrics_comparison(
    baseline: dict[str, Any],
    advanced: dict[str, Any],
    output_path: Path,
) -> None:
    """
	Plot baseline vs advanced loss/accuracy across train/val/test and save the figure.

	Creates a two-panel bar chart (loss and accuracy) comparing `baseline` and
	`advanced` split metrics, saves it to `output_path`, displays it, and closes
	the figure.

	Args:
		baseline: Baseline evaluation result dict (expects split_metrics per split).
		advanced: Advanced evaluation result dict (expects split_metrics per split).
		output_path: Destination path for the saved plot image.

	Returns:
		None
	"""
    splits = ["train", "val", "test"]
    baseline_losses = [baseline["split_metrics"][s]["loss"] for s in splits]
    advanced_losses = [advanced["split_metrics"][s]["loss"] for s in splits]
    baseline_accs = [baseline["split_metrics"][s]["acc"] for s in splits]
    advanced_accs = [advanced["split_metrics"][s]["acc"] for s in splits]

    x = list(range(len(splits)))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar([i - width / 2 for i in x], baseline_losses, width=width, label="baseline", color="#4C72B0")
    axes[0].bar([i + width / 2 for i in x], advanced_losses, width=width, label="advanced", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_title("Loss by Split (best.pth)")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.2, linestyle="--")
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x], baseline_accs, width=width, label="baseline", color="#4C72B0")
    axes[1].bar([i + width / 2 for i in x], advanced_accs, width=width, label="advanced", color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_title("Accuracy by Split (best.pth)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.2, linestyle="--")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.show()
    plt.close(fig)

def plot_test_head_to_head(
    baseline: dict[str, Any],
    advanced: dict[str, Any],
    deltas: dict[str, Any],
    output_path: Path,
) -> None:
    """
	Plot a baseline vs advanced head-to-head comparison on the test split.

	Creates two bar charts (test accuracy and test loss), annotates each with the
	advanced-minus-baseline delta from `deltas`, saves the figure to `output_path`,
	displays it, and closes the figure.

	Args:
		baseline: Baseline evaluation result dict (expects split_metrics->test).
		advanced: Advanced evaluation result dict (expects split_metrics->test).
		deltas: Dict containing test metric deltas (expects accuracy/loss delta keys).
		output_path: Destination path for the saved plot image.

	Returns:
		None
	"""
    models = ["baseline", "advanced"]
    test_acc = [baseline["split_metrics"]["test"]["acc"], advanced["split_metrics"]["test"]["acc"]]
    test_loss = [baseline["split_metrics"]["test"]["loss"], advanced["split_metrics"]["test"]["loss"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(models, test_acc, color=["#4C72B0", "#DD8452"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Test Accuracy Head-to-Head")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.2, linestyle="--")
    axes[0].text(
        0.5,
        0.05,
        f"delta (adv-base): {deltas['test_accuracy_delta_advanced_minus_baseline']:+.4f}",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    axes[1].bar(models, test_loss, color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Test Loss Head-to-Head")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.2, linestyle="--")
    axes[1].text(
        0.5,
        0.05,
        f"delta (adv-base): {deltas['test_loss_delta_advanced_minus_baseline']:+.4f}",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.show()
    plt.close(fig)

def plot_training_curves_comparison(
    baseline_history: list[dict[str, Any]] | None,
    advanced_history: list[dict[str, Any]] | None,
    output_path: Path,
) -> None:
    """
	Plot training/validation curves for baseline vs advanced runs and save the figure.

	If either history is missing, prints a warning and returns. Otherwise plots:
	- Train/val loss curves
	- Train/val accuracy curves
	- Learning rate curves
	Saves to `output_path`, displays the figure, and closes it.

	Args:
		baseline_history: Baseline training history (list of per-epoch dicts) or None.
		advanced_history: Advanced training history (list of per-epoch dicts) or None.
		output_path: Destination path for the saved plot image.

	Returns:
		None
	"""
    if baseline_history is None or advanced_history is None:
        print("[warn] Skipping training_curves_comparison.png due to missing/corrupt history.")
        return

    def _series(history: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row.get(key, 0.0)) for row in history]

    base_epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(baseline_history)]
    adv_epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(advanced_history)]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(base_epochs, _series(baseline_history, "train_loss"), color="#4C72B0", label="baseline-train")
    axes[0].plot(base_epochs, _series(baseline_history, "val_loss"), color="#4C72B0", linestyle="--", label="baseline-val")
    axes[0].plot(adv_epochs, _series(advanced_history, "train_loss"), color="#DD8452", label="advanced-train")
    axes[0].plot(adv_epochs, _series(advanced_history, "val_loss"), color="#DD8452", linestyle="--", label="advanced-val")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2, linestyle="--")
    axes[0].legend(fontsize=8)

    axes[1].plot(base_epochs, _series(baseline_history, "train_acc"), color="#4C72B0", label="baseline-train")
    axes[1].plot(base_epochs, _series(baseline_history, "val_acc"), color="#4C72B0", linestyle="--", label="baseline-val")
    axes[1].plot(adv_epochs, _series(advanced_history, "train_acc"), color="#DD8452", label="advanced-train")
    axes[1].plot(adv_epochs, _series(advanced_history, "val_acc"), color="#DD8452", linestyle="--", label="advanced-val")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.2, linestyle="--")
    axes[1].legend(fontsize=8)

    axes[2].plot(base_epochs, _series(baseline_history, "lr"), color="#4C72B0", label="baseline")
    axes[2].plot(adv_epochs, _series(advanced_history, "lr"), color="#DD8452", label="advanced")
    axes[2].set_title("Learning Rate Curves")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(alpha=0.2, linestyle="--")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.show()
    plt.close(fig)

def plot_confusion_matrices_comparison(
    baseline_cm: torch.Tensor,
    advanced_cm: torch.Tensor,
    class_names: list[str],
    output_path: Path,
) -> None:
    """
	Plot and save side-by-side confusion matrices for baseline and advanced models.

	Renders two heatmaps (baseline and advanced) with shared class labels on both
	axes, adds a colorbar per plot, saves the figure to `output_path`, displays it,
	and closes the figure.

	Args:
		baseline_cm: Confusion matrix tensor for the baseline model (C x C).
		advanced_cm: Confusion matrix tensor for the advanced model (C x C).
		class_names: Class labels in index order (length C).
		output_path: Destination path for the saved plot image.

	Returns:
		None
	"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, cm, title in (
        (axes[0], baseline_cm, "Baseline Test Confusion Matrix"),
        (axes[1], advanced_cm, "Advanced Test Confusion Matrix"),
    ):
        im = ax.imshow(cm.numpy(), interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=5)
        ax.set_yticklabels(class_names, fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.show()
    plt.close(fig)

def _compute_per_class_stats(cm: torch.Tensor) -> tuple[list[int], list[int], list[float]]:
    """
	Compute per-class support, correct counts, and accuracy from a confusion matrix.

	Args:
		cm: Confusion matrix tensor of shape (C, C).

	Returns:
		Tuple of (support, correct, accuracy) where:
		- support: total true samples per class
		- correct: diagonal counts per class
		- accuracy: correct/support per class (0.0 when support is 0)
	"""
    support = cm.sum(dim=1)
    correct = cm.diag()
    acc = torch.where(support > 0, correct.float() / support.float(), torch.zeros_like(correct, dtype=torch.float32))
    return (
        [int(x.item()) for x in support],
        [int(x.item()) for x in correct],
        [float(x.item()) for x in acc],
    )


def plot_and_save_per_class_comparison(
    class_names: list[str],
    baseline_cm: torch.Tensor,
    advanced_cm: torch.Tensor,
    output_png: Path,
    output_csv: Path,
) -> None:
    """
	Compute per-class test accuracy for baseline/advanced, save CSV, and plot comparison.

	Derives per-class support/correct/accuracy from both confusion matrices, writes a
	CSV sorted by (advanced - baseline) accuracy delta, and generates a horizontal
	bar chart comparing per-class accuracies. Saves the plot to `output_png`,
	displays it, and closes the figure.

	Args:
		class_names: Class labels in index order.
		baseline_cm: Baseline confusion matrix tensor (C x C).
		advanced_cm: Advanced confusion matrix tensor (C x C).
		output_png: Destination path for the saved plot image.
		output_csv: Destination path for the per-class comparison CSV.

	Returns:
		None
	"""
    b_support, b_correct, b_acc = _compute_per_class_stats(baseline_cm)
    a_support, a_correct, a_acc = _compute_per_class_stats(advanced_cm)

    rows: list[dict[str, Any]] = []
    for idx, class_name in enumerate(class_names):
        rows.append(
            {
                "class": class_name,
                "baseline_support": b_support[idx],
                "advanced_support": a_support[idx],
                "baseline_correct": b_correct[idx],
                "advanced_correct": a_correct[idx],
                "baseline_acc": b_acc[idx],
                "advanced_acc": a_acc[idx],
                "delta_advanced_minus_baseline": a_acc[idx] - b_acc[idx],
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["delta_advanced_minus_baseline"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "class",
                "baseline_support",
                "advanced_support",
                "baseline_correct",
                "advanced_correct",
                "baseline_acc",
                "advanced_acc",
                "delta_advanced_minus_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    labels = [row["class"] for row in rows_sorted]
    baseline_vals = [row["baseline_acc"] for row in rows_sorted]
    advanced_vals = [row["advanced_acc"] for row in rows_sorted]

    y = list(range(len(labels)))
    fig_h = max(10.0, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.barh([i - 0.2 for i in y], baseline_vals, height=0.4, color="#4C72B0", label="baseline")
    ax.barh([i + 0.2 for i in y], advanced_vals, height=0.4, color="#DD8452", label="advanced")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Per-class Test Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Per-Class Accuracy Comparison (advanced vs baseline)")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.show()
    plt.close(fig)

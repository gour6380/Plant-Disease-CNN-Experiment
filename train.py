from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from dataset import build_dataloaders
from models import AdvancedCNN, BaselineCNN

def _to_serializable(value: Any) -> Any:
    """
	Recursively convert common Python objects into JSON-serializable forms.

	Converts:
	- `Path` -> `str`
	- `tuple`/`list` -> lists with converted elements
	- `dict` -> stringified keys with converted values

	Args:
		value: Arbitrary value to convert.

	Returns:
		A JSON-serializable representation of `value` when possible.
	"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value

def _config_to_dict(cfg: Any) -> dict[str, Any]:
    """
	Convert a config object into a JSON-friendly dictionary.

	Iterates over `vars(cfg)`, skipping dunder attributes and callables, and
	recursively serializes values via `_to_serializable`.

	Args:
		cfg: Config-like object with attributes.

	Returns:
		Dictionary of config fields suitable for JSON serialization.
	"""
    payload: dict[str, Any] = {}
    for key, value in vars(cfg).items():
        if key.startswith("__"):
            continue
        if callable(value):
            continue
        payload[key] = _to_serializable(value)
    return payload

def _set_seed(seed: int) -> None:
    """
	Set random seeds for reproducibility across Python and PyTorch backends.

	Seeds Python's `random`, PyTorch CPU RNG, CUDA RNGs (if available), and
	MPS RNG (if supported by the installed PyTorch build).

	Args:
		seed: Seed value to apply.

	Returns:
		None
	"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

def _batch_to_device(images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
	Move a batch of tensors to the target device.

	Args:
		images: Batch of input images.
		labels: Batch of target labels.
		device: Destination `torch.device`.

	Returns:
		Tuple of (images, labels) moved to `device`.
	"""
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return images, labels

def _run_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
	Run one training epoch over a dataloader.

	Sets the model to train mode, iterates batches with a tqdm progress bar,
	performs forward/backward/optimizer steps, and tracks running loss/accuracy.

	Args:
		model: Model to train.
		loader: Training dataloader yielding (images, labels).
		criterion: Loss function.
		optimizer: Optimizer to update model parameters.
		device: Device to run computation on.
		epoch_idx: 1-based epoch index for display/logging.
		total_epochs: Total number of epochs for display/logging.

	Returns:
		Tuple of (epoch_loss, epoch_accuracy).
	"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    progress = tqdm(loader, desc=f"train {epoch_idx}/{total_epochs}", leave=True)
    for images, labels in progress:
        images, labels = _batch_to_device(images, labels, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_seen += batch_size
        batch_loss = float(loss.item())
        total_loss += batch_loss * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

        running_loss = total_loss / total_seen
        running_acc = total_correct / total_seen
        lr_now = float(optimizer.param_groups[0]["lr"])
        progress.set_postfix(
            batch_loss=f"{batch_loss:.4f}",
            loss=f"{running_loss:.4f}",
            acc=f"{running_acc:.4f}",
            lr=f"{lr_now:.6f}",
        )

    return total_loss / total_seen, total_correct / total_seen

@torch.no_grad()
def _run_eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    epoch_idx: int | None = None,
    total_epochs: int | None = None,
) -> tuple[float, float]:
    """
	Evaluate one epoch over a dataloader without gradient computation.

	Sets the model to eval mode, iterates batches with a tqdm progress bar,
	computes loss/accuracy, and returns dataset-averaged metrics.

	Args:
		model: Model to evaluate.
		loader: Dataloader yielding (images, labels).
		criterion: Loss function.
		device: Device to run computation on.
		split_name: Label for the split (e.g., "val", "test") used in progress text.
		epoch_idx: Optional 1-based epoch index for display/logging.
		total_epochs: Optional total epochs for display/logging.

	Returns:
		Tuple of (epoch_loss, epoch_accuracy).
	"""
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0

    if epoch_idx is not None and total_epochs is not None:
        desc = f"{split_name} {epoch_idx}/{total_epochs}"
    else:
        desc = split_name

    progress = tqdm(loader, desc=desc, leave=False)
    for step, (images, labels) in enumerate(progress, start=1):
        images, labels = _batch_to_device(images, labels, device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_seen += batch_size
        batch_loss = float(loss.item())
        total_loss += batch_loss * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

        running_loss = total_loss / total_seen
        running_acc = total_correct / total_seen
        progress.set_postfix(
            step=f"{step}/{len(loader)}",
            batch_loss=f"{batch_loss:.4f}",
            loss=f"{running_loss:.4f}",
            acc=f"{running_acc:.4f}",
        )

    return total_loss / total_seen, total_correct / total_seen

def _save_json(path: Path, payload: Any) -> None:
    """
	Write a JSON file, creating parent directories as needed.

	Args:
		path: Output JSON file path.
		payload: JSON-serializable object to write.

	Returns:
		None
	"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_acc: float,
    class_to_idx: dict[str, int],
    model_name: str,
) -> None:
    """
	Save a training checkpoint with model, optimizer, scheduler, and metadata.

	Creates parent directories for `path` and writes a `torch.save` payload
	including epoch, best validation accuracy, model name, state dicts, and
	the class-to-index mapping.

	Args:
		path: Destination checkpoint path.
		model: Model whose parameters will be saved.
		optimizer: Optimizer whose state will be saved.
		scheduler: LR scheduler whose state will be saved.
		epoch: Epoch number to record.
		best_val_acc: Best validation accuracy achieved so far.
		class_to_idx: Mapping of class name to index.
		model_name: Human-readable model identifier.

	Returns:
		None
	"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_to_idx": class_to_idx,
        },
        path,
    )

def _resolve_output_dir(cfg: Any) -> Path:
    """
	Resolve the model output directory from config, with legacy-path remapping.

	Builds the default output dir as `<output_root>/<model_name>_cnn`. If
	`cfg.output_dir` is unset, returns the default. If `cfg.output_dir` points
	to a legacy path (e.g., baseline_cnn/advanced_cnn) that doesn't match the
	expected default, remaps to the expected default.

	Args:
		cfg: Config-like object with `model_name`, and optional `output_root`
			and `output_dir`.

	Returns:
		Path to the resolved output directory.
	"""
    model_name = str(cfg.model_name).strip().lower()
    default_root = Path(getattr(cfg, "output_root", Path("artifacts")))
    expected_default = default_root / f"{model_name}_cnn"

    configured = getattr(cfg, "output_dir", None)
    if configured is None:
        return expected_default

    configured_path = Path(configured)
    legacy_paths = {
        default_root / "baseline_cnn",
        default_root / "advanced_cnn",
    }
    if configured_path in legacy_paths and configured_path != expected_default:
        return expected_default
    return configured_path

def _build_model(cfg: Any, class_count: int, device: torch.device) -> tuple[nn.Module, str]:
    """
	Instantiate the requested CNN model and validate configuration constraints.

	Builds either `BaselineCNN` or `AdvancedCNN` based on `cfg.model_name`,
	moves it to `device`, and verifies input size/output dimensions. The
	baseline path enforces fixed 256x256 inputs and a config constraint on
	`cfg.num_classes`.

	Args:
		cfg: Config-like object with `model_name`, image size fields, and
			advanced-model settings (e.g., `advanced_dropout`).
		class_count: Number of classes for the dataset/output head.
		device: Target device for the model.

	Returns:
		Tuple of (model, resolved_model_name).

	Raises:
		ValueError: If `model_name` is unsupported, image size is incompatible,
			required config constraints fail, or output dimension mismatches
			`class_count`.
	"""
    model_name = str(cfg.model_name).strip().lower()
    if int(cfg.image_width) != 256 or int(cfg.image_height) != 256:
        raise ValueError(
            "BaselineCNN expects 256x256 inputs due to fixed classifier dimensions. "
            f"Got {cfg.image_width}x{cfg.image_height}."
        )
    if int(cfg.num_classes) != 38:
        raise ValueError("Config num_classes must be 38 for current BaselineCNN output head.")

    if model_name == "baseline":
        model = BaselineCNN(input_features=3).to(device)
    elif model_name == "advanced":
        model = AdvancedCNN(
            num_classes=class_count,
            in_channels=3,
            dropout=float(cfg.advanced_dropout),
        ).to(device)
        return model, model_name
    else:
        raise ValueError(f"Unsupported model_name: {cfg.model_name}. Use one of: baseline, advanced")
    
    output_dim = int(model.classifier[-1].out_features)
    if output_dim != class_count:
        raise ValueError(f"Model output dim ({output_dim}) must equal class count ({class_count}).")
    return model, model_name

def train_from_config(cfg: Any) -> None:
    """
	Train a CNN experiment end-to-end using a config object.

	Sets precision/seed, builds dataloaders, validates shapes/class counts,
	resolves the output directory, snapshots config/class mappings, trains for
	`cfg.epochs` with checkpointing (best + last), saves training history, then
	evaluates the chosen checkpoint on the test split and writes a metrics JSON.

	Args:
		cfg: Config-like object providing dataset, model, training, checkpoint,
			and output settings.

	Returns:
		None

	Raises:
		ValueError: If class count, expected image shape, model constraints, or
			output dimensions do not match configuration expectations.
	"""
    torch.set_float32_matmul_precision("high")
    _set_seed(int(cfg.seed))
    device = str(cfg.device)

    loaders = build_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    class_to_idx: dict[str, int] = loaders["class_to_idx"]
    class_count = len(class_to_idx)

    if class_count != int(cfg.num_classes):
        raise ValueError(f"Class count mismatch: found {class_count}, config says {cfg.num_classes}")
    if int(cfg.num_classes) != 38:
        raise ValueError("Config num_classes must be 38 for current BaselineCNN output head.")

    sample_images, _ = next(iter(train_loader))
    expected_shape = (3, int(cfg.image_height), int(cfg.image_width))
    if tuple(sample_images.shape[1:]) != expected_shape:
        raise ValueError(
            f"Unexpected train batch image shape {tuple(sample_images.shape[1:])}, expected {expected_shape}"
        )
    print(f"[data] sample batch shape = {tuple(sample_images.shape)}")
    print(
        f"[data] sizes train={loaders['train_size']} val={loaders['val_size']} test={loaders['test_size']} classes={class_count}"
    )
    print(f"[data] normalization_mean={loaders['normalization_mean']} normalization_std={loaders['normalization_std']}")
    print(f"[data] weighted_sampler={'enabled' if loaders['train_sampler'] is not None else 'disabled'}")

    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = _config_to_dict(cfg)
    config_snapshot["resolved_output_dir"] = str(output_dir)
    _save_json(output_dir / "config_snapshot.json", config_snapshot)
    _save_json(output_dir / "class_to_idx.json", class_to_idx)

    model, model_name = _build_model(cfg, class_count, device)
    with torch.no_grad():
        probe_logits = model(sample_images[:1].to(device, non_blocking=True))
    if int(probe_logits.shape[-1]) != class_count:
        raise ValueError(
            f"Model output dim ({int(probe_logits.shape[-1])}) must equal class count ({class_count})."
        )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] name={model_name} trainable_params={trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(cfg.epochs)))

    best_ckpt_path = output_dir / str(cfg.best_ckpt_name)
    last_ckpt_path = output_dir / str(cfg.last_ckpt_name)

    history: list[dict[str, Any]] = []
    best_val_acc = float("-inf")
    best_epoch = -1

    print(f"[device] using {device}")
    print(f"[train] starting training for {cfg.epochs} epoch(s)")
    for epoch in range(1, int(cfg.epochs) + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = _run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=int(cfg.epochs),
        )
        val_loss, val_acc = _run_eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split_name="val",
            epoch_idx=epoch,
            total_epochs=int(cfg.epochs),
        )

        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        elapsed_sec = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr_now,
            "epoch_time_sec": elapsed_sec,
        }
        history.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            _save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc,
                class_to_idx=class_to_idx,
                model_name=model_name,
            )

        _save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_acc=best_val_acc,
            class_to_idx=class_to_idx,
            model_name=model_name,
        )

        print(
            f"[epoch {epoch}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.6f}"
        )

    _save_json(output_dir / str(cfg.history_file), history)

    ckpt_to_test = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
    state = torch.load(ckpt_to_test, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    test_loss, test_acc = _run_eval_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        split_name="test",
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "device": str(device),
        "train_size": loaders["train_size"],
        "val_size": loaders["val_size"],
        "test_size": loaders["test_size"],
        "num_classes": class_count,
        "model_name": model_name,
        "checkpoint_evaluated": str(ckpt_to_test),
    }
    _save_json(output_dir / str(cfg.metrics_file), metrics)

    print(
        f"[done] best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
    )
    print(f"[done] artifacts written to {output_dir.resolve()}")
    if cfg.device == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    return None
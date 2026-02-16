from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from transform import build_transforms
from tqdm import tqdm


def _norm_exts(allowed_exts: tuple[str, ...]) -> set[str]:
    """
	Normalize a tuple of allowed file extensions to a lowercase set.

	Args:
		allowed_exts: File extensions (e.g., (".JPG", ".jpg")).

	Returns:
		Set of lowercase extensions.
	"""
    return {ext.lower() for ext in allowed_exts}

def _list_classes(root: Path) -> list[str]:
    """
	List non-hidden class directory names under a dataset root.

	Args:
		root: Directory whose immediate subdirectories represent classes.

	Returns:
		Sorted list of class folder names.
	"""
    return sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])

def _collect_samples(
    split_root: Path,
    class_to_idx: dict[str, int],
    allowed_exts: tuple[str, ...]
    )-> list[tuple[Path, int]]:
    """
	Collect (path, class_index) samples for a split directory.

	Iterates classes in `class_to_idx` order, scans each class subdirectory under
	`split_root`, filters files by `allowed_exts`, and returns labeled paths.

	Args:
		split_root: Root directory for a dataset split.
		class_to_idx: Mapping of class name to integer index.
		allowed_exts: Allowed file extensions (typically lowercase).

	Returns:
		List of (file_path, class_index) tuples.
	"""
    samples: list[tuple[Path, int]] = []
    for class_name in sorted(class_to_idx):
        class_idx = class_to_idx[class_name]
        class_dir = split_root / class_name
        files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts])
        samples.extend(((p, class_idx) for p in files))
    return samples

class PlantVillageDataset(Dataset):
    """
	A simple image classification dataset backed by (path, label) samples.

	Loads an image from disk, converts it to RGB, applies an optional torchvision
	transform pipeline, and returns the transformed tensor with its integer label.

	Args:
		samples: List of (image_path, class_index) tuples.
		transform: Optional torchvision transform applied to the loaded image.

	Returns:
		__getitem__ returns (image_tensor, label).
	"""
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose | None = None) -> None:
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def _validate_split_classes(train_classes: list[str], other_classes: list[str], split_name: str) -> None:
    """
	Validate that a split's class names match the training split classes.

	Checks for missing and extra class names relative to `train_classes` and
	raises with a brief summary if a mismatch is found.

	Args:
		train_classes: Class names present in the training split.
		other_classes: Class names found in another split.
		split_name: Name of the split being validated (for error context).

	Returns:
		None

	Raises:
		ValueError: If the split class set differs from the training class set.
	"""
    missing = sorted(set(train_classes) - set(other_classes))
    extra = sorted(set(other_classes) - set(train_classes))
    if missing or extra:
        raise ValueError(
            f"{split_name} classes do not match train classes. Missing={missing[:5]} Extra={extra[:5]}"
        )

def _loader_kwargs(cfg: Any) -> dict[str, Any]:
    """
	Build DataLoader keyword arguments from config, normalizing prefetch_factor.

	Sets batch size, worker count, pinning, persistent workers, and prefetch factor.
	If `cfg.prefetch_factor` is falsy, uses None.

	Args:
		cfg: Config-like object providing DataLoader settings.

	Returns:
		Dictionary of kwargs suitable for constructing a PyTorch DataLoader.
	"""
    prefetch_factor = None if not cfg.prefetch_factor else int(cfg.prefetch_factor)
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
        "persistent_workers": bool(cfg.persistent_workers),
        "prefetch_factor": prefetch_factor
    }
    return kwargs

def _validate_stats_payload(payload: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """
	Validate and parse a normalization-stats payload.

	Expects a dict with "mean" and "std" as length-3 lists of numeric values, and
	requires all std values to be > 0.

	Args:
		payload: Parsed JSON-like object to validate.

	Returns:
		(mean, std) as tuples of floats if valid, otherwise None.
	"""
    if not isinstance(payload, dict):
        return None
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        return None
    if len(mean) != 3 or len(std) != 3:
        return None
    try:
        mean_t = tuple(float(x) for x in mean)
        std_t = tuple(float(x) for x in std)
    except (TypeError, ValueError):
        return None
    if any(v <= 0.0 for v in std_t):
        return None
    return mean_t, std_t

def _compute_normalization_stats(
    train_samples: list[tuple[Path, int]],
    image_height: int,
    image_width: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
	Compute dataset-wide RGB normalization mean/std from training samples.

	Resizes each image to (image_height, image_width), converts to a float tensor
	in [0, 1], and accumulates per-channel sums and squared sums to derive mean
	and standard deviation.

	Args:
		train_samples: List of (image_path, class_index) tuples.
		image_height: Target height used for resizing during stats computation.
		image_width: Target width used for resizing during stats computation.

	Returns:
		(mean, std) as tuples of 3 floats (RGB).
	
	Raises:
		ValueError: If no pixels are processed (pixel_count == 0).
	"""
    resize = transforms.Resize((image_height, image_width))
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    progress = tqdm(train_samples, desc="Computing normalization stats", unit="img", leave=False)
    for path, _ in progress:
        with Image.open(path) as img:
            image = resize(img.convert("RGB"))
        tensor = transforms.functional.pil_to_tensor(image).to(dtype=torch.float64) / 255.0
        c, h, w = tensor.shape
        flat = tensor.view(c, -1)
        channel_sum += flat.sum(dim=1)
        channel_sq_sum += (flat * flat).sum(dim=1)
        pixel_count += h * w

    if pixel_count == 0:
        raise ValueError("Cannot compute normalization stats: pixel_count is 0.")

    mean = channel_sum / pixel_count
    var = (channel_sq_sum / pixel_count) - (mean * mean)
    std = torch.sqrt(var.clamp(min=1e-12))

    mean_t = tuple(float(v) for v in mean.tolist())
    std_t = tuple(float(v) for v in std.tolist())
    return mean_t, std_t

def get_or_compute_normalization_stats(
    cfg: Any,
    train_samples: list[tuple[Path, int]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
	Load cached normalization stats or compute and cache them.

	Reads `cfg.normalization_stats_file` when available and valid (and when
	`cfg.recompute_normalization_stats` is False). Otherwise computes stats from
	`train_samples`, writes a JSON cache, and returns the computed mean/std.

	Args:
		cfg: Config-like object providing stats file path, image size, and
			recompute flag.
		train_samples: Training samples used to compute stats when needed.

	Returns:
		(mean, std) as tuples of 3 floats (RGB).
	"""
    stats_path = Path(cfg.normalization_stats_file)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_path.exists() and not bool(cfg.recompute_normalization_stats):
        try:
            with stats_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            parsed = _validate_stats_payload(payload)
            if parsed is not None:
                print(f"[stats] loaded normalization stats from {stats_path}")
                return parsed
            print(f"[stats] invalid stats cache at {stats_path}; recomputing.")
        except Exception as err:  # pragma: no cover - defensive
            print(f"[stats] failed to read stats cache ({err}); recomputing.")

    mean, std = _compute_normalization_stats(
        train_samples=train_samples,
        image_height=int(cfg.image_height),
        image_width=int(cfg.image_width),
    )
    payload = {
        "mean": list(mean),
        "std": list(std),
        "image_height": int(cfg.image_height),
        "image_width": int(cfg.image_width),
        "num_samples": len(train_samples),
    }
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"[stats] computed and cached normalization stats to {stats_path}")
    return mean, std

def build_dataloaders(cfg: Any) -> dict[str, Any]:
    """
	Build train/val/test DataLoaders and dataset metadata from a folder structure.

	Validates split directories and class consistency, collects labeled samples
	filtered by `cfg.allowed_exts`, computes or loads normalization stats,
	constructs transforms and datasets, optionally enables a weighted sampler for
	training, and returns loaders plus mappings and split sizes.

	Args:
		cfg: Config-like object providing split folder paths, allowed extensions,
			image size, DataLoader settings, and weighted-sampler options.

	Returns:
		Dictionary containing DataLoaders for {"train","val","test"} and metadata:
		class_to_idx, idx_to_class, class_counts, normalization_mean/std, sampler,
		and split sizes.

	Raises:
		FileNotFoundError: If any split folder does not exist.
		NotADirectoryError: If any split folder is not a directory.
		ValueError: If classes are missing/mismatched across splits or any split
			has no images after extension filtering.
	"""
    train_root = Path(cfg.train_folder)
    val_root = Path(cfg.val_folder)
    test_root = Path(cfg.test_folder)

    for split_root, split_name in (
        (train_root, "train"),
        (val_root, "val"),
        (test_root, "test"),
    ):
        if not split_root.exists():
            raise FileNotFoundError(f"{split_name} folder does not exist: {split_root}")
        if not split_root.is_dir():
            raise NotADirectoryError(f"{split_name} folder is not a directory: {split_root}")

    train_classes = _list_classes(train_root)
    if not train_classes:
        raise ValueError(f"No class subfolders found under train folder: {train_root}")

    val_classes = _list_classes(val_root)
    test_classes = _list_classes(test_root)
    _validate_split_classes(train_classes, val_classes, "val")
    _validate_split_classes(train_classes, test_classes, "test")

    class_to_idx = {name: idx for idx, name in enumerate(train_classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    allowed_exts = _norm_exts(tuple(cfg.allowed_exts))
    train_samples = _collect_samples(train_root, class_to_idx, allowed_exts)
    val_samples = _collect_samples(val_root, class_to_idx, allowed_exts)
    test_samples = _collect_samples(test_root, class_to_idx, allowed_exts)

    if not train_samples:
        raise ValueError(f"No training images found in {train_root} with extensions {sorted(allowed_exts)}")
    if not val_samples:
        raise ValueError(f"No validation images found in {val_root} with extensions {sorted(allowed_exts)}")
    if not test_samples:
        raise ValueError(f"No test images found in {test_root} with extensions {sorted(allowed_exts)}")

    class_counts = [0] * len(class_to_idx)
    for _, label in train_samples:
        class_counts[label] += 1

    mean, std = get_or_compute_normalization_stats(cfg=cfg, train_samples=train_samples)
    train_tfm, eval_tfm = build_transforms(int(cfg.image_height), int(cfg.image_width), mean, std)
    train_dataset = PlantVillageDataset(samples=train_samples, transform=train_tfm)
    val_dataset = PlantVillageDataset(samples=val_samples, transform=eval_tfm)
    test_dataset = PlantVillageDataset(samples=test_samples, transform=eval_tfm)

    sampler: WeightedRandomSampler | None = None
    shuffle = True
    if bool(cfg.use_weighted_sampler):
        class_weights = [0.0 if count == 0 else 1.0 / float(count) for count in class_counts]
        sample_weights = [class_weights[label] for _, label in train_samples]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    kwargs = _loader_kwargs(cfg)
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        **kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )

    cache_path = Path(cfg.class_names_cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(train_classes, fp, indent=2)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_counts": class_counts,
        "normalization_mean": mean,
        "normalization_std": std,
        "train_sampler": sampler,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }


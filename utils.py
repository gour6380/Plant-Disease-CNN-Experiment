from pathlib import Path
import os
from typing import Any
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from typing import Iterable
import torch
import json
import shutil


def get_classes2idx(path: Path) -> dict[str, int]:
    """
	Return a deterministic mapping from class folder names to integer indices.

	Scans immediate subdirectories under `path`, sorts them by directory name,
	and assigns indices starting from 0.

	Args:
		path: Root directory containing class subfolders.

	Returns:
		Mapping of class name -> class index.

	Raises:
		FileNotFoundError: If `path` does not exist.
		NotADirectoryError: If `path` is not a directory.
		ValueError: If no class subdirectories are found.
	"""
    if not path.exists():
        raise FileNotFoundError(f"get_classes2idx: path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"get_classes2idx: path is not a directory: {path}")

    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if not subdirs:
        raise ValueError(f"get_classes2idx: no class subdirectories found under: {path}")

    subdirs.sort(key=lambda p: p.name)
    classes2idx = {p.name: i for i, p in enumerate(subdirs)}

    return classes2idx

def class_wise_distribution(path: Path) -> "pd.DataFrame":
    """
	Compute per-class image counts for a folder-structured dataset.

	Counts `.jpg` and `.JPG` files inside each non-hidden class subdirectory under `path`.
	Class ids are assigned deterministically via `get_classes2idx(path)`.

	Args:
		path: Root directory whose immediate subdirectories represent classes.

	Returns:
		A pandas DataFrame with columns: ["id", "name", "total_images"].

	Raises:
		FileNotFoundError: If `path` does not exist.
		NotADirectoryError: If `path` is not a directory.
		ValueError: If no non-hidden class subdirectories are found.
	"""
    if not path.exists():
        raise FileNotFoundError(f"class_wise_distribution: path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"class_wise_distribution: path is not a directory: {path}")

    classes2idx = get_classes2idx(path)

    # non-hidden class dirs only
    items = [(cls_name, cls_id) for cls_name, cls_id in classes2idx.items() if not cls_name.startswith(".")]
    if not items:
        raise ValueError(f"class_wise_distribution: no non-hidden class subdirectories found under: {path}")

    rows: list[tuple[int, str, int]] = []

    for cls_name, cls_id in sorted(items, key=lambda x: x[1]):
        class_dir = path / cls_name
        if not class_dir.is_dir():
            continue

        n_images = 0
        for f in class_dir.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith("."):
                continue
            suf = f.suffix
            if suf == ".jpg" or suf == ".JPG":
                n_images += 1

        rows.append((cls_id, cls_name, n_images))

    df = pd.DataFrame(rows, columns=["id", "name", "total_images"])
    return df

def view_sample_images(path: Path, n_sample_per_class: int = 5, random_state: int = 42) -> None:
    """
	Display a grid of randomly sampled images per class subdirectory.

	Searches non-hidden class folders directly under `path`, samples up to
	`n_sample_per_class` `.jpg/.JPG` files per class using `random_state`,
	and renders them with matplotlib (expects `plt` and `PIL.Image` available).

	Args:
		path: Root directory whose immediate subdirectories represent classes.
		n_sample_per_class: Number of images to sample per class (> 0).
		random_state: Seed for reproducible sampling.

	Raises:
		FileNotFoundError: If `path` does not exist.
		NotADirectoryError: If `path` is not a directory.
		ValueError: If `n_sample_per_class` is not positive.
	"""
    if not path.exists():
        raise FileNotFoundError(f"view_sample_images: path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"view_sample_images: path is not a directory: {path}")
    if n_sample_per_class <= 0:
        raise ValueError(f"view_sample_images: n_sample_per_class must be > 0, got {n_sample_per_class}")

    random.seed(random_state)

    class_dirs = sorted(
        [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )

    if not class_dirs:
        print(f"[view_sample_images] No class subdirectories found under: {path}")
        return

    # Collect samples
    per_class: list[tuple[str, list[Path]]] = []
    for class_dir in class_dirs:
        files: list[Path] = []
        for f in class_dir.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith("."):
                continue
            suf = f.suffix
            if suf == ".jpg" or suf == ".JPG":
                files.append(f)

        if not files:
            per_class.append((class_dir.name, []))
            continue

        k = n_sample_per_class if n_sample_per_class < len(files) else len(files)
        per_class.append((class_dir.name, random.sample(files, k)))

    # Determine grid size
    n_classes = len(per_class)
    max_k = max((len(samples) for _, samples in per_class), default=0)

    if max_k == 0:
        print(f"[view_sample_images] No .jpg/.JPG images found under: {path}")
        return

    # assumes plt exists in notebook env (no new imports)
    fig, axes = plt.subplots(n_classes, max_k, figsize=(3.0 * max_k, 2.8 * n_classes))
    fig.suptitle(f"Samples per class (k={n_sample_per_class})", y=0.995)

    # Normalize axes to 2D list-like
    if n_classes == 1 and max_k == 1:
        axes = [[axes]]
    elif n_classes == 1:
        axes = [list(axes)]
    elif max_k == 1:
        axes = [[ax] for ax in axes]

    for r, (cls_name, samples) in enumerate(per_class):
        for c in range(max_k):
            ax = axes[r][c]
            ax.axis("off")

            if c >= len(samples):
                if c == 0:
                    ax.set_title(cls_name, fontsize=10)
                continue

            img_path = samples[c]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(cls_name if c == 0 else "", fontsize=10)
            except Exception as e:
                ax.set_title(cls_name if c == 0 else "", fontsize=10)
                ax.text(0.5, 0.5, f"Failed\n{img_path.name}", ha="center", va="center", fontsize=9)
                print(f"[view_sample_images] Failed to load: {img_path} err={type(e).__name__}: {e}")

    plt.tight_layout()
    plt.show()

def get_min_max_size(path: Path) -> None:
    """
    Print min/max image size stats for all images under class subfolders in `path`.

    - Recursively scans non-hidden class directories (immediate subdirs of `path`)
    - Includes `.jpg` / `.JPG` only
    - Excludes hidden files/dirs (starting with ".")
    - Uses tqdm progress bar (assumes tqdm is already available)

    Args:
        path: Root directory containing class subfolders.

    Raises:
        FileNotFoundError: If `path` does not exist.
        NotADirectoryError: If `path` is not a directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"get_min_max_size: path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"get_min_max_size: path is not a directory: {path}")

    min_wh: tuple[int, int] | None = None
    max_wh: tuple[int, int] | None = None
    min_path: Path | None = None
    max_path: Path | None = None

    total = 0
    failed = 0

    class_dirs = sorted(
        [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )

    # Build a file list once so tqdm has a stable total
    image_files: list[Path] = []
    for class_dir in class_dirs:
        for f in class_dir.rglob("*"):
            if not f.is_file():
                continue
            if f.name.startswith("."):
                continue
            suf = f.suffix
            if suf == ".jpg" or suf == ".JPG":
                image_files.append(f)

    if not image_files:
        print(f"[get_min_max_size] No .jpg/.JPG images found under: {path}")
        return

    for f in tqdm(image_files, desc="Scanning images", unit="img"):
        total += 1
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:
            failed += 1
            continue

        wh = (w, h)
        if min_wh is None or (wh[0] * wh[1]) < (min_wh[0] * min_wh[1]):
            min_wh = wh
            min_path = f
        if max_wh is None or (wh[0] * wh[1]) > (max_wh[0] * max_wh[1]):
            max_wh = wh
            max_path = f

    if min_wh is None or max_wh is None:
        print(f"[get_min_max_size] Found {total} images but couldn't read any (failed={failed}).")
        return

    print(f"[get_min_max_size] scanned={total} failed={failed}")
    print(f"[get_min_max_size] min_size(WxH)={min_wh[0]}x{min_wh[1]}")
    print(f"[get_min_max_size] max_size(WxH)={max_wh[0]}x{max_wh[1]}")

def select_device(device_pref: str) -> torch.device:
    """
	Select a torch device based on a user preference string.

	Supports "auto", "mps", "cuda", and "cpu". In "auto" mode, prefers MPS if
	available, then CUDA, otherwise CPU. If a requested accelerator is not
	available, prints a message and falls back to CPU.

	Args:
		device_pref: Device preference ("auto", "mps", "cuda", or "cpu").

	Returns:
		A `torch.device` matching the resolved preference.

	Raises:
		ValueError: If `device_pref` is not a supported option.
	"""
    pref = str(device_pref).strip().lower()
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    cuda_available = torch.cuda.is_available()

    if pref == "auto":
        if mps_available:
            return torch.device("mps")
        if cuda_available:
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "mps":
        if mps_available:
            return torch.device("mps")
        print("[device] MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if pref == "cuda":
        if cuda_available:
            return torch.device("cuda")
        print("[device] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {device_pref}. Use one of: auto, mps, cpu")

def setup_config_as_per_device(cfg: Any):
    """
	Mutate dataloader-related config fields based on the configured device.

	Sets `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers`
	for "mps", "cuda", or other (CPU) devices, then returns the updated config.

	Args:
		cfg: Config-like object with a `device` attribute and mutable fields.

	Returns:
		The same `cfg` object with updated settings.
	"""
    if cfg.device == "mps":
        cfg.num_workers = 4
        cfg.pin_memory = False
        cfg.prefetch_factor = 4
        cfg.persistent_workers = True
    elif cfg.device == "cuda":
        cfg.num_workers = os.cpu_count()//2
        cfg.pin_memory = True
        cfg.prefetch_factor = 4
        cfg.persistent_workers = True
    else:
        cfg.num_workers = 0
        cfg.pin_memory = False
        cfg.prefetch_factor = None
        cfg.persistent_workers = False
    return cfg

def artifact_model_dirs(cfg: Any) -> dict[str, ]:
    """
	Build output directories for model artifacts.

	Uses `cfg.output_root` if present, otherwise defaults to "artifacts", and
	returns paths for baseline and advanced model directories.

	Args:
		cfg: Config-like object that may define `output_root`.

	Returns:
		Mapping with keys {"baseline", "advanced"} pointing to `Path` locations.
	"""
    output_root = Path(getattr(cfg, "output_root", Path("artifacts")))
    return {
        "baseline": output_root / "baseline_cnn",
        "advanced": output_root / "advanced_cnn",
    }

def require_best_checkpoint(model_name: str, model_dir: Path) -> Path:
    """
	Ensure the best-checkpoint file exists and return its path.

	Looks for `best.pth` inside `model_dir` and raises if missing.

	Args:
		model_name: Model identifier used for error context.
		model_dir: Directory expected to contain the checkpoint.

	Returns:
		Path to `model_dir / "best.pth"`.

	Raises:
		FileNotFoundError: If the checkpoint does not exist.
	"""
    ckpt_path = model_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[{model_name}] required checkpoint missing: {ckpt_path}")
    return ckpt_path

def load_optional_metrics(model_name: str, model_dir: Path) -> dict[str, Any] | None:
    """
	Load `metrics.json` from a model directory if present.

	Reads and returns the JSON payload only if it is a dict. If the file is
	missing or cannot be read/parsed, prints a warning and returns None.

	Args:
		model_name: Model identifier used for log context.
		model_dir: Directory that may contain `metrics.json`.

	Returns:
		A dict of metrics if available and valid, otherwise None.
	"""
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"[warn] [{model_name}] metrics.json not found at {metrics_path}")
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            return payload
    except Exception as err:  # pragma: no cover - defensive
        print(f"[warn] [{model_name}] failed reading metrics.json: {err}")
    return None

def prepare_latest_output_dir(output_root: Path) -> Path:
    """
	Recreate and return the "latest" output directory under `output_root`.

	Deletes `output_root/model_comparison/latest` if it exists, then creates it.

	Args:
		output_root: Root directory for outputs.

	Returns:
		Path to the recreated `output_root / "model_comparison" / "latest"` directory.
	"""
    latest_dir = output_root / "model_comparison" / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    return latest_dir

def load_optional_history(model_name: str, model_dir: Path) -> list[dict[str, Any]] | None:
    """
	Load `history.json` from a model directory if present and non-empty.

	Expects the JSON payload to be a non-empty list (e.g., per-epoch dicts).
	If missing, empty, invalid, or unreadable, prints a warning and returns None.

	Args:
		model_name: Model identifier used for log context.
		model_dir: Directory that may contain `history.json`.

	Returns:
		A non-empty list of history records if available, otherwise None.
	"""
    history_path = model_dir / "history.json"
    if not history_path.exists():
        print(f"[warn] [{model_name}] history.json not found at {history_path}. Skipping training curve comparison.")
        return None
    try:
        with history_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, list) and payload:
            return payload
        print(f"[warn] [{model_name}] history.json is empty/invalid. Skipping training curve comparison.")
    except Exception as err:  # pragma: no cover - defensive
        print(f"[warn] [{model_name}] failed reading history.json: {err}. Skipping training curve comparison.")
    return None

def write_summary_json(
    output_path: Path,
    device: torch.device,
    loaders: dict[str, Any],
    baseline_result: dict[str, Any],
    advanced_result: dict[str, Any],
    baseline_metrics_json: dict[str, Any] | None,
    advanced_metrics_json: dict[str, Any] | None,
    deltas: dict[str, Any],
) -> None:
    """
	Write a consolidated model-comparison summary JSON.

	Serializes device info, dataset normalization metadata, baseline/advanced
	checkpoint and split metrics, optional best-epoch values from metrics.json,
	and computed deltas to `output_path`.

	Args:
		output_path: Destination path for the JSON file.
		device: Torch device used for evaluation/comparison.
		loaders: Loader metadata dict (expects class_names, normalization_mean/std).
		baseline_result: Baseline evaluation result dict (expects checkpoint_path,
			best_epoch_in_ckpt, split_metrics).
		advanced_result: Advanced evaluation result dict (expects checkpoint_path,
			best_epoch_in_ckpt, split_metrics).
		baseline_metrics_json: Optional parsed metrics.json dict for baseline.
		advanced_metrics_json: Optional parsed metrics.json dict for advanced.
		deltas: Dict of computed differences between baseline and advanced metrics.

	Returns:
		None
	"""
    payload = {
        "device": str(device),
        "num_classes": len(loaders["class_names"]),
        "normalization_mean": list(loaders["normalization_mean"]),
        "normalization_std": list(loaders["normalization_std"]),
        "baseline": {
            "checkpoint": baseline_result["checkpoint_path"],
            "best_epoch_in_ckpt": baseline_result["best_epoch_in_ckpt"],
            "best_epoch_in_metrics_json": baseline_metrics_json.get("best_epoch") if baseline_metrics_json else None,
            "split_metrics": baseline_result["split_metrics"],
        },
        "advanced": {
            "checkpoint": advanced_result["checkpoint_path"],
            "best_epoch_in_ckpt": advanced_result["best_epoch_in_ckpt"],
            "best_epoch_in_metrics_json": advanced_metrics_json.get("best_epoch") if advanced_metrics_json else None,
            "split_metrics": advanced_result["split_metrics"],
        },
        "deltas": deltas,
    }
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

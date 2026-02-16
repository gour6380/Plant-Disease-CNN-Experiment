from pathlib import Path

class Config:
    """
	Configuration container for PlantVillage CNN training and evaluation.

	Defines dataset paths, image size/class settings, training hyperparameters,
	runtime/DataLoader options, and artifact output locations used across the
	pipeline.
	"""
    # Data paths
    data_dir: Path = Path("data")
    dataset_dir: Path = data_dir / "plantvillage dataset"
    zip_file: Path = data_dir / "plantvillage-dataset.zip"
    images_folder: Path = dataset_dir / "color"
    split_folder: Path = data_dir / "output"
    train_folder: Path = split_folder / "train"
    val_folder: Path = split_folder / "val"
    test_folder: Path = split_folder / "test"

    # Dataset settings
    image_width: int = 256
    image_height: int = 256
    num_classes: int = 38
    class_names_cache_file: Path = split_folder / "class_names.json"
    normalization_stats_file: Path = split_folder / "normalization_stats.json"
    recompute_normalization_stats: bool = False
    allowed_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Training settings
    model_name: str = "baseline" # baseline | advanced
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    advanced_dropout: float = 0.3

    # Runtime settings
    device: str = "auto" # auto | mps | cpu | cuda
    num_workers: int = 4 # auto 0 for MPS and CPU
    pin_memory: bool = True
    use_weighted_sampler: bool = True
    prefetch_factor: int | None = 2 # auto 0 for MPS and CPU
    persistent_workers: bool = True # auto 0 for MPS and CPU

    # Output settings
    output_root: Path = Path("artifacts")
    # If None, trainer auto-selects: artifacts/{model_name}_cnn
    output_dir: Path | None = None
    best_ckpt_name: str = "best.pth"
    last_ckpt_name: str = "last.pth"
    history_file: str = "history.json"
    metrics_file: str = "metrics.json"

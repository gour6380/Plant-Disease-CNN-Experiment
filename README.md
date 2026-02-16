# Plant Disease CNN Experiment

A controlled deep learning experiment comparing a **Baseline CNN** vs. an **Advanced Residual CNN** for plant leaf disease classification on the [PlantVillage dataset](https://arxiv.org/abs/1511.08060) (38 classes, 54K images). Built with PyTorch and Apple MPS acceleration.

## Key Results

| Model | Test Accuracy | Test Loss | Best Epoch |
|-------|:------------:|:---------:|:----------:|
| Baseline CNN | 43.12% | 1.7000 | 27 |
| **Advanced CNN** | **99.43%** | **0.0216** | 29 |

- **Accuracy gain:** +56.31 percentage points (+130.59% relative)
- **Controlled variable:** Architecture only — all hyperparameters held constant

## Project Structure

```
.
├── config.py                   # Centralized configuration (paths, hyperparams, device)
├── dataset.py                  # PlantVillageDataset class, DataLoader builders
├── models.py                   # BaselineCNN & AdvancedCNN (ResidualGNBlock) architectures
├── train.py                    # Training loop, checkpointing, history logging
├── train_baseline.py           # Entry point for baseline model training
├── train_advanced.py           # Entry point for advanced model training
├── evaluate_model.py           # Evaluation pipeline, confusion matrices, metrics
├── transform.py                # Data augmentation & normalization transforms
├── utils.py                    # Helpers: device selection, class mapping, visualization
├── deep_learning.ipynb         # Main experiment notebook (documented with markdown)
├── requirements.txt            # Python dependencies
├── Deep_Learning_Systems_Analysis_Report.pdf  # Full analysis report (PDF with figures)
│
├── artifacts/
│   ├── baseline_cnn/
│   │   ├── best.pth            # Best model checkpoint (by val accuracy)
│   │   ├── class_to_idx.json   # Class name → index mapping
│   │   ├── config_snapshot.json # Training configuration snapshot
│   │   ├── history.json        # Epoch-by-epoch training history
│   │   └── metrics.json        # Final evaluation metrics
│   ├── advanced_cnn/
│   │   ├── best.pth
│   │   ├── class_to_idx.json
│   │   ├── config_snapshot.json
│   │   ├── history.json
│   │   └── metrics.json
│   └── model_comparison/
│       └── latest/
│           ├── comparison_summary.json              # Head-to-head comparison results
│           ├── per_class_accuracy_comparison.csv     # Per-class accuracy breakdown
│           ├── per_class_accuracy_comparison.png     # Per-class accuracy chart
│           ├── split_metrics_comparison.png          # Train/Val/Test metrics comparison
│           ├── test_confusion_matrices_comparison.png # Confusion matrix visualization
│           ├── test_head_to_head.png                 # Test accuracy/loss comparison
│           └── training_curves_comparison.png        # Loss/accuracy training curves
│
└── data/                       # (not tracked — see Setup)
    ├── plantvillage dataset/
    │   └── color/              # 54,305 raw images across 38 class folders
    └── output/
        ├── class_names.json    # Ordered class name list
        └── normalization_stats.json  # Channel-wise mean/std from training set
```

## Model Architectures

### Baseline CNN
A shallow 2-block CNN with BatchNorm, MaxPool, and a 3-layer MLP classifier head (262K flattened features → 128 → 64 → 38).

### Advanced CNN
A deeper residual network with 4 stages of `ResidualGNBlock` (GroupNorm + skip connections), progressive downsampling, global average pooling, and a single FC head (256 → 38).

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=30) |
| Batch size | 32 |
| Epochs | 30 |
| Data split | 80% train / 10% val / 10% test (seed=42) |
| Augmentation | RandomHorizontalFlip, RandomRotation(10°), ColorJitter |
| Device | Apple MPS |

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gour6380/Plant-Disease-CNN-Experiment.git
   cd Plant-Disease-CNN-Experiment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   Place the PlantVillage color images in `data/plantvillage dataset/color/` with one subfolder per class.

4. **Run the experiment:**
   ```bash
   # Option A: Run the full notebook
   jupyter notebook deep_learning.ipynb

   # Option B: Train individually
   python train_baseline.py
   python train_advanced.py
   ```

## Report

The full analysis report is available as:
- [PDF](Deep_Learning_Systems_Analysis_Report.pdf)

## Requirements

- Python 3.13
- PyTorch 2.0+ (with MPS support on Apple Silicon)
- See `requirements.txt` for full dependency list

## License

This project was developed as part of the Udacity Project.

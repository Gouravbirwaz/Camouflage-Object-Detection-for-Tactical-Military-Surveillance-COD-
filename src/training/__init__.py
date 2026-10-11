"""
Training, learning rate scheduling, and benchmark suites for DGNet.
"""

from .trainer import (
    train_dgnet,
    print_model_summary,
    WarmupCosineDecay,
)
from .kaggle_benchmark import (
    run_benchmark_training,
    download_dataset as download_kaggle_dataset,
    upload_checkpoints_to_hf,
    download_checkpoints_from_hf,
)

__all__ = [
    "train_dgnet",
    "print_model_summary",
    "WarmupCosineDecay",
    "run_benchmark_training",
    "download_kaggle_dataset",
    "upload_checkpoints_to_hf",
    "download_checkpoints_from_hf",
]

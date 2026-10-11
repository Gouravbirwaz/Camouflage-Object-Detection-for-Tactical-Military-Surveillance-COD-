"""
Utility modules for COD_PROJECT framework.
"""

from .config import load_project_config, get_project_root
from .metrics import (
    CODMetricsAccumulator,
    compute_all_metrics,
    compute_confusion_components,
    pixel_accuracy,
    mean_absolute_error,
    precision_score,
    recall_score,
    specificity_score,
    false_positive_rate,
    false_negative_rate,
    iou_score,
    dice_score,
    f_measure,
)
from .hardware import setup_gpu_environment

# Aliases
compute_mae = mean_absolute_error
compute_iou = iou_score
compute_f_measure = f_measure

__all__ = [
    "load_project_config",
    "get_project_root",
    "CODMetricsAccumulator",
    "compute_all_metrics",
    "compute_confusion_components",
    "pixel_accuracy",
    "mean_absolute_error",
    "compute_mae",
    "precision_score",
    "recall_score",
    "specificity_score",
    "false_positive_rate",
    "false_negative_rate",
    "iou_score",
    "compute_iou",
    "dice_score",
    "f_measure",
    "compute_f_measure",
    "setup_gpu_environment",
]

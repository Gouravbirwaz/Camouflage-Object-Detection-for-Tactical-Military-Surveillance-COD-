"""
Inference, prediction, and evaluation engines for COD models.
"""

from .predictor import (
    load_and_preprocess_image,
    load_ground_truth,
    process_prediction_mask,
    discover_all_models,
    find_default_sample_images,
    find_ground_truth_for_image,
    ModelRunner,
    run_inference_pipeline,
)
from .evaluator import evaluate_dgnet

__all__ = [
    "load_and_preprocess_image",
    "load_ground_truth",
    "process_prediction_mask",
    "discover_all_models",
    "find_default_sample_images",
    "find_ground_truth_for_image",
    "ModelRunner",
    "run_inference_pipeline",
    "evaluate_dgnet",
]

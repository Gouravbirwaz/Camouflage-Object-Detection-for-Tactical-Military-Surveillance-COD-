#!/usr/bin/env python3
"""
DGNet Kaggle Camouflage Benchmark Training & Remote Hugging Face Execution Script
CLI Entrypoint delegating to src.training.kaggle_benchmark.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.kaggle_benchmark import (
    KAGGLE_DATASETS,
    ensure_kagglehub,
    download_dataset,
    discover_image_mask_pairs,
    get_dataset_splits,
    run_benchmark_training,
    upload_checkpoints_to_hf,
    download_checkpoints_from_hf,
    main,
)
from src.integration.hf_job_launcher import generate_benchmark_job_command as generate_hf_job_command
from src.training.trainer import WarmupCosineDecay
from src.data.dataset import (
    compute_object_gradient_np,
    parse_sample,
    _random_resized_crop,
    _cutout_tf,
    create_tf_dataset,
)
from src.optimization.tflite_converter import export_tflite as export_tflite_model

if __name__ == "__main__":
    main()

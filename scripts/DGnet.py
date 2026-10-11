#!/usr/bin/env python3
"""
DGNet Execution & Training CLI Script - Realtime Camouflaged Target Detection
Pure TensorFlow 2.x / Keras Implementation for Tactical Military Surveillance.

This script serves as the primary CLI entrypoint, delegating modular functionality
to the unified 'src' package:
- Model Architecture: src.models.dgnet
- Training Loop:      src.training.trainer
- Evaluation Loop:    src.inference.evaluator
- TFLite Export:      src.optimization.tflite_converter
- Dataset Pipelines:  src.data.dataset

Usage:
    python scripts/DGnet.py --train --epochs 10 --batch-size 8
    python scripts/DGnet.py --train --resume
    python scripts/DGnet.py --evaluate
    python scripts/DGnet.py --summary --backbone mobilenet_v3_large
    python scripts/DGnet.py --export-tflite --output models/quantized/dgnet_mobilenet_v3.tflite
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
# Prevent threadpool contention and deadlock between OpenCV and TensorFlow tf.data workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Add project root directory to python path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Re-export and import modular components from src for 100% backward compatibility
from src.utils.hardware import setup_gpu_environment
from src.utils.config import load_project_config
from src.utils.metrics import CODMetricsAccumulator
from src.tracking.mlflow_utils import MLflowTracker
from src.models.dgnet import DGNet, DGNetLoss, build_dgnet_model
from src.data.dataset import (
    _find_dataset_splits_in_dir,
    discover_dataset_pairs,
    compute_object_gradient_np,
    parse_sample,
    _random_resized_crop,
    _gaussian_blur_tf,
    _cutout_tf,
    create_tf_dataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.training.trainer import (
    train_dgnet,
    print_model_summary,
    WarmupCosineDecay,
)
from src.inference.evaluator import evaluate_dgnet
from src.optimization.tflite_converter import export_tflite


def load_config(config_path: str = None) -> dict:
    """Loads configuration parameters dynamically from configs/ directory via central manager."""
    return load_project_config(config_path)


def main():
    cfg = load_project_config()
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    # Priority: Env Vars > Config Params > Default relative paths
    env_data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw/dataset-splitM"))
    env_augmented_dir = os.getenv("AUGMENTED_DIR", "data/augmented/military_tactical")
    env_ckpt_dir = os.getenv("CHECKPOINT_DIR", train_cfg.get("checkpoint_dir", "models/raw_checkpoints"))
    env_log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))
    env_resume = train_cfg.get("resume", os.getenv("RESUME", "false").lower() in ("true", "1", "t", "yes"))

    default_backbone = model_cfg.get("context_backbone", "mobilenet_v3_large")
    default_kernel = model_cfg.get("gradient_kernel", "sobel")
    default_img_size = tuple(data_cfg.get("image_size", [384, 384]))
    default_epochs = train_cfg.get("epochs", 50)
    default_batch_size = train_cfg.get("batch_size", data_cfg.get("batch_size", 8))
    default_lr = train_cfg.get("learning_rate", 0.0001)
    default_tflite_path = train_cfg.get("tflite_export_path", "models/quantized/dgnet_s_int8.tflite")
    default_version = os.getenv("MODEL_VERSION", train_cfg.get("model_version", model_cfg.get("version", "v1.0")))

    parser = argparse.ArgumentParser(description="DGNet: Deep Gradient Network Real Model Training & Execution (TensorFlow 2.x)")
    parser.add_argument("--train", action="store_true", help="Execute real model training on dataset")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model on test dataset split")
    parser.add_argument("--resume", nargs="?", const=True, default=env_resume, help="Resume training from latest checkpoint, or specify path to weights file")
    parser.add_argument("--version", "--model-version", type=str, default=default_version, help="Model version tag for checkpointing (e.g. v1.0)")
    parser.add_argument("--backbone", type=str, default=default_backbone, help="Backbone architecture")
    parser.add_argument("--kernel", type=str, default=default_kernel, help="Gradient kernel type")
    parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=default_lr, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=env_data_dir, help="Path to raw data directory (or via DATA_DIR env var)")
    parser.add_argument("--augmented-dir", type=str, default=env_augmented_dir, help="Path to augmented dataset directory (or via AUGMENTED_DIR env var)")
    parser.add_argument("--checkpoint-dir", type=str, default=env_ckpt_dir, help="Path to checkpoint directory (or via CHECKPOINT_DIR env var)")
    parser.add_argument("--log-dir", type=str, default=env_log_dir, help="Path to metrics/log directory (or via LOG_DIR env var)")
    parser.add_argument("--weights", type=str, default=None, help="Path to specific model weights checkpoint file for training or evaluation")
    parser.add_argument("--pred-dir", type=str, default="experiments/predictions", help="Directory to save test prediction and comparison masks")
    parser.add_argument("--model-tag", "--model-name", type=str, default=None, help="Custom model tag/name for evaluation output filenames")
    parser.add_argument("--no-save-preds", action="store_true", help="Disable saving prediction mask images during evaluation")
    parser.add_argument("--summary", action="store_true", help="Print model summary and parameter stats")
    parser.add_argument("--export-tflite", action="store_true", help="Export model to TensorFlow Lite format")
    parser.add_argument("--output", type=str, default=None, help="Custom output filepath for export/checkpoint")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name (or via MLFLOW_EXPERIMENT_NAME env var)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking for this run")

    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.summary, args.export_tflite]):
        args.train = True

    if args.train:
        train_dgnet(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_dir=args.data_dir,
            augmented_dir=args.augmented_dir,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            backbone=args.backbone,
            kernel=args.kernel,
            resume=args.resume,
            weights=args.weights,
            model_version=args.version,
            experiment_name=args.experiment_name,
            enable_mlflow=not args.no_mlflow
        )

    if args.evaluate:
        evaluate_dgnet(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            weights_path=args.weights,
            pred_dir=args.pred_dir,
            log_dir=args.log_dir,
            backbone=args.backbone,
            kernel=args.kernel,
            model_version=args.version,
            model_tag=args.model_tag,
            save_predictions=not args.no_save_preds
        )

    if args.summary:
        model = build_dgnet_model(
            backbone_name=args.backbone,
            input_shape=(default_img_size[0], default_img_size[1], 3),
            gradient_kernel=args.kernel,
            pretrained=False
        )
        print_model_summary(model, image_size=default_img_size)

    if args.export_tflite:
        model = build_dgnet_model(
            backbone_name=args.backbone,
            input_shape=(default_img_size[0], default_img_size[1], 3),
            gradient_kernel=args.kernel,
            pretrained=False
        )
        out_path = args.output if args.output else default_tflite_path
        export_tflite(model, output_path=out_path, image_size=default_img_size)


if __name__ == "__main__":
    main()

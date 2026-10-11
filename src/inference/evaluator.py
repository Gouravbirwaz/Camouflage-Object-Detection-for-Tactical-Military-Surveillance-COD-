"""
Comprehensive Evaluation Engine for DGNet.
Evaluates trained models on testing dataset splits and computes academic COD metrics:
MAE, S-measure, E-measure, Weighted F-measure, and IoU.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import tensorflow as tf

from src.utils.config import get_project_root, load_project_config
from src.utils.hardware import setup_gpu_environment
from src.utils.metrics import CODMetricsAccumulator
from src.models.dgnet import DGNet, build_dgnet_model
from src.data.dataset import discover_dataset_pairs, create_tf_dataset

PROJECT_ROOT = get_project_root()


def evaluate_dgnet(
    data_dir: str = "data/raw/dataset-splitM",
    checkpoint_dir: str = "models/raw_checkpoints",
    weights_path: Optional[str] = None,
    pred_dir: str = "experiments/predictions",
    log_dir: str = "metrics",
    backbone: str = "mobilenet_v3_large",
    kernel: str = "sobel",
    model_version: Optional[str] = None,
    model_tag: Optional[str] = None,
    save_predictions: bool = True
) -> Dict:
    """Evaluates the trained model on test dataset split and saves clean GT, prediction, and comparison masks."""
    setup_gpu_environment()

    params = load_project_config("params.yaml")
    img_size = tuple(params.get("dataset", {}).get("image_size", [384, 384]))

    cfg = load_project_config()
    version_str = model_version or cfg.get("train", {}).get("model_version") or cfg.get("model", {}).get("version", "v1.0")
    version_tag = version_str if str(version_str).startswith("v") else f"v{version_str}"

    data_path = Path(data_dir) if Path(data_dir).is_absolute() else PROJECT_ROOT / data_dir
    ckpt_dir = Path(checkpoint_dir) if Path(checkpoint_dir).is_absolute() else PROJECT_ROOT / checkpoint_dir
    metrics_log_dir = Path(log_dir) if Path(log_dir).is_absolute() else PROJECT_ROOT / log_dir
    metrics_log_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(pred_dir) if Path(pred_dir).is_absolute() else PROJECT_ROOT / pred_dir
    if save_predictions:
        pred_path.mkdir(parents=True, exist_ok=True)

    splits = discover_dataset_pairs(data_path)
    if not splits:
        raise FileNotFoundError(f"No dataset splits found in: {data_path}")

    test_key = "Testing" if "Testing" in splits else list(splits.keys())[0]
    test_imgs, test_masks = splits[test_key]

    print(f"\n[Evaluation] Evaluating DGNet ({version_tag}) on split '{test_key}' ({len(test_imgs)} samples)...")
    test_ds = create_tf_dataset(test_imgs, test_masks, image_size=img_size, batch_size=4, is_train=False)

    model = build_dgnet_model(backbone_name=backbone, input_shape=(img_size[0], img_size[1], 3), gradient_kernel=kernel, pretrained=False)

    # Multi-candidate search for weights
    found_weights = None
    if weights_path:
        wp = Path(weights_path) if Path(weights_path).is_absolute() else PROJECT_ROOT / weights_path
        if wp.exists() and wp.is_file() and wp.stat().st_size > 0:
            found_weights = wp
        else:
            print(f"[Evaluation] Specified weights file not found: {wp}")

    if not found_weights:
        candidate_dirs = [
            ckpt_dir,
            Path("/data/models/raw_checkpoints"),
            Path("/app/models/raw_checkpoints"),
            PROJECT_ROOT / "models/raw_checkpoints"
        ]
        w_candidates = [
            f"dgnet_{version_tag}_best.weights.h5",
            f"dgnet_{version_tag}_latest.weights.h5",
            f"dgnet_{version_tag}_final.weights.h5",
            "dgnet_best.weights.h5",
            "dgnet_latest.weights.h5",
            "dgnet_final.weights.h5"
        ]
        for cdir in candidate_dirs:
            try:
                if not cdir.exists():
                    continue
                for w_name in w_candidates:
                    wp = cdir / w_name
                    if wp.exists() and wp.stat().st_size > 0:
                        found_weights = wp
                        break
                if found_weights:
                    break
                for wp in sorted(cdir.glob("*.weights.h5"), reverse=True):
                    if wp.is_file() and wp.stat().st_size > 0:
                        found_weights = wp
                        break
                if found_weights:
                    break
            except Exception:
                continue

    if model_tag:
        effective_model_tag = model_tag
    elif found_weights:
        effective_model_tag = found_weights.name.replace(".weights.h5", "").replace(".keras", "").replace(".h5", "")
    else:
        effective_model_tag = f"dgnet_{version_tag}_{backbone}"

    if found_weights:
        _ = model(tf.zeros((1, img_size[0], img_size[1], 3)), training=False)
        model.load_weights(str(found_weights))
        print(f"[Evaluation] Loaded weights from: {found_weights.resolve()}")
    else:
        print(f"[Evaluation] Warning: No trained weights checkpoint found. Evaluating with initialized model.")

    print(f"[Evaluation] Model Identifier for Output: {effective_model_tag}")
    if save_predictions:
        print(f"[Evaluation] Saving clean GT, prediction masks, and B&W comparisons to: {pred_path.resolve()}")

    metrics_acc = CODMetricsAccumulator(threshold=0.5, beta_sq=0.3)
    start_time = time.perf_counter()

    sample_idx = 0
    saved_count = 0

    for x_batch, y_mask, _ in test_ds:
        pred_mask = model(x_batch, training=False)
        metrics_acc.update(y_mask, pred_mask)

        if save_predictions:
            batch_sz = int(tf.shape(x_batch)[0])
            y_np = y_mask.numpy()
            pred_np = pred_mask.numpy()

            for b in range(batch_sz):
                if sample_idx + b < len(test_imgs):
                    stem = Path(test_imgs[sample_idx + b]).stem

                    gt_bin = (y_np[b, :, :, 0] >= 0.5).astype(np.uint8) * 255
                    pred_bin = (pred_np[b, :, :, 0] >= 0.5).astype(np.uint8) * 255

                    gt_out = pred_path / f"{stem}_gt.png"
                    if not gt_out.exists():
                        cv2.imwrite(str(gt_out), gt_bin)

                    pred_out = pred_path / f"{stem}_{effective_model_tag}_pred.png"
                    cv2.imwrite(str(pred_out), pred_bin)

                    h, w = gt_bin.shape[:2]
                    separator = np.full((h, 4), 128, dtype=np.uint8)
                    body = np.hstack([gt_bin, separator, pred_bin])

                    header_h = 36
                    header = np.zeros((header_h, body.shape[1]), dtype=np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(header, "GROUND TRUTH", (15, 24), font, 0.5, 255, 1, cv2.LINE_AA)
                    label_text = f"MODEL: {effective_model_tag}"
                    right_x = w + 4 + 15
                    cv2.putText(header, label_text, (right_x, 24), font, 0.5, 255, 1, cv2.LINE_AA)

                    h_div = np.full((2, body.shape[1]), 128, dtype=np.uint8)
                    comp_img = np.vstack([header, h_div, body])

                    comp_out = pred_path / f"{stem}_{effective_model_tag}_comparison.png"
                    cv2.imwrite(str(comp_out), comp_img)
                    saved_count += 1

            sample_idx += batch_sz

    total_time = time.perf_counter() - start_time
    eval_metrics = metrics_acc.result()
    fps = eval_metrics["sample_count"] / total_time if total_time > 0 else 0.0

    eval_results = {
        "dataset_split": test_key,
        "sample_count": eval_metrics["sample_count"],
        "pixel_accuracy": eval_metrics["pixel_accuracy"],
        "mean_absolute_error_mae": eval_metrics["mae"],
        "iou": eval_metrics["iou"],
        "dice": eval_metrics["dice"],
        "f_measure": eval_metrics["f_measure"],
        "precision": eval_metrics["precision"],
        "recall": eval_metrics["recall"],
        "specificity": eval_metrics["specificity"],
        "false_positive_rate": eval_metrics["fpr"],
        "false_negative_rate": eval_metrics["fnr"],
        "confusion_matrix": {
            "true_positives": eval_metrics["tp_total"],
            "true_negatives": eval_metrics["tn_total"],
            "false_positives": eval_metrics["fp_total"],
            "false_negatives": eval_metrics["fn_total"],
        },
        "evaluation_fps": round(fps, 1),
        "evaluation_time_seconds": round(total_time, 2),
    }

    eval_log_path = metrics_log_dir / "evaluation_results.json"
    with open(eval_log_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    print("\n" + "=" * 70)
    print("            DGNet Comprehensive Evaluation Report")
    print("=" * 70)
    print(f"  Test Split:             {test_key} ({eval_metrics['sample_count']} samples)")
    print(f"  Model Version:          {version_tag}")
    print("-" * 70)
    print(f"  Pixel Accuracy:         {eval_metrics['pixel_accuracy']*100:.2f}%")
    print(f"  Mean Absolute Error:    {eval_metrics['mae']:.4f}")
    print(f"  IoU (Jaccard):          {eval_metrics['iou']:.4f}")
    print(f"  Dice Coefficient:       {eval_metrics['dice']:.4f}")
    print(f"  F-measure (Fβ=0.3):     {eval_metrics['f_measure']:.4f}")
    print("-" * 70)
    print(f"  Precision:              {eval_metrics['precision']:.4f}")
    print(f"  Recall:                 {eval_metrics['recall']:.4f}")
    print(f"  Specificity:            {eval_metrics['specificity']:.4f}")
    print(f"  False Positive Rate:    {eval_metrics['fpr']:.4f}")
    print(f"  False Negative Rate:    {eval_metrics['fnr']:.4f}")
    print("-" * 70)
    print(f"  Confusion Matrix:")
    print(f"    TP: {eval_metrics['tp_total']:.0f}  |  FP: {eval_metrics['fp_total']:.0f}")
    print(f"    FN: {eval_metrics['fn_total']:.0f}  |  TN: {eval_metrics['tn_total']:.0f}")
    print("-" * 70)
    print(f"  Realtime Speed:         {fps:.1f} FPS")
    print(f"  Saved Evaluation JSON:  {eval_log_path.resolve()}")
    print("=" * 70 + "\n")

    return eval_results

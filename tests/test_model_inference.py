#!/usr/bin/env python3
"""
DGNet Single Image Testing Script
Loads a trained DGNet checkpoint, runs inference on a sample RGB image,
and saves the predicted camouflage mask + a side-by-side comparison.

Usage:
    python tests/test_model_inference.py
    python tests/test_model_inference.py --image data/raw/dataset-splitM/Testing/Images/image501.jpg
    python tests/test_model_inference.py --image path/to/any/image.jpg --checkpoint models/raw_checkpoints/dgnet_best.weights.h5
    python tests/test_model_inference.py --threshold 0.4 --output-dir tests/output
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dgnet import build_dgnet_model
from src.utils.config import load_project_config
from src.utils.metrics import compute_all_metrics


def load_and_preprocess_image(image_path: str, image_size: tuple = (384, 384)) -> tuple:
    """
    Loads an RGB image, resizes it, and normalizes to [0, 1].

    Returns:
        (input_tensor, original_image)
        - input_tensor: [1, H, W, 3] float32 tensor normalized to [0, 1]
        - original_image: original-resolution uint8 numpy array for visualization
    """
    raw_bytes = tf.io.read_file(image_path)
    original = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    original_np = original.numpy()

    resized = tf.image.resize(original, image_size)
    normalized = tf.cast(resized, tf.float32) / 255.0
    input_tensor = tf.expand_dims(normalized, axis=0)  # [1, H, W, 3]

    return input_tensor, original_np


def load_ground_truth(gt_path: str, image_size: tuple = (384, 384)):
    """Loads ground truth mask if available. Returns None if not found."""
    if not os.path.exists(gt_path):
        return None
    raw_bytes = tf.io.read_file(gt_path)
    mask = tf.image.decode_image(raw_bytes, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, image_size, method="nearest")
    mask = tf.cast(mask >= 128, tf.float32)
    return mask


def save_mask_png(mask_array: np.ndarray, output_path: str):
    """Saves a [H, W] or [H, W, 1] float mask as a grayscale PNG (0-255)."""
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    encoded = tf.image.encode_png(tf.expand_dims(mask_uint8, -1))
    tf.io.write_file(output_path, encoded)


def save_comparison_image(original: np.ndarray, pred_mask: np.ndarray, output_path: str,
                          gt_mask: np.ndarray = None, image_size: tuple = (384, 384)):
    """
    Creates and saves a side-by-side comparison image:
    [Original | Predicted Mask | Overlay] or [Original | GT Mask | Predicted Mask | Overlay] if GT exists.
    """
    # Resize original to match mask size
    orig_resized = tf.image.resize(original, image_size).numpy().astype(np.uint8)

    # Predicted mask as 3-channel
    if pred_mask.ndim == 2:
        pred_mask = pred_mask[:, :, np.newaxis]
    pred_vis = np.repeat((pred_mask * 255).astype(np.uint8), 3, axis=-1)

    # Overlay: red tint on predicted regions
    overlay = orig_resized.copy().astype(np.float32)
    mask_bool = pred_mask[:, :, 0] > 0.5
    overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 128, 0, 255)  # Red channel boost
    overlay[mask_bool, 1] = overlay[mask_bool, 1] * 0.5  # Dim green
    overlay[mask_bool, 2] = overlay[mask_bool, 2] * 0.5  # Dim blue
    overlay = overlay.astype(np.uint8)

    panels = [orig_resized]

    if gt_mask is not None:
        if gt_mask.ndim == 2:
            gt_mask = gt_mask[:, :, np.newaxis]
        gt_vis = np.repeat((gt_mask * 255).astype(np.uint8), 3, axis=-1)
        panels.append(gt_vis)

    panels.extend([pred_vis, overlay])

    comparison = np.concatenate(panels, axis=1)
    encoded = tf.image.encode_png(comparison)
    tf.io.write_file(output_path, encoded)


def find_default_sample_image() -> str:
    """Finds a sample test image from the dataset."""
    candidates = [
        PROJECT_ROOT / "data/raw/dataset-splitM/Testing/Images/image501.jpg",
        PROJECT_ROOT / "data/raw/dataset-splitM/Testing/Images/image1000.jpg",
    ]
    # Try specific files first
    for c in candidates:
        if c.exists():
            return str(c)

    # Scan for any image in Testing/Images
    test_img_dir = PROJECT_ROOT / "data/raw/dataset-splitM/Testing/Images"
    if test_img_dir.exists():
        for f in sorted(test_img_dir.glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                return str(f)

    # Scan Training/Images as fallback
    train_img_dir = PROJECT_ROOT / "data/raw/dataset-splitM/Training/Images"
    if train_img_dir.exists():
        for f in sorted(train_img_dir.glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                return str(f)

    return None


def find_ground_truth_for_image(image_path: str) -> str:
    """Attempts to find the matching ground truth mask for a given image."""
    img_path = Path(image_path)
    stem = img_path.stem

    # Check sibling GT folder
    gt_candidates = [
        img_path.parent.parent / "GT" / f"{stem}.png",
        img_path.parent.parent / "Masks" / f"{stem}.png",
        img_path.parent.parent / "GT" / f"{stem}.jpg",
    ]
    for gt in gt_candidates:
        if gt.exists():
            return str(gt)
    return None


def find_best_checkpoint() -> str:
    """Finds the best available model checkpoint."""
    candidates = [
        # PROJECT_ROOT / "models/raw_checkpoints/dgnet_best.weights.h5",
        PROJECT_ROOT / "models/raw_checkpoints/dgnet_v1.0_latest.weights.h5"
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return str(c)

    # Fallback: any .weights.h5 file
    ckpt_dir = PROJECT_ROOT / "models/raw_checkpoints"
    if ckpt_dir.exists():
        for f in sorted(ckpt_dir.glob("*.weights.h5"), reverse=True):
            if f.stat().st_size > 0:
                return str(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="DGNet Single Image Inference Test")
    parser.add_argument("--image", type=str, default=None, help="Path to input RGB image (auto-detected if not set)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights (auto-detected if not set)")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone architecture (auto-detected from config)")
    parser.add_argument("--kernel", type=str, default=None, help="Gradient kernel type (auto-detected from config)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for mask (default: 0.5)")
    parser.add_argument("--invert", action="store_true", help="Invert predicted mask (1 - pred) so target is white and background is black")
    parser.add_argument("--auto-invert", action="store_true", default=False, help="Auto-detect and flip if background is detected as target (default: False)")
    parser.add_argument("--no-auto-invert", dest="auto_invert", action="store_false", help="Disable auto-invert")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for masks (default: tests/output)")
    parser.add_argument("--no-comparison", action="store_true", help="Skip generating comparison image")
    args = parser.parse_args()

    # ===== Load Config =====
    cfg = load_project_config()
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    img_size = tuple(data_cfg.get("image_size", [384, 384]))
    backbone = args.backbone or model_cfg.get("context_backbone", "mobilenet_v3_large")
    kernel = args.kernel or model_cfg.get("gradient_kernel", "sobel")

    # ===== Resolve Paths =====
    image_path = args.image or find_default_sample_image()
    if image_path is None:
        print("[ERROR] No input image found. Provide one with --image <path>")
        sys.exit(1)
    image_path = str(Path(image_path).resolve()) if not Path(image_path).is_absolute() else image_path

    checkpoint_path = args.checkpoint or find_best_checkpoint()
    if checkpoint_path is None:
        print("[ERROR] No model checkpoint found. Provide one with --checkpoint <path>")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "tests" / "unit" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(image_path).stem

    print("\n" + "=" * 70)
    print("         DGNet Single Image Inference Test")
    print("=" * 70)
    print(f"  Input Image:      {image_path}")
    print(f"  Checkpoint:       {checkpoint_path}")
    print(f"  Backbone:         {backbone}")
    print(f"  Gradient Kernel:  {kernel}")
    print(f"  Image Size:       {img_size}")
    print(f"  Threshold:        {args.threshold}")
    print(f"  Output Dir:       {output_dir.resolve()}")
    print("-" * 70)

    # ===== Build & Load Model =====
    print("\n[1/4] Building DGNet model...", flush=True)
    model = build_dgnet_model(
        backbone_name=backbone,
        input_shape=(img_size[0], img_size[1], 3),
        gradient_kernel=kernel,
        pretrained=False
    )

    # Initialize weights with dummy forward pass
    dummy = tf.zeros((1, img_size[0], img_size[1], 3))
    _ = model(dummy, training=False)

    print(f"[2/4] Loading checkpoint: {Path(checkpoint_path).name}...", flush=True)
    model.load_weights(checkpoint_path)
    print("       Checkpoint loaded successfully!", flush=True)

    # ===== Load & Preprocess Image =====
    print(f"[3/4] Loading image: {Path(image_path).name}...", flush=True)
    input_tensor, original_image = load_and_preprocess_image(image_path, image_size=img_size)

    # ===== Run Inference =====
    print("[4/4] Running inference...", flush=True)
    start_time = time.perf_counter()
    pred_prob = model(input_tensor, training=False)  # [1, H, W, 1] sigmoid output
    inference_time = time.perf_counter() - start_time

    pred_prob_np = pred_prob.numpy()[0]  # [H, W, 1]

    # Inversion handling (manual flag or auto-detection when background > 50%)
    initial_coverage = np.mean(pred_prob_np >= args.threshold)
    should_invert = args.invert or (args.auto_invert and initial_coverage > 0.5)
    if should_invert:
        reason = "CLI flag --invert" if args.invert else f"Auto-detected reversed polarity (initial coverage {initial_coverage*100:.1f}% > 50%)"
        print(f"  -> Inversion Applied: {reason}")
        print("     Flipping mask: Target set to WHITE (1) and Background set to BLACK (0).")
        pred_prob_np = 1.0 - pred_prob_np
        pred_prob = 1.0 - pred_prob

    pred_binary = (pred_prob_np >= args.threshold).astype(np.float32)

    # ===== Save Outputs =====
    # 1. Raw probability mask
    prob_mask_path = str(output_dir / f"{image_stem}_pred_prob.png")
    save_mask_png(pred_prob_np, prob_mask_path)

    # 2. Binary mask
    binary_mask_path = str(output_dir / f"{image_stem}_pred_mask.png")
    save_mask_png(pred_binary, binary_mask_path)

    print(f"\n  Saved probability mask:  {prob_mask_path}")
    print(f"  Saved binary mask:       {binary_mask_path}")

    # ===== Metrics Against GT (if available) =====
    gt_path = find_ground_truth_for_image(image_path)
    gt_mask = None
    if gt_path:
        gt_mask = load_ground_truth(gt_path, image_size=img_size)
        if gt_mask is not None:
            gt_mask_np = gt_mask.numpy()
            metrics = compute_all_metrics(
                tf.expand_dims(gt_mask, 0),
                pred_prob,
                threshold=args.threshold
            )
            print(f"\n  Ground Truth Found:      {gt_path}")
            print("-" * 70)
            print(f"  Pixel Accuracy:          {metrics['pixel_accuracy']*100:.2f}%")
            print(f"  IoU:                     {metrics['iou']:.4f}")
            print(f"  Dice:                    {metrics['dice']:.4f}")
            print(f"  F-measure:               {metrics['f_measure']:.4f}")
            print(f"  MAE:                     {metrics['mae']:.4f}")
            print(f"  Precision:               {metrics['precision']:.4f}")
            print(f"  Recall:                  {metrics['recall']:.4f}")
            print(f"  FPR:                     {metrics['fpr']:.4f}")
            print(f"  FNR:                     {metrics['fnr']:.4f}")
    else:
        print(f"\n  Ground Truth:            Not found (skipping metrics)")

    # ===== Save Comparison Image =====
    if not args.no_comparison:
        comparison_path = str(output_dir / f"{image_stem}_comparison.png")
        gt_for_vis = gt_mask.numpy() if gt_mask is not None else None
        save_comparison_image(original_image, pred_binary, comparison_path,
                              gt_mask=gt_for_vis, image_size=img_size)
        print(f"  Saved comparison image:  {comparison_path}")

    # ===== Summary =====
    pred_coverage = np.mean(pred_binary) * 100
    print("-" * 70)
    print(f"  Inference Time:          {inference_time*1000:.1f} ms ({1.0/inference_time:.1f} FPS)")
    print(f"  Predicted Coverage:      {pred_coverage:.2f}% of pixels detected as camouflaged")
    print(f"  Prediction Range:        [{pred_prob_np.min():.4f}, {pred_prob_np.max():.4f}]")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

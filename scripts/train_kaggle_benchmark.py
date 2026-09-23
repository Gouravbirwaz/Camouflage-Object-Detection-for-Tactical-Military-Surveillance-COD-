#!/usr/bin/env python3
"""
DGNet Kaggle Camouflage Benchmark Training & Remote Hugging Face Execution Script
Trains DGNet on public camouflage benchmarks (COD10K, NC4K, CAMO) from Kaggle.
Supports local execution, Hugging Face GPU Jobs, automated checkpoint upload, and local download.

Datasets (from dataset.txt):
1. https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset (COD10K: ~10,000 samples)
2. https://www.kaggle.com/datasets/ivanomelchenkoim11/nc4k-dataset (NC4K: ~4,120 samples)
3. https://www.kaggle.com/datasets/danhnt/camo-fs-dataset (CAMO: ~1,250 samples)

Usage:
    # 1. Train locally on COD10K:
    python scripts/train_kaggle_benchmark.py --train --dataset cod10k --epochs 30

    # 2. Train on all 3 combined benchmark datasets:
    python scripts/train_kaggle_benchmark.py --train --dataset all --epochs 40

    # 3. Launch training on Hugging Face Cloud GPU Job:
    python scripts/train_kaggle_benchmark.py --launch-hf-job --flavor t4-small --epochs 30 --run

    # 4. Download trained checkpoints from Hugging Face back to local machine:
    python scripts/train_kaggle_benchmark.py --download-checkpoints
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np

# Ensure NumPy 1.x compatibility with TensorFlow 2.16
if int(np.__version__.split(".")[0]) >= 2:
    print(f"[Compatibility Fix] Detected NumPy {np.__version__} which is incompatible with TensorFlow 2.16. Downgrading to numpy<2.0.0...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0.0"])
    os.execv(sys.executable, [sys.executable] + sys.argv)

import tensorflow as tf

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dgnet import DGNet, DGNetLoss, build_dgnet_model
from src.utils.config import load_project_config
from src.utils.metrics import CODMetricsAccumulator

# Public Kaggle Datasets mapping from dataset.txt
KAGGLE_DATASETS = {
    "cod10k": "ivanomelchenkoim11/cod10k-dataset",
    "nc4k": "ivanomelchenkoim11/nc4k-dataset",
    "camo": "danhnt/camo-fs-dataset",
}

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ==============================================================================
# 1. Dataset Ingestion & Pair Matching
# ==============================================================================

def ensure_kagglehub():
    """Ensures kagglehub is installed for direct dataset downloads."""
    try:
        import kagglehub
        return kagglehub
    except ImportError:
        print("[Setup] Installing kagglehub for automated dataset downloading...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0.0", "kagglehub"])
        import kagglehub
        return kagglehub


def download_dataset(dataset_key: str) -> Path:
    """Downloads requested Kaggle dataset using kagglehub."""
    if dataset_key not in KAGGLE_DATASETS and dataset_key != "all":
        raise ValueError(f"Unknown dataset key: {dataset_key}. Options: {list(KAGGLE_DATASETS.keys())} or 'all'")

    kh = ensure_kagglehub()
    slug = KAGGLE_DATASETS[dataset_key]
    print(f"\n[Kaggle Ingestion] Downloading '{slug}' via kagglehub...")
    path_str = kh.dataset_download(slug)
    data_path = Path(path_str)
    print(f"✓ Downloaded to: {data_path.resolve()}")
    return data_path


def discover_image_mask_pairs(data_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Recursively scans directory hierarchy to pair RGB camouflage images
    with their corresponding binary Ground Truth masks.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = list(data_dir.rglob("*"))

    img_candidates = {}
    gt_candidates = {}

    for f in all_files:
        if not f.is_file() or f.suffix.lower() not in valid_exts:
            continue
        p_str = str(f).replace("\\", "/")
        stem = f.stem

        # Mask indicators across COD10K, NC4K, and CAMO hierarchies
        is_gt = any(k in p_str for k in [
            "/GT_Object/", "/GT/", "/Masks/", "/masks/", "/gt/", "/SegmentationClass/",
            "_GT_Object", "_GT", "_mask", "_Mask"
        ])

        if is_gt:
            gt_candidates[stem] = str(f)
        else:
            img_candidates[stem] = str(f)

    matched_imgs, matched_masks = [], []
    for stem, ip in img_candidates.items():
        # Match exact stem or stem with GT suffix
        gp = (
            gt_candidates.get(stem) or
            gt_candidates.get(f"{stem}_GT") or
            gt_candidates.get(f"{stem}_mask") or
            gt_candidates.get(f"{stem}_GT_Object")
        )
        if gp:
            matched_imgs.append(ip)
            matched_masks.append(gp)

    return matched_imgs, matched_masks


def get_dataset_splits(dataset_name: str = "cod10k", val_ratio: float = 0.15) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Loads and pairs images from selected Kaggle benchmark dataset(s)."""
    selected_keys = list(KAGGLE_DATASETS.keys()) if dataset_name == "all" else [dataset_name]

    all_imgs, all_masks = [], []
    for key in selected_keys:
        dpath = download_dataset(key)
        imgs, masks = discover_image_mask_pairs(dpath)
        print(f"  • {key.upper()}: Matched {len(imgs):,} image-mask pairs.")
        all_imgs.extend(imgs)
        all_masks.extend(masks)

    if not all_imgs:
        raise RuntimeError(f"Failed to find image-mask pairs for dataset(s): {selected_keys}")

    # Deterministic train/validation split
    combined = list(zip(all_imgs, all_masks))
    random.seed(42)
    random.shuffle(combined)

    split_idx = int((1.0 - val_ratio) * len(combined))
    train_pairs = combined[:split_idx]
    val_pairs = combined[split_idx:]

    train_imgs, train_masks = [p[0] for p in train_pairs], [p[1] for p in train_pairs]
    val_imgs, val_masks = [p[0] for p in val_pairs], [p[1] for p in val_pairs]

    print(f"\n✓ Combined Dataset Loaded: {len(all_imgs):,} total pairs across {selected_keys}")
    print(f"  • Training Split:   {len(train_imgs):,} images")
    print(f"  • Validation Split: {len(val_imgs):,} images")

    return train_imgs, train_masks, val_imgs, val_masks


# ==============================================================================
# 2. High-Throughput `tf.data` Pipeline
# ==============================================================================

def compute_object_gradient_np(image_uint8: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    """Official DGNet Canny object gradient supervision (cv2.Canny * mask)."""
    if image_uint8.ndim == 4:
        image_uint8 = image_uint8[0]
    if mask_uint8.ndim == 3 and mask_uint8.shape[-1] == 1:
        mask_uint8 = mask_uint8[:, :, 0]
    canny = cv2.Canny(image_uint8, 64, 128)
    fore = ((canny > 0) & (mask_uint8 >= 128)).astype(np.float32)
    return fore[:, :, np.newaxis]


def parse_sample(img_path: tf.Tensor, mask_path: tf.Tensor, image_size: tuple):
    img_bytes = tf.io.read_file(img_path)
    image_raw = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image_raw, image_size)
    image_uint8 = tf.cast(image, tf.uint8)

    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask_bytes, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, image_size, method="nearest")
    mask_bin = tf.cast(mask >= 128, tf.float32)

    def _py_grad(im, m):
        return compute_object_gradient_np(im, m)

    grad_gt = tf.numpy_function(_py_grad, [image_uint8, tf.cast(mask, tf.uint8)], tf.float32)
    grad_gt.set_shape([image_size[0], image_size[1], 1])

    image_float = tf.cast(image, tf.float32) / 255.0
    image_float.set_shape([image_size[0], image_size[1], 3])
    mask_bin.set_shape([image_size[0], image_size[1], 1])

    return image_float, mask_bin, grad_gt


def _random_resized_crop(img, mask, grad_gt, target_size, scale_min=0.5, scale_max=1.0):
    h, w = target_size
    img_h, img_w = tf.shape(img)[0], tf.shape(img)[1]
    scale = tf.random.uniform((), minval=scale_min, maxval=scale_max)
    crop_h = tf.minimum(tf.cast(tf.cast(img_h, tf.float32) * tf.sqrt(scale), tf.int32), img_h)
    crop_w = tf.minimum(tf.cast(tf.cast(img_w, tf.float32) * tf.sqrt(scale), tf.int32), img_w)
    offset_h = tf.random.uniform((), minval=0, maxval=img_h - crop_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform((), minval=0, maxval=img_w - crop_w + 1, dtype=tf.int32)
    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, crop_h, crop_w)
    mask = tf.image.crop_to_bounding_box(mask, offset_h, offset_w, crop_h, crop_w)
    grad_gt = tf.image.crop_to_bounding_box(grad_gt, offset_h, offset_w, crop_h, crop_w)
    return tf.image.resize(img, [h, w]), tf.image.resize(mask, [h, w], method="nearest"), tf.image.resize(grad_gt, [h, w], method="nearest")


def _cutout_tf(img, max_fraction=0.25):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    cut_h = tf.cast(tf.cast(h, tf.float32) * tf.random.uniform((), 0.05, max_fraction), tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * tf.random.uniform((), 0.05, max_fraction), tf.int32)
    cy, cx = tf.random.uniform((), minval=0, maxval=h, dtype=tf.int32), tf.random.uniform((), minval=0, maxval=w, dtype=tf.int32)
    y1, y2 = tf.maximum(cy - cut_h // 2, 0), tf.minimum(cy + cut_h // 2, h)
    x1, x2 = tf.maximum(cx - cut_w // 2, 0), tf.minimum(cx + cut_w // 2, w)
    rows, cols = tf.range(h), tf.range(w)
    keep_mask = tf.cast(tf.expand_dims(tf.logical_or(rows < y1, rows >= y2), 1) | tf.expand_dims(tf.logical_or(cols < x1, cols >= x2), 0), tf.float32)
    return img * tf.expand_dims(keep_mask, -1)


def create_tf_dataset(image_paths, mask_paths, image_size=(384, 384), batch_size=16, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda x, y: parse_sample(x, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 512), seed=42)

        def _augment_fn(img, mask, grad_gt):
            if tf.random.uniform(()) > 0.5:
                img, mask, grad_gt = _random_resized_crop(img, mask, grad_gt, image_size)
            if tf.random.uniform(()) > 0.5:
                img = tf.image.flip_left_right(img)
                mask = tf.image.flip_left_right(mask)
                grad_gt = tf.image.flip_left_right(grad_gt)
            if tf.random.uniform(()) > 0.5:
                img = tf.image.flip_up_down(img)
                mask = tf.image.flip_up_down(mask)
                grad_gt = tf.image.flip_up_down(grad_gt)
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k=k)
            mask = tf.image.rot90(mask, k=k)
            grad_gt = tf.image.rot90(grad_gt, k=k)
            if tf.random.uniform(()) > 0.7:
                img = _cutout_tf(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
            img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD
            return img_norm, mask, grad_gt

        dataset = dataset.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(lambda img, m, g: ((img - IMAGENET_MEAN) / IMAGENET_STD, m, g), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ==============================================================================
# 3. Learning Rate Schedule & Training Loop
# ==============================================================================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = float(warmup_steps)
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=int(total_steps - warmup_steps)
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        cosine_lr = self.cosine_decay(step - self.warmup_steps)
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)


def export_tflite_model(model: tf.keras.Model, export_path: Path, image_size: tuple = (384, 384)):
    """Exports DGNet to INT8 optimized TFLite for edge devices."""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Edge Export] Quantizing DGNet to TFLite INT8: {export_path.resolve()}...")
    try:
        concrete_func = tf.function(lambda x: model(x, training=False)).get_concrete_function(
            tf.TensorSpec([1, image_size[0], image_size[1], 3], tf.float32)
        )
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
        with open(export_path, "wb") as f:
            f.write(tflite_bytes)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        print(f"✓ SUCCESS: Exported TFLite INT8 model ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"Notice (TFLite Export): {e}")
        return False


def run_benchmark_training(
    dataset_name: str = "cod10k",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    backbone: str = "mobilenet_v3_large",
    checkpoint_dir: Path = None,
    upload_on_finish: bool = False,
    repo_id: str = None
):
    """Executes full benchmark pretraining on Kaggle camouflage datasets."""
    checkpoint_dir = checkpoint_dir or (PROJECT_ROOT / "models/raw_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    quantized_dir = PROJECT_ROOT / "models/quantized"
    quantized_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = PROJECT_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_weights_path = checkpoint_dir / "dgnet_benchmark_best.weights.h5"
    best_keras_path = checkpoint_dir / "dgnet_benchmark_best_model.keras"
    latest_weights_path = checkpoint_dir / "dgnet_benchmark_latest.weights.h5"
    tflite_path = quantized_dir / "dgnet_benchmark_s_int8.tflite"

    print("=" * 70)
    print(f" DGNet Pretraining on Kaggle Benchmark: {dataset_name.upper()}")
    print("=" * 70)
    print(f" Backbone:        {backbone}")
    print(f" Epochs:          {epochs}")
    print(f" Batch Size:      {batch_size}")
    print(f" Learning Rate:   {learning_rate}")
    print(f" Checkpoints:     {checkpoint_dir}")
    print("=" * 70)

    # 1. Download and prepare data
    train_imgs, train_masks, val_imgs, val_masks = get_dataset_splits(dataset_name)

    train_ds = create_tf_dataset(train_imgs, train_masks, image_size=(384, 384), batch_size=batch_size, is_train=True)
    val_ds = create_tf_dataset(val_imgs, val_masks, image_size=(384, 384), batch_size=batch_size, is_train=False)

    # 2. Build DGNet Model
    print("\n[Model] Instantiating DGNet architecture with ImageNet context backbone...")
    model = build_dgnet_model(backbone_name=backbone, input_shape=(384, 384, 3), channel=64, pretrained=True)
    _ = model(tf.zeros((1, 384, 384, 3), dtype=tf.float32), training=False)
    total_params = model.count_params()
    print(f"✓ Model compiled: {total_params:,} parameters ({total_params*4 / (1024*1024):.2f} MB)")

    # 3. Setup Loss & Optimizer
    loss_fn = DGNetLoss(lambda_grad_loss=1.0, label_smoothing=0.02)
    total_batches = len(train_imgs) // batch_size
    total_steps = epochs * max(total_batches, 1)
    warmup_steps = min(5 * max(total_batches, 1), total_steps // 5)
    lr_schedule = WarmupCosineDecay(learning_rate, warmup_steps, total_steps)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)

    @tf.function
    def train_step(x_b, y_m, y_g):
        with tf.GradientTape() as tape:
            preds = model(x_b, training=True, return_dict=True)
            preds["gt_grad"] = y_g
            loss = loss_fn(y_m, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x_b, y_m, y_g):
        val_preds = model(x_b, training=False, return_dict=True)
        val_preds["gt_grad"] = y_g
        v_loss = loss_fn(y_m, val_preds)
        prob_mask = tf.nn.sigmoid(val_preds["mask_pred"])
        return v_loss, prob_mask

    # 4. Training Loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    print("\n[Training] Starting pretraining loop...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        batches = 0

        for step, (x_b, y_m, y_g) in enumerate(train_ds):
            loss = train_step(x_b, y_m, y_g)
            loss_val = float(loss)
            epoch_loss += loss_val
            batches += 1

            if batches % 50 == 0 or batches == total_batches:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] Batch [{batches:03d}/{total_batches:03d}] | Loss: {loss_val:.4f} | Avg: {epoch_loss/batches:.4f}")

        avg_train_loss = epoch_loss / max(batches, 1)

        # Validation Pass
        val_loss_total = 0.0
        val_batches = 0
        metrics_acc = CODMetricsAccumulator(threshold=0.5, beta_sq=0.3)

        for vx_b, vy_m, vy_g in val_ds:
            v_loss, prob_mask = val_step(vx_b, vy_m, vy_g)
            val_loss_total += float(v_loss)
            val_batches += 1
            metrics_acc.update(vy_m, prob_mask)

        avg_val_loss = val_loss_total / max(val_batches, 1)
        val_metrics = metrics_acc.result()
        val_iou = val_metrics["iou"]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_iou"].append(val_iou)

        print(f"--> Epoch {epoch:02d} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {val_iou:.4f} | Time: {elapsed:.1f}s")

        # Save ONLY the single best checkpoint whenever validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_weights(str(best_weights_path))
            print(f"  ★ New Best Weights Saved: {best_weights_path.name} (Val Loss: {best_val_loss:.4f}, IoU: {val_iou:.4f})")
            if upload_on_finish or os.getenv("HF_TOKEN"):
                upload_checkpoints_to_hf(repo_id=repo_id, ckpt_dir=checkpoint_dir)

    # Upload only the best weights checkpoint to Hugging Face if requested or running in container
    if upload_on_finish or os.getenv("HF_TOKEN"):
        upload_checkpoints_to_hf(repo_id=repo_id, ckpt_dir=checkpoint_dir)

    print("\n" + "=" * 70)
    print(" ✓ PRETRAINING COMPLETE!")
    print(f" Single Best Checkpoint Saved: {best_weights_path.resolve()}")
    print(" You can now fine-tune this model on your tactical military surveillance dataset!")
    print("=" * 70)


# ==============================================================================
# 4. Hugging Face Synchronization (Upload / Download)
# ==============================================================================

def upload_checkpoints_to_hf(repo_id: str = None, ckpt_dir: Path = None):
    """Uploads ONLY the single best checkpoint to Hugging Face Hub."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[HF Upload] Notice: HF_TOKEN not set. Skipping Hugging Face upload.")
        return False

    repo_id = repo_id or os.getenv("HF_UPLOAD_REPO_ID", os.getenv("HF_REPO_ID", "yakshithkd/COD_training_artifacts"))
    ckpt_dir = ckpt_dir or (PROJECT_ROOT / "models/raw_checkpoints")
    best_ckpt = ckpt_dir / "dgnet_benchmark_best.weights.h5"

    if not best_ckpt.exists():
        print(f"[HF Upload] Best checkpoint not found at: {best_ckpt}")
        return False

    print(f"\n[HF Upload] Uploading best checkpoint '{best_ckpt.name}' to '{repo_id}'...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token.strip())
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        api.upload_file(
            path_or_fileobj=str(best_ckpt),
            path_in_repo="models/raw_checkpoints/dgnet_benchmark_best.weights.h5",
            repo_id=repo_id,
            repo_type="dataset"
        )

        print(f"✓ SUCCESS: Best checkpoint uploaded to Hugging Face: {best_ckpt.name}")
        return True
    except Exception as e:
        print(f"Notice (HF Upload): {e}")
        return False


def download_checkpoints_from_hf(repo_id: str = None, target_dir: Path = None):
    """Downloads ONLY the single best checkpoint from Hugging Face Hub into local repo."""
    target_dir = target_dir or (PROJECT_ROOT / "models/raw_checkpoints")
    target_dir.mkdir(parents=True, exist_ok=True)
    local_dest = target_dir / "dgnet_benchmark_best.weights.h5"
    token = os.getenv("HF_TOKEN", None)

    # Prioritize user-specified repo or the dedicated training artifacts repo
    candidate_repos = []
    if repo_id:
        candidate_repos.append(repo_id)
    if os.getenv("HF_UPLOAD_REPO_ID"):
        candidate_repos.append(os.getenv("HF_UPLOAD_REPO_ID"))
    candidate_repos.extend(["yakshithkd/COD_training_artifacts", "gouravbirwaz/COD_dataset"])

    # Remove duplicates preserving order
    candidate_repos = list(dict.fromkeys(candidate_repos))
    repo_filename = "models/raw_checkpoints/dgnet_benchmark_best.weights.h5"

    for r_id in candidate_repos:
        print(f"\n[HF Download] Fetching best DGNet checkpoint from '{r_id}'...")
        try:
            from huggingface_hub import hf_hub_download
            dl_path = hf_hub_download(
                repo_id=r_id,
                repo_type="dataset",
                filename=repo_filename,
                token=token
            )
            if dl_path and Path(dl_path).exists():
                shutil.copy2(dl_path, str(local_dest))
                size_mb = local_dest.stat().st_size / (1024 * 1024)
                print("\n" + "=" * 70)
                print(" ✓ BEST CHECKPOINT DOWNLOADED SUCCESSFULLY!")
                print(f" Source Repo: {r_id}")
                print(f" Saved to:    {local_dest.resolve()} ({size_mb:.2f} MB)")
                print(" You can now fine-tune your model on the tactical military dataset using:")
                print("   python scripts/DGnet.py --train --resume models/raw_checkpoints/dgnet_benchmark_best.weights.h5 --lr 0.00002")
                print("=" * 70)
                return True
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg or "EntryNotFoundError" in type(e).__name__:
                print(f"  [Notice] Not found in '{r_id}'. Checking next source...")
                continue
            else:
                print(f"  [Notice] Error checking '{r_id}': {e}")
                continue

    print("\n" + "=" * 70)
    print(" ⚠ Checkpoint 'models/raw_checkpoints/dgnet_benchmark_best.weights.h5' was not found in candidate repos.")
    print(" Candidate Repositories checked:")
    for r in candidate_repos:
        print(f"   - {r}")
    print(" Note: If training recently finished, ensure the file was pushed successfully.")
    print("=" * 70)
    return False


# ==============================================================================
# 5. Hugging Face GPU Job Launcher
# ==============================================================================

def generate_hf_job_command(
    flavor: str = "t4-small",
    dataset: str = "cod10k",
    epochs: int = 30,
    batch_size: int = 16,
    backbone: str = "mobilenet_v3_large"
) -> Tuple[str, List[str]]:
    """Builds the exact `hf jobs run` command to launch this training script remotely."""
    hf_token = os.getenv("HF_TOKEN", "<YOUR_HF_TOKEN>")
    upload_repo = os.getenv("HF_UPLOAD_REPO_ID", "yakshithkd/COD_training_artifacts")
    space_id = os.getenv("HF_SPACE_ID", "yakshithkd/COD_training_artifacts")

    # Ensure latest script is uploaded to HF repo so the container can always fetch it
    if hf_token and upload_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=upload_repo, repo_type="dataset", exist_ok=True)
            script_path = PROJECT_ROOT / "scripts/train_kaggle_benchmark.py"
            if script_path.exists():
                api.upload_file(
                    path_or_fileobj=str(script_path),
                    path_in_repo="scripts/train_kaggle_benchmark.py",
                    repo_id=upload_repo,
                    repo_type="dataset"
                )
        except Exception:
            pass

    container_cmd = (
        f"apt-get update -qq && "
        f"apt-get install -y -qq git libgl1 libglib2.0-0 && "
        f"(git clone --depth 1 https://oauth2:{hf_token}@huggingface.co/spaces/{space_id} /app 2>/dev/null || "
        f"git clone --depth 1 https://github.com/Gouravbirwaz/Camouflage-Object-Detection-for-Tactical-Military-Surveillance-COD-.git /app) && "
        f"cd /app && "
        f"pip install --no-cache-dir \"numpy<2.0.0\" kagglehub opencv-python-headless huggingface_hub && "
        f"python3 -c \"from huggingface_hub import hf_hub_download; import shutil; (lambda f: shutil.copy2(f, 'scripts/train_kaggle_benchmark.py'))(hf_hub_download('{upload_repo}', 'scripts/train_kaggle_benchmark.py', repo_type='dataset', token='{hf_token}'))\" 2>/dev/null || true && "
        f"python3 scripts/train_kaggle_benchmark.py --train --dataset {dataset} --epochs {epochs} --batch-size {batch_size} --backbone {backbone} --upload-checkpoints"
    )

    env_args = [
        "-e", f"HF_TOKEN={hf_token}",
        "-e", f"HF_UPLOAD_REPO_ID={upload_repo}",
    ]

    # Forward Kaggle credentials if present
    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME"]:
        if os.getenv(k):
            env_args.extend(["-e", f"{k}={os.getenv(k)}"])

    cmd_list = [
        "hf", "jobs", "run",
        "--flavor", flavor,
        *env_args,
        "tensorflow/tensorflow:2.16.1-gpu",
        "bash", "-c", container_cmd
    ]

    cmd_display = (
        f"hf jobs run \\\n"
        f"  --flavor {flavor} \\\n"
        f"  -e HF_TOKEN={hf_token} \\\n"
        f"  -e HF_UPLOAD_REPO_ID={upload_repo} \\\n"
        f"  tensorflow/tensorflow:2.16.1-gpu \\\n"
        f"  bash -c \"{container_cmd}\""
    )

    return cmd_display, cmd_list


# ==============================================================================
# 6. Main CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="DGNet Kaggle Camouflage Benchmark Training & Remote HF Jobs")
    parser.add_argument("--train", action="store_true", help="Start training on Kaggle benchmark dataset")
    parser.add_argument("--dataset", type=str, default="cod10k", choices=["cod10k", "nc4k", "camo", "all"], help="Benchmark dataset to train on (default: cod10k)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large", help="Backbone encoder (default: mobilenet_v3_large)")
    parser.add_argument("--upload-checkpoints", action="store_true", help="Upload trained checkpoints to Hugging Face Hub")
    parser.add_argument("--download-checkpoints", action="store_true", help="Download trained checkpoints from Hugging Face Hub to local repo")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face repo ID for upload/download")

    # HF Job options
    parser.add_argument("--launch-hf-job", action="store_true", help="Generate or launch a remote Hugging Face GPU Job")
    parser.add_argument("--flavor", type=str, default="t4-small", help="HF Hardware flavor (e.g., t4-small, t4-medium, a10g-small, a100-large)")
    parser.add_argument("--dry-run", action="store_true", help="Only generate and display command without launching")
    parser.add_argument("--detach", "-d", action="store_true", help="Run HF job in background")

    args = parser.parse_args()

    # 1. Download Mode
    if args.download_checkpoints:
        download_checkpoints_from_hf(repo_id=args.repo_id)
        return

    # 2. Launch HF Job Mode
    if args.launch_hf_job:
        cmd_display, cmd_list = generate_hf_job_command(
            flavor=args.flavor,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            backbone=args.backbone
        )
        if args.detach:
            cmd_list.insert(3, "--detach")

        print("\n" + "=" * 70)
        print("      Hugging Face Cloud GPU Training Job Generator")
        print("=" * 70)
        print(f" Target Hardware Flavor: {args.flavor}")
        print(f" Dataset:                {args.dataset.upper()}")
        print(f" Epochs / Batch Size:    {args.epochs} / {args.batch_size}")
        print("\nJob Command:\n")
        print(cmd_display)
        print("\n" + "=" * 70)

        if args.dry_run:
            print("\n[Dry Run] Command generated successfully. Job was NOT submitted.")
            return

        print("\n[Launching Job] Executing command via Hugging Face CLI...")
        try:
            res = subprocess.run(cmd_list)
            if res.returncode == 0:
                print("\n[HF Jobs] Job submitted successfully!")
                print("Track your job with: hf jobs ps  or  hf jobs logs <job_id>")
            else:
                print(f"\n[HF Jobs] Process exited with code: {res.returncode}")
        except Exception as e:
            print(f"Error launching job: {e}")
        return

    # 3. Train Mode
    if args.train:
        run_benchmark_training(
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            backbone=args.backbone,
            upload_on_finish=args.upload_checkpoints,
            repo_id=args.repo_id
        )
        return

    # Default help
    parser.print_help()


if __name__ == "__main__":
    main()

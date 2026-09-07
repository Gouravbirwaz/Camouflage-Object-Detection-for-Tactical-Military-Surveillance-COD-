#!/usr/bin/env python3
"""
DGnet Real Dataset Training & Execution Script - Realtime Camouflaged Target Detection
Pure TensorFlow 2.x / Keras Implementation for Tactical Military Surveillance

Supports:
- Local Windows / Linux execution
- Remote Hugging Face GPU Jobs (NVIDIA A100 / T4 / V100)
- Configurable environment variable paths (DATA_DIR, CHECKPOINT_DIR, LOG_DIR, MODEL_DIR)
- Automatic GPU hardware detection
- Checkpoint persistence and training resumption

Usage:
    python scripts/DGnet.py --train --epochs 10 --batch-size 8
    python scripts/DGnet.py --train --resume
    python scripts/DGnet.py --evaluate
    python scripts/DGnet.py --summary --backbone mobilenet_v3_large
    python scripts/DGnet.py --export-tflite --output models/quantized/dgnet_mobilenet_v3.tflite
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import yaml

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root directory to python path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dgnet import DGNet, DGNetLoss, build_dgnet_model, TFGradientExtractor


def setup_gpu_environment():
    """Detects physical GPUs and configures TensorFlow memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n[GPU Hardware Detection] Found {len(gpus)} GPU(s):")
        for idx, gpu in enumerate(gpus):
            print(f"  • GPU [{idx}]: {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"    Notice (Memory Growth): {e}")
    else:
        print("\n[GPU Hardware Detection] No physical GPU detected by TensorFlow. Running on CPU.")


def load_config(config_path: str = "params.yaml") -> dict:
    """Loads configuration parameters from params.yaml."""
    cfg_file = PROJECT_ROOT / config_path
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            params = yaml.safe_load(f)
        return params
    return {}


def _find_dataset_splits_in_dir(target_dir: Path, valid_exts: set) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Safely inspects a candidate directory for explicit dataset splits (Training, Testing) or image/GT folders.
    Ignores non-dataset directories (e.g. dvc-store, .git, checkpoints) and handles disconnected FUSE mounts safely.
    """
    ignore_names = {"dvc-store", ".dvc", ".git", "__pycache__", ".cache", "raw_checkpoints", "logs", "models", "metrics"}
    splits = {}

    try:
        if not target_dir.exists() or not target_dir.is_dir():
            return {}
    except (OSError, PermissionError):
        return {}

    # Collect directory candidates while ignoring known noise folders
    split_folders = []
    try:
        for d in target_dir.rglob("*"):
            try:
                if any(ignored in d.parts for ignored in ignore_names):
                    continue
                if d.is_dir() and d.name.lower() in ["training", "testing", "validation", "val", "train", "test"]:
                    split_folders.append(d)
            except (OSError, PermissionError):
                continue
    except (OSError, PermissionError):
        pass

    if split_folders:
        for split_dir in split_folders:
            split_name = split_dir.name.capitalize()
            img_dir, mask_dir = None, None

            try:
                for child in split_dir.iterdir():
                    try:
                        if child.is_dir():
                            c_name = child.name.lower()
                            if any(k in c_name for k in ["image", "images", "img", "imgs", "rgb"]):
                                img_dir = child
                            elif any(k in c_name for k in ["gt", "mask", "masks", "label", "labels", "groundtruth"]):
                                mask_dir = child
                    except (OSError, PermissionError):
                        continue
            except (OSError, PermissionError):
                continue

            if img_dir and mask_dir:
                img_files = {}
                mask_files = {}
                try:
                    for f in img_dir.glob("*"):
                        try:
                            if f.is_file() and f.suffix.lower() in valid_exts:
                                img_files[f.stem] = str(f.resolve())
                        except (OSError, PermissionError):
                            continue
                except (OSError, PermissionError):
                    pass

                try:
                    for f in mask_dir.glob("*"):
                        try:
                            if f.is_file() and f.suffix.lower() in valid_exts:
                                mask_files[f.stem] = str(f.resolve())
                        except (OSError, PermissionError):
                            continue
                except (OSError, PermissionError):
                    pass

                common_stems = sorted(list(set(img_files.keys()).intersection(set(mask_files.keys()))))
                paired_imgs = [img_files[stem] for stem in common_stems]
                paired_masks = [mask_files[stem] for stem in common_stems]

                if paired_imgs:
                    splits[split_name] = (paired_imgs, paired_masks)

    # Fallback: scan directory directly if no explicit split folders matched
    if not splits:
        img_files = {}
        mask_files = {}

        try:
            for f in target_dir.rglob("*"):
                try:
                    if any(ignored in f.parts for ignored in ignore_names):
                        continue
                    if f.is_file() and f.suffix.lower() in valid_exts:
                        p_str = str(f.parent.name.lower())
                        if any(k in p_str for k in ["image", "images", "img", "imgs", "rgb"]):
                            img_files[f.stem] = str(f.resolve())
                        elif any(k in p_str for k in ["gt", "mask", "masks", "label", "labels", "groundtruth"]):
                            mask_files[f.stem] = str(f.resolve())
                except (OSError, PermissionError):
                    continue
        except (OSError, PermissionError):
            pass

        common_stems = sorted(list(set(img_files.keys()).intersection(set(mask_files.keys()))))
        paired_imgs = [img_files[stem] for stem in common_stems]
        paired_masks = [mask_files[stem] for stem in common_stems]

        if paired_imgs:
            splits["Default"] = (paired_imgs, paired_masks)

    return splits


def discover_dataset_pairs(data_dir: Path) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Scans candidate data directories and auto-discovers matched RGB images and GT mask pairs.
    Handles existing splits (Training, Testing) or raw image/mask folders.
    Supports mounted Hugging Face Storage Buckets / S3 mounts (/data, /mnt/data, etc.).
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    req_path = Path(data_dir)
    candidate_paths = []

    # Priority candidate paths
    if req_path.is_absolute():
        candidate_paths.append(req_path)
    else:
        candidate_paths.append(PROJECT_ROOT / req_path)
        candidate_paths.append(req_path)

    # Common mounted paths on HF Space S3 storage bucket /data
    candidate_paths.extend([
        Path("/data") / req_path,
        Path("/data/data/raw/dataset-splitM"),
        Path("/data/raw/dataset-splitM"),
        Path("/data/dataset-splitM"),
        Path("/data/data/raw"),
        Path("/data/raw"),
        Path("/data"),
        PROJECT_ROOT / "data/raw/dataset-splitM",
        PROJECT_ROOT / "data/raw",
        Path("/mnt/data"),
        Path("/bucket"),
        Path("/hf/COD_dataset"),
        Path("/datasets/gouravbirwaz/COD_dataset"),
        Path("/mnt/s3"),
    ])

    # De-duplicate candidate paths while preserving order
    seen = set()
    unique_candidates = []
    for cp in candidate_paths:
        try:
            resolved = str(cp.resolve())
        except Exception:
            resolved = str(cp)
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(cp)

    for p in unique_candidates:
        try:
            splits = _find_dataset_splits_in_dir(p, valid_exts)
            if splits:
                print(f"\n[Dataset Discovery] Successfully discovered dataset at: {p.resolve()}")
                for split_name, (imgs, masks) in splits.items():
                    print(f"  • Split '{split_name}': Found {len(imgs)} matched image-mask pairs.")
                return splits
        except Exception:
            continue

    # Fallback: Check if mounted DVC store exists and run DVC checkout automatically
    dvc_store_paths = [Path("/data/dvc-store"), Path("/data"), Path("/mnt/data/dvc-store")]
    found_dvc_store = None
    for dvc_p in dvc_store_paths:
        try:
            if dvc_p.exists() and dvc_p.is_dir():
                found_dvc_store = dvc_p
                break
        except Exception:
            continue

    if found_dvc_store:
        print(f"\n[Dataset Discovery] Attempting automatic DVC checkout from mounted store: {found_dvc_store.resolve()}...")
        try:
            import subprocess
            subprocess.run(["dvc", "config", "cache.dir", str(found_dvc_store.resolve())], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "config", "cache.type", "copy"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "checkout", "data/raw.dvc"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "checkout"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Re-scan candidate paths after DVC checkout
            for p in unique_candidates:
                splits = _find_dataset_splits_in_dir(p, valid_exts)
                if splits:
                    print(f"[Dataset Discovery] Successfully restored dataset via DVC at: {p.resolve()}")
                    for split_name, (imgs, masks) in splits.items():
                        print(f"  • Split '{split_name}': Found {len(imgs)} matched image-mask pairs.")
                    return splits
        except Exception as e:
            print(f"[Dataset Discovery] DVC checkout note: {e}")

    print(f"\n[Dataset Discovery] Warning: No dataset pairs discovered across candidate paths.")
    return {}


SOBEL_KX = tf.constant([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=tf.float32)[:, :, np.newaxis, np.newaxis]
SOBEL_KY = tf.constant([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=tf.float32)[:, :, np.newaxis, np.newaxis]


def compute_sobel_gradient_tf(mask: tf.Tensor) -> tf.Tensor:
    """Computes ground-truth Sobel edge magnitude map from a binary mask tensor."""
    is_3d = (len(mask.shape) == 3 or tf.rank(mask) == 3)
    mask_4d = tf.expand_dims(mask, 0) if is_3d else mask

    gx = tf.nn.conv2d(mask_4d, SOBEL_KX, strides=[1, 1, 1, 1], padding="SAME")
    gy = tf.nn.conv2d(mask_4d, SOBEL_KY, strides=[1, 1, 1, 1], padding="SAME")
    magnitude = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-8)
    grad_map = tf.cast(magnitude > 0.1, tf.float32)

    return tf.squeeze(grad_map, 0) if is_3d else grad_map


def parse_sample(img_path: tf.Tensor, mask_path: tf.Tensor, image_size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Reads, decodes, resizes, normalizes, and extracts Sobel edge map for an image-mask pair."""
    img_bytes = tf.io.read_file(img_path)
    image = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask_bytes, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, image_size, method="nearest")
    mask = tf.cast(mask >= 128, tf.float32)

    grad_gt = compute_sobel_gradient_tf(mask)
    return image, mask, grad_gt


def create_tf_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    image_size: Tuple[int, int] = (384, 384),
    batch_size: int = 8,
    is_train: bool = True
) -> tf.data.Dataset:
    """Builds a high-throughput TensorFlow tf.data.Dataset pipeline with RAM caching."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    def _map_fn(img_p, mask_p):
        return parse_sample(img_p, mask_p, image_size=image_size)

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()

    if is_train:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 1000), seed=42)

        def _augment_fn(img, mask, grad_gt):
            if tf.random.uniform(()) > 0.5:
                img = tf.image.flip_left_right(img)
                mask = tf.image.flip_left_right(mask)
                grad_gt = tf.image.flip_left_right(grad_gt)
            img = tf.image.random_brightness(img, max_delta=0.15)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, mask, grad_gt

        dataset = dataset.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def print_model_summary(model: DGNet, image_size: tuple = (384, 384)):
    """Prints architectural summary and parameter counts using Keras."""
    dummy_input = tf.zeros((1, image_size[0], image_size[1], 3))
    _ = model(dummy_input, training=False)

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    model_size_mb = (total_params * 4) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("      DGNet (Deep Gradient Network) Model Summary - TensorFlow 2.x")
    print("=" * 70)
    print(f" Context Backbone:    {model.backbone_name}")
    print(f" Gradient Kernel:     {model.gradient_kernel}")
    print(f" Target Input Size:   [{image_size[0]}, {image_size[1]}, 3]")
    print(f" Total Parameters:    {total_params:,}")
    print(f" Trainable Params:    {trainable_params:,}")
    print(f" Model Memory Size:   ~{model_size_mb:.2f} MB (FP32 precision)")
    print("=" * 70 + "\n")


def train_dgnet(
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    data_dir: str = "data/raw/dataset-splitM",
    checkpoint_dir: str = "models/raw_checkpoints",
    log_dir: str = "metrics",
    backbone: str = "mobilenet_v3_large",
    kernel: str = "sobel",
    lambda_grad: float = 1.5,
    resume: bool = False
):
    """Executes full model training on the dataset with checkpoint persistence and resume support."""
    setup_gpu_environment()

    params = load_config("params.yaml")
    img_size = tuple(params.get("dataset", {}).get("image_size", [384, 384]))

    # Resolve configurable paths
    data_path = Path(data_dir) if Path(data_dir).is_absolute() else PROJECT_ROOT / data_dir
    ckpt_dir = Path(checkpoint_dir) if Path(checkpoint_dir).is_absolute() else PROJECT_ROOT / checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_dir = Path(log_dir) if Path(log_dir).is_absolute() else PROJECT_ROOT / log_dir
    metrics_log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover Real Dataset
    splits = discover_dataset_pairs(data_path)
    if not splits:
        raise FileNotFoundError(f"No paired images and masks found in target directory: {data_path.resolve()}")

    if "Training" in splits:
        train_imgs, train_masks = splits["Training"]
        if "Testing" in splits:
            val_imgs, val_masks = splits["Testing"]
        else:
            val_idx = int(len(train_imgs) * 0.8)
            val_imgs, val_masks = train_imgs[val_idx:], train_masks[val_idx:]
            train_imgs, train_masks = train_imgs[:val_idx], train_masks[:val_idx]
    else:
        all_imgs, all_masks = list(splits.values())[0]
        val_idx = int(len(all_imgs) * 0.8)
        train_imgs, train_masks = all_imgs[:val_idx], all_masks[:val_idx]
        val_imgs, val_masks = all_imgs[val_idx:], all_masks[val_idx:]

    print(f"\n[Training Setup] Training Samples: {len(train_imgs)} | Validation Samples: {len(val_imgs)}")

    # 2. Build tf.data Datasets
    train_ds = create_tf_dataset(train_imgs, train_masks, image_size=img_size, batch_size=batch_size, is_train=True)
    val_ds = create_tf_dataset(val_imgs, val_masks, image_size=img_size, batch_size=batch_size, is_train=False)

    # 3. Build DGNet Model & Optimizer
    print(f"[Training Setup] Building DGNet model with backbone='{backbone}'...")
    model = build_dgnet_model(
        backbone_name=backbone,
        input_shape=(img_size[0], img_size[1], 3),
        gradient_kernel=kernel,
        pretrained=True
    )
    loss_fn = DGNetLoss(lambda_grad_loss=lambda_grad)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * len(train_ds)
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    best_weights_path = ckpt_dir / "dgnet_best.weights.h5"
    latest_weights_path = ckpt_dir / "dgnet_latest.weights.h5"
    final_weights_path = ckpt_dir / "dgnet_final.weights.h5"
    best_model_path = ckpt_dir / "dgnet_best_model.keras"
    metrics_log_path = metrics_log_dir / "train_metrics.json"

    # Initialize model weights shape by calling dummy forward pass
    dummy_x = tf.zeros((1, img_size[0], img_size[1], 3))
    _ = model(dummy_x, training=False)

    # Checkpoint Resume Logic - Multi-candidate search
    if resume:
        candidate_dirs = [
            ckpt_dir,
            Path("/data/models/raw_checkpoints"),
            Path("/app/models/raw_checkpoints"),
            PROJECT_ROOT / "models/raw_checkpoints"
        ]

        found_weights = None
        for cdir in candidate_dirs:
            try:
                if not cdir.exists():
                    continue
                for w_name in ["dgnet_latest.weights.h5", "dgnet_best.weights.h5", "dgnet_final.weights.h5"]:
                    wp = cdir / w_name
                    if wp.exists() and wp.stat().st_size > 0:
                        found_weights = wp
                        break
                if found_weights:
                    break
                for wp in cdir.glob("*.weights.h5"):
                    if wp.is_file() and wp.stat().st_size > 0:
                        found_weights = wp
                        break
                if found_weights:
                    break
            except Exception:
                continue

        if found_weights:
            print(f"\n[Resume Training] Found checkpoint weights: {found_weights.resolve()}")
            model.load_weights(str(found_weights))
            print("[Resume Training] Checkpoint weights loaded successfully! Continuing training...")
        else:
            print(f"\n[Resume Training] Notice: Resume flag set, but no checkpoint found in candidate locations. Starting fresh training.")

    best_val_loss = float("inf")
    history = {"train_total_loss": [], "val_total_loss": [], "val_iou": []}

    print("\n" + "=" * 70)
    print(f"             Starting DGNet Real Model Training ({epochs} Epochs)")
    print("=" * 70)

    # 4. Compiled Step Functions & Warmup
    @tf.function
    def train_step(x_b, y_m, y_g):
        with tf.GradientTape() as tape:
            preds = model(x_b, training=True, return_dict=True)
            preds_dict = {
                "mask_pred": preds["mask_pred"],
                "grad_pred": preds["grad_pred"],
                "gt_grad": y_g
            }
            loss = loss_fn(y_m, preds_dict)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x_b, y_m, y_g):
        val_preds = model(x_b, training=False, return_dict=True)
        val_preds_dict = {
            "mask_pred": val_preds["mask_pred"],
            "grad_pred": val_preds["grad_pred"],
            "gt_grad": y_g
        }
        v_loss = loss_fn(y_m, val_preds_dict)
        prob_mask = tf.nn.sigmoid(val_preds["mask_pred"])
        pred_bin = tf.cast(prob_mask >= 0.5, tf.float32)
        inter = tf.reduce_sum(pred_bin * y_m)
        union = tf.reduce_sum(pred_bin + y_m) - inter
        return v_loss, inter, union

    print("\n[Training Setup] Compiling @tf.function model graph...", flush=True)
    for w_x, w_y, w_g in train_ds.take(1):
        _ = train_step(w_x, w_y, w_g)
        _ = val_step(w_x, w_y, w_g)
    print("[Training Setup] Graph compilation complete! Starting training loop...\n", flush=True)

    try:
        from tqdm import tqdm
        use_tqdm = sys.stdout.isatty()
    except ImportError:
        use_tqdm = False

    start_train_time = time.time()
    total_train_batches = len(train_ds)
    total_val_batches = len(val_ds)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss_sum = 0.0
        num_batches = 0

        if use_tqdm:
            pbar = tqdm(train_ds, total=total_train_batches, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False)
            for x_batch, y_mask, y_grad in pbar:
                loss = train_step(x_batch, y_mask, y_grad)
                loss_val = float(loss)
                train_loss_sum += loss_val
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "avg_loss": f"{train_loss_sum / num_batches:.4f}"})
        else:
            print(f"---> Epoch [{epoch:02d}/{epochs:02d}] Training on {len(train_imgs)} samples...", flush=True)
            for x_batch, y_mask, y_grad in train_ds:
                loss = train_step(x_batch, y_mask, y_grad)
                loss_val = float(loss)
                train_loss_sum += loss_val
                num_batches += 1
                if num_batches % 10 == 0 or num_batches == total_train_batches:
                    print(f"  Batch [{num_batches:03d}/{total_train_batches:03d}] | Loss: {loss_val:.4f} | Avg Loss: {train_loss_sum / num_batches:.4f}", flush=True)

        avg_train_loss = train_loss_sum / num_batches if num_batches > 0 else 0.0

        val_loss_sum = 0.0
        val_batches = 0
        intersection_sum = 0.0
        union_sum = 0.0

        if use_tqdm:
            val_pbar = tqdm(val_ds, total=total_val_batches, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]  ", leave=False)
            for x_val, y_val_mask, y_val_grad in val_pbar:
                v_loss, inter, union = val_step(x_val, y_val_mask, y_val_grad)
                val_loss_sum += float(v_loss)
                intersection_sum += float(inter)
                union_sum += float(union)
                val_batches += 1
        else:
            for x_val, y_val_mask, y_val_grad in val_ds:
                v_loss, inter, union = val_step(x_val, y_val_mask, y_val_grad)
                val_loss_sum += float(v_loss)
                intersection_sum += float(inter)
                union_sum += float(union)
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
        val_iou = (intersection_sum + 1.0) / (union_sum + 1.0)

        epoch_time = time.time() - epoch_start
        history["train_total_loss"].append(avg_train_loss)
        history["val_total_loss"].append(avg_val_loss)
        history["val_iou"].append(val_iou)

        # Always update latest weights checkpoint after each epoch
        try:
            model.save_weights(str(latest_weights_path))
        except Exception as e:
            print(f"  Notice (Checkpoint Save): {e}")

        saved_str = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                model.save_weights(str(best_weights_path))
                model.save(str(best_model_path))
            except Exception:
                pass
            saved_str = " -> Best Checkpoint Saved!"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {val_iou:.4f}{saved_str}", flush=True)

        with open(metrics_log_path, "w") as f:
            json.dump(history, f, indent=4)

    # Save final checkpoint
    try:
        model.save_weights(str(final_weights_path))
        if not best_weights_path.exists() and latest_weights_path.exists():
            shutil.copy2(str(latest_weights_path), str(best_weights_path))
    except Exception as e:
        print(f"[Checkpoint Save] Note: {e}")

    # Sync checkpoints to local models/raw_checkpoints if saved elsewhere
    fallback_ckpt_dir = PROJECT_ROOT / "models/raw_checkpoints"
    if ckpt_dir.resolve() != fallback_ckpt_dir.resolve():
        fallback_ckpt_dir.mkdir(parents=True, exist_ok=True)
        for w_file in [latest_weights_path, best_weights_path, final_weights_path]:
            if w_file.exists():
                try:
                    shutil.copy2(str(w_file), str(fallback_ckpt_dir / w_file.name))
                except Exception:
                    pass

    total_training_time = time.time() - start_train_time
    print("=" * 70)
    print(f" Training Completed in {total_training_time / 60.0:.2f} minutes!")
    print(f" Best Validation Loss: {best_val_loss:.4f}")
    print(f" Checkpoint Location:  {ckpt_dir.resolve()}")
    print("=" * 70 + "\n")

    with open(metrics_log_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[Metrics] Saved training history to: {metrics_log_path.resolve()}\n")


def evaluate_dgnet(data_dir: str = "data/raw/dataset-splitM", checkpoint_dir: str = "models/raw_checkpoints", log_dir: str = "metrics", backbone: str = "mobilenet_v3_large", kernel: str = "sobel"):
    """Evaluates the trained model on test dataset split."""
    setup_gpu_environment()

    params = load_config("params.yaml")
    img_size = tuple(params.get("dataset", {}).get("image_size", [384, 384]))

    data_path = Path(data_dir) if Path(data_dir).is_absolute() else PROJECT_ROOT / data_dir
    ckpt_dir = Path(checkpoint_dir) if Path(checkpoint_dir).is_absolute() else PROJECT_ROOT / checkpoint_dir
    metrics_log_dir = Path(log_dir) if Path(log_dir).is_absolute() else PROJECT_ROOT / log_dir
    metrics_log_dir.mkdir(parents=True, exist_ok=True)

    splits = discover_dataset_pairs(data_path)
    test_key = "Testing" if "Testing" in splits else list(splits.keys())[0]
    test_imgs, test_masks = splits[test_key]

    print(f"\n[Evaluation] Evaluating DGNet on split '{test_key}' ({len(test_imgs)} samples)...")
    test_ds = create_tf_dataset(test_imgs, test_masks, image_size=img_size, batch_size=4, is_train=False)

    model = build_dgnet_model(backbone_name=backbone, input_shape=(img_size[0], img_size[1], 3), gradient_kernel=kernel, pretrained=False)

    # Multi-candidate search for weights
    candidate_dirs = [
        ckpt_dir,
        Path("/data/models/raw_checkpoints"),
        Path("/app/models/raw_checkpoints"),
        PROJECT_ROOT / "models/raw_checkpoints"
    ]
    found_weights = None
    for cdir in candidate_dirs:
        try:
            if not cdir.exists():
                continue
            for w_name in ["dgnet_best.weights.h5", "dgnet_latest.weights.h5", "dgnet_final.weights.h5"]:
                wp = cdir / w_name
                if wp.exists() and wp.stat().st_size > 0:
                    found_weights = wp
                    break
            if found_weights:
                break
        except Exception:
            continue

    if found_weights:
        _ = model(tf.zeros((1, img_size[0], img_size[1], 3)), training=False)
        model.load_weights(str(found_weights))
        print(f"[Evaluation] Loaded weights from: {found_weights.resolve()}")
    else:
        print(f"[Evaluation] Warning: No trained weights checkpoint found. Evaluating with initialized model.")

    mae_sum = 0.0
    count = 0
    start_time = time.perf_counter()

    for x_batch, y_mask, _ in test_ds:
        pred_mask = model(x_batch, training=False)
        mae = tf.reduce_mean(tf.abs(pred_mask - y_mask))
        mae_sum += float(mae) * len(x_batch)
        count += len(x_batch)

    total_time = time.perf_counter() - start_time
    avg_mae = mae_sum / count if count > 0 else 0.0
    fps = count / total_time if total_time > 0 else 0.0

    eval_results = {
        "dataset_split": test_key,
        "sample_count": count,
        "mean_absolute_error_mae": round(avg_mae, 4),
        "evaluation_fps": round(fps, 1)
    }

    eval_log_path = metrics_log_dir / "evaluation_results.json"
    with open(eval_log_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    print("\n" + "=" * 70)
    print("                    DGNet Evaluation Metrics")
    print("=" * 70)
    print(f" Test Set Samples:       {count}")
    print(f" Mean Absolute Error:    {avg_mae:.4f}")
    print(f" Realtime Speed:         {fps:.1f} FPS")
    print(f" Saved Evaluation JSON:  {eval_log_path.resolve()}")
    print("=" * 70 + "\n")


def export_tflite(
    model: DGNet,
    output_path: str = "models/quantized/dgnet_s_int8.tflite",
    image_size: tuple = (384, 384),
    quantize_int8: bool = False
):
    """Exports DGNet to TensorFlow Lite (.tflite) for edge deployment on Jetson / ARM / Raspberry Pi."""
    output_file = Path(output_path) if Path(output_path).is_absolute() else PROJECT_ROOT / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Export] Exporting DGNet (Backbone: {model.backbone_name}) to TFLite format...")
    print(f"[Export] Target File: {output_file.resolve()}")

    concrete_func = tf.function(lambda x: model(x, training=False)).get_concrete_function(
        tf.TensorSpec([1, image_size[0], image_size[1], 3], tf.float32)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize_int8:
        print("[Export] Applying INT8 Quantization...")
        def representative_dataset_gen():
            for _ in range(50):
                yield [np.random.normal(size=(1, image_size[0], image_size[1], 3)).astype(np.float32)]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(tflite_model)

    print(f"[Export] SUCCESS: TFLite model successfully saved ({output_file.stat().st_size / (1024*1024):.2f} MB).\n")


def main():
    params = load_config("params.yaml")
    model_cfg = params.get("model", {})
    train_cfg = params.get("train", {})

    # Priority: Env Vars > Config Params > Default relative paths
    env_data_dir = os.getenv("DATA_DIR", "data/raw/dataset-splitM")
    env_model_dir = os.getenv("MODEL_DIR", "models")
    env_ckpt_dir = os.getenv("CHECKPOINT_DIR", "models/raw_checkpoints")
    env_log_dir = os.getenv("LOG_DIR", "metrics")
    env_resume = os.getenv("RESUME", "false").lower() in ("true", "1", "t", "yes")

    default_backbone = model_cfg.get("context_backbone", "mobilenet_v3_large")
    default_kernel = model_cfg.get("gradient_kernel", "sobel")
    default_img_size = tuple(params.get("dataset", {}).get("image_size", [384, 384]))
    default_epochs = train_cfg.get("epochs", 10)
    default_batch_size = train_cfg.get("batch_size", params.get("dataset", {}).get("batch_size", 8))
    default_lr = train_cfg.get("learning_rate", 0.001)

    parser = argparse.ArgumentParser(description="DGNet: Deep Gradient Network Real Model Training & Execution (TensorFlow 2.x)")
    parser.add_argument("--train", action="store_true", help="Execute real model training on dataset")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model on test dataset split")
    parser.add_argument("--resume", action="store_true", default=env_resume, help="Resume training from latest checkpoint in CHECKPOINT_DIR")
    parser.add_argument("--backbone", type=str, default=default_backbone, help="Backbone architecture")
    parser.add_argument("--kernel", type=str, default=default_kernel, help="Gradient kernel type")
    parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=default_lr, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=env_data_dir, help="Path to data directory (or via DATA_DIR env var)")
    parser.add_argument("--checkpoint-dir", type=str, default=env_ckpt_dir, help="Path to checkpoint directory (or via CHECKPOINT_DIR env var)")
    parser.add_argument("--log-dir", type=str, default=env_log_dir, help="Path to metrics/log directory (or via LOG_DIR env var)")
    parser.add_argument("--model-dir", type=str, default=env_model_dir, help="Path to model directory (or via MODEL_DIR env var)")
    parser.add_argument("--summary", action="store_true", help="Print model summary and parameter stats")
    parser.add_argument("--export-tflite", action="store_true", help="Export model to TensorFlow Lite format")
    parser.add_argument("--output", type=str, default=None, help="Custom output filepath for export/checkpoint")

    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.summary, args.export_tflite]):
        args.train = True

    if args.train:
        train_dgnet(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            backbone=args.backbone,
            kernel=args.kernel,
            resume=args.resume
        )

    if args.evaluate:
        evaluate_dgnet(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            backbone=args.backbone,
            kernel=args.kernel
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
        out_path = args.output if args.output else os.path.join(args.model_dir, "quantized/dgnet_mobilenet_v3.tflite")
        export_tflite(model, output_path=out_path, image_size=default_img_size)


if __name__ == "__main__":
    main()

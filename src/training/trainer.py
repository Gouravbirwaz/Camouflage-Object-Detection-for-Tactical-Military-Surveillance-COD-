"""
DGNet Training Engine for Camouflaged Object Detection (COD).
Implements the training loop with AdamW, Warmup + Cosine Decay, multi-scale tactical augmentations,
checkpoint persistence, MLflow tracking, and remote Hugging Face sync.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

from src.utils.config import get_project_root, load_project_config
from src.utils.hardware import setup_gpu_environment
from src.utils.metrics import CODMetricsAccumulator
from src.tracking.mlflow_utils import MLflowTracker
from src.models.dgnet import DGNet, DGNetLoss, build_dgnet_model
from src.data.dataset import (
    discover_dataset_pairs,
    _find_dataset_splits_in_dir,
    create_tf_dataset,
)

PROJECT_ROOT = get_project_root()


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine learning rate decay."""

    def __init__(self, initial_lr: float, warmup_steps: float, total_steps: float):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=int(max(total_steps - warmup_steps, 1))
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / tf.maximum(self.warmup_steps, 1.0))
        cosine_lr = self.cosine_decay(step - self.warmup_steps)
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


def print_model_summary(model: DGNet, image_size: Tuple[int, int] = (384, 384)):
    """Prints architectural summary and parameter counts."""
    dummy_input = tf.zeros((1, image_size[0], image_size[1], 3))
    _ = model(dummy_input, training=False)

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    model_size_mb = (total_params * 4) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("      DGNet (Deep Gradient Network) Model Summary - TensorFlow 2.x")
    print("=" * 70)
    print(f" Context Backbone:    {getattr(model, 'backbone_name', 'mobilenet_v3_large')}")
    print(f" Gradient Kernel:     {getattr(model, 'gradient_kernel', 'sobel')}")
    print(f" Target Input Size:   [{image_size[0]}, {image_size[1]}, 3]")
    print(f" Total Parameters:    {total_params:,}")
    print(f" Trainable Params:    {trainable_params:,}")
    print(f" Model Memory Size:   ~{model_size_mb:.2f} MB (FP32 precision)")
    print("=" * 70 + "\n")


def train_dgnet(
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.00005,
    data_dir: str = "data/raw/dataset-splitM",
    augmented_dir: Optional[str] = None,
    checkpoint_dir: str = "models/raw_checkpoints",
    log_dir: str = "metrics",
    backbone: str = "mobilenet_v3_large",
    kernel: str = "sobel",
    lambda_grad: float = 1.0,
    resume: Union[bool, str] = False,
    weights: Optional[str] = None,
    model_version: Optional[str] = None,
    experiment_name: Optional[str] = None,
    enable_mlflow: bool = True
):
    """Executes full model training on the dataset with checkpoint persistence, versioning, and resume support."""
    setup_gpu_environment()

    params = load_project_config("params.yaml")
    img_size = tuple(params.get("dataset", {}).get("image_size", [384, 384]))

    cfg = load_project_config()
    version_str = model_version or cfg.get("train", {}).get("model_version") or cfg.get("model", {}).get("version", "v1.0")
    version_tag = version_str if str(version_str).startswith("v") else f"v{version_str}"

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

    # Merge augmented military tactical data
    augmented_candidates = []
    if augmented_dir:
        augmented_candidates.append(Path(augmented_dir) if Path(augmented_dir).is_absolute() else PROJECT_ROOT / augmented_dir)
        augmented_candidates.append(Path(augmented_dir))
    augmented_candidates.extend([
        PROJECT_ROOT / "data" / "augmented" / "military_tactical",
        PROJECT_ROOT / "data" / "augmented",
        Path("/data/augmented/military_tactical"),
        Path("/data/augmented"),
        Path("/data/data/augmented/military_tactical"),
        Path("/data/data/augmented"),
        Path("/mnt/data/augmented/military_tactical"),
        Path("/mnt/data/augmented"),
    ])

    seen_aug_dirs = set()
    for aug_dir in augmented_candidates:
        try:
            resolved_aug = str(aug_dir.resolve())
        except Exception:
            resolved_aug = str(aug_dir)
        if resolved_aug in seen_aug_dirs:
            continue
        seen_aug_dirs.add(resolved_aug)

        try:
            aug_splits = _find_dataset_splits_in_dir(aug_dir, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
            if aug_splits:
                for split_name, (aug_imgs, aug_masks) in aug_splits.items():
                    if split_name.lower() in ["training", "train", "default"]:
                        orig_count = len(train_imgs)
                        train_imgs.extend(aug_imgs)
                        train_masks.extend(aug_masks)
                        print(f"\n[Dataset Merge] Successfully merged {len(aug_imgs)} augmented military tactical samples from: {aug_dir.resolve()}")
                        print(f"  • Original Training: {orig_count} | Augmented: {len(aug_imgs)} | Combined Total: {len(train_imgs)}")
                break
        except Exception:
            continue

    print(f"\n[Training Setup] Training Version: {version_tag} | Training Samples: {len(train_imgs)} | Validation Samples: {len(val_imgs)}")

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
    loss_fn = DGNetLoss(lambda_grad_loss=lambda_grad, label_smoothing=0.02)

    total_steps = epochs * len(train_ds)
    warmup_steps = min(10 * len(train_ds), total_steps // 5)
    lr_schedule = WarmupCosineDecay(learning_rate, warmup_steps, total_steps)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.0001,
        clipnorm=1.0
    )

    versioned_best_weights_path = ckpt_dir / f"dgnet_{version_tag}_best.weights.h5"
    versioned_latest_weights_path = ckpt_dir / f"dgnet_{version_tag}_latest.weights.h5"
    versioned_final_weights_path = ckpt_dir / f"dgnet_{version_tag}_final.weights.h5"
    versioned_best_model_path = ckpt_dir / f"dgnet_{version_tag}_best_model.keras"

    best_weights_path = ckpt_dir / "dgnet_best.weights.h5"
    latest_weights_path = ckpt_dir / "dgnet_latest.weights.h5"
    final_weights_path = ckpt_dir / "dgnet_final.weights.h5"
    best_model_path = ckpt_dir / "dgnet_best_model.keras"
    metrics_log_path = metrics_log_dir / "train_metrics.json"

    dummy_x = tf.zeros((1, img_size[0], img_size[1], 3))
    _ = model(dummy_x, training=False)

    explicit_weights = weights if weights else (resume if isinstance(resume, str) and resume.lower() not in ("true", "1") else None)

    if explicit_weights:
        wp = Path(explicit_weights) if Path(explicit_weights).is_absolute() else PROJECT_ROOT / explicit_weights
        if not wp.exists() or wp.stat().st_size == 0:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    from huggingface_hub import hf_hub_download
                    rel_name = "models/raw_checkpoints/" + wp.name
                    for r in [os.getenv("HF_UPLOAD_REPO_ID", "yakshithkd/COD_training_artifacts"), "yakshithkd/COD_training_artifacts", "gouravbirwaz/COD_dataset"]:
                        try:
                            dl = hf_hub_download(repo_id=r, repo_type="dataset", filename=rel_name, token=hf_token)
                            if dl and Path(dl).exists():
                                wp.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(dl, str(wp))
                                print(f"[Resume Training] Auto-downloaded '{wp.name}' from Hugging Face repo '{r}'!")
                                break
                        except Exception:
                            continue
                except Exception as e:
                    print(f"[Resume Training] Notice: Could not auto-download weights: {e}")

        if wp.exists() and wp.stat().st_size > 0:
            print(f"\n[Resume Training] Loading specified checkpoint weights: {wp.resolve()}")
            try:
                model.load_weights(str(wp))
                print(f"[Resume Training] Successfully loaded checkpoint weights from {wp.name}!")
            except Exception as load_err:
                print(f"[Resume Training] Warning: Failed to load weights from {wp.name}: {load_err}")
                print(f"[Resume Training] Proceeding with fresh initialized weights.")
        else:
            print(f"\n[Resume Training] Notice: Specified weights file not found: {wp.resolve()}. Proceeding with fresh training.")
    elif resume:
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
                w_candidates = [
                    f"dgnet_{version_tag}_latest.weights.h5",
                    f"dgnet_{version_tag}_best.weights.h5",
                    f"dgnet_{version_tag}_final.weights.h5",
                    "dgnet_benchmark_best.weights.h5",
                    "dgnet_best.weights.h5",
                ]
                for w_name in w_candidates:
                    wp = cdir / w_name
                    if wp.exists() and wp.stat().st_size > 0:
                        found_weights = wp
                        break
                if found_weights:
                    break
            except Exception:
                continue

        if found_weights:
            print(f"\n[Resume Training] Found compatible checkpoint weights: {found_weights.resolve()}")
            try:
                model.load_weights(str(found_weights))
                print(f"[Resume Training] Successfully loaded checkpoint weights from {found_weights.name}!")
            except Exception as load_err:
                print(f"[Resume Training] Warning: Failed to load weights from {found_weights.name}: {load_err}")
                print(f"[Resume Training] Starting fresh training from scratch for version '{version_tag}'.")
        else:
            print(f"\n[Resume Training] Notice: Resume requested, but no existing checkpoints found for version '{version_tag}'. Starting fresh training.")

    best_val_loss = float("inf")
    best_val_iou = 0.0
    best_val_iou_epoch = 0
    best_val_loss_epoch = 0
    early_stop_patience = 15
    epochs_without_improvement = 0

    if os.getenv("TRAINING_PLATFORM", "huggingface").lower() in ("huggingface", "hf", "true", "1") and os.getenv("HF_TOKEN"):
        try:
            from src.integration.hf_sync import download_logs
            download_logs(str(metrics_log_dir))
        except Exception as e:
            print(f"[HF Sync] Initial log sync notice: {e}")

    mlflow_tracker = MLflowTracker(
        experiment_name=experiment_name,
        enabled=enable_mlflow
    )
    mlflow_tracker.start_run(
        run_name=f"DGNet-{version_tag}-{backbone}",
        tags={
            "model_version": version_tag,
            "backbone": backbone,
            "gradient_kernel": kernel,
            "resumed": str(resume)
        },
        description=f"DGNet COD training run for version {version_tag} with {backbone} backbone."
    )

    mlflow_tracker.log_params({
        "model_name": "DGNet",
        "model_architecture": "DGNet",
        "model_version": version_tag,
        "context_backbone": backbone,
        "gradient_kernel": kernel,
        "image_size": f"[{img_size[0]}, {img_size[1]}]",
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "learning_rate_scheduler": "WarmupCosineDecay",
        "loss_function": "DGNetLoss (StructureLoss + MSE)",
        "lambda_grad_loss": lambda_grad,
        "train_samples": len(train_imgs),
        "val_samples": len(val_imgs),
    })

    history = {
        "train_total_loss": [],
        "val_total_loss": [],
        "val_iou": [],
        "val_mae": [],
        "val_dice": [],
        "val_f_measure": [],
        "val_precision": [],
        "val_recall": [],
        "val_pixel_accuracy": [],
        "val_fpr": [],
        "val_fnr": [],
    }

    print("\n" + "=" * 70)
    print(f"             Starting DGNet Real Model Training ({epochs} Epochs)")
    print(f"             Early Stopping Patience: {early_stop_patience} epochs")
    print("=" * 70)

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
                if num_batches % 10 == 0:
                    global_step = (epoch - 1) * total_train_batches + num_batches
                    mlflow_tracker.log_metrics({
                        "batch_train_loss": loss_val,
                        "running_avg_train_loss": train_loss_sum / num_batches
                    }, step=global_step)
        else:
            print(f"---> Epoch [{epoch:02d}/{epochs:02d}] Training on {len(train_imgs)} samples...", flush=True)
            for x_batch, y_mask, y_grad in train_ds:
                loss = train_step(x_batch, y_mask, y_grad)
                loss_val = float(loss)
                train_loss_sum += loss_val
                num_batches += 1
                if num_batches == 1 or num_batches % 5 == 0 or num_batches == total_train_batches:
                    print(f"  Batch [{num_batches:03d}/{total_train_batches:03d}] | Loss: {loss_val:.4f} | Avg Loss: {train_loss_sum / num_batches:.4f}", flush=True)
                if num_batches % 10 == 0:
                    global_step = (epoch - 1) * total_train_batches + num_batches
                    mlflow_tracker.log_metrics({
                        "batch_train_loss": loss_val,
                        "running_avg_train_loss": train_loss_sum / num_batches
                    }, step=global_step)

        avg_train_loss = train_loss_sum / num_batches if num_batches > 0 else 0.0

        val_loss_sum = 0.0
        val_batches = 0
        metrics_acc = CODMetricsAccumulator(threshold=0.5, beta_sq=0.3)

        if use_tqdm:
            val_pbar = tqdm(val_ds, total=total_val_batches, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]  ", leave=False)
            for x_val, y_val_mask, y_val_grad in val_pbar:
                v_loss, prob_mask = val_step(x_val, y_val_mask, y_val_grad)
                val_loss_sum += float(v_loss)
                val_batches += 1
                metrics_acc.update(y_val_mask, prob_mask)
        else:
            print(f"---> Epoch [{epoch:02d}/{epochs:02d}] Evaluating on {len(val_imgs)} validation samples...", flush=True)
            for x_val, y_val_mask, y_val_grad in val_ds:
                v_loss, prob_mask = val_step(x_val, y_val_mask, y_val_grad)
                val_loss_sum += float(v_loss)
                val_batches += 1
                metrics_acc.update(y_val_mask, prob_mask)
                if val_batches == 1 or val_batches % 10 == 0 or val_batches == total_val_batches:
                    print(f"  Val Batch [{val_batches:02d}/{total_val_batches:02d}] | Val Loss: {val_loss_sum / val_batches:.4f}", flush=True)

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
        epoch_metrics = metrics_acc.result()

        epoch_time = time.time() - epoch_start
        print(f"\n[Epoch {epoch:02d}/{epochs:02d} Finished in {epoch_time:.1f}s] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {epoch_metrics['iou']:.4f}\n", flush=True)

        try:
            with open("/tmp/training_status.txt", "w", encoding="utf-8") as sf:
                sf.write(f"Epoch {epoch:02d}/{epochs:02d} in progress | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {epoch_metrics['iou']:.4f}")
        except Exception:
            pass

        history["train_total_loss"].append(avg_train_loss)
        history["val_total_loss"].append(avg_val_loss)
        history["val_iou"].append(epoch_metrics["iou"])
        history["val_mae"].append(epoch_metrics["mae"])
        history["val_dice"].append(epoch_metrics["dice"])
        history["val_f_measure"].append(epoch_metrics["f_measure"])
        history["val_precision"].append(epoch_metrics["precision"])
        history["val_recall"].append(epoch_metrics["recall"])
        history["val_pixel_accuracy"].append(epoch_metrics["pixel_accuracy"])
        history["val_fpr"].append(epoch_metrics["fpr"])
        history["val_fnr"].append(epoch_metrics["fnr"])

        try:
            model.save_weights(str(versioned_latest_weights_path))
            model.save_weights(str(latest_weights_path))
            if epoch % 5 == 0 or epoch == epochs:
                epoch_ckpt_path = ckpt_dir / f"dgnet_{version_tag}_epoch_{epoch:03d}.weights.h5"
                model.save_weights(str(epoch_ckpt_path))
        except Exception as e:
            print(f"  Notice (Checkpoint Save): {e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_loss_epoch = epoch

        saved_str = ""
        if epoch_metrics["iou"] > best_val_iou:
            best_val_iou = epoch_metrics["iou"]
            best_val_iou_epoch = epoch
            epochs_without_improvement = 0
            try:
                model.save_weights(str(versioned_best_weights_path))
                model.save(str(versioned_best_model_path))
                model.save_weights(str(best_weights_path))
                model.save(str(best_model_path))
            except Exception:
                pass
            saved_str = f" -> Best IoU Checkpoint ({version_tag}, IoU={epoch_metrics['iou']:.4f}) Saved!"
        else:
            epochs_without_improvement += 1

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{saved_str}", flush=True)
        print(f"  -> Pixel Acc: {epoch_metrics['pixel_accuracy']*100:.2f}% | IoU: {epoch_metrics['iou']:.4f} | Dice: {epoch_metrics['dice']:.4f} | MAE: {epoch_metrics['mae']:.4f}", flush=True)

        with open(metrics_log_path, "w") as f:
            json.dump(history, f, indent=4)

        mlflow_tracker.log_metrics({
            "train_total_loss": avg_train_loss,
            "val_total_loss": avg_val_loss,
            "val_iou": epoch_metrics["iou"],
            "val_mae": epoch_metrics["mae"],
            "val_dice": epoch_metrics["dice"],
            "val_f_measure": epoch_metrics["f_measure"],
            "val_precision": epoch_metrics["precision"],
            "val_recall": epoch_metrics["recall"],
            "val_pixel_accuracy": epoch_metrics["pixel_accuracy"],
            "val_fpr": epoch_metrics["fpr"],
            "val_fnr": epoch_metrics["fnr"],
        }, step=epoch)

        if os.getenv("TRAINING_PLATFORM", "huggingface").lower() in ("huggingface", "hf", "true", "1") or os.getenv("HF_TOKEN"):
            try:
                from src.integration.hf_sync import upload_logs, upload_checkpoints
                upload_logs()
                if saved_str or epoch % 5 == 0 or epoch == epochs:
                    upload_checkpoints()
            except Exception:
                pass

        if epochs_without_improvement >= early_stop_patience:
            print(f"\n[Early Stopping] No improvement in val loss for {early_stop_patience} epochs. Stopping training.")
            print(f"[Early Stopping] Best val loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}")
            break

    try:
        model.save_weights(str(versioned_final_weights_path))
        model.save_weights(str(final_weights_path))
        if not versioned_best_weights_path.exists() and versioned_latest_weights_path.exists():
            shutil.copy2(str(versioned_latest_weights_path), str(versioned_best_weights_path))
        if not best_weights_path.exists() and latest_weights_path.exists():
            shutil.copy2(str(latest_weights_path), str(best_weights_path))
    except Exception as e:
        print(f"[Checkpoint Save] Note: {e}")

    fallback_ckpt_dir = PROJECT_ROOT / "models/raw_checkpoints"
    if ckpt_dir.resolve() != fallback_ckpt_dir.resolve():
        fallback_ckpt_dir.mkdir(parents=True, exist_ok=True)
        for w_file in [versioned_latest_weights_path, versioned_best_weights_path, versioned_final_weights_path, versioned_best_model_path, latest_weights_path, best_weights_path, final_weights_path, best_model_path]:
            if w_file.exists():
                try:
                    shutil.copy2(str(w_file), str(fallback_ckpt_dir / w_file.name))
                except Exception:
                    pass

    total_training_time = time.time() - start_train_time
    final_epoch = len(history["train_total_loss"])
    best_dice = max(history["val_dice"]) if history["val_dice"] else 0.0
    best_dice_epoch = (history["val_dice"].index(best_dice) + 1) if history["val_dice"] else 0
    best_fm = max(history["val_f_measure"]) if history["val_f_measure"] else 0.0
    best_fm_epoch = (history["val_f_measure"].index(best_fm) + 1) if history["val_f_measure"] else 0
    best_acc = max(history["val_pixel_accuracy"]) if history["val_pixel_accuracy"] else 0.0
    best_acc_epoch = (history["val_pixel_accuracy"].index(best_acc) + 1) if history["val_pixel_accuracy"] else 0

    print("\n" + "=" * 70)
    print("            FINAL TRAINING REPORT - DGNet COD")
    print("=" * 70)
    print(f"  Training Time:          {total_training_time / 60.0:.2f} minutes ({final_epoch} epochs)")
    print(f"  Model Version:          {version_tag}")
    print(f"  Backbone:               {backbone}")
    print("-" * 70)
    print(f"  Best Val Loss:          {best_val_loss:.4f} (epoch {best_val_loss_epoch})")
    print(f"  Best IoU:               {best_val_iou:.4f} (epoch {best_val_iou_epoch})")
    print(f"  Best Dice:              {best_dice:.4f} (epoch {best_dice_epoch})")
    print(f"  Best F-measure:         {best_fm:.4f} (epoch {best_fm_epoch})")
    print(f"  Best Pixel Accuracy:    {best_acc*100:.2f}% (epoch {best_acc_epoch})")
    print("-" * 70)
    print(f"  Final Train Loss:       {history['train_total_loss'][-1]:.4f}")
    print(f"  Final Val Loss:         {history['val_total_loss'][-1]:.4f}")
    print(f"  Final MAE:              {history['val_mae'][-1]:.4f}")
    print("-" * 70)
    print(f"  Checkpoint Location:    {ckpt_dir.resolve()}")
    print("=" * 70 + "\n")

    history["final_summary"] = {
        "training_time_minutes": round(total_training_time / 60.0, 2),
        "total_epochs_run": final_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_iou": round(best_val_iou, 6),
        "best_val_iou_epoch": best_val_iou_epoch,
        "best_dice": round(best_dice, 6),
        "best_f_measure": round(best_fm, 6),
        "best_pixel_accuracy": round(best_acc, 6),
        "final_train_loss": round(history['train_total_loss'][-1], 6),
        "final_val_loss": round(history['val_total_loss'][-1], 6),
        "final_mae": round(history['val_mae'][-1], 6),
        "early_stopped": epochs_without_improvement >= early_stop_patience,
    }

    with open(metrics_log_path, "w") as f:
        json.dump(history, f, indent=4)

    best_weights_resolved = versioned_best_weights_path if versioned_best_weights_path.exists() else best_weights_path
    mlflow_tracker.log_checkpoint_metadata(
        best_weights_path=str(best_weights_resolved) if best_weights_resolved.exists() else None,
        best_epoch=best_val_loss_epoch,
        best_val_loss=best_val_loss,
        best_val_iou=best_val_iou,
        best_val_dice=best_dice,
    )
    mlflow_tracker.log_metrics({
        "final_train_loss": float(history['train_total_loss'][-1]) if history['train_total_loss'] else 0.0,
        "final_val_loss": float(history['val_total_loss'][-1]) if history['val_total_loss'] else 0.0,
        "best_val_loss": float(best_val_loss),
        "best_val_iou": float(best_val_iou),
        "best_dice": float(best_dice),
        "best_f_measure": float(best_fm),
        "best_pixel_accuracy": float(best_acc),
        "training_time_minutes": float(round(total_training_time / 60.0, 2)),
    })
    if metrics_log_path.exists():
        mlflow_tracker.log_artifact(str(metrics_log_path.resolve()), artifact_path="metrics")

    mlflow_tracker.log_model_summary(model)
    mlflow_tracker.log_model(model, artifact_path="model", registered_model_name="DGNet-Camouflage-Detector")
    mlflow_tracker.log_plots(str(metrics_log_dir))
    mlflow_tracker.log_quantized_models()
    mlflow_tracker.end_run(status="FINISHED")

    if os.getenv("TRAINING_PLATFORM", "huggingface").lower() in ("huggingface", "hf", "true", "1") and os.getenv("HF_TOKEN"):
        try:
            from src.integration.hf_sync import upload_logs, upload_checkpoints
            upload_logs(str(metrics_log_dir))
            upload_checkpoints(str(ckpt_dir))
        except Exception as e:
            print(f"[HF Sync] Final upload notice: {e}")

    return history

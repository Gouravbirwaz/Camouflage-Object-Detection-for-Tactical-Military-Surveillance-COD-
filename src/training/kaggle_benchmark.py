"""
DGNet Kaggle Camouflage Benchmark Training & Pretraining Runner.
Supports training on public camouflage benchmarks (COD10K, NC4K, CAMO) from Kaggle
with automatic downloading via kagglehub, warm-up scheduling, and checkpoint synchronization.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.utils.config import get_project_root, load_project_config
from src.utils.metrics import CODMetricsAccumulator
from src.models.dgnet import DGNet, DGNetLoss, build_dgnet_model
from src.data.dataset import create_tf_dataset
from src.training.trainer import WarmupCosineDecay
from src.integration.hf_job_launcher import generate_benchmark_job_command

PROJECT_ROOT = get_project_root()

# Public Kaggle Datasets mapping from dataset.txt
KAGGLE_DATASETS = {
    "cod10k": "ivanomelchenkoim11/cod10k-dataset",
    "nc4k": "ivanomelchenkoim11/nc4k-dataset",
    "camo": "danhnt/camo-fs-dataset",
}


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
    """Recursively scans directory hierarchy to pair RGB camouflage images with masks."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = list(data_dir.rglob("*"))

    img_candidates = {}
    gt_candidates = {}

    for f in all_files:
        if not f.is_file() or f.suffix.lower() not in valid_exts:
            continue
        p_str = str(f).replace("\\", "/")
        stem = f.stem

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


def upload_checkpoints_to_hf(repo_id: Optional[str] = None, ckpt_dir: Optional[Path] = None) -> bool:
    """Uploads the single best benchmark checkpoint to Hugging Face Hub."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[HF Upload] Notice: HF_TOKEN not set. Skipping Hugging Face upload.")
        return False

    target_repo = repo_id or os.getenv("HF_UPLOAD_REPO_ID", os.getenv("HF_REPO_ID", "yakshithkd/COD_training_artifacts"))
    c_dir = ckpt_dir or (PROJECT_ROOT / "models/raw_checkpoints")
    best_ckpt = c_dir / "dgnet_benchmark_best.weights.h5"

    if not best_ckpt.exists():
        print(f"[HF Upload] Best checkpoint not found at: {best_ckpt}")
        return False

    print(f"\n[HF Upload] Uploading best checkpoint '{best_ckpt.name}' to '{target_repo}'...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token.strip())
        api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(best_ckpt),
            path_in_repo="models/raw_checkpoints/dgnet_benchmark_best.weights.h5",
            repo_id=target_repo,
            repo_type="dataset"
        )
        print(f"✓ SUCCESS: Best checkpoint uploaded to Hugging Face: {best_ckpt.name}")
        return True
    except Exception as e:
        print(f"Notice (HF Upload): {e}")
        return False


def download_checkpoints_from_hf(repo_id: Optional[str] = None, target_dir: Optional[Path] = None) -> bool:
    """Downloads the benchmark checkpoint from Hugging Face Hub into local repository."""
    t_dir = target_dir or (PROJECT_ROOT / "models/raw_checkpoints")
    t_dir.mkdir(parents=True, exist_ok=True)
    local_dest = t_dir / "dgnet_benchmark_best.weights.h5"
    token = os.getenv("HF_TOKEN", None)

    candidate_repos = []
    if repo_id:
        candidate_repos.append(repo_id)
    if os.getenv("HF_UPLOAD_REPO_ID"):
        candidate_repos.append(os.getenv("HF_UPLOAD_REPO_ID"))
    candidate_repos.extend(["yakshithkd/COD_training_artifacts", "gouravbirwaz/COD_dataset"])
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
                print("=" * 70)
                return True
        except Exception as e:
            continue

    print(f"\n ⚠ Checkpoint was not found across {candidate_repos}")
    return False


def run_benchmark_training(
    dataset_name: str = "cod10k",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    backbone: str = "mobilenet_v3_large",
    checkpoint_dir: Optional[Path] = None,
    upload_on_finish: bool = False,
    repo_id: Optional[str] = None
):
    """Executes pretraining loop on Kaggle benchmark datasets."""
    checkpoint_dir = checkpoint_dir or (PROJECT_ROOT / "models/raw_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = checkpoint_dir / "dgnet_benchmark_best.weights.h5"

    print("=" * 70)
    print(f" DGNet Pretraining on Kaggle Benchmark: {dataset_name.upper()}")
    print("=" * 70)
    print(f" Backbone:        {backbone}")
    print(f" Epochs:          {epochs}")
    print(f" Batch Size:      {batch_size}")
    print(f" Learning Rate:   {learning_rate}")
    print(f" Checkpoints:     {checkpoint_dir}")
    print("=" * 70)

    train_imgs, train_masks, val_imgs, val_masks = get_dataset_splits(dataset_name)

    train_ds = create_tf_dataset(train_imgs, train_masks, image_size=(384, 384), batch_size=batch_size, is_train=True)
    val_ds = create_tf_dataset(val_imgs, val_masks, image_size=(384, 384), batch_size=batch_size, is_train=False)

    print("\n[Model] Instantiating DGNet architecture...")
    model = build_dgnet_model(backbone_name=backbone, input_shape=(384, 384, 3), channel=64, pretrained=True)
    _ = model(tf.zeros((1, 384, 384, 3), dtype=tf.float32), training=False)

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

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    print("\n[Training] Starting benchmark pretraining loop...")
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_weights(str(best_weights_path))
            print(f"  ★ New Best Weights Saved: {best_weights_path.name} (Val Loss: {best_val_loss:.4f}, IoU: {val_iou:.4f})")
            if upload_on_finish or os.getenv("HF_TOKEN"):
                upload_checkpoints_to_hf(repo_id=repo_id, ckpt_dir=checkpoint_dir)

    if upload_on_finish or os.getenv("HF_TOKEN"):
        upload_checkpoints_to_hf(repo_id=repo_id, ckpt_dir=checkpoint_dir)

    print("\n" + "=" * 70)
    print(" ✓ PRETRAINING COMPLETE!")
    print(f" Single Best Checkpoint Saved: {best_weights_path.resolve()}")
    print("=" * 70)
    return history


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="DGNet Kaggle Camouflage Benchmark Training")
    parser.add_argument("--train", action="store_true", help="Start benchmark pretraining")
    parser.add_argument("--dataset", type=str, default="cod10k", choices=["cod10k", "nc4k", "camo", "all"], help="Dataset to pretrain on")
    parser.add_argument("--epochs", type=int, default=30, help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large", help="Backbone encoder")
    parser.add_argument("--upload-checkpoints", action="store_true", help="Upload trained best checkpoint to HF")
    parser.add_argument("--download-checkpoints", action="store_true", help="Download trained checkpoint from HF")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face repo ID")

    args = parser.parse_args(argv)

    if args.download_checkpoints:
        download_checkpoints_from_hf(repo_id=args.repo_id)
        return

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


if __name__ == "__main__":
    main()

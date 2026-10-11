"""
Dataset Loading, Split Discovery, and TensorFlow tf.data Pipeline for COD.
Provides high-throughput streaming dataset pipelines with military tactical data augmentations,
Canny object gradient extraction, and cloud/local path resolution.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
# Prevent threadpool contention and deadlock between OpenCV and TensorFlow tf.data workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import tensorflow as tf

from src.utils.config import get_project_root
from src.data.augmentation import tf_random_military_fog_smoke, tf_random_thermal_noise

PROJECT_ROOT = get_project_root()
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


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
                            if f.is_file() and f.suffix.lower() in valid_exts and f.stat().st_size > 1024:
                                img_files[f.stem] = str(f.resolve())
                        except (OSError, PermissionError):
                            continue
                except (OSError, PermissionError):
                    pass

                try:
                    for f in mask_dir.glob("*"):
                        try:
                            if f.is_file() and f.suffix.lower() in valid_exts and f.stat().st_size > 1024:
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
                    if f.is_file() and f.suffix.lower() in valid_exts and f.stat().st_size > 1024:
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


def discover_dataset_pairs(data_dir: Union[str, Path]) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Scans candidate data directories and auto-discovers matched RGB images and GT mask pairs.
    Handles existing splits (Training, Testing) or raw image/mask folders.
    Supports mounted Hugging Face Storage Buckets / S3 mounts (/data, /mnt/data, etc.).
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    req_path = Path(data_dir)
    candidate_paths = []

    # Priority candidate paths (original raw data only — augmented data is merged separately in training)
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
            subprocess.run(["dvc", "config", "cache.dir", str(found_dvc_store.resolve())], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "config", "cache.type", "copy"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "checkout", "data/raw.dvc"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "checkout", "data/augmented.dvc"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "checkout"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "pull", "data/raw.dvc"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["dvc", "pull", "data/augmented.dvc"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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


def compute_object_gradient_np(image_uint8: np.ndarray, mask_uint8: np.ndarray) -> np.ndarray:
    """
    Computes official DGNet object-level gradient supervision map (MIR 2023 / dataset/Img2ObjGrad.py):
    grad = cv2.Canny(image, 64, 128) * (mask >= 128)
    Captures fine boundary and internal structural cues of the camouflaged object.
    """
    if image_uint8.ndim == 4:
        image_uint8 = image_uint8[0]
    if mask_uint8.ndim == 3 and mask_uint8.shape[-1] == 1:
        mask_uint8 = mask_uint8[:, :, 0]
    canny = cv2.Canny(image_uint8, 64, 128)
    fore = ((canny > 0) & (mask_uint8 >= 128)).astype(np.float32)
    return fore[:, :, np.newaxis]


def parse_sample(img_path: tf.Tensor, mask_path: tf.Tensor, image_size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Reads, decodes, resizes float image [0, 1], and extracts official Canny object gradient safely."""
    def _py_load(ip_bytes, mp_bytes, h, w):
        try:
            ip_str = ip_bytes.decode("utf-8") if isinstance(ip_bytes, bytes) else str(ip_bytes)
            mp_str = mp_bytes.decode("utf-8") if isinstance(mp_bytes, bytes) else str(mp_bytes)

            im_bgr = cv2.imread(ip_str)
            if im_bgr is not None:
                im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                im_res = cv2.resize(im_rgb, (w, h))
            else:
                im_res = np.zeros((h, w, 3), dtype=np.uint8)

            m_raw = cv2.imread(mp_str, cv2.IMREAD_GRAYSCALE)
            if m_raw is not None:
                m_res = cv2.resize(m_raw, (w, h), interpolation=cv2.INTER_NEAREST)
                m_bin = (m_res >= 128).astype(np.float32)[:, :, np.newaxis]
            else:
                m_res = np.zeros((h, w), dtype=np.uint8)
                m_bin = np.zeros((h, w, 1), dtype=np.float32)

            canny = cv2.Canny(im_res, 64, 128)
            fore = ((canny > 0) & (m_res >= 128)).astype(np.float32)[:, :, np.newaxis]
            im_f = im_res.astype(np.float32) / 255.0

            return im_f, m_bin, fore
        except Exception:
            return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w, 1), dtype=np.float32), np.zeros((h, w, 1), dtype=np.float32)

    image_float, mask_bin, grad_gt = tf.numpy_function(
        _py_load,
        [img_path, mask_path, image_size[0], image_size[1]],
        [tf.float32, tf.float32, tf.float32]
    )
    image_float.set_shape([image_size[0], image_size[1], 3])
    mask_bin.set_shape([image_size[0], image_size[1], 1])
    grad_gt.set_shape([image_size[0], image_size[1], 1])

    return image_float, mask_bin, grad_gt


def _random_resized_crop(img, mask, grad_gt, target_size, scale_min=0.5, scale_max=1.0, ratio_min=0.75, ratio_max=1.33):
    """
    Random Resized Crop: Crops a random sub-region at random scale/aspect ratio,
    then resizes back to target_size. Applied identically to image, mask, and gradient GT.
    Critical augmentation for multi-scale learning on small datasets.
    """
    h, w = target_size
    img_h = tf.shape(img)[0]
    img_w = tf.shape(img)[1]

    # Random scale factor
    scale = tf.random.uniform((), minval=scale_min, maxval=scale_max)
    # Random aspect ratio
    log_ratio = tf.random.uniform((), minval=tf.math.log(ratio_min), maxval=tf.math.log(ratio_max))
    aspect = tf.exp(log_ratio)

    crop_h = tf.cast(tf.cast(img_h, tf.float32) * tf.sqrt(scale / aspect), tf.int32)
    crop_w = tf.cast(tf.cast(img_w, tf.float32) * tf.sqrt(scale * aspect), tf.int32)
    crop_h = tf.minimum(crop_h, img_h)
    crop_w = tf.minimum(crop_w, img_w)

    offset_h = tf.random.uniform((), minval=0, maxval=img_h - crop_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform((), minval=0, maxval=img_w - crop_w + 1, dtype=tf.int32)

    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, crop_h, crop_w)
    mask = tf.image.crop_to_bounding_box(mask, offset_h, offset_w, crop_h, crop_w)
    grad_gt = tf.image.crop_to_bounding_box(grad_gt, offset_h, offset_w, crop_h, crop_w)

    img = tf.image.resize(img, [h, w])
    mask = tf.image.resize(mask, [h, w], method="nearest")
    grad_gt = tf.image.resize(grad_gt, [h, w], method="nearest")

    return img, mask, grad_gt


def _gaussian_blur_tf(img, sigma_min=0.1, sigma_max=2.0, kernel_size=7):
    """
    Random Gaussian blur applied only to the image (not mask).
    Teaches model to handle blurred/unfocused edges common in real tactical imagery.
    """
    sigma = tf.random.uniform((), minval=sigma_min, maxval=sigma_max)
    half = kernel_size // 2
    x = tf.cast(tf.range(-half, half + 1), tf.float32)
    kernel_1d = tf.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)

    kernel_h = tf.reshape(kernel_1d, [kernel_size, 1, 1, 1])
    kernel_w = tf.reshape(kernel_1d, [1, kernel_size, 1, 1])

    channels = tf.split(img, 3, axis=-1)
    blurred = []
    for ch in channels:
        ch = tf.expand_dims(ch, 0)
        ch = tf.nn.conv2d(ch, kernel_h, strides=[1, 1, 1, 1], padding="SAME")
        ch = tf.nn.conv2d(ch, kernel_w, strides=[1, 1, 1, 1], padding="SAME")
        blurred.append(ch[0])
    return tf.concat(blurred, axis=-1)


def _cutout_tf(img, max_fraction=0.25):
    """
    CutOut / Random Erasing: Zeros out a random rectangular patch (up to 25% of area).
    Prevents model from relying on single-region cues and reduces overfitting.
    Applied only to image, not mask or gradient GT.
    """
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    cut_h = tf.cast(tf.cast(h, tf.float32) * tf.random.uniform((), 0.05, max_fraction), tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * tf.random.uniform((), 0.05, max_fraction), tf.int32)
    cy = tf.random.uniform((), minval=0, maxval=h, dtype=tf.int32)
    cx = tf.random.uniform((), minval=0, maxval=w, dtype=tf.int32)

    y1 = tf.maximum(cy - cut_h // 2, 0)
    y2 = tf.minimum(cy + cut_h // 2, h)
    x1 = tf.maximum(cx - cut_w // 2, 0)
    x2 = tf.minimum(cx + cut_w // 2, w)

    rows = tf.range(h)
    cols = tf.range(w)
    row_mask = tf.logical_or(rows < y1, rows >= y2)
    col_mask = tf.logical_or(cols < x1, cols >= x2)
    keep_mask = tf.logical_or(
        tf.expand_dims(row_mask, 1),
        tf.expand_dims(col_mask, 0)
    )
    keep_mask = tf.cast(keep_mask, tf.float32)
    keep_mask = tf.expand_dims(keep_mask, -1)

    return img * keep_mask


def create_tf_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    image_size: Tuple[int, int] = (384, 384),
    batch_size: int = 8,
    is_train: bool = True
) -> tf.data.Dataset:
    """Builds a high-throughput streaming TensorFlow tf.data.Dataset pipeline with low memory footprint."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    def _map_fn(img_p, mask_p):
        return parse_sample(img_p, mask_p, image_size=image_size)

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 256), seed=42)

        def _augment_fn(img, mask, grad_gt):
            # 1. Random Resized Crop (50% probability)
            if tf.random.uniform(()) > 0.5:
                img, mask, grad_gt = _random_resized_crop(img, mask, grad_gt, image_size, scale_min=0.5, scale_max=1.0)

            # 2. Random horizontal flip
            if tf.random.uniform(()) > 0.5:
                img = tf.image.flip_left_right(img)
                mask = tf.image.flip_left_right(mask)
                grad_gt = tf.image.flip_left_right(grad_gt)

            # 3. Random vertical flip
            if tf.random.uniform(()) > 0.5:
                img = tf.image.flip_up_down(img)
                mask = tf.image.flip_up_down(mask)
                grad_gt = tf.image.flip_up_down(grad_gt)

            # 4. Random 90-degree rotation
            k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k=k)
            mask = tf.image.rot90(mask, k=k)
            grad_gt = tf.image.rot90(grad_gt, k=k)

            # 5. Fog / Smoke Screen Simulation (30% probability)
            if tf.random.uniform(()) > 0.7:
                img = tf_random_military_fog_smoke(img)

            # 6. Thermal & Electro-Optical Sensor Noise (30% probability)
            if tf.random.uniform(()) > 0.7:
                img = tf_random_thermal_noise(img)

            # 7. Color Jitter & Illumination Shift
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
            img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
            img = tf.image.random_hue(img, max_delta=0.05)
            img = tf.clip_by_value(img, 0.0, 1.0)

            # 8. Gaussian Blur / Heat Defocus (35% probability)
            if tf.random.uniform(()) > 0.65:
                img = _gaussian_blur_tf(img, sigma_min=0.3, sigma_max=1.8)
                img = tf.clip_by_value(img, 0.0, 1.0)

            # 9. CutOut / Foliage Mesh Occlusion (40% probability)
            if tf.random.uniform(()) > 0.6:
                img = _cutout_tf(img, max_fraction=0.25)

            # Apply ImageNet normalization at the end
            img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD
            return img_norm, mask, grad_gt

        dataset = dataset.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def _norm_fn(img, mask, grad_gt):
            img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD
            return img_norm, mask, grad_gt

        dataset = dataset.map(_norm_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

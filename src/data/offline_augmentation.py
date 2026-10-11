"""
Military Tactical Offline Dataset Generator - Camouflaged Object Detection (COD)
Reads original image-mask training pairs from raw dataset splits and generates
multiple realistic military tactical augmented variants (fog/smoke, thermal noise, motion blur,
mesh net cutout, dusk shadows, rain/dust streaks).

Output directory: data/augmented/military_tactical/Training/
  ├── images/
  └── GT/
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config import get_project_root
from src.data.augmentation import (
    apply_fog_smoke,
    apply_full_military_tactical_pipeline,
    apply_mesh_foliage_cutout,
    apply_motion_blur_defocus,
    apply_rain_dust_streaks,
    apply_tactical_lighting_shift,
    apply_thermal_sensor_noise,
)

PROJECT_ROOT = get_project_root()


def discover_raw_training_pairs(raw_dir: Union[str, Path]) -> Tuple[List[Path], List[Path]]:
    """Scans dataset directory for matched image and ground truth mask files."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    raw_path = Path(raw_dir) if Path(raw_dir).is_absolute() else PROJECT_ROOT / raw_dir

    candidate_dirs = [
        raw_path / "dataset-splitM" / "Training",
        raw_path / "Training",
        raw_path,
        PROJECT_ROOT / "data" / "raw" / "dataset-splitM" / "Training",
        PROJECT_ROOT / "data" / "raw",
    ]

    target_train_dir = None
    for cdir in candidate_dirs:
        if cdir.exists() and cdir.is_dir():
            target_train_dir = cdir
            break

    if not target_train_dir:
        raise FileNotFoundError(f"Could not locate training dataset directory under: {raw_dir}")

    img_dir, mask_dir = None, None
    for child in target_train_dir.iterdir():
        if child.is_dir():
            cname = child.name.lower()
            if any(k in cname for k in ["image", "images", "img", "rgb"]):
                img_dir = child
            elif any(k in cname for k in ["gt", "mask", "masks", "label", "groundtruth"]):
                mask_dir = child

    if not img_dir or not mask_dir:
        img_files = {f.stem: f for f in target_train_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts and any(k in str(f.parent).lower() for k in ["image", "images", "img"])}
        mask_files = {f.stem: f for f in target_train_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts and any(k in str(f.parent).lower() for k in ["gt", "mask", "masks", "label"])}
    else:
        img_files = {f.stem: f for f in img_dir.glob("*") if f.is_file() and f.suffix.lower() in valid_exts}
        mask_files = {f.stem: f for f in mask_dir.glob("*") if f.is_file() and f.suffix.lower() in valid_exts}

    common_stems = sorted(list(set(img_files.keys()).intersection(set(mask_files.keys()))))
    paired_imgs = [img_files[s] for s in common_stems]
    paired_masks = [mask_files[s] for s in common_stems]

    print(f"[Dataset Discovery] Found {len(paired_imgs)} paired training images in: {target_train_dir.resolve()}")
    return paired_imgs, paired_masks


def generate_augmented_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/augmented/military_tactical",
    multiplier: int = 4
) -> int:
    """
    Reads original dataset pairs and generates 'multiplier' tactical military augmented variants per image.
    Expands a 748 sample training set into 2,992+ matched image-mask pairs.

    Returns:
        Total count of generated image-mask pairs.
    """
    start_time = time.time()
    raw_path = Path(raw_dir) if Path(raw_dir).is_absolute() else PROJECT_ROOT / raw_dir
    out_path = Path(output_dir) if Path(output_dir).is_absolute() else PROJECT_ROOT / output_dir

    out_img_dir = out_path / "Training" / "images"
    out_gt_dir = out_path / "Training" / "GT"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    imgs, masks = discover_raw_training_pairs(raw_path)
    if not imgs:
        print("[ERROR] No training pairs found to augment.")
        return 0

    print("\n" + "=" * 70)
    print("     Military Tactical Dataset Expansion & Generator for COD")
    print("=" * 70)
    print(f"  Raw Training Samples:    {len(imgs)}")
    print(f"  Augmentation Multiplier: {multiplier}x")
    print(f"  Target Expanded Size:    ~{len(imgs) * (multiplier + 1)} paired samples")
    print(f"  Output Directory:        {out_path.resolve()}")
    print("-" * 70 + "\n")

    generated_count = 0
    pbar = tqdm(zip(imgs, masks), total=len(imgs), desc="Generating Military Data", unit="image")

    for img_p, mask_p in pbar:
        stem = img_p.stem

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            continue

        # 1. Save original copy into augmented dataset
        orig_out_img = out_img_dir / f"{stem}_orig.jpg"
        orig_out_gt = out_gt_dir / f"{stem}_orig.png"
        cv2.imwrite(str(orig_out_img), img_bgr)
        cv2.imwrite(str(orig_out_gt), mask_gray)
        generated_count += 1

        # 2. Generate 'multiplier' military tactical variants
        tactical_variants = [
            ("fog_lighting", lambda i, m: (apply_tactical_lighting_shift(apply_fog_smoke(i, 0.45)), m)),
            ("thermal_noise", lambda i, m: (apply_thermal_sensor_noise(apply_motion_blur_defocus(i)), m)),
            ("mesh_foliage", lambda i, m: apply_mesh_foliage_cutout(i, m)),
            ("rain_dusk", lambda i, m: (apply_rain_dust_streaks(apply_tactical_lighting_shift(i)), m)),
            ("full_tactical", lambda i, m: apply_full_military_tactical_pipeline(i, m, augmentation_prob=1.0)),
        ]

        for v_idx in range(multiplier):
            variant_name, aug_fn = tactical_variants[v_idx % len(tactical_variants)]
            aug_rgb, aug_mask = aug_fn(img_rgb.copy(), mask_gray.copy())

            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

            out_i_path = out_img_dir / f"{stem}_mil_{variant_name}_{v_idx+1}.jpg"
            out_g_path = out_gt_dir / f"{stem}_mil_{variant_name}_{v_idx+1}.png"

            cv2.imwrite(str(out_i_path), aug_bgr)
            cv2.imwrite(str(out_g_path), aug_mask)
            generated_count += 1

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("     Military Tactical Dataset Generation Complete!")
    print("=" * 70)
    print(f"  Total Samples Created:  {generated_count} paired image-mask files")
    print(f"  Time Elapsed:           {total_time:.2f} seconds ({total_time / 60.0:.2f} min)")
    print(f"  Images Directory:       {out_img_dir.resolve()}")
    print(f"  Masks Directory:        {out_gt_dir.resolve()}")
    print("=" * 70 + "\n")

    return generated_count


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Military Tactical Augmented Dataset for DGNet COD")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/augmented/military_tactical", help="Path to save augmented dataset")
    parser.add_argument("--multiplier", type=int, default=4, help="Number of military augmented variants per image (default: 4)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_arguments(argv)
    generate_augmented_dataset(raw_dir=args.raw_dir, output_dir=args.output_dir, multiplier=args.multiplier)


if __name__ == "__main__":
    main()

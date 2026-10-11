"""
Convert Ground Truth (GT) mask images from JPEG (.jpg) to lossless PNG (.png) format.

This module processes ground truth binary segmentation masks stored as lossy JPEGs
(which commonly exhibit DCT ringing and edge artifacts), cleans them via strict binary
re-thresholding (>= 128 -> 255, else 0), saves them as lossless PNG files, and removes
the original JPEG files.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from src.utils.config import get_project_root

PROJECT_ROOT = get_project_root()

# Default target directories relative to project root
DEFAULT_TARGET_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "dataset-splitM" / "Training" / "GT",
    PROJECT_ROOT / "data" / "raw" / "dataset-splitM" / "Testing" / "GT",
]

DEFAULT_BACKUP_DIR = PROJECT_ROOT / "data" / "backup" / "gt_jpg_backup"


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure a console logger with formatted output."""
    logger = logging.getLogger("mask_converter")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def find_jpg_files(directory: Path) -> List[Path]:
    """Find all .jpg and .jpeg files (case-insensitive) in the given directory."""
    if not directory.exists() or not directory.is_dir():
        return []

    jpg_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg"}:
            jpg_files.append(file_path)

    return sorted(jpg_files, key=lambda p: p.name)


def convert_single_image(
    jpg_path: Path,
    output_png_path: Path,
    threshold_value: int = 128,
    dry_run: bool = False,
    backup: bool = False,
    backup_path: Optional[Path] = None,
    delete_original: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Read a JPEG ground truth mask, threshold it to clean binary format, save as PNG,
    and safely delete the original JPEG.
    """
    if dry_run:
        action_desc = f"[DRY-RUN] Would convert: {jpg_path.name} -> {output_png_path.name}"
        if backup and backup_path:
            action_desc += f" (Backup to: {backup_path})"
        if delete_original:
            action_desc += " (Delete original)"
        if logger:
            logger.debug(action_desc)
        return True, "DRY_RUN"

    try:
        img = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, f"Failed to read image (corrupt or unreadable): {jpg_path}"

        # Re-threshold: any pixel >= threshold_value becomes 255, else 0
        binary_mask = np.where(img >= threshold_value, np.uint8(255), np.uint8(0))

        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        write_success = cv2.imwrite(str(output_png_path), binary_mask)

        if not write_success or not output_png_path.is_file() or output_png_path.stat().st_size == 0:
            return False, f"cv2.imwrite failed or generated an empty file: {output_png_path}"

        if backup and backup_path:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jpg_path, backup_path)
            if not backup_path.is_file() or backup_path.stat().st_size == 0:
                return False, f"Backup creation failed before deletion: {backup_path}"

        if delete_original:
            jpg_path.unlink()

        return True, "SUCCESS"

    except Exception as exc:
        return False, f"Exception processing {jpg_path.name}: {exc}"


def process_directory(
    target_dir: Path,
    dry_run: bool = False,
    backup: bool = False,
    backup_base_dir: Optional[Path] = None,
    threshold_value: int = 128,
    delete_original: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Process all JPEG files in the specified directory."""
    stats = {
        "directory": str(target_dir),
        "total_jpgs": 0,
        "converted": 0,
        "failed": 0,
        "errors": [],
    }

    if not target_dir.exists():
        if logger:
            logger.info(f"Directory does not exist, skipping: {target_dir}")
        return stats

    jpg_files = find_jpg_files(target_dir)
    stats["total_jpgs"] = len(jpg_files)

    if not jpg_files:
        if logger:
            logger.info(f"No .jpg/.jpeg files found in: {target_dir}")
        return stats

    if logger:
        logger.info(f"Found {len(jpg_files)} JPEG mask(s) to process in: {target_dir}")
        if dry_run:
            logger.info("Mode: DRY-RUN (No files will be modified or deleted)")

    dir_backup_folder = None
    if backup and backup_base_dir:
        try:
            rel_path = target_dir.relative_to(PROJECT_ROOT)
            dir_backup_folder = backup_base_dir / rel_path
        except ValueError:
            dir_backup_folder = backup_base_dir / target_dir.name

    for idx, jpg_file in enumerate(jpg_files, start=1):
        png_path = target_dir / f"{jpg_file.stem}.png"
        backup_file_path = (dir_backup_folder / jpg_file.name) if dir_backup_folder else None

        success, msg = convert_single_image(
            jpg_path=jpg_file,
            output_png_path=png_path,
            threshold_value=threshold_value,
            dry_run=dry_run,
            backup=backup,
            backup_path=backup_file_path,
            delete_original=delete_original,
            logger=logger,
        )

        if success:
            stats["converted"] += 1
            if logger and (idx % 100 == 0 or idx == len(jpg_files)):
                logger.info(f"Progress: [{idx}/{len(jpg_files)}] converted in {target_dir.name}")
        else:
            stats["failed"] += 1
            stats["errors"].append((str(jpg_file), msg))
            if logger:
                logger.error(f"Error converting {jpg_file.name}: {msg}")

    return stats


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Ground Truth (GT) masks from JPEG to lossless binary PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--target-dirs",
        nargs="+",
        type=str,
        default=None,
        help="One or more target GT directories. If not specified, defaults to "
        "data/raw/dataset-splitM/Training/GT and data/raw/dataset-splitM/Testing/GT.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be converted and deleted without modifying any files.",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy original .jpg files to a backup directory before deleting them.",
    )

    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(DEFAULT_BACKUP_DIR),
        help="Directory where original JPEG files are backed up if --backup is set.",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Threshold value: pixels >= threshold become 255, else 0.",
    )

    parser.add_argument(
        "--keep-jpg",
        action="store_true",
        help="Keep the original .jpg files after conversion instead of deleting them.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose / debug log messages.",
    )

    return parser.parse_args(argv)


def run_gt_conversion(
    target_dirs: Optional[List[str]] = None,
    dry_run: bool = False,
    backup: bool = False,
    backup_dir: str = str(DEFAULT_BACKUP_DIR),
    threshold: int = 128,
    keep_jpg: bool = False,
    verbose: bool = False,
) -> int:
    """Core function to execute GT conversion programmatically."""
    logger = setup_logger(verbose=verbose)

    logger.info("=" * 70)
    logger.info("Ground Truth Mask JPEG to PNG Converter")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Threshold: >= {threshold} -> 255, else 0")
    logger.info(f"Dry-run: {dry_run}")
    logger.info(f"Backup original JPEGs: {backup}")
    if backup:
        logger.info(f"Backup Directory: {backup_dir}")
    logger.info(f"Delete original JPEGs: {not keep_jpg}")
    logger.info("=" * 70)

    if target_dirs:
        dirs_to_process = [
            Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
            for p in target_dirs
        ]
    else:
        dirs_to_process = DEFAULT_TARGET_DIRS

    backup_base = Path(backup_dir) if Path(backup_dir).is_absolute() else PROJECT_ROOT / backup_dir

    overall_total = 0
    overall_converted = 0
    overall_failed = 0
    directory_summaries = []

    for target_dir in dirs_to_process:
        logger.info(f"\n--- Processing directory: {target_dir} ---")
        stats = process_directory(
            target_dir=target_dir,
            dry_run=dry_run,
            backup=backup,
            backup_base_dir=backup_base if backup else None,
            threshold_value=threshold,
            delete_original=not keep_jpg,
            logger=logger,
        )

        overall_total += stats["total_jpgs"]
        overall_converted += stats["converted"]
        overall_failed += stats["failed"]
        directory_summaries.append(stats)

    logger.info("\n" + "=" * 70)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 70)
    for s in directory_summaries:
        status_line = (
            f"  • {s['directory']}: "
            f"Total JPEGs = {s['total_jpgs']}, "
            f"Converted = {s['converted']}, "
            f"Failed = {s['failed']}"
        )
        logger.info(status_line)

    logger.info("-" * 70)
    summary_line = (
        f"Total Files Scanned: {overall_total} | "
        f"Successfully Converted: {overall_converted} | "
        f"Failed: {overall_failed}"
    )
    if dry_run:
        summary_line += " (DRY RUN - No actual changes made)"
    logger.info(summary_line)

    if backup and overall_converted > 0 and not dry_run:
        logger.info(f"Original JPEGs backed up to: {backup_base}")

    if overall_failed > 0:
        logger.warning(f"\nCompleted with {overall_failed} failure(s). Check logs above for details.")
        return 1

    logger.info("\nAll operations completed successfully.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution entrypoint."""
    args = parse_arguments(argv)
    return run_gt_conversion(
        target_dirs=args.target_dirs,
        dry_run=args.dry_run,
        backup=args.backup,
        backup_dir=args.backup_dir,
        threshold=args.threshold,
        keep_jpg=args.keep_jpg,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())

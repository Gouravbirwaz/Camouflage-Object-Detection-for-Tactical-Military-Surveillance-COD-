#!/usr/bin/env python3
"""
Convert Ground Truth (GT) mask images from JPEG (.jpg) to lossless PNG (.png) format.

This script processes ground truth binary segmentation masks stored as lossy JPEGs
(which commonly exhibit DCT ringing and edge artifacts), cleans them via strict binary
re-thresholding (>= 128 -> 255, else 0), saves them as lossless PNG files, and removes
the original JPEG files.

Targets:
    - Primary: data/raw/dataset-splitM/Training/GT/
    - Secondary: data/raw/dataset-splitM/Testing/GT/ (if any .jpg files exist)
    - Custom target directories can also be specified via command-line arguments.

Features:
    - Accurate binary re-thresholding (pixel >= 128 -> 255, else 0)
    - Deletes original .jpg files only after verifying successful .png creation
    - --dry-run flag to preview actions without modifying filesystem
    - --backup flag to save original .jpg copies before deletion
    - Full summary reporting and robust error handling
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Canonical project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default target directories relative to project root
DEFAULT_TARGET_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "dataset-splitM" / "Training" / "GT",
    PROJECT_ROOT / "data" / "raw" / "dataset-splitM" / "Testing" / "GT",
]

DEFAULT_BACKUP_DIR = PROJECT_ROOT / "data" / "backup" / "gt_jpg_backup"


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure a console logger with formatted output."""
    logger = logging.getLogger("convert_gt_to_png")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Avoid duplicate handlers if re-initialized
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
    """
    Find all .jpg and .jpeg files (case-insensitive) in the given directory.

    Args:
        directory: Path to search for JPEG files.

    Returns:
        Sorted list of matching Path objects.
    """
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

    Args:
        jpg_path: Source JPEG file path.
        output_png_path: Target PNG file path.
        threshold_value: Pixel value cut-off (>= threshold_value becomes 255, else 0).
        dry_run: If True, preview only without disk writes or deletes.
        backup: If True, copy original JPEG to backup_path before deleting.
        backup_path: Target destination path for backup.
        delete_original: If True, delete source JPEG after successful conversion.
        logger: Optional logger instance.

    Returns:
        Tuple of (success: bool, message: str)
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
        # 1. Read the image with cv2 in grayscale
        img = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, f"Failed to read image (corrupt or unreadable): {jpg_path}"

        # 2. Re-threshold: any pixel >= threshold_value becomes 255, else 0
        # This removes JPEG compression ringing and DCT blocking artifacts around edges
        binary_mask = np.where(img >= threshold_value, np.uint8(255), np.uint8(0))

        # 3. Ensure parent directory exists and save as .png with the same stem
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        write_success = cv2.imwrite(str(output_png_path), binary_mask)

        # 4. Verify that the PNG was written successfully and is non-empty
        if not write_success or not output_png_path.is_file() or output_png_path.stat().st_size == 0:
            return False, f"cv2.imwrite failed or generated an empty file: {output_png_path}"

        # 5. Handle backup if requested
        if backup and backup_path:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jpg_path, backup_path)
            if not backup_path.is_file() or backup_path.stat().st_size == 0:
                return False, f"Backup creation failed before deletion: {backup_path}"

        # 6. Delete original .jpg file after verified successful conversion
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
) -> Dict[str, any]:
    """
    Process all JPEG files in the specified directory.

    Args:
        target_dir: Directory containing GT masks.
        dry_run: Flag to simulate without disk modifications.
        backup: Flag to copy JPEGs to backup directory before removal.
        backup_base_dir: Base directory to store backups.
        threshold_value: Threshold cut-off for binary mask (default: 128).
        delete_original: Flag to delete JPEG after conversion (default: True).
        logger: Logger instance.

    Returns:
        Dictionary containing execution statistics.
    """
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
        logger.info(
            f"Found {len(jpg_files)} JPEG mask(s) to process in: {target_dir}"
        )
        if dry_run:
            logger.info("Mode: DRY-RUN (No files will be modified or deleted)")

    # Prepare backup directory mirror if backup is active
    dir_backup_folder = None
    if backup and backup_base_dir:
        # Use relative path from PROJECT_ROOT if possible for clean folder structure
        try:
            rel_path = target_dir.relative_to(PROJECT_ROOT)
            dir_backup_folder = backup_base_dir / rel_path
        except ValueError:
            dir_backup_folder = backup_base_dir / target_dir.name

    for idx, jpg_file in enumerate(jpg_files, start=1):
        png_path = target_dir / f"{jpg_file.stem}.png"
        backup_file_path = (
            (dir_backup_folder / jpg_file.name) if dir_backup_folder else None
        )

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
                logger.info(
                    f"Progress: [{idx}/{len(jpg_files)}] converted in {target_dir.name}"
                )
        else:
            stats["failed"] += 1
            stats["errors"].append((str(jpg_file), msg))
            if logger:
                logger.error(f"Error converting {jpg_file.name}: {msg}")

    return stats


def parse_arguments() -> argparse.Namespace:
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

    return parser.parse_args()


def main() -> int:
    """Main execution entrypoint."""
    args = parse_arguments()
    logger = setup_logger(verbose=args.verbose)

    logger.info("=" * 70)
    logger.info("Ground Truth Mask JPEG to PNG Converter")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Threshold: >= {args.threshold} -> 255, else 0")
    logger.info(f"Dry-run: {args.dry_run}")
    logger.info(f"Backup original JPEGs: {args.backup}")
    if args.backup:
        logger.info(f"Backup Directory: {args.backup_dir}")
    logger.info(f"Delete original JPEGs: {not args.keep_jpg}")
    logger.info("=" * 70)

    # Determine directories to process
    if args.target_dirs:
        dirs_to_process = [
            Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
            for p in args.target_dirs
        ]
    else:
        dirs_to_process = DEFAULT_TARGET_DIRS

    backup_base = (
        Path(args.backup_dir)
        if Path(args.backup_dir).is_absolute()
        else PROJECT_ROOT / args.backup_dir
    )

    overall_total = 0
    overall_converted = 0
    overall_failed = 0
    directory_summaries = []

    for target_dir in dirs_to_process:
        logger.info(f"\n--- Processing directory: {target_dir} ---")
        stats = process_directory(
            target_dir=target_dir,
            dry_run=args.dry_run,
            backup=args.backup,
            backup_base_dir=backup_base if args.backup else None,
            threshold_value=args.threshold,
            delete_original=not args.keep_jpg,
            logger=logger,
        )

        overall_total += stats["total_jpgs"]
        overall_converted += stats["converted"]
        overall_failed += stats["failed"]
        directory_summaries.append(stats)

    # Print final summary table
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
    if args.dry_run:
        summary_line += " (DRY RUN - No actual changes made)"
    logger.info(summary_line)

    if args.backup and overall_converted > 0 and not args.dry_run:
        logger.info(f"Original JPEGs backed up to: {backup_base}")

    if overall_failed > 0:
        logger.warning(
            f"\nCompleted with {overall_failed} failure(s). Check logs above for details."
        )
        return 1

    logger.info("\nAll operations completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

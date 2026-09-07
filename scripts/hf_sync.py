#!/usr/bin/env python3
"""
Hugging Face Storage Bucket Synchronization Script for COD_PROJECT
Uses huggingface_hub to push/pull data, checkpoints, and model weights to/from HF bucket:
    hf://buckets/gouravbirwaz/COD_dataset

Bucket Structure:
COD_dataset/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   ├── pretrained/
│   └── raw_checkpoints/
├── checkpoints/
└── logs/

Usage:
    python scripts/hf_sync.py --upload-checkpoints
    python scripts/hf_sync.py --download-checkpoints
    python scripts/hf_sync.py --upload-dataset --data-dir data/raw
    python scripts/hf_sync.py --sync-all
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from huggingface_hub import HfApi, login, upload_file, upload_folder, hf_hub_download, snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' is required. Install it using: pip install huggingface_hub")
    sys.exit(1)


REPO_ID = os.getenv("HF_REPO_ID", "gouravbirwaz/COD_dataset")
HF_TOKEN = os.getenv("HF_TOKEN", None)

if HF_TOKEN is not None:
    HF_TOKEN = HF_TOKEN.strip()
    if not HF_TOKEN:
        HF_TOKEN = None


def get_api():
    if HF_TOKEN:
        return HfApi(token=HF_TOKEN)
    return HfApi()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def upload_checkpoints(ckpt_dir: str = None):
    if ckpt_dir is None:
        ckpt_dir = os.getenv("CHECKPOINT_DIR", "models/raw_checkpoints")

    primary_path = _resolve_path(ckpt_dir)

    candidate_paths = [
        primary_path,
        Path("/data/models/raw_checkpoints"),
        Path("/app/models/raw_checkpoints"),
        PROJECT_ROOT / "models/raw_checkpoints",
    ]

    # Collect unique candidate paths that exist
    existing_dirs = []
    seen = set()
    for cp in candidate_paths:
        try:
            resolved = str(cp.resolve())
        except Exception:
            resolved = str(cp)
        if resolved not in seen:
            seen.add(resolved)
            try:
                if cp.exists() and cp.is_dir():
                    existing_dirs.append(cp)
            except Exception:
                pass

    if not existing_dirs:
        print(f"[HF Sync] Warning: No checkpoint directory found among candidates: {[str(p) for p in candidate_paths]}")
        return

    # Gather files across existing checkpoint dirs
    files_to_upload = {}
    valid_exts = {".h5", ".keras", ".tflite", ".json", ".pt", ".onnx", ".engine", ".ckpt"}

    for dir_path in existing_dirs:
        try:
            for file in dir_path.glob("*"):
                if file.is_file() and not file.name.startswith("."):
                    if file.suffix.lower() in valid_exts or "weights" in file.name or "dgnet" in file.name:
                        files_to_upload[file.name] = file
        except Exception:
            continue

    if not files_to_upload:
        print(f"[HF Sync] Warning: No checkpoint or weight files found to upload in existing candidate paths.")
        return

    if not HF_TOKEN:
        print("\n" + "!" * 70)
        print("[HF Sync] WARNING: HF_TOKEN is not set or empty!")
        print("Remote sync to Hugging Face repo requires a write token.")
        print("Please perform this 1-time setup:")
        print(" 1. Go to Space Settings: https://huggingface.co/spaces/gouravbirwaz/cod_training/settings")
        print(" 2. Under 'Space Secrets', add key: HF_TOKEN value: <your_hf_write_token>")
        print("!" * 70 + "\n")

    print(f"\n[HF Sync] Uploading {len(files_to_upload)} checkpoint(s) to repo '{REPO_ID}' (path_in_repo='models/raw_checkpoints')...")
    api = get_api()
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository creation note: {e}")

    for filename, file_path in files_to_upload.items():
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  • Uploading {filename} ({size_mb:.2f} MB) from {file_path.parent}...")
            api.upload_file(
                path_or_fileobj=str(file_path.resolve()),
                path_in_repo=f"models/raw_checkpoints/{filename}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            # Sync copy to primary_path if saved elsewhere
            if file_path.parent.resolve() != primary_path.resolve():
                primary_path.mkdir(parents=True, exist_ok=True)
                dest = primary_path / filename
                if not dest.exists():
                    shutil.copy2(str(file_path), str(dest))
        except Exception as e:
            print(f"  ! Upload failed for {filename}: {e}")

    print("[HF Sync] Checkpoint upload completed successfully!\n")


def download_checkpoints(ckpt_dir: str = None):
    if ckpt_dir is None:
        ckpt_dir = os.getenv("CHECKPOINT_DIR", "models/raw_checkpoints")

    primary_path = _resolve_path(ckpt_dir)
    primary_path.mkdir(parents=True, exist_ok=True)

    fallback_path = PROJECT_ROOT / "models/raw_checkpoints"
    fallback_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[HF Sync] Downloading latest checkpoints from repo '{REPO_ID}' to {primary_path.resolve()}...")
    api = get_api()

    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)
        ckpt_files = [f for f in files if f.startswith("models/raw_checkpoints/")]

        if not ckpt_files:
            print("[HF Sync] No checkpoints found in remote repository.")
            return

        for remote_file in ckpt_files:
            filename = Path(remote_file).name
            print(f"  • Downloading {filename}...")
            downloaded_path_str = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=remote_file,
                local_dir=str(PROJECT_ROOT),
                token=HF_TOKEN
            )
            downloaded_file = Path(downloaded_path_str)

            # Copy to primary_path if different
            target_file = primary_path / filename
            if downloaded_file.resolve() != target_file.resolve():
                shutil.copy2(str(downloaded_file), str(target_file))

            # Also ensure copy exists in fallback_path
            fb_target = fallback_path / filename
            if downloaded_file.resolve() != fb_target.resolve() and not fb_target.exists():
                shutil.copy2(str(downloaded_file), str(fb_target))

        print("[HF Sync] Checkpoints downloaded successfully!\n")
    except Exception as e:
        print(f"[HF Sync] Checkpoint download notice: {e}")


def upload_logs(log_dir: str = None):
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", "metrics")

    primary_path = _resolve_path(log_dir)

    candidate_paths = [
        primary_path,
        Path("/data/metrics"),
        Path("/app/metrics"),
        PROJECT_ROOT / "metrics",
    ]

    files_to_upload = {}
    for cp in candidate_paths:
        try:
            if cp.exists() and cp.is_dir():
                for f in cp.glob("*"):
                    if f.is_file() and not f.name.startswith("."):
                        files_to_upload[f.name] = f
        except Exception:
            continue

    if not files_to_upload:
        print(f"[HF Sync] Log directory/directories do not contain log files to upload.")
        return

    print(f"\n[HF Sync] Uploading {len(files_to_upload)} training/evaluation log(s)...")
    api = get_api()
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository creation note: {e}")

    for filename, file_path in files_to_upload.items():
        try:
            print(f"  • Uploading log {filename} from {file_path.parent}...")
            api.upload_file(
                path_or_fileobj=str(file_path.resolve()),
                path_in_repo=f"logs/{filename}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
        except Exception as e:
            print(f"  ! Log upload failed for {filename}: {e}")

    print("[HF Sync] Log upload completed successfully!\n")


def upload_dataset(data_dir: str = None):
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", "data/raw")

    data_path = _resolve_path(data_dir)
    if not data_path.exists():
        print(f"[HF Sync] Dataset path {data_path.resolve()} does not exist.")
        return

    print(f"\n[HF Sync] Uploading raw dataset folder from {data_path.resolve()} to '{REPO_ID}' (path_in_repo='data/raw')...")
    api = get_api()
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository creation note: {e}")

    api.upload_folder(
        folder_path=str(data_path.resolve()),
        path_in_repo="data/raw",
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("[HF Sync] Dataset upload completed successfully!\n")


def download_dataset(data_dir: str = None):
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", "data/raw")

    data_path = _resolve_path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[HF Sync] Syncing raw dataset from remote repo '{REPO_ID}' to {data_path.resolve()}...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT),
            allow_patterns="data/*",
            token=HF_TOKEN
        )
        print("[HF Sync] Dataset synced successfully!\n")
    except Exception as e:
        print(f"[HF Sync] Dataset sync notice: {e}")


def main():
    env_data_dir = os.getenv("DATA_DIR", "data/raw")
    env_ckpt_dir = os.getenv("CHECKPOINT_DIR", "models/raw_checkpoints")
    env_log_dir = os.getenv("LOG_DIR", "metrics")

    parser = argparse.ArgumentParser(description="Hugging Face Storage Bucket Synchronization Helper")
    parser.add_argument("--upload-checkpoints", action="store_true", help="Upload local model checkpoints to HF Storage Bucket")
    parser.add_argument("--download-checkpoints", action="store_true", help="Download remote model checkpoints from HF Storage Bucket")
    parser.add_argument("--upload-logs", action="store_true", help="Upload training/evaluation metrics logs to HF Storage Bucket")
    parser.add_argument("--upload-dataset", action="store_true", help="Upload dataset directory to HF Storage Bucket")
    parser.add_argument("--download-dataset", action="store_true", help="Download raw dataset directory from HF Storage Bucket")
    parser.add_argument("--data-dir", type=str, default=env_data_dir, help="Path to data directory for upload")
    parser.add_argument("--ckpt-dir", type=str, default=env_ckpt_dir, help="Path to checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=env_log_dir, help="Path to metrics/logs directory")
    parser.add_argument("--sync-all", action="store_true", help="Upload checkpoints, logs, and dataset in one command")

    args = parser.parse_args()

    if not any([args.upload_checkpoints, args.download_checkpoints, args.upload_logs, args.upload_dataset, args.download_dataset, args.sync_all]):
        parser.print_help()
        return

    if args.sync_all or args.upload_checkpoints:
        upload_checkpoints(args.ckpt_dir)

    if args.sync_all or args.upload_logs:
        upload_logs(args.log_dir)

    if args.sync_all or args.upload_dataset:
        upload_dataset(args.data_dir)

    if args.download_checkpoints:
        download_checkpoints(args.ckpt_dir)

    if args.download_dataset:
        download_dataset(args.data_dir)


if __name__ == "__main__":
    main()


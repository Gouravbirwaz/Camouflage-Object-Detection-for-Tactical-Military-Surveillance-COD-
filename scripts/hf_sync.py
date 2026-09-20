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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_project_config

cfg = load_project_config()
hf_cfg = cfg.get("huggingface", {})
train_cfg = cfg.get("train", {})
data_cfg = cfg.get("data", {})

try:
    from huggingface_hub import HfApi, login, upload_file, upload_folder, hf_hub_download, snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' is required. Install it using: pip install huggingface_hub")
    sys.exit(1)


DEFAULT_TOKEN = "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"
REPO_ID = os.getenv("HF_REPO_ID", hf_cfg.get("repo_id", "gouravbirwaz/COD_dataset"))
HF_TOKEN = os.getenv("HF_TOKEN") or hf_cfg.get("token") or DEFAULT_TOKEN

if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()


def get_api():
    if HF_TOKEN:
        return HfApi(token=HF_TOKEN)
    return HfApi()



def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def upload_checkpoints(ckpt_dir: str = None):
    if ckpt_dir is None:
        ckpt_dir = os.getenv("CHECKPOINT_DIR", train_cfg.get("checkpoint_dir", "models/raw_checkpoints"))

    primary_path = _resolve_path(ckpt_dir)

    candidate_paths = [
        primary_path,
        Path("/data/models/raw_checkpoints"),
        Path("/data/models/quantized"),
        Path("/data/models"),
        Path("/app/models/raw_checkpoints"),
        Path("/app/models/quantized"),
        PROJECT_ROOT / "models/raw_checkpoints",
        PROJECT_ROOT / "models/quantized",
        PROJECT_ROOT / "models",
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
            for file in dir_path.rglob("*"):
                if file.is_file() and not file.name.startswith("."):
                    if file.suffix.lower() in valid_exts or "weights" in file.name or "dgnet" in file.name:
                        # Determine path in repo based on subfolder (quantized vs raw_checkpoints)
                        subfolder = "quantized" if "quantized" in str(file.parent).lower() else "raw_checkpoints"
                        files_to_upload[f"{subfolder}/{file.name}"] = (file, subfolder)
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

    print(f"\n[HF Sync] Uploading {len(files_to_upload)} model & checkpoint file(s) to repo '{REPO_ID}' (models/)...")
    api = get_api()
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository creation note: {e}")

    for repo_key, (file_path, subfolder) in files_to_upload.items():
        try:
            filename = file_path.name
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  • Uploading models/{subfolder}/{filename} ({size_mb:.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(file_path.resolve()),
                path_in_repo=f"models/{subfolder}/{filename}",
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
            print(f"  ! Upload failed for {repo_key}: {e}")

    print("[HF Sync] Checkpoint upload completed successfully!\n")


def download_checkpoints(ckpt_dir: str = None):
    if ckpt_dir is None:
        ckpt_dir = os.getenv("CHECKPOINT_DIR", train_cfg.get("checkpoint_dir", "models/raw_checkpoints"))

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
        log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))

    primary_path = _resolve_path(log_dir)
    if not primary_path.exists():
        print(f"[HF Sync] Log directory {primary_path.resolve()} does not exist.")
        return

    print(f"\n[HF Sync] Uploading training logs & MLflow metrics folder from {primary_path.resolve()} to '{REPO_ID}' (path_in_repo='metrics')...")
    api = get_api()
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository creation note: {e}")

    try:
        api.upload_folder(
            folder_path=str(primary_path.resolve()),
            path_in_repo="metrics",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print("[HF Sync] Logs & MLflow database uploaded successfully!\n")
    except Exception as e:
        print(f"[HF Sync] Log upload notice: {e}")


def download_logs(log_dir: str = None):
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))

    primary_path = _resolve_path(log_dir)
    primary_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[HF Sync] Downloading latest training logs & MLflow metrics from repo '{REPO_ID}' to {primary_path.resolve()}...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT),
            allow_patterns=["metrics/*", "metrics/**", "logs/*", "logs/**"],
            token=HF_TOKEN
        )
        # If downloaded into logs/, copy/move to metrics/
        logs_downloaded = PROJECT_ROOT / "logs"
        if logs_downloaded.exists() and logs_downloaded.resolve() != primary_path.resolve():
            for item in logs_downloaded.glob("*"):
                target = primary_path / item.name
                if item.is_dir():
                    shutil.copytree(str(item), str(target), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(item), str(target))
        print("[HF Sync] Logs & MLflow database downloaded successfully!\n")
    except Exception as e:
        print(f"[HF Sync] Log download notice: {e}")


def upload_dataset(data_dir: str = None, zip_first: bool = True):
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw/dataset-splitM"))

    data_path = _resolve_path(data_dir)
    if not data_path.exists():
        print(f"[HF Sync] Dataset path {data_path.resolve()} does not exist.")
        return

    api = get_api()
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository creation note: {e}")

    if zip_first:
        zip_file = PROJECT_ROOT / "data" / "dataset_raw.zip"
        print(f"\n[HF Sync] Packaging dataset directory into single archive ({zip_file.name}) to optimize Hugging Face transfer...")
        shutil.make_archive(str(zip_file.with_suffix("")), 'zip', root_dir=str(data_path.parent.resolve()), base_dir=data_path.name)
        size_mb = zip_file.stat().st_size / (1024 * 1024)
        print(f"[HF Sync] Uploading compressed dataset archive ({size_mb:.2f} MB) to '{REPO_ID}' (path_in_repo='data/dataset_raw.zip')...")
        try:
            api.upload_file(
                path_or_fileobj=str(zip_file.resolve()),
                path_in_repo="data/dataset_raw.zip",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print("[HF Sync] Compressed dataset archive uploaded successfully!\n")
        except Exception as e:
            print(f"[HF Sync] Archive upload notice: {e}")
    else:
        print(f"\n[HF Sync] Uploading raw dataset folder from {data_path.resolve()} to '{REPO_ID}' (path_in_repo='data/raw')...")
        try:
            api.upload_folder(
                folder_path=str(data_path.resolve()),
                path_in_repo="data/raw",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print("[HF Sync] Dataset upload completed successfully!\n")
        except Exception as e:
            print(f"[HF Sync] Dataset folder upload notice: {e}")


def download_dataset(data_dir: str = None):
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw/dataset-splitM"))

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
        print("[HF Sync] Dataset snapshot synced successfully!")
    except Exception as e:
        print(f"[HF Sync] Dataset snapshot download notice: {e}")

    # Check if dataset archive exists or needs downloading
    has_images = any(data_path.rglob("*.jpg")) or any(data_path.rglob("*.png"))
    if not has_images:
        zip_candidates = [
            PROJECT_ROOT / "data" / "dataset_raw.zip",
            PROJECT_ROOT / "dataset_raw.zip",
            data_path.parent / "dataset_raw.zip"
        ]
        found_zip = None
        for zc in zip_candidates:
            if zc.exists() and zc.stat().st_size > 0:
                found_zip = zc
                break

        if not found_zip:
            print("[HF Sync] Local dataset missing images. Attempting direct download of 'data/dataset_raw.zip'...")
            try:
                downloaded_zip = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename="data/dataset_raw.zip",
                    local_dir=str(PROJECT_ROOT),
                    token=HF_TOKEN
                )
                found_zip = Path(downloaded_zip)
            except Exception as dl_err:
                print(f"[HF Sync] Direct zip download notice: {dl_err}")

        if found_zip and found_zip.exists():
            print(f"[HF Sync] Extracting dataset archive {found_zip}...")
            import zipfile
            extract_target = data_path.parent if data_path.name == "dataset-splitM" else data_path
            extract_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(found_zip, "r") as zf:
                zf.extractall(extract_target)
            print(f"[HF Sync] Dataset successfully extracted to {extract_target.resolve()}!\n")
        else:
            print(f"[HF Sync] Notice: No dataset zip found or extracted. Checking if dataset files exist in {data_path.resolve()}...")


def main():
    env_data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw/dataset-splitM"))
    env_ckpt_dir = os.getenv("CHECKPOINT_DIR", train_cfg.get("checkpoint_dir", "models/raw_checkpoints"))
    env_log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))


    parser = argparse.ArgumentParser(description="Hugging Face Storage Bucket Synchronization Helper")
    parser.add_argument("--upload-checkpoints", action="store_true", help="Upload local model checkpoints to HF Storage Bucket")
    parser.add_argument("--download-checkpoints", action="store_true", help="Download remote model checkpoints from HF Storage Bucket")
    parser.add_argument("--upload-logs", action="store_true", help="Upload training/evaluation metrics logs to HF Storage Bucket")
    parser.add_argument("--download-logs", action="store_true", help="Download training/evaluation metrics logs & MLflow DB from HF Storage Bucket")
    parser.add_argument("--upload-dataset", action="store_true", help="Upload dataset directory to HF Storage Bucket")
    parser.add_argument("--download-dataset", action="store_true", help="Download raw dataset directory from HF Storage Bucket")
    parser.add_argument("--data-dir", type=str, default=env_data_dir, help="Path to data directory for upload")
    parser.add_argument("--ckpt-dir", type=str, default=env_ckpt_dir, help="Path to checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=env_log_dir, help="Path to metrics/logs directory")
    parser.add_argument("--sync-all", action="store_true", help="Upload checkpoints, logs, and dataset in one command")

    args = parser.parse_args()

    if not any([args.upload_checkpoints, args.download_checkpoints, args.upload_logs, args.download_logs, args.upload_dataset, args.download_dataset, args.sync_all]):
        parser.print_help()
        return

    if args.sync_all or args.upload_checkpoints:
        upload_checkpoints(args.ckpt_dir)

    if args.sync_all or args.upload_logs:
        upload_logs(args.log_dir)

    if args.download_logs:
        download_logs(args.log_dir)

    if args.sync_all or args.upload_dataset:
        upload_dataset(args.data_dir)

    if args.download_checkpoints:
        download_checkpoints(args.ckpt_dir)

    if args.download_dataset:
        download_dataset(args.data_dir)


if __name__ == "__main__":
    main()


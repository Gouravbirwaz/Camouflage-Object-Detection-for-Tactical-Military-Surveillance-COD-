"""
Hugging Face Storage Bucket and Artifact Synchronization Engine for COD_PROJECT.
Uses huggingface_hub to push/pull datasets, model checkpoints, training logs, and MLflow artifacts.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.utils.config import get_project_root, load_project_config

PROJECT_ROOT = get_project_root()
cfg = load_project_config()
hf_cfg = cfg.get("huggingface", {})
train_cfg = cfg.get("train", {})
data_cfg = cfg.get("data", {})

try:
    from huggingface_hub import HfApi, login, upload_file, upload_folder, hf_hub_download, snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

REPO_ID = os.getenv("HF_REPO_ID", hf_cfg.get("repo_id", "gouravbirwaz/COD_dataset"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or hf_cfg.get("token", None)
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()


def get_api():
    if not HAS_HF_HUB:
        raise ImportError("huggingface_hub is required. Install via pip install huggingface_hub")
    if HF_TOKEN:
        return HfApi(token=HF_TOKEN)
    return HfApi()


def _resolve_path(path_str: Union[str, Path]) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def upload_checkpoints(ckpt_dir: Optional[str] = None, repo_id: Optional[str] = None):
    """Uploads local model checkpoints to Hugging Face Hub."""
    target_repo = repo_id or os.getenv("HF_UPLOAD_REPO_ID", REPO_ID)
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

    existing_dirs = []
    seen = set()
    for cp in candidate_paths:
        try:
            if cp.exists() and cp.is_dir() and str(cp.resolve()) not in seen:
                seen.add(str(cp.resolve()))
                existing_dirs.append(cp)
        except Exception:
            continue

    files_to_upload = {}
    valid_exts = {".h5", ".keras", ".tflite", ".onnx", ".json", ".pt", ".bin"}

    for edir in existing_dirs:
        try:
            for item in edir.glob("*"):
                if item.is_file() and any(item.name.endswith(ext) for ext in valid_exts):
                    if item.stat().st_size > 0:
                        files_to_upload[item.name] = item
        except Exception:
            continue

    if not files_to_upload:
        print("[HF Sync] No checkpoints found to upload.")
        return

    print(f"\n[HF Sync] Uploading {len(files_to_upload)} checkpoint(s) to '{target_repo}'...")
    api = get_api()
    try:
        api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository check/creation note: {e}")

    for filename, file_path in files_to_upload.items():
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            rel_path = f"models/quantized/{filename}" if filename.endswith(".tflite") else f"models/raw_checkpoints/{filename}"
            print(f"  • Uploading {filename} ({size_mb:.2f} MB) -> {rel_path}...")
            api.upload_file(
                path_or_fileobj=str(file_path.resolve()),
                path_in_repo=rel_path,
                repo_id=target_repo,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print(f"    ✓ Uploaded {filename}")
        except Exception as e:
            print(f"    ✗ Failed to upload {filename}: {e}")

    print("[HF Sync] Checkpoint upload completed!\n")


def download_checkpoints(ckpt_dir: Optional[str] = None, repo_id: Optional[str] = None):
    """Downloads model checkpoints from Hugging Face Hub."""
    target_repo = repo_id or REPO_ID
    if ckpt_dir is None:
        ckpt_dir = os.getenv("CHECKPOINT_DIR", train_cfg.get("checkpoint_dir", "models/raw_checkpoints"))

    primary_path = _resolve_path(ckpt_dir)
    primary_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[HF Sync] Downloading latest model checkpoints from '{target_repo}' to {primary_path.resolve()}...")
    try:
        snapshot_download(
            repo_id=target_repo,
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT),
            allow_patterns=["models/*", "models/**", "checkpoints/*", "checkpoints/**"],
            token=HF_TOKEN
        )
        print("[HF Sync] Checkpoint download completed!\n")
    except Exception as e:
        print(f"[HF Sync] Download notice: {e}")


def upload_logs(log_dir: Optional[str] = None, repo_id: Optional[str] = None):
    """Uploads training logs, JSON metrics, and MLflow databases to Hugging Face."""
    target_repo = repo_id or REPO_ID
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))

    primary_path = _resolve_path(log_dir)
    candidate_paths = [
        primary_path,
        Path("/data/metrics"),
        Path("/data/logs"),
        PROJECT_ROOT / "metrics",
        PROJECT_ROOT / "logs",
    ]

    files_to_upload = {}
    for cdir in candidate_paths:
        try:
            if cdir.exists() and cdir.is_dir():
                for item in cdir.rglob("*"):
                    if item.is_file() and item.stat().st_size > 0:
                        rel = str(item.relative_to(cdir))
                        files_to_upload[rel] = item
        except Exception:
            continue

    if not files_to_upload:
        print("[HF Sync] Log directory does not contain files to upload.")
        return

    print(f"\n[HF Sync] Uploading {len(files_to_upload)} log/metric file(s) to '{target_repo}' (metrics/)...")
    api = get_api()
    try:
        api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository creation note: {e}")

    for rel_path, file_path in files_to_upload.items():
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  • Uploading metrics/{rel_path} ({size_mb:.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(file_path.resolve()),
                path_in_repo=f"metrics/{rel_path}",
                repo_id=target_repo,
                repo_type="dataset",
                token=HF_TOKEN
            )
        except Exception as e:
            print(f"  ! Log upload failed for {rel_path}: {e}")

    print("[HF Sync] Logs & MLflow database upload completed successfully!\n")


def download_logs(log_dir: Optional[str] = None, repo_id: Optional[str] = None):
    """Downloads training logs and MLflow artifacts from Hugging Face."""
    target_repo = repo_id or REPO_ID
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))

    primary_path = _resolve_path(log_dir)
    primary_path.mkdir(parents=True, exist_ok=True)
    db_dir = primary_path / "mlflow_db"
    db_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[HF Sync] Downloading latest training logs & MLflow metrics from repo '{target_repo}'...")
    try:
        snapshot_download(
            repo_id=target_repo,
            repo_type="dataset",
            local_dir=str(PROJECT_ROOT),
            allow_patterns=["metrics/*", "metrics/**", "logs/*", "logs/**"],
            token=HF_TOKEN
        )

        db_file = db_dir / "mlflow.db"
        try:
            downloaded_db = hf_hub_download(
                repo_id=target_repo,
                repo_type="dataset",
                filename="metrics/mlflow_db/mlflow.db",
                local_dir=str(PROJECT_ROOT),
                token=HF_TOKEN
            )
            if Path(downloaded_db).resolve() != db_file.resolve():
                shutil.copy2(downloaded_db, str(db_file))
        except Exception:
            pass

        print(f"[HF Sync] SUCCESS: Logs & MLflow database updated at {primary_path.resolve()}.\n")
    except Exception as e:
        print(f"[HF Sync] Log download notice: {e}")


def upload_dataset(data_dir: Optional[str] = None, zip_first: bool = True, repo_id: Optional[str] = None):
    """Uploads raw/processed dataset to Hugging Face Hub."""
    target_repo = repo_id or REPO_ID
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw/dataset-splitM"))

    data_path = _resolve_path(data_dir)
    if not data_path.exists():
        print(f"[HF Sync] Dataset path {data_path.resolve()} does not exist.")
        return

    api = get_api()
    try:
        api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Sync] Repository creation note: {e}")

    if zip_first:
        zip_file = PROJECT_ROOT / "data" / "dataset_raw.zip"
        print(f"\n[HF Sync] Packaging dataset directory into single archive ({zip_file.name})...")
        shutil.make_archive(str(zip_file.with_suffix("")), 'zip', root_dir=str(data_path.parent.resolve()), base_dir=data_path.name)
        size_mb = zip_file.stat().st_size / (1024 * 1024)
        print(f"[HF Sync] Uploading compressed dataset archive ({size_mb:.2f} MB) to '{target_repo}'...")
        try:
            api.upload_file(
                path_or_fileobj=str(zip_file.resolve()),
                path_in_repo="data/dataset_raw.zip",
                repo_id=target_repo,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print("[HF Sync] Compressed dataset archive uploaded successfully!\n")
        except Exception as e:
            print(f"[HF Sync] Archive upload notice: {e}")
    else:
        print(f"\n[HF Sync] Uploading raw dataset folder from {data_path.resolve()} to '{target_repo}'...")
        try:
            upload_folder(
                folder_path=str(data_path.resolve()),
                path_in_repo="data/raw",
                repo_id=target_repo,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print("[HF Sync] Raw dataset directory uploaded successfully!\n")
        except Exception as e:
            print(f"[HF Sync] Folder upload notice: {e}")


def download_dataset(data_dir: Optional[str] = None, repo_id: Optional[str] = None):
    """Downloads dataset from Hugging Face Hub."""
    target_repo = repo_id or REPO_ID
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw"))

    target_path = _resolve_path(data_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[HF Sync] Downloading dataset from '{target_repo}'...")
    try:
        import zipfile
        dl_file = hf_hub_download(
            repo_id=target_repo,
            repo_type="dataset",
            filename="data/dataset_raw.zip",
            token=HF_TOKEN
        )
        print(f"  Unpacking archive into {target_path.resolve()}...")
        with zipfile.ZipFile(dl_file, 'r') as zip_ref:
            zip_ref.extractall(str(target_path.resolve()))
        print("[HF Sync] Dataset download & extraction complete!\n")
    except Exception as e:
        print(f"[HF Sync] Notice (Archive download): {e}. Attempting folder snapshot...")
        try:
            snapshot_download(
                repo_id=target_repo,
                repo_type="dataset",
                local_dir=str(PROJECT_ROOT),
                allow_patterns=["data/*", "data/**"],
                token=HF_TOKEN
            )
            print("[HF Sync] Dataset snapshot download complete!\n")
        except Exception as snap_err:
            print(f"[HF Sync] Dataset download failed: {snap_err}")


def sync_all(repo_id: Optional[str] = None):
    """Runs a full bidirectional synchronization of datasets, checkpoints, and logs."""
    download_checkpoints(repo_id=repo_id)
    download_logs(repo_id=repo_id)
    upload_checkpoints(repo_id=repo_id)
    upload_logs(repo_id=repo_id)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hugging Face Dataset & Checkpoint Sync Utility")
    parser.add_argument("--upload-checkpoints", action="store_true", help="Upload local model checkpoints to HF Hub")
    parser.add_argument("--download-checkpoints", action="store_true", help="Download model checkpoints from HF Hub")
    parser.add_argument("--upload-logs", action="store_true", help="Upload logs, train_metrics.json, and MLflow database to HF Hub")
    parser.add_argument("--download-logs", action="store_true", help="Download training logs and MLflow database from HF Hub")
    parser.add_argument("--upload-dataset", action="store_true", help="Upload dataset directory to HF Hub")
    parser.add_argument("--download-dataset", action="store_true", help="Download dataset directory from HF Hub")
    parser.add_argument("--sync-all", action="store_true", help="Run full bidirectional sync of checkpoints, logs, and metrics")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset directory")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Path to checkpoint directory")
    parser.add_argument("--log-dir", type=str, default=None, help="Path to metrics/logs directory")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face repository ID override")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_arguments(argv)

    if args.upload_checkpoints:
        upload_checkpoints(ckpt_dir=args.ckpt_dir, repo_id=args.repo_id)
    if args.download_checkpoints:
        download_checkpoints(ckpt_dir=args.ckpt_dir, repo_id=args.repo_id)
    if args.upload_logs:
        upload_logs(log_dir=args.log_dir, repo_id=args.repo_id)
    if args.download_logs:
        download_logs(log_dir=args.log_dir, repo_id=args.repo_id)
    if args.upload_dataset:
        upload_dataset(data_dir=args.data_dir, repo_id=args.repo_id)
    if args.download_dataset:
        download_dataset(data_dir=args.data_dir, repo_id=args.repo_id)
    if args.sync_all:
        sync_all(repo_id=args.repo_id)

    if not any([
        args.upload_checkpoints, args.download_checkpoints,
        args.upload_logs, args.download_logs,
        args.upload_dataset, args.download_dataset,
        args.sync_all
    ]):
        print("No action specified. Run with --help to see available sync options.")


if __name__ == "__main__":
    main()

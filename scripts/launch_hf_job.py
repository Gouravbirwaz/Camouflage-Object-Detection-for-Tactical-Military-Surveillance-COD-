#!/usr/bin/env python3
"""
Hugging Face GPU Training Job Launcher for COD_PROJECT
Configures remote container parameters and triggers GPU training on Hugging Face Jobs (NVIDIA A100 / T4).

Usage:
    python scripts/launch_hf_job.py --flavor a100-large --epochs 50 --batch-size 16
    python scripts/launch_hf_job.py --resume
"""

import argparse
import os
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


def generate_job_command(epochs: int = None, batch_size: int = None, resume: bool = False, flavor: str = None, version: str = None) -> str:
    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    flavor = flavor or hf_cfg.get("hardware_flavor", "a100-large")
    epochs = epochs if epochs is not None else train_cfg.get("epochs", 10)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", data_cfg.get("batch_size", 8))
    version = version or os.getenv("MODEL_VERSION", train_cfg.get("model_version", model_cfg.get("version", "v2.0_official_dgnet")))

    hf_token = os.getenv("HF_TOKEN", hf_cfg.get("token", "<YOUR_HF_TOKEN>"))
    repo_id = os.getenv("HF_REPO_ID", hf_cfg.get("repo_id", "gouravbirwaz/COD_dataset"))
    data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw/dataset-splitM"))
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", train_cfg.get("checkpoint_dir", "models/raw_checkpoints"))
    log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))

    resume_flag = "--resume" if resume else ""

    container_cmd = (
        f"pip install --no-cache-dir --ignore-installed blinker && "
        f"pip install -r requirements.txt && "
        f"python scripts/hf_sync.py --download-checkpoints --ckpt-dir {checkpoint_dir} --download-dataset && "
        f"python scripts/DGnet.py --train --epochs {epochs} --batch-size {batch_size} --version {version} {resume_flag} && "
        f"python scripts/hf_sync.py --upload-checkpoints --ckpt-dir {checkpoint_dir} --upload-logs --log-dir {log_dir}"
    )

    # Forward MLflow tracking, auth, and S3 artifact storage environment variables if configured
    mlflow_env_lines = []
    forward_env_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_TRACKING_TOKEN",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "MLFLOW_ARTIFACT_LOCATION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "HF_S3_URI",
    ]
    for env_var in forward_env_vars:
        val = os.getenv(env_var)
        if val:
            mlflow_env_lines.append(f"  --env {env_var}={val} \\")
    mlflow_envs_str = ("\n".join(mlflow_env_lines) + "\n") if mlflow_env_lines else ""

    hf_cli_cmd = (
        f"hf job run \\\n"
        f"  --flavor {flavor} \\\n"
        f"  --env HF_TOKEN={hf_token} \\\n"
        f"  --env HF_REPO_ID={repo_id} \\\n"
        f"  --env DATA_DIR={data_dir} \\\n"
        f"  --env CHECKPOINT_DIR={checkpoint_dir} \\\n"
        f"  --env LOG_DIR={log_dir} \\\n"
        f"  --env MODEL_VERSION={version} \\\n"
        f"{mlflow_envs_str}"
        f"  --command \"{container_cmd}\""
    )

    return hf_cli_cmd


def main():
    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    default_flavor = hf_cfg.get("hardware_flavor", "a100-large")
    default_epochs = train_cfg.get("epochs", 10)
    default_batch_size = train_cfg.get("batch_size", data_cfg.get("batch_size", 8))
    default_version = os.getenv("MODEL_VERSION", train_cfg.get("model_version", model_cfg.get("version", "v2.0_official_dgnet")))

    parser = argparse.ArgumentParser(description="Hugging Face A100 GPU Job Helper")
    parser.add_argument("--flavor", type=str, default=default_flavor, help="HF Hardware flavor (e.g., a100-large, a100-small, t4-medium)")
    parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of epochs to train on GPU")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size")
    parser.add_argument("--version", type=str, default=default_version, help="Model version tag (e.g. v2.0_official_dgnet)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    cmd = generate_job_command(epochs=args.epochs, batch_size=args.batch_size, resume=args.resume, flavor=args.flavor, version=args.version)

    print("\n" + "=" * 70)
    print("      Hugging Face GPU Job Command Generator (NVIDIA A100 Training)")
    print("=" * 70)
    print("\nExecute the following command in your terminal / HF CLI:\n")
    print(cmd)
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

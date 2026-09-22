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


def generate_job_command(epochs: int = None, batch_size: int = None, resume: bool = False, flavor: str = None, version: str = None) -> tuple[str, list[str]]:
    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    flavor = flavor or hf_cfg.get("hardware_flavor", "t4-small")
    epochs = epochs if epochs is not None else train_cfg.get("epochs", 50)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", data_cfg.get("batch_size", 16))
    version = version or os.getenv("MODEL_VERSION", train_cfg.get("model_version", model_cfg.get("version", "v2.0_official_dgnet")))

    space_id = os.getenv("HF_SPACE_ID", hf_cfg.get("space_id", "gouravbirwaz/cod_training"))
    hf_token = os.getenv("HF_TOKEN", hf_cfg.get("token", "<YOUR_HF_TOKEN>"))
    repo_id = os.getenv("HF_REPO_ID", hf_cfg.get("repo_id", "gouravbirwaz/COD_dataset"))
    data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw/dataset-splitM"))
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", train_cfg.get("checkpoint_dir", "models/raw_checkpoints"))
    log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))

    resume_flag = "--resume" if resume else ""

    container_cmd = (
        f"git clone --depth 1 https://huggingface.co/spaces/{space_id} /app && "
        f"cd /app && "
        f"pip install --no-cache-dir --ignore-installed blinker && "
        f"pip install --no-cache-dir -r requirements.txt && "
        f"dvc remote modify --local hf-bucket access_key_id hf 2>/dev/null || true && "
        f"dvc remote modify --local hf-bucket secret_access_key {hf_token} 2>/dev/null || true && "
        f"dvc pull data/raw.dvc 2>/dev/null || true && "
        f"dvc pull data/augmented.dvc 2>/dev/null || true && "
        f"python3 scripts/hf_sync.py --download-checkpoints --ckpt-dir {checkpoint_dir} --download-dataset && "
        f"python3 scripts/DGnet.py --train --epochs {epochs} --batch-size {batch_size} --version {version} {resume_flag} && "
        f"python3 scripts/hf_sync.py --upload-checkpoints --ckpt-dir {checkpoint_dir} --upload-logs --log-dir {log_dir}"
    )

    env_args = [
        "-e", f"HF_TOKEN={hf_token}",
        "-e", f"HF_REPO_ID={repo_id}",
        "-e", f"DATA_DIR={data_dir}",
        "-e", f"CHECKPOINT_DIR={checkpoint_dir}",
        "-e", f"LOG_DIR={log_dir}",
        "-e", f"MODEL_VERSION={version}",
    ]

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
            env_args.extend(["-e", f"{env_var}={val}"])

    cmd_list = [
        "hf", "jobs", "run",
        "--flavor", flavor,
        *env_args,
        "tensorflow/tensorflow:2.16.1-gpu",
        "bash", "-c", container_cmd
    ]

    # Render formatted shell string
    env_str_lines = []
    for i in range(0, len(env_args), 2):
        env_str_lines.append(f"  {env_args[i]} {env_args[i+1]}")
    envs_str = " \\\n".join(env_str_lines)

    cmd_display = (
        f"hf jobs run \\\n"
        f"  --flavor {flavor} \\\n"
        f"{envs_str} \\\n"
        f"  tensorflow/tensorflow:2.16.1-gpu \\\n"
        f"  bash -c \"{container_cmd}\""
    )

    return cmd_display, cmd_list


def main():
    import subprocess

    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    default_flavor = hf_cfg.get("hardware_flavor", "t4-small")
    default_epochs = train_cfg.get("epochs", 50)
    default_batch_size = train_cfg.get("batch_size", data_cfg.get("batch_size", 16))
    default_version = os.getenv("MODEL_VERSION", train_cfg.get("model_version", model_cfg.get("version", "v2.0_official_dgnet")))

    parser = argparse.ArgumentParser(description="Hugging Face Cloud GPU Job Launcher")
    parser.add_argument("--flavor", type=str, default=default_flavor, help="HF Hardware flavor (e.g., t4-small, t4-medium, a10g-small, a100-large)")
    parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of epochs to train on GPU")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size")
    parser.add_argument("--version", type=str, default=default_version, help="Model version tag (e.g. v2.0_official_dgnet)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--run", action="store_true", help="Execute the job immediately using HF CLI")
    parser.add_argument("--detach", "-d", action="store_true", help="Run job in background and print Job ID")

    args = parser.parse_args()

    cmd_display, cmd_list = generate_job_command(
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        flavor=args.flavor,
        version=args.version
    )

    if args.detach:
        cmd_list.insert(3, "--detach")

    print("\n" + "=" * 70)
    print("      Hugging Face Cloud GPU Training Job Generator / Runner")
    print("=" * 70)
    print("\nTarget Hardware Flavor: ", args.flavor)
    print("Model Version:          ", args.version)
    print("Epochs / Batch Size:    ", f"{args.epochs} / {args.batch_size}")
    print("\nJob Command:\n")
    print(cmd_display)
    print("\n" + "=" * 70)

    if args.run:
        print("\n[Launching Job] Executing command via Hugging Face CLI...")
        try:
            res = subprocess.run(cmd_list)
            if res.returncode == 0:
                print("\n[HF Jobs] Job submitted successfully!")
                print("Run 'hf jobs ps' to list active jobs or 'hf jobs logs <job_id>' to inspect live output.")
            else:
                print(f"\n[HF Jobs] Process exited with return code: {res.returncode}")
        except Exception as e:
            print(f"\n[HF Jobs] Error running job: {e}")
    else:
        print("\nTo trigger this job immediately, run:")
        print(f"    python scripts/launch_hf_job.py --run --flavor {args.flavor} --epochs {args.epochs}")
        print("Or copy and paste the command above directly into your terminal.\n")


if __name__ == "__main__":
    main()

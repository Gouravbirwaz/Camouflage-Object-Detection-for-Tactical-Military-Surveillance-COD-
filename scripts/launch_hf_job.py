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


def generate_job_command(epochs: int = 50, batch_size: int = 16, resume: bool = False, flavor: str = "a100-large") -> str:
    hf_token = os.getenv("HF_TOKEN", "<YOUR_HF_TOKEN>")
    repo_id = os.getenv("HF_REPO_ID", "gouravbirwaz/COD_dataset")

    resume_flag = "--resume" if resume else ""

    container_cmd = (
        f"pip install -r requirements.txt && "
        f"python scripts/hf_sync.py --download-checkpoints --ckpt-dir models/raw_checkpoints --download-dataset && "
        f"python scripts/DGnet.py --train --epochs {epochs} --batch-size {batch_size} {resume_flag} && "
        f"python scripts/hf_sync.py --upload-checkpoints --ckpt-dir models/raw_checkpoints --upload-logs --log-dir metrics"
    )

    hf_cli_cmd = (
        f"hf job run \\\n"
        f"  --flavor {flavor} \\\n"
        f"  --env HF_TOKEN={hf_token} \\\n"
        f"  --env HF_REPO_ID={repo_id} \\\n"
        f"  --env DATA_DIR=data/raw/dataset-splitM \\\n"
        f"  --env CHECKPOINT_DIR=models/raw_checkpoints \\\n"
        f"  --env LOG_DIR=metrics \\\n"
        f"  --command \"{container_cmd}\""
    )

    return hf_cli_cmd


def main():
    parser = argparse.ArgumentParser(description="Hugging Face A100 GPU Job Helper")
    parser.add_argument("--flavor", type=str, default="a100-large", help="HF Hardware flavor (e.g., a100-large, a100-small, t4-medium)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train on GPU")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    cmd = generate_job_command(epochs=args.epochs, batch_size=args.batch_size, resume=args.resume, flavor=args.flavor)

    print("\n" + "=" * 70)
    print("      Hugging Face GPU Job Command Generator (NVIDIA A100 Training)")
    print("=" * 70)
    print("\nExecute the following command in your terminal / HF CLI:\n")
    print(cmd)
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

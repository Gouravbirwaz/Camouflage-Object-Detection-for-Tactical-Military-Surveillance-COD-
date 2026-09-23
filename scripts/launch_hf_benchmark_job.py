#!/usr/bin/env python3
"""
Hugging Face GPU Training Job Launcher for Kaggle Camouflage Benchmark Pretraining
Triggers DGNet pretraining on public camouflage datasets (COD10K, NC4K, CAMO)
on remote Hugging Face GPU Jobs (e.g. T4 / A100), automatically uploads weights to HF,
and allows seamless local download for military fine-tuning.

Datasets:
- COD10K: https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset
- NC4K:   https://www.kaggle.com/datasets/ivanomelchenkoim11/nc4k-dataset
- CAMO:   https://www.kaggle.com/datasets/danhnt/camo-fs-dataset

Usage:
    # 1. Print the HF Jobs command:
    python scripts/launch_hf_benchmark_job.py --flavor t4-small --dataset cod10k --epochs 30

    # 2. Launch the job immediately on Hugging Face:
    python scripts/launch_hf_benchmark_job.py --flavor t4-small --dataset cod10k --epochs 30 --run

    # 3. Launch on NVIDIA A100 with all 3 datasets combined:
    python scripts/launch_hf_benchmark_job.py --flavor a100-large --dataset all --epochs 40 --run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_project_config


def generate_job_command(
    dataset: str = "cod10k",
    epochs: int = 30,
    batch_size: int = 16,
    flavor: str = "t4-small",
    backbone: str = "mobilenet_v3_large"
):
    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})

    flavor = flavor or hf_cfg.get("hardware_flavor", "t4-small")
    space_id = os.getenv("HF_SPACE_ID", hf_cfg.get("space_id", "gouravbirwaz/cod_training"))
    hf_token = os.getenv("HF_TOKEN", hf_cfg.get("token", "<YOUR_HF_TOKEN>"))
    upload_repo_id = os.getenv("HF_UPLOAD_REPO_ID", "yakshithkd/COD_training_artifacts")

    # Ensure latest script is uploaded to HF repo so the container can always fetch it
    if hf_token and upload_repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=upload_repo_id, repo_type="dataset", exist_ok=True)
            script_path = PROJECT_ROOT / "scripts/train_kaggle_benchmark.py"
            if script_path.exists():
                api.upload_file(
                    path_or_fileobj=str(script_path),
                    path_in_repo="scripts/train_kaggle_benchmark.py",
                    repo_id=upload_repo_id,
                    repo_type="dataset"
                )
                print(f"[Launcher] Synced scripts/train_kaggle_benchmark.py to '{upload_repo_id}'")
        except Exception as e_sync:
            print(f"[Launcher] Notice (Pre-sync script): {e_sync}")

    container_cmd = (
        f"apt-get update -qq && "
        f"apt-get install -y -qq git libgl1 libglib2.0-0 && "
        f"(git clone --depth 1 https://oauth2:{hf_token}@huggingface.co/spaces/{space_id} /app 2>/dev/null || "
        f"git clone --depth 1 https://github.com/Gouravbirwaz/Camouflage-Object-Detection-for-Tactical-Military-Surveillance-COD-.git /app) && "
        f"cd /app && "
        f"pip install --no-cache-dir \"numpy<2.0.0\" kagglehub opencv-python-headless huggingface_hub && "
        f"python3 -c \"from huggingface_hub import hf_hub_download; import shutil; (lambda f: shutil.copy2(f, 'scripts/train_kaggle_benchmark.py'))(hf_hub_download('{upload_repo_id}', 'scripts/train_kaggle_benchmark.py', repo_type='dataset', token='{hf_token}'))\" 2>/dev/null || true && "
        f"python3 scripts/train_kaggle_benchmark.py --train --dataset {dataset} --epochs {epochs} --batch-size {batch_size} --backbone {backbone} --upload-checkpoints"
    )

    env_args = [
        "-e", f"HF_TOKEN={hf_token}",
        "-e", f"HF_UPLOAD_REPO_ID={upload_repo_id}",
    ]

    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME"]:
        val = os.getenv(k)
        if val:
            env_args.extend(["-e", f"{k}={val}"])

    cmd_list = [
        "hf", "jobs", "run",
        "--flavor", flavor,
        *env_args,
        "tensorflow/tensorflow:2.16.1-gpu",
        "bash", "-c", container_cmd
    ]

    # Shell display string
    env_str_lines = [f"  {env_args[i]} {env_args[i+1]}" for i in range(0, len(env_args), 2)]
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
    parser = argparse.ArgumentParser(description="Hugging Face Cloud GPU Benchmark Job Launcher")
    parser.add_argument("--dataset", type=str, default="cod10k", choices=["cod10k", "nc4k", "camo", "all"], help="Benchmark dataset (default: cod10k)")
    parser.add_argument("--flavor", type=str, default="t4-small", help="HF Hardware flavor (e.g., t4-small, t4-medium, a10g-small, a100-large)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train on GPU")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large", help="Backbone encoder")
    parser.add_argument("--dry-run", action="store_true", help="Only generate and display the command without launching")
    parser.add_argument("--detach", "-d", action="store_true", help="Run job in background and print Job ID")

    args = parser.parse_args()

    cmd_display, cmd_list = generate_job_command(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        flavor=args.flavor,
        backbone=args.backbone
    )

    if args.detach:
        cmd_list.insert(3, "--detach")

    print("\n" + "=" * 70)
    print("      Hugging Face Cloud GPU Benchmark Training Launcher")
    print("=" * 70)
    print(f" Target Hardware Flavor:  {args.flavor}")
    print(f" Benchmark Dataset:       {args.dataset.upper()}")
    print(f" Epochs / Batch Size:     {args.epochs} / {args.batch_size}")
    print(f" Context Backbone:        {args.backbone}")
    print("\nExecuting Job Command:\n")
    print(cmd_display)
    print("\n" + "=" * 70)

    if args.dry_run:
        print("\n[Dry Run] Command generated successfully. Job was NOT submitted.")
        return

    print("\n[Launching HF Job] Submitting job to Hugging Face Cloud GPUs...")
    try:
        res = subprocess.run(cmd_list)
        if res.returncode == 0:
            print("\n[HF Jobs] ✓ Job submitted successfully!")
            print("Track your job with:  hf jobs ps")
            print("Check live logs with: hf jobs logs <job_id>")
            print("\nOnce finished, download the single best checkpoint to your local machine using:")
            print("  python scripts/train_kaggle_benchmark.py --download-checkpoints")
        else:
            print(f"\n[HF Jobs] Process exited with return code: {res.returncode}")
    except Exception as e:
        print(f"\n[HF Jobs] Error running job: {e}")


if __name__ == "__main__":
    main()

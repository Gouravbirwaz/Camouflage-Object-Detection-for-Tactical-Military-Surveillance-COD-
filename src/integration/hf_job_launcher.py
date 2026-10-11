"""
Hugging Face GPU Training & Benchmark Job Launcher.
Configures remote container parameters and triggers GPU training jobs (A100 / T4) on Hugging Face.
Supports both tactical military fine-tuning and public benchmark pretraining (COD10K, NC4K, CAMO).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.utils.config import get_project_root, load_project_config

PROJECT_ROOT = get_project_root()


def generate_job_command(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    resume: bool = False,
    flavor: Optional[str] = None,
    version: Optional[str] = None,
    lr: Optional[float] = None,
    weights: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Generates the hf jobs run command for military training."""
    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    flavor = flavor or hf_cfg.get("hardware_flavor", "t4-small")
    epochs = epochs if epochs is not None else train_cfg.get("epochs", 50)
    batch_size = batch_size if batch_size is not None else train_cfg.get("batch_size", data_cfg.get("batch_size", 16))
    version = version or os.getenv("MODEL_VERSION", train_cfg.get("model_version", model_cfg.get("version", "v2.0_official_dgnet")))
    lr = lr if lr is not None else train_cfg.get("learning_rate", 0.00005)

    space_id = os.getenv("HF_SPACE_ID", hf_cfg.get("space_id", "gouravbirwaz/cod_training"))
    hf_token = os.getenv("HF_TOKEN", hf_cfg.get("token", "<YOUR_HF_TOKEN>"))
    repo_id = os.getenv("HF_REPO_ID", hf_cfg.get("repo_id", "gouravbirwaz/COD_dataset"))
    data_dir = os.getenv("DATA_DIR", data_cfg.get("raw_data_dir", "data/raw/dataset-splitM"))
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", train_cfg.get("checkpoint_dir", "models/raw_checkpoints"))
    log_dir = os.getenv("LOG_DIR", train_cfg.get("log_dir", "metrics"))

    resume_flag = "--resume" if resume else ""
    weights_flag = f"--weights {weights}" if weights else ""
    upload_repo_id = os.getenv("HF_UPLOAD_REPO_ID", "yakshithkd/COD_training_artifacts")

    if hf_token and upload_repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=upload_repo_id, repo_type="dataset", exist_ok=True)
            dgnet_script = PROJECT_ROOT / "scripts/DGnet.py"
            if dgnet_script.exists():
                api.upload_file(
                    path_or_fileobj=str(dgnet_script),
                    path_in_repo="scripts/DGnet.py",
                    repo_id=upload_repo_id,
                    repo_type="dataset"
                )
        except Exception:
            pass

    container_cmd = (
        f"apt-get update -qq && "
        f"apt-get install -y -qq git libgl1 libglib2.0-0 && "
        f"(git clone --depth 1 https://oauth2:{hf_token}@huggingface.co/spaces/{space_id} /app 2>/dev/null || "
        f"git clone --depth 1 https://github.com/Gouravbirwaz/Camouflage-Object-Detection-for-Tactical-Military-Surveillance-COD-.git /app) && "
        f"cd /app && "
        f"pip install --no-cache-dir \"numpy<2.0.0\" --ignore-installed blinker && "
        f"pip install --no-cache-dir -r requirements.txt && "
        f"pip install --no-cache-dir hf_xet 2>/dev/null || true && "
        f"python3 -c \"from huggingface_hub import hf_hub_download; import shutil; (lambda f: shutil.copy2(f, 'scripts/DGnet.py'))(hf_hub_download('{upload_repo_id}', 'scripts/DGnet.py', repo_type='dataset', token='{hf_token}'))\" 2>/dev/null || true && "
        f"python3 -c \"from huggingface_hub import hf_hub_download; import os, shutil; os.makedirs('models/raw_checkpoints', exist_ok=True); (lambda f: shutil.copy2(f, 'models/raw_checkpoints/dgnet_benchmark_best.weights.h5'))(hf_hub_download('{upload_repo_id}', 'models/raw_checkpoints/dgnet_benchmark_best.weights.h5', repo_type='dataset', token='{hf_token}'))\" 2>/dev/null || true && "
        f"rm -rf data/raw data/augmented && "
        f"python3 -c \"import zipfile; from huggingface_hub import hf_hub_download; print('[S3/HF Fetch] Downloading real 372MB soldier dataset archive from {repo_id}...'); z = hf_hub_download(repo_id='{repo_id}', repo_type='dataset', filename='data/dataset_raw.zip'); print('[S3/HF Fetch] Unpacking into data/raw...'); zipfile.ZipFile(z).extractall('data/raw'); print('[S3/HF Fetch] SUCCESS: Real soldier dataset ready!')\" && "
        f"python3 scripts/generate_augmented_dataset.py --raw-dir data/raw/dataset-splitM --multiplier 4 && "
        f"python3 scripts/DGnet.py --train --epochs {epochs} --batch-size {batch_size} --lr {lr} --version {version} {resume_flag} {weights_flag} && "
        f"python3 scripts/hf_sync.py --upload-checkpoints --ckpt-dir {checkpoint_dir} --upload-logs --log-dir {log_dir} --repo-id {upload_repo_id}"
    )

    env_args = [
        "-e", f"HF_TOKEN={hf_token}",
        "-e", f"HF_REPO_ID={upload_repo_id}",
        "-e", f"DATA_DIR={data_dir}",
        "-e", f"CHECKPOINT_DIR={checkpoint_dir}",
        "-e", f"LOG_DIR={log_dir}",
        "-e", f"MODEL_VERSION={version}",
    ]

    for k in ["KAGGLE_USERNAME", "KAGGLE_KEY", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "DVC_REMOTE_URL"]:
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

    env_str = " \\\n  ".join([f"{env_args[i]} {env_args[i+1]}" for i in range(0, len(env_args), 2)])
    cmd_display = (
        f"hf jobs run \\\n"
        f"  --flavor {flavor} \\\n"
        f"  {env_str} \\\n"
        f"  tensorflow/tensorflow:2.16.1-gpu \\\n"
        f"  bash -c \"{container_cmd}\""
    )

    return cmd_display, cmd_list


def generate_benchmark_job_command(
    dataset: str = "cod10k",
    epochs: int = 30,
    batch_size: int = 16,
    flavor: str = "t4-small",
    backbone: str = "mobilenet_v3_large"
) -> Tuple[str, List[str]]:
    """Generates the hf jobs run command for Kaggle benchmark pretraining."""
    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})

    flavor = flavor or hf_cfg.get("hardware_flavor", "t4-small")
    space_id = os.getenv("HF_SPACE_ID", hf_cfg.get("space_id", "gouravbirwaz/cod_training"))
    hf_token = os.getenv("HF_TOKEN", hf_cfg.get("token", "<YOUR_HF_TOKEN>"))
    upload_repo_id = os.getenv("HF_UPLOAD_REPO_ID", "yakshithkd/COD_training_artifacts")

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
        except Exception:
            pass

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

    env_str = " \\\n  ".join([f"{env_args[i]} {env_args[i+1]}" for i in range(0, len(env_args), 2)])
    cmd_display = (
        f"hf jobs run \\\n"
        f"  --flavor {flavor} \\\n"
        f"  {env_str} \\\n"
        f"  tensorflow/tensorflow:2.16.1-gpu \\\n"
        f"  bash -c \"{container_cmd}\""
    )

    return cmd_display, cmd_list


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Hugging Face GPU Training Job Launcher")
    parser.add_argument("--benchmark", action="store_true", help="Launch Kaggle benchmark pretraining job")
    parser.add_argument("--dataset", type=str, default="cod10k", help="Kaggle dataset for benchmark (cod10k, nc4k, camo, all)")
    parser.add_argument("--flavor", type=str, default=None, help="Hardware flavor (e.g. t4-small, a100-large)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--version", type=str, default=None, help="Model version tag")
    parser.add_argument("--weights", type=str, default=None, help="Initial weights path")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large", help="Backbone encoder")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--run", action="store_true", help="Execute the job immediately")

    args = parser.parse_args(argv)

    if args.benchmark:
        cmd_display, cmd_list = generate_benchmark_job_command(
            dataset=args.dataset,
            epochs=args.epochs or 30,
            batch_size=args.batch_size or 16,
            flavor=args.flavor or "t4-small",
            backbone=args.backbone
        )
    else:
        cmd_display, cmd_list = generate_job_command(
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=args.resume,
            flavor=args.flavor,
            version=args.version,
            lr=args.lr,
            weights=args.weights
        )

    print("\n" + "=" * 70)
    print("      Hugging Face Cloud GPU Training Job Configuration")
    print("=" * 70)
    print(cmd_display)
    print("=" * 70 + "\n")

    if args.run:
        print("[Launcher] Launching remote job on Hugging Face...")
        try:
            res = subprocess.run(cmd_list)
            if res.returncode == 0:
                print("\n[Launcher] Job submitted successfully to Hugging Face!")
            else:
                print(f"\n[Launcher] Job exited with code {res.returncode}")
        except Exception as e:
            print(f"[Launcher] Error executing job: {e}")


if __name__ == "__main__":
    main()

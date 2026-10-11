"""
Pushes COD_PROJECT code directly to a Hugging Face Space (Docker SDK) for GPU Training.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.utils.config import get_project_root, load_project_config

PROJECT_ROOT = get_project_root()


def push_to_space(space_id: Optional[str] = None, token: Optional[str] = None):
    """Pushes project source and config files to Hugging Face Space."""
    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})

    target_space_id = space_id or os.getenv("HF_SPACE_ID", hf_cfg.get("space_id", "gouravbirwaz/cod_training"))
    target_token = token or os.getenv("HF_TOKEN", hf_cfg.get("token", None))
    space_sdk = hf_cfg.get("space_sdk", "docker")

    if target_token:
        target_token = target_token.strip()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: 'huggingface_hub' is required. Run: pip install huggingface_hub")
        return 1

    api = HfApi(token=target_token) if target_token else HfApi()

    print(f"\n[HF Space Deploy] Checking Hugging Face Space '{target_space_id}'...")

    repo_exists = False
    try:
        api.repo_info(repo_id=target_space_id, repo_type="space", token=target_token)
        repo_exists = True
        print(f"[HF Space Deploy] Space '{target_space_id}' found!")
    except Exception:
        print(f"[HF Space Deploy] Space '{target_space_id}' not found yet. Attempting creation...")

    if not repo_exists:
        try:
            api.create_repo(
                repo_id=target_space_id,
                repo_type="space",
                space_sdk=space_sdk,
                exist_ok=True,
                token=target_token
            )
            print(f"[HF Space Deploy] Successfully created Space '{target_space_id}'!")
        except Exception as e:
            print("\n" + "!" * 70)
            print(f"[HF Space Deploy] Automatic Space Creation Note: {e}")
            print("\nPlease initialize the Docker Space via web UI if not already present.")
            print("!" * 70 + "\n")
            return 1

    ignore_patterns = [
        ".git/*",
        ".venv/*",
        "venv/*",
        "env/*",
        ".env*",
        ".dvc/config.local*",
        ".dvc/cache/*",
        "data/raw/**",
        "data/augmented/**",
        "data/processed/**",
        "data/interim/**",
        "data/synthetic/**",
        "models/*",
        "experiments/*",
        "metrics/*",
        "logs/*",
        "*.weights.h5",
        "*.h5",
        "*.keras",
        "*.tflite",
        "*.onnx",
        "*.tar.gz",
        "*.zip",
        "__pycache__/*",
        "*.pyc",
    ]

    print(f"[HF Space Deploy] Uploading source files to Space '{target_space_id}'...")
    try:
        api.upload_folder(
            folder_path=str(PROJECT_ROOT),
            repo_id=target_space_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            token=target_token
        )
        print(f"\n[HF Space Deploy] SUCCESS: Code pushed to https://huggingface.co/spaces/{target_space_id}\n")
        return 0
    except Exception as e:
        print(f"[HF Space Deploy] Push failed: {e}")
        return 1


def main(argv: Optional[List[str]] = None):
    cfg = load_project_config()
    hf_cfg = cfg.get("huggingface", {})

    default_space_id = os.getenv("HF_SPACE_ID", hf_cfg.get("space_id", "gouravbirwaz/cod_training"))
    default_token = os.getenv("HF_TOKEN", hf_cfg.get("token", None))

    parser = argparse.ArgumentParser(description="Push COD_PROJECT code to Hugging Face Space (Docker SDK)")
    parser.add_argument("--space-id", type=str, default=default_space_id, help="HF Space ID")
    parser.add_argument("--token", type=str, default=default_token, help="Hugging Face write token")

    args = parser.parse_args(argv)
    return push_to_space(space_id=args.space_id, token=args.token)


if __name__ == "__main__":
    sys.exit(main())

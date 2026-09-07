#!/usr/bin/env python3
"""
Pushes COD_PROJECT code directly to a Hugging Face Space (Docker SDK) for GPU Training.

Usage:
    python scripts/push_to_hf_space.py --space-id gouravbirwaz/cod_training
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

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: 'huggingface_hub' is required. Run: pip install huggingface_hub")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Push COD_PROJECT code to Hugging Face Space (Docker SDK)")
    parser.add_argument("--space-id", type=str, default=os.getenv("HF_SPACE_ID", "gouravbirwaz/cod_training"), help="HF Space ID (e.g. gouravbirwaz/cod_training)")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN", None), help="Hugging Face write token")

    args = parser.parse_args()

    token = args.token.strip() if args.token and args.token.strip() else None
    space_id = args.space_id

    if not token:
        print("Error: HF_TOKEN is required. Set it in .env or pass --token YOUR_TOKEN")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"\n[HF Space Deploy] Checking Hugging Face Space '{space_id}'...")

    repo_exists = False
    try:
        api.repo_info(repo_id=space_id, repo_type="space", token=token)
        repo_exists = True
        print(f"[HF Space Deploy] Space '{space_id}' found!")
    except Exception:
        print(f"[HF Space Deploy] Space '{space_id}' not found yet. Attempting creation...")

    if not repo_exists:
        try:
            api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="docker",
                exist_ok=True,
                token=token
            )
            print(f"[HF Space Deploy] Successfully created Space '{space_id}'!")
        except Exception as e:
            print("\n" + "!" * 70)
            print(f"[HF Space Deploy] Automatic Space Creation Note: {e}")
            print("\nHugging Face requires Docker Spaces to be initialized on the web interface first.")
            print("Please perform these 1-time setup steps in your browser:\n")
            print(f" 1. Open: https://huggingface.co/new-space")
            print(f" 2. Set Space Name: {space_id.split('/')[-1]}")
            print(f" 3. Set Space SDK: Docker")
            print(f" 4. Set Space Hardware: NVIDIA T4 Small (or A10G)")
            print(f" 5. Click 'Create Space'")
            print(f"\nAfter creating the Space on the web UI, re-run this script!")
            print("!" * 70 + "\n")
            sys.exit(1)

    ignore_patterns = [
        ".git/*",
        ".venv/*",
        "venv/*",
        "env/*",
        ".env*",
        ".dvc/config.local*",
        ".dvc/cache/*",
        "data/*",
        "models/*",
        "metrics/*",
        "__pycache__/*",
        "*.pyc",
        "*.pdf",
        "*.h5",
        "*.keras",
        "*.tflite",
        "*.zip",
        "*.tar",
        "*.gz"
    ]

    print(f"[HF Space Deploy] Uploading project code to Hugging Face Space '{space_id}'...")
    api.upload_folder(
        folder_path=str(PROJECT_ROOT),
        repo_id=space_id,
        repo_type="space",
        token=token,
        ignore_patterns=ignore_patterns
    )

    print("\n" + "=" * 70)
    print("      SUCCESS: Project Code Pushed to Hugging Face Docker Space!")
    print("=" * 70)
    print(f"Space URL:    https://huggingface.co/spaces/{space_id}")
    print(f"Settings:     https://huggingface.co/spaces/{space_id}/settings")
    print("\nNext Steps:")
    print(" 1. Open your Space Settings link above.")
    print(" 2. In 'Space Secrets', add: HF_TOKEN = <your_huggingface_token>")
    print(" 3. In 'Space Hardware', select: NVIDIA T4 Small (or A10G)")
    print(" 4. The Space will automatically build the container and run GPU training!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

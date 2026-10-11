#!/usr/bin/env python3
"""
Hugging Face Storage Bucket Synchronization Script for COD_PROJECT
CLI Entrypoint delegating to src.integration.hf_sync.
"""

import sys
from pathlib import Path

# Add project root directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.hf_sync import (
    get_api,
    _resolve_path,
    upload_checkpoints,
    download_checkpoints,
    upload_logs,
    download_logs,
    upload_dataset,
    download_dataset,
    sync_all,
    main,
)

if __name__ == "__main__":
    main()

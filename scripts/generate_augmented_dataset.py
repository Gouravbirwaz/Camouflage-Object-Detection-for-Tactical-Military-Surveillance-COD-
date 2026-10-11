#!/usr/bin/env python3
"""
Military Tactical Offline Dataset Generator - Camouflaged Object Detection (COD)
CLI Entrypoint delegating to src.data.offline_augmentation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.offline_augmentation import (
    discover_raw_training_pairs,
    generate_augmented_dataset,
    main,
)

if __name__ == "__main__":
    main()

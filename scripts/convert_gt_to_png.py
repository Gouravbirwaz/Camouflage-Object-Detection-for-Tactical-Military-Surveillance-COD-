#!/usr/bin/env python3
"""
Convert Ground Truth (GT) mask images from JPEG (.jpg) to lossless PNG (.png) format.
CLI Entrypoint delegating to src.data.mask_converter.
"""

import sys
from pathlib import Path

# Add project root directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mask_converter import (
    main,
    setup_logger,
    find_jpg_files,
    convert_single_image,
    process_directory,
    parse_arguments,
    run_gt_conversion,
    DEFAULT_TARGET_DIRS,
    DEFAULT_BACKUP_DIR,
)

if __name__ == "__main__":
    sys.exit(main())

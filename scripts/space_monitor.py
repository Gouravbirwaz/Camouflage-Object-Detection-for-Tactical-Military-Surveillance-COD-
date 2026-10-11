#!/usr/bin/env python3
"""
Lightweight HTTP Status Monitor for Hugging Face Space (Port 7860)
CLI Entrypoint delegating to src.integration.space_monitor.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.space_monitor import (
    PORT,
    StatusHandler,
    run_monitor,
    main,
)

if __name__ == "__main__":
    main()

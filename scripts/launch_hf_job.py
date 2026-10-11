#!/usr/bin/env python3
"""
Hugging Face GPU Training Job Launcher for COD_PROJECT
CLI Entrypoint delegating to src.integration.hf_job_launcher.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.hf_job_launcher import (
    generate_job_command,
    main,
)

if __name__ == "__main__":
    main()

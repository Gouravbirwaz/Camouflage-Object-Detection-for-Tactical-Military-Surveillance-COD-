#!/usr/bin/env python3
"""
Pushes COD_PROJECT code directly to a Hugging Face Space (Docker SDK) for GPU Training.
CLI Entrypoint delegating to src.integration.hf_deploy.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.hf_deploy import push_to_space, main

if __name__ == "__main__":
    sys.exit(main())

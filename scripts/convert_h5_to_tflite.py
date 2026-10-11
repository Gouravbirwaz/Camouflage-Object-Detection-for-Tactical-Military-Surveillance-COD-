#!/usr/bin/env python3
"""
High-Fidelity H5 / Checkpoint to TensorFlow Lite (.tflite) Converter
CLI Entrypoint delegating to src.optimization.tflite_converter.
"""

import sys
from pathlib import Path

# Add project root directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.tflite_converter import (
    ensure_weights_available,
    collect_calibration_images,
    build_tflite_inference_model,
    convert_h5_to_tflite,
    verify_tflite_parity,
    upload_to_hf,
    generate_hf_conversion_command,
    main,
)

if __name__ == "__main__":
    main()

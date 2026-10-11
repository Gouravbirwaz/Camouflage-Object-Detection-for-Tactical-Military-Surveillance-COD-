"""
Hardware and GPU Environment Detection Utilities.
"""

from typing import List
import tensorflow as tf


def setup_gpu_environment() -> List[str]:
    """
    Detects physical GPUs and configures TensorFlow memory growth.

    Returns:
        List of detected GPU device names.
    """
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = []
    if gpus:
        print(f"\n[GPU Hardware Detection] Found {len(gpus)} GPU(s):")
        for idx, gpu in enumerate(gpus):
            gpu_names.append(gpu.name)
            print(f"  • GPU [{idx}]: {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"    Notice (Memory Growth): {e}")
    else:
        print("\n[GPU Hardware Detection] No physical GPU detected by TensorFlow. Running on CPU.")
    return gpu_names

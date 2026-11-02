"""
Camouflaged Object Detection (COD) for Tactical Military Surveillance.
Root package providing models, data processing, training, inference, optimization,
tracking, visualization, and cloud integration modules.
"""

import importlib

__version__ = "2.0.0"

__all__ = [
    "utils",
    "data",
    "models",
    "training",
    "inference",
    "optimization",
    "visualization",
    "tracking",
    "integration",
    "api",
]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

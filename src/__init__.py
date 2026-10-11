"""
Camouflaged Object Detection (COD) for Tactical Military Surveillance.
Root package providing models, data processing, training, inference, optimization,
tracking, visualization, and cloud integration modules.
"""

from . import utils
from . import data
from . import models
from . import training
from . import inference
from . import optimization
from . import visualization
from . import tracking
from . import integration

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
]

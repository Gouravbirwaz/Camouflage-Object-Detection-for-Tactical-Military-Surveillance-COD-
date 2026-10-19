"""
Edge AI Inference Service and Web API Package for DGNet Camouflaged Object Detection.
"""

from src.api.runtime import EdgeInferenceRuntime
from src.api.main import app

__all__ = ["EdgeInferenceRuntime", "app"]

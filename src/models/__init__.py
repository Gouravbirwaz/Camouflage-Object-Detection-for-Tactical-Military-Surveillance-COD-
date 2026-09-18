"""
Source Models Package for COD Framework (TensorFlow 2.x)
Exposes official DGNet architecture, loss functions, and component modules.
"""

from .dgnet import (
    DGNet,
    DGNetLoss,
    ConvBR,
    DimensionalReduction,
    SoftGroupingStrategy,
    GradientInducedTransition,
    NeighborConnectionDecoder,
    TextureEncoder,
    ContextEncoder,
    build_dgnet_model,
)

# Aliases for backward compatibility
ContextBranch = ContextEncoder
TextureBranch = TextureEncoder
PartialDecoder = NeighborConnectionDecoder

__all__ = [
    "DGNet",
    "DGNetLoss",
    "ConvBR",
    "DimensionalReduction",
    "SoftGroupingStrategy",
    "GradientInducedTransition",
    "NeighborConnectionDecoder",
    "TextureEncoder",
    "ContextEncoder",
    "ContextBranch",
    "TextureBranch",
    "PartialDecoder",
    "build_dgnet_model",
]

"""Continuous Guided-Attention Transformer with Bivariate Gaussian MDN.

Completely isolated from the canonical Alex Graves (2013) baseline.
"""

from .config import NewModelConfig
from .model import ContinuousGuidedTransformer
from .loss import ContinuousGuidedSynthesisLoss

# Backward compatibility aliases
GuidedHandwritingTransformer = ContinuousGuidedTransformer
GuidedSynthesisLoss = ContinuousGuidedSynthesisLoss

__all__ = [
    "NewModelConfig",
    "ContinuousGuidedTransformer",
    "ContinuousGuidedSynthesisLoss",
    "GuidedHandwritingTransformer",
    "GuidedSynthesisLoss",
]

"""
resonance_folding
=================
Octonion weight representation and geodesic model merging
for native 8-channel convolutional networks.

Key results (CIFAR-10, OctConvNet-M, March 2026):
  - RF fold: cos=1.000000, holo=0.000000, accuracy delta=0.0000
  - Oct-SLERP at t=0.20 beats both individual fine-tunes (+0.19%)
    and float averaging (+0.43%) simultaneously

Quickstart
----------
>>> import torch
>>> from resonance_folding import OctConvNet, fold_model, slerp_merge, verify_fold
>>>
>>> # Build and train a native OctConvNet
>>> model = OctConvNet(size="M", n_classes=10)
>>>
>>> # Fold weights onto S⁷ (lossless for native 8-ch networks)
>>> folded = fold_model(model)
>>> verify_fold(model, folded)
>>>
>>> # Merge two trained checkpoints via geodesic SLERP
>>> merged, holo = slerp_merge(model_a, model_b, t=0.20)
>>> print(f"Holographic coherence: {holo:.6f}")  # → 0.000000
"""

__version__ = "0.1.0"
__author__  = "Daniel Frokido"
__license__ = "Apache-2.0"

from .algebra import (
    oct_mul,
    oct_conj,
    oct_normalize,
    holo_loss,
    assoc_loss,
    oct_slerp,
    task_vector,
    task_vector_apply,
)

from .arch import (
    OctBlock,
    OctConvNet,
    oct_init_,
)

from .fold import (
    FoldedLayer,
    fold_model,
    patch_model,
    verify_fold,
    weight_to_oct,
    oct_to_weight,
)

from .merge import (
    float_average,
    slerp_merge,
    slerp_sweep,
    triple_slerp,
)

__all__ = [
    # algebra
    "oct_mul", "oct_conj", "oct_normalize",
    "holo_loss", "assoc_loss",
    "oct_slerp", "task_vector", "task_vector_apply",
    # arch
    "OctBlock", "OctConvNet", "oct_init_",
    # fold
    "FoldedLayer", "fold_model", "patch_model",
    "verify_fold", "weight_to_oct", "oct_to_weight",
    # merge
    "float_average", "slerp_merge", "slerp_sweep", "triple_slerp",
]

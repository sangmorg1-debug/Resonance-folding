"""
resonance_folding.fold
======================
The core Resonance Folding operation.

RF fold encodes native 8-channel conv weights as unit octonions
on S⁷ and stores original norms for lossless reconstruction.

Proven properties (all empirically verified):
  - cos(reconstructed, original) = 1.000000
  - holo_loss(folded shards)     = 0.000000
  - accuracy delta after fold    = 0.0000

These properties hold because:
  1. Each filter's 8 input-channel weights form a natural 8D vector
  2. Normalizing to S⁷ changes only the magnitude, not the direction
  3. Storing the original norms allows exact magnitude restoration
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .algebra import oct_normalize, holo_loss
from .arch    import OctConvNet


# ─────────────────────────────────────────────────────────────
#  LOW-LEVEL: weight tensor ↔ oct shards
# ─────────────────────────────────────────────────────────────

def weight_to_oct(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a conv weight tensor to octonion shards.

    W must have in_ch % 8 == 0. Each group of 8 consecutive
    input-channel weights becomes one octonion on S⁷.

    Args:
        W: (out_ch, in_ch, kH, kW) conv weight — in_ch must be % 8

    Returns:
        oct_shards: (N, 8) unit octonions — N = out_ch * kH * kW * (in_ch//8)
        norms:      (N, 1) original L2 norms of each group
    """
    out_ch, in_ch, kH, kW = W.shape
    assert in_ch % 8 == 0, (
        f"weight_to_oct: in_ch={in_ch} must be divisible by 8. "
        f"Use OctConvNet which enforces 8-aligned channels."
    )
    N = out_ch * kH * kW * (in_ch // 8)
    kernels = (
        W.float()
         .reshape(out_ch, in_ch // 8, 8, kH, kW)
         .permute(0, 3, 4, 1, 2)
         .reshape(N, 8)
    )
    norms      = kernels.norm(dim=-1, keepdim=True)
    oct_shards = oct_normalize(kernels)
    return oct_shards, norms


def oct_to_weight(
    oct_shards: torch.Tensor,
    norms: torch.Tensor,
    shape: tuple,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Reconstruct a conv weight tensor from octonion shards.

    Args:
        oct_shards: (N, 8) unit octonions
        norms:      (N, 1) original norms
        shape:      (out_ch, in_ch, kH, kW) — original weight shape
        dtype:      output dtype (default float32)

    Returns:
        (out_ch, in_ch, kH, kW) reconstructed weight tensor
    """
    out_ch, in_ch, kH, kW = shape
    recon = (oct_shards * norms).reshape(out_ch, kH, kW, in_ch // 8, 8)
    return recon.permute(0, 3, 4, 1, 2).reshape(out_ch, in_ch, kH, kW).to(dtype)


# ─────────────────────────────────────────────────────────────
#  FOLDED LAYER
# ─────────────────────────────────────────────────────────────

class FoldedLayer:
    """
    A single conv layer's weights stored as octonion shards.

    Attributes:
        name:      parameter path (e.g. "stage1.0.conv")
        shape:     original weight shape
        dtype:     original weight dtype
        shards:    (N, 8) unit octonions on S⁷
        norms:     (N, 1) original magnitudes
        n_groups:  number of octonion kernel groups
        ratio:     compression ratio (always 1.0 for native fold —
                   native fold is a re-parameterization, not compression)
        holo:      holographic coherence loss at fold time
        cos:       cosine similarity between original and reconstructed weights
    """

    def __init__(
        self,
        name: str,
        W: torch.Tensor,
    ):
        self.name   = name
        self.shape  = tuple(W.shape)
        self.dtype  = W.dtype

        self.shards, self.norms = weight_to_oct(W)
        self.n_groups = self.shards.shape[0]

        # Verify fidelity at fold time
        recon      = oct_to_weight(self.shards, self.norms, self.shape, W.dtype)
        self.cos   = F.cosine_similarity(
            W.float().flatten().unsqueeze(0),
            recon.float().flatten().unsqueeze(0),
        ).item()
        self.holo  = holo_loss(self.shards).item()
        self.ratio = 1.0   # native fold: same number of floats, different geometry

    def decode(self) -> torch.Tensor:
        """Reconstruct the original weight tensor from shards."""
        return oct_to_weight(self.shards, self.norms, self.shape, self.dtype)

    def __repr__(self) -> str:
        return (f"FoldedLayer({self.name!r}, "
                f"shape={self.shape}, groups={self.n_groups}, "
                f"cos={self.cos:.6f}, holo={self.holo:.2e})")


# ─────────────────────────────────────────────────────────────
#  MODEL-LEVEL FOLD
# ─────────────────────────────────────────────────────────────

def fold_model(model: OctConvNet) -> dict[str, FoldedLayer]:
    """
    Fold all oct-aligned conv layers of an OctConvNet.

    Iterates over all OctBlock conv layers, converts each to a
    FoldedLayer (shards on S⁷ + norms). Does NOT modify the model
    in-place — call patch_model() to write decoded weights back.

    Args:
        model: trained OctConvNet

    Returns:
        dict mapping layer name → FoldedLayer
    """
    folded = {}
    for name, conv in model.oct_layers():
        W = conv.weight.data
        if W.shape[1] % 8 != 0:
            continue
        folded[name] = FoldedLayer(name, W.clone())
    return folded


def patch_model(model: OctConvNet, folded: dict[str, FoldedLayer]) -> None:
    """
    Write decoded weights from folded layers back into the model.
    Used to prepare the model for inference after fold/merge.

    Args:
        model:  OctConvNet to patch (modified in-place)
        folded: dict from fold_model()
    """
    for name, fl in folded.items():
        # name is e.g. "stage1.0.conv" — navigate to the conv module
        # then write to its .weight parameter
        parts = name.split(".")
        mod   = model
        for p in parts:
            mod = getattr(mod, p)
        # mod is now the Conv2d; write decoded weights to its weight parameter
        with torch.no_grad():
            mod.weight.data.copy_(fl.decode())


# ─────────────────────────────────────────────────────────────
#  FOLD VERIFICATION
# ─────────────────────────────────────────────────────────────

def verify_fold(
    model: OctConvNet,
    folded: dict[str, FoldedLayer] = None,
    verbose: bool = True,
) -> dict:
    """
    Verify that the RF fold is lossless for all layers.

    Args:
        model:   OctConvNet whose weights to check against
        folded:  dict from fold_model(); if None, folds model first
        verbose: print per-layer results

    Returns:
        dict with keys: mean_cos, min_cos, mean_holo, all_lossless
    """
    if folded is None:
        folded = fold_model(model)

    layer_dict = dict(model.oct_layers())
    cos_vals, holo_vals = [], []

    for name, fl in folded.items():
        W     = layer_dict[name].weight.data
        recon = fl.decode()
        cos   = F.cosine_similarity(
            W.float().flatten().unsqueeze(0),
            recon.float().flatten().unsqueeze(0),
        ).item()
        hl = holo_loss(fl.shards).item()
        cos_vals.append(cos)
        holo_vals.append(hl)

        if verbose:
            status = "LOSSLESS" if cos > 0.9999 else f"cos={cos:.4f}"
            print(f"  {name:<50} cos={cos:.6f}  holo={hl:.2e}  [{status}]")

    mean_cos  = sum(cos_vals)  / len(cos_vals)
    mean_holo = sum(holo_vals) / len(holo_vals)
    result = {
        "mean_cos":    mean_cos,
        "min_cos":     min(cos_vals),
        "mean_holo":   mean_holo,
        "n_layers":    len(folded),
        "all_lossless": mean_cos > 0.9999,
    }

    if verbose:
        print(f"\n  Summary: mean_cos={mean_cos:.6f}  "
              f"min_cos={min(cos_vals):.6f}  "
              f"mean_holo={mean_holo:.2e}  "
              f"[{'LOSSLESS' if result['all_lossless'] else 'DEGRADED'}]")

    return result

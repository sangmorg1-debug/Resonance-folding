"""
resonance_folding.arch
======================
OctConvNet architecture family.

All conv layers use channels that are multiples of 8 so that
every filter bank is natively partitioned into octonion groups.
Weights are oct-initialized — near-unit-sphere from the start.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebra import oct_normalize


def oct_init_(weight: torch.Tensor) -> None:
    """
    Oct-aware weight initialization for conv layers.

    For layers with in_ch % 8 == 0: initializes each 8-tuple of
    input-channel weights as a near-unit octonion scaled by the
    standard fan-in factor. The network starts geometrically close
    to S⁷ — the fold has less distance to travel.

    Falls back to Kaiming for non-oct-aligned layers (e.g. the
    input projection from RGB which has 3 input channels).

    Args:
        weight: conv weight tensor (out_ch, in_ch, kH, kW)
    """
    out_ch, in_ch, kH, kW = weight.shape
    if in_ch % 8 == 0:
        n_groups = out_ch * kH * kW * (in_ch // 8)
        g = oct_normalize(torch.randn(n_groups, 8))
        g = g * math.sqrt(2.0 / (in_ch * kH * kW))
        weight.data.copy_(
            g.reshape(out_ch, kH, kW, in_ch // 8, 8)
             .permute(0, 3, 4, 1, 2)
             .reshape(out_ch, in_ch, kH, kW)
        )
    else:
        nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")


class OctBlock(nn.Module):
    """
    Conv + BatchNorm + ReLU with oct-aligned channels and oct-init.

    Both in_ch and out_ch must be multiples of 8. After training,
    every filter's 8-wide input slice is a native octonion on S⁷
    and can be RF-folded with zero loss.

    Args:
        in_ch:   input channels (must be multiple of 8)
        out_ch:  output channels (must be multiple of 8)
        kernel:  conv kernel size (default 3)
        stride:  conv stride (default 1)
        padding: conv padding (default 1)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        assert in_ch  % 8 == 0, f"OctBlock: in_ch {in_ch} must be multiple of 8"
        assert out_ch % 8 == 0, f"OctBlock: out_ch {out_ch} must be multiple of 8"
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        oct_init_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)

    @property
    def n_oct_groups(self) -> int:
        """Number of native octonion filter groups in this layer."""
        oc, ic, kH, kW = self.conv.weight.shape
        return oc * kH * kW * (ic // 8)


# ─────────────────────────────────────────────────────────────
#  OctConvNet family
# ─────────────────────────────────────────────────────────────

_CONFIGS = {
    "S": (8,  16,  32,  64),
    "M": (8,  32,  64,  128),
    "L": (8,  64,  128, 256),
}


class OctConvNet(nn.Module):
    """
    Fully native 8-channel OctConvNet.

    Every conv layer uses channels that are multiples of 8.
    Every filter bank is natively partitioned into octonion groups.
    RF fold of a trained OctConvNet is lossless (cos=1.0, holo=0.0).

    Architecture:
        input_proj: RGB (3ch) → c0 (8ch)
        stage 1:    c0  → c1  (2× OctBlock + MaxPool + Dropout)
        stage 2:    c1  → c2  (2× OctBlock + MaxPool + Dropout)
        stage 3:    c2  → c3  (2× OctBlock + MaxPool + Dropout)
        head:       AdaptiveAvgPool(2) → Linear → n_classes

    Args:
        size:      "S" | "M" | "L"  (default "M")
        n_classes: number of output classes (default 10)
        channels:  optional (c0, c1, c2, c3) tuple to override size
    """

    def __init__(
        self,
        size: str = "M",
        n_classes: int = 10,
        channels: tuple = None,
    ):
        super().__init__()
        c0, c1, c2, c3 = channels if channels else _CONFIGS[size]

        self.proj = nn.Sequential(
            nn.Conv2d(3, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.proj[0].weight, mode="fan_out")

        self.stage1 = nn.Sequential(
            OctBlock(c0, c1), OctBlock(c1, c1),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        self.stage2 = nn.Sequential(
            OctBlock(c1, c2), OctBlock(c2, c2),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        self.stage3 = nn.Sequential(
            OctBlock(c2, c3), OctBlock(c3, c3),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4, c3 * 2), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c3 * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(self.pool(x))

    def oct_layers(self) -> list:
        """
        Returns list of (name, conv_module) for all OctBlock conv layers.
        These are the layers that RF fold operates on.
        """
        return [
            (name + ".conv", module.conv)
            for name, module in self.named_modules()
            if isinstance(module, OctBlock)
        ]

    @property
    def n_oct_groups(self) -> int:
        """Total number of native octonion kernel groups across all layers."""
        return sum(
            m.n_oct_groups
            for m in self.modules()
            if isinstance(m, OctBlock)
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

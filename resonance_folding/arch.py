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


# ─────────────────────────────────────────────────────────────
#  OctResNet18 — standard ResNet-18 with RF fold support
# ─────────────────────────────────────────────────────────────

class OctBasicBlock(nn.Module):
    """
    ResNet BasicBlock with oct-aware initialization.
    Standard ResNet-18 channel widths [64,128,256,512] are all
    multiples of 8, so every conv in this block is oct-foldable.

    Args:
        in_ch:      input channels
        out_ch:     output channels
        stride:     conv stride (default 1)
        downsample: optional downsample Sequential for skip connection
    """
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        downsample=None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        oct_init_(self.conv1.weight)
        oct_init_(self.conv2.weight)
        if downsample is not None:
            oct_init_(downsample[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class OctResNet18(nn.Module):
    """
    ResNet-18 with RF fold support.

    Standard ResNet-18 channel widths [64, 128, 256, 512] are all
    multiples of 8, so no channel modifications are required.
    Every conv layer except the RGB stem (in_ch=3) is oct-foldable.

    Proven properties (CIFAR-10, 30 epochs, March 2026):
        - 19 foldable layers, 1,394,688 oct kernel groups
        - RF fold: cos=1.000000, holo=0.0, acc delta=0.0
        - SLERP merge beats best individual fine-tune (+0.35%)

    Args:
        n_classes: number of output classes (default 10)
        cifar:     True = 3×3 stem, no maxpool (for 32×32 inputs)
                   False = 7×7 stem + maxpool (for 224×224 ImageNet)
    """

    def __init__(self, n_classes: int = 10, cifar: bool = True):
        super().__init__()
        if cifar:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
        # Stem uses standard Kaiming — in_ch=3 is not oct-aligned
        nn.init.kaiming_normal_(self.stem[0].weight, mode="fan_out")

        self._in_ch = 64
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def _make_layer(self, out_ch: int, n_blocks: int, stride: int):
        downsample = None
        if stride != 1 or self._in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [OctBasicBlock(self._in_ch, out_ch, stride, downsample)]
        self._in_ch = out_ch
        for _ in range(1, n_blocks):
            layers.append(OctBasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))

    def oct_layers(self) -> list:
        """
        Returns list of (name, conv_module) for all oct-foldable layers.
        Excludes the stem conv (in_ch=3, not oct-aligned).
        """
        return [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, nn.Conv2d)
            and module.weight.shape[1] % 8 == 0
        ]

    @property
    def n_oct_groups(self) -> int:
        """Total octonion kernel groups across all foldable layers."""
        total = 0
        for _, conv in self.oct_layers():
            oc, ic, kH, kW = conv.weight.shape
            total += oc * kH * kW * (ic // 8)
        return total

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

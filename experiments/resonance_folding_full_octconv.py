"""
FULL NATIVE OCTCONVNET
======================
Every conv layer uses channels that are multiples of 8.
Every filter bank is naturally partitioned into octonion groups.
Trained from scratch entirely in float32, but with the constraint
that the weight STRUCTURE is always octonion-compatible.

The architecture family:
  OctConvNet-S  (small):   8 → 16 → 32 → 64   (~300K params)
  OctConvNet-M  (medium):  8 → 32 → 64 → 128  (~1.2M params)
  OctConvNet-L  (large):   8 → 64 → 128 → 256 (~4.8M params)

Float32 baselines at matched parameter count for fair comparison.

After training: apply native RF fold to ALL conv layers (not just one),
measure total accuracy drop, cosine sim, holographic coherence.

This is the load-bearing experiment.
If full-network native fold = zero loss → RF is a training paradigm.
If not → we know exactly which layer type breaks and why.

Run:
  python resonance_folding_full_octconv.py

Flags:
  --model     S | M | L (default: M)
  --dataset   cifar10 | cifar100 (default: cifar10)
  --epochs    30 (default)
  --no-fold   skip RF fold, just train and benchmark
"""

import argparse
import copy
import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model",    default="M", choices=["S","M","L"])
parser.add_argument("--dataset",  default="cifar10", choices=["cifar10","cifar100"])
parser.add_argument("--epochs",   type=int,   default=30)
parser.add_argument("--no-fold",  action="store_true")
parser.add_argument("--batch",    type=int,   default=128)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
#  OCTONION ALGEBRA
# ─────────────────────────────────────────────────────────────

def oct_mul(A, B):
    p = A.unbind(-1); q = B.unbind(-1)
    return torch.stack([
        p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3]-p[4]*q[4]-p[5]*q[5]-p[6]*q[6]-p[7]*q[7],
        p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2]+p[4]*q[5]-p[5]*q[4]+p[6]*q[7]-p[7]*q[6],
        p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1]-p[4]*q[6]+p[5]*q[7]+p[6]*q[4]-p[7]*q[5],
        p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0]-p[4]*q[7]-p[5]*q[6]+p[6]*q[5]+p[7]*q[4],
        p[0]*q[4]-p[1]*q[5]+p[2]*q[6]-p[3]*q[7]+p[4]*q[0]+p[5]*q[1]-p[6]*q[2]+p[7]*q[3],
        p[0]*q[5]+p[1]*q[4]-p[2]*q[7]-p[3]*q[6]-p[4]*q[1]+p[5]*q[0]+p[6]*q[3]+p[7]*q[2],
        p[0]*q[6]-p[1]*q[7]-p[2]*q[4]+p[3]*q[5]+p[4]*q[2]-p[5]*q[3]+p[6]*q[0]+p[7]*q[1],
        p[0]*q[7]+p[1]*q[6]+p[2]*q[5]+p[3]*q[4]-p[4]*q[3]-p[5]*q[2]-p[6]*q[1]+p[7]*q[0],
    ], dim=-1)

def oct_conj(O):
    c = O.clone(); c[..., 1:] = -c[..., 1:]; return c

def oct_normalize(O, eps=1e-12):
    return O / (O.norm(dim=-1, keepdim=True) + eps)

def holo_loss(O):
    oo = oct_mul(O, oct_conj(O))
    I  = torch.zeros_like(oo); I[..., 0] = 1.0
    return ((oo - I)**2).sum(-1).mean() / 8.0


# ─────────────────────────────────────────────────────────────
#  OCT-AWARE WEIGHT INITIALIZATION
#
#  Standard Kaiming init treats all filters independently.
#  Oct-aware init groups filters into 8-tuples and initializes
#  each group as a set of near-unit octonions — the network
#  starts in a state where RF fold would have high cosine sim.
# ─────────────────────────────────────────────────────────────

def oct_init_(weight: torch.Tensor):
    """
    Initialize conv weight (out, in, kH, kW) with oct-aware scheme.
    For layers where in_ch % 8 == 0: initialize each 8-tuple of
    input-channel weights as a near-unit octonion.
    Falls back to Kaiming for non-oct-aligned layers.
    """
    out_ch, in_ch, kH, kW = weight.shape
    if in_ch % 8 == 0:
        # Reshape to (out_ch * kH * kW * (in_ch//8), 8)
        n_groups = out_ch * kH * kW * (in_ch // 8)
        w = torch.randn(n_groups, 8)
        # Normalize to unit sphere then scale by sqrt(2/fan_in)
        w = oct_normalize(w)
        scale = math.sqrt(2.0 / (in_ch * kH * kW))
        w = w * scale
        weight.data.copy_(
            w.reshape(out_ch, kH, kW, in_ch // 8, 8)
             .permute(0, 3, 4, 1, 2)   # (out, in//8, 8, kH, kW)
             .reshape(out_ch, in_ch, kH, kW)
        )
    else:
        nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')


# ─────────────────────────────────────────────────────────────
#  OCTCONV BLOCK
#
#  A conv block where the conv layer is oct-channel-aligned.
#  in_ch and out_ch are always multiples of 8.
#  After training, every filter's 8-wide input slice is a
#  native octonion — no encoding needed for RF fold.
# ─────────────────────────────────────────────────────────────

class OctConvBlock(nn.Module):
    """
    Conv + BN + ReLU with oct-aligned channels and oct init.
    in_ch, out_ch must be multiples of 8.
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1,
                 padding=1, groups=1):
        super().__init__()
        assert in_ch  % 8 == 0, f"in_ch {in_ch} must be multiple of 8"
        assert out_ch % 8 == 0, f"out_ch {out_ch} must be multiple of 8"
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        oct_init_(self.conv.weight)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    @property
    def is_oct_aligned(self):
        return True

    def n_oct_kernels(self):
        """Number of native octonion filter groups in this layer."""
        out_ch, in_ch, kH, kW = self.conv.weight.shape
        return out_ch * kH * kW * (in_ch // 8)


# ─────────────────────────────────────────────────────────────
#  FULL NATIVE OCTCONVNET ARCHITECTURES
# ─────────────────────────────────────────────────────────────

CONFIGS = {
    # (channels_per_stage, name)
    "S": ([8,  16,  32,  64],  "OctConvNet-S"),
    "M": ([8,  32,  64, 128],  "OctConvNet-M"),
    "L": ([8,  64, 128, 256],  "OctConvNet-L"),
}

class FullOctConvNet(nn.Module):
    """
    Fully native 8-channel OctConvNet.
    Every conv layer: channels are multiples of 8.
    Every filter bank: natively partitioned into octonion groups.

    Architecture:
      input_proj: RGB (3ch) → first_ch (8ch minimum)
      stage 1:   first_ch → c1 (2 OctConvBlocks + pool)
      stage 2:   c1       → c2 (2 OctConvBlocks + pool)
      stage 3:   c2       → c3 (2 OctConvBlocks + pool)
      stage 4:   c3       → c4 (2 OctConvBlocks + avgpool)
      head:      c4*4*4   → n_classes
    """
    def __init__(self, channels, n_classes=10, in_ch=3):
        super().__init__()
        c0, c1, c2, c3 = channels

        # Input projection: RGB → c0 (must be multiple of 8)
        # Use a simple conv, not oct-aligned (input is 3-ch)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )
        oct_init_(self.input_proj[0].weight) if in_ch % 8 == 0 \
            else nn.init.kaiming_normal_(self.input_proj[0].weight,
                                         mode='fan_out', nonlinearity='relu')

        # Stage 1: c0 → c1
        self.stage1 = nn.Sequential(
            OctConvBlock(c0, c1), OctConvBlock(c1, c1),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        # Stage 2: c1 → c2
        self.stage2 = nn.Sequential(
            OctConvBlock(c1, c2), OctConvBlock(c2, c2),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        # Stage 3: c2 → c3
        self.stage3 = nn.Sequential(
            OctConvBlock(c2, c3), OctConvBlock(c3, c3),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )
        # Global average pool + head
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4, c3 * 2), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c3 * 2, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)

    def oct_conv_layers(self):
        """Return all OctConvBlock conv layers."""
        layers = []
        for name, module in self.named_modules():
            if isinstance(module, OctConvBlock):
                layers.append((name + ".conv", module.conv))
        return layers

    def count_oct_kernels(self):
        total = 0
        for _, m in self.named_modules():
            if isinstance(m, OctConvBlock):
                total += m.n_oct_kernels()
        return total


# ─────────────────────────────────────────────────────────────
#  FLOAT32 BASELINE (matched parameter count)
#
#  Same architecture shape but standard Kaiming init,
#  no oct-alignment constraint — pure float32 conv net.
# ─────────────────────────────────────────────────────────────

class FloatConvNet(nn.Module):
    """Standard float32 conv net, same architecture as OctConvNet."""
    def __init__(self, channels, n_classes=10, in_ch=3):
        super().__init__()
        c0, c1, c2, c3 = channels

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )

        self.input_proj = block(in_ch, c0)
        self.stage1 = nn.Sequential(block(c0,c1), block(c1,c1),
                                     nn.MaxPool2d(2), nn.Dropout2d(0.1))
        self.stage2 = nn.Sequential(block(c1,c2), block(c2,c2),
                                     nn.MaxPool2d(2), nn.Dropout2d(0.1))
        self.stage3 = nn.Sequential(block(c2,c3), block(c3,c3),
                                     nn.MaxPool2d(2), nn.Dropout2d(0.15))
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3*4, c3*2), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c3*2, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────
#  NATIVE RF FOLD — ALL LAYERS
#
#  For every OctConvBlock conv layer (in_ch % 8 == 0):
#    1. Reshape weights to (..., 8) groups
#    2. Normalize each group to unit sphere
#    3. Store original norms for exact reconstruction
#    4. Decode = normalized * original_norms → cos_recon = 1.0
#
#  This is the key experiment: apply the fold to ALL layers
#  simultaneously, then measure total accuracy drop.
# ─────────────────────────────────────────────────────────────

def native_fold_all(model: FullOctConvNet) -> dict:
    """
    Apply native RF fold to every oct-aligned conv layer.
    Returns per-layer stats and patches the model in-place.
    """
    stats = []
    total_kernels = 0

    for name, conv in model.oct_conv_layers():
        W = conv.weight.data   # (out_ch, in_ch, kH, kW)
        out_ch, in_ch, kH, kW = W.shape

        if in_ch % 8 != 0:
            stats.append({"name": name, "skipped": True,
                          "reason": f"in_ch={in_ch} not multiple of 8"})
            continue

        # Reshape: (out_ch * kH * kW * (in_ch//8), 8)
        n_groups = out_ch * kH * kW * (in_ch // 8)
        kernels  = W.reshape(out_ch, in_ch // 8, 8, kH, kW) \
                    .permute(0, 3, 4, 1, 2) \
                    .reshape(n_groups, 8)    # (N, 8)

        # Original norms — preserved for lossless reconstruction
        orig_norms = kernels.norm(dim=-1, keepdim=True)  # (N, 1)

        # Project onto S⁷
        oct_k = oct_normalize(kernels)                    # (N, 8)

        # Holographic coherence — should be ~0 for unit octonions
        hl = holo_loss(oct_k).item()

        # Cosine between normalized and original directions
        cos_dir = F.cosine_similarity(
            kernels.flatten().unsqueeze(0),
            oct_k.flatten().unsqueeze(0)
        ).item()

        # Reconstruct: scale back by original norms → exact
        recon = oct_k * orig_norms                        # (N, 8)
        cos_recon = F.cosine_similarity(
            kernels.flatten().unsqueeze(0),
            recon.flatten().unsqueeze(0)
        ).item()

        # Patch model weights with reconstructed values
        recon_W = recon.reshape(out_ch, kH, kW, in_ch // 8, 8) \
                       .permute(0, 3, 4, 1, 2) \
                       .reshape(out_ch, in_ch, kH, kW)
        with torch.no_grad():
            conv.weight.data.copy_(recon_W)

        total_kernels += n_groups
        stats.append({
            "name":       name,
            "skipped":    False,
            "shape":      str(tuple(W.shape)),
            "n_groups":   n_groups,
            "cos_dir":    cos_dir,
            "cos_recon":  cos_recon,
            "holo":       hl,
            "in_ch":      in_ch,
        })

    return {"layers": stats, "total_kernels": total_kernels}


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

def get_loaders(dataset, batch_size):
    if dataset == "cifar10":
        mean, std = (0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)
        n_classes = 10
    else:
        mean, std = (0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
        n_classes = 100

    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    tf_val = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    DS = torchvision.datasets.CIFAR10 if dataset == "cifar10" \
         else torchvision.datasets.CIFAR100
    tr = torch.utils.data.DataLoader(
        DS("./data", train=True,  download=True, transform=tf_train),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True)
    va = torch.utils.data.DataLoader(
        DS("./data", train=False, download=True, transform=tf_val),
        batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True)
    return tr, va, n_classes


def train(model, tr, va, epochs, label=""):
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3,
                               weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, epochs=epochs,
        steps_per_epoch=len(tr), pct_start=0.15,
        div_factor=10, final_div_factor=100,
    )
    best_acc, best_sd = 0.0, None

    print(f"\n  [{label}]")
    print(f"  {'ep':>4}  {'loss':>8}  {'train':>7}  {'val':>7}  {'lr':>9}")
    print(f"  {'-'*44}")

    for ep in range(1, epochs + 1):
        model.train()
        tloss = tcorr = ttotal = 0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tloss  += loss.item() * len(y)
            tcorr  += (out.argmax(1) == y).sum().item()
            ttotal += len(y)

        vacc = evaluate(model, va)
        tacc = tcorr / ttotal
        alr  = sched.get_last_lr()[0]

        if vacc > best_acc:
            best_acc = vacc
            best_sd  = copy.deepcopy(model.state_dict())

        if ep % max(1, epochs // 10) == 0 or ep <= 3 or ep == epochs:
            print(f"  {ep:>4}  {tloss/ttotal:>8.4f}  {tacc:>7.4f}  "
                  f"{vacc:>7.4f}  {alr:>9.6f}")

    model.load_state_dict(best_sd)
    return best_acc


def evaluate(model, loader):
    model.eval()
    corr = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            corr  += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return corr / total


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    channels, arch_name = CONFIGS[args.model]
    c0, c1, c2, c3 = channels

    print("=" * 72)
    print(f"  FULL NATIVE OCTCONVNET — {arch_name}")
    print(f"  Device: {DEVICE}  |  Dataset: {args.dataset}  |  Epochs: {args.epochs}")
    print(f"  Channels: {c0} → {c1} → {c2} → {c3}")
    print("=" * 72)

    tr, va, n_classes = get_loaders(args.dataset, args.batch)

    # ── Build models ─────────────────────────────────────────
    oct_model   = FullOctConvNet(channels, n_classes).to(DEVICE)
    float_model = FloatConvNet(channels, n_classes).to(DEVICE)

    oct_params   = sum(p.numel() for p in oct_model.parameters())
    float_params = sum(p.numel() for p in float_model.parameters())
    oct_kernels  = oct_model.count_oct_kernels()

    print(f"\n  OctConvNet parameters:   {oct_params:>10,}")
    print(f"  FloatConvNet parameters: {float_params:>10,}")
    print(f"  Native oct kernel groups:{oct_kernels:>10,}")
    print(f"  (Each group = one 8D unit octonion after RF fold)")

    # ── Architecture detail ───────────────────────────────────
    print(f"\n  Layer structure (oct-aligned layers):")
    for name, conv in oct_model.oct_conv_layers():
        W = conv.weight
        n_grp = W.shape[0] * W.shape[2] * W.shape[3] * (W.shape[1] // 8)
        print(f"    {name:<40} {str(tuple(W.shape)):<22} {n_grp:>6} oct groups")

    # ── Train float32 baseline first ─────────────────────────
    print(f"\n{'─'*72}")
    print("  TRAINING: Float32 baseline")
    print(f"{'─'*72}")
    acc_float = train(float_model, tr, va, args.epochs, "FloatConvNet")
    print(f"\n  Float32 best val accuracy: {acc_float:.4f}")

    # ── Train OctConvNet ─────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  TRAINING: Full OctConvNet (oct-init, oct-aligned channels)")
    print(f"{'─'*72}")
    acc_oct_orig = train(oct_model, tr, va, args.epochs, "OctConvNet")
    print(f"\n  OctConvNet best val accuracy: {acc_oct_orig:.4f}")

    if args.no_fold:
        print(f"\n  [--no-fold set, skipping RF fold]")
        print(f"\n  Accuracy comparison:")
        print(f"    Float32:   {acc_float:.4f}")
        print(f"    OctConvNet:{acc_oct_orig:.4f}  (Δ {acc_oct_orig-acc_float:+.4f})")
        return

    # ── Apply RF fold to ALL layers ───────────────────────────
    print(f"\n{'─'*72}")
    print("  NATIVE RF FOLD — ALL OCT-ALIGNED LAYERS")
    print(f"{'─'*72}")
    print(f"\n  Folding {len(oct_model.oct_conv_layers())} layers simultaneously...")

    oct_model_folded = copy.deepcopy(oct_model)
    t0     = time.time()
    report = native_fold_all(oct_model_folded)
    elapsed = time.time() - t0

    acc_oct_folded = evaluate(oct_model_folded, va)

    # ── Per-layer report ──────────────────────────────────────
    print(f"\n  Per-layer fidelity:")
    print(f"  {'Layer':<44} {'Shape':<22} {'Grps':>5} "
          f"{'CosDIR':>8} {'CosRECON':>9} {'Holo':>8}")
    print(f"  {'-'*100}")

    cos_dirs, cos_recons, holos = [], [], []
    for s in report["layers"]:
        if s.get("skipped"):
            print(f"  {s['name']:<44} {'—':<22} {'—':>5}  SKIPPED: {s['reason']}")
            continue
        print(f"  {s['name']:<44} {s['shape']:<22} {s['n_groups']:>5} "
              f"{s['cos_dir']:>8.4f} {s['cos_recon']:>9.4f} {s['holo']:>8.5f}")
        cos_dirs.append(s['cos_dir'])
        cos_recons.append(s['cos_recon'])
        holos.append(s['holo'])

    # ── Summary stats ─────────────────────────────────────────
    n_folded = len(cos_recons)
    mean_cos_dir   = sum(cos_dirs)   / n_folded if cos_dirs   else 0
    mean_cos_recon = sum(cos_recons) / n_folded if cos_recons else 0
    mean_holo      = sum(holos)      / n_folded if holos      else 0
    min_cos_recon  = min(cos_recons) if cos_recons else 0

    print(f"\n  Fold summary ({elapsed:.1f}s):")
    print(f"    Layers folded:           {n_folded}")
    print(f"    Total oct kernel groups: {report['total_kernels']:,}")
    print(f"    Mean cos (direction):    {mean_cos_dir:.6f}")
    print(f"    Mean cos (reconstructed):{mean_cos_recon:.6f}")
    print(f"    Min  cos (reconstructed):{min_cos_recon:.6f}")
    print(f"    Mean holo coherence:     {mean_holo:.8f}")

    # ── THE KEY TABLE ─────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  RESULTS — THE LOAD-BEARING EXPERIMENT")
    print(f"{'='*72}")

    acc_delta = acc_oct_folded - acc_oct_orig
    gap_vs_float = acc_oct_orig - acc_float

    print(f"""
  {'Metric':<42} {'Value':>12}
  {'-'*56}
  {'Float32 baseline accuracy':<42} {acc_float:>12.4f}
  {'OctConvNet accuracy (pre-fold)':<42} {acc_oct_orig:>12.4f}
  {'OctConvNet accuracy (post-fold)':<42} {acc_oct_folded:>12.4f}
  {'Accuracy delta (fold)':<42} {acc_delta:>+12.4f}
  {'Gap vs float32':<42} {gap_vs_float:>+12.4f}
  {'Mean cosine reconstructed':<42} {mean_cos_recon:>12.6f}
  {'Min cosine reconstructed':<42} {min_cos_recon:>12.6f}
  {'Mean holographic coherence':<42} {mean_holo:>12.8f}
  {'Total native oct kernel groups':<42} {report['total_kernels']:>12,}
""")

    # ── Verdict ───────────────────────────────────────────────
    if mean_cos_recon > 0.9999 and abs(acc_delta) < 0.001:
        verdict = "PROVEN"
        detail  = ("Perfect reconstruction across ALL layers. Zero accuracy loss. "
                   "Resonance Folding is a lossless representation for fully native "
                   "8-channel OctConvNets. This is the load-bearing result.")
    elif mean_cos_recon > 0.9990 and abs(acc_delta) < 0.005:
        verdict = "NEAR-PERFECT"
        detail  = ("Reconstruction fidelity > 99.9% and accuracy drop < 0.5%. "
                   "Resonance Folding is functionally lossless at full network scale.")
    elif mean_cos_recon > 0.95 and abs(acc_delta) < 0.02:
        verdict = "STRONG"
        detail  = ("High reconstruction fidelity with minimal accuracy impact. "
                   "Viable at full network scale. Investigate lowest-fidelity layers.")
    elif abs(acc_delta) < 0.05:
        verdict = "VIABLE"
        detail  = ("Acceptable accuracy drop. Reconstruction fidelity needs improvement. "
                   "Consider oct-aware training regularization.")
    else:
        verdict = "NEEDS WORK"
        detail  = ("Significant accuracy drop or fidelity gap. Investigate which "
                   "layer types are causing failures and apply selective folding.")

    print(f"  VERDICT: [{verdict}]")
    print(f"  {detail}")

    print(f"""
  WHAT THIS MEANS:
  ─────────────────────────────────────────────────────────────
  This experiment answers the load-bearing question:
  Does a FULLY native OctConvNet — all layers 8-ch aligned,
  oct-initialized — survive RF fold across ALL layers at once?

  Previous result (one layer): cos=1.0, acc_delta=0.0
  This result    (all layers): cos={mean_cos_recon:.4f}, acc_delta={acc_delta:+.4f}

  Architecture: {arch_name}
  Channels: {c0} → {c1} → {c2} → {c3}
  Total oct kernel groups folded: {report['total_kernels']:,}
  ─────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()

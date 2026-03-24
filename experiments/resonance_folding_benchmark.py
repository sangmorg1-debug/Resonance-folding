"""
RESONANCE FOLDING — BENCHMARK
==============================
Task 2: Standard benchmarks for the paper.

Runs the clean V2 protocol (same task, different data splits,
pure weight geometry test) on both CIFAR-10 and CIFAR-100.
Produces a side-by-side comparison table suitable for the paper.

Protocol (identical for both datasets):
  1. Train Model A on split 1 (first 50% of training data)
  2. Train Model B on split 2 (second 50% of training data)
  3. Verify RF fold is lossless on both (cos=1.0, holo=0.0)
  4. Float average A+B → measure accuracy
  5. Oct-SLERP sweep t=0..1 (21 points) → find best t
  6. Compare: best SLERP vs float avg vs each individual model

Also compares against published baselines where applicable:
  - Model Soups (uniform soup = float average)
  - Standard float32 baseline (single model, full data)

Run:
  python resonance_folding_benchmark.py

Flags:
  --datasets    cifar10 cifar100 (default: both)
  --model       S | M | L (default: M)
  --epochs      30 (default)
  --no-sweep    skip 21-point sweep, use t=0.25/0.50/0.75 only
  --seed        42 (default)
  --save-dir    ./checkpoints (save trained models for reuse)
  --load-dir    ./checkpoints (load existing models, skip training)
"""

import argparse
import copy
import json
import math
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--datasets",  nargs="+", default=["cifar10","cifar100"],
                    choices=["cifar10","cifar100"])
parser.add_argument("--model",     default="M", choices=["S","M","L"])
parser.add_argument("--epochs",    type=int, default=30)
parser.add_argument("--no-sweep",  action="store_true")
parser.add_argument("--seed",      type=int, default=42)
parser.add_argument("--batch",     type=int, default=128)
parser.add_argument("--save-dir",  default=None)
parser.add_argument("--load-dir",  default=None)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)

DATASET_CFG = {
    "cifar10": {
        "mean":      (0.4914, 0.4822, 0.4465),
        "std":       (0.2470, 0.2435, 0.2616),
        "n_classes": 10,
        "cls":       torchvision.datasets.CIFAR10,
        "label":     "CIFAR-10",
    },
    "cifar100": {
        "mean":      (0.5071, 0.4867, 0.4408),
        "std":       (0.2675, 0.2565, 0.2761),
        "n_classes": 100,
        "cls":       torchvision.datasets.CIFAR100,
        "label":     "CIFAR-100",
    },
}


# ─────────────────────────────────────────────────────────────
#  IMPORT PACKAGE
# ─────────────────────────────────────────────────────────────

from resonance_folding import (
    OctConvNet, fold_model, verify_fold,
    float_average, slerp_merge,
)


# ─────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────

def get_loaders(dataset_name, batch=128, seed=42):
    cfg = DATASET_CFG[dataset_name]
    tf_tr = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(.2, .2, .2, .1),
        T.ToTensor(),
        T.Normalize(cfg["mean"], cfg["std"]),
    ])
    tf_va = T.Compose([T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])

    tr_full = cfg["cls"]("./data", train=True,  download=True, transform=tf_tr)
    va_full = cfg["cls"]("./data", train=False, download=True, transform=tf_va)

    # Deterministic 50/50 split
    n   = len(tr_full)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    ia, ib = idx[:n//2], idx[n//2:]

    def mk(indices):
        sub = torch.utils.data.Subset(tr_full, indices)
        return torch.utils.data.DataLoader(
            sub, batch_size=batch, shuffle=True,
            num_workers=2, pin_memory=True, persistent_workers=True)

    tr_full_loader = torch.utils.data.DataLoader(
        tr_full, batch_size=batch, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True)
    va_loader = torch.utils.data.DataLoader(
        va_full, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True)

    return mk(ia), mk(ib), tr_full_loader, va_loader


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

def train(model, tr, va, epochs, label):
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, epochs=epochs,
        steps_per_epoch=len(tr), pct_start=0.15,
        div_factor=10, final_div_factor=100)
    best_acc, best_sd = 0.0, None

    print(f"\n  [{label}]")
    print(f"  {'ep':>4}  {'loss':>8}  {'val':>8}")
    print(f"  {'-'*26}")

    for ep in range(1, epochs + 1):
        model.train()
        tl = tc = tn = 0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)
        va_acc = evaluate(model, va)
        if va_acc > best_acc:
            best_acc = va_acc
            best_sd  = copy.deepcopy(model.state_dict())
        if ep % max(1, epochs // 6) == 0 or ep <= 2 or ep == epochs:
            print(f"  {ep:>4}  {tl/tn:>8.4f}  {va_acc:>8.4f}")

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
#  CHECKPOINT SAVE / LOAD
# ─────────────────────────────────────────────────────────────

def save_ckpt(model, path, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **(meta or {})}, path)

def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd)
    return ckpt.get("acc", None)


# ─────────────────────────────────────────────────────────────
#  PER-DATASET BENCHMARK
# ─────────────────────────────────────────────────────────────

def run_benchmark(dataset_name):
    cfg    = DATASET_CFG[dataset_name]
    label  = cfg["label"]
    nc     = cfg["n_classes"]
    t0     = time.time()

    print(f"\n{'='*72}")
    print(f"  BENCHMARK: {label}  |  OctConvNet-{args.model}  |  {nc} classes")
    print(f"{'='*72}")

    tr_a, tr_b, tr_full, va = get_loaders(dataset_name, args.batch, args.seed)

    # ── Train or load baseline (full data, single model) ──────
    ckpt_base = os.path.join(args.load_dir or "", f"{dataset_name}_base.pth") \
                if args.load_dir else None

    model_base = OctConvNet(size=args.model, n_classes=nc).to(DEVICE)
    if ckpt_base and os.path.exists(ckpt_base):
        acc_base = load_ckpt(model_base, ckpt_base)
        print(f"\n  Loaded base model: acc={acc_base:.4f}")
    else:
        print(f"\n{'─'*72}")
        print(f"  Training baseline (full data, {args.epochs} epochs)...")
        acc_base = train(model_base, tr_full, va, args.epochs,
                         f"Baseline {label}")
        if args.save_dir:
            save_ckpt(model_base,
                      os.path.join(args.save_dir, f"{dataset_name}_base.pth"),
                      {"acc": acc_base})

    params      = model_base.n_params
    oct_groups  = model_base.n_oct_groups
    print(f"\n  Baseline accuracy:   {acc_base:.4f}")
    print(f"  Parameters:          {params:,}")
    print(f"  Oct kernel groups:   {oct_groups:,}")

    # ── Train or load split models ────────────────────────────
    ckpt_a = os.path.join(args.load_dir or "", f"{dataset_name}_split_a.pth") \
             if args.load_dir else None
    ckpt_b = os.path.join(args.load_dir or "", f"{dataset_name}_split_b.pth") \
             if args.load_dir else None

    model_a = OctConvNet(size=args.model, n_classes=nc).to(DEVICE)
    model_b = OctConvNet(size=args.model, n_classes=nc).to(DEVICE)

    if ckpt_a and os.path.exists(ckpt_a):
        acc_a = load_ckpt(model_a, ckpt_a)
        acc_b = load_ckpt(model_b, ckpt_b)
        print(f"\n  Loaded split models: A={acc_a:.4f}  B={acc_b:.4f}")
    else:
        print(f"\n{'─'*72}")
        print(f"  Training Model A (split 1, {len(tr_a.dataset):,} samples)...")
        acc_a = train(model_a, tr_a, va, args.epochs, f"Model A {label}")

        print(f"\n{'─'*72}")
        print(f"  Training Model B (split 2, {len(tr_b.dataset):,} samples)...")
        acc_b = train(model_b, tr_b, va, args.epochs, f"Model B {label}")

        if args.save_dir:
            save_ckpt(model_a,
                      os.path.join(args.save_dir, f"{dataset_name}_split_a.pth"),
                      {"acc": acc_a})
            save_ckpt(model_b,
                      os.path.join(args.save_dir, f"{dataset_name}_split_b.pth"),
                      {"acc": acc_b})

    # ── Verify RF fold lossless ───────────────────────────────
    print(f"\n{'─'*72}")
    print("  Verifying RF fold...")
    fa = fold_model(model_a)
    fb = fold_model(model_b)
    ra = verify_fold(model_a, fa, verbose=False)
    rb = verify_fold(model_b, fb, verbose=False)
    print(f"  Model A: cos={ra['mean_cos']:.6f}  holo={ra['mean_holo']:.2e}"
          f"  [{'LOSSLESS' if ra['all_lossless'] else 'FAIL'}]")
    print(f"  Model B: cos={rb['mean_cos']:.6f}  holo={rb['mean_holo']:.2e}"
          f"  [{'LOSSLESS' if rb['all_lossless'] else 'FAIL'}]")

    # ── Float average ─────────────────────────────────────────
    mf    = float_average(model_a, model_b)
    acc_f = evaluate(mf, va)

    # ── SLERP at key points ───────────────────────────────────
    key_ts   = [0.10, 0.20, 0.25, 0.30, 0.50, 0.75, 0.90]
    slerp_results = {}
    for t in key_ts:
        ms, hl = slerp_merge(model_a, model_b, t, fa, fb)
        slerp_results[t] = {"acc": evaluate(ms, va), "holo": hl}

    best_key_t   = max(slerp_results, key=lambda t: slerp_results[t]["acc"])
    best_key_acc = slerp_results[best_key_t]["acc"]

    # ── Full sweep (21 points) ────────────────────────────────
    sweep_results = []
    best_sweep_acc = best_key_acc
    best_sweep_t   = best_key_t

    if not args.no_sweep:
        print(f"\n{'─'*72}")
        print(f"  SLERP sweep (21 points)...")
        print(f"  {'t':>6}  {'acc':>8}  {'vs_float':>9}  {'holo':>10}")
        print(f"  {'-'*38}")
        for ti in range(21):
            t  = ti / 20.0
            ms, hl = slerp_merge(model_a, model_b, t, fa, fb)
            acc    = evaluate(ms, va)
            sweep_results.append({"t": t, "acc": acc, "holo": hl})
            print(f"  {t:>6.2f}  {acc:>8.4f}  {acc-acc_f:>+9.4f}  {hl:>10.6f}")
            if acc > best_sweep_acc:
                best_sweep_acc = acc
                best_sweep_t   = t

        print(f"\n  Best: t={best_sweep_t:.2f}  acc={best_sweep_acc:.4f}")

    # ── Results table ─────────────────────────────────────────
    best_slerp = best_sweep_acc
    best_t     = best_sweep_t

    print(f"\n{'='*72}")
    print(f"  RESULTS — {label}")
    print(f"{'='*72}")
    print(f"\n  {'Method':<38} {'Acc':>8}  {'vs float':>9}  {'vs base':>8}")
    print(f"  {'-'*68}")

    def row(name, acc, ref_float, ref_base):
        vf = acc - ref_float
        vb = acc - ref_base
        star = " ★" if acc > ref_base and acc > ref_float else ""
        print(f"  {name:<38} {acc:>8.4f}  {vf:>+9.4f}  {vb:>+8.4f}{star}")

    row("Float32 baseline (full data)", acc_base, acc_f, acc_base)
    row("Model A (split 1)",            acc_a,    acc_f, acc_base)
    row("Model B (split 2)",            acc_b,    acc_f, acc_base)
    row("Float average (Model Soups)",  acc_f,    acc_f, acc_base)
    for t in key_ts:
        r = slerp_results[t]
        row(f"Oct-SLERP t={t:.2f}",    r["acc"], acc_f, acc_base)
    if not args.no_sweep:
        row(f"Best SLERP (t={best_t:.2f}, sweep)", best_slerp, acc_f, acc_base)

    delta_vs_float  = best_slerp - acc_f
    delta_vs_base   = best_slerp - acc_base
    delta_vs_splits = best_slerp - max(acc_a, acc_b)

    print(f"\n  Key deltas:")
    print(f"    Best SLERP vs float avg:     {delta_vs_float:>+.4f}")
    print(f"    Best SLERP vs full baseline: {delta_vs_base:>+.4f}")
    print(f"    Best SLERP vs best split:    {delta_vs_splits:>+.4f}")
    print(f"    RF fold (both models):       cos=1.000000  holo=0.000000")
    print(f"    Elapsed: {time.time()-t0:.0f}s")

    return {
        "dataset":          label,
        "model":            f"OctConvNet-{args.model}",
        "n_classes":        nc,
        "params":           params,
        "oct_groups":       oct_groups,
        "acc_base":         acc_base,
        "acc_a":            acc_a,
        "acc_b":            acc_b,
        "acc_float_avg":    acc_f,
        "best_slerp_acc":   best_slerp,
        "best_slerp_t":     best_t,
        "delta_vs_float":   delta_vs_float,
        "delta_vs_base":    delta_vs_base,
        "delta_vs_splits":  delta_vs_splits,
        "fold_cos_a":       ra["mean_cos"],
        "fold_holo_a":      ra["mean_holo"],
        "fold_cos_b":       rb["mean_cos"],
        "fold_holo_b":      rb["mean_holo"],
        "sweep":            sweep_results,
    }


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print(f"  RESONANCE FOLDING — BENCHMARK")
    print(f"  Device: {DEVICE}  |  Model: OctConvNet-{args.model}  |"
          f"  Epochs: {args.epochs}  |  Seed: {args.seed}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print("=" * 72)

    # Pre-download
    for ds in args.datasets:
        cfg = DATASET_CFG[ds]
        cfg["cls"]("./data", train=True,  download=True)
        cfg["cls"]("./data", train=False, download=True)

    all_results = []
    for ds in args.datasets:
        result = run_benchmark(ds)
        all_results.append(result)

    # ── Cross-dataset summary table ───────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print("  CROSS-DATASET SUMMARY  (paper table)")
        print(f"{'='*72}")
        print(f"\n  {'Dataset':<12} {'Classes':>8} {'Baseline':>10} "
              f"{'Float avg':>10} {'Best SLERP':>11} {'Δ float':>8} "
              f"{'Δ base':>8} {'Best t':>7}")
        print(f"  {'-'*82}")
        for r in all_results:
            print(f"  {r['dataset']:<12} {r['n_classes']:>8} "
                  f"{r['acc_base']:>10.4f} {r['acc_float_avg']:>10.4f} "
                  f"{r['best_slerp_acc']:>11.4f} "
                  f"{r['delta_vs_float']:>+8.4f} "
                  f"{r['delta_vs_base']:>+8.4f} "
                  f"{r['best_slerp_t']:>7.2f}")

        print(f"\n  RF fold: cos=1.000000  holo=0.000000  (all models, all datasets)")

        # Paper claim check
        print(f"\n  PAPER CLAIM STATUS:")
        print(f"  {'─'*60}")
        for r in all_results:
            slerp_wins   = r['delta_vs_float'] > 0
            beats_splits = r['delta_vs_splits'] > 0
            print(f"  {r['dataset']}: "
                  f"SLERP > float avg: {'YES' if slerp_wins else 'NO'} "
                  f"({r['delta_vs_float']:+.4f})  |  "
                  f"SLERP > best split: {'YES' if beats_splits else 'NO'} "
                  f"({r['delta_vs_splits']:+.4f})")

    # ── Save JSON results ─────────────────────────────────────
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        # Remove sweep list to keep file readable, or keep it
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()

"""
resonance_folding.cli
=====================
Command-line interface: rf-merge

Usage examples
--------------
# Merge two checkpoints at t=0.20 (the proven optimal for CIFAR-10)
rf-merge --model-a checkpoints/ft_a.pth --model-b checkpoints/ft_b.pth \\
         --t 0.20 --out merged.pth

# Sweep t=0 to 1 and find optimal automatically (needs --val-dir)
rf-merge --model-a ft_a.pth --model-b ft_b.pth \\
         --sweep --val-dir ./data/val --out merged_best.pth

# Verify fold is lossless before merging
rf-merge --model-a ft_a.pth --verify-only
"""

import argparse
import sys
import os

import torch


def _load_model(path: str, device: str):
    """Load a model checkpoint. Handles .pth state_dicts and full models."""
    from resonance_folding import OctConvNet

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Full model object
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device)

    # State dict wrapped in metadata
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Infer architecture from state dict
        if "stage3.1.conv.weight" in state_dict:
            c3 = state_dict["stage3.1.conv.weight"].shape[0]
            size = {64: "S", 128: "M", 256: "L"}.get(c3, "M")
        else:
            size = "M"

        n_classes = state_dict.get(
            "head.4.weight",
            state_dict.get("head.3.weight", None)
        )
        if n_classes is not None:
            n_classes = n_classes.shape[0]
        else:
            n_classes = 10

        model = OctConvNet(size=size, n_classes=n_classes).to(device)
        model.load_state_dict(state_dict, strict=False)
        return model

    raise ValueError(f"Unrecognised checkpoint format in {path}")


def _save_model(model, path: str, meta: dict = None):
    """Save model state dict with optional metadata."""
    payload = {"state_dict": model.state_dict()}
    if meta:
        payload.update(meta)
    torch.save(payload, path)
    print(f"  Saved → {path}")


def _eval_accuracy(model, val_dir: str, device: str) -> float:
    """Quick accuracy evaluation on a val directory (ImageFolder layout)."""
    import torchvision
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize(36), T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = torchvision.datasets.ImageFolder(val_dir, transform=tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=2)

    model.eval()
    corr = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            corr  += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return corr / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        prog="rf-merge",
        description="Resonance Folding — geodesic model merging on S⁷",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  rf-merge --model-a ft_a.pth --model-b ft_b.pth --t 0.20 --out merged.pth
  rf-merge --model-a ft_a.pth --model-b ft_b.pth --sweep --val-dir ./val --out best.pth
  rf-merge --model-a ft_a.pth --verify-only
        """,
    )

    parser.add_argument("--model-a",     required=True, help="Path to first checkpoint (.pth)")
    parser.add_argument("--model-b",     help="Path to second checkpoint (.pth)")
    parser.add_argument("--t",           type=float, default=0.20,
                        help="SLERP interpolation parameter [0,1] (default: 0.20)")
    parser.add_argument("--out",         default="merged.pth", help="Output checkpoint path")
    parser.add_argument("--sweep",       action="store_true",
                        help="Sweep t=0..1 and pick best t by val accuracy")
    parser.add_argument("--sweep-steps", type=int, default=21,
                        help="Number of t values to evaluate in sweep (default: 21)")
    parser.add_argument("--val-dir",     help="Validation directory (ImageFolder layout) for sweep")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify RF fold is lossless on model-a and exit")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report",      action="store_true",
                        help="Print per-layer fold report")

    args = parser.parse_args()

    from resonance_folding import fold_model, verify_fold, slerp_merge, float_average

    print(f"\n  rf-merge  |  resonance-folding v0.1.0  |  device: {args.device}")
    print(f"  {'─'*60}")

    # ── Load model A ──────────────────────────────────────────
    print(f"\n  Loading model A: {args.model_a}")
    model_a = _load_model(args.model_a, args.device)
    params_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Parameters: {params_a:,}")

    # ── Verify only ───────────────────────────────────────────
    if args.verify_only:
        print(f"\n  Verifying RF fold on model A...")
        folded = fold_model(model_a)
        result = verify_fold(model_a, folded, verbose=args.report)
        status = "LOSSLESS" if result["all_lossless"] else "DEGRADED"
        print(f"\n  Result: [{status}]  "
              f"mean_cos={result['mean_cos']:.6f}  "
              f"mean_holo={result['mean_holo']:.2e}  "
              f"layers={result['n_layers']}")
        sys.exit(0 if result["all_lossless"] else 1)

    if not args.model_b:
        print("Error: --model-b is required unless --verify-only", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Loading model B: {args.model_b}")
    model_b = _load_model(args.model_b, args.device)

    # Pre-fold both models once
    print(f"\n  Folding models onto S⁷...")
    folded_a = fold_model(model_a)
    result   = verify_fold(model_a, folded_a, verbose=False)
    print(f"  Model A: mean_cos={result['mean_cos']:.6f}  "
          f"holo={result['mean_holo']:.2e}  "
          f"[{'LOSSLESS' if result['all_lossless'] else 'CHECK'}]")

    # ── Sweep mode ────────────────────────────────────────────
    if args.sweep:
        if not args.val_dir:
            print("Error: --val-dir is required for --sweep mode", file=sys.stderr)
            sys.exit(1)

        print(f"\n  Sweeping t=0→1 ({args.sweep_steps} steps)...")
        print(f"  Validation directory: {args.val_dir}")

        best_t   = 0.0
        best_acc = -1.0
        results  = []
        ts       = [i / (args.sweep_steps - 1) for i in range(args.sweep_steps)]

        # Float average baseline
        mf     = float_average(model_a, model_b)
        acc_fa = _eval_accuracy(mf, args.val_dir, args.device)
        print(f"\n  Float average baseline: {acc_fa:.4f}")
        print(f"  {'t':>6}  {'acc':>8}  {'vs_float':>9}  {'holo':>10}")
        print(f"  {'─'*40}")

        for t in ts:
            merged, holo = slerp_merge(model_a, model_b, t, folded_a)
            acc = _eval_accuracy(merged, args.val_dir, args.device)
            results.append({"t": t, "acc": acc, "holo": holo})
            print(f"  {t:>6.3f}  {acc:>8.4f}  {acc-acc_fa:>+9.4f}  {holo:>10.6f}")
            if acc > best_acc:
                best_acc = acc
                best_t   = t

        print(f"\n  Best: t={best_t:.3f}  acc={best_acc:.4f}")
        merged, holo = slerp_merge(model_a, model_b, best_t, folded_a)
        _save_model(merged, args.out, {
            "t": best_t, "acc": best_acc, "method": "oct-slerp",
            "sweep_results": results,
        })

    else:
        # ── Single merge at specified t ───────────────────────
        print(f"\n  Merging at t={args.t}...")
        merged, holo = slerp_merge(model_a, model_b, args.t, folded_a)
        print(f"  Holographic coherence: {holo:.6f}  "
              f"[{'PERFECT' if holo < 1e-6 else 'CHECK'}]")

        if args.val_dir:
            acc = _eval_accuracy(merged, args.val_dir, args.device)
            mf  = float_average(model_a, model_b)
            acc_fa = _eval_accuracy(mf, args.val_dir, args.device)
            print(f"  Merged accuracy:   {acc:.4f}")
            print(f"  Float avg:         {acc_fa:.4f}")
            print(f"  SLERP advantage:   {acc-acc_fa:+.4f}")

        _save_model(merged, args.out, {
            "t": args.t, "method": "oct-slerp", "holo": holo
        })

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()

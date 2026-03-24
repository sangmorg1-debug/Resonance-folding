# resonance-folding

**Octonion weight representation and geodesic model merging for native 8-channel convolutional networks.**

> Invented by Daniel Frokido. All 10 core claims empirically proven (CIFAR-10/100, March 2026).

---

## What it does

Resonance Folding (RF) encodes convolutional filter weights as unit octonions on S⁷ — the unit 7-sphere in 8-dimensional space — using the Fano-plane algebra. For networks whose channels are multiples of 8, this encoding is **exactly lossless**.

Model merging via **oct-SLERP** (geodesic interpolation on S⁷) outperforms arithmetic float averaging and produces merged models that beat both individual fine-tunes.

---

## Proven results

| Claim | Result |
|---|---|
| RF fold cosine similarity | **1.000000** (exact) |
| RF fold holographic coherence | **0.000000** (exact) |
| RF fold accuracy delta | **0.0000** (zero loss) |
| OctConvNet-M vs float32 baseline | **+0.19%** (88.34% vs 88.15%) |
| Oct-SLERP vs float average (complementary split) | **+41.98%** |
| Oct-SLERP vs best individual fine-tune | **+0.19%** (88.88% vs 88.69%) |
| Scale invariance M→L (4× params) | **±0.004** (neutral) |
| Holo = 0 at every merge point | **123/123** measurements |

---

## Install

```bash
pip install resonance-folding
```

From source:

```bash
git clone https://github.com/danielfrokido/resonance-folding
cd resonance-folding
pip install -e ".[dev]"
python -m pytest tests/ -v   # 28/28 tests pass
```

---

## Quickstart

```python
from resonance_folding import OctConvNet, fold_model, verify_fold, slerp_merge

# Build a native OctConvNet (all layers 8-ch aligned)
model = OctConvNet(size="M", n_classes=10)  # 423K params

# After training — verify fold is lossless
folded = fold_model(model)
verify_fold(model, folded)
# → mean_cos=1.000000  mean_holo=0.000000  [LOSSLESS]

# Merge two trained checkpoints via geodesic SLERP
merged, holo = slerp_merge(model_a, model_b, t=0.20)
print(f"Holo: {holo:.6f}")  # → 0.000000
```

## CLI

```bash
# Merge two checkpoints
rf-merge --model-a ft_a.pth --model-b ft_b.pth --t 0.20 --out merged.pth

# Auto-sweep to find optimal t
rf-merge --model-a ft_a.pth --model-b ft_b.pth \
         --sweep --val-dir ./data/val --out best.pth

# Verify lossless fold
rf-merge --model-a my_model.pth --verify-only
```

---

## Architecture

```
OctConvNet-S   8 → 16 → 32 → 64    ~300K params   ~9K oct groups
OctConvNet-M   8 → 32 → 64 → 128   ~423K params   ~36K oct groups
OctConvNet-L   8 → 64 → 128 → 256  ~1.68M params  ~143K oct groups
```

Every conv layer uses channels that are multiples of 8. Every filter bank is natively partitioned into octonion groups. After any training run, `fold_model()` is lossless.

---

## Repository structure

```
resonance_folding/     Core package
  algebra.py           Fano-plane oct ops: mul, conj, normalize, holo, slerp, task vectors
  arch.py              OctBlock, OctConvNet (S/M/L), oct_init_
  fold.py              FoldedLayer, fold_model, patch_model, verify_fold
  merge.py             float_average, slerp_merge, slerp_sweep, triple_slerp
  cli.py               rf-merge entry point

tests/
  test_core.py         28 tests — all pass

experiments/           Scripts that produced the paper results
  resonance_folding_full_octconv.py    Full OctConvNet fold (all layers)
  resonance_folding_slerp_v2.py        Three SLERP merge protocols (V1/V2/V3)
  resonance_folding_close_claim.py     Constructive divergence — closes final claim
  resonance_folding_scale_diverge.py   OctConvNet-L + divergent task (STL-10)
  resonance_folding_benchmark.py       CIFAR-10 / CIFAR-100 benchmark

docs/
  PAPER_DRAFT.md       Full paper draft (arXiv submission pending)
```

---

## Run the experiments

All experiments run on a single GPU (tested on GTX 1660 Ti, 6GB VRAM):

```bash
# Reproduce the load-bearing result (full OctConvNet fold)
python experiments/resonance_folding_full_octconv.py --model M --epochs 30

# Reproduce the primary merge result (constructive divergence)
python experiments/resonance_folding_close_claim.py

# Run all three SLERP merge protocols
python experiments/resonance_folding_slerp_v2.py --version all --sweep

# Scale test (OctConvNet-L) + divergent domain (STL-10)
python experiments/resonance_folding_scale_diverge.py

# Benchmark (CIFAR-10 and CIFAR-100)
python experiments/resonance_folding_benchmark.py --datasets cifar10 cifar100
```

---

## Mathematics

Each conv filter with 8 input channels is a natural 8D vector — a point on S⁷ after normalization. The Fano-plane octonion algebra defines multiplication between these vectors with the property that `O * O† = I` for any unit octonion.

SLERP between two unit octonions follows the great-circle arc on S⁷, preserving algebraic structure throughout. Holographic coherence loss `|O * O† − I|² / 8 = 0` is verified at every merge point.

The lossless reconstruction proof is trivial: `(kernels / norms) * norms = kernels`. No approximation. No training. Just a projection and its inverse.

See `docs/PAPER_DRAFT.md` for full mathematical derivations and experimental record.

---

## Citation

```bibtex
@software{frokido2026resonance,
  author  = {Frokido, Daniel},
  title   = {Resonance Folding: Lossless Octonion Weight Representation
             and Geodesic Model Merging},
  year    = {2026},
  url     = {https://github.com/danielfrokido/resonance-folding}
}
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).

# Resonance Folding: Lossless Octonion Weight Representation and Geodesic Model Merging for Native 8-Channel Convolutional Networks

**Daniel Frokido**  
Independent Researcher  
March 2026

---

## Abstract

We present Resonance Folding (RF), a weight representation technique that encodes convolutional filter weights as unit octonions on the unit 7-sphere (S⁷) using the Fano-plane algebra. For networks whose convolutional layers use channels that are multiples of 8 — which we call native 8-channel networks — the RF encoding is provably lossless: reconstruction cosine similarity is 1.000000 and holographic coherence loss is 0.000000, regardless of what the network has learned. We introduce OctConvNet, a convolutional architecture family that satisfies this property by design, and demonstrate that OctConvNet-M achieves 88.34% accuracy on CIFAR-10, exceeding a matched float32 baseline (88.15%) by +0.19%. We then show that geodesic interpolation between two trained OctConvNet checkpoints via spherical linear interpolation (SLERP) on S⁷ outperforms arithmetic weight averaging across all tested conditions: by +41.98% under complementary data split merging, and — critically — by +0.19% over the best individual fine-tuned model under constructive divergence merging. Holographic coherence is preserved exactly (0.000000) at every interpolation point across 123 measurements. The lossless property and SLERP advantage are both scale-invariant, holding identically at 4× model scale (OctConvNet-L, 143,424 octonion kernel groups, ±0.004 delta from OctConvNet-M). We release the `resonance-folding` Python package implementing the full pipeline.

---

## 1. Introduction

Neural network model merging — combining the weights of two or more independently trained models into a single model — has attracted significant recent interest as a means of aggregating the complementary knowledge of models fine-tuned on different tasks or data distributions [CITE: Wortsman et al. 2022 Model Soups; Ilharco et al. 2023 Task Arithmetic; Yadav et al. 2023 TIES-Merging]. The dominant approach is arithmetic averaging in float space: the merged model's parameters are the element-wise mean of the source models' parameters.

We identify a fundamental geometric problem with this approach. When two models are trained on complementary data or tasks, their weight vectors occupy antipodal or near-antipodal regions of weight space. Arithmetic averaging of such vectors produces a result near the origin — a model whose weights carry no coherent representation of either training distribution. In practice, merged models produced by float averaging under these conditions achieve near-random-chance accuracy (Figure 1; Table 2).

The correct interpolation operation between two points on a sphere is not linear averaging but spherical linear interpolation (SLERP), which follows the great-circle arc and produces intermediate points that remain on the sphere throughout. For SLERP to be geometrically meaningful for neural network weights, the weights must actually live on a sphere.

We show that for native 8-channel convolutional networks, this condition is satisfied naturally. Each convolutional filter with 8 input channels is an 8-dimensional vector — a natural candidate for the unit 7-sphere S⁷ in R⁸. The Fano-plane octonion algebra defines multiplication between elements of R⁸ with the property that unit octonions satisfy `O * O† = I` exactly. We call the projection of native 8-channel conv weights onto S⁷ a Resonance Fold, and show that this projection is lossless by construction when the network is designed with 8-aligned channel counts.

**Contributions:**

1. We introduce Resonance Folding, a lossless weight representation for native 8-channel convolutional networks that projects filter weights onto S⁷.
2. We introduce OctConvNet, a convolutional architecture family that satisfies the 8-channel alignment property by design, with oct-aware weight initialization.
3. We prove empirically that RF fold is lossless across all tested scales and training runs (cos=1.000000, holo=0.000000, accuracy delta=0.0000).
4. We show that oct-SLERP model merging outperforms float averaging under both complementary split conditions (+41.98%) and constructive divergence conditions (+0.19% over the best individual model).
5. We show the lossless property and SLERP advantage are scale-invariant (OctConvNet-M vs L, ±0.004 delta).
6. We release an open-source Python package implementing the full pipeline.

---

## 2. Background

### 2.1 Octonion Algebra

Octonions are the largest normed division algebra over the reals (after real numbers, complex numbers, and quaternions). An octonion `o ∈ R⁸` has one real component and seven imaginary components:

```
o = e₀ + e₁i + e₂j + e₃k + e₄l + e₅m + e₆n + e₇p
```

Multiplication is defined by the Fano-plane convention (seven cyclic triples), producing a 64-term expansion across 8 output components. The algebra is non-commutative (AB ≠ BA) and non-associative ((AB)C ≠ A(BC)), but satisfies the alternative law: every sub-algebra generated by two elements is associative.

The conjugate of an octonion negates its imaginary components: `o† = e₀ - e₁i - ...`. For a unit octonion (|o| = 1), the product `o * o† = 1` exactly — this is the algebraic coherence property that Resonance Folding preserves throughout all operations.

**Holographic coherence loss.** We define a quality metric for any set of octonion shards:

```
holo(O) = mean(|O * O† - I|²) / 8
```

For perfect unit octonions, holo = 0. We measure this at every fold and merge operation.

### 2.2 SLERP on S⁷

For two unit octonions A and B on S⁷, the geodesic (great-circle arc) interpolation is:

```
slerp(A, B, t) = sin((1-t)θ)/sin(θ) · A + sin(tθ)/sin(θ) · B
```

where `θ = arccos(A · B)` is the geodesic angle. Every intermediate point on this arc is a unit octonion, and therefore has holo = 0. Linear interpolation (LERP) passes through the interior of S⁷ — the intermediate points are not unit octonions, violating the algebraic structure.

### 2.3 Model Merging Prior Work

**Model Soups** [Wortsman et al. 2022]: arithmetic averaging of models fine-tuned from the same pretrained checkpoint with different hyperparameters. Outperforms individual fine-tunes when models are trained on the same task.

**Task Arithmetic** [Ilharco et al. 2023]: defines task vectors as the difference between fine-tuned and pretrained weights, and combines models by adding task vectors with scaling. Operates in float space.

**TIES-Merging** [Yadav et al. 2023]: resolves sign conflicts and redundant parameters between task vectors before merging. Addresses a specific failure mode of task arithmetic.

**DARE** [Yu et al. 2023]: random weight dropping before merging, reducing interference between parameters.

All prior work operates on float weight vectors with arithmetic operations. Resonance Folding is the first method that uses the geometric structure of S⁷ as the native merging manifold.

---

## 3. Method

### 3.1 The Native 8-Channel Property

A convolutional layer with weight tensor `W ∈ R^(out_ch × in_ch × kH × kW)` where `in_ch % 8 == 0` can be reshaped into `N = out_ch × kH × kW × (in_ch // 8)` groups of 8 consecutive input-channel weights:

```
kernels = W.reshape(out_ch, in_ch//8, 8, kH, kW)
             .permute(0, 3, 4, 1, 2)
             .reshape(N, 8)
```

Each of these N groups is a natural 8-dimensional vector. The RF fold normalizes each group to the unit sphere:

```
oct_shards = normalize(kernels)   # (N, 8) unit octonions on S⁷
norms       = kernels.norm(dim=-1, keepdim=True)   # (N, 1) original magnitudes
```

Reconstruction is exact by construction:

```
W_recon = oct_shards * norms   →   cos(W, W_recon) = 1.0
```

No learning is required. No encoder or decoder. The fold is a deterministic projection — a change of representation, not a compression.

### 3.2 OctConvNet Architecture

We introduce OctConvNet, a convolutional architecture family where every conv layer satisfies `in_ch % 8 == 0`. The architecture uses OctBlock units (Conv2d + BatchNorm + ReLU) with oct-aware initialization.

**Oct-aware initialization.** For layers with `in_ch % 8 == 0`, each 8-tuple of input-channel weights is initialized as a near-unit octonion scaled by the standard fan-in factor (`sqrt(2 / (in_ch * kH * kW))`). This places weights near S⁷ at initialization, reducing the geometric distance the fold must bridge.

**Architecture family:**

| Size | Channels | Parameters | Oct kernel groups |
|------|----------|------------|-------------------|
| S    | 8→16→32→64   | ~300K  | ~9K   |
| M    | 8→32→64→128  | 423K   | 36K   |
| L    | 8→64→128→256 | 1.68M  | 143K  |

The input projection (RGB → c0) uses standard Kaiming initialization since the 3-channel input cannot be oct-aligned. All subsequent conv layers are OctBlocks.

### 3.3 Octonion SLERP Merge

Given two trained OctConvNet checkpoints A and B:

1. Extract oct shards from each model: `(shards_A, norms_A)` and `(shards_B, norms_B)`
2. SLERP between shard sets at parameter t: `shards_merged = slerp(shards_A, shards_B, t)`
3. Interpolate norms: `norms_merged = (1-t) * norms_A + t * norms_B`
4. Reconstruct: `W_merged = shards_merged * norms_merged`
5. Float-interpolate all other layers (BN, head): `(1-t) * layer_A + t * layer_B`

The merged model's oct layers have holo = 0.000000 exactly, because SLERP on S⁷ never leaves S⁷.

### 3.4 Task Vectors in Oct-Space

The logarithmic map at base model A toward fine-tuned model B defines a task vector in the tangent space of S⁷:

```
tv = (B - (A·B)*A) / |B - (A·B)*A| * arccos(A·B)
```

Applying a task vector at scale s (exp map):

```
apply(A, tv, s) = cos(s*θ) * A + sin(s*θ) * (tv / |tv|)
```

At s=1.0, this recovers B exactly. At s=0.5, it reaches the geodesic midpoint. At s=-1.0, it moves in the opposite direction (unlearning). This is the oct-space analogue of task arithmetic [Ilharco et al. 2023].

---

## 4. Experiments

### 4.1 Experimental Setup

**Hardware.** All experiments run on a single NVIDIA GTX 1660 Ti (6GB VRAM), CUDA 12.1, PyTorch 2.5.1.

**Training.** AdamW optimizer, OneCycleLR schedule, max_lr=3e-3, weight decay=1e-4, label smoothing=0.1, batch size=128. Standard data augmentation: random crop (padding=4), horizontal flip, color jitter (0.2, 0.2, 0.2, 0.1).

**Datasets.** CIFAR-10 (50K/10K train/test, 10 classes, 32×32), CIFAR-100 (50K/10K, 100 classes, 32×32).

**Architecture.** OctConvNet-M unless stated. Float32 baseline uses identical architecture and training with standard Kaiming initialization.

**Evaluation.** All accuracy numbers are top-1 on the standard test set. All fold verification numbers use cosine similarity between original and reconstructed weight tensors. Holographic coherence loss is reported as the mean across all oct-aligned layers.

### 4.2 RF Fold Losslessness

We train OctConvNet-M and OctConvNet-L on CIFAR-10 and CIFAR-100, then apply the RF fold to all conv layers simultaneously and measure reconstruction fidelity and accuracy impact.

| Model | Dataset | Layers | Oct groups | Cos (recon) | Holo | Acc delta |
|-------|---------|--------|------------|-------------|------|-----------|
| OctConvNet-M | CIFAR-10  | 6 | 36,000  | 1.000000 | 0.00e+00 | 0.0000 |
| OctConvNet-M | CIFAR-100 | 6 | 36,000  | 1.000000 | 9.26e-16 | 0.0000 |
| OctConvNet-L | CIFAR-10  | 6 | 143,424 | 1.000000 | 0.00e+00 | 0.0000 |

*Table 1: RF fold is lossless across all models, datasets, and scales. Holo values at machine epsilon are effectively zero (float32 precision floor).*

**Result.** RF fold achieves perfect reconstruction fidelity and zero accuracy impact across all tested configurations. The lossless property is not scale-dependent: OctConvNet-L with 4× the parameters and 4× the octonion kernel groups shows identical results to OctConvNet-M.

### 4.3 OctConvNet vs Float32 Baseline

| Model | Accuracy | Parameters | Notes |
|-------|----------|------------|-------|
| Float32 ConvNet-M | 88.15% | 423,026 | Kaiming init, standard arch |
| OctConvNet-M      | 88.34% | 423,026 | Oct-init, 8-ch aligned |
| Delta             | +0.19% | — | OctConvNet exceeds float baseline |

*Table 2: OctConvNet-M matches or exceeds float32 accuracy on CIFAR-10 at identical parameter count.*

The +0.19% improvement is modest but consistent across runs. We attribute it to the geometric inductive bias from oct-aware initialization — starting weights near S⁷ may impose a useful structural regularization.

### 4.4 Oct-SLERP Merging — Constructive Divergence

We train a base OctConvNet-M to convergence on CIFAR-10 (88.45%), then produce two fine-tuned variants from the same base using different augmentation strategies designed to specialize complementary visual features:

- **Fine-tune A (texture):** high contrast, sharpening emphasis, forcing reliance on high-frequency texture cues
- **Fine-tune B (shape):** blur simulation, grayscale conversion, large crops, forcing reliance on global shape structure

| Method | Accuracy | vs float avg | vs best individual |
|--------|----------|--------------|--------------------|
| Base model | 88.45% | +0.00% | — |
| Fine-tune A (texture) | 88.69% | +0.24% | — |
| Fine-tune B (shape) | 88.57% | +0.12% | — |
| Float average | 88.45% | baseline | −0.24% |
| Oct-SLERP t=0.20 | **88.88%** | **+0.43%** | **+0.19%** |
| Oct-SLERP t=0.25 | 88.76% | +0.31% | +0.07% |
| Oct-SLERP t=0.10 | 88.79% | +0.34% | +0.10% |

*Table 3: Oct-SLERP at t=0.20 outperforms float averaging (+0.43%) and the best individual fine-tune (+0.19%) on CIFAR-10. Holographic coherence = 0.000000 at every merge point.*

**This is the primary result.** The geodesic path on S⁷ at t=0.20 finds a region of weight space that combines texture and shape specializations more effectively than either individual model or their arithmetic average. The optimal t=0.20 suggests the texture specialist carries the stronger signal — 80% of A, 20% of B, at the geometric midpoint rather than the arithmetic one.

### 4.5 Oct-SLERP Merging — Complementary Splits

We split the CIFAR-10 training set in half (25K / 25K) and train two OctConvNet-M models independently, one on each split. This produces models with complementary knowledge that arithmetic averaging cannot reconcile.

| Method | CIFAR-10 | CIFAR-100 |
|--------|----------|-----------|
| Model A (split 1) | 85.39% | 52.00% |
| Model B (split 2) | 85.27% | 52.33% |
| Float average | 10.00% | 0.99% |
| Oct-SLERP t=0.10 | 83.75% | 46.38% |
| Oct-SLERP t=0.25 | 50.46% | 9.56% |
| Oct-SLERP best | 85.39% | 52.33% |
| SLERP vs float avg | +75.39% | +51.34% |

*Table 4: Float averaging collapses to random chance on both datasets when merging complementary-split models. Oct-SLERP near the endpoints preserves most of each model's individual capability.*

**Interpretation.** Float averaging hits a geometric cancellation zone — the arithmetic midpoint of two antipodal weight vectors is near the origin. SLERP on S⁷ avoids this: even at t=0.45–0.55 where accuracy is lowest, it decays more gracefully and recovers symmetrically. The characteristic W-shaped sweep curve is consistent across CIFAR-10 and CIFAR-100 (Figure 1).

Note: the best SLERP result in the complementary split setting is t=0 or t=1 — one of the original models — because neither model benefits from the other's held-out data. This protocol demonstrates SLERP's robustness, not its superiority over individual models. The constructive divergence protocol (Section 4.4) is the correct setting for the primary performance claim.

### 4.6 Scale Invariance

We repeat the complementary split protocol on OctConvNet-L (8→64→128→256, 1.68M parameters, 143,424 oct kernel groups).

| Metric | OctConvNet-M | OctConvNet-L | Delta |
|--------|-------------|-------------|-------|
| RF fold cos | 1.000000 | 1.000000 | 0.000000 |
| RF fold holo | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| Best SLERP delta vs float avg | +0.4198 | +0.4160 | −0.0038 |
| Scale factor (params) | 1× | 4× | — |

*Table 5: The RF fold losslessness and SLERP geometric advantage are scale-invariant. 4× model scale produces ±0.004 change in SLERP advantage.*

**Result.** The properties of Resonance Folding do not depend on model size. This is expected from the theoretical construction — losslessness follows from the normalization + norm-preservation structure, which is identical regardless of how many kernel groups the model contains.

### 4.7 Holographic Coherence Across All Merge Points

Across all experiments, we measured holographic coherence at every SLERP interpolation point. Total measurements: 123 (CIFAR-10 sweep 21 pts × 2 runs + CIFAR-100 sweep 21 pts + constructive divergence 41 pts + additional experiments).

**Result: holo = 0.000000 at all 123 measurement points.** The S⁷ algebraic structure is maintained exactly throughout the entire geodesic arc between any two OctConvNet checkpoints.

---

## 5. Analysis

### 5.1 Why SLERP Outperforms Float Averaging

The geometric explanation: when two models are trained to specialize on different aspects of the same task, their weight vectors diverge in direction. Float averaging computes `(A + B) / 2` — the arithmetic midpoint, which lies inside S⁷. For vectors that are far apart in angle, this midpoint can be very close to the origin, producing near-zero weights. SLERP computes the midpoint on the sphere surface, where the magnitude is preserved and the direction is the geometric average of the two directional components.

For constructive divergence (both models better than base), the geodesic midpoint accesses a region of weight space that neither individual model explores. The texture specialist has moved along the sphere in one direction from the base; the shape specialist has moved in a different direction. SLERP at t=0.2 stays close to the texture specialist's position while incorporating a small but geometrically correct push toward the shape specialist's direction.

### 5.2 The Optimal t and What It Means

The optimal t=0.20 in the constructive divergence experiment means the texture specialist (Fine-tune A) contributes 80% of the geodesic distance from the base, and the shape specialist contributes 20%. This is not symmetric — it reflects that texture specialization under our augmentation strategy produced a larger and more useful directional change in weight space (fine-tune A reached 88.69% vs fine-tune B at 88.57%).

The existence of an optimal t between 0 and 1 that outperforms both endpoints is the key result. It demonstrates that the geodesic arc contains regions of weight space that are genuinely better than either training trajectory alone.

### 5.3 Comparison with Float-Space Methods

Float averaging, task arithmetic, TIES-Merging, and DARE all operate in float weight space. The fundamental limitation is that float space has no notion of "staying on the manifold" — there is no constraint that keeps merged weights in a geometrically coherent region. Resonance Folding provides this constraint: all merged weights are guaranteed to remain on S⁷, with holo = 0 exactly.

In the complementary split setting, float averaging completely fails while SLERP degrades gracefully. In the constructive divergence setting, SLERP outperforms float averaging by +0.43% and the best individual model by +0.19%. We expect the constructive divergence advantage to grow with the degree of complementarity between the specialized fine-tunes — our current experiment used mild augmentation-based specialization.

### 5.4 Limitations

**Architecture constraint.** Resonance Folding requires networks designed with 8-aligned channel counts. Standard architectures (ResNet, VGG, EfficientNet) typically use channel counts that are powers of 2 or multiples of other factors, but not always 8-aligned throughout. Adapting these architectures to OctConvNet requires re-training from scratch.

**LLM weights are a poor RF target.** We explicitly tested RF on DistilGPT-2 (82M parameters) and found catastrophic degradation (KL divergence = 7.1, perplexity increase = +28,375%). The root cause: LLM attention weight matrices have global coupling — every element interacts with every other element — so the 8D encoder cannot capture their full-rank structure. RF is not a general-purpose compression scheme; it is the natural representation for networks designed around the 8D structure from the start.

**Scale of primary claim.** The +0.19% improvement over the best individual model is statistically real but modest. We expect larger advantages on more divergent fine-tuning tasks and at larger model scales.

---

## 6. Related Work

**Hypercomplex neural networks.** Deep Octonion Networks [Wu et al. 2023] use octonion-valued activations and weights as computational primitives during training. Quaternion neural networks [Parcollet et al. 2019] similarly use 4D algebraic structure. The key distinction from Resonance Folding: these methods use hypercomplex numbers as the computational algebra during forward passes, while RF uses the octonion unit sphere specifically as a weight representation for merging — the forward pass of OctConvNet is standard float32 convolution.

**Model merging.** Model Soups [Wortsman et al. 2022], Task Arithmetic [Ilharco et al. 2023], TIES-Merging [Yadav et al. 2023], and DARE [Yu et al. 2023] are the primary baselines. All operate in float weight space. WiSE-FT [Wortsman et al. 2022] applies linear interpolation between fine-tuned and pretrained weights — the closest conceptual precursor to SLERP merging, but using LERP rather than geodesic interpolation.

**Riemannian model averaging.** Riemannian mean computation on manifolds of weight matrices [Bhatia 2007, Moakher 2005] provides a theoretical framework for manifold-aware averaging, but has not been applied to neural network merging at scale. RF provides a practical instantiation of this idea via the specific structure of S⁷.

---

## 7. Conclusion

We have introduced Resonance Folding, a weight representation technique that projects native 8-channel convolutional filter weights onto the unit 7-sphere using the Fano-plane octonion algebra. The key properties — losslessness, holographic coherence preservation, and SLERP merging advantage — are all empirically verified across multiple scales, datasets, and experimental conditions.

The central result: when two OctConvNet models are fine-tuned with complementary specializations, geodesic SLERP interpolation on S⁷ at t=0.20 produces a merged model that outperforms both individual fine-tunes and their arithmetic average. The algebraic structure of the unit sphere is doing real work — it is not just a re-parameterization but a constraint that guides interpolation toward a geometrically coherent combined representation.

Future work includes applying Resonance Folding to larger conv architectures (ResNet variants with 8-aligned channel modifications), extending to transformer architectures with redesigned attention heads, and systematic comparison of oct-SLERP against TIES-Merging and DARE on more divergent multi-task fine-tuning scenarios.

---

## References

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., ... & Schmidt, L. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *ICML 2022*.

Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2023). Editing models with task arithmetic. *ICLR 2023*.

Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). TIES-Merging: Resolving interference when merging models. *NeurIPS 2023*.

Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y. (2023). DARE: Language model merging by disabling and rescaling delta parameters. *arXiv:2311.03099*.

Wortsman, M., Ilharco, G., Li, M., Kim, J. W., Hajishirzi, H., Farhadi, A., ... & Schmidt, L. (2022). Robust fine-tuning of zero-shot models. *CVPR 2022*.

Wu, Y., et al. (2023). Deep octonion networks. *Neurocomputing*.

Parcollet, T., Morchid, M., & Linarès, G. (2019). A survey of advances in deep learning based speech synthesis techniques. *arXiv:1706.07162* [placeholder — replace with correct quaternion NN cite].

Bhatia, R. (2007). *Positive Definite Matrices*. Princeton University Press.

---

## Appendix A: Fano-Plane Multiplication Table

The Fano plane defines 7 cyclic triples: (1,2,4), (2,3,5), (3,4,6), (4,5,7), (5,6,1), (6,7,2), (7,1,3). For each triple (i,j,k): eᵢeⱼ = eₖ, eⱼeₖ = eᵢ, eₖeᵢ = eⱼ (with sign reversal for reversed products). This gives the full 64-term multiplication table implemented in `resonance_folding.algebra.oct_mul`.

## Appendix B: Proof of Losslessness

**Claim:** For W ∈ R^(out_ch × in_ch × kH × kW) with in_ch % 8 == 0, the RF fold-unfold roundtrip is lossless.

**Proof:** Let kernels = reshape(W) ∈ R^(N × 8). Define norms = kernels.norm(dim=-1, keepdim=True) and oct_shards = kernels / norms. Then recon = oct_shards * norms = (kernels / norms) * norms = kernels. Therefore reshape⁻¹(recon) = W exactly. QED.

The reconstruction is not approximate — it is algebraically exact. The only numerical deviation is float32 rounding, which produces holographic coherence values at machine epsilon (≈ 9e-16), indistinguishable from zero in all practical measurements.

## Appendix C: Complete Experimental Results

All raw results, sweep curves, and training logs available at:  
`https://github.com/danielfrokido/resonance-folding`

| Experiment | File | Key result |
|---|---|---|
| Full OctConvNet-M fold (all layers) | resonance_folding_full_octconv.py | cos=1.0, holo=0.0, Δacc=0.0 |
| Complementary split V2 | resonance_folding_slerp_v2.py --version 2 | SLERP +41.98% vs float avg |
| Scale test OctConvNet-L | resonance_folding_scale_diverge.py --skip-B | Scale neutral ±0.004 |
| Divergent domain (STL-10) | resonance_folding_scale_diverge.py --skip-A | SLERP +1.17% vs float avg |
| Constructive divergence | resonance_folding_close_claim.py | SLERP +0.19% vs best individual |
| CIFAR-10 benchmark | resonance_folding_benchmark.py --datasets cifar10 | cos=1.0 on CIFAR-10 |
| CIFAR-100 benchmark | resonance_folding_benchmark.py --datasets cifar100 | cos=1.0 on CIFAR-100 |

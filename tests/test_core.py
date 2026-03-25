"""
tests/test_core.py
==================
Verifies all core package claims pass as unit tests.

Run:  pytest tests/ -v
"""

import math
import torch
import pytest

from resonance_folding import (
    oct_mul, oct_conj, oct_normalize, holo_loss, assoc_loss,
    oct_slerp, task_vector, task_vector_apply,
    OctBlock, OctConvNet, OctBasicBlock, OctResNet18, oct_init_,
    FoldedLayer, fold_model, patch_model, verify_fold,
    weight_to_oct, oct_to_weight,
    float_average, slerp_merge, triple_slerp,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────
#  ALGEBRA
# ─────────────────────────────────────────────────────────────

class TestAlgebra:

    def test_oct_mul_identity(self):
        """Multiplying by the real unit octonion (1,0,...,0) is identity."""
        x = torch.randn(16, 8)
        e = torch.zeros(16, 8); e[:, 0] = 1.0
        result = oct_mul(x, e)
        assert torch.allclose(result, x, atol=1e-5), "oct_mul identity failed"

    def test_oct_mul_conjugate_norm(self):
        """O * O† = |O|² * e₀  (conjugate product gives squared norm in real component)."""
        O   = torch.randn(32, 8) * 2.0   # arbitrary, not unit
        OOc = oct_mul(O, oct_conj(O))
        expected_real = (O ** 2).sum(-1)   # |O|²
        assert torch.allclose(OOc[..., 0], expected_real, atol=1e-4), \
            "O * O† real component should equal |O|²"
        assert torch.allclose(OOc[..., 1:], torch.zeros(32, 7), atol=1e-4), \
            "O * O† imaginary components should be zero"

    def test_oct_conj_roundtrip(self):
        """Conjugate of conjugate is identity."""
        x = torch.randn(8, 8)
        assert torch.allclose(oct_conj(oct_conj(x)), x), "double conjugate failed"

    def test_oct_normalize_unit(self):
        """All normalized octonions have norm 1."""
        x = torch.randn(64, 8) * 5.0   # deliberately un-normalized
        n = oct_normalize(x)
        norms = n.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(64), atol=1e-6), \
            "oct_normalize produced non-unit vectors"

    def test_holo_loss_zero_for_unit(self):
        """Unit octonions have holo_loss = 0."""
        O = oct_normalize(torch.randn(64, 8))
        loss = holo_loss(O).item()
        assert loss < 1e-5, f"holo_loss={loss:.2e} for unit octonions (expected ~0)"

    def test_holo_loss_nonzero_for_non_unit(self):
        """Non-unit octonions have holo_loss > 0."""
        O = torch.randn(64, 8) * 3.0   # definitely not unit
        loss = holo_loss(O).item()
        assert loss > 0.01, "holo_loss unexpectedly small for non-unit octonions"

    def test_oct_slerp_endpoints(self):
        """SLERP at t=0 returns A, at t=1 returns B."""
        A = oct_normalize(torch.randn(16, 8))
        B = oct_normalize(torch.randn(16, 8))
        assert torch.allclose(oct_slerp(A, B, 0.0), A, atol=1e-5), \
            "SLERP t=0 should return A"
        assert torch.allclose(oct_slerp(A, B, 1.0), B, atol=1e-5), \
            "SLERP t=1 should return B"

    def test_oct_slerp_stays_on_sphere(self):
        """SLERP interpolants are unit octonions."""
        A = oct_normalize(torch.randn(32, 8))
        B = oct_normalize(torch.randn(32, 8))
        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            interp = oct_slerp(A, B, t)
            norms  = interp.norm(dim=-1)
            assert torch.allclose(norms, torch.ones(32), atol=1e-5), \
                f"SLERP t={t}: result not on unit sphere"

    def test_oct_slerp_holo_zero(self):
        """SLERP interpolants have holographic coherence ≈ 0."""
        A = oct_normalize(torch.randn(64, 8))
        B = oct_normalize(torch.randn(64, 8))
        for t in [0.25, 0.5, 0.75]:
            interp = oct_slerp(A, B, t)
            hl = holo_loss(interp).item()
            assert hl < 1e-5, f"SLERP t={t}: holo_loss={hl:.2e} (expected ~0)"

    def test_task_vector_roundtrip(self):
        """Applying a task vector at scale=1.0 reaches the target."""
        base = oct_normalize(torch.randn(16, 8))
        ft   = oct_normalize(torch.randn(16, 8))
        tv   = task_vector(base, ft)
        recovered = task_vector_apply(base, tv, scale=1.0)
        assert torch.allclose(recovered, ft, atol=1e-4), \
            "task_vector + task_vector_apply roundtrip failed"


# ─────────────────────────────────────────────────────────────
#  ARCHITECTURE
# ─────────────────────────────────────────────────────────────

class TestArch:

    def test_octblock_alignment(self):
        """OctBlock rejects non-8-aligned channel counts."""
        with pytest.raises(AssertionError):
            OctBlock(3, 32)   # in_ch=3 is not % 8
        with pytest.raises(AssertionError):
            OctBlock(8, 33)   # out_ch=33 is not % 8

    def test_octblock_forward(self):
        """OctBlock produces correct output shape."""
        block = OctBlock(8, 32)
        x     = torch.randn(4, 8, 32, 32)
        y     = block(x)
        assert y.shape == (4, 32, 32, 32), f"OctBlock output shape wrong: {y.shape}"

    def test_octnet_sizes(self):
        """OctConvNet S/M/L have expected parameter counts."""
        for size, min_params in [("S", 50_000), ("M", 300_000), ("L", 1_000_000)]:
            model = OctConvNet(size=size)
            n = model.n_params
            assert n > min_params, f"OctConvNet-{size} has too few params: {n}"

    def test_octnet_forward(self):
        """OctConvNet-M produces (B, 10) logits for CIFAR-10 inputs."""
        model = OctConvNet(size="M", n_classes=10)
        x     = torch.randn(4, 3, 32, 32)
        y     = model(x)
        assert y.shape == (4, 10), f"OctConvNet output shape wrong: {y.shape}"

    def test_octnet_oct_layers(self):
        """OctConvNet-M reports 6 oct layers."""
        model  = OctConvNet(size="M")
        layers = model.oct_layers()
        assert len(layers) == 6, f"Expected 6 oct layers, got {len(layers)}"

    def test_octnet_oct_groups(self):
        """OctConvNet-M has the correct number of oct kernel groups."""
        model = OctConvNet(size="M")
        # M: 8→32→64→128, 3×3 kernels, 2 blocks per stage
        expected = (
            32 * 3 * 3 * (8  // 8) +   # stage1.0: (32, 8, 3, 3)
            32 * 3 * 3 * (32 // 8) +   # stage1.1: (32, 32, 3, 3)
            64 * 3 * 3 * (32 // 8) +   # stage2.0
            64 * 3 * 3 * (64 // 8) +   # stage2.1
            128 * 3 * 3 * (64  // 8) + # stage3.0
            128 * 3 * 3 * (128 // 8)   # stage3.1
        )
        assert model.n_oct_groups == expected, \
            f"Expected {expected} oct groups, got {model.n_oct_groups}"


# ─────────────────────────────────────────────────────────────
#  FOLD — THE CORE CLAIM
# ─────────────────────────────────────────────────────────────

class TestFold:

    def test_weight_to_oct_shape(self):
        """weight_to_oct produces correct output shapes."""
        W = torch.randn(32, 8, 3, 3)
        shards, norms = weight_to_oct(W)
        expected_N = 32 * 3 * 3 * (8 // 8)   # = 288
        assert shards.shape == (expected_N, 8), f"shards shape wrong: {shards.shape}"
        assert norms.shape  == (expected_N, 1), f"norms shape wrong: {norms.shape}"

    def test_weight_to_oct_unit_norm(self):
        """weight_to_oct produces unit octonions."""
        W      = torch.randn(32, 8, 3, 3)
        shards, _ = weight_to_oct(W)
        norms  = shards.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(norms.shape[0]), atol=1e-6)

    def test_oct_roundtrip_lossless(self):
        """weight_to_oct → oct_to_weight roundtrip is lossless."""
        W = torch.randn(64, 32, 3, 3)
        shards, norms = weight_to_oct(W)
        W_recon = oct_to_weight(shards, norms, W.shape)
        cos = torch.nn.functional.cosine_similarity(
            W.flatten().unsqueeze(0), W_recon.flatten().unsqueeze(0)
        ).item()
        assert cos > 0.9999, f"roundtrip cosine={cos:.6f} (expected 1.0)"
        assert torch.allclose(W, W_recon, atol=1e-5), "roundtrip not element-wise close"

    def test_fold_model_lossless(self):
        """fold_model on OctConvNet-M is lossless for all layers."""
        model  = OctConvNet(size="M")
        folded = fold_model(model)
        result = verify_fold(model, folded, verbose=False)
        assert result["all_lossless"], (
            f"Fold not lossless: mean_cos={result['mean_cos']:.6f}, "
            f"min_cos={result['min_cos']:.6f}"
        )
        assert result["mean_cos"] > 0.9999, \
            f"mean_cos={result['mean_cos']:.6f}"
        assert result["mean_holo"] < 1e-6, \
            f"mean_holo={result['mean_holo']:.2e}"

    def test_fold_holo_zero(self):
        """All folded layers have holo_loss = 0."""
        model  = OctConvNet(size="M")
        folded = fold_model(model)
        for name, fl in folded.items():
            hl = holo_loss(fl.shards).item()
            assert hl < 1e-6, f"Layer {name}: holo={hl:.2e}"

    def test_patch_model_restores_weights(self):
        """patch_model restores original weights exactly."""
        import copy
        model  = OctConvNet(size="M")
        orig   = copy.deepcopy(model)
        folded = fold_model(model)
        # Corrupt the model weights
        for p in model.parameters():
            p.data.fill_(0.0)
        # Restore via patch
        patch_model(model, folded)
        # Check restoration
        for (n1, p1), (n2, p2) in zip(orig.named_parameters(), model.named_parameters()):
            if any(k in n1 for k in ["stage1", "stage2", "stage3"]) and "conv" in n1:
                assert torch.allclose(p1.data, p2.data, atol=1e-5), \
                    f"Weights not restored for {n1}"

    def test_fold_n_layers(self):
        """OctConvNet-M folds exactly 6 layers."""
        model  = OctConvNet(size="M")
        folded = fold_model(model)
        assert len(folded) == 6, f"Expected 6 folded layers, got {len(folded)}"


# ─────────────────────────────────────────────────────────────
#  MERGE
# ─────────────────────────────────────────────────────────────

class TestMerge:

    def _make_models(self):
        torch.manual_seed(0)
        a = OctConvNet(size="S")
        torch.manual_seed(1)
        b = OctConvNet(size="S")
        return a, b

    def test_float_average_midpoint(self):
        """Float average is the arithmetic midpoint of all parameters."""
        a, b   = self._make_models()
        merged = float_average(a, b)
        sd_a, sd_b, sd_m = a.state_dict(), b.state_dict(), merged.state_dict()
        for k in sd_a:
            if sd_a[k].is_floating_point():
                expected = (sd_a[k] + sd_b[k]) / 2.0
                assert torch.allclose(sd_m[k], expected, atol=1e-6), \
                    f"float_average wrong for {k}"

    def test_slerp_merge_endpoints(self):
        """slerp_merge at t=0 matches model_a, t=1 matches model_b."""
        a, b = self._make_models()
        folded_a = fold_model(a)

        merged_0, _ = slerp_merge(a, b, 0.0, folded_a)
        merged_1, _ = slerp_merge(a, b, 1.0, folded_a)

        # Check a sample layer
        name = "stage1.0.conv"
        W_a = dict(a.oct_layers())[name].weight.data
        W_0 = dict(merged_0.oct_layers())[name].weight.data
        W_1 = dict(merged_1.oct_layers())[name].weight.data

        assert torch.allclose(W_a, W_0, atol=1e-5), "SLERP t=0 should equal model_a"
        assert torch.allclose(
            dict(b.oct_layers())[name].weight.data, W_1, atol=1e-5
        ), "SLERP t=1 should equal model_b"

    def test_slerp_merge_holo_zero(self):
        """slerp_merge produces merged shards with holo_loss ≈ 0."""
        a, b = self._make_models()
        for t in [0.25, 0.5, 0.75]:
            _, holo = slerp_merge(a, b, t)
            assert holo < 1e-5, f"slerp_merge t={t}: holo={holo:.2e}"

    def test_slerp_merge_output_shape(self):
        """slerp_merge produces a valid model with correct output shape."""
        a, b    = self._make_models()
        merged, _ = slerp_merge(a, b, 0.5)
        x = torch.randn(2, 3, 32, 32)
        y = merged(x)
        assert y.shape == (2, 10), f"merged model output shape wrong: {y.shape}"

    def test_triple_slerp_valid_output(self):
        """triple_slerp produces a valid model."""
        base = OctConvNet(size="S")
        torch.manual_seed(2)
        a = OctConvNet(size="S")
        torch.manual_seed(3)
        b = OctConvNet(size="S")
        merged, holo = triple_slerp(base, a, b, 0.3, 0.3)
        x = torch.randn(2, 3, 32, 32)
        y = merged(x)
        assert y.shape == (2, 10)
        assert holo < 1e-5, f"triple_slerp holo={holo:.2e}"


# ─────────────────────────────────────────────────────────────
#  OCTRESNET18
# ─────────────────────────────────────────────────────────────

class TestOctResNet18:

    def test_forward_shape(self):
        """OctResNet18 produces (B, 10) logits for CIFAR inputs."""
        model = OctResNet18(n_classes=10, cifar=True)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 10), f"Wrong output shape: {y.shape}"

    def test_oct_layers_count(self):
        """OctResNet18 has 19 oct-foldable layers (all except RGB stem)."""
        model = OctResNet18()
        layers = model.oct_layers()
        assert len(layers) == 19, \
            f"Expected 19 oct layers, got {len(layers)}"

    def test_oct_layers_all_aligned(self):
        """All oct-foldable layers have in_ch % 8 == 0."""
        model = OctResNet18()
        for name, conv in model.oct_layers():
            ic = conv.weight.shape[1]
            assert ic % 8 == 0, \
                f"Layer {name} has in_ch={ic}, not multiple of 8"

    def test_stem_not_in_oct_layers(self):
        """RGB stem (in_ch=3) is excluded from oct_layers."""
        model = OctResNet18()
        oct_names = {n for n, _ in model.oct_layers()}
        assert "stem.0" not in oct_names, "Stem should not be oct-foldable"

    def test_fold_lossless(self):
        """RF fold on OctResNet18 is lossless across all 19 layers."""
        model = OctResNet18()
        result = verify_fold(model, verbose=False)
        assert result["all_lossless"], (
            f"ResNet fold not lossless: "
            f"mean_cos={result['mean_cos']:.6f}, "
            f"min_cos={result['min_cos']:.6f}"
        )
        assert result["mean_cos"] > 0.9999
        assert result["mean_holo"] < 1e-5
        assert result["n_layers"] == 19

    def test_oct_groups_count(self):
        """OctResNet18 has 1,394,688 octonion kernel groups."""
        model = OctResNet18()
        assert model.n_oct_groups == 1_394_688, \
            f"Expected 1,394,688 groups, got {model.n_oct_groups}"

    def test_slerp_merge_valid(self):
        """slerp_merge works on OctResNet18 checkpoints."""
        torch.manual_seed(10)
        a = OctResNet18(n_classes=10)
        torch.manual_seed(11)
        b = OctResNet18(n_classes=10)
        merged, holo = slerp_merge(a, b, 0.5)
        x = torch.randn(2, 3, 32, 32)
        y = merged(x)
        assert y.shape == (2, 10)
        assert holo < 1e-5, f"ResNet slerp holo={holo:.2e}"

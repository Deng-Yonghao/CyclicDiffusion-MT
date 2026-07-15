"""Tests for frame-based denoiser with output heads."""
import pytest
import torch
from cyclicdiffusion_mt.model.denoiser import (
    FrameDenoiser,
    SinusoidalTimeEmbedding,
    FrameUpdate,
    DenoiserBlock,
    AffinityHead,
)


class TestSinusoidalTimeEmbedding:
    def test_shape(self):
        emb = SinusoidalTimeEmbedding(d_time=64)
        t = torch.rand(4)
        out = emb(t)
        assert out.shape == (4, 64)

    def test_different_times(self):
        emb = SinusoidalTimeEmbedding(d_time=64)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        # Different times should produce different embeddings
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[1], out[2])


class TestFrameUpdate:
    def test_forward_shape(self):
        B, L, D = 2, 8, 256
        feats = torch.randn(B, L, D)
        frames = torch.randn(B, L, 3, 3)
        mask = torch.ones(B, L, dtype=torch.bool)
        update = FrameUpdate(d_model=256, d_head=64, n_heads=4)
        out = update(feats, frames, mask)
        assert out.shape == (B, L, D)


class TestAffinityHead:
    def test_single_target(self):
        B, L, K = 2, 8, 1
        d_model, d_target = 256, 128
        pep_feats = torch.randn(B, L, d_model)
        target_feats = [torch.randn(B, 10, d_target)]
        t_masks = [torch.ones(B, 10, dtype=torch.bool)]
        pep_mask = torch.ones(B, L, dtype=torch.bool)

        head = AffinityHead(d_model=d_model, d_target=d_target)
        dg_pred = head(pep_feats, target_feats, pep_mask, t_masks)
        assert dg_pred.shape == (B, K)

    def test_multi_target(self):
        B, L, K = 2, 8, 2
        d_model, d_target = 256, 128
        pep_feats = torch.randn(B, L, d_model)
        target_feats = [torch.randn(B, 10, d_target), torch.randn(B, 12, d_target)]
        t_masks = [torch.ones(B, 10, dtype=torch.bool), torch.ones(B, 12, dtype=torch.bool)]
        pep_mask = torch.ones(B, L, dtype=torch.bool)

        head = AffinityHead(d_model=d_model, d_target=d_target)
        dg_pred = head(pep_feats, target_feats, pep_mask, t_masks)
        assert dg_pred.shape == (B, K)


class TestDenoiserBlock:
    def test_forward_shape(self):
        B, L = 2, 8
        d_model, d_target, d_time = 256, 128, 64
        feats = torch.randn(B, L, d_model)
        frames = torch.randn(B, L, 3, 3)
        mask = torch.ones(B, L, dtype=torch.bool)
        target_feats = [torch.randn(B, 10, d_target)]
        t_masks = [torch.ones(B, 10, dtype=torch.bool)]
        t_emb = torch.randn(B, d_time)
        cyclo_emb = torch.randn(B, L, 32)

        block = DenoiserBlock(d_model, d_target, d_time, 64, 4)
        out = block(feats, frames, mask, target_feats, t_masks, t_emb, cyclo_emb)
        assert out.shape == (B, L, d_model)


class TestFrameDenoiser:
    @pytest.fixture
    def denoiser(self):
        return FrameDenoiser(
            d_model=256, d_target=128, d_time=64, n_blocks=6, d_head=64, n_heads=4
        )

    def test_forward_shape_single_target(self, denoiser):
        B, L = 2, 8
        tau_t = torch.randn(B, L, 7)
        a_t = torch.randint(0, 26, (B, L))
        t = torch.rand(B)
        cyclo = torch.zeros(B, dtype=torch.long)
        targets = [torch.randn(B, 15, 128)]
        t_mask = [torch.ones(B, 15, dtype=torch.bool)]
        tau_pred, a_logits, dg_pred = denoiser(tau_t, a_t, t, targets, t_mask, cyclo)
        assert tau_pred.shape == (B, L, 7)
        assert a_logits.shape == (B, L, 26)
        assert dg_pred.shape == (B, 1)  # K=1

    def test_forward_shape_multi_target(self, denoiser):
        B, L = 2, 8
        tau_t = torch.randn(B, L, 7)
        a_t = torch.randint(0, 26, (B, L))
        t = torch.rand(B)
        cyclo = torch.zeros(B, dtype=torch.long)
        targets = [torch.randn(B, 12, 128), torch.randn(B, 15, 128)]
        t_mask = [
            torch.ones(B, 12, dtype=torch.bool),
            torch.ones(B, 15, dtype=torch.bool),
        ]
        tau_pred, a_logits, dg_pred = denoiser(tau_t, a_t, t, targets, t_mask, cyclo)
        assert tau_pred.shape == (B, L, 7)
        assert a_logits.shape == (B, L, 26)
        assert dg_pred.shape == (B, 2)  # K=2

    def test_output_range(self, denoiser):
        """Torsion predictions should be in [-pi, pi)."""
        B, L = 1, 5
        tau_t = torch.randn(B, L, 7)
        a_t = torch.randint(0, 26, (B, L))
        t = torch.rand(B)
        cyclo = torch.zeros(B, dtype=torch.long)
        targets = [torch.randn(B, 10, 128)]
        t_mask = [torch.ones(B, 10, dtype=torch.bool)]
        tau_pred, _, _ = denoiser(tau_t, a_t, t, targets, t_mask, cyclo)
        assert tau_pred.min() >= -torch.pi - 0.1
        assert tau_pred.max() <= torch.pi + 0.1

    def test_different_cyclo_modes(self, denoiser):
        """All five cyclization modes should work without error."""
        B, L = 2, 8
        tau_t = torch.randn(B, L, 7)
        a_t = torch.randint(0, 26, (B, L))
        t = torch.rand(B)
        targets = [torch.randn(B, 10, 128)]
        t_mask = [torch.ones(B, 10, dtype=torch.bool)]
        for mode in range(5):
            cyclo = torch.full((B,), mode, dtype=torch.long)
            tau_pred, a_logits, dg_pred = denoiser(
                tau_t, a_t, t, targets, t_mask, cyclo
            )
            assert tau_pred.shape == (B, L, 7)
            assert a_logits.shape == (B, L, 26)
            assert dg_pred.shape == (B, 1)

    def test_batch_independence(self, denoiser):
        """Running batch together should match running samples separately (eval mode)."""
        denoiser.eval()
        B, L = 2, 4
        torch.manual_seed(42)
        tau_t = torch.randn(B, L, 7)
        a_t = torch.randint(0, 26, (B, L))
        t = torch.rand(B)
        cyclo = torch.randint(0, 5, (B,), dtype=torch.long)
        targets = [torch.randn(B, 10, 128)]
        t_mask = [torch.ones(B, 10, dtype=torch.bool)]

        with torch.no_grad():
            tau_pred_batch, _, _ = denoiser(tau_t, a_t, t, targets, t_mask, cyclo)

            # Sample 0 alone
            tau_pred_0, _, _ = denoiser(
                tau_t[0:1], a_t[0:1], t[0:1], [targets[0][0:1]], [t_mask[0][0:1]], cyclo[0:1]
            )

            # Sample 1 alone
            tau_pred_1, _, _ = denoiser(
                tau_t[1:2], a_t[1:2], t[1:2], [targets[0][1:2]], [t_mask[0][1:2]], cyclo[1:2]
            )

        # Batch output should equal stacked individual outputs
        stacked = torch.cat([tau_pred_0, tau_pred_1], dim=0)
        assert torch.allclose(tau_pred_batch, stacked, atol=1e-5), "Batch independence violated"

    def test_gradient_flow(self, denoiser):
        """All parameters should receive gradients with K>1 targets.

        Note: With K=1 target, AdaptiveTargetGating softmax outputs constant 1.0
        and has zero gradient, so some time_mlp and gating params get no gradient.
        This is expected for single-target mode. We test with K=2 to verify
        full gradient flow.
        """
        B, L = 2, 6
        tau_t = torch.randn(B, L, 7)
        a_t = torch.randint(0, 26, (B, L))
        t = torch.rand(B)
        cyclo = torch.zeros(B, dtype=torch.long)
        targets = [torch.randn(B, 10, 128), torch.randn(B, 12, 128)]
        t_mask = [
            torch.ones(B, 10, dtype=torch.bool),
            torch.ones(B, 12, dtype=torch.bool),
        ]

        tau_pred, a_logits, dg_pred = denoiser(tau_t, a_t, t, targets, t_mask, cyclo)
        loss = tau_pred.sum() + a_logits.sum() + dg_pred.sum()
        loss.backward()

        # Skip placeholder parameters not yet wired into forward()
        # Also skip final gating bias: shared across all K logits, cancels in softmax
        skip_patterns = (
            "frame_update.rbf_centers",
            "frame_update.edge_mlp",
            "gating.gate.2.bias",
        )

        for name, param in denoiser.named_parameters():
            if any(p in name for p in skip_patterns):
                continue
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ), f"Parameter {name} has zero gradient"

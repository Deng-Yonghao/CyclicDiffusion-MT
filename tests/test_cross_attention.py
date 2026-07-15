import pytest
import torch
from cyclicdiffusion_mt.model.cross_attention import MultiTargetCrossAttention, AdaptiveTargetGating


class TestAdaptiveGating:
    @pytest.fixture
    def gating(self):
        return AdaptiveTargetGating(d_target=128, d_time=64)

    def test_output_shape_2_targets(self, gating):
        targets = [torch.randn(2, 10, 128), torch.randn(2, 15, 128)]
        t = torch.tensor([100.0, 200.0])
        alpha = gating(targets, t)
        assert alpha.shape == (2, 2)  # B, K
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_output_shape_3_targets(self, gating):
        targets = [torch.randn(4, 8, 128), torch.randn(4, 12, 128), torch.randn(4, 6, 128)]
        t = torch.randn(4, 64)
        alpha = gating(targets, t)
        assert alpha.shape == (4, 3)
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(4), atol=1e-6)

    def test_gradient_flow(self, gating):
        targets = [torch.randn(2, 10, 128, requires_grad=True)]
        t = torch.randn(2, 64, requires_grad=True)
        alpha = gating(targets, t)
        loss = alpha.sum()
        loss.backward()
        assert targets[0].grad is not None
        assert t.grad is not None


class TestCrossAttention:
    @pytest.fixture
    def xattn(self):
        return MultiTargetCrossAttention(d_model=256, d_target=128, d_head=64, n_heads=4)

    def test_output_shape(self, xattn):
        B, L = 2, 10
        pep = torch.randn(B, L, 256)
        pep_mask = torch.ones(B, L, dtype=torch.bool)
        targets = [torch.randn(B, 12, 128), torch.randn(B, 15, 128)]
        t_mask = [torch.ones(B, 12, dtype=torch.bool), torch.ones(B, 15, dtype=torch.bool)]
        t_emb = torch.randn(B, 64)
        out = xattn(pep, pep_mask, targets, t_mask, t_emb)
        assert out.shape == (B, L, 256)

    def test_single_target(self, xattn):
        B, L = 2, 8
        pep = torch.randn(B, L, 256)
        pep_mask = torch.ones(B, L, dtype=torch.bool)
        targets = [torch.randn(B, 20, 128)]
        t_mask = [torch.ones(B, 20, dtype=torch.bool)]
        t_emb = torch.randn(B, 64)
        out = xattn(pep, pep_mask, targets, t_mask, t_emb)
        assert out.shape == (B, L, 256)

    def test_masked_targets(self, xattn):
        B, L = 1, 6
        pep = torch.randn(B, L, 256)
        pep_mask = torch.ones(B, L, dtype=torch.bool)
        targets = [torch.randn(B, 8, 128)]
        # mask out last 3 residues
        t_mask = [torch.tensor([[True, True, True, True, True, False, False, False]])]
        t_emb = torch.randn(B, 64)
        out = xattn(pep, pep_mask, targets, t_mask, t_emb)
        assert out.shape == (B, L, 256)

    def test_with_cyclo_emb(self, xattn):
        B, L = 2, 10
        pep = torch.randn(B, L, 256)
        pep_mask = torch.ones(B, L, dtype=torch.bool)
        targets = [torch.randn(B, 12, 128)]
        t_mask = [torch.ones(B, 12, dtype=torch.bool)]
        t_emb = torch.randn(B, 64)
        cyclo_emb = torch.randn(B, 64)
        out = xattn(pep, pep_mask, targets, t_mask, t_emb, cyclo_emb=cyclo_emb)
        assert out.shape == (B, L, 256)

    def test_gradient_flow(self, xattn):
        B, L = 2, 8
        pep = torch.randn(B, L, 256, requires_grad=True)
        pep_mask = torch.ones(B, L, dtype=torch.bool)
        targets = [torch.randn(B, 10, 128, requires_grad=True)]
        t_mask = [torch.ones(B, 10, dtype=torch.bool)]
        t_emb = torch.randn(B, 64, requires_grad=True)
        out = xattn(pep, pep_mask, targets, t_mask, t_emb)
        loss = out.sum()
        loss.backward()
        assert pep.grad is not None
        assert targets[0].grad is not None
        assert t_emb.grad is not None

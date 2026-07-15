import pytest
import torch
from cyclicdiffusion_mt.model.target_encoder import TargetEncoder


class TestTargetEncoder:
    @pytest.fixture
    def encoder(self):
        return TargetEncoder(d_target=128, n_blocks=3, d_head=32, n_heads=4)

    def test_output_shape_single_target(self, encoder):
        B, K, N = 2, 1, 10
        coords = [torch.randn(B, N, 14, 3)]
        mask = [torch.ones(B, N, dtype=torch.bool)]
        out = encoder(coords, mask)
        assert len(out) == 1
        assert out[0].shape == (B, N, 128)

    def test_output_shape_multi_target(self, encoder):
        B, N1, N2 = 2, 10, 15
        coords = [torch.randn(B, N1, 14, 3), torch.randn(B, N2, 14, 3)]
        mask = [torch.ones(B, N1, dtype=torch.bool), torch.ones(B, N2, dtype=torch.bool)]
        out = encoder(coords, mask)
        assert len(out) == 2
        assert out[0].shape == (B, N1, 128)
        assert out[1].shape == (B, N2, 128)

    def test_shared_weights(self, encoder):
        """Two forward passes with same weights should give same result."""
        encoder.eval()
        B, N = 1, 5
        coords = [torch.randn(B, N, 14, 3)]
        mask = [torch.ones(B, N, dtype=torch.bool)]
        o1 = encoder(coords, mask)
        o2 = encoder(coords, mask)
        assert torch.allclose(o1[0], o2[0], atol=1e-6)

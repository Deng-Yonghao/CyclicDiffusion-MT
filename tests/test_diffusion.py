# tests/test_diffusion.py
import pytest, torch
from cyclicdiffusion_mt.model.diffusion import (
    WrappedNormalDiffusion, MaskedDiscreteDiffusion, cosine_schedule,
)

class TestCosineSchedule:
    def test_start_is_1(self):
        sched = cosine_schedule(500)
        assert torch.allclose(sched[0], torch.tensor(1.0), atol=0.05)
    def test_end_is_0(self):
        sched = cosine_schedule(500)
        assert sched[-1] < 0.01
    def test_monotonic_decreasing(self):
        sched = cosine_schedule(500)
        assert (sched[1:] <= sched[:-1]).all()

class TestWrappedNormalDiffusion:
    @pytest.fixture
    def diff(self): return WrappedNormalDiffusion(T=500)
    def test_q_sample_shape(self, diff):
        tau_0 = torch.randn(2,10,7)
        t = torch.randint(0,500,(2,))
        tau_t, noise = diff.q_sample(tau_0, t)
        assert tau_t.shape == tau_0.shape
        assert noise.shape == tau_0.shape
    def test_t0_is_clean(self, diff):
        tau_0 = torch.randn(2,5,7)
        tau_0 = torch.atan2(torch.sin(tau_0), torch.cos(tau_0))  # wrap to [-pi, pi)
        tau_t, _ = diff.q_sample(tau_0, torch.zeros(2,dtype=torch.long))
        assert torch.allclose(tau_t, tau_0, atol=1e-5)
    def test_loss_shape(self, diff):
        tau_0 = torch.randn(2,10,7); tau_pred = torch.randn(2,10,7)
        loss = diff.loss(tau_0, tau_pred)
        assert loss.ndim == 0  # scalar
    def test_wrapped_loss_handles_boundary(self, diff):
        """Loss between -pi+eps and pi-eps should be small."""
        tau_0 = torch.tensor([[[3.1,0,0,0,0,0,0]]])  # near pi
        tau_pred = torch.tensor([[[-3.1,0,0,0,0,0,0]]])  # near -pi but same angle
        loss = diff.loss(tau_0, tau_pred)
        assert loss.item() < 0.1  # should be close since angles are equivalent

class TestMaskedDiscreteDiffusion:
    @pytest.fixture
    def diff(self): return MaskedDiscreteDiffusion(T=500, num_classes=26, mask_idx=25)
    def test_q_sample_shape(self, diff):
        a_0 = torch.randint(0,25,(2,10)); t = torch.randint(0,500,(2,))
        a_t = diff.q_sample(a_0, t)
        assert a_t.shape == a_0.shape
    def test_t0_is_clean(self, diff):
        a_0 = torch.randint(0,25,(2,5))
        a_t = diff.q_sample(a_0, torch.zeros(2,dtype=torch.long))
        assert (a_t == a_0).all()
    def test_loss_shape(self, diff):
        a_0 = torch.randint(0,25,(2,10))
        logits = torch.randn(2,10,26)
        loss = diff.loss(a_0, logits)
        assert loss.ndim == 0

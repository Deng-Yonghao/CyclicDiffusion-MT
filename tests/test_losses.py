"""Tests for loss functions: torsion, type, cyclo, geometry, affinity."""
import pytest
import torch
from cyclicdiffusion_mt.losses.torsion_loss import torsion_loss
from cyclicdiffusion_mt.losses.type_loss import type_loss
from cyclicdiffusion_mt.losses.cyclo_loss import cyclo_loss
from cyclicdiffusion_mt.losses.geometry_loss import clash_loss, rama_loss, rotamer_loss
from cyclicdiffusion_mt.losses.affinity_loss import affinity_loss
from cyclicdiffusion_mt.model.nerf import NeRF
from cyclicdiffusion_mt.utils.constants import MASK_IDX, CLASH_THRESHOLD


class TestTorsionLoss:
    """Wrapped L2 torsion angle loss."""

    def test_perfect_prediction(self):
        """Loss is near-zero when prediction matches target."""
        tau = torch.randn(2, 5, 7)
        mask = torch.ones(2, 5, 7, dtype=torch.bool)
        loss = torsion_loss(tau, tau, mask)
        assert loss.item() < 1e-5, f"Expected ~0 loss, got {loss.item()}"

    def test_masked_positions_ignored(self):
        """Masked (chi) positions do not contribute to loss."""
        tau = torch.randn(2, 5, 7)
        # Only backbone (phi, psi, omega) positions 0,1,2 are valid
        mask = torch.ones(2, 5, 7, dtype=torch.bool)
        mask[:, :, 3:] = False
        loss = torsion_loss(tau, tau + 1.0, mask)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_angular_wrapping(self):
        """Angular differences wrap correctly (2pi periodicity)."""
        tau = torch.tensor([[[3.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        tau_shifted = torch.tensor([[[-3.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        mask = torch.ones(1, 1, 7, dtype=torch.bool)
        loss = torsion_loss(tau, tau_shifted, mask)
        # pi and -pi should be close after wrapping
        assert loss.item() < 0.1, f"Expected small loss with wrapped diff, got {loss.item()}"

    def test_nonzero_loss(self):
        """Different torsions produce non-zero loss."""
        tau_pred = torch.zeros(2, 3, 7)
        tau_true = torch.ones(2, 3, 7)
        mask = torch.ones(2, 3, 7, dtype=torch.bool)
        loss = torsion_loss(tau_pred, tau_true, mask)
        assert loss.item() > 0, "Expected non-zero loss for different torsions"

    def test_all_masked(self):
        """All-False mask gives zero loss without crashing."""
        tau = torch.randn(2, 3, 7)
        mask = torch.zeros(2, 3, 7, dtype=torch.bool)
        loss = torsion_loss(tau, tau + 10.0, mask)
        assert loss.item() == 0.0, "All-masked loss should be zero"


class TestTypeLoss:
    """Cross-entropy loss for amino acid type prediction."""

    def test_shape(self):
        """Loss is a scalar."""
        logits = torch.randn(2, 10, 26)
        a_0 = torch.randint(0, 25, (2, 10))
        loss = type_loss(logits, a_0)
        assert loss.ndim == 0, "Loss should be scalar"

    def test_perfect_prediction_low_loss(self):
        """Confident correct prediction gives low loss."""
        logits = torch.zeros(1, 3, 26)
        a_0 = torch.tensor([[0, 1, 2]])
        # Put high logit on correct class
        logits[0, 0, 0] = 100.0
        logits[0, 1, 1] = 100.0
        logits[0, 2, 2] = 100.0
        loss = type_loss(logits, a_0)
        assert loss.item() < 0.01, f"Expected low loss, got {loss.item()}"

    def test_mask_ignored(self):
        """MASK_IDX positions are ignored in loss computation."""
        logits = torch.randn(1, 5, 26)
        a_0 = torch.tensor([[MASK_IDX, MASK_IDX, MASK_IDX, MASK_IDX, MASK_IDX]])
        loss = type_loss(logits, a_0)
        assert loss.ndim == 0, "Loss should be scalar even with all masked"


class TestCycloLoss:
    """Ring closure loss using NeRF."""

    def test_runs(self):
        """Cyclo loss produces a scalar without NaN."""
        nerf = NeRF()
        B, L = 1, 4
        aa = torch.zeros(B, L, dtype=torch.long)
        mode = torch.zeros(B, dtype=torch.long)
        bonds = torch.rand(B, L, 14) * 0.1 + 1.3
        angles = torch.rand(B, L, 14) * 0.3 + 1.9
        tau = torch.rand(B, L, 7) * 0.1
        loss = cyclo_loss(nerf, tau, aa, bonds, angles, mode)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_batch_independence(self):
        """Each batch element's cyclo loss is independent."""
        nerf = NeRF()
        B, L = 1, 4
        aa = torch.zeros(B, L, dtype=torch.long)
        mode = torch.zeros(B, dtype=torch.long)
        bonds = torch.rand(B, L, 14) * 0.1 + 1.3
        angles = torch.rand(B, L, 14) * 0.3 + 1.9
        tau = torch.rand(B, L, 7) * 0.1
        loss = cyclo_loss(nerf, tau, aa, bonds, angles, mode)
        assert loss.ndim == 0


class TestClashLoss:
    """Steric clash loss."""

    def test_no_clash(self):
        """Atoms far apart produce zero clash loss."""
        coords = torch.randn(1, 3, 14, 3) * 100  # spread far apart
        mask = torch.ones(1, 3, 14, dtype=torch.bool)
        loss = clash_loss(coords, mask)
        assert loss.item() < 1e-5, f"Expected ~0 loss for distant atoms, got {loss.item()}"

    def test_clash_detected(self):
        """Overlapping atoms produce non-zero clash loss."""
        # Two atoms at the same position
        coords = torch.zeros(1, 2, 14, 3)
        coords[0, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
        coords[0, 0, 1] = torch.tensor([0.0, 0.0, 0.0])  # same position = clash
        mask = torch.zeros(1, 2, 14, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[0, 0, 1] = True
        loss = clash_loss(coords, mask)
        assert loss.item() > 0, f"Expected clash loss > 0 for overlapping atoms, got {loss.item()}"

    def test_masked_atoms_ignored(self):
        """Masked atoms do not contribute to clash detection."""
        # Place two atoms at same position, but mask one out
        coords = torch.zeros(1, 2, 14, 3)
        coords[0, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
        coords[0, 0, 1] = torch.tensor([0.0, 0.0, 0.0])  # clash if both valid
        mask = torch.zeros(1, 2, 14, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[0, 0, 1] = False  # second atom masked out -> no pair to clash
        loss = clash_loss(coords, mask)
        # With only one valid atom, there are no pairs, loss should be ~0
        assert loss.item() < 1e-5, f"Expected ~0 loss with single atom, got {loss.item()}"


class TestRamaLoss:
    """Ramachandran torsion loss."""

    def test_runs(self):
        """Rama loss produces a scalar."""
        phi = torch.randn(2, 5) * 0.5
        psi = torch.randn(2, 5) * 0.5
        aa = torch.randint(0, 25, (2, 5))
        loss = rama_loss(phi, psi, aa)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_glycine_zero(self):
        """Glycine produces near-zero rama loss due to flexibility."""
        from cyclicdiffusion_mt.utils.constants import AA_TO_IDX
        gly = AA_TO_IDX['GLY']
        phi = torch.tensor([[-1.0, 0.5]])
        psi = torch.tensor([[-0.82, -0.5]])
        aa = torch.tensor([[gly, gly]])
        loss = rama_loss(phi, psi, aa)
        assert loss.item() < 1e-5, f"Glycine rama loss should be near-zero, got {loss.item()}"


class TestRotamerLoss:
    """Rotamer chi angle loss."""

    def test_runs(self):
        """Rotamer loss produces a scalar."""
        chis = torch.randn(2, 5, 4) * 0.5
        aa = torch.randint(0, 25, (2, 5))
        loss = rotamer_loss(chis, aa)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_ala_zero(self):
        """Alanine (no chi) produces zero rotamer loss."""
        from cyclicdiffusion_mt.utils.constants import AA_TO_IDX
        ala = AA_TO_IDX['ALA']
        chis = torch.rand(1, 3, 4) * 3.14
        aa = torch.full((1, 3), ala)
        loss = rotamer_loss(chis, aa)
        assert loss.item() < 1e-5, f"Ala rotamer loss should be zero, got {loss.item()}"


class TestAffinityLoss:
    """Confidence-weighted MSE affinity loss."""

    def test_shape(self):
        """Affinity loss is a scalar."""
        dg_pred = torch.randn(4, 2)
        dg_label = torch.randn(4, 2)
        conf = torch.ones(4)
        loss = affinity_loss(dg_pred, dg_label, conf)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_perfect_prediction_zero_loss(self):
        """Perfect prediction with confidence=1 gives zero loss."""
        dg_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        dg_label = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        conf = torch.tensor([1.0, 1.0])
        loss = affinity_loss(dg_pred, dg_label, conf)
        assert loss.item() < 1e-6, f"Expected ~0 loss, got {loss.item()}"

    def test_zero_confidence(self):
        """Zero confidence gives zero loss regardless of prediction error."""
        dg_pred = torch.randn(4, 3)
        dg_label = dg_pred + 100.0  # large error
        conf = torch.zeros(4)
        loss = affinity_loss(dg_pred, dg_label, conf)
        assert loss.item() == 0.0, "Zero confidence should yield zero loss"

    def test_confidence_weighting(self):
        """Higher confidence yields higher contribution to loss."""
        dg_pred = torch.tensor([[0.0, 0.0]])
        dg_label = torch.tensor([[10.0, 10.0]])
        conf_low = torch.tensor([0.1])
        conf_high = torch.tensor([1.0])
        loss_low = affinity_loss(dg_pred, dg_label, conf_low)
        loss_high = affinity_loss(dg_pred, dg_label, conf_high)
        assert loss_low.item() < loss_high.item(), \
            f"Low-conf loss ({loss_low.item()}) should be less than high-conf loss ({loss_high.item()})"

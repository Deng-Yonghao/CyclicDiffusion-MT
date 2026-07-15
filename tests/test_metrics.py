import torch, pytest
from cyclicdiffusion_mt.eval.metrics import (
    ring_closure_precision,
    ramachandran_outlier_rate,
    steric_clash_count,
    internal_diversity,
)
from cyclicdiffusion_mt.utils.constants import AA_TO_IDX


class TestRingClosure:
    def test_perfect_closure(self):
        # N at (0,0,0), C at (d,0,0) where d = ideal bond
        coords = torch.zeros(2, 8, 14, 3)
        coords[:, 0, 0, 0] = 0.0
        coords[:, -1, 2, 0] = 1.329
        err = ring_closure_precision(coords)
        assert err < 0.1

    def test_broken_closure(self):
        coords = torch.zeros(2, 8, 14, 3)
        coords[:, 0, 0, 0] = 0.0
        coords[:, -1, 2, 0] = 5.0  # 5A gap
        err = ring_closure_precision(coords)
        assert err > 2.0


class TestRamachandran:
    def test_alpha_helix_low_outlier(self):
        # Alpha-helical phi=-57, psi=-47 are favored
        phi = torch.full((2, 10), -0.995)  # -57 deg
        psi = torch.full((2, 10), -0.820)  # -47 deg
        aa = torch.randint(0, 25, (2, 10))
        rate = ramachandran_outlier_rate(phi, psi, aa)
        assert rate < 0.5  # alpha-helix should be mostly allowed

    def test_bad_angles_high_outlier(self):
        # phi=0, psi=0 is disallowed for non-gly
        phi = torch.zeros(2, 10)
        psi = torch.zeros(2, 10)
        aa = torch.full((2, 10), AA_TO_IDX['ALA'])
        rate = ramachandran_outlier_rate(phi, psi, aa)
        assert rate > 0.5


class TestStericClash:
    def test_no_clash(self):
        coords = torch.zeros(1, 5, 14, 3)
        # Place N atom of each residue far apart
        for i in range(5):
            coords[0, i, 0, 0] = i * 5.0
        atom_mask = torch.zeros(1, 5, 14, dtype=torch.bool)
        atom_mask[:, :, 0] = True  # only N atoms are present
        count = steric_clash_count(coords, atom_mask)
        assert count == 0

    def test_clash_detected(self):
        coords = torch.zeros(1, 2, 14, 3)
        # Two atoms at nearly same position
        coords[0, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
        coords[0, 1, 0] = torch.tensor([0.5, 0.0, 0.0])  # 0.5A apart
        atom_mask = torch.ones(1, 2, 14, dtype=torch.bool)
        # Only those two atoms
        atom_mask[0, 0, 1:] = False
        atom_mask[0, 1, 1:] = False
        count = steric_clash_count(coords, atom_mask)
        assert count > 0


class TestDiversity:
    def test_identical_diversity(self):
        tau = torch.randn(1, 8, 7)
        div = internal_diversity([tau, tau.clone()])
        assert div == 0.0

    def test_different_diversity(self):
        tau1 = torch.randn(1, 8, 7)
        tau2 = torch.randn(1, 8, 7) + 5.0  # clearly different
        div = internal_diversity([tau1, tau2])
        assert div > 0.0

    def test_returns_float(self):
        samples = [torch.randn(1, 8, 7) for _ in range(5)]
        div = internal_diversity(samples)
        assert isinstance(div, float)
        assert div >= 0.0

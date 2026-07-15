# tests/test_geometry.py
import torch, pytest
from cyclicdiffusion_mt.data.geometry import build_ideal_bonds, build_ideal_angles, build_ideal_geometry
from cyclicdiffusion_mt.utils.constants import AA_TO_IDX, MAX_ATOMS_PER_RES, IDEAL_BOND_LENGTHS, IDEAL_BOND_ANGLES


class TestIdealBonds:
    def test_shape(self):
        aa = torch.tensor([[AA_TO_IDX['ALA'], AA_TO_IDX['GLY']]])
        bonds = build_ideal_bonds(aa)
        assert bonds.shape == (1, 2, MAX_ATOMS_PER_RES)

    def test_ala_bonds_nonzero(self):
        aa = torch.tensor([[AA_TO_IDX['ALA']]])
        bonds = build_ideal_bonds(aa)
        # ALA has 5 atoms, bonded: atoms 1-4 should have values
        assert bonds[0, 0, 1:5].sum() > 0
        # Padding atoms should be zero
        assert bonds[0, 0, 5:].sum() == 0

    def test_gly_only_backbone(self):
        aa = torch.tensor([[AA_TO_IDX['GLY']]])
        bonds = build_ideal_bonds(aa)
        # GLY has 4 atoms, so atoms 4+ should be zero
        assert bonds[0, 0, 4:].sum() == 0

    def test_batch_shape(self):
        aa = torch.randint(0, 25, (4, 10))
        bonds = build_ideal_bonds(aa)
        assert bonds.shape == (4, 10, MAX_ATOMS_PER_RES)


class TestIdealAngles:
    def test_shape(self):
        aa = torch.tensor([[AA_TO_IDX['ALA'], AA_TO_IDX['GLY']]])
        angles = build_ideal_angles(aa)
        assert angles.shape == (1, 2, MAX_ATOMS_PER_RES)

    def test_angles_in_radians(self):
        aa = torch.randint(0, 25, (2, 3))
        angles = build_ideal_angles(aa)
        # Bond angles should be in radian range (approx 1.9-2.1 rad for ~110-120 deg)
        valid = angles > 0
        if valid.any():
            assert angles[valid].max() < 3.15  # < pi


class TestBuildIdealGeometry:
    def test_returns_pair(self):
        aa = torch.randint(0, 25, (2, 5))
        bonds, angles = build_ideal_geometry(aa)
        assert bonds.shape == (2, 5, MAX_ATOMS_PER_RES)
        assert angles.shape == (2, 5, MAX_ATOMS_PER_RES)

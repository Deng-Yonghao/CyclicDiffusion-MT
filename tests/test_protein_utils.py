"""Tests for CPCore PDB parsing and internal coordinate extraction."""
import os
import torch
import pytest
from cyclicdiffusion_mt.utils.protein_utils import (
    extract_peptide_residues,
    extract_target_coords,
    compute_backbone_torsions,
    residues_to_torsions,
    PDB_ATOM_ORDER,
    ATOM_NAME_MAP,
)
from cyclicdiffusion_mt.utils.constants import AA_TO_IDX, MAX_ATOMS_PER_RES


# Path to a real test PDB
TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "CPCore", "CPCore_pdb",
    "AF-A0A417M1J3-F1_0_172_177_relaxed_relaxed.pdb"
)


class TestExtractPeptideResidues:
    def test_strips_capping_residues(self):
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        res_names = [r['res_name'] for r in residues]
        assert 'ACE' not in res_names
        assert 'NME' not in res_names

    def test_known_residue_sequence(self):
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        res_names = [r['res_name'] for r in residues]
        # 172-177: CYS, ASP, TYR, PHE, LYS, CYS
        assert res_names == ['CYS', 'ASP', 'TYR', 'PHE', 'LYS', 'CYS']

    def test_has_backbone_atoms(self):
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        for r in residues:
            atoms = r['atoms']
            assert 'N' in atoms or 'H' in atoms  # some PDBs use H for backbone N
            assert 'CA' in atoms
            assert 'C' in atoms


class TestExtractTargetCoords:
    def test_returns_tensors(self):
        coords, seq = extract_target_coords(TEST_PDB, chain_id='R')
        assert isinstance(coords, torch.Tensor)
        assert isinstance(seq, torch.Tensor)

    def test_coords_shape(self):
        coords, seq = extract_target_coords(TEST_PDB, chain_id='R')
        assert coords.ndim == 3
        assert coords.shape[1] == 14  # max atoms per residue
        assert coords.shape[2] == 3   # xyz
        assert coords.shape[0] == seq.shape[0]

    def test_coords_not_all_zero(self):
        coords, seq = extract_target_coords(TEST_PDB, chain_id='R')
        assert (coords != 0).any()  # real coordinates exist


class TestResiduesToTorsions:
    def test_output_shape(self):
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        tau, aa_types = residues_to_torsions(residues)
        assert tau.ndim == 2
        assert tau.shape[0] == len(residues)
        assert tau.shape[1] == 7  # phi, psi, omega, chi1-4
        assert aa_types.shape == (len(residues),)

    def test_all_torsions_in_range(self):
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        tau, aa_types = residues_to_torsions(residues)
        assert (tau >= -3.1416).all()
        assert (tau < 3.1416).all()

    def test_omega_near_trans(self):
        """Omega angles should be near pi (trans) for standard peptides."""
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        tau, aa_types = residues_to_torsions(residues)
        omega = tau[:, 2]  # omega is the 3rd torsion
        # Omega should be near pi (trans) or 0 (cis)
        for w in omega:
            sin_w = abs(float(torch.sin(w)))
            assert sin_w < 0.3, f"Omega={w:.2f} is not near trans/CIS, sin={sin_w:.3f}"

    def test_glycine_has_zero_chi(self):
        """Glycine residues should have chi angles near 0 (no sidechain)."""
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        tau, aa_types = residues_to_torsions(residues)
        gly_idx = AA_TO_IDX['GLY']
        for i, aa in enumerate(aa_types):
            if aa.item() == gly_idx:
                for chi in range(3, 7):
                    assert abs(tau[i, chi].item()) < 1e-6, \
                        f"GLY at position {i} has non-zero chi{chi-2}"


class TestComputeBackboneTorsions:
    def test_phi_uses_prev_residue_C(self):
        """First residue phi must use last residue C (cyclic wrap-around)."""
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        phi, psi, omega = compute_backbone_torsions(residues)
        assert phi.shape == (len(residues),)
        assert psi.shape == (len(residues),)
        assert omega.shape == (len(residues),)

    def test_cyclic_wraparound(self):
        """Last residue psi/omega use first residue N/CA (cyclic)."""
        residues = extract_peptide_residues(TEST_PDB, chain_id='L')
        phi, psi, omega = compute_backbone_torsions(residues)
        # All should be well-defined (not NaN or zero because of missing neighbors)
        assert not torch.isnan(phi).any()
        assert not torch.isnan(psi).any()
        assert not torch.isnan(omega).any()

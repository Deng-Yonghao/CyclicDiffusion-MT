"""Tests for CPCore PDB parsing and internal coordinate extraction."""
import os
import torch
import pytest
from cyclicdiffusion_mt.utils.protein_utils import (
    extract_peptide_residues,
    extract_target_coords,
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

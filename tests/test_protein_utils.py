"""Tests for CPCore PDB parsing and internal coordinate extraction."""
import os
import torch
import pytest
from cyclicdiffusion_mt.utils.protein_utils import (
    parse_pdb_cyclic,
    extract_peptide_residues,
    extract_target_coords,
    PDB_ATOM_ORDER,
    ATOM_NAME_MAP,
)
from cyclicdiffusion_mt.utils.constants import AA_TO_IDX


# Path to a real test PDB
TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "CPCore", "CPCore_pdb",
    "AF-A0A417M1J3-F1_0_172_177_relaxed_relaxed.pdb"
)


class TestParsePDBCyclic:
    def test_parse_returns_dict(self):
        result = parse_pdb_cyclic(TEST_PDB)
        assert isinstance(result, dict)
        assert "peptide_torsions" in result
        assert "peptide_aa_types" in result
        assert "target_coords" in result
        assert "target_sequences" in result

    def test_peptide_torsions_shape_and_range(self):
        result = parse_pdb_cyclic(TEST_PDB)
        tau = result["peptide_torsions"]
        assert tau.shape[1] == 7  # 7 torsion angles
        assert tau.shape[0] <= 20  # max residues
        # Torsions should be in [-pi, pi)
        assert (tau >= -3.1416).all()
        assert (tau < 3.1416).all()

    def test_peptide_aa_types_valid(self):
        result = parse_pdb_cyclic(TEST_PDB)
        aa = result["peptide_aa_types"]
        assert aa.ndim == 1
        # All indices should be 0-24 (not 25=MASK)
        assert (aa >= 0).all() and (aa < 25).all()

    def test_target_coords_shape(self):
        result = parse_pdb_cyclic(TEST_PDB)
        coords = result["target_coords"][0]  # list of [(N_k, 14, 3)]
        assert coords.ndim == 3
        assert coords.shape[1] == 14
        assert coords.shape[2] == 3

    def test_target_sequences_valid(self):
        result = parse_pdb_cyclic(TEST_PDB)
        seq = result["target_sequences"][0]
        assert seq.ndim == 1
        assert (seq >= 0).all() and (seq < 25).all()

    def test_known_peptide_length(self):
        # This PDB has residues CYS(172) through CYS(177) = 6 residues
        # Plus ACE and NME caps which should be stripped
        result = parse_pdb_cyclic(TEST_PDB)
        # AF-A0A417M1J3-F1_0_172_177 = 6 residues (CYS, ASP, TYR, PHE, LYS, CYS)
        assert result["peptide_torsions"].shape[0] == 6

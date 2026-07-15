# tests/test_constants.py
import pytest
from cyclicdiffusion_mt.utils.constants import (
    AA_TO_IDX, IDX_TO_AA, NUM_AA_TYPES, MASK_IDX,
    AA_CHI_COUNTS, AA_ATOM_NAMES, AA_ATOM_COUNT,
    BACKBONE_ATOMS, CYCLO_MODES, MAX_RESIDUES,
    MAX_ATOMS_PER_RES, MAX_CHI_PER_RES, NUM_TORSIONS,
    IDEAL_BOND_LENGTHS, IDEAL_BOND_ANGLES,
)

class TestAAVocabulary:
    def test_num_aa_types(self):
        assert NUM_AA_TYPES == 25
    def test_mask_idx_is_25(self):
        assert MASK_IDX == 25
    def test_standard_20_present(self):
        std = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
               'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
        for aa in std:
            assert aa in AA_TO_IDX
    def test_noncanonical_5_present(self):
        for aa in ['ORN','DAL','BAL','NMA','DPH']:
            assert aa in AA_TO_IDX
    def test_idx_roundtrip(self):
        for aa, idx in AA_TO_IDX.items():
            assert IDX_TO_AA[idx] == aa
    def test_chi_counts_length(self):
        assert len(AA_CHI_COUNTS) == NUM_AA_TYPES
    def test_ala_gly_no_chi(self):
        assert AA_CHI_COUNTS[AA_TO_IDX['ALA']] == 0
        assert AA_CHI_COUNTS[AA_TO_IDX['GLY']] == 0
    def test_arg_lys_4_chi(self):
        assert AA_CHI_COUNTS[AA_TO_IDX['ARG']] == 4
        assert AA_CHI_COUNTS[AA_TO_IDX['LYS']] == 4
    def test_max_chi_is_4(self):
        assert max(AA_CHI_COUNTS) <= MAX_CHI_PER_RES
    def test_atom_names_count_match(self):
        for aa in AA_ATOM_NAMES:
            assert len(AA_ATOM_NAMES[aa]) == AA_ATOM_COUNT[aa]
    def test_backbone_order(self):
        assert BACKBONE_ATOMS == ['N', 'CA', 'C', 'O']
    def test_ala_5_atoms(self):
        assert AA_ATOM_COUNT['ALA'] == 5

class TestCycloModes:
    def test_five_modes(self):
        assert len(CYCLO_MODES) == 5
    def test_head_to_tail(self):
        assert 'head_to_tail' in CYCLO_MODES

class TestConstants:
    def test_dimensional_constants(self):
        assert MAX_RESIDUES == 20
        assert MAX_ATOMS_PER_RES == 14
        assert MAX_CHI_PER_RES == 4
        assert NUM_TORSIONS == 7
    def test_ideal_bonds(self):
        assert 'N-CA' in IDEAL_BOND_LENGTHS
        assert 'CA-C' in IDEAL_BOND_LENGTHS
        assert 'C-N' in IDEAL_BOND_LENGTHS
    def test_ideal_angles(self):
        assert 'N-CA-C' in IDEAL_BOND_ANGLES
        assert 'CA-C-N' in IDEAL_BOND_ANGLES

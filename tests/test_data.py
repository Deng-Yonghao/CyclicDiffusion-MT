"""Tests for data pipeline: transforms and dataset."""
import pytest
import torch
from cyclicdiffusion_mt.data.transforms import compute_chi_mask, compute_atom_mask, cartesian_to_internal
from cyclicdiffusion_mt.data.dataset import MultiTargetDataset, PeptideDataCollate
from cyclicdiffusion_mt.utils.constants import MAX_RESIDUES


class TestChiMask:
    def test_shape(self):
        aa = torch.tensor([[0, 7, 1]])  # ALA, GLY, ARG
        mask = compute_chi_mask(aa)
        assert mask.shape == (1, 3, 7)

    def test_ala_no_chi(self):
        aa = torch.tensor([[0]])  # ALA
        mask = compute_chi_mask(aa)
        assert mask[0, 0, 0] and mask[0, 0, 1] and mask[0, 0, 2]  # phi,psi,omega always True
        assert not mask[0, 0, 3]  # chi1 False for ALA

    def test_arg_all_chi(self):
        aa = torch.tensor([[1]])  # ARG
        mask = compute_chi_mask(aa)
        assert mask[0, 0, 3] and mask[0, 0, 4] and mask[0, 0, 5] and mask[0, 0, 6]  # all 4 chi


class TestAtomMask:
    def test_shape(self):
        aa = torch.tensor([[0, 7]])  # ALA, GLY
        mask = compute_atom_mask(aa)
        assert mask.shape == (1, 2, 14)

    def test_gly_only_4_atoms(self):
        aa = torch.tensor([[7]])  # GLY
        mask = compute_atom_mask(aa)
        assert mask[0, 0, 0] and mask[0, 0, 1] and mask[0, 0, 2] and mask[0, 0, 3]
        assert not mask[0, 0, 4]  # no CB for GLY


class TestCartesianToInternal:
    def test_simple_ala(self):
        # N, CA, C, O, CB coords for ALA
        coords = torch.tensor([
            [0.0, 0.0, 0.0],       # N
            [1.458, 0.0, 0.0],     # CA
            [2.458, 0.0, 0.0],     # C (simplified)
            [2.458, 1.231, 0.0],   # O
            [1.458, 1.0, 0.0],     # CB
        ])
        atom_names = ['N', 'CA', 'C', 'O', 'CB']
        bonds, angles, torsions = cartesian_to_internal(coords, atom_names, 'ALA')
        assert bonds is not None
        assert angles is not None


class TestMultiTargetDataset:
    def test_len(self):
        manifest = [
            {
                'peptide_torsions': [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0] for _ in range(5)],
                'peptide_aa_types': [0, 1, 2, 3, 4],
                'target_coords': [],
                'target_sequences': [],
                'cyclo_mode': 0,
                'dG_rosetta': -10.5,
                'confidence': 0.9,
            },
            {
                'peptide_torsions': [[0.5, 0.6, 0.7, 0.0, 0.0, 0.0, 0.0] for _ in range(3)],
                'peptide_aa_types': [7, 8, 9],
                'target_coords': [],
                'target_sequences': [],
                'cyclo_mode': 2,
                'dG_rosetta': -5.0,
                'confidence': 0.8,
            },
        ]
        ds = MultiTargetDataset(manifest)
        assert len(ds) == 2

    def test_getitem_keys(self):
        manifest = [
            {
                'peptide_torsions': [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0] for _ in range(4)],
                'peptide_aa_types': [0, 1, 0, 1],
                'target_coords': [],
                'target_sequences': [],
                'cyclo_mode': 0,
                'dG_rosetta': -10.5,
                'confidence': 0.9,
            },
        ]
        ds = MultiTargetDataset(manifest)
        item = ds[0]
        assert 'peptide_torsions' in item
        assert 'peptide_aa_types' in item
        assert 'target_coords' in item
        assert 'target_sequences' in item
        assert 'cyclo_mode' in item
        assert 'dG_rosetta' in item
        assert 'confidence' in item

    def test_getitem_shapes(self):
        manifest = [
            {
                'peptide_torsions': [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0] for _ in range(5)],
                'peptide_aa_types': [0, 1, 2, 3, 4],
                'target_coords': [],
                'target_sequences': [],
                'cyclo_mode': 0,
                'dG_rosetta': -10.5,
                'confidence': 0.9,
            },
        ]
        ds = MultiTargetDataset(manifest)
        item = ds[0]
        assert item['peptide_torsions'].shape == (5, 7)
        assert item['peptide_aa_types'].shape == (5,)
        assert item['cyclo_mode'] == 0

    def test_defaults(self):
        manifest = [
            {
                'peptide_torsions': [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]],
                'peptide_aa_types': [0],
            },
        ]
        ds = MultiTargetDataset(manifest)
        item = ds[0]
        assert item['dG_rosetta'].item() == 0.0
        assert item['confidence'].item() == 1.0
        assert item['cyclo_mode'] == 0
        assert item['target_coords'] == []
        assert item['target_sequences'] == []


class TestPeptideDataCollate:
    def _make_item(self, L, aa_types, torsions, n_targets=0):
        return {
            'peptide_torsions': torch.tensor(torsions, dtype=torch.float32),
            'peptide_aa_types': torch.tensor(aa_types, dtype=torch.long),
            'target_coords': [torch.zeros(1, 3) for _ in range(n_targets)],
            'target_sequences': [torch.zeros(1, dtype=torch.long) for _ in range(n_targets)],
            'cyclo_mode': 0,
            'dG_rosetta': torch.tensor(-10.0),
            'confidence': torch.tensor(0.9),
        }

    def test_collate_shape(self):
        collate = PeptideDataCollate()
        batch = [
            self._make_item(3, [0, 1, 2], [[0.1]*7, [0.2]*7, [0.3]*7]),
            self._make_item(5, [3, 4, 5, 6, 7], [[0.4]*7]*5),
        ]
        result = collate(batch)
        assert result['peptide_torsions'].shape == (2, 5, 7)
        assert result['peptide_aa_types'].shape == (2, 5)
        assert result['peptide_mask'].shape == (2, 5)

    def test_mask_padding(self):
        collate = PeptideDataCollate()
        batch = [
            self._make_item(3, [0, 1, 2], [[0.1]*7]*3),
            self._make_item(6, [3, 4, 5, 6, 7, 8], [[0.2]*7]*6),
        ]
        result = collate(batch)
        # First sample: 3 real, 3 padded
        assert result['peptide_mask'][0, 0].item()
        assert result['peptide_mask'][0, 1].item()
        assert result['peptide_mask'][0, 2].item()
        assert not result['peptide_mask'][0, 3].item()
        assert not result['peptide_mask'][0, 4].item()
        assert not result['peptide_mask'][0, 5].item()
        # Second sample: all 6 real
        assert result['peptide_mask'][1, 0].item()
        assert result['peptide_mask'][1, 5].item()

    def test_max_residues_clamp(self):
        collate = PeptideDataCollate(max_residues=4)
        batch = [
            self._make_item(10, list(range(10)), [[0.1]*7]*10),
        ]
        result = collate(batch)
        assert result['peptide_torsions'].shape == (1, 4, 7)

    def test_pad_values(self):
        collate = PeptideDataCollate(pad_torsion=0.0, pad_aa=25)
        batch = [
            self._make_item(2, [0, 1], [[0.1]*7]*2),
            self._make_item(4, [2, 3, 4, 5], [[0.2]*7]*4),
        ]
        result = collate(batch)
        # Padded AA positions should be pad_aa (25)
        assert result['peptide_aa_types'][0, 2].item() == 25
        assert result['peptide_aa_types'][0, 3].item() == 25
        # Padded torsion positions should be 0.0
        assert result['peptide_torsions'][0, 2, 0].item() == 0.0

    def test_cyclo_modes(self):
        collate = PeptideDataCollate()
        item1 = self._make_item(2, [0, 1], [[0.1]*7]*2)
        item1['cyclo_mode'] = 1
        item2 = self._make_item(3, [2, 3, 4], [[0.2]*7]*3)
        item2['cyclo_mode'] = 3
        result = collate([item1, item2])
        assert result['cyclo_modes'][0].item() == 1
        assert result['cyclo_modes'][1].item() == 3

    def test_empty_targets(self):
        collate = PeptideDataCollate()
        batch = [
            self._make_item(2, [0, 1], [[0.1]*7]*2, n_targets=0),
            self._make_item(3, [2, 3, 4], [[0.2]*7]*3, n_targets=0),
        ]
        result = collate(batch)
        # Targets are passed through from first item
        assert isinstance(result['target_coords'], list)

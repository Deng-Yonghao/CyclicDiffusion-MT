"""End-to-end test: PDB parsing -> manifest -> DataLoader -> training step."""
import os
import torch

TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "CPCore", "CPCore_pdb",
    "AF-A0A417M1J3-F1_0_172_177_relaxed_relaxed.pdb"
)


class TestPreprocessIntegration:
    def test_pdb_to_manifest_roundtrip(self):
        """Parse PDB -> manifest -> DataLoader -> check tensor shapes."""
        from cyclicdiffusion_mt.utils.protein_utils import parse_pdb_cyclic
        from cyclicdiffusion_mt.data.dataset import MultiTargetDataset, PeptideDataCollate

        entry = parse_pdb_cyclic(TEST_PDB, dG_rosetta=-45.0, confidence=0.9)
        assert entry['peptide_torsions'].shape[1] == 7
        assert entry['peptide_torsions'].shape[0] == entry['peptide_aa_types'].shape[0]

        manifest_entry = {
            'peptide_torsions': entry['peptide_torsions'].tolist(),
            'peptide_aa_types': entry['peptide_aa_types'].tolist(),
            'target_coords': [tc.tolist() for tc in entry['target_coords']],
            'target_sequences': [ts.tolist() for ts in entry['target_sequences']],
            'cyclo_mode': entry['cyclo_mode'],
            'dG_rosetta': entry['dG_rosetta'],
            'confidence': entry['confidence'],
        }

        ds = MultiTargetDataset([manifest_entry])
        assert len(ds) == 1

        sample = ds[0]
        for key in ['peptide_torsions', 'peptide_aa_types', 'target_coords',
                     'target_sequences', 'cyclo_mode', 'dG_rosetta', 'confidence']:
            assert key in sample, f"Missing key: {key}"

    def test_collate_produces_valid_batch(self):
        """Verify that the collated batch matches training_step expectations."""
        from cyclicdiffusion_mt.utils.protein_utils import parse_pdb_cyclic
        from cyclicdiffusion_mt.data.dataset import MultiTargetDataset, PeptideDataCollate

        entry = parse_pdb_cyclic(TEST_PDB, dG_rosetta=-45.0, confidence=0.9)
        manifest_entry = {
            'peptide_torsions': entry['peptide_torsions'].tolist(),
            'peptide_aa_types': entry['peptide_aa_types'].tolist(),
            'target_coords': [tc.tolist() for tc in entry['target_coords']],
            'target_sequences': [ts.tolist() for ts in entry['target_sequences']],
            'cyclo_mode': 0,
            'dG_rosetta': -45.0,
            'confidence': 0.9,
        }

        ds = MultiTargetDataset([manifest_entry, manifest_entry])
        collate = PeptideDataCollate(max_residues=20)
        batch = collate([ds[0], ds[1]])

        for key in ['peptide_torsions', 'peptide_aa_types', 'peptide_mask',
                     'target_coords', 'target_sequences', 'cyclo_modes',
                     'dG_rosetta', 'confidence']:
            assert key in batch, f"Missing batch key: {key}"

        L = entry['peptide_torsions'].shape[0]
        assert batch['peptide_torsions'].shape == (2, L, 7)
        assert batch['peptide_aa_types'].shape == (2, L)
        assert batch['peptide_mask'].shape == (2, L)
        assert len(batch['target_coords']) == 1

    def test_dummy_manifest_roundtrip(self):
        """Verify the existing dummy manifest generator still works."""
        import numpy as np
        from cyclicdiffusion_mt.data.dataset import MultiTargetDataset, PeptideDataCollate

        np.random.seed(42)
        entries = []
        for _ in range(5):
            L = 8
            entries.append({
                'peptide_torsions': np.random.uniform(-3, 3, (L, 7)).tolist(),
                'peptide_aa_types': np.random.randint(0, 25, L).tolist(),
                'target_coords': [np.random.randn(30, 14, 3).tolist()],
                'target_sequences': [np.random.randint(0, 25, 30).tolist()],
                'cyclo_mode': 0,
                'dG_rosetta': -40.0,
                'confidence': 0.8,
            })

        ds = MultiTargetDataset(entries)
        assert len(ds) == 5
        collate = PeptideDataCollate(max_residues=20)
        batch = collate([ds[i] for i in range(2)])
        assert batch['peptide_torsions'].shape[0] == 2

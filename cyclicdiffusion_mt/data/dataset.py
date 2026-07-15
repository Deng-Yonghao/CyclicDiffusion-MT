"""Multi-target peptide dataset."""
import torch
from torch.utils.data import Dataset
from cyclicdiffusion_mt.utils.constants import AA_TO_IDX, MAX_RESIDUES


class MultiTargetDataset(Dataset):
    """Dataset for multi-target cyclic peptide data.
    Returns per sample: dict with peptide_torsions, peptide_aa_types,
    target_coords, target_sequences, cyclo_mode, dG_rosetta, confidence.
    """
    def __init__(self, data_manifest, max_residues=MAX_RESIDUES, max_targets=3):
        self.manifest = data_manifest  # list of dicts
        self.max_residues = max_residues
        self.max_targets = max_targets

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        return {
            'peptide_torsions': torch.tensor(entry['peptide_torsions'], dtype=torch.float32),
            'peptide_aa_types': torch.tensor(entry['peptide_aa_types'], dtype=torch.long),
            'target_coords': [torch.tensor(tc, dtype=torch.float32) for tc in entry.get('target_coords', [])],
            'target_sequences': [torch.tensor(ts, dtype=torch.long) for ts in entry.get('target_sequences', [])],
            'cyclo_mode': entry.get('cyclo_mode', 0),
            'dG_rosetta': torch.tensor(entry.get('dG_rosetta', 0.0), dtype=torch.float32),
            'confidence': torch.tensor(entry.get('confidence', 1.0), dtype=torch.float32),
        }


class PeptideDataCollate:
    """Collate function to pad variable-length peptides and targets into batches.

    NOTE: All samples in a batch must share the same target structures.
    target_coords and target_sequences are taken from batch[0] only.
    """
    def __init__(self, max_residues=MAX_RESIDUES, max_atoms=14, pad_torsion=0.0, pad_aa=25):
        self.max_residues = max_residues
        self.max_atoms = max_atoms
        self.pad_torsion = pad_torsion
        self.pad_aa = pad_aa

    def __call__(self, batch):
        B = len(batch)

        # Verify homogeneous targets (all samples share same targets)
        if B > 1:
            first_k = len(batch[0].get('target_coords', []))
            for b_idx in range(1, B):
                assert len(batch[b_idx].get('target_coords', [])) == first_k, \
                    "PeptideDataCollate requires all batch samples to have the same number of targets"

        L_max = max(item['peptide_aa_types'].shape[0] for item in batch)
        L_max = min(L_max, self.max_residues)

        peptide_torsions = torch.full((B, L_max, 7), self.pad_torsion)
        peptide_aa_types = torch.full((B, L_max), self.pad_aa, dtype=torch.long)
        peptide_mask = torch.zeros(B, L_max, dtype=torch.bool)
        cyclo_modes = torch.zeros(B, dtype=torch.long)
        dG_rosetta = torch.zeros(B)
        confidence = torch.zeros(B)

        # Targets: padded to max targets across batch
        K_max = max(len(item.get('target_coords', [])) for item in batch)
        K_max = max(K_max, 1)

        for b, item in enumerate(batch):
            L = min(item['peptide_aa_types'].shape[0], L_max)
            peptide_torsions[b,:L] = item['peptide_torsions'][:L]
            peptide_aa_types[b,:L] = item['peptide_aa_types'][:L]
            peptide_mask[b,:L] = True
            cyclo_modes[b] = item.get('cyclo_mode', 0)
            dG_rosetta[b] = item.get('dG_rosetta', 0.0)
            confidence[b] = item.get('confidence', 1.0)

        return {
            'peptide_torsions': peptide_torsions,
            'peptide_aa_types': peptide_aa_types,
            'peptide_mask': peptide_mask,
            'target_coords': batch[0].get('target_coords', []),
            'target_sequences': batch[0].get('target_sequences', []),
            'cyclo_modes': cyclo_modes,
            'dG_rosetta': dG_rosetta,
            'confidence': confidence,
        }

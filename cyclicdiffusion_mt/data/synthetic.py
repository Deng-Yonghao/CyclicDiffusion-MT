# cyclicdiffusion_mt/data/synthetic.py
"""Synthetic multi-target data construction from single-target co-crystal data.

Implements the Tier 3 data pipeline: given a cyclic peptide-targetA complex,
pair it with additional target structures, check for clashes, estimate contact
surfaces, and assign a composite quality score.

Reference: spec Section 7.2 Synthetic Multi-Target Data Pipeline
"""

import math
import torch


def compute_quality_score(clash_count, contact_area, docking_score):
    """Composite confidence score c = c_clash * c_contact * c_docking.

    Each component maps a raw metric to [0, 1], with 1 being perfect quality.
    The product ensures that a single bad dimension can pull the score low.

    Args:
        clash_count: number of steric clash pairs between peptide and target.
        contact_area: estimated contact surface area in square Angstroms.
        docking_score: Rosetta dG-based quality (lower is better, ~0 = perfect).

    Returns:
        confidence: float in [0, 1].
    """
    # c_clash: exponential decay with clash count
    c_clash = math.exp(-clash_count / 10.0)

    # c_contact: sigmoid centered at 200 A^2 (reasonable min interface)
    c_contact = 1.0 / (1.0 + math.exp(-(contact_area - 200.0) / 100.0))

    # c_docking: exponential decay with |dG| -- penalize very poor binders
    c_docking = math.exp(-abs(docking_score) / 20.0)

    return c_clash * c_contact * c_docking


class SyntheticMultiTargetBuilder:
    """Builds synthetic multi-target data from single-target peptide complexes.

    For each peptide, pairs it with combinations of target structures from
    the pool, applies quality filtering, and returns a manifest of entries
    suitable for MultiTargetDataset ingestion.
    """

    def __init__(self, max_targets=3, clash_threshold=1.5, min_contact=100.0):
        self.max_targets = max_targets
        self.clash_threshold = clash_threshold
        self.min_contact = min_contact

    def build(self, peptide_data, target_pool):
        """Build synthetic multi-target entries for one peptide.

        Args:
            peptide_data: dict with keys:
                peptide_torsions: (L, 7) tensor
                peptide_aa_types: (L,) tensor
                cyclo_mode: int
                dG_rosetta: float (dG with primary target)
                confidence: float (base confidence)
            target_pool: list of target dicts, each with:
                coords: (N_k, 14, 3) tensor
                sequence: (N_k,) tensor
                name: str

        Returns:
            list of manifest entries, each ready for MultiTargetDataset.
        """
        if not target_pool:
            return []

        results = []

        # Single-target entry (primary target only)
        primary = target_pool[0]
        results.append({
            "peptide_torsions": peptide_data["peptide_torsions"].clone(),
            "peptide_aa_types": peptide_data["peptide_aa_types"].clone(),
            "target_coords": [primary["coords"].clone()],
            "target_sequences": [primary["sequence"].clone()],
            "cyclo_mode": peptide_data.get("cyclo_mode", 0),
            "dG_rosetta": peptide_data.get("dG_rosetta", 0.0),
            "confidence": peptide_data.get("confidence", 1.0),
        })

        # Multi-target entries: pair with additional targets
        for k in range(1, min(len(target_pool), self.max_targets)):
            extra = target_pool[k]

            # Estimate quality metrics for this pairing
            clash_count = self._estimate_clashes(
                peptide_data.get("peptide_coords", None), extra["coords"]
            )
            contact_area = self._estimate_contact(
                peptide_data.get("peptide_coords", None), extra["coords"]
            )
            dg_combined = peptide_data.get("dG_rosetta", 0.0)

            quality = compute_quality_score(clash_count, contact_area, dg_combined)

            if quality < 0.1:
                continue  # skip very poor pairings

            # Collect K targets for this entry
            target_coords = [target_pool[0]["coords"].clone()]
            target_seqs = [target_pool[0]["sequence"].clone()]
            for j in range(1, k + 1):
                target_coords.append(target_pool[j]["coords"].clone())
                target_seqs.append(target_pool[j]["sequence"].clone())

            results.append({
                "peptide_torsions": peptide_data["peptide_torsions"].clone(),
                "peptide_aa_types": peptide_data["peptide_aa_types"].clone(),
                "target_coords": target_coords,
                "target_sequences": target_seqs,
                "cyclo_mode": peptide_data.get("cyclo_mode", 0),
                "dG_rosetta": peptide_data.get("dG_rosetta", 0.0),
                "confidence": quality * peptide_data.get("confidence", 1.0),
            })

        return results

    def _estimate_clashes(self, peptide_coords, target_coords):
        """Estimate number of steric clash pairs.

        Simplified: returns 0 when peptide coords unavailable (coords are
        reconstructed on-the-fly by NeRF during training). Real implementation
        would compute Cartesian coords first.
        """
        if peptide_coords is None:
            return 0  # will be recomputed during training
        # Placeholder: compute pairwise distances if coords available
        pep_flat = peptide_coords.reshape(-1, 3)
        tar_flat = target_coords.reshape(-1, 3)
        if pep_flat.shape[0] == 0 or tar_flat.shape[0] == 0:
            return 0
        dists = torch.cdist(pep_flat[:100], tar_flat[:100])  # truncated for speed
        return (dists < self.clash_threshold).sum().item()

    def _estimate_contact(self, peptide_coords, target_coords):
        """Estimate contact surface area.

        Simplified: returns a default reasonable interface area when
        peptide coords are unavailable.
        """
        if peptide_coords is None:
            return 400.0  # typical small peptide interface
        pep_flat = peptide_coords.reshape(-1, 3)
        tar_flat = target_coords.reshape(-1, 3)
        if pep_flat.shape[0] == 0 or tar_flat.shape[0] == 0:
            return 400.0
        dists = torch.cdist(pep_flat[:100], tar_flat[:100])
        contacts = (dists < 8.0).sum().item()
        return contacts * 10.0  # rough A^2 per contact pair

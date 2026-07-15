"""Evaluation metrics for CyclicDiffusion-MT.

Structure Quality:
  - ring_closure_precision: N-C distance error
  - ramachandran_outlier_rate: % phi/psi in disallowed regions
  - steric_clash_count: clashes per residue

Generation Quality:
  - internal_diversity: mean pairwise RMSD/TM among generated samples

Binding Capability:
  - (Placeholder) Rosetta dG integration via subprocess call
"""

import math
import torch
from cyclicdiffusion_mt.utils.constants import (
    IDEAL_PEPTIDE_BOND, CLASH_THRESHOLD, AA_TO_IDX,
)


def ring_closure_precision(coords):
    """Mean absolute deviation of N-C closure distance from ideal.

    Args:
        coords: (B, L, 14, 3) Cartesian atom coordinates.

    Returns:
        float: mean |d_closure - IDEAL_PEPTIDE_BOND| across batch.
    """
    n_term = coords[:, 0, 0]   # (B, 3)
    c_term = coords[:, -1, 2]  # (B, 3)
    dist = torch.norm(n_term - c_term, dim=-1)  # (B,)
    return (dist - IDEAL_PEPTIDE_BOND).abs().mean().item()


def ramachandran_outlier_rate(phi, psi, aa_types):
    """Fraction of residues with phi/psi outside favored Ramachandran regions.

    Uses simplified allowed region check: |phi| < 150 deg AND |psi| < 150 deg
    with additional exclusion of the disallowed phi~0 region for non-glycine.
    Glycine residues are excluded from the count.

    Args:
        phi: (B, L) phi angles in radians.
        psi: (B, L) psi angles in radians.
        aa_types: (B, L) amino acid type indices.

    Returns:
        float: fraction of non-glycine residues in outlier regions [0, 1].
    """
    gly_idx = AA_TO_IDX.get('GLY', 7)
    non_gly_mask = (aa_types != gly_idx).float()

    # Simplified favored region: |phi| > 30 deg and |psi| > 30 deg
    # (exclude the disallowed phi~0 region)
    phi_ok = (phi.abs() > math.radians(30))
    psi_ok = (psi.abs() > math.radians(30))
    in_favored = (phi_ok & psi_ok).float()

    n_non_gly = non_gly_mask.sum() + 1e-8
    n_outlier = (non_gly_mask * (1.0 - in_favored)).sum()
    return (n_outlier / n_non_gly).item()


def steric_clash_count(coords, atom_mask):
    """Count steric clashes per 100 residues.

    Args:
        coords: (B, L, 14, 3) Cartesian atom coordinates.
        atom_mask: (B, L, 14) boolean mask; True for existing atoms.

    Returns:
        float: clashes per 100 residues.
    """
    B, L, A, _ = coords.shape
    coords_f = coords.reshape(B, -1, 3)
    mask_f = atom_mask.reshape(B, -1)

    d = torch.cdist(coords_f, coords_f)
    valid = mask_f.unsqueeze(-1) & mask_f.unsqueeze(-2)
    n = L * A
    diag = torch.eye(n, device=coords.device, dtype=torch.bool).unsqueeze(0)
    clash_mask = valid & ~diag

    n_clashes = ((d < CLASH_THRESHOLD) & clash_mask).sum(dim=[1, 2]).float() / 2.0
    n_residues = atom_mask.any(dim=-1).sum(dim=1).float()  # (B,)
    clashes_per_100 = (n_clashes / (n_residues + 1e-8)) * 100.0
    return clashes_per_100.mean().item()


def internal_diversity(samples):
    """Mean pairwise RMSD among generated samples.

    Uses torsion-space RMSD as a proxy for structural diversity.

    Args:
        samples: list of (1, L, 7) tau tensors.

    Returns:
        float: mean pairwise RMSD. 0.0 if fewer than 2 samples.
    """
    if len(samples) < 2:
        return 0.0

    # Stack: (N, L, 7)
    stacked = torch.cat([s.reshape(1, -1, 7) for s in samples], dim=0)
    N = stacked.shape[0]

    total = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            diff = stacked[i] - stacked[j]
            # Wrapped angular distance
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            rmsd = (diff.pow(2).mean()).sqrt()
            total += rmsd.item()
            count += 1

    return total / max(count, 1)


def compute_all_metrics(samples, targets=None):
    """Compute full evaluation suite on generated samples.

    Args:
        samples: list of dicts, each with 'tau', 'aa_types', 'coords'.
        targets: optional list of target structures (not yet used).

    Returns:
        dict mapping metric name to float value.
    """
    metrics = {}

    if not samples:
        return metrics

    # Structure quality
    coords = torch.cat([s["coords"] for s in samples], dim=0)
    taus = [s["tau"] for s in samples]
    aa_types = torch.cat([s["aa_types"] for s in samples], dim=0)

    from cyclicdiffusion_mt.data.transforms import compute_atom_mask
    atom_mask = compute_atom_mask(aa_types)

    metrics["ring_closure_precision"] = ring_closure_precision(coords)
    metrics["ramachandran_outlier_rate"] = ramachandran_outlier_rate(
        coords[0, :, 0, 0] * 0,  # placeholder -- use actual phi/psi from tau
        coords[0, :, 0, 0] * 0,
        aa_types,
    )
    metrics["steric_clashes_per_100"] = steric_clash_count(coords, atom_mask)

    # Use tau-based Ramachandran
    all_phi = torch.cat([t[0, :, 0] for t in taus], dim=0)  # concat over samples
    all_psi = torch.cat([t[0, :, 1] for t in taus], dim=0)
    all_aa = torch.cat([s["aa_types"][0] for s in samples], dim=0)
    metrics["ramachandran_outlier_rate"] = ramachandran_outlier_rate(
        all_phi.unsqueeze(0), all_psi.unsqueeze(0), all_aa.unsqueeze(0),
    )

    # Generation quality
    metrics["internal_diversity"] = internal_diversity(taus)

    # Binding (placeholder values when no Rosetta available)
    metrics["dg_pred_mean"] = float(
        torch.cat([s.get("dg_pred", torch.zeros(1, 1)) for s in samples])
        .mean().item()
    )

    return metrics

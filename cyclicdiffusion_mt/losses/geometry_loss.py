"""Geometry-based losses: steric clash, Ramachandran, rotamer."""

import torch
from cyclicdiffusion_mt.utils.constants import CLASH_THRESHOLD


def clash_loss(coords, atom_mask):
    """Steric clash loss penalising atom pairs closer than CLASH_THRESHOLD.

    Computes all pairwise distances among valid atoms (as determined by
    ``atom_mask``), excluding self-pairs via a diagonal mask, and sums the
    squared violations ``max(0, CLASH_THRESHOLD - d)^2``.  The result is
    normalised by the total number of valid pairs.

    Args:
        coords: (B, L, A, 3) Cartesian atom coordinates.
        atom_mask: (B, L, A) boolean mask; True for existing atoms.

    Returns:
        Scalar loss.
    """
    B, L, A, _ = coords.shape
    coords_f = coords.reshape(B, -1, 3)        # (B, L*A, 3)
    mask_f = atom_mask.reshape(B, -1)           # (B, L*A)

    d = torch.cdist(coords_f, coords_f)          # (B, L*A, L*A)
    valid = mask_f.unsqueeze(-1) & mask_f.unsqueeze(-2)  # pair mask

    # Exclude self-pairs
    n = L * A
    diag = torch.eye(n, device=coords.device, dtype=torch.bool).unsqueeze(0)
    clash_mask = valid & ~diag

    violations = (CLASH_THRESHOLD - d).clamp(min=0)
    numerator = (violations * clash_mask.float()).sum()
    denominator = clash_mask.float().sum() + 1e-8
    return numerator / denominator


def rama_loss(phi, psi, aa_types):
    """Ramachandran-based backbone torsion potential.

    Simplified penalty that pushes phi/psi towards the alpha-helical
    basin (phi ~ -57 deg, psi ~ -47 deg).  Glycine is exempt because of
    its much wider Ramachandran map.

    Args:
        phi: (B, L) backbone phi angles in radians.
        psi: (B, L) backbone psi angles in radians.
        aa_types: (B, L) amino acid type indices.

    Returns:
        Scalar loss.
    """
    from cyclicdiffusion_mt.utils.constants import AA_TO_IDX

    # alpha-helix centre in radians: -57 deg ~ -0.995, -47 deg ~ -0.820
    phi_alpha = -0.995
    psi_alpha = -0.820

    phi_diff = torch.atan2(torch.sin(phi - phi_alpha), torch.cos(phi - phi_alpha))
    psi_diff = torch.atan2(torch.sin(psi - psi_alpha), torch.cos(psi - psi_alpha))

    gly_idx = AA_TO_IDX['GLY']
    gly_mask = (aa_types == gly_idx).float()
    non_gly = 1.0 - gly_mask

    loss = non_gly * (phi_diff.pow(2) + psi_diff.pow(2)) * 0.1
    return loss.mean()


def rotamer_loss(chis, aa_types):
    """Rotamer-based chi angle loss.

    Penalises chi angles that deviate from staggered conformers
    (60 deg, 180 deg, -60 deg).  The loss for each chi is the minimum
    squared angular distance to any staggered rotamer.  Positions
    where the AA has no chi angle (according to ``AA_CHI_COUNTS``) are
    masked out.

    Args:
        chis: (B, L, 4) chi angles in radians (padded with zeros).
        aa_types: (B, L) amino acid type indices.

    Returns:
        Scalar loss.
    """
    from cyclicdiffusion_mt.utils.constants import AA_CHI_COUNTS

    staggered = torch.tensor(
        [-torch.pi, -torch.pi / 3.0, torch.pi / 3.0],
        device=chis.device,
    )

    # (B, L, 4, 1)  vs  (1, 1, 1, 3)  ->  (B, L, 4, 3)
    chis_exp = chis.unsqueeze(-1)
    stag_exp = staggered.view(1, 1, 1, -1)
    diff = torch.atan2(torch.sin(chis_exp - stag_exp), torch.cos(chis_exp - stag_exp))
    min_dist_sq = diff.pow(2).min(dim=-1).values  # (B, L, 4)

    # Build chi mask from AA_CHI_COUNTS
    B, L = aa_types.shape
    chi_mask = torch.zeros(B, L, 4, device=aa_types.device)
    for b in range(B):
        for l in range(L):
            aa = aa_types[b, l].item()
            n_chi = AA_CHI_COUNTS[aa]
            chi_mask[b, l, :n_chi] = 1.0

    numerator = (min_dist_sq * chi_mask).sum()
    denominator = chi_mask.sum() + 1e-8
    return numerator / denominator

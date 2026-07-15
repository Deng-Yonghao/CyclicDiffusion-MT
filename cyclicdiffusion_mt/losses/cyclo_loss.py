"""Ring closure distance error loss."""

import torch
from cyclicdiffusion_mt.utils.constants import IDEAL_PEPTIDE_BOND


def cyclo_loss(nerf, tau_pred, aa_types, bonds, angles, cyclo_mode):
    """Ring closure loss via squared distance error at the cyclization point.

    Uses the NeRF module to reconstruct Cartesian coordinates from internal
    coordinates and then measures the distance between the N-terminal N atom
    (residue 0, atom 0) and the C-terminal C atom (residue L-1, atom 2).
    The loss is the squared deviation of this distance from the ideal peptide
    bond length.

    Currently supports the ``head_to_tail`` cyclization mode (mode 0).
    Other modes are reserved for future expansion (sidechain-to-tail,
    sidechain-to-sidechain, head-to-sidechain, bicyclic).

    Args:
        nerf: NeRF module with ``forward(bonds, angles, torsions, aa_types)``
            returning Cartesian coordinates ``(B, L, 14, 3)``.
        tau_pred: (B, L, 7) predicted torsion angles in radians.
        aa_types: (B, L) amino acid type indices.
        bonds: (B, L, 14) bond lengths in Angstroms.
        angles: (B, L, 14) bond angles in radians.
        cyclo_mode: (B,) cyclization mode indices (0 = head_to_tail).

    Returns:
        Scalar loss: MSE of (d_closure - IDEAL_PEPTIDE_BOND)^2.
    """
    coords = nerf(bonds, angles, tau_pred, aa_types)
    # head-to-tail: N-term N (residue 0, atom 0) vs C-term C (residue L-1, atom 2)
    n_term = coords[:, 0, 0]    # (B, 3)
    c_term = coords[:, -1, 2]   # (B, 3)
    dist = torch.norm(n_term - c_term, dim=-1)  # (B,)
    return (dist - IDEAL_PEPTIDE_BOND).pow(2).mean()

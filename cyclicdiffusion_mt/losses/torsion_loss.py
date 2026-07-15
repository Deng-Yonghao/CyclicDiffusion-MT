"""Wrapped L2 loss on torsion angle variables."""

import torch


def torsion_loss(tau_pred, tau_0, chi_mask):
    """Masked wrapped L2 loss for torsion angles.

    Handles the circular (wrapped) nature of angle variables by computing
    the minimum angular distance on the unit circle using atan2(sin, cos).

    Args:
        tau_pred: (B, L, 7) predicted torsion angles in radians.
        tau_0: (B, L, 7) ground truth torsion angles in radians.
        chi_mask: (B, L, 7) boolean mask; True for valid torsion positions
            (phi/psi/omega always True; chi1-chi4 True only when the
            corresponding sidechain torsion exists for the AA type).

    Returns:
        Scalar loss: mean squared wrapped angular error over valid positions.
    """
    diff = tau_0 - tau_pred
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    loss_per_pos = diff.pow(2)
    numerator = (loss_per_pos * chi_mask.float()).sum()
    denominator = chi_mask.float().sum() + 1e-8
    return numerator / denominator

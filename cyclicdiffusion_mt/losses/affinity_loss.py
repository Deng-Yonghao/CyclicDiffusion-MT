"""Confidence-weighted MSE loss for binding affinity regression."""

import torch.nn.functional as F


def affinity_loss(dg_pred, dg_label, confidence):
    """Confidence-weighted mean squared error for dG prediction.

    Each sample is weighted by its confidence score so that high-confidence
    predictions contribute more to the total loss.

    Args:
        dg_pred: (B, C) predicted binding free-energy values.
        dg_label: (B, C) ground truth dG values.
        confidence: (B,) confidence scores in [0, 1].

    Returns:
        Scalar loss.
    """
    mse = F.mse_loss(dg_pred, dg_label, reduction='none').mean(dim=-1)  # (B,)
    return (mse * confidence).mean()

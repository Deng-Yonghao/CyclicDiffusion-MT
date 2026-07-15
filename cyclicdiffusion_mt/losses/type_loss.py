"""Cross-entropy loss for amino acid type prediction."""

import torch.nn.functional as F
from cyclicdiffusion_mt.utils.constants import NUM_AA_TYPES_WITH_MASK, MASK_IDX


def type_loss(a_logits, a_0):
    """Cross-entropy loss for discrete AA type diffusion.

    Positions where ``a_0 == MASK_IDX`` are ignored (MASK_IDX is the
    special token used in masked discrete diffusion for positions that
    have not yet been unmasked).

    Args:
        a_logits: (B, L, NUM_AA_TYPES_WITH_MASK) predicted logits.
        a_0: (B, L) ground truth AA type indices (may contain MASK_IDX).

    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(
        a_logits.reshape(-1, NUM_AA_TYPES_WITH_MASK),
        a_0.reshape(-1),
        ignore_index=MASK_IDX,
    )

"""IPA-based target protein encoder with shared weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IPABlock(nn.Module):
    """Single Invariant Point Attention block."""

    def __init__(self, d_model=128, d_head=32, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model, self.d_head, self.n_heads = d_model, d_head, n_heads
        self.qkv = nn.Linear(d_model, 3 * d_head * n_heads)
        self.o_proj = nn.Linear(d_head * n_heads, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """x:(B,N,d_model) mask:(B,N)"""
        B, N, D = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        x = residual + self.o_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class TargetEncoder(nn.Module):
    """IPA-based encoder for protein target structures. Shared weights across targets."""

    def __init__(self, d_target=128, n_blocks=3, d_head=32, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_target = d_target
        # Input projection: mean over 14 atoms per residue + atom type embedding
        self.atom_proj = nn.Linear(3, d_target)  # per-atom xyz -> features
        self.ipa_blocks = nn.ModuleList([
            IPABlock(d_target, d_head, n_heads, dropout) for _ in range(n_blocks)
        ])

    def forward(self, target_coords, target_masks):
        """target_coords: list[(B,N_k,14,3)]  target_masks: list[(B,N_k)]
        Returns: list[(B,N_k,d_target)]"""
        outputs = []
        for coords, mask in zip(target_coords, target_masks):
            B, N, _, _ = coords.shape
            # Mean-pool atom features to residue-level
            atom_feats = self.atom_proj(coords)  # (B,N,14,d_target)
            valid_mask = mask.unsqueeze(-1).unsqueeze(-1).float()
            res_feats = (atom_feats * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-8)
            for block in self.ipa_blocks:
                res_feats = block(res_feats, mask)
            outputs.append(res_feats)
        return outputs

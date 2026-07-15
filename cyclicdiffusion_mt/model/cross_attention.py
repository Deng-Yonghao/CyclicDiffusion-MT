"""Multi-target cross-attention with adaptive per-target gating."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveTargetGating(nn.Module):
    """Learns per-target importance weights alpha_k = softmax(g(h_k, t))."""

    def __init__(self, d_target=128, d_time=64, hidden=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_target + d_time, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1, bias=False),  # bias cancels in softmax
        )

    def forward(self, target_feats, t_emb):
        """target_feats: list[(B,N_k,d_target)]  t_emb: (B,d_time) -> alpha: (B,K)

        Args:
            target_feats: list of tensors, each (B, N_k, d_target)
            t_emb: time embedding (B, d_time) or (B,) for scalar timestep

        Returns:
            alpha: per-target gating weights (B, K), softmax-normalized
        """
        K = len(target_feats)
        B = target_feats[0].shape[0]

        # Handle scalar timestep input
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(-1)  # (B,) -> (B, 1)
            # If d_time > 1, pad or project
            if t_emb.shape[-1] < self.gate[0].in_features - target_feats[0].shape[-1]:
                d_time_expected = self.gate[0].in_features - target_feats[0].shape[-1]
                t_emb = t_emb.expand(-1, d_time_expected)

        scores = []
        for k in range(K):
            # Pool target features: mean over residues
            pooled = target_feats[k].mean(dim=1)  # (B, d_target)
            inp = torch.cat([pooled, t_emb], dim=-1)  # (B, d_target + d_time)
            scores.append(self.gate(inp))  # (B, 1)
        scores = torch.cat(scores, dim=-1)  # (B, K)
        return F.softmax(scores, dim=-1)


class MultiTargetCrossAttention(nn.Module):
    """Multi-head cross-attention from peptide (Q) to concatenated targets (K,V).

    Injects target-conditioned information into peptide residue representations
    via cross-attention with adaptive per-target gating and target index embeddings.
    """

    def __init__(self, d_model=256, d_target=128, d_head=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, d_head * n_heads)
        self.kv_proj = nn.Linear(d_target, 2 * d_head * n_heads)
        self.o_proj = nn.Linear(d_head * n_heads, d_model)

        self.gating = AdaptiveTargetGating(d_target)
        self.target_idx_embed = nn.Embedding(10, d_target)  # max 10 targets
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, peptide_feats, peptide_mask, target_feats, target_masks, t_emb, cyclo_emb=None):
        """Cross-attention from peptide to multi-target features.

        Args:
            peptide_feats: (B, L, d_model) - peptide residue features
            peptide_mask: (B, L) - boolean peptide residue mask
            target_feats: list[(B, N_k, d_target)] - per-target residue features
            target_masks: list[(B, N_k)] - boolean per-target residue masks
            t_emb: (B, d_time) - diffusion time embedding
            cyclo_emb: optional (B, d_cyclo) - cyclization mode embedding

        Returns:
            updated peptide features (B, L, d_model)
        """
        B, L, _ = peptide_feats.shape
        K = len(target_feats)

        # Add target index embedding and concatenate targets
        target_parts = []
        target_mask_parts = []
        for k, (tf, tm) in enumerate(zip(target_feats, target_masks)):
            idx_emb = self.target_idx_embed(torch.tensor(k, device=tf.device))
            target_parts.append(tf + idx_emb)
            target_mask_parts.append(tm)
        all_targets = torch.cat(target_parts, dim=1)  # (B, sum(N_k), d_target)
        all_tmasks = torch.cat(target_mask_parts, dim=1)  # (B, sum(N_k))

        # Adaptive gating
        alpha = self.gating(target_feats, t_emb)  # (B, K)

        # Build segment ID for each key position (which target it belongs to)
        seg_ids = torch.cat([
            torch.full((tf.shape[1],), k, device=tf.device, dtype=torch.long)
            for k, tf in enumerate(target_feats)
        ], dim=0)  # (sum(N_k),)
        # Gather gating weight per key position: (B, sum(N_k))
        alpha_per_pos = alpha[:, seg_ids]  # (B, sum(N_k))
        # Log-alpha bias for attention: (B, 1, 1, sum(N_k))
        attn_bias = torch.log(alpha_per_pos + 1e-8).unsqueeze(1).unsqueeze(2)

        # Compute Q, K, V
        Q = self.q_proj(self.norm(peptide_feats)).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        kv = self.kv_proj(all_targets).view(B, -1, 2, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        K_target, V = kv[0], kv[1]  # each (B, n_heads, sum(N_k), d_head)

        # Cross-attention with gating bias
        attn = (Q @ K_target.transpose(-2, -1)) / math.sqrt(self.d_head)
        # attn shape: (B, n_heads, L, sum(N_k))
        attn = attn + attn_bias  # inject per-target gating as additive log-probability
        attn_mask = ~all_tmasks.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, sum(N_k))
        attn = attn.masked_fill(attn_mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, -1)
        out = self.o_proj(out)

        # Apply as residual update
        return peptide_feats + out

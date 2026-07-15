"""SE(3) frame-based denoiser with multi-target cross-attention."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cyclicdiffusion_mt.model.cross_attention import MultiTargetCrossAttention


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time-step embedding as used in diffusion models."""

    def __init__(self, d_time=64):
        super().__init__()
        self.d_time = d_time

    def forward(self, t):
        """t: (B,) -> (B, d_time)"""
        device = t.device
        half = self.d_time // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=device, dtype=torch.float32)
            / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FrameUpdate(nn.Module):
    """SE(3) frame update via self-attention on residue features.

    Uses multi-head self-attention as a simplified frame update mechanism.
    Full SE(3) equivariant message passing can replace this later.
    """

    def __init__(self, d_model=256, d_head=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_head * n_heads)
        self.o_proj = nn.Linear(d_head * n_heads, d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 16, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.rbf_centers = nn.Parameter(torch.linspace(2.0, 20.0, 16))
        self.rbf_sigma = 2.0
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, frames, mask):
        """Self-attention update on residue features.

        Args:
            feats: (B, L, d_model) residue features
            frames: (B, L, 3, 3) or None — placeholder; full SE(3) not yet used
            mask: (B, L) boolean mask

        Returns:
            updated features (B, L, d_model)
        """
        B, L, D = feats.shape
        qkv = (
            self.qkv(feats)
            .view(B, L, 3, self.n_heads, self.d_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(
                ~mask.unsqueeze(1).unsqueeze(2), -1e9
            )
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class AffinityHead(nn.Module):
    """Lightweight Rosetta dG regression head.

    Pools peptide and target features, then predicts per-target binding affinity.
    """

    def __init__(self, d_model=256, d_target=128, hidden_dim=128):
        super().__init__()
        self.pep_pool = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tar_pool = nn.Sequential(
            nn.Linear(d_target, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pep_feats, target_feats, pep_mask, t_masks):
        """Predict binding affinity for each target.

        Args:
            pep_feats: (B, L, d_model) peptide residue features
            target_feats: list[(B, N_k, d_target)] per-target features
            pep_mask: (B, L) boolean peptide mask
            t_masks: list[(B, N_k)] per-target boolean masks

        Returns:
            dg_pred: (B, K) predicted affinity per target
        """
        # Pool peptide features via masked mean
        pep = (pep_feats * pep_mask.unsqueeze(-1).float()).sum(dim=1) / (
            pep_mask.sum(dim=1, keepdim=True).float() + 1e-8
        )
        pep = self.pep_pool(pep)  # (B, hidden_dim)

        dg_preds = []
        for k, (tf, tm) in enumerate(zip(target_feats, t_masks)):
            t_pooled = (tf * tm.unsqueeze(-1).float()).sum(dim=1) / (
                tm.sum(dim=1, keepdim=True).float() + 1e-8
            )
            t_emb = self.tar_pool(t_pooled)  # (B, hidden_dim)
            dg_preds.append(
                self.predictor(torch.cat([pep, t_emb], dim=-1))
            )  # (B, 1)
        return torch.cat(dg_preds, dim=-1)  # (B, K)


class DenoiserBlock(nn.Module):
    """One denoiser block: frame update + cross-attention + cyclo injection + FFN."""

    def __init__(
        self,
        d_model=256,
        d_target=128,
        d_time=64,
        d_head=64,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.frame_update = FrameUpdate(d_model, d_head, n_heads, dropout)
        self.cross_attn = MultiTargetCrossAttention(
            d_model, d_target, d_head, n_heads, dropout
        )
        self.cyclo_proj = nn.Linear(32, d_model)  # cyclo embedding projected to d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, feats, frames, mask, targets, t_masks, t_emb, cyclo_emb):
        """Process features through one denoising block.

        Args:
            feats: (B, L, d_model) residue features
            frames: (B, L, 3, 3) or None — placeholder
            mask: (B, L) boolean mask
            targets: list[(B, N_k, d_target)] per-target features
            t_masks: list[(B, N_k)] per-target boolean masks
            t_emb: (B, d_time) time embedding
            cyclo_emb: (B, L, 32) cyclization mode features per residue

        Returns:
            updated features (B, L, d_model)
        """
        # Frame update
        feats = feats + self.frame_update(feats, frames, mask)
        feats = self.norm1(feats)
        # Cross-attention to targets (returns residual-included output)
        feats = self.cross_attn(feats, mask, targets, t_masks, t_emb)
        feats = self.norm2(feats)
        # Cyclization bias injection
        cyclo_bias = self.cyclo_proj(cyclo_emb)  # (B, L, d_model)
        feats = feats + cyclo_bias
        # FFN with pre-norm
        feats = feats + self.ffn(self.norm3(feats))
        return feats


class FrameDenoiser(nn.Module):
    """Main frame-based denoiser: predicts clean torsions, AA types, and affinity.

    Takes noisy torsions + AA types + timestep + target features + cyclo mode,
    and predicts denoised torsion angles, amino acid type logits, and binding
    affinity for each target.
    """

    def __init__(
        self,
        d_model=256,
        d_target=128,
        d_time=64,
        n_blocks=6,
        d_head=64,
        n_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projections
        self.torsion_proj = nn.Linear(7, d_model)  # 7 torsions -> d_model
        self.aa_embed = nn.Embedding(26, d_model)  # 25 AA + MASK
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_time),
            nn.Linear(d_time, d_time),
            nn.SiLU(),
            nn.Linear(d_time, d_time),
        )
        self.cyclo_embed = nn.Embedding(5, 32)  # 5 cyclization modes -> 32

        # Denoiser blocks
        self.blocks = nn.ModuleList([
            DenoiserBlock(d_model, d_target, d_time, d_head, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # Output heads
        self.torsion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 7),
        )
        self.aa_head = nn.Linear(d_model, 26)  # 25 AA + MASK
        self.affinity_head = AffinityHead(d_model, d_target)

    def forward(self, tau_t, a_t, t, target_feats, target_masks, cyclo_mode):
        """Denoise torsion angles and predict AA types and binding affinity.

        Args:
            tau_t: (B, L, 7) noisy torsion angles at time t
            a_t: (B, L) noisy amino acid types (int indices 0..25)
            t: (B,) diffusion timestep in [0, 1]
            target_feats: list[(B, N_k, d_target)] per-target residue features
            target_masks: list[(B, N_k)] per-target residue masks
            cyclo_mode: (B,) cyclization mode indices {0..4}

        Returns:
            tau_pred: (B, L, 7) predicted clean torsion angles in [-pi, pi)
            aa_logits: (B, L, 26) predicted AA type logits
            dg_pred: (B, K) predicted binding affinity per target
        """
        B, L = tau_t.shape[0], tau_t.shape[1]

        # Merge torsion + AA embeddings
        torsion_feats = self.torsion_proj(tau_t)  # (B, L, d_model)
        aa_feats = self.aa_embed(a_t.clamp(0, 25))  # (B, L, d_model)
        feats = torsion_feats + aa_feats

        # Time embedding
        t_emb = self.time_mlp(t)  # (B, d_time=64)

        # Cyclization embedding — expand to per-residue
        cyclo_emb = self.cyclo_embed(cyclo_mode)  # (B, 32)
        cyclo_emb_expanded = cyclo_emb.unsqueeze(1).expand(
            -1, L, -1
        )  # (B, L, 32)

        # Dummy frames (placeholder for full SE(3) implementation)
        frames = None
        mask = torch.ones(B, L, dtype=torch.bool, device=tau_t.device)

        # Process through blocks
        for block in self.blocks:
            feats = block(
                feats, frames, mask, target_feats, target_masks, t_emb,
                cyclo_emb_expanded,
            )

        # Output predictions
        tau_pred = self.torsion_head(feats)  # (B, L, 7)
        tau_pred = torch.atan2(
            torch.sin(tau_pred), torch.cos(tau_pred)
        )  # wrap to [-pi, pi)
        aa_logits = self.aa_head(feats)  # (B, L, 26)
        dg_pred = self.affinity_head(feats, target_feats, mask, target_masks)

        return tau_pred, aa_logits, dg_pred

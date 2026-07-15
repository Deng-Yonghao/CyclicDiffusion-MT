"""Main training script for CyclicDiffusion-MT.

Three-phase training:
  Phase 1: Single-target pretrain (K=1, no cyclo/affinity loss)
  Phase 2: Multi-target + cyclization (K=2-3, full loss)
  Phase 3: Affinity fine-tune (high-quality subset, Rosetta dG)

Usage:
    python -m cyclicdiffusion_mt.train --config config/config.yaml
"""

import os
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from cyclicdiffusion_mt.config.train_config import TrainConfig, load_config
from cyclicdiffusion_mt.data.dataset import MultiTargetDataset, PeptideDataCollate
from cyclicdiffusion_mt.data.geometry import build_ideal_geometry
from cyclicdiffusion_mt.data.transforms import compute_chi_mask, compute_atom_mask
from cyclicdiffusion_mt.model.nerf import NeRF
from cyclicdiffusion_mt.model.target_encoder import TargetEncoder
from cyclicdiffusion_mt.model.diffusion import WrappedNormalDiffusion, MaskedDiscreteDiffusion
from cyclicdiffusion_mt.model.denoiser import FrameDenoiser
from cyclicdiffusion_mt.losses import (
    torsion_loss, type_loss, cyclo_loss,
    clash_loss, rama_loss, rotamer_loss, affinity_loss,
)


class TrainingLoop:
    """Orchestrates the full training pipeline for CyclicDiffusion-MT."""

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model components
        mc = config.model
        self.target_encoder = TargetEncoder(
            d_target=mc.d_target, n_blocks=3, d_head=32, n_heads=4, dropout=mc.dropout,
        ).to(self.device)

        self.diffusion_wn = WrappedNormalDiffusion(T=mc.T).to(self.device)
        self.diffusion_md = MaskedDiscreteDiffusion(T=mc.T, num_classes=mc.num_aa_types).to(self.device)

        self.denoiser = FrameDenoiser(
            d_model=mc.d_model, d_target=mc.d_target, d_time=mc.d_time,
            n_blocks=mc.n_blocks, d_head=mc.d_head, n_heads=mc.n_heads,
            dropout=mc.dropout,
        ).to(self.device)

        self.nerf = NeRF().to(self.device)
        self.nerf.init_frame.requires_grad_(False)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.target_encoder.parameters())
            + list(self.denoiser.parameters()),
            lr=config.lr, weight_decay=config.weight_decay,
        )

        self.scaler = GradScaler(enabled=config.mixed_precision and self.device.type == "cuda")
        self.global_step = 0
        self.current_phase = 0

    @property
    def model(self):
        """Convenience access for tests."""
        return self.denoiser

    def _encode_targets(self, target_coords, target_masks):
        """Run target encoder on each target's coordinates.

        Args:
            target_coords: list[(B, N_k, 14, 3)] — coords already on device
            target_masks: list[(B, N_k)] — boolean masks already on device

        Returns:
            list[(B, N_k, d_target)] encoded features
        """
        if not target_coords:
            return [], []
        encoded = self.target_encoder(target_coords, target_masks)
        return encoded, target_masks

    def training_step(self, batch):
        """Single training step for one batch.

        Args:
            batch: dict from PeptideDataCollate with keys:
                peptide_torsions, peptide_aa_types, peptide_mask,
                target_coords, target_sequences, cyclo_modes,
                dG_rosetta, confidence

        Returns:
            dict of loss components (all scalars on current device).
        """
        tau_0 = batch["peptide_torsions"].to(self.device)        # (B, L, 7)
        a_0 = batch["peptide_aa_types"].to(self.device)           # (B, L)
        pep_mask = batch["peptide_mask"].to(self.device)          # (B, L)
        cyclo_mode = batch["cyclo_modes"].to(self.device)         # (B,)
        dg_label = batch["dG_rosetta"].to(self.device)            # (B,)
        confidence = batch["confidence"].to(self.device)          # (B,)

        B, L = tau_0.shape[0], tau_0.shape[1]

        # --- Prepare target coordinates and masks ---
        target_coords_raw = batch.get("target_coords", [])
        target_seqs_raw = batch.get("target_sequences", [])

        target_coords = []
        target_masks = []
        for tc, ts in zip(target_coords_raw, target_seqs_raw):
            tc = tc.to(self.device)  # (N_k, 14, 3) per-target from collate
            ts = ts.to(self.device)  # (N_k,)
            # Add batch dim and expand to match peptide batch size
            tc = tc.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N_k, 14, 3)
            ts = ts.unsqueeze(0).expand(B, -1)           # (B, N_k)
            # Build residue mask: any atom present
            t_mask = (tc.abs().sum(dim=-1).sum(dim=-1) > 1e-6)  # (B, N_k)
            target_coords.append(tc)
            target_masks.append(t_mask)

        # --- Encode targets ---
        phase = self.cfg.phases[self.current_phase]
        target_feats, target_masks_enc = self._encode_targets(
            target_coords[:phase.max_targets],
            target_masks[:phase.max_targets],
        )

        # --- Sample diffusion timestep ---
        t = torch.rand(B, device=self.device)  # (B,) continuous in [0, 1]
        t_int = (t * (self.diffusion_wn.T - 1)).long().clamp(0, self.diffusion_wn.T - 1)

        # --- Forward diffusion ---
        tau_t, noise = self.diffusion_wn.q_sample(tau_0, t_int)
        a_t = self.diffusion_md.q_sample(a_0, t_int)

        # --- Denoise ---
        tau_pred, aa_logits, dg_pred = self.denoiser(
            tau_t, a_t, t, target_feats, target_masks_enc, cyclo_mode,
        )

        # --- Loss computation ---
        chi_mask = compute_chi_mask(a_0).to(self.device)

        # Core diffusion losses
        loss_t = torsion_loss(tau_pred, tau_0, chi_mask)
        loss_ty = type_loss(aa_logits, a_0)

        losses = {
            "torsion": loss_t,
            "type": loss_ty,
        }

        lw = phase.loss_weights

        # Cyclization loss (requires NeRF)
        if phase.use_cyclo and lw.cyclo > 0:
            bonds, angles = build_ideal_geometry(a_0)
            bonds, angles = bonds.to(self.device), angles.to(self.device)
            loss_c = cyclo_loss(self.nerf, tau_pred, a_0, bonds, angles, cyclo_mode)
            losses["cyclo"] = loss_c
        else:
            loss_c = torch.tensor(0.0, device=self.device)
            losses["cyclo"] = loss_c

        # Geometry loss (requires NeRF Cartesian output)
        if lw.geometry > 0:
            bonds, angles = build_ideal_geometry(a_0)
            bonds, angles = bonds.to(self.device), angles.to(self.device)
            coords = self.nerf(bonds, angles, tau_pred, a_0)
            atom_mask = compute_atom_mask(a_0).to(self.device)

            loss_clash = clash_loss(coords, atom_mask)
            loss_rama = rama_loss(tau_pred[..., 0], tau_pred[..., 1], a_0)
            loss_rot = rotamer_loss(tau_pred[..., 3:], a_0)
            loss_g = loss_clash + loss_rama + loss_rot
            losses["geometry"] = loss_g
        else:
            loss_g = torch.tensor(0.0, device=self.device)
            losses["geometry"] = loss_g

        # Affinity loss
        if phase.use_affinity and lw.affinity > 0:
            loss_a = affinity_loss(dg_pred, dg_label.unsqueeze(-1).expand_as(dg_pred), confidence)
            losses["affinity"] = loss_a
        else:
            loss_a = torch.tensor(0.0, device=self.device)
            losses["affinity"] = loss_a

        # Weighted total
        total = (
            lw.torsion * loss_t
            + lw.type * loss_ty
            + lw.cyclo * loss_c
            + lw.geometry * loss_g
            + lw.affinity * loss_a
        )
        losses["total"] = total

        return losses

    def train_epoch(self, loader, phase_idx):
        """Run one training epoch. Returns average loss dict."""
        self.current_phase = phase_idx
        self.target_encoder.train()
        self.denoiser.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for batch in loader:
            self.optimizer.zero_grad()

            with autocast(enabled=self.cfg.mixed_precision and self.device.type == "cuda"):
                loss_dict = self.training_step(batch)

            self.scaler.scale(loss_dict["total"]).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.target_encoder.parameters())
                + list(self.denoiser.parameters()),
                self.cfg.grad_clip,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()
            n_batches += 1
            self.global_step += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    def save_checkpoint(self, path):
        """Save model state."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "global_step": self.global_step,
            "current_phase": self.current_phase,
            "target_encoder": self.target_encoder.state_dict(),
            "denoiser": self.denoiser.state_dict(),
            "nerf": self.nerf.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.cfg.to_dict(),
        }, path)

    def load_checkpoint(self, path):
        """Load model state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.denoiser.load_state_dict(ckpt["denoiser"])
        self.nerf.load_state_dict(ckpt["nerf"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["global_step"]
        self.current_phase = ckpt["current_phase"]
        return ckpt


def main(config_path=None):
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="CyclicDiffusion-MT Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = parser.parse_args()

    path = config_path or args.config
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")

    cfg = load_config(path)
    torch.manual_seed(cfg.seed)

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Config loaded from: {path}")

    # --- Data loading ---
    # In production, data_manifest comes from preprocessing pipeline.
    # Here we demonstrate the structure; actual data paths come from config.
    manifest_path = os.path.join(cfg.data.data_root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Data manifest not found at {manifest_path}. "
            "Run the data preprocessing pipeline first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    dataset = MultiTargetDataset(manifest, max_residues=cfg.data.max_residues, max_targets=cfg.data.max_targets)
    collate = PeptideDataCollate(max_residues=cfg.data.max_residues, max_atoms=cfg.data.max_atoms)
    loader = DataLoader(
        dataset, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory, collate_fn=collate,
    )

    # --- Training loop ---
    loop = TrainingLoop(cfg)

    for phase_idx, phase in enumerate(cfg.phases):
        print(f"\n{'='*50}")
        print(f"Phase {phase_idx+1}/{len(cfg.phases)}: {phase.name}")
        print(f"  Epochs: {phase.epochs}")
        print(f"  Loss weights: {phase.loss_weights}")
        print(f"  Max targets: {phase.max_targets}")
        print(f"{'='*50}")

        for epoch in range(phase.epochs):
            avg_losses = loop.train_epoch(loader, phase_idx)

            if epoch % cfg.log_interval == 0 or epoch == phase.epochs - 1:
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
                print(f"  Epoch {epoch+1}/{phase.epochs} | {loss_str}")

            if loop.global_step % cfg.save_interval == 0:
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir,
                    f"phase{phase_idx+1}_step{loop.global_step}.pt",
                )
                loop.save_checkpoint(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

        # Save phase completion checkpoint
        phase_ckpt = os.path.join(cfg.checkpoint_dir, f"phase{phase_idx+1}_complete.pt")
        loop.save_checkpoint(phase_ckpt)
        print(f"Phase {phase.name} complete. Checkpoint: {phase_ckpt}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

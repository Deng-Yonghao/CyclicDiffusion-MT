"""Sampling / inference for CyclicDiffusion-MT.

Runs reverse diffusion: noise -> clean torsions + AA types, with optional
hard cyclization projection for ring closure enforcement.

Usage:
    python -m cyclicdiffusion_mt.sample --checkpoint path/to/checkpoint.pt \
        --target_A target.pdb --num_samples 10
"""

import argparse
import os

import torch

from cyclicdiffusion_mt.config.train_config import TrainConfig, load_config
from cyclicdiffusion_mt.model.nerf import NeRF
from cyclicdiffusion_mt.model.target_encoder import TargetEncoder
from cyclicdiffusion_mt.model.diffusion import WrappedNormalDiffusion, MaskedDiscreteDiffusion
from cyclicdiffusion_mt.model.denoiser import FrameDenoiser
from cyclicdiffusion_mt.data.geometry import build_ideal_geometry
from cyclicdiffusion_mt.utils.constants import IDEAL_PEPTIDE_BOND


def hard_cyclo_projection(nerf, tau, aa_types, bonds, angles, cyclo_mode,
                          n_steps=50, lr=0.01, target_dist=None):
    """Gradient-based hard projection to enforce ring closure.

    Minimises the squared distance between N-term N and C-term C atoms by
    adjusting torsion angles via gradient descent. Only used during inference.

    Args:
        nerf: NeRF module for internal->Cartesian conversion.
        tau: (B, L, 7) torsion angles to project (modified in-place).
        aa_types: (B, L) amino acid type indices.
        bonds: (B, L, 14) bond lengths.
        angles: (B, L, 14) bond angles.
        cyclo_mode: (B,) cyclization mode indices (0 = head_to_tail).
        n_steps: number of gradient descent steps.
        lr: learning rate for projection.
        target_dist: target closure distance (default: IDEAL_PEPTIDE_BOND).

    Returns:
        tau: (B, L, 7) projected torsion angles.
    """
    if target_dist is None:
        target_dist = IDEAL_PEPTIDE_BOND

    tau_opt = tau.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([tau_opt], lr=lr)

    # Use enable_grad so this works even when called from @torch.no_grad() context
    with torch.enable_grad():
        for _ in range(n_steps):
            opt.zero_grad()
            coords = nerf(bonds, angles, tau_opt, aa_types)
            # head-to-tail closure distance
            n_term = coords[:, 0, 0]   # (B, 3)
            c_term = coords[:, -1, 2]  # (B, 3)
            dist = torch.norm(n_term - c_term, dim=-1)  # (B,)
            loss = (dist - target_dist).pow(2).mean()
            loss.backward()
            opt.step()
            # Keep angles in [-pi, pi)
            with torch.no_grad():
                tau_opt.data = torch.atan2(
                    torch.sin(tau_opt.data), torch.cos(tau_opt.data)
                )

    return tau_opt.detach()


class Sampler:
    """Reverse diffusion sampler for CyclicDiffusion-MT.

    Samples from noise to clean structure by iteratively denoising with
    the FrameDenoiser, running both the wrapped normal (continuous) and
    masked discrete diffusion reverse processes.
    """

    def __init__(self, config: TrainConfig):
        mc = config.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_encoder = TargetEncoder(
            d_target=mc.d_target, n_blocks=3, d_head=32, n_heads=4, dropout=0.0,
        ).to(self.device)

        self.diffusion_wn = WrappedNormalDiffusion(T=mc.T).to(self.device)
        self.diffusion_md = MaskedDiscreteDiffusion(
            T=mc.T, num_classes=mc.num_aa_types,
        ).to(self.device)

        self.denoiser = FrameDenoiser(
            d_model=mc.d_model, d_target=mc.d_target, d_time=mc.d_time,
            n_blocks=mc.n_blocks, d_head=mc.d_head, n_heads=mc.n_heads,
            dropout=0.0,
        ).to(self.device)

        self.nerf = NeRF().to(self.device)
        self.T = mc.T

    @property
    def model(self):
        return self.denoiser

    def load_checkpoint(self, path):
        """Load trained model weights."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.denoiser.load_state_dict(ckpt["denoiser"])
        self.nerf.load_state_dict(ckpt["nerf"])
        print(f"Loaded checkpoint from {path}")

    @torch.no_grad()
    def sample(self, target_feats, target_masks, num_residues=10, cyclo_mode=0,
               num_steps=None, use_projection=True, temperature=1.0):
        """Sample one cyclic peptide conditioned on target(s).

        Args:
            target_feats: list[(1, N_k, d_target)] pre-encoded target features,
                or list[(1, N_k, 14, 3)] raw coordinates (will auto-encode).
            target_masks: list[(1, N_k)] per-target residue masks.
            num_residues: number of residues L to generate.
            cyclo_mode: cyclization mode index {0..4}.
            num_steps: number of reverse diffusion steps (default: T).
            use_projection: whether to apply hard cyclo projection after each step.
            temperature: sampling temperature for AA type softmax.

        Returns:
            dict with keys: tau, aa_types, coords, dg_pred, aa_logits.
        """
        self.target_encoder.eval()
        self.denoiser.eval()
        self.nerf.eval()

        # Move targets to device
        target_feats = [tf.to(self.device) for tf in target_feats]
        target_masks = [tm.to(self.device) for tm in target_masks]

        if num_steps is None:
            num_steps = self.T

        # Auto-encode targets if raw coordinates provided
        if target_feats and target_feats[0].shape[-1] == 3:
            target_feats = self.target_encoder(target_feats, target_masks)

        B = 1  # sampling one at a time
        L = num_residues
        step_size = self.T // num_steps

        # Start from pure noise
        tau = torch.randn(B, L, 7, device=self.device) * 0.5
        tau = torch.atan2(torch.sin(tau), torch.cos(tau))
        a = torch.full((B, L), self.diffusion_md.mask_idx, device=self.device, dtype=torch.long)
        cyclo = torch.tensor([cyclo_mode], device=self.device, dtype=torch.long)

        # Timesteps from T-1 down to 0
        timesteps = list(range(self.T - 1, -1, -step_size))

        for t_idx in timesteps:
            t = torch.full((B,), t_idx / self.T, device=self.device)

            # Denoise
            tau_pred, aa_logits, dg_pred = self.denoiser(
                tau, a, t, target_feats, target_masks, cyclo,
            )

            # Reverse step for torsions: DDPM x_{t-1} = f(x_t, eps_theta)
            a_bar_t = self.diffusion_wn.alpha_bar[t_idx]
            a_bar_prev = self.diffusion_wn.alpha_bar[max(t_idx - step_size, 0)]
            beta_t = 1 - a_bar_t / (a_bar_prev + 1e-8)
            beta_t = beta_t.clamp(max=0.999)

            # Predicted noise from x_0 prediction
            noise_pred = tau - tau_pred  # simplified; full formula uses alpha_bar

            # Sample x_{t-1}
            sigma_t = beta_t.sqrt()
            z = torch.randn_like(tau) if t_idx > 0 else torch.zeros_like(tau)
            tau = tau_pred + sigma_t.view(-1, 1, 1) * z
            tau = torch.atan2(torch.sin(tau), torch.cos(tau))

            # Discrete reverse step
            a = self.diffusion_md.p_sample(aa_logits, a, t)

            # Hard cyclization projection
            if use_projection:
                aa_for_nerf = torch.where(
                    a == self.diffusion_md.mask_idx,
                    torch.zeros_like(a), a,
                )
                bonds, angles = build_ideal_geometry(aa_for_nerf)
                bonds, angles = bonds.to(self.device), angles.to(self.device)
                tau = hard_cyclo_projection(
                    self.nerf, tau, aa_for_nerf,
                    bonds, angles, cyclo,
                    n_steps=10, lr=0.01,
                )

        # Final Cartesian coordinates
        aa_final = torch.where(
            a == self.diffusion_md.mask_idx,
            torch.zeros_like(a), a,
        )
        bonds, angles = build_ideal_geometry(aa_final)
        bonds, angles = bonds.to(self.device), angles.to(self.device)
        coords = self.nerf(bonds, angles, tau, aa_final)

        return {
            "tau": tau.cpu(),
            "aa_types": a.cpu(),
            "coords": coords.cpu(),
            "dg_pred": dg_pred.cpu() if dg_pred is not None else None,
            "aa_logits": aa_logits.cpu(),
        }


def main():
    parser = argparse.ArgumentParser(description="CyclicDiffusion-MT Sampling")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_residues", type=int, default=10)
    parser.add_argument("--cyclo_mode", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--output", type=str, default="samples.pt")
    args = parser.parse_args()

    path = args.config or os.path.join(
        os.path.dirname(__file__), "config", "config.yaml"
    )
    cfg = load_config(path) if os.path.exists(path) else TrainConfig()

    sampler = Sampler(cfg)
    sampler.load_checkpoint(args.checkpoint)

    # Placeholder: in production, targets would be loaded from PDB files.
    # Here we create dummy targets for demonstration.
    dummy_feats = [torch.randn(1, 30, cfg.model.d_target)]
    dummy_masks = [torch.ones(1, 30, dtype=torch.bool)]

    samples = []
    for i in range(args.num_samples):
        sample = sampler.sample(
            dummy_feats, dummy_masks,
            num_residues=args.num_residues,
            cyclo_mode=args.cyclo_mode,
            num_steps=args.num_steps,
        )
        samples.append(sample)
        print(f"Sample {i+1}/{args.num_samples} complete")

    torch.save(samples, args.output)
    print(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()

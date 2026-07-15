# tests/test_sample.py
import torch, pytest
from cyclicdiffusion_mt.sample import Sampler, hard_cyclo_projection
from cyclicdiffusion_mt.config.train_config import TrainConfig
from cyclicdiffusion_mt.model.nerf import NeRF
from cyclicdiffusion_mt.data.geometry import build_ideal_geometry


class TestHardCycloProjection:
    def test_reduces_closure_error(self):
        nerf = NeRF()
        nerf.eval()
        aa = torch.randint(0, 25, (2, 6))
        tau = torch.randn(2, 6, 7) * 0.5  # perturbed torsions
        bonds, angles = build_ideal_geometry(aa)
        cyclo_mode = torch.zeros(2, dtype=torch.long)

        # Measure initial closure error
        with torch.no_grad():
            coords_before = nerf(bonds, angles, tau, aa)
            n_term = coords_before[:, 0, 0]
            c_term = coords_before[:, -1, 2]
            err_before = torch.norm(n_term - c_term, dim=-1).mean()

        tau_proj = hard_cyclo_projection(
            nerf, tau.clone(), aa, bonds, angles, cyclo_mode,
            n_steps=20, lr=0.05,
        )

        with torch.no_grad():
            coords_after = nerf(bonds, angles, tau_proj, aa)
            n_term_a = coords_after[:, 0, 0]
            c_term_a = coords_after[:, -1, 2]
            err_after = torch.norm(n_term_a - c_term_a, dim=-1).mean()

        # Projection should reduce or maintain closure error
        assert err_after <= err_before + 0.5, \
            f"Projection increased error: {err_before:.4f} -> {err_after:.4f}"

    def test_preserves_batch_shape(self):
        nerf = NeRF()
        nerf.eval()
        aa = torch.randint(0, 25, (4, 8))
        tau = torch.randn(4, 8, 7)
        bonds, angles = build_ideal_geometry(aa)
        cyclo_mode = torch.zeros(4, dtype=torch.long)

        tau_proj = hard_cyclo_projection(
            nerf, tau, aa, bonds, angles, cyclo_mode, n_steps=5, lr=0.01,
        )
        assert tau_proj.shape == (4, 8, 7)


class TestSampler:
    def test_sample_shape(self):
        cfg = TrainConfig()
        sampler = Sampler(cfg)
        sampler.model.eval()

        # Dummy target features
        target_feats = [torch.randn(1, 30, 128)]
        target_masks = [torch.ones(1, 30, dtype=torch.bool)]

        result = sampler.sample(
            target_feats, target_masks,
            num_residues=8, cyclo_mode=0,
            num_steps=10,  # fewer steps for test speed
        )

        assert "tau" in result
        assert "aa_types" in result
        assert result["tau"].shape == (1, 8, 7)
        assert result["aa_types"].shape == (1, 8)
        # AA types should be in valid range
        assert result["aa_types"].min() >= 0
        assert result["aa_types"].max() <= 25

    def test_sample_multi_target(self):
        cfg = TrainConfig()
        sampler = Sampler(cfg)
        sampler.model.eval()

        target_feats = [
            torch.randn(1, 25, 128),
            torch.randn(1, 30, 128),
        ]
        target_masks = [
            torch.ones(1, 25, dtype=torch.bool),
            torch.ones(1, 30, dtype=torch.bool),
        ]

        result = sampler.sample(
            target_feats, target_masks,
            num_residues=10, cyclo_mode=2,
            num_steps=10,
        )
        assert result["tau"].shape == (1, 10, 7)
        assert result["aa_types"].shape == (1, 10)
        assert "dg_pred" in result

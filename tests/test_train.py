"""Smoke test: single training step runs without error."""
import torch, pytest
from cyclicdiffusion_mt.train import TrainingLoop
from cyclicdiffusion_mt.config.train_config import TrainConfig, PhaseConfig, LossWeights
from cyclicdiffusion_mt.data.dataset import MultiTargetDataset, PeptideDataCollate


def _make_dummy_manifest(n_samples=4, L=8, K=2):
    """Build a minimal manifest for smoke testing."""
    manifest = []
    for i in range(n_samples):
        entry = {
            "peptide_torsions": torch.randn(L, 7),
            "peptide_aa_types": torch.randint(0, 25, (L,)),
            "target_coords": [
                torch.randn(torch.randint(20, 40, (1,)).item(), 14, 3)
                for _ in range(K)
            ],
            "target_sequences": [
                torch.randint(0, 25, (torch.randint(20, 40, (1,)).item(),))
                for _ in range(K)
            ],
            "cyclo_mode": i % 5,
            "dG_rosetta": float(torch.randn(1).item()),
            "confidence": float(torch.rand(1).item()),
        }
        manifest.append(entry)
    return manifest


class TestTrainingLoop:
    def test_one_step_runs(self):
        manifest = _make_dummy_manifest()
        dataset = MultiTargetDataset(manifest, max_targets=2)
        collate = PeptideDataCollate()
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate)

        cfg = TrainConfig()
        cfg.phases = [
            PhaseConfig(
                name="pretrain", epochs=1,
                loss_weights=LossWeights(torsion=1.0, type=0.1, cyclo=0.5, affinity=0.1, geometry=0.01),
                max_targets=2, use_cyclo=True, use_affinity=True,
            )
        ]
        loop = TrainingLoop(cfg)
        loop.model.train()

        batch = next(iter(loader))
        loss_dict = loop.training_step(batch)

        assert "total" in loss_dict
        assert "torsion" in loss_dict
        assert torch.isfinite(loss_dict["total"])

    def test_gradient_flows(self):
        manifest = _make_dummy_manifest(n_samples=4)
        dataset = MultiTargetDataset(manifest, max_targets=2)
        collate = PeptideDataCollate()
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate)

        cfg = TrainConfig()
        cfg.phases = [
            PhaseConfig(
                name="pretrain", epochs=1,
                loss_weights=LossWeights(torsion=1.0, type=0.1, cyclo=0.5, affinity=0.1, geometry=0.01),
                max_targets=2, use_cyclo=True, use_affinity=True,
            )
        ]
        loop = TrainingLoop(cfg)
        loop.model.train()

        batch = next(iter(loader))
        loss_dict = loop.training_step(batch)

        loss_dict["total"].backward()
        # Check that at least one parameter received gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in loop.model.parameters()
        )
        assert has_grad, "No parameters received gradients"

    def test_phase_without_cyclo(self):
        """Phase 1 pretrain runs without cyclo/affinity."""
        manifest = _make_dummy_manifest(n_samples=4, K=1)
        dataset = MultiTargetDataset(manifest, max_targets=1)
        collate = PeptideDataCollate()
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate)

        cfg = TrainConfig()
        cfg.phases = [
            PhaseConfig(
                name="pretrain", epochs=1,
                loss_weights=LossWeights(torsion=1.0, type=0.1, cyclo=0.0, affinity=0.0, geometry=0.01),
                max_targets=1, use_cyclo=False, use_affinity=False,
            )
        ]
        loop = TrainingLoop(cfg)
        loop.model.train()

        batch = next(iter(loader))
        loss_dict = loop.training_step(batch)

        assert torch.isfinite(loss_dict["total"])
        assert loss_dict["cyclo"] == 0.0 or loss_dict.get("cyclo") is None

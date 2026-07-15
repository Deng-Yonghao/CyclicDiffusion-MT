# tests/test_config.py
import pytest, tempfile, os
from cyclicdiffusion_mt.config.model_config import ModelConfig
from cyclicdiffusion_mt.config.data_config import DataConfig
from cyclicdiffusion_mt.config.train_config import TrainConfig, PhaseConfig, LossWeights


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.d_model == 256
        assert cfg.d_target == 128
        assert cfg.n_blocks == 6
        assert cfg.T == 500

    def test_override(self):
        cfg = ModelConfig(d_model=128, n_blocks=4)
        assert cfg.d_model == 128
        assert cfg.n_blocks == 4


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.max_residues == 20
        assert cfg.max_targets == 3
        assert cfg.batch_size == 16

    def test_paths(self):
        cfg = DataConfig(data_root="/tmp/data")
        assert cfg.data_root == "/tmp/data"


class TestPhaseConfig:
    def test_fields(self):
        p = PhaseConfig(name="test", epochs=10, loss_weights=LossWeights())
        assert p.name == "test"
        assert p.epochs == 10

    def test_loss_weights_defaults(self):
        lw = LossWeights()
        assert lw.torsion == 1.0
        assert lw.type == 0.1
        assert lw.cyclo == 0.0
        assert lw.affinity == 0.0
        assert lw.geometry == 0.01


class TestTrainConfig:
    def test_default_phases(self):
        cfg = TrainConfig()
        assert len(cfg.phases) == 3  # three-phase training

    def test_phase_names(self):
        cfg = TrainConfig()
        names = [p.name for p in cfg.phases]
        assert names == ["pretrain", "multi_target_cyclo", "affinity_finetune"]


class TestLoadConfig:
    def test_roundtrip(self):
        import yaml
        from cyclicdiffusion_mt.config.train_config import load_config
        cfg = TrainConfig()
        d = cfg.to_dict()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(d, f)
            tmp = f.name
        loaded = load_config(tmp)
        assert loaded.model.d_model == cfg.model.d_model
        assert len(loaded.phases) == 3
        os.unlink(tmp)

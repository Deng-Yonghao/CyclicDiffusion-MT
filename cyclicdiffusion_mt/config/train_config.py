"""Training configuration with three-phase scheduling."""
from dataclasses import dataclass, field
from typing import List
import yaml

from cyclicdiffusion_mt.config.model_config import ModelConfig
from cyclicdiffusion_mt.config.data_config import DataConfig


@dataclass
class LossWeights:
    """Per-loss weight multipliers. Tuned per training phase."""
    torsion: float = 1.0
    type: float = 0.1
    cyclo: float = 0.0
    affinity: float = 0.0
    geometry: float = 0.01

    def to_dict(self):
        return {
            "torsion": self.torsion,
            "type": self.type,
            "cyclo": self.cyclo,
            "affinity": self.affinity,
            "geometry": self.geometry,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PhaseConfig:
    """One training phase."""
    name: str = "pretrain"
    epochs: int = 100
    loss_weights: LossWeights = field(default_factory=LossWeights)
    max_targets: int = 1  # K for this phase
    use_cyclo: bool = False
    use_affinity: bool = False

    def to_dict(self):
        return {
            "name": self.name,
            "epochs": self.epochs,
            "loss_weights": self.loss_weights.to_dict(),
            "max_targets": self.max_targets,
            "use_cyclo": self.use_cyclo,
            "use_affinity": self.use_affinity,
        }

    @classmethod
    def from_dict(cls, d):
        lw = LossWeights.from_dict(d.get("loss_weights", {}))
        return cls(
            name=d.get("name", "pretrain"),
            epochs=d.get("epochs", 100),
            loss_weights=lw,
            max_targets=d.get("max_targets", 1),
            use_cyclo=d.get("use_cyclo", False),
            use_affinity=d.get("use_affinity", False),
        )


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    mixed_precision: bool = True
    warmup_steps: int = 1000
    log_interval: int = 50
    save_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    seed: int = 42
    phases: List[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig(
            name="pretrain",
            epochs=100,
            loss_weights=LossWeights(torsion=1.0, type=0.1, cyclo=0.0, affinity=0.0, geometry=0.01),
            max_targets=1,
            use_cyclo=False,
            use_affinity=False,
        ),
        PhaseConfig(
            name="multi_target_cyclo",
            epochs=50,
            loss_weights=LossWeights(torsion=1.0, type=0.1, cyclo=0.5, affinity=0.0, geometry=0.01),
            max_targets=3,
            use_cyclo=True,
            use_affinity=False,
        ),
        PhaseConfig(
            name="affinity_finetune",
            epochs=30,
            loss_weights=LossWeights(torsion=1.0, type=0.05, cyclo=0.5, affinity=0.1, geometry=0.01),
            max_targets=3,
            use_cyclo=True,
            use_affinity=True,
        ),
    ])

    def to_dict(self):
        return {
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "mixed_precision": self.mixed_precision,
            "warmup_steps": self.warmup_steps,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "seed": self.seed,
            "phases": [p.to_dict() for p in self.phases],
        }


def load_config(path: str) -> TrainConfig:
    """Load YAML config file into TrainConfig dataclass hierarchy."""
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    model = ModelConfig.from_dict(d.get("model", {}))
    data = DataConfig.from_dict(d.get("data", {}))
    phases = [PhaseConfig.from_dict(p) for p in d.get("phases", [])]
    # Start with phases from YAML, fall back to defaults if empty
    if not phases:
        phases = TrainConfig().phases
    return TrainConfig(
        model=model,
        data=data,
        lr=d.get("lr", 1e-4),
        weight_decay=d.get("weight_decay", 1e-5),
        grad_clip=d.get("grad_clip", 1.0),
        mixed_precision=d.get("mixed_precision", True),
        warmup_steps=d.get("warmup_steps", 1000),
        log_interval=d.get("log_interval", 50),
        save_interval=d.get("save_interval", 1000),
        checkpoint_dir=d.get("checkpoint_dir", "./checkpoints"),
        log_dir=d.get("log_dir", "./logs"),
        seed=d.get("seed", 42),
        phases=phases,
    )

"""Model architecture hyperparameters."""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    d_model: int = 256
    d_target: int = 128
    d_time: int = 64
    n_blocks: int = 6
    d_head: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    T: int = 500  # diffusion steps
    num_aa_types: int = 26  # 25 AA + MASK
    max_chi: int = 4

    def to_dict(self):
        return {
            "d_model": self.d_model,
            "d_target": self.d_target,
            "d_time": self.d_time,
            "n_blocks": self.n_blocks,
            "d_head": self.d_head,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "T": self.T,
            "num_aa_types": self.num_aa_types,
            "max_chi": self.max_chi,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

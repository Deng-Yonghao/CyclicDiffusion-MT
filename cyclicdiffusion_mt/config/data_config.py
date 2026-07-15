"""Data pipeline configuration."""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    data_root: str = "./data"
    max_residues: int = 20
    max_targets: int = 3
    max_atoms: int = 14
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    def to_dict(self):
        return {
            "data_root": self.data_root,
            "max_residues": self.max_residues,
            "max_targets": self.max_targets,
            "max_atoms": self.max_atoms,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

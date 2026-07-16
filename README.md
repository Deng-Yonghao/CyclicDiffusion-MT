# CyclicDiffusion-MT

**Multi-Target Full-Atom Cyclic Peptide Generation via Diffusion Models**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

CyclicDiffusion-MT is a diffusion-based generative model that produces **full-atom cyclic peptides** conditioned on **multiple protein target structures**. It jointly generates both the 3D coordinates (backbone + sidechain) and the amino acid sequence, with explicit cyclization constraints and multi-target binding optimization.

<p align="center">
  <img src="https://img.shields.io/badge/status-Implementation%20Complete-brightgreen" alt="Status">
</p>

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Full-Atom Generation** | All backbone + sidechain atoms generated in torsion space via differentiable NeRF |
| **Joint Sequence-Structure** | Unified continuous (wrapped normal) + discrete (masked) diffusion framework |
| **Explicit Cyclization** | Two-level constraint: soft loss during training + hard projection during inference |
| **Multi-Target Conditioning** | Cross-attention fusion with adaptive per-target gating for K ≥ 1 targets |
| **5 Cyclization Modes** | Head-to-tail, sidechain-to-tail, sidechain-to-sidechain, head-to-sidechain, bicyclic |
| **25 Amino Acid Support** | 20 standard + 5 common non-canonical (Orn, D-Ala, β-Ala, NMA, D-Phe) |
| **Three-Phase Training** | Pretrain → multi-target + cyclo → affinity fine-tune |
| **Binding Affinity** | Lightweight Rosetta dG regression head with confidence weighting |

## Architecture Overview

```
Input: K target protein 3D structures
       │
  ┌────┴──────────────────────────────────────────────┐
  │  ① Target Encoder (IPA, 3 blocks, d=128)          │
  │     Shared weights across targets                 │
  └────┬──────────────────────────────────────────────┘
       │
  ┌────┴──────────────────────────────────────────────┐
  │  ② Joint Diffusion (DDPM, T=500, cosine schedule) │
  │     · Wrapped normal diffusion on 7 torsion angles│
  │     · Masked discrete diffusion on AA types       │
  └────┬──────────────────────────────────────────────┘
       │
  ┌────┴──────────────────────────────────────────────┐
  │  ③ Frame-Based Denoiser (N=6 blocks, d=256)      │
  │     · SE(3) frame update via self-attention       │
  │     · Multi-target cross-attention                │
  │     · Cyclization bias injection                  │
  │     · FFN + Residual                              │
  └────┬──────────────────────────────────────────────┘
       │
  ┌────┴──────────────────────────────────────────────┐
  │  ④ Differentiable NeRF                            │
  │     Internal → Cartesian coordinate conversion    │
  └────┬──────────────────────────────────────────────┘
       │
Output: cyclic peptide (τ₀, a₀) → full-atom 3D structure + sequence + dG
```

## Project Structure

```
cyclicdiffusion_mt/
├── config/                      # YAML + dataclass configuration
│   ├── config.yaml              # Main configuration file
│   ├── model_config.py          # Model hyperparameters
│   ├── data_config.py           # Data pipeline settings
│   └── train_config.py          # Training config with 3-phase scheduling
├── data/                        # Data pipeline
│   ├── dataset.py               # Multi-target dataset + collate
│   ├── synthetic.py             # Synthetic multi-target data builder
│   ├── geometry.py              # Ideal bond/angle tensor builders
│   └── transforms.py            # Masks, coordinate transforms
├── model/                       # Core model modules
│   ├── diffusion.py             # DDPM processes (wrapped normal + discrete)
│   ├── denoiser.py              # SE(3) frame-based denoiser
│   ├── target_encoder.py        # IPA target structure encoder
│   ├── cross_attention.py       # Multi-target cross-attention + gating
│   └── nerf.py                  # Differentiable internal↔Cartesian NeRF
├── losses/                      # Loss functions
│   ├── torsion_loss.py          # Wrapped L2 torsion loss
│   ├── type_loss.py             # Cross-entropy AA type loss
│   ├── cyclo_loss.py            # Ring closure distance loss
│   ├── geometry_loss.py         # Clash, Ramachandran, rotamer losses
│   └── affinity_loss.py         # Confidence-weighted dG regression loss
├── eval/                        # Evaluation
│   └── metrics.py               # Structure, binding & generation metrics
├── utils/                       # Utilities
│   └── constants.py             # AA vocabulary, atom mappings, constants
├── train.py                     # Three-phase training entry point
└── sample.py                    # Reverse diffusion sampling/inference
tests/                           # Comprehensive test suite
CPCore/                          # Reference cyclic peptide dataset
docs/                            # Design specs and implementation plans
```

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA-capable GPU recommended (single RTX 4090 sufficient)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/CyclicDiffusion-MT.git
cd CyclicDiffusion-MT

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Training

```bash
# Three-phase training with default config
python -m cyclicdiffusion_mt.train --config cyclicdiffusion_mt/config/config.yaml
```

The training proceeds through three phases:
1. **Phase 1 (Pretrain)**: Single-target conditioning, no cyclization/affinity loss
2. **Phase 2 (Multi-target + Cyclo)**: Multi-target cross-attention + ring closure loss
3. **Phase 3 (Affinity Fine-tune)**: Rosetta dG regression with confidence weighting

### Sampling / Inference

```bash
# Generate novel cyclic peptides conditioned on target structures
python -m cyclicdiffusion_mt.sample \
    --checkpoint checkpoints/phase3_complete.pt \
    --num_residues 10 \
    --cyclo_mode 0 \
    --num_samples 5 \
    --output generated_samples.pt
```

### Running Tests

```bash
pytest tests/ -v
```

## Model Configuration

Key hyperparameters (configurable in `config/config.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 256 | Hidden dimension for denoiser |
| `d_target` | 128 | Target encoder output dimension |
| `n_blocks` | 6 | Number of denoiser blocks |
| `T` | 500 | Diffusion timesteps |
| `max_residues` | 20 | Maximum peptide length |
| `num_aa_types` | 26 | 25 amino acids + mask token |
| `max_chi` | 4 | Maximum sidechain torsion angles |

## Loss Function

The total loss is a weighted sum of five components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{torsion}} + \lambda_1\mathcal{L}_{\text{type}} + \lambda_2\mathcal{L}_{\text{cyclo}} + \lambda_3\mathcal{L}_{\text{affinity}} + \lambda_4\mathcal{L}_{\text{geometry}}$$

| Loss | Form | Purpose |
|------|------|---------|
| $\mathcal{L}_{\text{torsion}}$ | Wrapped L2: $\|\varepsilon - \hat{\varepsilon}_\theta\|^2$ | Core diffusion denoising |
| $\mathcal{L}_{\text{type}}$ | Cross-entropy: $-\log p_\theta(a_0 \mid a_t, \tau_t, t)$ | Amino acid type prediction |
| $\mathcal{L}_{\text{cyclo}}$ | $\|\text{dist}(N, C) - d_{\text{bond}}\|^2$ | Ring closure constraint |
| $\mathcal{L}_{\text{geometry}}$ | Clash + Rama + Rotamer | Structural plausibility |
| $\mathcal{L}_{\text{affinity}}$ | $\|f_\phi(\hat{\tau}_0) - \Delta G_{\text{Rosetta}}\|^2$ | Binding affinity signal |

Loss weights adjust dynamically across the three training phases.

## Data Strategy

The model uses a hierarchical data approach:

| Tier | Source | Scale | Usage |
|------|--------|-------|-------|
| Tier 1 | Cyclic peptide-target co-crystal structures | ~10²–10³ | Base training + validation |
| Tier 2 | Linear peptide-protein co-crystal structures | ~10³–10⁴ | Transfer learning pretrain |
| Tier 3 | Synthetic multi-target (docking + alignment) | ~10⁴–10⁵ | Multi-target main training |

Synthetic data is quality-filtered using a composite confidence score:
$$c = c_{\text{clash}} \times c_{\text{contact}} \times c_{\text{docking}}$$

## Evaluation Metrics

### Structure Quality
- Ring closure precision (N–C distance error)
- Ramachandran outlier rate
- Steric clash count per residue
- Bond angle/length deviation from ideality

### Generation Quality
- Internal diversity (pairwise torsion RMSD)
- Novelty (nearest-neighbor similarity to training set)
- Uniqueness (% unique among generated samples)

### Binding Capability
- Predicted Rosetta ΔG
- Contact surface area estimation
- Multi-target affinity balance score

## Technical Stack

| Component | Choice |
|-----------|--------|
| Deep learning | PyTorch 2.x |
| Protein parsing | Biopython |
| Internal coordinates | Self-implemented NeRF (PyTorch, fully differentiable) |
| Structure validation | RDKit |
| Offline scoring | Rosetta |
| Configuration | YAML + Python dataclasses |
| Experiment tracking | TensorBoard / Weights & Biases |

## Reference Dataset

The `CPCore/` directory contains a curated dataset of cyclic peptides with experimentally characterized structures and properties, including:
- PDB structures of cyclic peptides
- Binding affinity measurements
- Clustering and validity annotations

## Citation

If you use CyclicDiffusion-MT in your research, please cite:

```bibtex
@article{cyclicdiffusion-mt,
  title={CyclicDiffusion-MT: Multi-Target Full-Atom Cyclic Peptide Diffusion},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Project Status

- ✅ Design specification complete
- ✅ Core implementation complete (model, losses, data pipeline)
- ✅ Training loop with three-phase scheduling
- ✅ Reverse diffusion sampling with hard cyclo projection
- ✅ Evaluation metrics
- ⏳ Real data preprocessing pipeline
- ⏳ Full training run and benchmarking
- ⏳ Paper writing

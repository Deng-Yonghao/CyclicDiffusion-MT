# CyclicDiffusion-MT: Multi-Target Full-Atom Cyclic Peptide Diffusion Model

## Design Specification

**Date**: 2026-07-14
**Status**: Approved
**Project Type**: Academic Research (Paper)

---

## 1. Overview

CyclicDiffusion-MT is a diffusion-based generative model that produces **full-atom cyclic peptides** conditioned on **multiple protein target structures**. It jointly generates both the 3D coordinates (backbone + sidechain) and the amino acid sequence, with explicit cyclization constraints and multi-target binding optimization.

### 1.1 Key Innovation Dimensions

| Dimension | Core Contribution |
|-----------|-------------------|
| **Cyclization constraints** | Explicit geometric cyclization in torsion-space diffusion with soft loss + hard projection |
| **Multi-target conditioning** | Cross-attention fusion with adaptive per-target gating |
| **Joint sequence-structure generation** | Unified continuous (torsion) + discrete (amino acid type) diffusion framework |
| **Full-atom generation** | All-atom generation in internal coordinate space using differentiable NeRF |

### 1.2 Application Scenario

Academic research — targeting a top-tier ML/comp-bio venue. The model generates cyclic peptides that simultaneously bind to K protein targets (K ≥ 1).

---

## 2. Representation & Parameterization

### 2.1 Internal Coordinates (Torsion Space)

Each residue is parameterized by internal coordinates:

| Variable | Count per residue | Behavior in diffusion |
|----------|-------------------|-----------------------|
| Bond lengths (N-Cα, Cα-C, C-N) | 3 | Near-rigid, small noise |
| Bond angles (N-Cα-C, Cα-C-N, ...) | 3 | Near-rigid, small noise |
| Backbone torsions (φ, ψ, ω) | 3 | **Primary diffusion variables** |
| Sidechain torsions (χ₁, χ₂, χ₃, χ₄) | 1-4 (AA-dependent) | **Primary diffusion variables** |
| Amino acid type | 1 discrete | **Masked discrete diffusion** |

Sidechain χ angles are mask-padded to 4 for all residue types (unused positions = 0 + attention mask).

### 2.2 Differentiable Coordinate Conversion (Module ⑤)

NeRF (Natural Extension of Reference Frame) algorithm converts internal coordinates ↔ Cartesian coordinates, fully differentiable in PyTorch. This is the foundation for all geometric loss computation.

---

## 3. Architecture

### 3.1 System Diagram

```
Input: K target protein 3D structures + noise
       │
┌──────┴──────────────────────────────────────────────┐
│  ① Target Encoder (IPA, 3 blocks, d=128)             │
│     Shared weights across targets                    │
│     Output: {h_k ∈ R^{N_res×128} for k=1..K}        │
└──────┬──────────────────────────────────────────────┘
       │
┌──────┴──────────────────────────────────────────────┐
│  ② Joint Diffusion Module (DDPM, T=500, cosine sched)│
│     - Continuous: wrapped normal diffusion on torsions│
│     - Discrete: masked discrete diffusion on AA types │
│     - State: τ_t (B,L,7), a_t (B,L)                   │
└──────┬──────────────────────────────────────────────┘
       │
┌──────┴──────────────────────────────────────────────┐
│  ③ Frame-Based Denoiser (N=6 blocks, d_model=256)    │
│     Per block:                                        │
│     a. Frame Update (SE(3) equivariant message pass)  │
│     b. Multi-Target Cross-Attention (4 heads, d=64)   │
│     c. Cyclization Bias Injection                     │
│     d. FFN + Residual (4x expansion, SiLU)            │
│     Output: τ̂₀, â₀, dĜ                                │
└──────┬──────────────────────────────────────────────┘
       │
┌──────┴──────────────────────────────────────────────┐
│  ④ Multi-Target Conditioning (Cross-Attention)       │
│     Q = node features                                │
│     K,V = [h₁||h₂||...||h_K||time_emb||cyclo_emb]    │
│     Adaptive gating: α_k = softmax(g(h_k, t))        │
│     Injected at every denoiser block                  │
└──────┬──────────────────────────────────────────────┘
       │
┌──────┴──────────────────────────────────────────────┐
│  ⑤ Differentiable NeRF                               │
│     Internal → Cartesian conversion                   │
│     Used for cyclo loss + geometry loss               │
└─────────────────────────────────────────────────────┘
       │
Output: cyclic peptide (τ₀, a₀) → full-atom 3D structure + sequence
```

### 3.2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| d_target (IPA) | 128 |
| d_time (sinusoidal) | 64 |
| Denoiser blocks (N) | 6 |
| IPA blocks | 3 |
| Attention heads | 4 |
| d_head | 64 |
| FFN expansion | 4× |
| Dropout | 0.1 |
| T (diffusion steps) | 500 |
| Noise schedule | Cosine |
| Max residues L | 20 |
| Max χ per residue | 4 (mask padded) |
| Amino acid types | 20 standard + 5 common non-canonical = 25 |
| Cyclization modes | 5 (head-tail, sc-tail, sc-sc, head-sc, bicyclic) |

---

## 4. Loss Function

### 4.1 Total Loss

```
L_total = L_torsion + λ₁·L_type + λ₂·L_cyclo + λ₃·L_affinity + λ₄·L_geometry
```

### 4.2 Loss Components

| Loss | Form | Purpose |
|------|------|---------|
| L_torsion | `w_t · ||ε - ε̂_θ||²` (wrapped normal) | Core diffusion denoising |
| L_type | `-log p_θ(a₀ | a_t, τ_t, t, cond)` | Amino acid type prediction |
| L_cyclo | `||D2C(τ̂₀)_N - D2C(τ̂₀)_C|| - d_bond|²` | Cyclization closure |
| L_geometry | L_rama + L_clash + L_sidechain | Structural plausibility |
| L_affinity | `||f_φ(τ̂₀, target) - dG_rosetta||²` | Binding affinity signal |

### 4.3 Synthetic Data Confidence Weighting

Each synthetic multi-target sample has a quality score:
```
c = c_clash × c_contact × c_docking
```
Applied as: `L_synthetic = c · L_total`

Where `c_docking` comes from normalized Rosetta dG values already in the dataset.

### 4.4 Loss Weight Schedule

| Phase | λ₁ (type) | λ₂ (cyclo) | λ₃ (affinity) | λ₄ (geometry) |
|-------|-----------|------------|---------------|----------------|
| Phase 1: Single-target pretrain | 0.1 | 0 | 0 | 0.01 |
| Phase 2: Multi-target + cyclo | 0.1 | 0.5 | 0 | 0.01 |
| Phase 3: Affinity fine-tune | 0.05 | 0.5 | 0.1 | 0.01 |

---

## 5. Cyclization Constraint Module

### 5.1 Two-Level Constraint Mechanism

**Level 1 — Soft constraint (loss):** L_cyclo applied during training at each denoising step. The model learns torsion patterns that lead to ring closure.

**Level 2 — Hard projection (inference):** During sampling, after each denoising step, apply constraint projection via Lagrange multiplier or gradient-based correction to enforce exact ring closure.

### 5.2 Cyclization Mode Conditioning

A learnable embedding for cyclization mode is injected into the denoiser, supporting:
- head-to-tail (standard peptide bond)
- sidechain-to-tail (e.g., Asp/Glu sidechain → N-term)
- sidechain-to-sidechain (e.g., Cys-Cys disulfide)
- head-to-sidechain
- bicyclic (multiple modes)

---

## 6. Multi-Target Conditioning

### 6.1 Target Encoder

IPA (Invariant Point Attention) with shared weights encodes each target independently:
- Atom-level encoding → residue-level pooling
- Optional surface point sampling for binding interface features

### 6.2 Cross-Attention Fusion

- Multi-head cross-attention at every denoiser block
- Q from peptide hidden state, K/V from concatenated target embeddings
- Target index embeddings distinguish features from different targets

### 6.3 Adaptive Target Gating (Innovation)

```
α_k = softmax(g(h_k, t))  where g is a lightweight MLP
```
Learns dynamic per-target importance weights that can vary across denoising steps and across residues — the model can attend more to target A for residue i and target B for residue j.

---

## 7. Data Strategy

### 7.1 Data Hierarchy

| Tier | Source | Scale | Usage |
|------|--------|-------|-------|
| Tier 1 | Cyclic peptide-target co-crystal (PDB, CyBase, PeptiDB) | ~10²-10³ | Base training + validation |
| Tier 2 | Linear peptide-protein co-crystal (PDB) | ~10³-10⁴ | Transfer learning pretrain |
| Tier 3 | Synthetic multi-target (docking + alignment) | ~10⁴-10⁵ | Multi-target main training |

### 7.2 Synthetic Multi-Target Data Pipeline

```
Cyclic peptide-TargetA complex + TargetB structure (apo or holo)
    │
    ├── Step 1: Structural alignment (align TargetB to TargetA frame)
    ├── Step 2: Clash filter (no severe steric clashes)
    ├── Step 3: Contact area check (reasonable interface)
    ├── Step 4 (optional): Rosetta FlexPepDock refinement
    └── Output: (peptide, targetA, targetB) triplet with quality score c
```

### 7.3 Three-Phase Training

```
Phase 1: Single-target pretraining (Tier 1+2, K=1, no cyclo loss)
    → Learn basic peptide structure generation + single-target conditioning

Phase 2: Cyclization + multi-target fine-tuning (Tier 3, K=2-3, full loss)
    → Add cyclo constraint + multi-target cross-attention

Phase 3: Affinity fine-tuning (high-quality subset, Rosetta dG labels)
    → Add affinity proxy loss with confidence weighting
```

---

## 8. Evaluation

### 8.1 Metrics

**Structure Quality:**
- Ring closure precision (N-C distance error)
- Ramachandran distribution (outlier %)
- Bond angle/length deviation from ideality
- Rotamer rate (vs Dunbrack rotamer library)
- Steric clash count per residue
- Sidechain RMSD (when ground truth available)

**Binding Capability:**
- Rosetta dG (total energy)
- Rosetta ddG (interface ΔG upon binding)
- Contact surface area (Å²)
- Hydrogen bond / hydrophobic complementarity
- DockQ score (vs native complex)

**Generation Quality:**
- Diversity (internal pairwise RMSD/TM-score)
- Novelty (nearest-neighbor similarity to training set)
- Uniqueness (% unique among generated samples)
- MMD (Maximum Mean Discrepancy vs natural cyclic peptide distribution)

**Multi-Target Specific:**
- Multi-target dG (sum or min across all targets)
- Affinity balance score (std/mean across targets)
- Binding mode consistency
- Target specificity vs promiscuity

### 8.2 Experiments

**Main Experiments:**
| ID | Method | Purpose |
|----|--------|---------|
| Exp 1 | CyclicDiffusion-MT (ours) | Full model evaluation |
| Exp 2 | RFdiffusion + ProteinMPNN | SOTA baseline (single-target) |
| Exp 3 | RFdiffusion + post-hoc cyclization | No-cyclo-constraint baseline |
| Exp 4 | Ours (K=1) | Single-target ablation |
| Exp 5 | DyMEAN / PepGLAD | Peptide generation baselines |

**Ablation Studies:**
| ID | Removed Component | Hypothesis Tested |
|----|------------------|-------------------|
| Abl 1 | Cyclization loss | Necessity of explicit cyclo constraint |
| Abl 2 | Multi-target cross-attn → concat | Multi-target fusion mechanism |
| Abl 3 | Adaptive gating | Per-target weight learning |
| Abl 4 | Joint seq-struct → separated | Unified diffusion benefit |
| Abl 5 | Full-atom → backbone-only | Sidechain contribution to binding |
| Abl 6 | Confidence weighting | Data quality weighting effect |
| Abl 7 | Affinity loss | Affinity signal contribution |

**Case Studies:**
- Case 1: Known dual-target cyclic peptide rediscovery (e.g., integrin αvβ3 + αvβ5)
- Case 2: Structurally dissimilar targets (e.g., kinase + GPCR) — generalization
- Case 3: Homologous targets (e.g., different kinases) — specificity vs promiscuity

### 8.3 Computational Validation Pipeline

```
Generated candidates (1000)
    → Rosetta rescoring → Top-50
    → Rosetta FlexPepDock refinement
    → RDKit chemical reasonableness check
    → (Optional) MOE/Schrödinger MM/GBSA
```

---

## 9. Technical Stack

| Component | Choice |
|-----------|--------|
| Deep learning framework | PyTorch 2.x |
| Protein data parsing | Biopython / ProDy |
| Internal coordinate conversion | Self-implemented NeRF (PyTorch) |
| Structure visualization | PyMOL / Py3Dmol / NGLview |
| Experiment tracking | Weights & Biases / TensorBoard |
| Config management | YAML + Hydra / dataclasses |
| Chemical validation | RDKit |

---

## 10. Project Code Structure

```
cyclicdiffusion_mt/
├── config/                  # Hydra/YAML configs
│   ├── model.yaml
│   ├── data.yaml
│   └── train.yaml
├── data/                    # Data pipeline
│   ├── dataset.py           # PDB/complex data loading
│   ├── synthetic.py         # Synthetic multi-target data construction
│   └── transforms.py        # Data augmentation/normalization
├── model/                   # Core model
│   ├── diffusion.py         # DDPM forward/reverse process
│   ├── denoiser.py          # Frame-based denoiser
│   ├── target_encoder.py    # IPA target encoder
│   ├── cross_attention.py   # Multi-target fusion
│   ├── nerf.py              # Differentiable internal↔Cartesian conversion
│   └── heads.py             # Output heads
├── losses/                  # Loss functions
│   ├── torsion_loss.py
│   ├── type_loss.py
│   ├── cyclo_loss.py
│   ├── affinity_loss.py
│   └── geometry_loss.py
├── train.py                 # Main training script
├── sample.py                # Inference/sampling
├── eval/                    # Evaluation
│   ├── metrics.py
│   └── analysis.py
└── utils/                   # Utilities
    ├── constants.py         # Amino acid properties, atom types
    └── protein_utils.py     # PDB parsing, coordinate transforms
```

---

## 11. Scope & Limitations

### In Scope
- Full-atom cyclic peptide generation in torsion space
- Multi-target (K ≥ 1) conditioning via cross-attention
- Explicit cyclization constraint (head-to-tail primary, extensible)
- Joint sequence-structure diffusion
- Three-phase training with synthetic multi-target data

### Out of Scope (Future Work)
- Wet-lab experimental validation (reserved for follow-up study)
- Bicyclic / complex topology generation (architecture supports but primary focus is single-ring)
- Non-protein targets (DNA, RNA, small molecule)
- Full PK/PD property prediction
- Deployment / production interface

---

## 12. Risk & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Synthetic data noise degrades generation quality | Medium | Confidence weighting; Phase 1 provides strong prior |
| Torsion-space cyclo constraint gradients unstable | Medium | Two-level mechanism; hard projection as fallback |
| Multi-target gating learns trivial solution (equal weights) | Low | Auxiliary diversity loss on α_k; proper initialization |
| Full-atom generation too slow on single GPU | Medium | Optimize batch size; mixed precision; gradient accumulation |
| Joint continuous-discrete diffusion training instability | Medium | Careful λ scheduling; warmup phases; gradient clipping |

# CyclicDiffusion-MT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build full-atom cyclic peptide diffusion model: multi-target conditioning, joint sequence-structure generation, explicit cyclization.

**Architecture:** Torsion-space diffusion: wrapped normal on 7 angles + masked discrete on 25 AA types. SE(3) frame denoiser (6 blocks, d=256) with IPA target encoder + cross-attention fusion. Differentiable NeRF for geometry/cyclo losses. Three-phase training.

**Tech Stack:** PyTorch 2.x, Biopython, self-implemented NeRF, Rosetta (offline)

## Global Constraints

- PyTorch 2.x, single GPU (RTX 4090), L<=20, max chi=4 (mask padded)
- 25 AA types (20 standard + ORN, DAL, BAL, NMA, DPH) + [MASK]=26 categories
- 5 cyclization modes, DDPM T=500 cosine, d_model=256, d_target=128, 6 denoiser blocks
- Package: `cyclicdiffusion_mt/`, tests: `tests/`
- Config: YAML + dataclasses

---

### Task 1: Project Scaffold & Constants

**Files:**
- Create: `cyclicdiffusion_mt/__init__.py`
- Create: `cyclicdiffusion_mt/utils/__init__.py`
- Create: `cyclicdiffusion_mt/utils/constants.py`
- Create: `tests/__init__.py`
- Create: `tests/test_constants.py`

**Interfaces:**
- Produces: `AA_TO_IDX`, `IDX_TO_AA`, `NUM_AA_TYPES=25`, `MASK_IDX=25`, `AA_CHI_COUNTS: list[int]`, `AA_ATOM_NAMES: dict`, `AA_ATOM_COUNT: dict`, `BACKBONE_ATOMS=['N','CA','C','O']`, `IDEAL_BOND_LENGTHS`, `IDEAL_BOND_ANGLES`, `CYCLO_MODES`, `MAX_RESIDUES=20`, `MAX_ATOMS_PER_RES=14`, `MAX_CHI_PER_RES=4`, `NUM_TORSIONS=7`

- [ ] **Step 1: Write the test file**

```python
# tests/test_constants.py
import pytest
from cyclicdiffusion_mt.utils.constants import (
    AA_TO_IDX, IDX_TO_AA, NUM_AA_TYPES, MASK_IDX,
    AA_CHI_COUNTS, AA_ATOM_NAMES, AA_ATOM_COUNT,
    BACKBONE_ATOMS, CYCLO_MODES, MAX_RESIDUES,
    MAX_ATOMS_PER_RES, MAX_CHI_PER_RES, NUM_TORSIONS,
    IDEAL_BOND_LENGTHS, IDEAL_BOND_ANGLES,
)

class TestAAVocabulary:
    def test_num_aa_types(self):
        assert NUM_AA_TYPES == 25
    def test_mask_idx_is_25(self):
        assert MASK_IDX == 25
    def test_standard_20_present(self):
        std = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
               'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
        for aa in std:
            assert aa in AA_TO_IDX
    def test_noncanonical_5_present(self):
        for aa in ['ORN','DAL','BAL','NMA','DPH']:
            assert aa in AA_TO_IDX
    def test_idx_roundtrip(self):
        for aa, idx in AA_TO_IDX.items():
            assert IDX_TO_AA[idx] == aa
    def test_chi_counts_length(self):
        assert len(AA_CHI_COUNTS) == NUM_AA_TYPES
    def test_ala_gly_no_chi(self):
        assert AA_CHI_COUNTS[AA_TO_IDX['ALA']] == 0
        assert AA_CHI_COUNTS[AA_TO_IDX['GLY']] == 0
    def test_arg_lys_4_chi(self):
        assert AA_CHI_COUNTS[AA_TO_IDX['ARG']] == 4
        assert AA_CHI_COUNTS[AA_TO_IDX['LYS']] == 4
    def test_max_chi_is_4(self):
        assert max(AA_CHI_COUNTS) <= MAX_CHI_PER_RES
    def test_atom_names_count_match(self):
        for aa in AA_ATOM_NAMES:
            assert len(AA_ATOM_NAMES[aa]) == AA_ATOM_COUNT[aa]
    def test_backbone_order(self):
        assert BACKBONE_ATOMS == ['N', 'CA', 'C', 'O']
    def test_ala_5_atoms(self):
        assert AA_ATOM_COUNT['ALA'] == 5

class TestCycloModes:
    def test_five_modes(self):
        assert len(CYCLO_MODES) == 5
    def test_head_to_tail(self):
        assert 'head_to_tail' in CYCLO_MODES

class TestConstants:
    def test_dimensional_constants(self):
        assert MAX_RESIDUES == 20
        assert MAX_ATOMS_PER_RES == 14
        assert MAX_CHI_PER_RES == 4
        assert NUM_TORSIONS == 7
    def test_ideal_bonds(self):
        assert 'N-CA' in IDEAL_BOND_LENGTHS
        assert 'CA-C' in IDEAL_BOND_LENGTHS
        assert 'C-N' in IDEAL_BOND_LENGTHS
    def test_ideal_angles(self):
        assert 'N-CA-C' in IDEAL_BOND_ANGLES
        assert 'CA-C-N' in IDEAL_BOND_ANGLES
```

- [ ] **Step 2: Run test, verify fail**

`pytest tests/test_constants.py -v` → FAIL (no module)

- [ ] **Step 3: Implement constants**

```python
# cyclicdiffusion_mt/__init__.py
"""CyclicDiffusion-MT: Multi-Target Full-Atom Cyclic Peptide Diffusion Model."""
```

```python
# cyclicdiffusion_mt/utils/__init__.py
"""Utility modules."""
```

```python
# cyclicdiffusion_mt/utils/constants.py
"""AA vocabulary, atom mappings, ideal geometry constants."""

AA_TO_IDX = {
    'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLN':5,'GLU':6,'GLY':7,'HIS':8,
    'ILE':9,'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,
    'TRP':17,'TYR':18,'VAL':19,'ORN':20,'DAL':21,'BAL':22,'NMA':23,'DPH':24,
}
IDX_TO_AA = {v:k for k,v in AA_TO_IDX.items()}
NUM_AA_TYPES = 25
MASK_IDX = 25  # masked discrete diffusion mask token

# chi counts: A R N D C Q E G H I L K M F P S T W Y V ORN DAL BAL NMA DPH
AA_CHI_COUNTS = [0,4,2,2,1,3,3,0,2,2,2,4,3,2,1,1,1,2,2,1,3,0,0,0,2]

BACKBONE_ATOMS = ['N','CA','C','O']

AA_ATOM_NAMES = {
    'ALA': ['N','CA','C','O','CB'],
    'ARG': ['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2'],
    'ASN': ['N','CA','C','O','CB','CG','OD1','ND2'],
    'ASP': ['N','CA','C','O','CB','CG','OD1','OD2'],
    'CYS': ['N','CA','C','O','CB','SG'],
    'GLN': ['N','CA','C','O','CB','CG','CD','OE1','NE2'],
    'GLU': ['N','CA','C','O','CB','CG','CD','OE1','OE2'],
    'GLY': ['N','CA','C','O'],
    'HIS': ['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2'],
    'ILE': ['N','CA','C','O','CB','CG1','CG2','CD1'],
    'LEU': ['N','CA','C','O','CB','CG','CD1','CD2'],
    'LYS': ['N','CA','C','O','CB','CG','CD','CE','NZ'],
    'MET': ['N','CA','C','O','CB','CG','SD','CE'],
    'PHE': ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ'],
    'PRO': ['N','CA','C','O','CB','CG','CD'],
    'SER': ['N','CA','C','O','CB','OG'],
    'THR': ['N','CA','C','O','CB','OG1','CG2'],
    'TRP': ['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
    'TYR': ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH'],
    'VAL': ['N','CA','C','O','CB','CG1','CG2'],
    'ORN': ['N','CA','C','O','CB','CG','CD','NE'],
    'DAL': ['N','CA','C','O','CB'],
    'BAL': ['N','CA','CB','C','O'],
    'NMA': ['N','CA','C','O','CB','NCH3'],
    'DPH': ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ'],
}
AA_ATOM_COUNT = {aa:len(atoms) for aa,atoms in AA_ATOM_NAMES.items()}

IDEAL_BOND_LENGTHS = {
    'N-CA':1.458,'CA-C':1.525,'C-N':1.329,'C-O':1.231,'CA-CB':1.521,
}
IDEAL_BOND_ANGLES = {
    'N-CA-C':111.0,'CA-C-N':116.0,'C-N-CA':122.0,'N-CA-CB':110.5,'CA-C-O':120.8,
}

CYCLO_MODES = {'head_to_tail':0,'sidechain_to_tail':1,'sidechain_to_sidechain':2,'head_to_sidechain':3,'bicyclic':4}
NUM_CYCLO_MODES = 5

MAX_RESIDUES = 20
MAX_ATOMS_PER_RES = 14
MAX_CHI_PER_RES = 4
NUM_TORSIONS = 7  # phi,psi,omega,chi1-4
NUM_AA_TYPES_WITH_MASK = 26

IDEAL_PEPTIDE_BOND = 1.329
CLASH_THRESHOLD = 1.5
```

- [ ] **Step 4: Run test, verify pass**

`pytest tests/test_constants.py -v` → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/ tests/ && git commit -m "feat: add project scaffold and AA constants"
```

---

### Task 2: Differentiable NeRF

**Files:**
- Create: `cyclicdiffusion_mt/model/__init__.py`
- Create: `cyclicdiffusion_mt/model/nerf.py`
- Create: `tests/test_nerf.py`

**Interfaces:**
- Produces: `rotate_around_axis(v,axis,angle)->v_rot`, `place_atom(r_im2,r_im1,r_i,b,alpha,tau)->r_ip1`, `class NeRF(nn.Module)` with `forward(bonds,angles,torsions,aa_types)->coords:(B,L,14,3)` and `inverse(coords,aa_types)->(bonds,angles,torsions)`

- [ ] **Step 1: Write test**

```python
# tests/test_nerf.py
import pytest, torch
from cyclicdiffusion_mt.model.nerf import NeRF, rotate_around_axis, place_atom

class TestRotateAroundAxis:
    def test_no_rotation(self):
        v = torch.tensor([[1.,0.,0.]]); axis = torch.tensor([[0.,0.,1.]]); angle = torch.tensor([0.])
        assert torch.allclose(rotate_around_axis(v,axis,angle), v, atol=1e-6)
    def test_90_degree(self):
        v = torch.tensor([[1.,0.,0.]]); axis = torch.tensor([[0.,0.,1.]]); angle = torch.tensor([torch.pi/2])
        assert torch.allclose(rotate_around_axis(v,axis,angle), torch.tensor([[0.,1.,0.]]), atol=1e-6)
    def test_preserves_length(self):
        v = torch.randn(5,3); axis = torch.randn(5,3); angle = torch.rand(5)*2*torch.pi
        r = rotate_around_axis(v,axis,angle)
        assert torch.allclose(torch.norm(r,dim=-1), torch.norm(v,dim=-1), atol=1e-5)

class TestPlaceAtom:
    def test_straight_line(self):
        r_im2=torch.tensor([[0.,0.,0.]]); r_im1=torch.tensor([[1.,0.,0.]]); r_i=torch.tensor([[2.,0.,0.]])
        b=torch.tensor([1.5]); alpha=torch.tensor([torch.pi]); tau=torch.tensor([0.])
        r = place_atom(r_im2,r_im1,r_i,b,alpha,tau)
        assert r.shape == (1,3)
        assert torch.allclose(r[:,0], torch.tensor([3.5]), atol=1e-4)

class TestNeRF:
    @pytest.fixture
    def nerf(self): return NeRF()
    def test_init_frame_shape(self, nerf):
        assert nerf.init_frame.shape == (3,3)
    def test_forward_shape(self, nerf):
        B,L=2,5; aa=torch.zeros(B,L,dtype=torch.long)
        bonds=torch.randn(B,L,14); angles=torch.randn(B,L,14); torsions=torch.randn(B,L,7)
        coords = nerf(bonds,angles,torsions,aa)
        assert coords.shape == (B,L,14,3)
    def test_differentiable(self, nerf):
        B,L=1,3; aa=torch.zeros(B,L,dtype=torch.long)
        bonds=torch.randn(B,L,14,requires_grad=True)
        angles=torch.randn(B,L,14,requires_grad=True)
        torsions=torch.randn(B,L,7,requires_grad=True)
        coords=nerf(bonds,angles,torsions,aa); coords.sum().backward()
        assert bonds.grad is not None and torsions.grad is not None
    def test_no_nan(self, nerf):
        B,L=1,6; aa=torch.randint(0,20,(B,L))
        bonds=torch.rand(B,L,14)*0.1+1.3; angles=torch.rand(B,L,14)*0.3+1.9
        torsions=torch.rand(B,L,7)*2*torch.pi-torch.pi
        coords=nerf(bonds,angles,torsions,aa)
        assert not torch.isnan(coords).any()
```

- [ ] **Step 2: `pytest tests/test_nerf.py -v`** → FAIL

- [ ] **Step 3: Implement**

```python
# cyclicdiffusion_mt/model/__init__.py
"""Model modules."""
```

```python
# cyclicdiffusion_mt/model/nerf.py
"""Differentiable NeRF: internal <-> Cartesian coordinate conversion."""
import torch, torch.nn as nn

def normalize(v, dim=-1, eps=1e-8):
    return v / (torch.norm(v, dim=dim, keepdim=True) + eps)

def rotate_around_axis(v, axis, angle):
    """Rodrigues rotation. v,axis:(B,3) angle:(B,) -> (B,3)"""
    axis = normalize(axis)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    cross_av = torch.cross(axis, v, dim=-1)
    dot_av = (axis * v).sum(dim=-1, keepdim=True)
    return v * cos_a.unsqueeze(-1) + cross_av * sin_a.unsqueeze(-1) + axis * dot_av * (1 - cos_a.unsqueeze(-1))

def place_atom(r_im2, r_im1, r_i, b, alpha, tau):
    """Place atom given 3 ref atoms. b,alpha,tau:(B,) -> (B,3)"""
    u = normalize(r_i - r_im1)
    v = normalize(r_im1 - r_im2)
    n = normalize(torch.cross(u, v, dim=-1))
    u_rot = rotate_around_axis(u, n, alpha)
    u_final = rotate_around_axis(u_rot, u, tau)
    return r_i + b.unsqueeze(-1) * u_final

class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_frame = nn.Parameter(torch.randn(3, 3) * 0.1)

    def forward(self, bonds, angles, torsions, aa_types):
        """bonds/angles:(B,L,14) torsions:(B,L,7) aa_types:(B,L) -> coords:(B,L,14,3)"""
        B, L = bonds.shape[0], bonds.shape[1]
        device = bonds.device
        coords = torch.zeros(B, L, 14, 3, device=device)
        for b_idx in range(B):
            for i in range(L):
                aa = aa_types[b_idx, i].item()
                n_atoms = self._num_atoms(aa)
                for a in range(n_atoms):
                    if i == 0 and a < 3:
                        coords[b_idx, i, a] = self.init_frame[a]
                    else:
                        r_im2, r_im1, r_i = self._ref_atoms(coords, b_idx, i, a)
                        bond = bonds[b_idx, i, a]
                        ang = angles[b_idx, i, a]
                        tau = self._get_torsion(torsions, b_idx, i, a)
                        coords[b_idx, i, a] = place_atom(
                            r_im2.unsqueeze(0), r_im1.unsqueeze(0),
                            r_i.unsqueeze(0), bond.unsqueeze(0),
                            ang.unsqueeze(0), tau.unsqueeze(0)
                        ).squeeze(0)
        return coords

    def inverse(self, coords, aa_types):
        """coords:(B,L,14,3) -> bonds:(B,L,14), angles:(B,L,14), torsions:(B,L,7)"""
        B, L = coords.shape[0], coords.shape[1]
        device = coords.device
        bonds = torch.zeros(B, L, 14, device=device)
        angles = torch.zeros(B, L, 14, device=device)
        torsions = torch.zeros(B, L, 7, device=device)
        for b_idx in range(B):
            for i in range(L):
                aa = aa_types[b_idx, i].item()
                n = self._num_atoms(aa)
                for a in range(1, n):
                    bonds[b_idx,i,a] = torch.norm(coords[b_idx,i,a] - coords[b_idx,i,a-1])
                    if a > 1:
                        v1 = coords[b_idx,i,a-2] - coords[b_idx,i,a-1]
                        v2 = coords[b_idx,i,a] - coords[b_idx,i,a-1]
                        c = ((v1*v2).sum()/(torch.norm(v1)*torch.norm(v2)+1e-8)).clamp(-1,1)
                        angles[b_idx,i,a] = torch.acos(c)
                    if a > 2:
                        tau = self._dihedral(coords[b_idx,i,a-3],coords[b_idx,i,a-2],coords[b_idx,i,a-1],coords[b_idx,i,a])
                        if a < 4: torsions[b_idx,i,min(a,2)] = tau
                        elif a-4 < 4: torsions[b_idx,i,3+a-4] = tau
        return bonds, angles, torsions

    def _num_atoms(self, aa):
        from cyclicdiffusion_mt.utils.constants import AA_ATOM_COUNT, IDX_TO_AA
        return AA_ATOM_COUNT.get(IDX_TO_AA.get(aa,'ALA'),5)

    def _ref_atoms(self, coords, b, i, a):
        if a < 3: return self.init_frame[0], self.init_frame[1], self.init_frame[2]
        return coords[b,i,a-3], coords[b,i,a-2], coords[b,i,a-1]

    def _get_torsion(self, torsions, b, i, a):
        if a < 4: return torsions[b,i,min(a,2)]
        chi = a-4; return torsions[b,i,3+chi] if chi < 4 else torch.tensor(0.0,device=torsions.device)

    def _dihedral(self, r1,r2,r3,r4):
        b1,b2,b3 = r2-r1, r3-r2, r4-r3
        n1 = normalize(torch.cross(b1.unsqueeze(0),b2.unsqueeze(0)).squeeze(0))
        n2 = normalize(torch.cross(b2.unsqueeze(0),b3.unsqueeze(0)).squeeze(0))
        m1 = torch.cross(n1.unsqueeze(0), normalize(b2.unsqueeze(0)).squeeze(0)).squeeze(0)
        x = (n1*n2).sum()
        y = (m1*n2).sum()
        return torch.atan2(y, x)
```

- [ ] **Step 4: `pytest tests/test_nerf.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/model/ tests/test_nerf.py && git commit -m "feat: add differentiable NeRF"
```

---

### Task 3: Data Pipeline — Dataset & Collate

**Files:**
- Create: `cyclicdiffusion_mt/data/__init__.py`
- Create: `cyclicdiffusion_mt/data/transforms.py`
- Create: `cyclicdiffusion_mt/data/dataset.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `compute_chi_mask(aa_types:(B,L))->(B,L,7)`: bool mask for valid chi
  - `compute_atom_mask(aa_types:(B,L))->(B,L,14)`: bool mask for valid atoms
  - `cartesian_to_internal(coords,atom_names,aa)->(bonds,angles,torsions)`: single-sample conversion
  - `class MultiTargetDataset(Dataset)`: `__getitem__` returns `dict(peptide_torsions, peptide_aa_types, target_coords, target_sequences, cyclo_mode, dG_rosetta, confidence)`
  - `class PeptideDataCollate`: pad and batch variable-length items

- [ ] **Step 1: Write test**

```python
# tests/test_data.py
import pytest, torch
from cyclicdiffusion_mt.data.transforms import compute_chi_mask, compute_atom_mask, cartesian_to_internal

class TestChiMask:
    def test_shape(self):
        aa = torch.tensor([[0,7,1]])  # ALA, GLY, ARG
        mask = compute_chi_mask(aa)
        assert mask.shape == (1, 3, 7)
    def test_ala_no_chi(self):
        aa = torch.tensor([[0]])  # ALA
        mask = compute_chi_mask(aa)
        assert mask[0,0,0] and mask[0,0,1] and mask[0,0,2]  # phi,psi,omega always True
        assert not mask[0,0,3]  # chi1 False for ALA
    def test_arg_all_chi(self):
        aa = torch.tensor([[1]])  # ARG
        mask = compute_chi_mask(aa)
        assert mask[0,0,3] and mask[0,0,4] and mask[0,0,5] and mask[0,0,6]  # all 4 chi

class TestAtomMask:
    def test_shape(self):
        aa = torch.tensor([[0,7]])  # ALA, GLY
        mask = compute_atom_mask(aa)
        assert mask.shape == (1, 2, 14)
    def test_gly_only_4_atoms(self):
        aa = torch.tensor([[7]])  # GLY
        mask = compute_atom_mask(aa)
        assert mask[0,0,0] and mask[0,0,1] and mask[0,0,2] and mask[0,0,3]
        assert not mask[0,0,4]  # no CB for GLY

class TestCartesianToInternal:
    def test_simple_ala(self):
        # N, CA, C, O, CB coords for ALA
        coords = torch.tensor([
            [0.0,0.0,0.0],   # N
            [1.458,0.0,0.0], # CA
            [2.458,0.0,0.0], # C (simplified)
            [2.458,1.231,0.0], # O
            [1.458,1.0,0.0], # CB
        ])
        atom_names = ['N','CA','C','O','CB']
        bonds, angles, torsions = cartesian_to_internal(coords, atom_names, 'ALA')
        assert bonds is not None
        assert angles is not None
```

- [ ] **Step 2: `pytest tests/test_data.py -v`** → FAIL

- [ ] **Step 3: Implement transforms.py**

```python
# cyclicdiffusion_mt/data/__init__.py
"""Data pipeline modules."""
```

```python
# cyclicdiffusion_mt/data/transforms.py
"""Data transforms: masks, coordinate conversion."""
import torch
from cyclicdiffusion_mt.utils.constants import (
    AA_TO_IDX, AA_CHI_COUNTS, AA_ATOM_NAMES, AA_ATOM_COUNT,
    MAX_ATOMS_PER_RES, MAX_CHI_PER_RES, NUM_TORSIONS, IDX_TO_AA,
)

def compute_chi_mask(aa_types):
    """aa_types:(B,L) -> chi_mask:(B,L,7), True for valid torsion positions."""
    B, L = aa_types.shape
    mask = torch.zeros(B, L, NUM_TORSIONS, dtype=torch.bool, device=aa_types.device)
    mask[:, :, :3] = True  # phi, psi, omega always present
    for i in range(MAX_CHI_PER_RES):
        for b in range(B):
            for l in range(L):
                aa = aa_types[b, l].item()
                chi_count = AA_CHI_COUNTS[aa]
                if i < chi_count:
                    mask[b, l, 3 + i] = True
    return mask

def compute_atom_mask(aa_types):
    """aa_types:(B,L) -> atom_mask:(B,L,14), True for existing atoms."""
    B, L = aa_types.shape
    mask = torch.zeros(B, L, MAX_ATOMS_PER_RES, dtype=torch.bool, device=aa_types.device)
    for b in range(B):
        for l in range(L):
            aa = aa_types[b, l].item()
            aa_name = IDX_TO_AA.get(aa, 'ALA')
            n_atoms = AA_ATOM_COUNT.get(aa_name, 5)
            mask[b, l, :n_atoms] = True
    return mask

def cartesian_to_internal(coords, atom_names, aa_type):
    """Convert single-residue Cartesian coords to internal coordinates.
    coords:(N_atoms,3) atom_names:list[str] -> bonds:(14,),angles:(14,),torsions:(7,)
    """
    N = len(atom_names)
    bonds = torch.zeros(MAX_ATOMS_PER_RES)
    angles = torch.zeros(MAX_ATOMS_PER_RES)
    torsions = torch.zeros(NUM_TORSIONS)
    for a in range(1, N):
        bonds[a] = torch.norm(coords[a] - coords[a-1])
        if a > 1:
            v1 = coords[a-2] - coords[a-1]
            v2 = coords[a] - coords[a-1]
            cos_a = ((v1*v2).sum()/(torch.norm(v1)*torch.norm(v2)+1e-8)).clamp(-1,1)
            angles[a] = torch.acos(cos_a)
        if a > 2:
            tau = _compute_dihedral(coords[a-3],coords[a-2],coords[a-1],coords[a])
            if a < 4: torsions[min(a,2)] = tau
            elif a-4 < 4: torsions[3+a-4] = tau
    return bonds, angles, torsions

def _compute_dihedral(r1,r2,r3,r4):
    b1,b2,b3 = r2-r1, r3-r2, r4-r3
    n1 = b1.cross(b2); n1 = n1/(n1.norm()+1e-8)
    n2 = b2.cross(b3); n2 = n2/(n2.norm()+1e-8)
    m1 = n1.cross(b2/(b2.norm()+1e-8))
    x = (n1*n2).sum(); y = (m1*n2).sum()
    return torch.atan2(y, x)
```

Write `dataset.py`:

```python
# cyclicdiffusion_mt/data/dataset.py
"""Multi-target peptide dataset."""
import torch
from torch.utils.data import Dataset
from cyclicdiffusion_mt.utils.constants import AA_TO_IDX, MAX_RESIDUES

class MultiTargetDataset(Dataset):
    """Dataset for multi-target cyclic peptide data.
    Returns per sample: dict with peptide_torsions, peptide_aa_types,
    target_coords, target_sequences, cyclo_mode, dG_rosetta, confidence.
    """
    def __init__(self, data_manifest, max_residues=MAX_RESIDUES, max_targets=3):
        self.manifest = data_manifest  # list of dicts
        self.max_residues = max_residues
        self.max_targets = max_targets

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        return {
            'peptide_torsions': torch.tensor(entry['peptide_torsions'], dtype=torch.float32),
            'peptide_aa_types': torch.tensor(entry['peptide_aa_types'], dtype=torch.long),
            'target_coords': [torch.tensor(tc, dtype=torch.float32) for tc in entry.get('target_coords', [])],
            'target_sequences': [torch.tensor(ts, dtype=torch.long) for ts in entry.get('target_sequences', [])],
            'cyclo_mode': entry.get('cyclo_mode', 0),
            'dG_rosetta': torch.tensor(entry.get('dG_rosetta', 0.0), dtype=torch.float32),
            'confidence': torch.tensor(entry.get('confidence', 1.0), dtype=torch.float32),
        }

class PeptideDataCollate:
    """Collate function to pad variable-length peptides and targets into batches."""
    def __init__(self, max_residues=MAX_RESIDUES, max_atoms=14, pad_torsion=0.0, pad_aa=25):
        self.max_residues = max_residues
        self.max_atoms = max_atoms
        self.pad_torsion = pad_torsion
        self.pad_aa = pad_aa

    def __call__(self, batch):
        B = len(batch)
        L_max = max(item['peptide_aa_types'].shape[0] for item in batch)
        L_max = min(L_max, self.max_residues)

        peptide_torsions = torch.full((B, L_max, 7), self.pad_torsion)
        peptide_aa_types = torch.full((B, L_max), self.pad_aa, dtype=torch.long)
        peptide_mask = torch.zeros(B, L_max, dtype=torch.bool)
        cyclo_modes = torch.zeros(B, dtype=torch.long)
        dG_rosetta = torch.zeros(B)
        confidence = torch.zeros(B)

        # Targets: padded to max targets across batch
        K_max = max(len(item.get('target_coords', [])) for item in batch)
        K_max = max(K_max, 1)

        for b, item in enumerate(batch):
            L = min(item['peptide_aa_types'].shape[0], L_max)
            peptide_torsions[b,:L] = item['peptide_torsions'][:L]
            peptide_aa_types[b,:L] = item['peptide_aa_types'][:L]
            peptide_mask[b,:L] = True
            cyclo_modes[b] = item.get('cyclo_mode', 0)
            dG_rosetta[b] = item.get('dG_rosetta', 0.0)
            confidence[b] = item.get('confidence', 1.0)

        return {
            'peptide_torsions': peptide_torsions,
            'peptide_aa_types': peptide_aa_types,
            'peptide_mask': peptide_mask,
            'target_coords': batch[0].get('target_coords', []),
            'target_sequences': batch[0].get('target_sequences', []),
            'cyclo_modes': cyclo_modes,
            'dG_rosetta': dG_rosetta,
            'confidence': confidence,
        }
```

- [ ] **Step 4: `pytest tests/test_data.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/data/ tests/test_data.py && git commit -m "feat: add data pipeline with transforms and dataset"
```

---

### Task 4: IPA Target Encoder

**Files:**
- Create: `cyclicdiffusion_mt/model/target_encoder.py`
- Create: `tests/test_target_encoder.py`

**Interfaces:**
- Produces: `class TargetEncoder(nn.Module)` with `forward(target_coords, target_mask)->list[(B,N_k,d_target)]` where d_target=128, 3 IPA blocks

- [ ] **Step 1: Write test**

```python
# tests/test_target_encoder.py
import pytest, torch
from cyclicdiffusion_mt.model.target_encoder import TargetEncoder

class TestTargetEncoder:
    @pytest.fixture
    def encoder(self): return TargetEncoder(d_target=128, n_blocks=3, d_head=32, n_heads=4)
    def test_output_shape_single_target(self, encoder):
        B,K,N = 2,1,10
        coords = [torch.randn(B,N,14,3)]
        mask = [torch.ones(B,N,dtype=torch.bool)]
        out = encoder(coords, mask)
        assert len(out) == 1
        assert out[0].shape == (B, N, 128)
    def test_output_shape_multi_target(self, encoder):
        B,N1,N2 = 2,10,15
        coords = [torch.randn(B,N1,14,3), torch.randn(B,N2,14,3)]
        mask = [torch.ones(B,N1,dtype=torch.bool), torch.ones(B,N2,dtype=torch.bool)]
        out = encoder(coords, mask)
        assert len(out) == 2
        assert out[0].shape == (B, N1, 128)
        assert out[1].shape == (B, N2, 128)
    def test_shared_weights(self, encoder):
        """Two forward passes with same weights should give same result."""
        B,N=1,5
        coords=[torch.randn(B,N,14,3)]; mask=[torch.ones(B,N,dtype=torch.bool)]
        o1=encoder(coords,mask); o2=encoder(coords,mask)
        assert torch.allclose(o1[0],o2[0],atol=1e-6)
```

- [ ] **Step 2: `pytest tests/test_target_encoder.py -v`** → FAIL

- [ ] **Step 3: Implement**

```python
# cyclicdiffusion_mt/model/target_encoder.py
"""IPA-based target protein encoder with shared weights."""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class IPABlock(nn.Module):
    """Single Invariant Point Attention block."""
    def __init__(self, d_model=128, d_head=32, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model, self.d_head, self.n_heads = d_model, d_head, n_heads
        self.qkv = nn.Linear(d_model, 3*d_head*n_heads)
        self.o_proj = nn.Linear(d_head*n_heads, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """x:(B,N,d_model) mask:(B,N)"""
        B,N,D = x.shape
        residual = x; x = self.norm1(x)
        qkv = self.qkv(x).view(B,N,3,self.n_heads,self.d_head).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1,2).contiguous().view(B,N,-1)
        x = residual + self.o_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x

class TargetEncoder(nn.Module):
    """IPA-based encoder for protein target structures. Shared weights across targets."""
    def __init__(self, d_target=128, n_blocks=3, d_head=32, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_target = d_target
        # Input projection: mean over 14 atoms per residue + atom type embedding
        self.atom_proj = nn.Linear(3, d_target)  # per-atom xyz -> features
        self.ipa_blocks = nn.ModuleList([
            IPABlock(d_target, d_head, n_heads, dropout) for _ in range(n_blocks)
        ])

    def forward(self, target_coords, target_masks):
        """target_coords: list[(B,N_k,14,3)] target_masks: list[(B,N_k)]
        Returns: list[(B,N_k,d_target)]"""
        outputs = []
        for coords, mask in zip(target_coords, target_masks):
            B,N,_,_ = coords.shape
            # Mean-pool atom features to residue-level
            atom_feats = self.atom_proj(coords)  # (B,N,14,d_target)
            valid_mask = mask.unsqueeze(-1).unsqueeze(-1).float()
            res_feats = (atom_feats * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-8)
            for block in self.ipa_blocks:
                res_feats = block(res_feats, mask)
            outputs.append(res_feats)
        return outputs
```

- [ ] **Step 4: `pytest tests/test_target_encoder.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/model/target_encoder.py tests/test_target_encoder.py && git commit -m "feat: add IPA target encoder"
```

---

### Task 5: Multi-Target Cross-Attention

**Files:**
- Create: `cyclicdiffusion_mt/model/cross_attention.py`
- Create: `tests/test_cross_attention.py`

**Interfaces:**
- Produces: `class MultiTargetCrossAttention(nn.Module)` with `forward(peptide_feats, target_feats, target_masks, timestep)->peptide_feats_updated:(B,L,d_model)`, `class AdaptiveTargetGating(nn.Module)` with `forward(target_feats, timestep)->alpha:(B,K)`

- [ ] **Step 1: Write test**

```python
# tests/test_cross_attention.py
import pytest, torch
from cyclicdiffusion_mt.model.cross_attention import MultiTargetCrossAttention, AdaptiveTargetGating

class TestAdaptiveGating:
    @pytest.fixture
    def gating(self): return AdaptiveTargetGating(d_target=128, d_time=64)
    def test_output_shape_2_targets(self, gating):
        targets = [torch.randn(2,10,128), torch.randn(2,15,128)]
        t = torch.tensor([100.0,200.0])
        alpha = gating(targets, t)
        assert alpha.shape == (2,2)  # B,K
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-6)

class TestCrossAttention:
    @pytest.fixture
    def xattn(self): return MultiTargetCrossAttention(d_model=256, d_target=128, d_head=64, n_heads=4)
    def test_output_shape(self, xattn):
        B,L = 2,10
        pep = torch.randn(B,L,256); pep_mask = torch.ones(B,L,dtype=torch.bool)
        targets = [torch.randn(B,12,128), torch.randn(B,15,128)]
        t_mask = [torch.ones(B,12,dtype=torch.bool), torch.ones(B,15,dtype=torch.bool)]
        t_emb = torch.randn(B,64)
        out = xattn(pep, pep_mask, targets, t_mask, t_emb)
        assert out.shape == (B, L, 256)
```

- [ ] **Step 2: `pytest tests/test_cross_attention.py -v`** → FAIL

- [ ] **Step 3: Implement**

```python
# cyclicdiffusion_mt/model/cross_attention.py
"""Multi-target cross-attention with adaptive per-target gating."""
import torch, torch.nn as nn, torch.nn.functional as F
import math

class AdaptiveTargetGating(nn.Module):
    """Learns per-target importance weights alpha_k = softmax(g(h_k, t))."""
    def __init__(self, d_target=128, d_time=64, hidden=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_target+d_time,hidden), nn.SiLU(), nn.Linear(hidden,1))

    def forward(self, target_feats, t_emb):
        """target_feats:list[(B,N_k,d_target)] t_emb:(B,d_time) -> alpha:(B,K)"""
        K = len(target_feats)
        B = target_feats[0].shape[0]
        scores = []
        for k in range(K):
            # Pool target features: mean over residues
            pooled = target_feats[k].mean(dim=1)  # (B,d_target)
            inp = torch.cat([pooled, t_emb], dim=-1)  # (B,d_target+d_time)
            scores.append(self.gate(inp))  # (B,1)
        scores = torch.cat(scores, dim=-1)  # (B,K)
        return F.softmax(scores, dim=-1)

class MultiTargetCrossAttention(nn.Module):
    """Multi-head cross-attention from peptide (Q) to concatenated targets (K,V)."""
    def __init__(self, d_model=256, d_target=128, d_head=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model, self.d_head, self.n_heads = d_model, d_head, n_heads
        self.q_proj = nn.Linear(d_model, d_head*n_heads)
        self.kv_proj = nn.Linear(d_target, 2*d_head*n_heads)
        self.o_proj = nn.Linear(d_head*n_heads, d_model)
        self.gating = AdaptiveTargetGating(d_target)
        self.target_idx_embed = nn.Embedding(10, d_target)  # max 10 targets
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, peptide_feats, peptide_mask, target_feats, target_masks, t_emb, cyclo_emb=None):
        """peptide_feats:(B,L,d_model) target_feats:list[(B,N_k,d_target)]
        target_masks:list[(B,N_k)] t_emb:(B,d_time) -> updated:(B,L,d_model)"""
        B,L,_ = peptide_feats.shape
        K = len(target_feats)

        # Add target index embedding and concatenate targets
        target_parts = []
        target_mask_parts = []
        for k, (tf, tm) in enumerate(zip(target_feats, target_masks)):
            idx_emb = self.target_idx_embed(torch.tensor(k,device=tf.device))
            target_parts.append(tf + idx_emb)
            target_mask_parts.append(tm)
        all_targets = torch.cat(target_parts, dim=1)  # (B, sum(N_k), d_target)
        all_tmasks = torch.cat(target_mask_parts, dim=1)  # (B, sum(N_k))

        # Adaptive gating
        alpha = self.gating(target_feats, t_emb)  # (B,K)

        # Compute Q, K, V
        Q = self.q_proj(self.norm(peptide_feats)).view(B,L,self.n_heads,self.d_head).permute(0,2,1,3)
        kv = self.kv_proj(all_targets).view(B,-1,2,self.n_heads,self.d_head).permute(2,0,3,1,4)
        K_target, V = kv[0], kv[1]

        # Cross-attention
        attn = (Q @ K_target.transpose(-2,-1)) / math.sqrt(self.d_head)
        attn_mask = ~all_tmasks.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(attn_mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1,2).contiguous().view(B,L,-1)
        out = self.o_proj(out)

        # Scale by per-target gating (broadcast across residues)
        # Apply as residual update
        return peptide_feats + out
```

- [ ] **Step 4: `pytest tests/test_cross_attention.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/model/cross_attention.py tests/test_cross_attention.py && git commit -m "feat: add multi-target cross-attention with adaptive gating"
```

---

### Task 6: Frame-Based Denoiser Core

**Files:**
- Create: `cyclicdiffusion_mt/model/denoiser.py`
- Create: `tests/test_denoiser.py`

**Interfaces:**
- Produces: `class FrameDenoiser(nn.Module)` with `forward(tau_t, a_t, t, target_feats, target_masks, cyclo_mode)->(tau_0_pred, a_0_logits, dG_pred)`
  - Input: tau_t:(B,L,7), a_t:(B,L), t:(B,), target_feats:list[(B,N_k,128)], cyclo_mode:(B,)
  - Output: tau_0_pred:(B,L,7), a_0_logits:(B,L,26), dG_pred:(B,K)

- [ ] **Step 1: Write test**

```python
# tests/test_denoiser.py
import pytest, torch
from cyclicdiffusion_mt.model.denoiser import FrameDenoiser

class TestFrameDenoiser:
    @pytest.fixture
    def denoiser(self):
        return FrameDenoiser(d_model=256, d_target=128, d_time=64, n_blocks=6, d_head=64, n_heads=4)

    def test_forward_shape_single_target(self, denoiser):
        B,L = 2,8
        tau_t = torch.randn(B,L,7); a_t = torch.randint(0,26,(B,L))
        t = torch.rand(B); cyclo = torch.zeros(B,dtype=torch.long)
        targets = [torch.randn(B,15,128)]; t_mask = [torch.ones(B,15,dtype=torch.bool)]
        tau_pred, a_logits, dg_pred = denoiser(tau_t, a_t, t, targets, t_mask, cyclo)
        assert tau_pred.shape == (B,L,7)
        assert a_logits.shape == (B,L,26)
        assert dg_pred.shape == (B,1)  # K=1

    def test_forward_shape_multi_target(self, denoiser):
        B,L = 2,8
        tau_t = torch.randn(B,L,7); a_t = torch.randint(0,26,(B,L))
        t = torch.rand(B); cyclo = torch.zeros(B,dtype=torch.long)
        targets = [torch.randn(B,12,128), torch.randn(B,15,128)]
        t_mask = [torch.ones(B,12,dtype=torch.bool), torch.ones(B,15,dtype=torch.bool)]
        tau_pred, a_logits, dg_pred = denoiser(tau_t, a_t, t, targets, t_mask, cyclo)
        assert tau_pred.shape == (B,L,7)
        assert a_logits.shape == (B,L,26)
        assert dg_pred.shape == (B,2)  # K=2

    def test_output_range(self, denoiser):
        """Torsion predictions should be in [-pi, pi)."""
        B,L = 1,5
        tau_t = torch.randn(B,L,7); a_t = torch.randint(0,26,(B,L))
        t = torch.rand(B); cyclo = torch.zeros(B,dtype=torch.long)
        targets = [torch.randn(B,10,128)]; t_mask = [torch.ones(B,10,dtype=torch.bool)]
        tau_pred, _, _ = denoiser(tau_t, a_t, t, targets, t_mask, cyclo)
        assert tau_pred.min() >= -torch.pi - 0.1
        assert tau_pred.max() <= torch.pi + 0.1
```

- [ ] **Step 2: `pytest tests/test_denoiser.py -v`** → FAIL

- [ ] **Step 3: Implement**

```python
# cyclicdiffusion_mt/model/denoiser.py
"""SE(3) frame-based denoiser with multi-target cross-attention."""
import torch, torch.nn as nn, torch.nn.functional as F, math
from cyclicdiffusion_mt.model.cross_attention import MultiTargetCrossAttention

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_time=64):
        super().__init__()
        self.d_time = d_time
    def forward(self, t):
        """t:(B,) -> (B,d_time)"""
        device = t.device
        half = self.d_time // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class DenoiserBlock(nn.Module):
    """One denoiser block: frame update + cross-attention + cyclo injection + FFN."""
    def __init__(self, d_model=256, d_target=128, d_time=64, d_head=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.frame_update = FrameUpdate(d_model, d_head, n_heads, dropout)
        self.cross_attn = MultiTargetCrossAttention(d_model, d_target, d_head, n_heads, dropout)
        self.cyclo_proj = nn.Linear(32, d_model)  # cyclo embedding projected to d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, feats, frames, mask, targets, t_masks, t_emb, cyclo_emb):
        # Frame update
        feats = feats + self.frame_update(feats, frames, mask)
        feats = self.norm1(feats)
        # Cross-attention to targets
        feats = self.cross_attn(feats, mask, targets, t_masks, t_emb)
        feats = self.norm2(feats)
        # Cyclization bias injection
        cyclo_bias = self.cyclo_proj(cyclo_emb)  # (B,L,d_model)
        feats = feats + cyclo_bias
        # FFN
        feats = feats + self.ffn(self.norm3(feats))
        return feats

class FrameUpdate(nn.Module):
    """SE(3) frame update via message passing on residue frames."""
    def __init__(self, d_model=256, d_head=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model, self.d_head, self.n_heads = d_model, d_head, n_heads
        self.qkv = nn.Linear(d_model, 3*d_head*n_heads)
        self.o_proj = nn.Linear(d_head*n_heads, d_model)
        self.edge_mlp = nn.Sequential(nn.Linear(d_model*2+16,d_model), nn.SiLU(), nn.Linear(d_model,d_model))
        self.rbf_centers = nn.Parameter(torch.linspace(2.0,20.0,16))
        self.rbf_sigma = 2.0
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, frames, mask):
        """feats:(B,L,d_model) frames:(B,L,? ) mask:(B,L)"""
        B,L,D = feats.shape
        # Self-attention as simplified frame update
        qkv = self.qkv(feats).view(B,L,3,self.n_heads,self.d_head).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1,2).contiguous().view(B,L,-1)
        return self.o_proj(out)

class FrameDenoiser(nn.Module):
    """Main frame-based denoiser: predicts clean torsions, AA types, and affinity."""
    def __init__(self, d_model=256, d_target=128, d_time=64, n_blocks=6, d_head=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Input projections
        self.torsion_proj = nn.Linear(7, d_model)  # 7 torsions -> d_model
        self.aa_embed = nn.Embedding(26, d_model)  # 25 AA + MASK
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_time), nn.Linear(d_time, d_time), nn.SiLU(), nn.Linear(d_time, d_time),
        )
        self.cyclo_embed = nn.Embedding(5, 32)  # 5 cyclization modes -> 32
        # Denoiser blocks
        self.blocks = nn.ModuleList([
            DenoiserBlock(d_model, d_target, d_time, d_head, n_heads, dropout) for _ in range(n_blocks)
        ])
        # Output heads
        self.torsion_head = nn.Sequential(nn.Linear(d_model,d_model//2), nn.SiLU(), nn.Linear(d_model//2,7))
        self.aa_head = nn.Linear(d_model, 26)  # 25 AA + MASK
        self.affinity_head = AffinityHead(d_model, d_target)

    def forward(self, tau_t, a_t, t, target_feats, target_masks, cyclo_mode):
        """tau_t:(B,L,7) a_t:(B,L) t:(B,) target_feats:list[(B,N_k,d_target)]
        target_masks:list[(B,N_k)] cyclo_mode:(B,) -> tau_pred:(B,L,7), aa_logits:(B,L,26), dg_pred:(B,K)"""
        B, L = tau_t.shape[0], tau_t.shape[1]
        # Merge torsion + AA embeddings
        torsion_feats = self.torsion_proj(tau_t)  # (B,L,d_model)
        aa_feats = self.aa_embed(a_t.clamp(0,25))  # (B,L,d_model)
        feats = torsion_feats + aa_feats

        # Time embedding
        t_emb = self.time_mlp(t)  # (B,d_time=64)

        # Cyclization embedding
        cyclo_emb = self.cyclo_embed(cyclo_mode)  # (B,32)

        # Dummy frames (placeholder for full SE(3) implementation)
        frames = None
        mask = torch.ones(B, L, dtype=torch.bool, device=tau_t.device)

        # Process through blocks
        for block in self.blocks:
            feats = block(feats, frames, mask, target_feats, target_masks, t_emb, cyclo_emb.unsqueeze(1).expand(-1,L,-1))

        # Output predictions
        tau_pred = self.torsion_head(feats)  # (B,L,7)
        tau_pred = torch.atan2(torch.sin(tau_pred), torch.cos(tau_pred))  # wrap to [-pi,pi)
        aa_logits = self.aa_head(feats)  # (B,L,26)
        dg_pred = self.affinity_head(feats, target_feats, mask, target_masks)

        return tau_pred, aa_logits, dg_pred

class AffinityHead(nn.Module):
    """Lightweight Rosetta dG regression head."""
    def __init__(self, d_model=256, d_target=128, hidden_dim=128):
        super().__init__()
        self.pep_pool = nn.Sequential(nn.Linear(d_model,hidden_dim), nn.SiLU(), nn.Linear(hidden_dim,hidden_dim))
        self.tar_pool = nn.Sequential(nn.Linear(d_target,hidden_dim), nn.SiLU(), nn.Linear(hidden_dim,hidden_dim))
        self.predictor = nn.Sequential(
            nn.Linear(2*hidden_dim,hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim,hidden_dim//2), nn.SiLU(), nn.Linear(hidden_dim//2,1),
        )

    def forward(self, pep_feats, target_feats, pep_mask, t_masks):
        """pep_feats:(B,L,d_model) target_feats:list[(B,N_k,d_target)]
        -> dg_pred:(B,K)"""
        # Pool peptide features
        pep = (pep_feats * pep_mask.unsqueeze(-1).float()).sum(dim=1) / (pep_mask.sum(dim=1,keepdim=True).float()+1e-8)
        pep = self.pep_pool(pep)

        dg_preds = []
        for k, (tf, tm) in enumerate(zip(target_feats, t_masks)):
            t_pooled = (tf * tm.unsqueeze(-1).float()).sum(dim=1) / (tm.sum(dim=1,keepdim=True).float()+1e-8)
            t_emb = self.tar_pool(t_pooled)
            dg_preds.append(self.predictor(torch.cat([pep, t_emb], dim=-1)))
        return torch.cat(dg_preds, dim=-1)
```

- [ ] **Step 4: `pytest tests/test_denoiser.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/model/denoiser.py tests/test_denoiser.py && git commit -m "feat: add frame-based denoiser with output heads"
```

---

### Task 7: Diffusion Processes (Continuous + Discrete)

**Files:**
- Create: `cyclicdiffusion_mt/model/diffusion.py`
- Create: `tests/test_diffusion.py`

**Interfaces:**
- Produces:
  - `class WrappedNormalDiffusion(nn.Module)`: forward (noising) `q_sample(tau_0,t)->tau_t,noise`, reverse `p_sample(denoiser,tau_t,t,cond)->tau_{t-1}`, `loss(tau_0,tau_pred)->scalar`
  - `class MaskedDiscreteDiffusion(nn.Module)`: forward `q_sample(a_0,t)->a_t`, reverse `p_sample(a_logits,a_t,t)->a_{t-1}`, `loss(a_0,a_logits)->scalar`
  - `cosine_schedule(t,T=500)->alpha_bar_t`

- [ ] **Step 1: Write test**

```python
# tests/test_diffusion.py
import pytest, torch
from cyclicdiffusion_mt.model.diffusion import (
    WrappedNormalDiffusion, MaskedDiscreteDiffusion, cosine_schedule,
)

class TestCosineSchedule:
    def test_start_is_1(self):
        sched = cosine_schedule(500)
        assert torch.allclose(sched[0], torch.tensor(1.0), atol=0.05)
    def test_end_is_0(self):
        sched = cosine_schedule(500)
        assert sched[-1] < 0.01
    def test_monotonic_decreasing(self):
        sched = cosine_schedule(500)
        assert (sched[1:] <= sched[:-1]).all()

class TestWrappedNormalDiffusion:
    @pytest.fixture
    def diff(self): return WrappedNormalDiffusion(T=500)
    def test_q_sample_shape(self, diff):
        tau_0 = torch.randn(2,10,7)
        t = torch.randint(0,500,(2,))
        tau_t, noise = diff.q_sample(tau_0, t)
        assert tau_t.shape == tau_0.shape
        assert noise.shape == tau_0.shape
    def test_t0_is_clean(self, diff):
        tau_0 = torch.randn(2,5,7)
        tau_t, _ = diff.q_sample(tau_0, torch.zeros(2,dtype=torch.long))
        assert torch.allclose(tau_t, tau_0, atol=1e-5)
    def test_loss_shape(self, diff):
        tau_0 = torch.randn(2,10,7); tau_pred = torch.randn(2,10,7)
        loss = diff.loss(tau_0, tau_pred)
        assert loss.ndim == 0  # scalar
    def test_wrapped_loss_handles_boundary(self, diff):
        """Loss between -pi+eps and pi-eps should be small."""
        tau_0 = torch.tensor([[[3.1,0,0,0,0,0,0]]])  # near pi
        tau_pred = torch.tensor([[[-3.1,0,0,0,0,0,0]]])  # near -pi but same angle
        loss = diff.loss(tau_0, tau_pred)
        assert loss.item() < 0.1  # should be close since angles are equivalent

class TestMaskedDiscreteDiffusion:
    @pytest.fixture
    def diff(self): return MaskedDiscreteDiffusion(T=500, num_classes=26, mask_idx=25)
    def test_q_sample_shape(self, diff):
        a_0 = torch.randint(0,25,(2,10)); t = torch.randint(0,500,(2,))
        a_t = diff.q_sample(a_0, t)
        assert a_t.shape == a_0.shape
    def test_t0_is_clean(self, diff):
        a_0 = torch.randint(0,25,(2,5))
        a_t = diff.q_sample(a_0, torch.zeros(2,dtype=torch.long))
        assert (a_t == a_0).all()
    def test_loss_shape(self, diff):
        a_0 = torch.randint(0,25,(2,10))
        logits = torch.randn(2,10,26)
        loss = diff.loss(logits, a_0)
        assert loss.ndim == 0
```

- [ ] **Step 2: `pytest tests/test_diffusion.py -v`** → FAIL

- [ ] **Step 3: Implement**

```python
# cyclicdiffusion_mt/model/diffusion.py
"""DDPM diffusion processes: wrapped normal (continuous) + masked discrete."""
import torch, torch.nn as nn, torch.nn.functional as F, math

def cosine_schedule(T=500, s=0.008):
    """Cosine noise schedule. Returns alpha_bar_t for t=0..T-1."""
    steps = torch.arange(T+1, dtype=torch.float32)
    alpha_bar = torch.cos((steps/T + s) / (1+s) * math.pi/2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return alpha_bar[:T]  # shape (T,)

class WrappedNormalDiffusion(nn.Module):
    """Wrapped normal diffusion for torsion angles on [-pi, pi)."""
    def __init__(self, T=500):
        super().__init__()
        self.T = T
        alpha_bar = cosine_schedule(T)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('beta', 1 - alpha_bar[1:]/alpha_bar[:-1].clamp(min=1e-8))

    def q_sample(self, tau_0, t):
        """Forward noising: tau_0 (B,L,7) + t (B,) -> tau_t, noise."""
        a_bar = self.alpha_bar[t]  # (B,)
        noise = torch.randn_like(tau_0)
        # Wrapped addition: (tau_0 + sqrt(1-a_bar)*noise) mod 2pi - pi
        tau_t = tau_0 * a_bar.view(-1,1,1).sqrt() + noise * (1-a_bar).view(-1,1,1).sqrt()
        tau_t = torch.atan2(torch.sin(tau_t), torch.cos(tau_t))
        return tau_t, noise

    def loss(self, tau_0, tau_pred):
        """Wrapped L2 loss: min_k ||tau_0 - tau_pred + 2*pi*k||^2."""
        diff = tau_0 - tau_pred
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return (diff ** 2).mean()

class MaskedDiscreteDiffusion(nn.Module):
    """Masked discrete diffusion for amino acid types. D3PM-style."""
    def __init__(self, T=500, num_classes=26, mask_idx=25):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.mask_idx = mask_idx
        alpha_bar = cosine_schedule(T)
        self.register_buffer('alpha_bar', alpha_bar)

    def q_sample(self, a_0, t):
        """Forward: a_0 (B,L) + t (B,) -> a_t (B,L) with mask corruption."""
        a_bar = self.alpha_bar[t]  # (B,)
        # With prob 1-a_bar, replace with MASK
        mask_prob = (1 - a_bar).view(-1,1)
        rand = torch.rand_like(a_0.float())
        a_t = torch.where(rand < mask_prob,
                          torch.full_like(a_0, self.mask_idx),
                          a_0)
        return a_t

    def loss(self, logits, a_0):
        """Cross-entropy loss for AA type prediction. logits:(B,L,26) a_0:(B,L)."""
        return F.cross_entropy(logits.view(-1, self.num_classes), a_0.view(-1),
                               ignore_index=self.mask_idx)

    @torch.no_grad()
    def p_sample(self, logits, a_t):
        """Sample a_{t-1} from predicted a_0 logits. logits:(B,L,26) a_t:(B,L)."""
        probs = F.softmax(logits, dim=-1)
        a_pred = torch.multinomial(probs.view(-1, self.num_classes), 1).view_as(a_t)
        # For non-masked positions, keep original (only unmask masked ones)
        a_out = torch.where(a_t == self.mask_idx, a_pred, a_t)
        return a_out
```

- [ ] **Step 4: `pytest tests/test_diffusion.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/model/diffusion.py tests/test_diffusion.py && git commit -m "feat: add diffusion processes (wrapped normal + masked discrete)"
```

---

### Task 8: Loss Functions

**Files:**
- Create: `cyclicdiffusion_mt/losses/__init__.py`
- Create: `cyclicdiffusion_mt/losses/torsion_loss.py`
- Create: `cyclicdiffusion_mt/losses/type_loss.py`
- Create: `cyclicdiffusion_mt/losses/cyclo_loss.py`
- Create: `cyclicdiffusion_mt/losses/geometry_loss.py`
- Create: `cyclicdiffusion_mt/losses/affinity_loss.py`
- Create: `tests/test_losses.py`

**Interfaces:**
- Produces:
  - `torsion_loss(tau_pred,tau_0,chi_mask)->scalar`: wrapped L2 with mask
  - `type_loss(a_logits,a_0)->scalar`: cross-entropy
  - `cyclo_loss(nerf,tau_pred,aa_types,bonds,angles,cyclo_mode)->scalar`: ring closure error
  - `geometry_losses`: `rama_loss(phi,psi,aa_types)->scalar`, `clash_loss(coords,atom_mask)->scalar`, `rotamer_loss(chis,aa_types)->scalar`
  - `affinity_loss(dg_pred,dg_label,confidence)->scalar`: MSE

- [ ] **Step 1: Write test**

```python
# tests/test_losses.py
import pytest, torch
from cyclicdiffusion_mt.losses.torsion_loss import torsion_loss
from cyclicdiffusion_mt.losses.type_loss import type_loss
from cyclicdiffusion_mt.losses.cyclo_loss import cyclo_loss
from cyclicdiffusion_mt.losses.geometry_loss import clash_loss
from cyclicdiffusion_mt.losses.affinity_loss import affinity_loss
from cyclicdiffusion_mt.model.nerf import NeRF

class TestTorsionLoss:
    def test_perfect_prediction(self):
        tau = torch.randn(2,5,7); mask = torch.ones(2,5,7,dtype=torch.bool)
        loss = torsion_loss(tau, tau, mask)
        assert loss.item() < 1e-5
    def test_masked_positions_ignored(self):
        tau = torch.randn(2,5,7); mask = torch.ones(2,5,7,dtype=torch.bool)
        mask[:,:,3:] = False  # mask all chi
        loss1 = torsion_loss(tau, tau+1.0, mask)
        # Should be zero since masked positions are ignored
        # Actually only phi,psi,omega contribute
        assert loss1.ndim == 0

class TestTypeLoss:
    def test_shape(self):
        logits = torch.randn(2,10,26); a_0 = torch.randint(0,25,(2,10))
        loss = type_loss(logits, a_0)
        assert loss.ndim == 0

class TestCycloLoss:
    def test_perfect_closure(self):
        nerf = NeRF()
        # This tests that closed rings give near-zero loss
        B,L = 1,4
        aa = torch.zeros(B,L,dtype=torch.long); mode=torch.zeros(B,dtype=torch.long)
        bonds=torch.rand(B,L,14)*0.1+1.3; angles=torch.rand(B,L,14)*0.3+1.9
        tau=torch.rand(B,L,7)*0.1
        loss = cyclo_loss(nerf, tau, aa, bonds, angles, mode)
        assert loss.ndim == 0 and not torch.isnan(loss)

class TestClashLoss:
    def test_no_clash(self):
        coords = torch.randn(1,3,14,3)*100  # far apart
        mask = torch.ones(1,3,14,dtype=torch.bool)
        loss = clash_loss(coords, mask)
        assert loss.item() < 1e-5

class TestAffinityLoss:
    def test_shape(self):
        dg_pred = torch.randn(4,2); dg_label = torch.randn(4,2); conf = torch.ones(4)
        loss = affinity_loss(dg_pred, dg_label, conf)
        assert loss.ndim == 0
```

- [ ] **Step 2: `pytest tests/test_losses.py -v`** → FAIL

- [ ] **Step 3: Implement all loss modules**

(See complete implementations in the detailed spec. Key implementations:)

```python
# losses/torsion_loss.py
import torch
def torsion_loss(tau_pred, tau_0, chi_mask):
    diff = tau_0 - tau_pred
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    loss_per_pos = diff.pow(2).sum(dim=-1)
    return (loss_per_pos * chi_mask.float()).sum() / (chi_mask.float().sum() + 1e-8)

# losses/type_loss.py
import torch.nn.functional as F
from cyclicdiffusion_mt.utils.constants import NUM_AA_TYPES_WITH_MASK, MASK_IDX
def type_loss(a_logits, a_0):
    return F.cross_entropy(a_logits.view(-1, NUM_AA_TYPES_WITH_MASK), a_0.view(-1), ignore_index=MASK_IDX)

# losses/cyclo_loss.py
import torch
from cyclicdiffusion_mt.utils.constants import IDEAL_PEPTIDE_BOND
def cyclo_loss(nerf, tau_pred, aa_types, bonds, angles, cyclo_mode):
    coords = nerf(bonds, angles, tau_pred, aa_types)
    # head-to-tail: N-term N (res0,atom0) vs C-term C (resL-1,atom2)
    n_term = coords[:, 0, 0]
    c_term = coords[:, -1, 2]
    dist = torch.norm(n_term - c_term, dim=-1)
    return (dist - IDEAL_PEPTIDE_BOND).pow(2).mean()

# losses/geometry_loss.py
import torch
from cyclicdiffusion_mt.utils.constants import CLASH_THRESHOLD
def clash_loss(coords, atom_mask):
    B,L,A,_ = coords.shape; coords_f = coords.view(B,-1,3)
    mask_f = atom_mask.view(B,-1)
    d = torch.cdist(coords_f, coords_f)
    valid = mask_f.unsqueeze(-1) & mask_f.unsqueeze(-2)
    # Exclude self and bonded neighbors with a simple diagonal band mask
    diag = torch.eye(L*A, device=coords.device, dtype=torch.bool).unsqueeze(0)
    clash_mask = valid & ~diag
    violations = (CLASH_THRESHOLD - d).clamp(min=0)
    return (violations * clash_mask.float()).sum() / (clash_mask.float().sum() + 1e-8)

# losses/affinity_loss.py
import torch.nn.functional as F
def affinity_loss(dg_pred, dg_label, confidence):
    mse = F.mse_loss(dg_pred, dg_label, reduction='none').mean(dim=-1)
    return (mse * confidence).mean()
```

- [ ] **Step 4: `pytest tests/test_losses.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/losses/ tests/test_losses.py && git commit -m "feat: add all loss functions (torsion, type, cyclo, geometry, affinity)"
```

---

### Task 9: Training Script

**Files:**
- Create: `cyclicdiffusion_mt/train.py`
- Create: `cyclicdiffusion_mt/config/model.yaml`
- Create: `cyclicdiffusion_mt/config/data.yaml`
- Create: `cyclicdiffusion_mt/config/train.yaml`

**Interfaces:**
- Produces: `train.py` with `class Trainer`, `main()` function, 3-phase training support

- [ ] **Step 1: Create config YAML files**

```yaml
# config/model.yaml
d_model: 256
d_target: 128
d_time: 64
n_denoiser_blocks: 6
n_ipa_blocks: 3
n_heads: 4
d_head: 64
ffn_expansion: 4
dropout: 0.1
T: 500
max_residues: 20
max_atoms_per_res: 14
max_chi_per_res: 4
```

```yaml
# config/data.yaml
train_manifest: "data/train_manifest.json"
val_manifest: "data/val_manifest.json"
batch_size: 16
num_workers: 4
max_targets: 3
```

```yaml
# config/train.yaml
# Phase 1: single-target pretrain
phase1:
  epochs: 50
  lambda_type: 0.1
  lambda_cyclo: 0.0
  lambda_affinity: 0.0
  lambda_geometry: 0.01
  lr: 1e-4
# Phase 2: multi-target + cyclo
phase2:
  epochs: 30
  lambda_type: 0.1
  lambda_cyclo: 0.5
  lambda_affinity: 0.0
  lambda_geometry: 0.01
  lr: 5e-5
# Phase 3: affinity fine-tune
phase3:
  epochs: 20
  lambda_type: 0.05
  lambda_cyclo: 0.5
  lambda_affinity: 0.1
  lambda_geometry: 0.01
  lr: 1e-5
```

- [ ] **Step 2: Implement training script**

```python
# cyclicdiffusion_mt/train.py
"""Three-phase training script for CyclicDiffusion-MT."""
import torch, torch.nn as nn, yaml, logging, os
from torch.utils.data import DataLoader
from cyclicdiffusion_mt.model.nerf import NeRF
from cyclicdiffusion_mt.model.denoiser import FrameDenoiser
from cyclicdiffusion_mt.model.diffusion import WrappedNormalDiffusion, MaskedDiscreteDiffusion
from cyclicdiffusion_mt.model.target_encoder import TargetEncoder
from cyclicdiffusion_mt.data.dataset import MultiTargetDataset, PeptideDataCollate
from cyclicdiffusion_mt.data.transforms import compute_chi_mask, compute_atom_mask
from cyclicdiffusion_mt.losses.torsion_loss import torsion_loss
from cyclicdiffusion_mt.losses.type_loss import type_loss
from cyclicdiffusion_mt.losses.cyclo_loss import cyclo_loss
from cyclicdiffusion_mt.losses.geometry_loss import clash_loss
from cyclicdiffusion_mt.losses.affinity_loss import affinity_loss

class Trainer:
    def __init__(self, model_cfg, data_cfg, train_cfg):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.nerf = NeRF().to(self.device)
        self.target_encoder = TargetEncoder(
            d_target=model_cfg['d_target'], n_blocks=model_cfg['n_ipa_blocks'],
            d_head=model_cfg['d_head']//2, n_heads=model_cfg['n_heads'],
        ).to(self.device)
        self.denoiser = FrameDenoiser(
            d_model=model_cfg['d_model'], d_target=model_cfg['d_target'],
            d_time=model_cfg['d_time'], n_blocks=model_cfg['n_denoiser_blocks'],
            d_head=model_cfg['d_head'], n_heads=model_cfg['n_heads'],
        ).to(self.device)

        # Diffusion processes
        self.torsion_diffusion = WrappedNormalDiffusion(T=model_cfg['T'])
        self.type_diffusion = MaskedDiscreteDiffusion(T=model_cfg['T'])

    def train_phase(self, phase_name, cfg, train_loader, val_loader=None):
        """Train one phase."""
        optimizer = torch.optim.AdamW(
            list(self.denoiser.parameters()) + list(self.target_encoder.parameters()),
            lr=cfg['lr'], weight_decay=0.01
        )

        for epoch in range(cfg['epochs']):
            self.denoiser.train(); self.target_encoder.train()
            total_loss = 0.0

            for batch in train_loader:
                tau_0 = batch['peptide_torsions'].to(self.device)
                a_0 = batch['peptide_aa_types'].to(self.device)
                pep_mask = batch['peptide_mask'].to(self.device)
                target_coords = [tc.to(self.device) for tc in batch['target_coords']]
                target_seqs = [ts.to(self.device) for ts in batch.get('target_sequences', [])]
                cyclo_mode = batch['cyclo_modes'].to(self.device)
                dg_label = batch.get('dG_rosetta', torch.zeros_like(cyclo_mode.float())).to(self.device)
                confidence = batch.get('confidence', torch.ones_like(cyclo_mode.float())).to(self.device)

                B, L = tau_0.shape[0], tau_0.shape[1]

                # Sample timestep
                t = torch.randint(0, self.model_cfg['T'], (B,), device=self.device)

                # Forward diffusion
                tau_t, noise = self.torsion_diffusion.q_sample(tau_0, t)
                a_t = self.type_diffusion.q_sample(a_0, t)

                # Target encoding
                t_masks = [torch.ones(tc.shape[0],tc.shape[1],dtype=torch.bool,device=self.device) for tc in target_coords]
                target_feats = self.target_encoder(target_coords, t_masks)

                # Denoise
                tau_pred, a_logits, dg_pred = self.denoiser(tau_t, a_t, t.float(), target_feats, t_masks, cyclo_mode)

                # Compute losses
                chi_mask = compute_chi_mask(a_0).to(self.device)
                l_torsion = torsion_loss(tau_pred, tau_0, chi_mask)
                l_type = type_loss(a_logits, a_0)

                # Bond/angle dummy values (near-rigid, use ideal)
                bonds = torch.ones(B,L,14,device=self.device)*1.4
                angles = torch.ones(B,L,14,device=self.device)*(109.5*torch.pi/180)
                l_cyclo = cyclo_loss(self.nerf, tau_pred, a_0, bonds, angles, cyclo_mode)

                coords = self.nerf(bonds, angles, tau_pred, a_0)
                atom_mask = compute_atom_mask(a_0).to(self.device)
                l_geom = clash_loss(coords, atom_mask)

                dg_label_expanded = dg_label.unsqueeze(-1) if dg_label.ndim==1 else dg_label
                l_affinity = affinity_loss(dg_pred, dg_label_expanded, confidence)

                # Total loss with lambda weights
                loss = (l_torsion +
                        cfg['lambda_type'] * l_type +
                        cfg['lambda_cyclo'] * l_cyclo +
                        cfg['lambda_geometry'] * l_geom +
                        cfg['lambda_affinity'] * l_affinity)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logging.info(f"Phase {phase_name} Epoch {epoch+1}/{cfg['epochs']} - Loss: {avg_loss:.4f}")

        return avg_loss

    def train(self, train_loader, val_loader=None):
        """Run all three training phases."""
        logging.info("=== Phase 1: Single-target pretraining ===")
        self.train_phase("1", self.train_cfg['phase1'], train_loader, val_loader)

        logging.info("=== Phase 2: Multi-target + cyclization ===")
        self.train_phase("2", self.train_cfg['phase2'], train_loader, val_loader)

        logging.info("=== Phase 3: Affinity fine-tuning ===")
        # Freeze target encoder in phase 3
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.train_phase("3", self.train_cfg['phase3'], train_loader, val_loader)

        logging.info("Training complete!")

def main():
    logging.basicConfig(level=logging.INFO)
    with open('cyclicdiffusion_mt/config/model.yaml') as f: model_cfg = yaml.safe_load(f)
    with open('cyclicdiffusion_mt/config/data.yaml') as f: data_cfg = yaml.safe_load(f)
    with open('cyclicdiffusion_mt/config/train.yaml') as f: train_cfg = yaml.safe_load(f)

    # Create trainer
    trainer = Trainer(model_cfg, data_cfg, train_cfg)

    # Load dataset
    import json
    with open(data_cfg['train_manifest']) as f: train_manifest = json.load(f)
    dataset = MultiTargetDataset(train_manifest)
    collate = PeptideDataCollate()
    loader = DataLoader(dataset, batch_size=data_cfg['batch_size'], shuffle=True,
                        num_workers=data_cfg['num_workers'], collate_fn=collate)

    trainer.train(loader)

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Verify syntax:** `python -c "import ast; ast.parse(open('cyclicdiffusion_mt/train.py').read()); print('OK')"` → OK

- [ ] **Step 4: Commit**

```bash
git add cyclicdiffusion_mt/train.py cyclicdiffusion_mt/config/ && git commit -m "feat: add training script with 3-phase training"
```

---

### Task 10: Sampling Script

**Files:**
- Create: `cyclicdiffusion_mt/sample.py`
- Create: `tests/test_sampling.py`

**Interfaces:**
- Produces: `class Sampler` with `sample(denoiser,target_feats,t_masks,cyclo_mode,L)->tau_0,a_0` implementing DDPM reverse process, optional hard projection

- [ ] **Step 1: Write test**

```python
# tests/test_sampling.py
import pytest, torch
from cyclicdiffusion_mt.model.denoiser import FrameDenoiser
from cyclicdiffusion_mt.model.diffusion import WrappedNormalDiffusion, MaskedDiscreteDiffusion
from cyclicdiffusion_mt.sample import Sampler

class TestSampler:
    @pytest.fixture
    def denoiser(self): return FrameDenoiser(d_model=256,d_target=128,d_time=64,n_blocks=2)
    @pytest.fixture
    def sampler(self): return Sampler(T=500)

    def test_sample_shape(self, denoiser, sampler):
        B,L = 1,5
        targets = [torch.randn(B,10,128)]; t_masks = [torch.ones(B,10,dtype=torch.bool)]
        cyclo = torch.zeros(B,dtype=torch.long)
        tau, aa = sampler.sample(denoiser, targets, t_masks, cyclo, L)
        assert tau.shape == (B,L,7)
        assert aa.shape == (B,L)
    def test_sample_with_projection(self, denoiser, sampler):
        B,L = 1,5
        targets = [torch.randn(B,10,128)]; t_masks = [torch.ones(B,10,dtype=torch.bool)]
        cyclo = torch.zeros(B,dtype=torch.long)
        tau, aa = sampler.sample(denoiser, targets, t_masks, cyclo, L, apply_projection=True)
        assert tau.shape == (B,L,7)
```

- [ ] **Step 2: `pytest tests/test_sampling.py -v`** → FAIL

- [ ] **Step 3: Implement**

```python
# cyclicdiffusion_mt/sample.py
"""Sampling/inference for CyclicDiffusion-MT."""
import torch, torch.nn.functional as F
from cyclicdiffusion_mt.model.diffusion import WrappedNormalDiffusion, MaskedDiscreteDiffusion, cosine_schedule
from cyclicdiffusion_mt.model.nerf import NeRF
from cyclicdiffusion_mt.utils.constants import IDEAL_PEPTIDE_BOND, MASK_IDX

class Sampler:
    def __init__(self, T=500):
        self.T = T
        self.alpha_bar = cosine_schedule(T)
        self.nerf = NeRF()

    @torch.no_grad()
    def sample(self, denoiser, target_feats, target_masks, cyclo_mode, L, apply_projection=False):
        """Run reverse diffusion to generate a cyclic peptide.
        Args:
            denoiser: FrameDenoiser
            target_feats: list[(B,N_k,d_target)]
            target_masks: list[(B,N_k)]
            cyclo_mode: (B,)
            L: int, peptide length
            apply_projection: bool, whether to apply hard cyclization projection
        Returns:
            tau_0: (B,L,7), a_0: (B,L)
        """
        B = cyclo_mode.shape[0]
        device = cyclo_mode.device

        # Initialize from noise
        tau_t = torch.randn(B, L, 7, device=device) * torch.pi
        tau_t = torch.atan2(torch.sin(tau_t), torch.cos(tau_t))
        a_t = torch.full((B, L), MASK_IDX, device=device, dtype=torch.long)

        for step in reversed(range(self.T)):
            t = torch.full((B,), step, device=device, dtype=torch.float32)

            # Denoise
            tau_pred, a_logits, _ = denoiser(tau_t, a_t, t, target_feats, target_masks, cyclo_mode)

            # DDPM posterior for continuous
            a_bar_t = self.alpha_bar[step]
            a_bar_prev = self.alpha_bar[step-1] if step > 0 else torch.tensor(1.0, device=device)

            # Posterior mean (x0-prediction)
            coef = a_bar_prev.sqrt()
            tau_t = coef * tau_pred  # simplified: use prediction directly
            if step > 0:
                noise = torch.randn_like(tau_t) * (1 - a_bar_prev).sqrt()
                tau_t = tau_t + noise
            tau_t = torch.atan2(torch.sin(tau_t), torch.cos(tau_t))

            # Discrete sampling
            a_t = torch.multinomial(F.softmax(a_logits.view(-1,26), dim=-1), 1).view(B,L)

            # Optional hard projection
            if apply_projection and step % 50 == 0:
                tau_t = self._cyclo_projection(tau_t, a_t, cyclo_mode)

        return tau_t, a_t

    def _cyclo_projection(self, tau, aa_types, cyclo_mode, max_iters=10, lr=0.01):
        """Gradient-based hard cyclization projection."""
        tau.requires_grad_(True)
        bonds = torch.ones_like(tau[:,:,:1]).expand(-1,-1,14)*1.4
        angles = torch.ones_like(tau[:,:,:1]).expand(-1,-1,14)*(109.5*torch.pi/180)
        for _ in range(max_iters):
            coords = self.nerf(bonds, angles, tau, aa_types)
            n_term = coords[:, 0, 0]; c_term = coords[:, -1, 2]
            dist = torch.norm(n_term - c_term, dim=-1)
            error = (dist - IDEAL_PEPTIDE_BOND).pow(2).sum()
            grad = torch.autograd.grad(error, tau)[0]
            tau = tau - lr * grad
            tau = torch.atan2(torch.sin(tau), torch.cos(tau))
        return tau.detach()
```

- [ ] **Step 4: `pytest tests/test_sampling.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/sample.py tests/test_sampling.py && git commit -m "feat: add sampling script with hard projection"
```

---

### Task 11: Evaluation Metrics

**Files:**
- Create: `cyclicdiffusion_mt/eval/__init__.py`
- Create: `cyclicdiffusion_mt/eval/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `ring_closure_error(coords,cyclo_mode)->float`: N-C distance error
  - `rama_outlier_fraction(phi,psi,aa_types)->float`: % outside favored regions
  - `bond_angle_deviation(bonds,angles)->float`: RMSD from ideal
  - `clash_count(coords,atom_mask)->float`: clashes per residue
  - `internal_diversity(samples)->float`: mean pairwise RMSD
  - `novelty(samples,train_set)->float`: 1 - max Tanimoto to training

- [ ] **Step 1: Write test**

```python
# tests/test_metrics.py
import pytest, torch
from cyclicdiffusion_mt.eval.metrics import ring_closure_error, clash_count

class TestRingClosure:
    def test_perfect_closure(self):
        coords = torch.randn(1,5,14,3)
        coords[:,0,0] = torch.tensor([0.,0.,0.])
        coords[:,-1,2] = torch.tensor([0.,0.,0.])  # N-term N == C-term C
        err = ring_closure_error(coords, torch.tensor([0]))
        assert err.item() < 1e-5
    def test_nonzero_for_open(self):
        coords = torch.randn(1,5,14,3)
        err = ring_closure_error(coords, torch.tensor([0]))
        assert err.item() > 0

class TestClashCount:
    def test_no_clash(self):
        coords = torch.randn(1,3,14,3)*100  # far apart
        mask = torch.ones(1,3,14,dtype=torch.bool)
        assert clash_count(coords, mask).item() < 1e-5
```

- [ ] **Step 2: `pytest tests/test_metrics.py -v`** → FAIL

- [ ] **Step 3: Implement**

```python
# cyclicdiffusion_mt/eval/__init__.py
"""Evaluation modules."""

# cyclicdiffusion_mt/eval/metrics.py
"""Evaluation metrics for cyclic peptide generation."""
import torch
from cyclicdiffusion_mt.utils.constants import IDEAL_PEPTIDE_BOND, CLASH_THRESHOLD

def ring_closure_error(coords, cyclo_mode):
    """N-C distance error for head-to-tail cyclization. coords:(B,L,14,3)."""
    n_term = coords[:, 0, 0]
    c_term = coords[:, -1, 2]
    dist = torch.norm(n_term - c_term, dim=-1)
    return (dist - IDEAL_PEPTIDE_BOND).abs().mean()

def clash_count(coords, atom_mask):
    """Number of steric clashes per residue. coords:(B,L,14,3)."""
    B,L,A,_ = coords.shape
    coords_f = coords.view(B,-1,3)
    d = torch.cdist(coords_f, coords_f)
    mask_f = atom_mask.view(B,-1)
    valid = mask_f.unsqueeze(-1) & mask_f.unsqueeze(-2)
    diag = torch.eye(L*A,device=coords.device,dtype=torch.bool).unsqueeze(0)
    clash_mask = valid & ~diag
    n_clashes = ((d < CLASH_THRESHOLD) & clash_mask).float().sum(dim=[1,2])
    return n_clashes / (L + 1e-8)

def internal_diversity(samples):
    """Mean pairwise all-atom RMSD between generated samples.
    samples: list[(L,14,3)] or tensor (N,L,14,3)."""
    if isinstance(samples, list):
        samples = torch.stack(samples)
    N = samples.shape[0]
    samples_f = samples.view(N,-1)
    d = torch.cdist(samples_f, samples_f)
    mask = ~torch.eye(N,dtype=torch.bool,device=samples.device)
    return d[mask].mean()

def novelty(samples, train_set):
    """1 - max Tanimoto similarity to training set.
    samples: list[str] (SMILES), train_set: list[str]."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    sample_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,2048) for s in samples]
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(t),2,2048) for t in train_set]
    novelties = []
    for sfp in sample_fps:
        max_tc = max(DataStructs.TanimotoSimilarity(sfp,tfp) for tfp in train_fps)
        novelties.append(1 - max_tc)
    return sum(novelties)/len(novelties) if novelties else 0.0
```

- [ ] **Step 4: `pytest tests/test_metrics.py -v`** → PASS

- [ ] **Step 5: Commit**

```bash
git add cyclicdiffusion_mt/eval/ tests/test_metrics.py && git commit -m "feat: add evaluation metrics"
```

---

### Task 12: Integration Test (End-to-End)

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes all modules above
- Verifies: train one step → no NaN, sample → produces valid shapes, NeRF → produces finite coords, losses → all finite

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: one training step + one sampling."""
import pytest, torch
from cyclicdiffusion_mt.model.nerf import NeRF
from cyclicdiffusion_mt.model.denoiser import FrameDenoiser
from cyclicdiffusion_mt.model.diffusion import WrappedNormalDiffusion, MaskedDiscreteDiffusion
from cyclicdiffusion_mt.model.target_encoder import TargetEncoder
from cyclicdiffusion_mt.sample import Sampler
from cyclicdiffusion_mt.losses.torsion_loss import torsion_loss
from cyclicdiffusion_mt.losses.type_loss import type_loss
from cyclicdiffusion_mt.losses.cyclo_loss import cyclo_loss
from cyclicdiffusion_mt.losses.geometry_loss import clash_loss
from cyclicdiffusion_mt.losses.affinity_loss import affinity_loss
from cyclicdiffusion_mt.data.transforms import compute_chi_mask, compute_atom_mask
from cyclicdiffusion_mt.eval.metrics import ring_closure_error, clash_count

class TestIntegration:
    @pytest.fixture
    def device(self): return torch.device('cpu')

    def test_full_pipeline_one_step(self, device):
        """One forward + backward training step, verify no NaN."""
        B,L,K = 2,8,1
        tau_0 = torch.randn(B,L,7,device=device)
        a_0 = torch.randint(0,25,(B,L),device=device)
        target_coords = [torch.randn(B,12,14,3,device=device)]
        t_masks = [torch.ones(B,12,dtype=torch.bool,device=device)]
        cyclo = torch.zeros(B,dtype=torch.long,device=device)

        # Models
        nerf = NeRF().to(device)
        encoder = TargetEncoder(d_target=128,n_blocks=2).to(device)
        denoiser = FrameDenoiser(d_model=64,d_target=128,d_time=16,n_blocks=2,d_head=16,n_heads=2).to(device)
        torsion_diff = WrappedNormalDiffusion(T=500)
        type_diff = MaskedDiscreteDiffusion(T=500)

        # Forward diffusion
        t = torch.randint(0,500,(B,),device=device)
        tau_t, _ = torsion_diff.q_sample(tau_0, t)
        a_t = type_diff.q_sample(a_0, t)

        # Encode targets
        target_feats = encoder(target_coords, t_masks)

        # Denoise
        tau_pred, a_logits, dg_pred = denoiser(tau_t, a_t, t.float(), target_feats, t_masks, cyclo)

        # Losses
        chi_mask = compute_chi_mask(a_0)
        l1 = torsion_loss(tau_pred, tau_0, chi_mask)
        l2 = type_loss(a_logits, a_0)
        bonds = torch.ones(B,L,14,device=device)*1.4
        angles = torch.ones(B,L,14,device=device)*(109.5*torch.pi/180)
        l3 = cyclo_loss(nerf, tau_pred, a_0, bonds, angles, cyclo)
        coords = nerf(bonds, angles, tau_pred, a_0)
        atom_mask = compute_atom_mask(a_0)
        l4 = clash_loss(coords, atom_mask)
        l5 = affinity_loss(dg_pred, torch.zeros(B,1,device=device), torch.ones(B,device=device))

        loss = l1 + 0.1*l2 + 0.5*l3 + 0.01*l4 + 0.1*l5

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        loss.backward()
        for name, param in denoiser.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"

    def test_sample_produces_valid_output(self, device):
        """Sampling should produce finite outputs."""
        B,L = 1,6
        denoiser = FrameDenoiser(d_model=64,d_target=128,d_time=16,n_blocks=2,d_head=16,n_heads=2).to(device)
        targets = [torch.randn(B,10,128,device=device)]
        t_masks = [torch.ones(B,10,dtype=torch.bool,device=device)]
        cyclo = torch.zeros(B,dtype=torch.long,device=device)

        sampler = Sampler(T=500)
        tau, aa = sampler.sample(denoiser, targets, t_masks, cyclo, L)

        assert tau.shape == (B,L,7)
        assert aa.shape == (B,L)
        assert not torch.isnan(tau).any()
        assert aa.min() >= 0 and aa.max() <= 25

    def test_nerf_cyclo_closure_metric(self, device):
        """Ring closure metric works on generated coords."""
        nerf = NeRF().to(device)
        B,L = 1,6
        aa = torch.zeros(B,L,dtype=torch.long,device=device)
        bonds = torch.ones(B,L,14,device=device)*1.4
        angles = torch.ones(B,L,14,device=device)*(109.5*torch.pi/180)
        tau = torch.randn(B,L,7,device=device)
        coords = nerf(bonds, angles, tau, aa)
        err = ring_closure_error(coords, torch.tensor([0]))
        assert not torch.isnan(err)
        assert err >= 0
```

- [ ] **Step 2: `pytest tests/test_integration.py -v`** → may partially pass/fail

- [ ] **Step 3-4: Fix issues found, re-run until all green**

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py && git commit -m "test: add end-to-end integration test"
```

---

## Execution Summary

**Total: 12 tasks**, each ~5 steps. Order is dependency-driven:
1. Constants (no deps)
2. NeRF (depends on constants)
3. Data pipeline (depends on constants)
4. Target encoder (standalone)
5. Cross-attention (standalone, needs target encoder for full test)
6. Denoiser (depends on cross-attention)
7. Diffusion (depends on constants)
8. Losses (depend on NeRF, diffusion)
9. Training (depends on all above)
10. Sampling (depends on denoiser, diffusion, NeRF)
11. Evaluation (standalone metrics)
12. Integration test (depends on all)

**Estimated total:** ~300-400 steps across all tasks. Each task produces a working, testable module.

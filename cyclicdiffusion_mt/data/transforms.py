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
    n1 = torch.linalg.cross(b1,b2); n1 = n1/(torch.norm(n1)+1e-8)
    n2 = torch.linalg.cross(b2,b3); n2 = n2/(torch.norm(n2)+1e-8)
    m1 = torch.linalg.cross(n1, b2/(torch.norm(b2)+1e-8))
    x = (n1*n2).sum(); y = (m1*n2).sum()
    return torch.atan2(y, x)

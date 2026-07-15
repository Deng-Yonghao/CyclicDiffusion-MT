"""Ideal bond length and angle tensor builders for NeRF input."""

import torch
import math
from cyclicdiffusion_mt.utils.constants import (
    AA_ATOM_NAMES, IDX_TO_AA, MAX_ATOMS_PER_RES,
    IDEAL_BOND_LENGTHS, IDEAL_BOND_ANGLES,
)

# Per-residue ideal bond lengths for each atom position (1-indexed, atom 0 has no bond)
# Keyed by atom name tuple: (prev_atom, this_atom)
_BOND_MAP = {
    # Backbone
    ('N', 'CA'): IDEAL_BOND_LENGTHS['N-CA'],
    ('CA', 'C'): IDEAL_BOND_LENGTHS['CA-C'],
    ('C', 'O'): IDEAL_BOND_LENGTHS['C-O'],
    ('C', 'N'): IDEAL_BOND_LENGTHS['C-N'],
    # Sidechain common
    ('CA', 'CB'): IDEAL_BOND_LENGTHS['CA-CB'],
}

# Ideal bond angles in degrees -> radians
_ANGLE_TABLE = {
    'N-CA-C': math.radians(IDEAL_BOND_ANGLES['N-CA-C']),
    'CA-C-N': math.radians(IDEAL_BOND_ANGLES['CA-C-N']),
    'C-N-CA': math.radians(IDEAL_BOND_ANGLES['C-N-CA']),
    'N-CA-CB': math.radians(IDEAL_BOND_ANGLES['N-CA-CB']),
    'CA-C-O': math.radians(IDEAL_BOND_ANGLES['CA-C-O']),
}

# Default bond length for atoms not in _BOND_MAP
_DEFAULT_BOND = 1.52
_DEFAULT_ANGLE = math.radians(109.5)


def build_ideal_bonds(aa_types):
    """Build ideal bond lengths tensor from AA types.

    Args:
        aa_types: (B, L) amino acid type indices.

    Returns:
        bonds: (B, L, MAX_ATOMS_PER_RES) bond lengths in Angstroms.
            Position 0 is always 0 (no preceding atom). Positions beyond
            the residue's atom count are 0.
    """
    B, L = aa_types.shape
    bonds = torch.zeros(B, L, MAX_ATOMS_PER_RES, device=aa_types.device)

    for b in range(B):
        for l in range(L):
            aa_idx = aa_types[b, l].item()
            aa_name = IDX_TO_AA.get(aa_idx, 'ALA')
            atom_names = AA_ATOM_NAMES.get(aa_name, AA_ATOM_NAMES['ALA'])
            n_atoms = len(atom_names)

            for a in range(1, n_atoms):
                key = (atom_names[a - 1], atom_names[a])
                bonds[b, l, a] = _BOND_MAP.get(key, _DEFAULT_BOND)

    return bonds


def build_ideal_angles(aa_types):
    """Build ideal bond angles tensor from AA types.

    Args:
        aa_types: (B, L) amino acid type indices.

    Returns:
        angles: (B, L, MAX_ATOMS_PER_RES) bond angles in radians.
            Positions 0 and 1 are 0 (need 3 atoms for an angle).
            Positions beyond the residue's atom count are 0.
    """
    B, L = aa_types.shape
    angles = torch.zeros(B, L, MAX_ATOMS_PER_RES, device=aa_types.device)

    for b in range(B):
        for l in range(L):
            aa_idx = aa_types[b, l].item()
            aa_name = IDX_TO_AA.get(aa_idx, 'ALA')
            atom_names = AA_ATOM_NAMES.get(aa_name, AA_ATOM_NAMES['ALA'])
            n_atoms = len(atom_names)

            for a in range(2, n_atoms):
                # Build angle name from atom triplet
                angle_name = f"{atom_names[a-2]}-{atom_names[a-1]}-{atom_names[a]}"
                angles[b, l, a] = _ANGLE_TABLE.get(angle_name, _DEFAULT_ANGLE)

    return angles


def build_ideal_geometry(aa_types):
    """Convenience: return both ideal bonds and angles for a batch of AA types."""
    return build_ideal_bonds(aa_types), build_ideal_angles(aa_types)

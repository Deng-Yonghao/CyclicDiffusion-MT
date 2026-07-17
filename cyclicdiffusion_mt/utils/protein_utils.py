"""PDB parsing and internal coordinate extraction for cyclic peptide data.

Handles the CPCore PDB format: chain L = cyclic peptide (ACE/NME capped),
chain R = target protein.
"""

import torch
from cyclicdiffusion_mt.utils.constants import (
    AA_TO_IDX, AA_ATOM_NAMES, MAX_ATOMS_PER_RES,
    NUM_TORSIONS, MAX_CHI_PER_RES,
)

# Standard backbone + sidechain atom order per residue (14 positions)
PDB_ATOM_ORDER = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'SD', 'SG',
                  'CD1', 'CD2', 'CE1']

# Map unusual PDB atom names to standard ones.
# H1/H2/H3 on backbone N are stripped (they're in capping residues only).
ATOM_NAME_MAP = {
    'H': 'N', 'HA': 'CA', 'H1': 'N', 'H2': 'N', 'H3': 'N',
    'HA2': 'CA', 'HA3': 'CA', 'HB1': 'CB', 'HB2': 'CB', 'HB3': 'CB',
    'HG1': 'CG', 'HG2': 'CG', 'HG3': 'CG',
    'HD1': 'CD', 'HD2': 'CD', 'HD3': 'CD',
    'HE1': 'CE', 'HE2': 'CE', 'HE3': 'CE',
    'HH': 'OH',
    # Glycine-specific
    '1H': 'N', '2H': 'N', '3H': 'N',
    # Terminal capping atoms (ACE/NME) — not standard AA atoms
    'CH3': None, 'HH31': None, 'HH32': None, 'HH33': None,
}

# Residue names to skip (capping groups, non-standard)
CAP_RESIDUE_NAMES = {'ACE', 'NME', 'NH2', 'CHO', 'AC'}

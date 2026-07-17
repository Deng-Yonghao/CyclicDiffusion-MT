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


def _parse_pdb_lines(pdb_path):
    """Read PDB file, return list of ATOM/HETATM line dicts.

    Each dict has keys: record_type, atom_name, res_name, chain_id,
    res_seq, x, y, z, element.
    """
    records = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            records.append({
                'record_type': line[0:6].strip(),
                'atom_name': line[12:16].strip(),
                'res_name': line[17:20].strip(),
                'chain_id': line[21].strip(),
                'res_seq': int(line[22:26].strip()),
                'x': float(line[30:38].strip()),
                'y': float(line[38:46].strip()),
                'z': float(line[46:54].strip()),
                'element': line[76:78].strip(),
            })
    return records


def extract_peptide_residues(pdb_path, chain_id='L'):
    """Extract cyclic peptide residues from PDB, stripping ACE/NME caps.

    Args:
        pdb_path: path to PDB file.
        chain_id: chain identifier for the peptide (default 'L').

    Returns:
        list of dicts, each with:
            res_name: str (3-letter AA code)
            res_seq_num: int (original PDB residue number)
            atoms: dict[str, tuple(float, float, float)]  atom_name -> (x,y,z)
    """
    records = _parse_pdb_lines(pdb_path)
    # Filter to target chain and real amino acids (skip caps)
    peptide_records = [
        r for r in records
        if r['chain_id'] == chain_id
        and r['res_name'] not in CAP_RESIDUE_NAMES
    ]
    # Group by residue sequence number
    residues = {}
    for r in peptide_records:
        seq = r['res_seq']
        if seq not in residues:
            residues[seq] = {
                'res_name': r['res_name'],
                'res_seq_num': seq,
                'atoms': {},
            }
        # Store all atoms — no hydrogen filter. ATOM_NAME_MAP handles
        # mapping 'H' (backbone amide hydrogen) to 'N' downstream.
        residues[seq]['atoms'][r['atom_name']] = (r['x'], r['y'], r['z'])

    # Sort by sequence number and return
    return [residues[s] for s in sorted(residues.keys())]


def extract_target_coords(pdb_path, chain_id='R'):
    """Extract target protein coordinates in standardized 14-atom order.

    Args:
        pdb_path: path to PDB file.
        chain_id: chain identifier for the target (default 'R').

    Returns:
        coords: (N, 14, 3) tensor of atom coordinates.
        aa_sequence: (N,) tensor of amino acid type indices (0-24).
    """
    records = _parse_pdb_lines(pdb_path)
    target_records = [r for r in records if r['chain_id'] == chain_id]

    # Group by residue
    by_res = {}
    for r in target_records:
        seq = r['res_seq']
        if seq not in by_res:
            by_res[seq] = {
                'res_name': r['res_name'],
                'atoms': {},
            }
        # Store all atoms — no hydrogen filter. ATOM_NAME_MAP handles
        # mapping hydrogen names ('H', 'HA', 'HB1', etc.) to their
        # corresponding heavy-atom names during coordinate extraction.
        by_res[seq]['atoms'][r['atom_name']] = (r['x'], r['y'], r['z'])

    sorted_seqs = sorted(by_res.keys())
    N = len(sorted_seqs)
    coords = torch.zeros(N, MAX_ATOMS_PER_RES, 3)
    aa_sequence = torch.zeros(N, dtype=torch.long)

    for i, seq in enumerate(sorted_seqs):
        info = by_res[seq]
        aa_name = info['res_name']
        aa_sequence[i] = AA_TO_IDX.get(aa_name, AA_TO_IDX['ALA'])

        # Get standard atom order for this residue type
        standard_atoms = AA_ATOM_NAMES.get(aa_name, AA_ATOM_NAMES['ALA'])
        for a_idx, std_name in enumerate(standard_atoms):
            if a_idx >= MAX_ATOMS_PER_RES:
                break
            # Try exact match first, then mapped name
            if std_name in info['atoms']:
                coords[i, a_idx] = torch.tensor(info['atoms'][std_name])
            else:
                # Try to find by mapped name
                found = False
                for pdb_name, (x, y, z) in info['atoms'].items():
                    mapped = ATOM_NAME_MAP.get(pdb_name, pdb_name)
                    if mapped == std_name:
                        coords[i, a_idx] = torch.tensor([x, y, z])
                        found = True
                        break

    return coords, aa_sequence

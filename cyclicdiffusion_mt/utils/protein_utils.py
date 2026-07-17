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


def _dihedral(r1, r2, r3, r4):
    """Compute dihedral angle (radians) from 4 3D points.

    Args:
        r1, r2, r3, r4: (3,) tensors.

    Returns:
        float: dihedral angle in [-pi, pi).
    """
    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1 = n1 / (torch.norm(n1) + 1e-8)
    n2 = n2 / (torch.norm(n2) + 1e-8)
    b2_n = b2 / (torch.norm(b2) + 1e-8)
    m1 = torch.linalg.cross(n1, b2_n)
    x = -(n1 * n2).sum()
    y = (m1 * n2).sum()
    return torch.atan2(y, x)


def _atom_coord_3d(residue, atom_name):
    """Get 3D coordinates for a named atom in a residue dict, or None."""
    pos = residue['atoms'].get(atom_name)
    if pos is not None:
        return torch.tensor(pos)
    # Try mapped names
    for pdb_name, pos2 in residue['atoms'].items():
        mapped = ATOM_NAME_MAP.get(pdb_name, pdb_name)
        if mapped == atom_name:
            return torch.tensor(pos2)
    return None


def compute_backbone_torsions(residues):
    """Compute backbone torsion angles (phi, psi, omega) for all residues.

    Handles cyclic wrap-around: phi of first residue uses C of last residue;
    psi and omega of last residue use N and CA of first residue.

    Args:
        residues: list of residue dicts from extract_peptide_residues.

    Returns:
        phi: (L,) tensor of phi angles (C_{i-1} - N_i - CA_i - C_i).
        psi: (L,) tensor of psi angles (N_i - CA_i - C_i - N_{i+1}).
        omega: (L,) tensor of omega angles (CA_i - C_i - N_{i+1} - CA_{i+1}).
    """
    L = len(residues)
    phi = torch.zeros(L)
    psi = torch.zeros(L)
    omega = torch.zeros(L)

    for i in range(L):
        # Get atoms for current residue
        c_i = _atom_coord_3d(residues[i], 'C')
        n_i = _atom_coord_3d(residues[i], 'N')
        ca_i = _atom_coord_3d(residues[i], 'CA')

        # PHI: C_{i-1} - N_i - CA_i - C_i
        prev_idx = (i - 1) % L  # cyclic wrap-around
        c_prev = _atom_coord_3d(residues[prev_idx], 'C')
        if c_prev is not None and n_i is not None and ca_i is not None and c_i is not None:
            phi[i] = _dihedral(c_prev, n_i, ca_i, c_i)

        # PSI: N_i - CA_i - C_i - N_{i+1}
        next_idx = (i + 1) % L
        n_next = _atom_coord_3d(residues[next_idx], 'N')
        if n_i is not None and ca_i is not None and c_i is not None and n_next is not None:
            psi[i] = _dihedral(n_i, ca_i, c_i, n_next)

        # OMEGA: CA_i - C_i - N_{i+1} - CA_{i+1}
        ca_next = _atom_coord_3d(residues[next_idx], 'CA')
        if ca_i is not None and c_i is not None and n_next is not None and ca_next is not None:
            omega[i] = _dihedral(ca_i, c_i, n_next, ca_next)

    return phi, psi, omega


def _compute_sidechain_chi(residue, aa_name):
    """Compute sidechain chi angles for one residue.

    Follows standard chi definitions using the atoms listed in
    AA_ATOM_NAMES. Returns a (4,) tensor; unused chi slots are zero.

    For each chi_k, the dihedral involves atoms at backbone positions and
    sidechain positions:
      chi1: N - CA - CB - CG   (or atoms[0], atoms[1], atoms[4], atoms[5])
      chi2: CA - CB - CG - CD  (or atoms[1], atoms[4], atoms[5], atoms[6])
      chi3: CB - CG - CD - CE  (or atoms[4], atoms[5], atoms[6], atoms[7])
      chi4: CG - CD - CE - NZ  (or atoms[5], atoms[6], atoms[7], atoms[8])
    Exact atom tuples depend on residue type.
    """
    from cyclicdiffusion_mt.utils.constants import AA_CHI_COUNTS, AA_TO_IDX

    chi = torch.zeros(4)
    aa_idx = AA_TO_IDX.get(aa_name, 0)
    n_chi = AA_CHI_COUNTS[aa_idx]

    std_atoms = AA_ATOM_NAMES.get(aa_name, AA_ATOM_NAMES['ALA'])
    # Build ordered list of atoms with coordinates
    atom_coords = []
    for name in std_atoms:
        c = _atom_coord_3d(residue, name)
        if c is not None:
            atom_coords.append(c)

    # Chi angle definitions per atom-position indices
    chi_atom_indices = [
        (0, 1, 4, 5),  # chi1: N-CA-CB-CG
        (1, 4, 5, 6),  # chi2: CA-CB-CG-CD
        (4, 5, 6, 7),  # chi3: CB-CG-CD-CE
        (5, 6, 7, 8),  # chi4: CG-CD-CE-NZ (approximate)
    ]

    for k in range(n_chi):
        if k >= 4:
            break
        indices = chi_atom_indices[k]
        if all(idx < len(atom_coords) for idx in indices):
            chi[k] = _dihedral(
                atom_coords[indices[0]],
                atom_coords[indices[1]],
                atom_coords[indices[2]],
                atom_coords[indices[3]],
            )

    return chi


def residues_to_torsions(residues):
    """Convert parsed peptide residues to torsion angles and AA types.

    Args:
        residues: list of residue dicts from extract_peptide_residues.

    Returns:
        torsions: (L, 7) tensor [phi, psi, omega, chi1, chi2, chi3, chi4].
        aa_types: (L,) tensor of AA type indices (0-24).
    """
    L = len(residues)
    torsions = torch.zeros(L, 7)
    aa_types = torch.zeros(L, dtype=torch.long)

    # Backbone torsions (handles cyclic wrap-around)
    phi, psi, omega = compute_backbone_torsions(residues)

    for i, res in enumerate(residues):
        aa_name = res['res_name']
        aa_types[i] = AA_TO_IDX.get(aa_name, AA_TO_IDX['ALA'])

        torsions[i, 0] = phi[i]
        torsions[i, 1] = psi[i]
        torsions[i, 2] = omega[i]

        # Sidechain chi angles
        chi = _compute_sidechain_chi(res, aa_name)
        torsions[i, 3:7] = chi

    return torsions, aa_types


def parse_pdb_cyclic(pdb_path, cyclo_mode=0, dG_rosetta=0.0, confidence=1.0):
    """Parse one CPCore PDB file into a manifest-ready dict.

    Extracts:
      - Cyclic peptide (chain L): torsion angles (L,7), AA types (L,)
      - Target protein (chain R): coordinates (N,14,3), sequence (N,)
      - Strips ACE/NME capping residues from peptide

    Args:
        pdb_path: path to PDB file.
        cyclo_mode: cyclization mode index {0..4}, default 0 (head-to-tail).
        dG_rosetta: Rosetta binding energy (kcal/mol). Set to 0.0 if unknown.
        confidence: data quality confidence weight in [0, 1].

    Returns:
        dict with keys: peptide_torsions (L,7), peptide_aa_types (L,),
            target_coords [(N,14,3)], target_sequences [(N,)],
            cyclo_mode, dG_rosetta, confidence.
    """
    residues = extract_peptide_residues(pdb_path, chain_id='L')
    torsions, aa_types = residues_to_torsions(residues)
    target_coords, target_seq = extract_target_coords(pdb_path, chain_id='R')

    return {
        'peptide_torsions': torsions,
        'peptide_aa_types': aa_types,
        'target_coords': [target_coords],
        'target_sequences': [target_seq],
        'cyclo_mode': cyclo_mode,
        'dG_rosetta': dG_rosetta,
        'confidence': confidence,
    }

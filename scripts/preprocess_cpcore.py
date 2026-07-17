#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess CPCore PDB files into manifest.json for CyclicDiffusion-MT.

Reads all PDB files from the CPCore directory, extracts peptide torsions
and target coordinates, merges with property data from TSV files, and
outputs a manifest.json consumable by MultiTargetDataset.

Usage:
    python scripts/preprocess_cpcore.py \
        --pdb_dir CPCore/CPCore_pdb \
        --props_dir CPCore/CPCore_properties \
        --output data/manifest.json \
        --max_samples 1000
"""

import argparse
import json
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_tsv_dict(tsv_path, key_col='id'):
    """Load a TSV file into a dict keyed by the id column.

    Args:
        tsv_path: path to tab-separated file with header.
        key_col: column name to use as key.

    Returns:
        dict[str, dict]: id -> row data dict.
    """
    result = {}
    if not os.path.exists(tsv_path):
        return result
    with open(tsv_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split('\t')
            row = dict(zip(header, values))
            key = row.get(key_col, '')
            if key:
                result[key] = row
    return result


def merge_properties(pdb_id, basic_dict, affinity_dict, validity_dict, hydro_dict):
    """Merge property data from CPCore TSV files for one PDB entry.

    Args:
        pdb_id: the PDB identifier (matches TSV id column).
        basic_dict: from CPCore_Basic.tsv.
        affinity_dict: from CPCore_Affinity.tsv.
        validity_dict: from CPCore_Validity.tsv.
        hydro_dict: from CPCore_Hydrophobic.tsv.

    Returns:
        dict with keys: dG_rosetta, confidence, cyclo_mode.
    """
    # Affinity
    aff = affinity_dict.get(pdb_id, {})
    dg = float(aff.get('rosetta_dG', 0.0))

    # Basic properties
    basic = basic_dict.get(pdb_id, {})

    # Cyclization mode mapping
    cyclic_type = basic.get('cyclic_type', 'HEADTAIL')
    cyclo_map = {'HEADTAIL': 0, 'ISOPEPTIDE': 1}
    cyclo_mode = cyclo_map.get(cyclic_type, 0)

    # Confidence from validity metrics
    valid = validity_dict.get(pdb_id, {})
    accept_rate = float(valid.get('accept_rate', 1.0))
    favoured_rate = float(valid.get('favoured_rate', 1.0))
    hbonds = int(valid.get('hydrogen_bonds', 0))

    base_conf = accept_rate * favoured_rate
    hbond_boost = min(hbonds / 10.0, 0.2)
    confidence = min(base_conf + hbond_boost, 1.0)

    return {
        'dG_rosetta': dg,
        'cyclo_mode': cyclo_mode,
        'confidence': confidence,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CPCore PDB files into manifest.json'
    )
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Path to CPCore_pdb directory')
    parser.add_argument('--props_dir', type=str, required=True,
                        help='Path to CPCore_properties directory')
    parser.add_argument('--output', type=str, default='data/manifest.json',
                        help='Output path for manifest.json')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--max_residues', type=int, default=20,
                        help='Maximum peptide residues')
    args = parser.parse_args()

    # Load property databases
    print("Loading property databases...")
    basic_dict = load_tsv_dict(os.path.join(args.props_dir, 'CPCore_Basic.tsv'))
    affinity_dict = load_tsv_dict(os.path.join(args.props_dir, 'CPCore_Affinity.tsv'))
    validity_dict = load_tsv_dict(os.path.join(args.props_dir, 'CPCore_Validity.tsv'))
    hydro_dict = load_tsv_dict(os.path.join(args.props_dir, 'CPCore_Hydrophobic.tsv'))
    print(f"  Loaded {len(basic_dict)} basic, {len(affinity_dict)} affinity, "
          f"{len(validity_dict)} validity, {len(hydro_dict)} hydrophobic entries")

    # Find all PDB files
    pdb_files = sorted([f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')])
    print(f"\nFound {len(pdb_files)} PDB files in {args.pdb_dir}")

    if args.max_samples:
        pdb_files = pdb_files[:args.max_samples]
        print(f"  Limiting to {args.max_samples} samples")

    from cyclicdiffusion_mt.utils.protein_utils import parse_pdb_cyclic

    # Process each PDB
    manifest = []
    skipped = 0
    for i, pdb_file in enumerate(pdb_files):
        pdb_path = os.path.join(args.pdb_dir, pdb_file)
        pdb_id = pdb_file.replace('.pdb', '')

        try:
            props = merge_properties(
                pdb_id, basic_dict, affinity_dict, validity_dict, hydro_dict
            )

            entry = parse_pdb_cyclic(
                pdb_path,
                cyclo_mode=props['cyclo_mode'],
                dG_rosetta=props['dG_rosetta'],
                confidence=props['confidence'],
            )

            L = entry['peptide_torsions'].shape[0]
            if L > args.max_residues:
                skipped += 1
                continue

            manifest_entry = {
                'peptide_torsions': entry['peptide_torsions'].tolist(),
                'peptide_aa_types': entry['peptide_aa_types'].tolist(),
                'target_coords': [tc.tolist() for tc in entry['target_coords']],
                'target_sequences': [ts.tolist() for ts in entry['target_sequences']],
                'cyclo_mode': entry['cyclo_mode'],
                'dG_rosetta': entry['dG_rosetta'],
                'confidence': entry['confidence'],
            }
            manifest.append(manifest_entry)

        except Exception as e:
            print(f"  WARNING: Failed to process {pdb_file}: {e}")
            skipped += 1
            continue

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(pdb_files)}...")

    # Write manifest
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f)

    print(f"\nDone! Wrote {len(manifest)} entries to {args.output}")
    print(f"  Skipped: {skipped} (too long or errors)")
    if manifest:
        avg_len = sum(len(e['peptide_torsions']) for e in manifest) / len(manifest)
        print(f"  Average peptide length: {avg_len:.1f}")


if __name__ == '__main__':
    main()

"""
OR51E2 (PDB 8F76) binding pocket analysis.

Extracts:
- Propionate (PPI) coordinates
- Residues within 5 Å of propionate
- Pocket volume estimate
- Key residue distances (R262, F155, L158)

Reference: Billesbølle et al. 2023, Nature 615:535-540
"""

import json
from pathlib import Path
import numpy as np

PDB_FILE = Path(__file__).parent / "8F76.pdb"
LIGAND_RESNAME = "PPI"  # Propanoic acid

# Element-name short forms for parsing PDB
ELEMENT_RADII = {  # Bondi vdW radii (Å)
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
}


def parse_pdb_atoms(pdb_path):
    """Parse ATOM/HETATM records."""
    atoms = []
    for line in Path(pdb_path).read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            atoms.append({
                "record": line[:6].strip(),
                "atom_name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21],
                "resnum": int(line[22:26]),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "element": line[76:78].strip(),
            })
        except (ValueError, IndexError):
            continue
    return atoms


def main():
    atoms = parse_pdb_atoms(PDB_FILE)
    print(f"Total atoms parsed: {len(atoms)}")

    # 1. Extract propionate
    ligand_atoms = [a for a in atoms if a["resname"] == LIGAND_RESNAME]
    print(f"\n=== Propionate (PPI) ligand ===")
    for a in ligand_atoms:
        print(f"  {a['atom_name']:>4} {a['element']:>2}  "
              f"({a['x']:8.3f}, {a['y']:8.3f}, {a['z']:8.3f})")
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in ligand_atoms])
    lig_center = lig_coords.mean(axis=0)
    print(f"\nLigand center: ({lig_center[0]:.2f}, {lig_center[1]:.2f}, "
          f"{lig_center[2]:.2f})")

    # 2. Find pocket residues (any atom within 5 Å of any ligand atom)
    cutoff = 5.0
    pocket_residues = {}
    for a in atoms:
        if a["record"] != "ATOM" or a["chain"] != "A":
            continue
        coord = np.array([a['x'], a['y'], a['z']])
        for lig in lig_coords:
            d = np.linalg.norm(coord - lig)
            if d <= cutoff:
                key = (a['resnum'], a['resname'])
                if key not in pocket_residues:
                    pocket_residues[key] = {
                        'min_dist': d, 'closest_atom': a['atom_name']
                    }
                elif d < pocket_residues[key]['min_dist']:
                    pocket_residues[key] = {
                        'min_dist': d, 'closest_atom': a['atom_name']
                    }
                break

    print(f"\n=== Pocket residues (≤ {cutoff} Å from propionate) ===")
    sorted_res = sorted(pocket_residues.items(), key=lambda x: x[0][0])
    for (resnum, resname), info in sorted_res:
        marker = ""
        if resnum == 262 and resname == "ARG":
            marker = "  ← ANCHOR (carboxylate H-bond/ionic)"
        elif resnum == 155 and resname == "PHE":
            marker = "  ← GATE (size restrict)"
        elif resnum == 158 and resname == "LEU":
            marker = "  ← GATE (size restrict)"
        elif resnum in (250, 251):
            marker = "  ← Activation connector"
        print(f"  {resname}{resnum:<4} dist={info['min_dist']:.2f} Å "
              f"(closest atom: {info['closest_atom']}){marker}")

    # 3. Pocket volume estimate (convex hull of pocket atoms within 5 Å)
    pocket_atom_coords = []
    for a in atoms:
        if a["record"] != "ATOM" or a["chain"] != "A":
            continue
        coord = np.array([a['x'], a['y'], a['z']])
        for lig in lig_coords:
            if np.linalg.norm(coord - lig) <= 5.0:
                pocket_atom_coords.append(coord)
                break

    pocket_atom_coords = np.array(pocket_atom_coords)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pocket_atom_coords)
        print(f"\n=== Pocket volume estimate ===")
        print(f"  Convex hull of {len(pocket_atom_coords)} atoms")
        print(f"  Volume ≈ {hull.volume:.1f} Å³")
        print(f"  (Reference: Billesbølle 2023 reports ~31 Å³ for ligand-accessible cavity)")
    except ImportError:
        print("scipy not available — install for volume calc")

    # 4. Distance from R262 NH groups to ligand carboxylate (key interaction)
    r262_nh_atoms = [a for a in atoms
                     if a['resnum'] == 262 and a['resname'] == 'ARG'
                     and a['atom_name'] in ('NH1', 'NH2', 'NE')]
    ligand_o_atoms = [a for a in ligand_atoms if a['element'] == 'O']

    print(f"\n=== R262–propionate carboxylate interactions ===")
    for r in r262_nh_atoms:
        rc = np.array([r['x'], r['y'], r['z']])
        for o in ligand_o_atoms:
            oc = np.array([o['x'], o['y'], o['z']])
            d = np.linalg.norm(rc - oc)
            print(f"  R262-{r['atom_name']:>3} ↔ PPI-{o['atom_name']}  "
                  f"= {d:.2f} Å")

    # 5. Save summary JSON
    summary = {
        "pdb_id": "8F76",
        "ligand": "Propionate (PPI)",
        "ligand_center": lig_center.tolist(),
        "pocket_residues": [
            {
                "resname": rn, "resnum": rnum,
                "min_dist_A": round(info['min_dist'], 2),
                "closest_atom": info['closest_atom'],
            }
            for (rnum, rn), info in sorted_res
        ],
        "key_residues": {
            "anchor": "R262 (TM6, position 6x59)",
            "gates": ["F155 (TM4, 4x57)", "L158 (TM4, 4x60)"],
            "activation": ["F250 (6x47)", "Y251 (6x48)"],
        },
    }
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pocket_atom_coords)
        summary["pocket_volume_A3_convex"] = round(float(hull.volume), 1)
    except Exception:
        pass

    out_json = Path(__file__).parent / "or51e2_pocket_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved: {out_json}")


if __name__ == "__main__":
    main()

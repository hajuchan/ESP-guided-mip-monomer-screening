"""
Draw ESP maps for Hexanal-MAA and Nonanal-MAA complexes
using the existing generate_esp_map() from stage2_dft.

Custom label produces title like "Hexanal + MAA Complex ESP".
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

from rdkit import Chem
from rdkit.Geometry import Point3D

from pipeline.stage2_dft import (
    generate_esp_map, smiles_to_mol3d, mol_to_pyscf_atom,
)
from pipeline.config import SOLVENTS, MONOMER_LIBRARY, DFT_SP_BASIS


def rebuild_complex_atom_str(template_smiles, monomer_smiles, complex_coords):
    """Build PySCF atom string from Stage 1 complex_coords."""
    template_mol = smiles_to_mol3d(template_smiles)
    monomer_mol = smiles_to_mol3d(monomer_smiles)
    combo = Chem.CombineMols(template_mol, monomer_mol)
    combo = Chem.RWMol(combo)
    conf = combo.GetConformer()
    for i, (sym, x, y, z) in enumerate(complex_coords):
        if i < combo.GetNumAtoms():
            conf.SetAtomPosition(i, Point3D(x, y, z))
    return mol_to_pyscf_atom(combo.GetMol())


def main():
    eps = SOLVENTS["Acetonitrile"]
    basis = DFT_SP_BASIS
    monomer_smiles = MONOMER_LIBRARY["MAA"]

    targets = [
        ("hexanal", "Hexanal", "CCCCCC=O"),
        ("nonanal", "Nonanal", "CCCCCCCCC=O"),
    ]

    for tmpl_dir, tmpl_name, tmpl_smiles in targets:
        s1_path = Path(f"results/{tmpl_dir}/stage1/stage1_all.json")
        with open(s1_path) as f:
            s1 = json.load(f)
        entry = next((d for d in s1 if d["name"] == "MAA"), None)
        if not entry or not entry.get("complex_coords"):
            print(f"Skipping {tmpl_dir}: MAA complex_coords missing")
            continue

        atom_str = rebuild_complex_atom_str(
            tmpl_smiles, monomer_smiles, entry["complex_coords"])

        out_dir = f"results/{tmpl_dir}/stage2"
        # Label becomes plot title — "{Template} + MAA Complex"
        label = f"{tmpl_name} + MAA Complex"

        print(f"\n=== Generating ESP map: {label} "
              f"({len(entry['complex_coords'])} atoms, dE={entry['dE_kcal']:.3f}) ===")

        generate_esp_map(atom_str, basis, eps, label, out_dir)

        # Rename to match existing convention: esp_<Template>_MAA_complex_3d.html
        from pathlib import Path as P
        out = P(out_dir)
        src_html = out / f"stage2_esp_{label}_3d.html"
        src_png = out / f"stage2_esp_{label}.png"
        dst_html = out / f"esp_{tmpl_name}_MAA_complex_3d.html"
        dst_png = out / f"esp_{tmpl_name}_MAA_complex.png"
        if src_html.exists():
            src_html.rename(dst_html)
            print(f"  → {dst_html}")
        if src_png.exists():
            src_png.rename(dst_png)
            print(f"  → {dst_png}")


if __name__ == "__main__":
    main()

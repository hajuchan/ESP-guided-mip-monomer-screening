"""
Draw template-monomer binding interaction (Singh 2012 style).

Shows ball-and-stick of both molecules, highlighting the closest H-bond
contact between heteroatoms (N, O) and connecting H atoms with a dashed line.

Reference: Singh et al. 2012 — "Computational and Experimental Studies of
Molecularly Imprinted Polymers for Organochlorine Pesticides Heptachlor and DDT"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

import numpy as np
import plotly.graph_objects as go


# CPK colors and vdW radii (Bondi 1964) — Å
ELEM_INFO = {
    1:  ('H',  '#FFFFFF', 1.20, 0.30),
    5:  ('B',  '#FFB5B5', 1.92, 0.85),
    6:  ('C',  '#909090', 1.70, 0.70),
    7:  ('N',  '#3050F8', 1.55, 0.65),
    8:  ('O',  '#FF0D0D', 1.52, 0.60),
    9:  ('F',  '#90E050', 1.47, 0.50),
    15: ('P',  '#FF8000', 1.80, 1.00),
    16: ('S',  '#FFFF30', 1.80, 1.00),
    17: ('Cl', '#1FF01F', 1.75, 1.00),
}
SYM_TO_Z = {v[0]: k for k, v in ELEM_INFO.items()}


def find_hbond_contacts(template_coords, monomer_coords, max_dist=3.5):
    """Find H-bond candidates: heteroatom (N,O) ↔ H pairs across molecules.

    Returns list of (template_idx, monomer_idx, distance) tuples.
    H-bonds: donor H is within 3.5 Å of acceptor heteroatom.
    """
    n_tmpl = len(template_coords)
    contacts = []

    # Template heavy atom (acceptor) ↔ Monomer H (donor)
    for ti, (sym_t, *pos_t) in enumerate(template_coords):
        if sym_t not in ('N', 'O', 'S', 'F'):
            continue
        pt = np.array(pos_t)
        for mi, (sym_m, *pos_m) in enumerate(monomer_coords):
            if sym_m != 'H':
                continue
            pm = np.array(pos_m)
            d = np.linalg.norm(pt - pm)
            if d < max_dist:
                contacts.append((ti, n_tmpl + mi, d, 'tmpl_acceptor'))

    # Monomer heavy atom (acceptor) ↔ Template H (donor)
    for mi, (sym_m, *pos_m) in enumerate(monomer_coords):
        if sym_m not in ('N', 'O', 'S', 'F'):
            continue
        pm = np.array(pos_m)
        for ti, (sym_t, *pos_t) in enumerate(template_coords):
            if sym_t != 'H':
                continue
            pt = np.array(pos_t)
            d = np.linalg.norm(pm - pt)
            if d < max_dist:
                contacts.append((ti, n_tmpl + mi, d, 'tmpl_donor'))

    contacts.sort(key=lambda c: c[2])
    return contacts


def draw_binding(template_name, template_coords, monomer_name, monomer_coords,
                  output_path, dE_kcal=None):
    """Plot template-monomer binding interaction."""
    n_tmpl = len(template_coords)
    all_coords = list(template_coords) + list(monomer_coords)

    # Build atom traces by element
    atoms_by_z = {}
    for idx, (sym, x, y, z) in enumerate(all_coords):
        Z = SYM_TO_Z.get(sym)
        if Z is None:
            continue
        atoms_by_z.setdefault(Z, []).append((idx, x, y, z))

    fig = go.Figure()

    # Bonds within each molecule (1.8 Å cutoff)
    for label, coords, idx_offset in [
        (template_name, template_coords, 0),
        (monomer_name, monomer_coords, n_tmpl),
    ]:
        bx, by, bz = [], [], []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                pi = np.array(coords[i][1:])
                pj = np.array(coords[j][1:])
                if np.linalg.norm(pi - pj) < 1.8:
                    bx += [pi[0], pj[0], None]
                    by += [pi[1], pj[1], None]
                    bz += [pi[2], pj[2], None]
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(color='#444', width=5),
            name=f'{label} bonds', showlegend=False,
        ))

    # Atom markers by element (CPK colors)
    for Z, atoms in atoms_by_z.items():
        sym, color, vdw, ball = ELEM_INFO[Z]
        xs = [a[1] for a in atoms]
        ys = [a[2] for a in atoms]
        zs = [a[3] for a in atoms]
        # Slightly different size for template vs monomer? Same for now.
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers',
            marker=dict(size=ball * 18, color=color,
                        line=dict(width=1.5, color='#222')),
            name=sym, hovertext=[f'{sym}{a[0]}' for a in atoms],
            showlegend=True,
        ))

    # H-bond contacts (dashed line + distance label)
    contacts = find_hbond_contacts(template_coords, monomer_coords)
    annotations = []
    for ci, (ti, mi_global, d, kind) in enumerate(contacts[:3]):  # top 3
        p1 = np.array(all_coords[ti][1:])
        p2 = np.array(all_coords[mi_global][1:])
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color='#E91E63', width=6, dash='dash'),
            name=f'H-bond {ci+1}: {d:.2f} Å',
            showlegend=True,
        ))
        # Midpoint label
        mid = (p1 + p2) / 2
        annotations.append(dict(
            x=mid[0], y=mid[1], z=mid[2],
            text=f"<b>{d:.2f} Å</b>",
            showarrow=False, font=dict(size=14, color='#E91E63'),
        ))

    title = f"{template_name} + {monomer_name} Binding Interaction"
    if dE_kcal is not None:
        title += f"  (ΔE = {dE_kcal:+.2f} kcal/mol)"

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False,
                       showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(visible=False, showbackground=False,
                       showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(visible=False, showbackground=False,
                       showgrid=False, zeroline=False, showticklabels=False),
            aspectmode='data',
            bgcolor='white',
            annotations=annotations,
        ),
        paper_bgcolor='white',
        width=1100, height=900,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(x=0.85, y=0.95, bgcolor='rgba(255,255,255,0.85)'),
    )

    # Save HTML and PNG
    out = Path(output_path)
    fig.write_html(str(out.with_suffix('.html')), include_plotlyjs='cdn')
    try:
        fig.write_image(str(out.with_suffix('.png')), scale=2)
    except Exception:
        pass

    print(f"  Saved: {out.with_suffix('.html')}")
    print(f"  Saved: {out.with_suffix('.png')}")
    if contacts:
        print(f"  Top contacts: " +
              ", ".join(f"{d:.2f}Å" for _,_,d,_ in contacts[:3]))


def main():
    targets = [
        ("hexanal", "Hexanal", 6),  # n_tmpl_atoms_with_H = 6+12-1 = need to compute
        ("nonanal", "Nonanal", 9),
    ]

    # Determine template atom count from RDKit (with H)
    from rdkit import Chem
    from rdkit.Chem import AllChem

    for tmpl_dir, tmpl_name, n_carbon in targets:
        s1_path = Path(f"results/{tmpl_dir}/stage1/stage1_all.json")
        with open(s1_path) as f:
            s1 = json.load(f)
        entry = next((d for d in s1 if d["name"] == "MAA"), None)
        if not entry or not entry.get("complex_coords"):
            print(f"Skipping {tmpl_dir}: complex_coords missing")
            continue

        # Build template mol to know atom count
        tmpl_smiles = "C" * n_carbon + "C=O"  # CCCCCC=O for hexanal
        tmpl_mol = Chem.AddHs(Chem.MolFromSmiles(tmpl_smiles))
        n_tmpl_atoms = tmpl_mol.GetNumAtoms()

        coords = entry["complex_coords"]
        template_coords = coords[:n_tmpl_atoms]
        monomer_coords = coords[n_tmpl_atoms:]

        out_path = Path(f"results/{tmpl_dir}/stage2/binding_{tmpl_name}_MAA")
        print(f"\n=== {tmpl_name}-MAA binding interaction ===")
        print(f"  Template: {n_tmpl_atoms} atoms, Monomer: {len(monomer_coords)} atoms")

        draw_binding(
            tmpl_name, template_coords,
            "MAA", monomer_coords,
            str(out_path),
            dE_kcal=entry.get("dE_kcal"),
        )


if __name__ == "__main__":
    main()

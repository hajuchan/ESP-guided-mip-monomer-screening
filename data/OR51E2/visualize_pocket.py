"""
Visualize OR51E2 (8F76) binding pocket with propionate ligand.

Generates:
- Interactive HTML (plotly) — rotatable, zoomable
- Static PNG — publication quality

Shows:
- Propionate (ball-and-stick, colored by element)
- Pocket residues (sticks): R262 (anchor), F155/L158 (gate), others (light)
- H-bond/ionic interactions (dashed lines with distance labels)
- Pocket cavity outline (translucent surface)
"""

import json
from pathlib import Path
import numpy as np

import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PDB_FILE = Path(__file__).parent / "8F76.pdb"
LIGAND_RESNAME = "PPI"

ELEM_COLORS = {
    'H': '#FFFFFF', 'C': '#404040', 'N': '#2050FF', 'O': '#E00000',
    'S': '#E0D040', 'P': '#FF8000', 'F': '#70E060',
}
ELEM_SIZE = {'H': 6, 'C': 12, 'N': 14, 'O': 14, 'S': 16}

# Highlight residues
KEY_RESIDUES = {
    262: ('R262', '#FF6600', 'ANCHOR (Salt bridge)'),
    155: ('F155', '#9B59B6', 'GATE'),
    158: ('L158', '#9B59B6', 'GATE'),
    180: ('H180', '#3498DB', 'polar contact'),
    181: ('Q181', '#3498DB', 'polar contact'),
    104: ('H104', '#3498DB', 'polar contact'),
    194: ('N194', '#3498DB', 'polar contact'),
}


def parse_pdb(path):
    atoms = []
    for line in Path(path).read_text().splitlines():
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
                "element": line[76:78].strip() or line[12:14].strip()[0],
            })
        except (ValueError, IndexError):
            continue
    return atoms


def bonded_pairs(atoms, cutoff=1.8):
    """Find bonded atom pairs (within `cutoff`)."""
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    pairs = []
    n = len(atoms)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if 0.5 < d < cutoff:
                # Skip H-H bonds, allow C-H up to 1.3
                if atoms[i]['element'] == 'H' and atoms[j]['element'] == 'H':
                    continue
                if 'H' in (atoms[i]['element'], atoms[j]['element']):
                    if d < 1.3:
                        pairs.append((i, j))
                else:
                    pairs.append((i, j))
    return pairs


def make_plotly_figure(atoms, ligand_atoms, key_res_atoms,
                       hbond_pairs, out_html):
    fig = go.Figure()

    # ── Trace: Ligand bonds ──
    bonds_lig = bonded_pairs(ligand_atoms, cutoff=1.7)
    bx, by, bz = [], [], []
    for i, j in bonds_lig:
        a, b = ligand_atoms[i], ligand_atoms[j]
        bx += [a['x'], b['x'], None]
        by += [a['y'], b['y'], None]
        bz += [a['z'], b['z'], None]
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz, mode='lines',
        line=dict(color='#222', width=8),
        name='Propionate bonds', showlegend=False, hoverinfo='skip'))

    # ── Trace: Ligand atoms ──
    by_elem = {}
    for a in ligand_atoms:
        by_elem.setdefault(a['element'], []).append(a)
    for elem, alist in by_elem.items():
        fig.add_trace(go.Scatter3d(
            x=[a['x'] for a in alist], y=[a['y'] for a in alist],
            z=[a['z'] for a in alist],
            mode='markers',
            marker=dict(size=ELEM_SIZE.get(elem, 10),
                        color=ELEM_COLORS.get(elem, '#888'),
                        line=dict(width=2, color='black')),
            name=f'Propionate {elem}', showlegend=True,
        ))

    # ── Trace: Pocket residue atoms (by resnum) ──
    for resnum, (label, color, role) in KEY_RESIDUES.items():
        ratoms = [a for a in key_res_atoms if a['resnum'] == resnum]
        if not ratoms:
            continue
        # Bonds within residue
        bonds_r = bonded_pairs(ratoms, cutoff=1.8)
        rx, ry, rz = [], [], []
        for i, j in bonds_r:
            a, b = ratoms[i], ratoms[j]
            rx += [a['x'], b['x'], None]
            ry += [a['y'], b['y'], None]
            rz += [a['z'], b['z'], None]
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz, mode='lines',
            line=dict(color=color, width=5),
            name=f'{label} ({role})', showlegend=True,
        ))
        # Heavy atoms
        for elem, alist in {e: [a for a in ratoms if a['element']==e and e!='H']
                             for e in set(a['element'] for a in ratoms)}.items():
            if not alist: continue
            fig.add_trace(go.Scatter3d(
                x=[a['x'] for a in alist], y=[a['y'] for a in alist],
                z=[a['z'] for a in alist],
                mode='markers',
                marker=dict(size=ELEM_SIZE.get(elem, 10),
                            color=color, opacity=0.8,
                            line=dict(width=1, color='#222')),
                name=f'{label} {elem}', showlegend=False, hoverinfo='text',
                text=[f"{label} {a['atom_name']}" for a in alist],
            ))

    # ── Trace: H-bond/ionic interactions (dashed pink) ──
    annotations = []
    for (a1, a2, d, label) in hbond_pairs:
        fig.add_trace(go.Scatter3d(
            x=[a1['x'], a2['x']], y=[a1['y'], a2['y']], z=[a1['z'], a2['z']],
            mode='lines', line=dict(color='#E91E63', width=6, dash='dash'),
            name=label, showlegend=True, hoverinfo='text',
            text=[label]*2,
        ))
        mid = [(a1['x']+a2['x'])/2, (a1['y']+a2['y'])/2, (a1['z']+a2['z'])/2]
        annotations.append(dict(
            x=mid[0], y=mid[1], z=mid[2],
            text=f"<b>{d:.2f} Å</b>", showarrow=False,
            font=dict(size=12, color='#E91E63'),
        ))

    fig.update_layout(
        title=dict(text='<b>OR51E2 binding pocket (PDB 8F76)</b><br>'
                        '<sub>Propionate bound — R262 salt bridge anchor + F155/L158 size gate</sub>',
                   x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data', bgcolor='white',
            annotations=annotations,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        paper_bgcolor='white',
        width=1400, height=1000,
        margin=dict(l=10, r=10, t=80, b=10),
        legend=dict(x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#888', borderwidth=1, font=dict(size=12)),
    )
    fig.write_html(str(out_html), include_plotlyjs='cdn')
    print(f"HTML: {out_html}")


def make_matplotlib_figure(atoms, ligand_atoms, key_res_atoms,
                            hbond_pairs, out_png):
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.set_axis_off()

    # Ligand bonds
    bonds_lig = bonded_pairs(ligand_atoms, cutoff=1.7)
    for i, j in bonds_lig:
        a, b = ligand_atoms[i], ligand_atoms[j]
        ax.plot([a['x'], b['x']], [a['y'], b['y']], [a['z'], b['z']],
                color='#222', lw=3.5, zorder=8)
    # Ligand atoms
    for a in ligand_atoms:
        ax.scatter(a['x'], a['y'], a['z'],
                    s=ELEM_SIZE.get(a['element'], 100)*8,
                    c=ELEM_COLORS.get(a['element'], '#888'),
                    edgecolors='black', linewidths=1.5, zorder=10)

    # Pocket residues
    for resnum, (label, color, role) in KEY_RESIDUES.items():
        ratoms = [a for a in key_res_atoms if a['resnum'] == resnum]
        if not ratoms:
            continue
        bonds_r = bonded_pairs(ratoms, cutoff=1.8)
        for i, j in bonds_r:
            a, b = ratoms[i], ratoms[j]
            ax.plot([a['x'], b['x']], [a['y'], b['y']], [a['z'], b['z']],
                    color=color, lw=2, zorder=5)
        # Heavy atoms
        for a in ratoms:
            if a['element'] == 'H': continue
            ax.scatter(a['x'], a['y'], a['z'],
                        s=ELEM_SIZE.get(a['element'], 100)*4,
                        c=color, edgecolors='#222', linewidths=0.8, alpha=0.85,
                        zorder=6)
        # Label residue (use closest atom position)
        cx = np.mean([a['x'] for a in ratoms])
        cy = np.mean([a['y'] for a in ratoms])
        cz = np.mean([a['z'] for a in ratoms])
        ax.text(cx, cy, cz+2, label, fontsize=11, color=color,
                fontweight='bold', ha='center', zorder=20)

    # H-bond dashes
    for (a1, a2, d, _) in hbond_pairs:
        ax.plot([a1['x'], a2['x']], [a1['y'], a2['y']], [a1['z'], a2['z']],
                color='#E91E63', lw=2, linestyle='--', zorder=15)
        mid = [(a1['x']+a2['x'])/2, (a1['y']+a2['y'])/2, (a1['z']+a2['z'])/2]
        ax.text(mid[0], mid[1], mid[2], f'{d:.2f} Å',
                color='#E91E63', fontsize=10, fontweight='bold', ha='center',
                zorder=25)

    # Equal aspect, center on ligand
    lig_coords = np.array([[a['x'], a['y'], a['z']] for a in ligand_atoms])
    pocket_coords = np.array([[a['x'], a['y'], a['z']] for a in key_res_atoms])
    all_c = np.vstack([lig_coords, pocket_coords])
    pad = 1.5
    centers = (all_c.min(0) + all_c.max(0)) / 2
    mr = max(all_c.max(0) - all_c.min(0)) / 2 + pad
    for i, c in enumerate(centers):
        getattr(ax, f"set_{'xyz'[i]}lim")(c - mr, c + mr)
    try: ax.set_box_aspect((1, 1, 1))
    except: pass
    ax.view_init(elev=25, azim=-65)

    fig.suptitle('OR51E2 binding pocket (PDB 8F76) — bound to propionate',
                  fontsize=16, fontweight='bold', y=0.95)
    fig.text(0.5, 0.90,
              'R262 forms bidentate salt bridge; F155/L158 gate the small pocket',
              ha='center', fontsize=11, color='#444', style='italic')

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"PNG: {out_png}")


def main():
    atoms = parse_pdb(PDB_FILE)

    # Ligand atoms
    ligand_atoms = [a for a in atoms if a['resname'] == LIGAND_RESNAME]

    # Pocket residue atoms (key residues only)
    key_res_atoms = [a for a in atoms
                     if a['record'] == 'ATOM'
                     and a['chain'] == 'A'
                     and a['resnum'] in KEY_RESIDUES]

    # H-bond pairs (R262 NH ↔ PPI O within 3.5 Å)
    hbond_pairs = []
    r262_donors = [a for a in atoms
                    if a['resnum'] == 262 and a['resname'] == 'ARG'
                    and a['atom_name'] in ('NH1', 'NH2', 'NE')]
    ppi_o = [a for a in ligand_atoms if a['element'] == 'O']
    for r in r262_donors:
        for o in ppi_o:
            d = np.linalg.norm(np.array([r['x'], r['y'], r['z']]) -
                               np.array([o['x'], o['y'], o['z']]))
            if d <= 3.5:
                hbond_pairs.append(
                    (r, o, d, f"R262-{r['atom_name']}...PPI-{o['atom_name']}"))

    out_dir = Path(__file__).parent
    make_plotly_figure(atoms, ligand_atoms, key_res_atoms, hbond_pairs,
                       out_dir / "or51e2_pocket.html")
    make_matplotlib_figure(atoms, ligand_atoms, key_res_atoms, hbond_pairs,
                            out_dir / "or51e2_pocket.png")


if __name__ == "__main__":
    main()

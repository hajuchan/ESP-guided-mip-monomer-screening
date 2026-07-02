"""
Publication-quality ESP figures for MAA binding verification.

Goal (narrative): MAA was selected as functional monomer based on literature.
DFT calculations verify that MAA actually forms stable, well-defined complexes
with the aldehyde templates (hexanal, nonanal) at the expected binding site.

Method (consistent for all panels):
  ωB97X-D/def2-TZVP + PCM (acetonitrile)
  Iso-surface: electron density at 0.0015 a.u.
  Color: ESP in atomic units (BWR: red=δ-, white=0, blue=δ+)

Outputs (per system):
  - Isolated template MEP (Hexanal, Nonanal)
  - Template–MAA complex ESP
  Both as static PNG and interactive HTML.
"""

import json
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go


# ── Parameters ────────────────────────────────────────────────
DFT_FUNCTIONAL = 'wb97xd'    # ωB97X-D (consistent with H-bond complex DFT calculations)
DFT_BASIS = 'def2-tzvp'
PCM_EPS = 35.69              # acetonitrile
ISO_VAL = 0.0015             # electron density isosurface (a.u.)
ESP_VMAX = 0.05              # ESP color range (a.u.)

ELEM_INFO = {
    1:  ('H',  '#E8E8E8', 0.31),
    6:  ('C',  '#404040', 0.50),
    7:  ('N',  '#2050FF', 0.55),
    8:  ('O',  '#E00000', 0.55),
    9:  ('F',  '#70E060', 0.50),
    16: ('S',  '#E0D040', 0.65),
    17: ('Cl', '#20D020', 0.60),
}

BWR_COLORSCALE = [
    [0.00, '#B00000'], [0.25, '#FF4040'],
    [0.50, '#FFFFFF'],
    [0.75, '#4080FF'], [1.00, '#0000A0'],
]


# ── DFT cube generation ───────────────────────────────────────

def generate_cubes(atom_str, label, out_dir):
    """Compute electron density + ESP cubes at ωB97X-D/def2-TZVP + PCM."""
    import os
    os.environ.setdefault("PYSCF_TMPDIR", "/tmp")
    from pyscf import gto, dft, tools
    from pyscf.solvent import PCM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rho_path = out / f"esp_{label}_density.cube"
    esp_path = out / f"esp_{label}_esp.cube"

    if rho_path.exists() and esp_path.exists():
        print(f"  Cubes already exist for {label}, skipping DFT")
        return rho_path, esp_path

    print(f"  Running DFT ({DFT_FUNCTIONAL}/{DFT_BASIS}, PCM ε={PCM_EPS})...")
    mol = gto.M(atom=atom_str, basis=DFT_BASIS, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = DFT_FUNCTIONAL
    mf = PCM(mf)
    mf.with_solvent.eps = PCM_EPS
    try:
        mf = mf.to_gpu()
    except Exception:
        pass
    mf.kernel()
    dm = mf.make_rdm1()
    if hasattr(dm, 'get'):
        dm = dm.get()

    print(f"  Generating cubes (80x80x80)...")
    tools.cubegen.density(mol, str(rho_path), dm, nx=80, ny=80, nz=80)
    tools.cubegen.mep(mol, str(esp_path), dm, nx=80, ny=80, nz=80)
    return rho_path, esp_path


# ── Cube parsing ──────────────────────────────────────────────

def parse_cube(path):
    lines = Path(path).read_text().splitlines()
    n_atoms = abs(int(lines[2].split()[0]))
    origin = np.array([float(x) for x in lines[2].split()[1:4]])
    nx = int(lines[3].split()[0])
    ny = int(lines[4].split()[0])
    nz = int(lines[5].split()[0])
    axes = np.zeros((3, 3))
    for i in range(3):
        parts = lines[3 + i].split()
        axes[i] = [float(parts[j]) for j in range(1, 4)]
    atoms = []
    for i in range(n_atoms):
        parts = lines[6 + i].split()
        atoms.append({
            'Z': int(float(parts[0])),
            'pos': np.array([float(parts[2]), float(parts[3]), float(parts[4])])
        })
    vals = []
    for line in lines[6 + n_atoms:]:
        vals.extend(float(v) for v in line.split())
    data = np.array(vals).reshape(nx, ny, nz)
    return data, origin, axes, nx, ny, nz, atoms


def make_bwr_cmap():
    return LinearSegmentedColormap.from_list(
        'ESP_BWR',
        [(0.00, '#B00000'), (0.25, '#FF4040'),
         (0.50, '#FFFFFF'),
         (0.75, '#4080FF'), (1.00, '#0000A0')],
        N=256,
    )


# ── Surface extraction (shared) ───────────────────────────────

def extract_surface(rho_cube, esp_cube):
    """Returns (cart_verts, faces, esp_v, atoms_cart_A)."""
    from skimage.measure import marching_cubes
    from scipy.interpolate import RegularGridInterpolator

    BOHR2ANG = 0.529177
    rho, origin, axes, nx, ny, nz, atoms = parse_cube(rho_cube)
    esp, _, _, _, _, _, _ = parse_cube(esp_cube)

    iv = ISO_VAL
    if rho.max() < iv:
        iv = float(rho.max() * 0.05)

    verts, faces, _, _ = marching_cubes(rho, iv)
    cart = np.zeros_like(verts)
    for i in range(3):
        cart[:, i] = (origin[i] + verts[:, 0] * axes[0, i]
                      + verts[:, 1] * axes[1, i]
                      + verts[:, 2] * axes[2, i]) * BOHR2ANG

    interp = RegularGridInterpolator(
        (np.arange(nx), np.arange(ny), np.arange(nz)), esp,
        bounds_error=False, fill_value=0)
    esp_v = np.clip(interp(verts), -ESP_VMAX, ESP_VMAX)

    atom_pos = np.array([a['pos'] * BOHR2ANG for a in atoms])
    return cart, faces, esp_v, atom_pos, atoms


# ── matplotlib PNG renderer ───────────────────────────────────

def render_png(cart, faces, esp_v, atom_pos, atoms, out_path, title):
    cmap = make_bwr_cmap()
    norm = matplotlib.colors.Normalize(vmin=-ESP_VMAX, vmax=ESP_VMAX)
    face_esp = esp_v[faces].mean(axis=1)
    face_colors = cmap(norm(face_esp))
    face_colors[:, 3] = 0.55

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white'); ax.set_axis_off()

    poly = Poly3DCollection(cart[faces], facecolors=face_colors,
                             edgecolors='none', linewidths=0)
    poly.set_zsort('max')
    ax.add_collection3d(poly)

    # Heavy-heavy bonds
    for i in range(len(atoms)):
        if atoms[i]['Z'] == 1: continue
        for j in range(i+1, len(atoms)):
            if atoms[j]['Z'] == 1: continue
            d = np.linalg.norm(atom_pos[i] - atom_pos[j])
            if d < 1.75:
                ax.plot(*zip(atom_pos[i], atom_pos[j]),
                        color='#222', lw=2.5, zorder=10)
    # C-H bonds
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            if 1 not in (atoms[i]['Z'], atoms[j]['Z']): continue
            d = np.linalg.norm(atom_pos[i] - atom_pos[j])
            if d < 1.30:
                ax.plot(*zip(atom_pos[i], atom_pos[j]),
                        color='#666', lw=1.5, zorder=10)
    # Atoms
    for atom, pos in zip(atoms, atom_pos):
        Z = atom['Z']
        _, color, _ = ELEM_INFO.get(Z, ('X', '#FF00FF', 0.4))
        s = 250 if Z != 1 else 80
        ax.scatter(*pos, s=s, c=color, edgecolors='#111',
                   linewidths=0.8, zorder=20)

    # Equal aspect
    all_c = np.vstack([cart, atom_pos])
    pad = 1.5
    centers = (all_c.min(0) + all_c.max(0)) / 2
    mr = max(all_c.max(0) - all_c.min(0)) / 2 + pad
    for i, c in enumerate(centers):
        getattr(ax, f'set_{"xyz"[i]}lim')(c - mr, c + mr)
    try: ax.set_box_aspect((1, 1, 1))
    except: pass
    ax.view_init(elev=20, azim=-65)

    fig.text(0.5, 0.95, title, ha='center', fontsize=18,
             fontweight='bold', color='#222')
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    cbar = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('ESP (a.u.)', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  PNG: {out_path}")


# ── plotly HTML renderer ──────────────────────────────────────

def render_html(cart, faces, esp_v, atom_pos, atoms, out_path, title):
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=cart[:, 0], y=cart[:, 1], z=cart[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=esp_v,
        colorscale=BWR_COLORSCALE,
        cmin=-ESP_VMAX, cmax=ESP_VMAX,
        opacity=0.6,
        colorbar=dict(
            title=dict(text='ESP (a.u.)', font=dict(size=15)),
            len=0.7, thickness=18, x=0.96, tickfont=dict(size=13),
        ),
        name='ESP surface',
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3,
                      roughness=0.5, fresnel=0.1),
        lightposition=dict(x=100, y=100, z=200),
    ))

    # Bonds
    bond_xyz = [[], [], []]
    bond_xyz_h = [[], [], []]
    for i in range(len(atoms)):
        Zi = atoms[i]['Z']
        for j in range(i+1, len(atoms)):
            Zj = atoms[j]['Z']
            d = np.linalg.norm(atom_pos[i] - atom_pos[j])
            if Zi != 1 and Zj != 1 and d < 1.75:
                for k in range(3):
                    bond_xyz[k] += [atom_pos[i, k], atom_pos[j, k], None]
            elif 1 in (Zi, Zj) and d < 1.30:
                for k in range(3):
                    bond_xyz_h[k] += [atom_pos[i, k], atom_pos[j, k], None]

    fig.add_trace(go.Scatter3d(
        x=bond_xyz[0], y=bond_xyz[1], z=bond_xyz[2],
        mode='lines', line=dict(color='#222', width=8),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter3d(
        x=bond_xyz_h[0], y=bond_xyz_h[1], z=bond_xyz_h[2],
        mode='lines', line=dict(color='#666', width=4),
        showlegend=False, hoverinfo='skip',
    ))

    # Atom markers grouped by element
    atoms_by_z = {}
    for atom, pos in zip(atoms, atom_pos):
        atoms_by_z.setdefault(atom['Z'], []).append(pos)
    for Z, positions in atoms_by_z.items():
        sym, color, _ = ELEM_INFO.get(Z, ('X', '#FF00FF', 0.4))
        positions = np.array(positions)
        size = 14 if Z != 1 else 7
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode='markers',
            marker=dict(size=size, color=color,
                        line=dict(width=1.5, color='#111')),
            name=sym, opacity=1.0,
        ))

    n_traces = len(fig.data)
    opacity_steps = []
    for op in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        opacity_values = [op] + [1.0] * (n_traces - 1)
        opacity_steps.append(dict(
            args=["opacity", opacity_values],
            label=f"{int(op*100)}%", method="restyle",
        ))

    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', x=0.5, y=0.97,
                   font=dict(size=22, color='#222')),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data', bgcolor='white',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        paper_bgcolor='white', plot_bgcolor='white',
        width=1400, height=1100,
        margin=dict(l=0, r=0, t=60, b=80),
        legend=dict(x=0.02, y=0.92,
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='#888', borderwidth=1,
                    font=dict(size=13)),
        sliders=[dict(
            active=4,
            currentvalue=dict(prefix='Surface Opacity: ',
                              font=dict(size=14)),
            len=0.8, pad=dict(t=20, b=10), x=0.1, y=0,
            steps=opacity_steps,
        )],
    )

    fig.write_html(str(out_path), include_plotlyjs='cdn')
    print(f"  HTML: {out_path}")


# ── Main driver ───────────────────────────────────────────────

def build_complex_atom_str(template_smiles, monomer_smiles, complex_coords):
    """Rebuild PySCF atom string from Stage 1 complex coords."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    from pipeline.stage2_dft import smiles_to_mol3d, mol_to_pyscf_atom
    template_mol = smiles_to_mol3d(template_smiles)
    monomer_mol = smiles_to_mol3d(monomer_smiles)
    combo = Chem.CombineMols(template_mol, monomer_mol)
    combo = Chem.RWMol(combo)
    conf = combo.GetConformer()
    for i, (sym, x, y, z) in enumerate(complex_coords):
        if i < combo.GetNumAtoms():
            conf.SetAtomPosition(i, Point3D(x, y, z))
    return mol_to_pyscf_atom(combo.GetMol())


def build_single_atom_str(smiles):
    """Build atom string for an isolated molecule."""
    from pipeline.stage2_dft import smiles_to_mol3d, mol_to_pyscf_atom
    return mol_to_pyscf_atom(smiles_to_mol3d(smiles))


def main():
    monomer_smiles = "CC(=C)C(=O)O"  # MAA
    targets = [
        ("hexanal", "Hexanal", "CCCCCC=O"),
        ("nonanal", "Nonanal", "CCCCCCCCC=O"),
    ]

    for tmpl_dir, tmpl_name, tmpl_smiles in targets:
        out_dir = Path(f"results/{tmpl_dir}/stage2")

        # Template–MAA complex ESP only (ωB97X-D/def2-TZVP)
        s1 = json.loads(Path(f"results/{tmpl_dir}/stage1/stage1_all.json").read_text())
        entry = next((d for d in s1 if d["name"] == "MAA"), None)
        if not entry:
            print(f"  Skip: MAA entry not found for {tmpl_name}")
            continue

        print(f"\n=== {tmpl_name}–MAA complex ESP (ωB97X-D/def2-TZVP) ===")
        atom_str_complex = build_complex_atom_str(
            tmpl_smiles, monomer_smiles, entry["complex_coords"])
        rho, esp = generate_cubes(
            atom_str_complex, f"{tmpl_name}_MAA_complex", str(out_dir))
        cart, faces, esp_v, atom_pos, atoms = extract_surface(rho, esp)
        render_png(cart, faces, esp_v, atom_pos, atoms,
                   out_dir / f"esp_{tmpl_name}_MAA_complex.png",
                   f"{tmpl_name} + MAA Complex")
        render_html(cart, faces, esp_v, atom_pos, atoms,
                    out_dir / f"esp_{tmpl_name}_MAA_complex.html",
                    f"{tmpl_name} + MAA Complex")


if __name__ == "__main__":
    main()

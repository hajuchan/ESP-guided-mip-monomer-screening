"""
Publication-quality ESP figures using matplotlib 3D ray-trace style.

Approach:
1. Reuse existing density + ESP cube files
2. Extract iso-surface via marching cubes (smooth molecular envelope)
3. Color by ESP using BWR colorscale (Singh 2012 convention)
4. Render with matplotlib mplot3d using Poly3DCollection + Phong shading
5. Add CPK ball-and-stick inside
6. White background, no axes — clean publication figure
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Bondi vdW radii (Å) + CPK colors
ELEM_INFO = {
    1:  ('H',  '#E8E8E8', 0.31),  # H — light grey
    6:  ('C',  '#404040', 0.50),  # C — dark grey
    7:  ('N',  '#2050FF', 0.55),  # N — blue
    8:  ('O',  '#E00000', 0.55),  # O — red
    9:  ('F',  '#70E060', 0.50),
    16: ('S',  '#E0D040', 0.65),
    17: ('Cl', '#20D020', 0.60),
}


def parse_cube(path):
    """Parse Gaussian cube file. Returns (data, origin, axes, atoms)."""
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
    data_start = 6 + n_atoms
    vals = []
    for line in lines[data_start:]:
        vals.extend(float(v) for v in line.split())
    data = np.array(vals).reshape(nx, ny, nz)
    return data, origin, axes, nx, ny, nz, atoms


def make_bwr_cmap():
    """BWR colormap matching Singh 2012 (red=δ-, white=0, blue=δ+)."""
    return LinearSegmentedColormap.from_list(
        'ESP_BWR',
        [(0.00, '#B00000'),  # deep red (most negative)
         (0.25, '#FF4040'),
         (0.50, '#FFFFFF'),  # white at zero
         (0.75, '#4080FF'),
         (1.00, '#0000A0')], # deep blue (most positive)
        N=256,
    )


def make_rwb_cmap():
    """Alternative: red→white→blue with broader white range."""
    return LinearSegmentedColormap.from_list(
        'ESP_RWB',
        [(0.00, '#990000'), (0.20, '#FF3333'), (0.40, '#FFAAAA'),
         (0.50, '#FFFFFF'),
         (0.60, '#AACCFF'), (0.80, '#3366FF'), (1.00, '#000099')],
        N=256,
    )


def render_esp_publication(rho_cube, esp_cube, out_path, title,
                            iso_val=0.0015, esp_vmax=0.05, view=(20, -65)):
    """Render publication-quality ESP figure using matplotlib 3D."""
    from skimage.measure import marching_cubes
    from scipy.interpolate import RegularGridInterpolator

    BOHR2ANG = 0.529177

    rho, origin, axes, nx, ny, nz, atoms = parse_cube(rho_cube)
    esp, _, _, _, _, _, _ = parse_cube(esp_cube)

    # Adjust iso if too low
    if rho.max() < iso_val:
        iso_val = float(rho.max() * 0.05)
        print(f"  Adjusted iso_val to {iso_val:.4g}")

    # Marching cubes on density grid
    verts, faces, normals, _ = marching_cubes(rho, iso_val)

    # Convert to Cartesian (Å)
    cart = np.zeros_like(verts)
    for i in range(3):
        cart[:, i] = (origin[i] + verts[:, 0] * axes[0, i]
                      + verts[:, 1] * axes[1, i]
                      + verts[:, 2] * axes[2, i]) * BOHR2ANG

    # Interpolate ESP on vertices
    interp = RegularGridInterpolator(
        (np.arange(nx), np.arange(ny), np.arange(nz)), esp,
        bounds_error=False, fill_value=0)
    esp_v = interp(verts)

    # Average ESP per triangle for face coloring
    face_esp = esp_v[faces].mean(axis=1)
    face_esp_clipped = np.clip(face_esp, -esp_vmax, esp_vmax)

    cmap = make_bwr_cmap()
    norm = matplotlib.colors.Normalize(vmin=-esp_vmax, vmax=esp_vmax)
    face_colors = cmap(norm(face_esp_clipped))
    face_colors[:, 3] = 0.55  # alpha for transparency

    # Triangle coordinates: shape (n_faces, 3, 3)
    tri_coords = cart[faces]

    # Figure setup
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.set_axis_off()

    # Add isosurface
    poly = Poly3DCollection(tri_coords, facecolors=face_colors,
                             edgecolors='none', linewidths=0)
    poly.set_zsort('max')
    ax.add_collection3d(poly)

    # Convert atom positions to Å
    atom_pos = np.array([a['pos'] * BOHR2ANG for a in atoms])

    # Draw bonds (1.8 Å cutoff)
    for i in range(len(atoms)):
        Zi = atoms[i]['Z']
        if Zi == 1: continue  # skip H bonds first pass
        for j in range(i+1, len(atoms)):
            Zj = atoms[j]['Z']
            if Zj == 1: continue
            d = np.linalg.norm(atom_pos[i] - atom_pos[j])
            if d < 1.75:
                ax.plot(*zip(atom_pos[i], atom_pos[j]),
                        color='#222222', lw=2.5, zorder=10)
    # Draw C-H bonds (thinner)
    for i in range(len(atoms)):
        Zi = atoms[i]['Z']
        for j in range(i+1, len(atoms)):
            Zj = atoms[j]['Z']
            if 1 not in (Zi, Zj): continue
            d = np.linalg.norm(atom_pos[i] - atom_pos[j])
            if d < 1.30:
                ax.plot(*zip(atom_pos[i], atom_pos[j]),
                        color='#666666', lw=1.5, zorder=10)

    # Draw atoms (heavy atoms larger, H smaller)
    for atom, pos in zip(atoms, atom_pos):
        Z = atom['Z']
        info = ELEM_INFO.get(Z, ('X', '#FF00FF', 0.4))
        sym, color, r = info
        size = 250 if Z != 1 else 80
        ax.scatter(*pos, s=size, c=color, edgecolors='#111111',
                   linewidths=0.8, zorder=20)

    # Set equal aspect
    all_coords = np.vstack([cart, atom_pos])
    pad = 1.5
    xmin, xmax = all_coords[:, 0].min() - pad, all_coords[:, 0].max() + pad
    ymin, ymax = all_coords[:, 1].min() - pad, all_coords[:, 1].max() + pad
    zmin, zmax = all_coords[:, 2].min() - pad, all_coords[:, 2].max() + pad
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2
    cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    ax.set_xlim(cx - max_range, cx + max_range)
    ax.set_ylim(cy - max_range, cy + max_range)
    ax.set_zlim(cz - max_range, cz + max_range)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # View angle
    ax.view_init(elev=view[0], azim=view[1])

    # Title
    if title:
        fig.text(0.5, 0.95, title, ha='center', fontsize=18,
                 fontweight='bold', color='#222')

    # Colorbar
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.25, 0.025, 0.5])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('ESP (a.u.)', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 0.86, 0.93])
    fig.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    targets = [
        ("hexanal", "Hexanal"),
        ("nonanal", "Nonanal"),
    ]
    for tmpl_dir, tmpl_name in targets:
        out_dir = Path(f"results/{tmpl_dir}/stage2")
        # Use already-generated cubes
        rho = out_dir / f"vmd_density_{tmpl_name}_MAA_complex.cube"
        esp = out_dir / f"vmd_esp_{tmpl_name}_MAA_complex.cube"
        if not rho.exists() or not esp.exists():
            print(f"  Skipping {tmpl_name}: cubes not found")
            continue

        print(f"\n=== {tmpl_name} + MAA ESP (publication style) ===")
        out_png = out_dir / f"pub_esp_{tmpl_name}_MAA_complex.png"
        title = f"{tmpl_name} + MAA Complex ESP"
        render_esp_publication(rho, esp, out_png, title)


if __name__ == "__main__":
    main()

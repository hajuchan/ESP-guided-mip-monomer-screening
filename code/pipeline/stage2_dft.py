"""
Stage 2: DFT Binding Energy with BSSE Correction
=================================================
B3LYP-D3BJ/6-311+G* with ddCOSMO solvation and counterpoise BSSE correction.

Refs: Singh et al. 2012 (DFT MIP screening),
      Boys & Bernardi 1970 (counterpoise BSSE),
      Grimme et al. 2010 (DFT-D3),
      Lipparini et al. 2013 (ddCOSMO),
      Wu et al. 2024 (GPU4PySCF).
"""

import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import (
    TEMPLATE_SMILES,
    MONOMER_LIBRARY,
    SOLVENTS,
    N_WORKERS,
    N_GPU_WORKERS,
    USE_GPU,
    HARTREE_TO_KCAL,
    OUTPUT_DIR,
    OUTPUT_DIRS,
    USE_ESP_MAP,
    DFT_RELAX_HEAVY_THRESHOLD,
    DFT_RELAX_STEPS,
    DFT_OPT_BASIS,
    DFT_SP_BASIS,
    DFT_FUNCTIONAL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage2] %(message)s")
logger = logging.getLogger(__name__)


# ── Utility: SMILES → 3D mol ────────────────────────────────────────

def smiles_to_mol3d(smiles: str) -> Chem.Mol:
    """Generate optimized 3D structure from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)
    if len(cids) == 0:
        raise RuntimeError(f"Embedding failed for {smiles}")
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    energies = [(e, i) for i, (c, e) in enumerate(results) if c == 0]
    if not energies:
        energies = [(results[0][1], 0)]
    energies.sort()
    best_cid = energies[0][1]
    conf = mol.GetConformer(best_cid)
    new_mol = Chem.RWMol(mol)
    for cid in [c.GetId() for c in new_mol.GetConformers()]:
        if cid != best_cid:
            new_mol.RemoveConformer(cid)
    return new_mol.GetMol()


def mol_to_pyscf_atom(mol: Chem.Mol) -> str:
    """Convert RDKit mol to PySCF atom string: 'C x y z; H x y z; ...'"""
    conf = mol.GetConformer()
    atoms = []
    for i in range(mol.GetNumAtoms()):
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        pos = conf.GetAtomPosition(i)
        atoms.append(f"{sym}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}")
    return "; ".join(atoms)


def mol_to_xyz_arrays(mol: Chem.Mol):
    """Extract positions as numpy array from RDKit mol."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(n)])


def build_complex_mol(template_mol: Chem.Mol, monomer_mol: Chem.Mol,
                      gap: float = 3.0, direction: str = "+x") -> Chem.Mol:
    """Combine template and monomer with bounding-box based placement.

    Args:
        direction: '+x', '-x', '+y', '-y', '+z', '-z'
    """
    from rdkit.Geometry import Point3D

    t_pos = mol_to_xyz_arrays(template_mol)
    m_pos = mol_to_xyz_arrays(monomer_mol)
    t_centroid = t_pos.mean(axis=0)
    m_centroid = m_pos.mean(axis=0)

    dir_map = {
        "+x": (0, 1), "-x": (0, -1),
        "+y": (1, 1), "-y": (1, -1),
        "+z": (2, 1), "-z": (2, -1),
    }
    axis, sign = dir_map.get(direction, (0, 1))

    if sign > 0:
        t_edge = t_pos[:, axis].max()
        m_edge = m_pos[:, axis].min()
    else:
        t_edge = t_pos[:, axis].min()
        m_edge = m_pos[:, axis].max()

    shift = np.zeros(3)
    shift[axis] = (t_edge - m_edge) + sign * gap
    for ax in range(3):
        if ax != axis:
            shift[ax] = t_centroid[ax] - m_centroid[ax]

    combo = Chem.CombineMols(template_mol, monomer_mol)
    combo = Chem.RWMol(combo)
    conf = combo.GetConformer()
    n_template = template_mol.GetNumAtoms()
    n_total = combo.GetNumAtoms()

    for i in range(n_template, n_total):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(
            pos.x + shift[0], pos.y + shift[1], pos.z + shift[2]
        ))
    return combo.GetMol()


def get_best_direction(template_smiles: str, monomer_smiles: str) -> str:
    """Determine best placement via xTB binding site search (fallback for Stage 2)."""
    from .stage1_xtb import build_complex_binding_site
    result = build_complex_binding_site(
        template_smiles, monomer_smiles,
        n_orientations=50, top_n=10,
    )
    return result.get("best_direction", "+x")


# ── DFT Calculations ────────────────────────────────────────────────

def _dft_optimize(atom_str: str, basis: str, eps: float,
                  tmpdir: str) -> str:
    """GPU-accelerated DFT geometry optimization with PCM solvation.

    Uses ωB97XD/6-311+G* + PCM on GPU via gpu4pyscf + geomeTRIC.
    PCM (not ddCOSMO) is used because gpu4pyscf supports PCM analytical
    gradient on GPU, enabling full geometry optimization with solvation.

    Returns updated atom string with optimized coordinates.
    """
    os.environ["PYSCF_TMPDIR"] = tmpdir
    from pyscf import gto, dft, lib
    from pyscf.solvent import PCM
    try:
        from pyscf.geomopt.geometric_solver import optimize as geo_opt
    except ImportError:
        logger.warning("  geomeTRIC not available, using SP only")
        return atom_str

    lib.param.TMPDIR = tmpdir

    # 2-level optimization: small basis + RI-J for speed, then large basis for SP
    mol = gto.M(atom=atom_str, basis=DFT_OPT_BASIS, verbose=0)
    mf = dft.RKS(mol).density_fit()  # RI-J density fitting (~3x faster)
    mf.xc = DFT_FUNCTIONAL
    mf = PCM(mf)
    mf.with_solvent.eps = eps

    try:
        mf = mf.to_gpu()
    except Exception:
        logger.info("  GPU not available for optimization, using CPU")

    try:
        mol_eq = geo_opt(mf, maxsteps=50)
        atoms = []
        for i in range(mol_eq.natm):
            sym = mol_eq.atom_symbol(i)
            coord = mol_eq.atom_coord(i, unit="Ang")
            atoms.append(f"{sym}  {coord[0]:.6f}  {coord[1]:.6f}  {coord[2]:.6f}")
        optimized_str = "; ".join(atoms)
        logger.info(f"  DFT optimization converged ({mol_eq.natm} atoms)")
        return optimized_str
    except Exception as e:
        logger.warning(f"  DFT optimization failed: {e}, using original geometry")
        return atom_str


def _try_gpu(mf):
    """Attempt GPU acceleration via gpu4pyscf; fall back to CPU."""
    if not USE_GPU:
        return mf
    try:
        mf = mf.to_gpu()
        logger.debug("Using GPU-accelerated DFT")
    except Exception:
        logger.debug("GPU unavailable, using CPU")
    return mf


def _apply_d3bj(mf, mol_pyscf):
    """Add D3BJ dispersion correction and return correction energy."""
    try:
        from pyscf import dftd3 as pyscf_dftd3
        mf_d3 = pyscf_dftd3.dftd3(mf)
        return mf_d3
    except ImportError:
        try:
            import dftd3.pyscf as dftd3_pyscf
            mf_d3 = dftd3_pyscf.energy(mf)
            return mf_d3
        except Exception:
            logger.warning("D3BJ correction unavailable; proceeding without dispersion.")
            return mf


def dft_energy(atom_str: str, basis: str, eps: float,
               tmpdir: str, use_gpu: bool = True) -> float:
    """Run B3LYP-D3BJ/basis with ddCOSMO solvation. Returns energy in Hartree."""
    os.environ["PYSCF_TMPDIR"] = tmpdir

    from pyscf import gto, dft, solvent, lib
    lib.param.TMPDIR = tmpdir

    mol = gto.M(atom=atom_str, basis=basis, verbose=0)
    mf = dft.RKS(mol).density_fit()  # RI-J density fitting
    mf.xc = DFT_FUNCTIONAL

    from pyscf.solvent import PCM
    mf = PCM(mf)
    mf.with_solvent.eps = eps

    # GPU acceleration
    if use_gpu:
        mf = _try_gpu(mf)

    mf.kernel()
    return mf.e_tot


def dft_energy_ghost(real_atom_str: str, ghost_atom_str: str,
                     basis: str, eps: float, tmpdir: str,
                     use_gpu: bool = True) -> float:
    """DFT energy with ghost atoms for BSSE counterpoise correction.

    Boys & Bernardi, Mol. Phys. 1970.
    real_atom_str: atoms of the fragment being computed (at complex geometry)
    ghost_atom_str: atoms of the OTHER fragment (basis only, at complex geometry)
    """
    os.environ["PYSCF_TMPDIR"] = tmpdir

    from pyscf import gto, dft, solvent, lib
    lib.param.TMPDIR = tmpdir

    # Combine real atoms + ghost atoms (ghost provides basis functions only)
    combined_atom = real_atom_str + "; " + _make_ghost_string(ghost_atom_str)
    mol = gto.M(atom=combined_atom, basis=basis, verbose=0)

    mf = dft.RKS(mol).density_fit()  # RI-J
    mf.xc = DFT_FUNCTIONAL
    # No PCM for ghost atoms — ghost distorts PCM cavity
    if use_gpu:
        mf = _try_gpu(mf)

    mf.kernel()
    return mf.e_tot


def _make_ghost_string(atom_str: str) -> str:
    """Convert atom string to ghost atom string (prefix elements with 'ghost-')."""
    parts = atom_str.split(";")
    ghost_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        tokens[0] = f"ghost-{tokens[0]}"
        ghost_parts.append("  ".join(tokens))
    return "; ".join(ghost_parts)


# ── Per-monomer DFT calculation ─────────────────────────────────────

def compute_dft_binding(monomer_name: str, monomer_smiles: str,
                        template_smiles: str, solvent_name: str,
                        eps: float, direction: str = "+x",
                        esp_output_dir: str = None,
                        prebuilt_complex_mol: "Chem.Mol" = None) -> dict:
    """Compute DFT binding energy with BSSE for one (monomer, solvent) pair.

    If prebuilt_complex_mol is provided, use it instead of building from direction.
    """
    basis = DFT_SP_BASIS

    with tempfile.TemporaryDirectory(prefix=f"dft_{monomer_name}_{solvent_name}_") as tmpdir:
        try:
            template_mol = smiles_to_mol3d(template_smiles)
            monomer_mol = smiles_to_mol3d(monomer_smiles)
            if prebuilt_complex_mol is not None:
                complex_mol = prebuilt_complex_mol
            else:
                complex_mol = build_complex_mol(template_mol, monomer_mol,
                                                direction=direction)

            # DFT binding energy: ωB97XD/6-311+G* + PCM, GPU-accelerated
            # 1) GPU geometry optimization on complex
            # 2) BSSE counterpoise correction (valid: DFT structure + DFT BSSE)
            template_atom = mol_to_pyscf_atom(template_mol)
            monomer_atom = mol_to_pyscf_atom(monomer_mol)
            complex_atom = mol_to_pyscf_atom(complex_mol)

            logger.info(f"  GPU DFT optimization (wB97M-V+PCM)...")
            complex_atom = _dft_optimize(complex_atom, basis, eps, tmpdir)

            # Extract fragment coordinates from DFT-optimized complex for BSSE
            n_template = template_mol.GetNumAtoms()
            opt_parts = [p.strip() for p in complex_atom.split(";") if p.strip()]
            template_cp_str = "; ".join(opt_parts[:n_template])
            monomer_cp_str = "; ".join(opt_parts[n_template:])

            # Raw binding energy (no BSSE)
            e_complex = dft_energy(complex_atom, basis, eps, tmpdir)
            e_template = dft_energy(template_atom, basis, eps, tmpdir)
            e_monomer = dft_energy(monomer_atom, basis, eps, tmpdir)
            raw_de = (e_complex - e_template - e_monomer) * HARTREE_TO_KCAL

            # BSSE counterpoise (DFT structure → DFT BSSE = consistent)
            logger.info(f"  BSSE counterpoise correction...")
            e_template_cp = dft_energy_ghost(
                template_cp_str, monomer_cp_str, basis, eps, tmpdir
            )
            e_monomer_cp = dft_energy_ghost(
                monomer_cp_str, template_cp_str, basis, eps, tmpdir
            )
            bsse_de = (e_complex - e_template_cp - e_monomer_cp) * HARTREE_TO_KCAL

            logger.info(f"  raw_dE={raw_de:+.3f}, bsse_dE={bsse_de:+.3f} kcal/mol")

            return {
                "monomer": monomer_name,
                "solvent": solvent_name,
                "raw_dE_kcal": round(raw_de, 3),
                "bsse_dE_kcal": round(bsse_de, 3),
                "bsse_correction_kcal": round(raw_de - bsse_de, 3),
                "success": True,
            }
        except Exception as e:
            logger.warning(f"DFT failed for {monomer_name}/{solvent_name}: {e}")
            return {
                "monomer": monomer_name,
                "solvent": solvent_name,
                "success": False,
                "error": str(e),
            }


def run_stage2(template_smiles: str = None,
               monomer_names: list[str] = None,
               monomer_library: dict = None,
               solvents: dict = None,
               output_dir: str = OUTPUT_DIRS["stage2"]) -> dict:
    """Run DFT calculations for all (monomer, solvent) pairs.

    Parameters
    ----------
    monomer_names : list of monomer names from stage1 screening
    """
    template_smiles = template_smiles or TEMPLATE_SMILES
    monomer_library = monomer_library or MONOMER_LIBRARY
    solvents = solvents or SOLVENTS

    if monomer_names is None:
        # Try loading stage1 results
        stage1_path = Path(output_dir).parent / "stage1" / "stage1_top.json"
        if stage1_path.exists():
            with open(stage1_path) as f:
                stage1_data = json.load(f)
            monomer_names = [r["name"] for r in stage1_data]
        else:
            monomer_names = list(monomer_library.keys())

    monomers_to_run = {n: monomer_library[n] for n in monomer_names
                       if n in monomer_library}

    # Load best_direction from Stage 1 results
    direction_map = {}
    stage1_path = Path(output_dir).parent / "stage1" / "stage1_top.json"
    if stage1_path.exists():
        with open(stage1_path) as f:
            for entry in json.load(f):
                if "best_direction" in entry:
                    direction_map[entry["name"]] = entry["best_direction"]

    n_tasks = len(monomers_to_run) * len(solvents)
    max_workers = N_GPU_WORKERS if USE_GPU else N_WORKERS
    logger.info(f"Stage 2: {len(monomers_to_run)} monomers × {len(solvents)} solvents = "
                f"{n_tasks} DFT jobs (workers={max_workers}, GPU={USE_GPU})")

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for m_name, m_smiles in monomers_to_run.items():
            # Get best direction: from Stage1, or run xTB surface scan
            if m_name in direction_map:
                best_dir = direction_map[m_name]
            else:
                logger.info(f"  {m_name}: no Stage1 direction, running xTB scan...")
                best_dir = get_best_direction(template_smiles, m_smiles)

            for s_name, eps in solvents.items():
                fut = executor.submit(
                    compute_dft_binding,
                    m_name, m_smiles, template_smiles, s_name, eps,
                    best_dir, output_dir,
                )
                futures[fut] = (m_name, s_name)

        for future in as_completed(futures):
            m_name, s_name = futures[future]
            res = future.result()
            if res["success"]:
                results.setdefault(m_name, {})[s_name] = {
                    "raw_dE": res["raw_dE_kcal"],
                    "bsse_dE": res["bsse_dE_kcal"],
                    "bsse_correction": res["bsse_correction_kcal"],
                }
                logger.info(
                    f"  {m_name:>10s}/{s_name:<15s}: "
                    f"raw={res['raw_dE_kcal']:+.3f}, "
                    f"bsse={res['bsse_dE_kcal']:+.3f} kcal/mol"
                )
            else:
                logger.warning(f"  {m_name}/{s_name}: FAILED")

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "stage2_dft.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Stage 2 complete. Results: {out_path / 'stage2_dft.json'}")
    return results


# ── Feature 4: ESP Map Visualization ─────────────────────────────────

def generate_esp_map(atom_str: str, basis: str, eps: float,
                     label: str, output_dir: str):
    """Generate ESP (Molecular Electrostatic Potential) map.

    Saves a cube file and a 2D contour PNG.
    Ref: Singh 2012 — ESP maps for template-monomer interaction analysis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    out_path = Path(output_dir)

    try:
        from pyscf import gto, dft, solvent, lib, tools

        mol = gto.M(atom=atom_str, basis=basis, verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        mf = solvent.ddCOSMO(mf)
        mf.with_solvent.eps = eps
        mf.kernel()

        # Generate cube file for MEP
        cube_path = str(out_path / f"stage2_esp_{label}.cube")
        dm = mf.make_rdm1()
        tools.cubegen.mep(mol, cube_path, dm, nx=60, ny=60, nz=60)

        # Read cube and create 2D contour plot (xy plane at z-midpoint)
        with open(cube_path) as f:
            lines = f.readlines()

        n_atoms = abs(int(lines[2].split()[0]))
        origin = [float(x) for x in lines[2].split()[1:4]]
        nx = int(lines[3].split()[0])
        ny = int(lines[4].split()[0])
        nz = int(lines[5].split()[0])
        dx = float(lines[3].split()[1])
        dy = float(lines[4].split()[2])
        dz = float(lines[5].split()[3])

        # Read volumetric data
        data_start = 6 + n_atoms
        values = []
        for line in lines[data_start:]:
            values.extend([float(v) for v in line.split()])
        data = np.array(values).reshape(nx, ny, nz)

        # Take z-midpoint slice
        z_mid = nz // 2
        slice_xy = data[:, :, z_mid]

        # Read atom positions for overlay
        atom_xs, atom_ys = [], []
        for i in range(n_atoms):
            parts = lines[6 + i].split()
            atom_xs.append(float(parts[2]))
            atom_ys.append(float(parts[3]))

        # Convert to grid coordinates
        bohr_to_ang = 0.529177
        atom_xs_grid = [(x - origin[0]) / dx for x in atom_xs]
        atom_ys_grid = [(y - origin[1]) / dy for y in atom_ys]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 7))
        vmax = max(abs(slice_xy.min()), abs(slice_xy.max()))
        vmax = min(vmax, 0.1)  # Clamp for visibility
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.contourf(slice_xy.T, levels=30, cmap="RdBu_r", norm=norm)
        ax.scatter(atom_xs_grid, atom_ys_grid, c="black", s=30, zorder=5)
        ax.set_title(f"ESP Map: {label}")
        ax.set_xlabel("x (grid)")
        ax.set_ylabel("y (grid)")
        plt.colorbar(im, ax=ax, label="ESP (a.u.)")
        fig.tight_layout()
        png_path = out_path / f"stage2_esp_{label}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info(f"  ESP map saved: {png_path}")

    except Exception as e:
        logger.warning(f"  ESP map generation failed for {label}: {e}")


def generate_esp_maps_for_binding(monomer_name: str, template_smiles: str,
                                  monomer_smiles: str, solvent_name: str,
                                  eps: float, output_dir: str,
                                  direction: str = "+x"):
    """Generate ESP maps for template, monomer, and complex."""
    if not USE_ESP_MAP:
        return

    template_mol = smiles_to_mol3d(template_smiles)
    monomer_mol = smiles_to_mol3d(monomer_smiles)
    complex_mol = build_complex_mol(template_mol, monomer_mol, direction=direction)

    basis = DFT_SP_BASIS
    tag = f"{monomer_name}_{solvent_name}"

    generate_esp_map(mol_to_pyscf_atom(template_mol), basis, eps,
                     f"{tag}_template", output_dir)
    generate_esp_map(mol_to_pyscf_atom(monomer_mol), basis, eps,
                     f"{tag}_monomer", output_dir)
    generate_esp_map(mol_to_pyscf_atom(complex_mol), basis, eps,
                     f"{tag}_complex", output_dir)


if __name__ == "__main__":
    run_stage2()

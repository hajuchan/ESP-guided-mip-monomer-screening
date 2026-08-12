"""
Stage 1: GFN2-xTB Fast Screening
=================================
QuantumDock-style exhaustive docking with GFN2-xTB single-point screening.

Method (faithful to Mukasa et al., Adv. Mater. 2023):
  1. Generate orientations across the entire molecular surface
  2. Fast GFN2-xTB single-point screening (replaces PM3)
  3. Full GFN2-xTB optimization on top candidates
  4. Binding sites identified post-hoc from results

Ref: Bannwarth et al., J. Chem. Theory Comput. 2019, 15, 1652-1671.
     Mukasa et al., Adv. Mater. 2023, 35, 2212161.
"""

import json
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import (
    TEMPLATE_SMILES,
    MONOMER_LIBRARY,
    N_WORKERS,
    STAGE1_TOP_N,
    HARTREE_TO_KCAL,
    OUTPUT_DIRS,
    COMPLEX_SEARCH_MODE,
    N_DOCK_ORIENTATIONS,
    N_TOP_FOR_OPTIMIZATION,
    DOCK_SURFACE_OFFSET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage1] %(message)s")
logger = logging.getLogger(__name__)

ANG_TO_BOHR = 1.8897259886


# ═══════════════════════════════════════════════════════════════════
#  Utility functions (shared by all modes)
# ═══════════════════════════════════════════════════════════════════

def smiles_to_mol3d(smiles: str, n_confs: int = 10) -> Chem.Mol:
    """Generate optimized 3D conformer from SMILES using ETKDGv3."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(cids) == 0:
        raise RuntimeError(f"Embedding failed for {smiles}")
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    energies = [(e, i) for i, (c, e) in enumerate(results) if c == 0]
    if not energies:
        energies = [(results[0][1], 0)]
    energies.sort()
    best_cid = energies[0][1]
    new_mol = Chem.RWMol(mol)
    for cid in [c.GetId() for c in new_mol.GetConformers()]:
        if cid != best_cid:
            new_mol.RemoveConformer(cid)
    return new_mol.GetMol()


def smiles_to_mol3d_ensemble(smiles: str, n_confs: int = 10,
                              n_keep: int = 3) -> list:
    """Generate multiple diverse 3D conformers for ensemble docking.

    Returns list of RDKit Mol objects (up to n_keep), sorted by MMFF energy.
    Conformers are filtered by RMSD diversity (> 0.5Å).
    Ref: Bio project Phase 1 ensemble docking approach.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(cids) == 0:
        return [smiles_to_mol3d(smiles)]

    results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
    energies = [(e, i) for i, (c, e) in enumerate(results) if c == 0]
    if not energies:
        return [smiles_to_mol3d(smiles)]
    energies.sort()

    # Select diverse conformers by RMSD filtering
    from rdkit.Chem import AllChem as _AC
    kept = []
    for e, cid in energies:
        if len(kept) >= n_keep:
            break
        is_diverse = True
        for _, kept_cid in kept:
            rmsd = _AC.GetConformerRMS(mol, cid, kept_cid)
            if rmsd < 0.5:  # too similar
                is_diverse = False
                break
        if is_diverse:
            kept.append((e, cid))

    # Return individual Mol objects per conformer
    conformers = []
    for e, cid in kept:
        new_mol = Chem.RWMol(mol)
        for c in [c.GetId() for c in new_mol.GetConformers()]:
            if c != cid:
                new_mol.RemoveConformer(c)
        conformers.append(new_mol.GetMol())

    return conformers if conformers else [smiles_to_mol3d(smiles)]


def mol_to_arrays(mol: Chem.Mol):
    """Extract atomic numbers and coordinates (Angstrom) from RDKit Mol."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    numbers = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)])
    positions = np.array(conf.GetPositions())
    return numbers, positions


def xtb_single_point(numbers: np.ndarray, positions: np.ndarray) -> float:
    """GFN2-xTB single point energy (Hartree)."""
    from tblite.interface import Calculator
    calc = Calculator("GFN2-xTB", numbers, positions * ANG_TO_BOHR)
    return calc.singlepoint().get("energy")


def xtb_optimize(numbers: np.ndarray, positions: np.ndarray,
                 tmpdir: str = None, max_iter: int = 200,
                 tol: float = 1e-4) -> tuple:
    """Optimize geometry with GFN2-xTB.

    Uses xtb binary's native ANCOPT optimizer (superior for non-covalent
    complexes on flat PES). Falls back to scipy L-BFGS-B if xtb not available.

    Returns (energy_hartree, optimized_positions_angstrom).
    """
    # Try native xtb optimizer first (ANCOPT — better for weak complexes)
    try:
        return _xtb_optimize_native(numbers, positions, max_iter)
    except Exception:
        pass

    # Fallback: scipy L-BFGS-B with tblite
    from tblite.interface import Calculator
    from scipy.optimize import minimize

    n_atoms = len(numbers)

    def energy_and_grad(flat_pos_bohr):
        pos_bohr = flat_pos_bohr.reshape(n_atoms, 3)
        calc = Calculator("GFN2-xTB", numbers, pos_bohr)
        res = calc.singlepoint()
        return res.get("energy"), res.get("gradient").flatten()

    x0 = (positions * ANG_TO_BOHR).flatten()
    result = minimize(energy_and_grad, x0, method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iter, "ftol": tol})
    opt_pos = result.x.reshape(n_atoms, 3) / ANG_TO_BOHR
    return result.fun, opt_pos


def _xtb_optimize_native(numbers, positions, max_iter=200):
    """Run xtb --opt using the native binary (ANCOPT algorithm)."""
    import subprocess, tempfile, os, shutil

    xtb_bin = shutil.which("xtb")
    if not xtb_bin:
        raise FileNotFoundError("xtb binary not found")

    with tempfile.TemporaryDirectory(prefix="xtb_opt_") as td:
        # Write xyz
        n = len(numbers)
        _ELEM = {1:'H',6:'C',7:'N',8:'O',9:'F',16:'S',17:'Cl',5:'B',35:'Br',15:'P'}
        xyz_path = os.path.join(td, "input.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{n}\n\n")
            for z, pos in zip(numbers, positions):
                sym = _ELEM.get(int(z), f"X{z}")
                f.write(f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")

        result = subprocess.run(
            [xtb_bin, "input.xyz", "--opt", "normal", "--iterations", str(max_iter),
             "--gfn", "2"],
            capture_output=True, text=True, timeout=300, cwd=td
        )

        # Parse optimized geometry from xtbopt.xyz
        opt_xyz = os.path.join(td, "xtbopt.xyz")
        if not os.path.exists(opt_xyz):
            raise RuntimeError("xtb optimization failed: no xtbopt.xyz")

        with open(opt_xyz) as f:
            lines = f.readlines()
        opt_pos = np.zeros((n, 3))
        for i in range(n):
            parts = lines[i + 2].split()
            opt_pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

        # Parse energy from output
        energy = None
        for line in result.stdout.split("\n"):
            if "TOTAL ENERGY" in line:
                energy = float(line.split()[3])
        if energy is None:
            raise RuntimeError("Could not parse energy from xtb output")

        return energy, opt_pos


# ═══════════════════════════════════════════════════════════════════
#  QuantumDock-style molecular surface docking
#  "a monomer is docked to a potential binding site on a target
#   molecule until all binding sites have been probed.
#   All noncovalent interactions (hydrogen bond, electrostatic,
#   and van der Waals) are considered." — Mukasa 2023 p.3
# ═══════════════════════════════════════════════════════════════════

def _vdw_radius(atomic_num: int) -> float:
    """Approximate van der Waals radius (Angstrom)."""
    _radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
              15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}
    return _radii.get(atomic_num, 1.70)


def generate_surface_points(mol: Chem.Mol, density: float = 1.0) -> np.ndarray:
    """Generate points on the van der Waals surface of a molecule.

    Uses a simple Fibonacci sphere sampling around each atom,
    then filters to keep only surface-exposed points.
    Returns array of shape (N, 3) in Angstrom.
    """
    nums, pos = mol_to_arrays(mol)
    n_atoms = len(nums)

    # Generate candidate points on each atom's vdW sphere
    all_pts = []
    n_per_atom = max(10, int(20 * density))
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n_atoms):
        r = _vdw_radius(nums[i])
        center = pos[i]
        for j in range(n_per_atom):
            theta = np.arccos(1 - 2 * (j + 0.5) / n_per_atom)
            phi = 2 * np.pi * j / golden_ratio
            pt = center + r * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            all_pts.append(pt)

    all_pts = np.array(all_pts)

    # Filter: keep only surface-exposed points (not buried inside molecule)
    from scipy.spatial.distance import cdist
    dists_to_atoms = cdist(all_pts, pos)  # (n_pts, n_atoms)
    radii = np.array([_vdw_radius(z) for z in nums])

    surface_pts = []
    for k, pt in enumerate(all_pts):
        # A point is surface-exposed if it's not inside any OTHER atom's vdW sphere
        is_buried = False
        for j in range(n_atoms):
            if dists_to_atoms[k, j] < radii[j] * 0.8:
                # Check if this is NOT the atom that generated this point
                parent_atom = k // n_per_atom
                if j != parent_atom:
                    is_buried = True
                    break
        if not is_buried:
            surface_pts.append(pt)

    if len(surface_pts) < 10:
        return all_pts  # Fallback: use all points
    return np.array(surface_pts)


def _compute_dft_charges(mol_3d: "Chem.Mol") -> np.ndarray:
    """Compute B3LYP/6-311+G* Mulliken partial charges via GPU-accelerated DFT.

    Much more accurate than xTB Mulliken, especially for molecules with
    B, Cl, or other heteroatoms where xTB charge distribution is unreliable.
    Takes ~7s per molecule (GPU SCF + CPU Mulliken analysis).
    """
    import os, tempfile
    from pyscf import gto, dft, lib

    conf = mol_3d.GetConformer()
    atoms = []
    for i in range(mol_3d.GetNumAtoms()):
        sym = mol_3d.GetAtomWithIdx(i).GetSymbol()
        pos = conf.GetAtomPosition(i)
        atoms.append(f"{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    atom_str = "; ".join(atoms)

    with tempfile.TemporaryDirectory(prefix="dft_chg_") as tmpdir:
        os.environ["PYSCF_TMPDIR"] = tmpdir
        lib.param.TMPDIR = tmpdir

        mol = gto.M(atom=atom_str, basis="6-311+g*", verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"

        try:
            mf_gpu = mf.to_gpu()
            mf_gpu.kernel()
            dm = mf_gpu.make_rdm1().get()  # cupy → numpy
            charges = mf.mulliken_pop(verbose=0, dm=dm)[1]
        except Exception:
            mf.kernel()
            charges = mf.mulliken_pop(verbose=0)[1]

    return charges.astype(float)


def _esp_guided_rotation(t_pos, t_charges, m_centered, m_charges,
                         target_pos) -> np.ndarray:
    """Orient monomer so its electrostatic-complementary atoms face template.

    Principle (Singh 2012): negative monomer atoms should face positive
    template regions and vice versa. This creates favorable electrostatic
    interactions (H-bond, ionic, dipole-dipole).

    Algorithm:
    1. Find the most positive and most negative atoms on template
       (weighted by distance to target_pos — prioritize nearby atoms)
    2. Compute template local dipole direction at target_pos
    3. Find monomer's dipole direction from its charges
    4. Rotate monomer so its dipole is anti-parallel to template's local dipole
    """
    from scipy.spatial.transform import Rotation

    # Template local electric field direction at target_pos
    # E = sum_i (q_i * (r_target - r_i) / |r_target - r_i|^3)
    dr = target_pos - t_pos  # (n_atoms, 3)
    dist = np.linalg.norm(dr, axis=1, keepdims=True)
    dist = np.maximum(dist, 0.5)  # Avoid division by zero
    field = np.sum(t_charges[:, None] * dr / dist**3, axis=0)
    field_norm = np.linalg.norm(field)

    if field_norm < 1e-10:
        return Rotation.random(1, random_state=42)

    field_dir = field / field_norm

    # Monomer dipole: sum(q_i * r_i) from centered coords
    m_dipole = np.sum(m_charges[:, None] * m_centered, axis=0)
    m_dipole_norm = np.linalg.norm(m_dipole)

    if m_dipole_norm < 1e-10:
        return Rotation.random(1, random_state=42)

    m_dipole_dir = m_dipole / m_dipole_norm

    # Rotate monomer dipole to be ANTI-PARALLEL to template field
    # (negative charges face positive field, and vice versa)
    target_dir = -field_dir
    rot = _rotation_between_vectors(m_dipole_dir, target_dir)
    return rot


def _rotation_between_vectors(v1: np.ndarray, v2: np.ndarray):
    """Compute rotation matrix that aligns v1 to v2."""
    from scipy.spatial.transform import Rotation

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.linalg.norm(cross) < 1e-10:
        if dot > 0:
            return Rotation.identity()
        else:
            # 180 degree rotation around any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(v1, perp)
            axis = axis / np.linalg.norm(axis)
            return Rotation.from_rotvec(np.pi * axis)

    skew = np.array([[0, -cross[2], cross[1]],
                     [cross[2], 0, -cross[0]],
                     [-cross[1], cross[0], 0]])
    R = np.eye(3) + skew + skew @ skew / (1 + dot)
    return Rotation.from_matrix(R)


def generate_docked_orientations(template_mol: Chem.Mol, monomer_mol: Chem.Mol,
                                  n_orientations: int = 100,
                                  surface_offset: float = 2.5,
                                  random_seed: int = 42) -> list:
    """100% ESP-guided orientation generation.

    All orientations use electrostatic complementarity (Singh 2012 principle):
    monomer dipole is aligned anti-parallel to template's local electric field.

    Surface points are weighted by |partial charge| so that stronger
    electrostatic interaction sites are sampled more densely.

    Each orientation gets a small random perturbation (±20°) for diversity,
    with multiple offsets (1.5, 2.5, 3.5 A) to cover different distances.
    """
    from scipy.spatial.transform import Rotation
    from scipy.spatial.distance import cdist

    t_num, t_pos = mol_to_arrays(template_mol)
    m_num, m_pos = mol_to_arrays(monomer_mol)
    t_centroid = t_pos.mean(axis=0)
    m_centroid = m_pos.mean(axis=0)
    m_centered = m_pos - m_centroid

    # DFT-level partial charges for ESP-guided placement
    # B3LYP/6-311+G* Mulliken charges — much more accurate than xTB
    # for heteroatom-rich molecules (B, Cl, etc.)
    logger.info("    Computing DFT charges for ESP-guided docking...")
    t_charges = _compute_dft_charges(template_mol)
    m_charges = _compute_dft_charges(monomer_mol)

    surface_pts = generate_surface_points(template_mol)

    # ── Identify reactive functional groups on template ──
    # Heteroatoms (O, N, S) and their neighbors are H-bond sites
    # These should get 50% of total orientations (focused sampling)
    functional_group_atoms = set()
    for i in range(template_mol.GetNumAtoms()):
        atom = template_mol.GetAtomWithIdx(i)
        Z = atom.GetAtomicNum()
        if Z in (7, 8, 16):  # N, O, S
            functional_group_atoms.add(i)
            # Include neighbors (C=O carbon, N-H hydrogen, etc.)
            for neighbor in atom.GetNeighbors():
                functional_group_atoms.add(neighbor.GetIdx())

    if functional_group_atoms:
        fg_names = [template_mol.GetAtomWithIdx(i).GetSymbol() for i in functional_group_atoms]
        logger.info(f"    Functional group atoms: {len(functional_group_atoms)} "
                    f"({', '.join(set(fg_names))})")

    # Weight surface points: functional group region gets much higher weight
    surface_weights = np.zeros(len(surface_pts))
    for i, pt in enumerate(surface_pts):
        dists_to_atoms = np.linalg.norm(t_pos - pt, axis=1)
        closest = np.argmin(dists_to_atoms)
        # Base weight from charge
        w = abs(t_charges[closest]) + 0.05
        # Boost 5x if nearest atom is part of a functional group
        if closest in functional_group_atoms:
            w *= 5.0
        surface_weights[i] = w
    surface_weights /= surface_weights.sum()

    # Multi-offset distances
    offsets = [1.5, surface_offset, surface_offset + 1.0]
    n_per_offset = max(1, n_orientations // len(offsets))

    rng = np.random.RandomState(random_seed)
    valid = []

    for offset in offsets:
        # Sample surface points weighted by charge + functional group boost
        indices = rng.choice(len(surface_pts), n_per_offset,
                             replace=True, p=surface_weights)

        for idx in indices:
            pt = surface_pts[idx]
            direction = pt - t_centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm

            target = pt + direction * offset

            # ESP-guided rotation: align dipoles for complementary interaction
            rot = _esp_guided_rotation(t_pos, t_charges, m_centered,
                                        m_charges, target)
            m_rotated = rot.apply(m_centered) + target

            # Random perturbation (±20°) for diversity around ESP optimum
            perturb = Rotation.from_rotvec(rng.randn(3) * 0.35)
            m_rotated = perturb.apply(m_rotated - target) + target

            # Clash check
            dists = cdist(t_pos, m_rotated)
            if dists.min() < 1.2:
                continue

            c_num = np.concatenate([t_num, m_num])
            c_pos = np.concatenate([t_pos, m_rotated])
            valid.append((c_num, c_pos))

    return valid


def _xtb_sp_worker(args):
    """Module-level worker for parallel xTB single-point screening."""
    from tblite.interface import Calculator
    idx, c_num, c_pos, n_template, e_template = args
    try:
        c_bohr = c_pos * ANG_TO_BOHR
        e_complex = Calculator("GFN2-xTB", c_num, c_bohr).singlepoint().get("energy")
        m_num, m_bohr = c_num[n_template:], c_bohr[n_template:]
        e_monomer = Calculator("GFN2-xTB", m_num, m_bohr).singlepoint().get("energy")
        de = (e_complex - e_template - e_monomer) * HARTREE_TO_KCAL
        return (idx, de)
    except Exception:
        return (idx, float("inf"))


def screen_orientations_sp(template_mol: Chem.Mol,
                           orientations: list,
                           top_n: int = 20) -> list:
    """GFN2-xTB single-point screening of docked orientations.

    CPU-parallel via ProcessPoolExecutor for ~4x speedup.
    Returns top_n (orientation, sp_energy) pairs sorted by energy.
    """
    t_num, t_pos = mol_to_arrays(template_mol)
    n_template = len(t_num)
    e_template = xtb_single_point(t_num, t_pos)

    # Serial SP screening (tblite deadlocks in ProcessPoolExecutor on large molecules)
    sp_args = [(i, c_num, c_pos, n_template, e_template)
               for i, (c_num, c_pos) in enumerate(orientations)]
    results = [_xtb_sp_worker(a) for a in sp_args]

    results.sort(key=lambda x: x[1])
    top = [(idx, de) for idx, de in results[:top_n] if de < float("inf")]
    return [(orientations[idx], de) for idx, de in top]


def optimize_top_candidates(template_mol: Chem.Mol, monomer_mol: Chem.Mol,
                            top_orientations: list) -> dict:
    """Full GFN2-xTB optimization on top SP candidates.

    Optimization: template and monomer energies computed ONCE (not per candidate).
    Only complex optimization is repeated for each candidate.
    """
    t_num, t_pos = mol_to_arrays(template_mol)
    m_num_ref, m_pos_ref = mol_to_arrays(monomer_mol)

    # Pre-compute isolated energies ONCE
    with tempfile.TemporaryDirectory(prefix="xtb_iso_") as td:
        e_template, _ = xtb_optimize(t_num, t_pos, td)
        e_monomer, _ = xtb_optimize(m_num_ref, m_pos_ref, td)
    logger.info(f"    E_template={e_template:.6f}, E_monomer={e_monomer:.6f} Ha")

    best_de = float("inf")
    best_nums = None
    best_poss = None
    best_idx = 0
    all_de = []                      # GAP 1: keep the whole pose ensemble

    for rank, ((c_num, c_pos), sp_energy) in enumerate(top_orientations):
        try:
            with tempfile.TemporaryDirectory(prefix=f"xtb_opt{rank}_") as td:
                e_complex, opt_pos = xtb_optimize(c_num, c_pos, td)
            de = (e_complex - e_template - e_monomer) * HARTREE_TO_KCAL
            all_de.append(round(de, 3))
            if de < best_de:
                best_de = de
                best_nums = c_num
                best_poss = opt_pos
                best_idx = rank
        except Exception:
            continue

    # Post-hoc: identify which template atoms are closest to monomer
    # (= discovered binding site, as in QuantumDock)
    binding_site_info = _identify_binding_site(template_mol, best_poss, len(t_num))

    # Build RDKit Mol from optimized coordinates for DFT handoff
    best_complex_mol = _arrays_to_mol(template_mol, monomer_mol, best_poss)

    return {
        "best_energy_kcal": round(best_de, 3),
        "best_energy_hartree": round(best_de / HARTREE_TO_KCAL, 6),
        "best_complex_mol": best_complex_mol,
        "best_numbers": best_nums,
        "best_positions": best_poss,
        "best_rank": best_idx,
        "binding_site": binding_site_info,
        "all_energies_kcal": all_de,      # GAP 1: pose ensemble
    }


def _vina_dock(template_mol: Chem.Mol, monomer_mol: Chem.Mol,
               n_poses: int = 20) -> list:
    """Generate docked poses using AutoDock Vina.

    Professional docking software with optimized scoring function
    for noncovalent interactions. Returns list of (numbers, positions).
    Falls back gracefully if Vina is not available.
    """
    try:
        from vina import Vina
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        import tempfile
    except ImportError:
        logger.warning("    Vina/meeko not installed, skipping Vina docking")
        return []

    try:
        t_num, t_pos = mol_to_arrays(template_mol)
        m_num, m_pos = mol_to_arrays(monomer_mol)
        t_centroid = t_pos.mean(axis=0)

        # Compute box size from template extent + padding
        extent = t_pos.max(axis=0) - t_pos.min(axis=0)
        box_size = extent + 15.0  # 15A padding

        with tempfile.TemporaryDirectory(prefix="vina_") as td:
            # Prepare PDBQT files
            t_pdbqt = f"{td}/template.pdbqt"
            m_pdbqt = f"{td}/monomer.pdbqt"

            # Use meeko for PDBQT preparation
            # meeko 0.7+: prepare() returns list of MoleculeSetup objects
            # Default merge: nonpolar H merged into parent (standard for Vina)
            preparator = MoleculePreparation()

            mol_setups = preparator.prepare(template_mol)
            pdbqt_str = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
            # Vina receptor format: strip ROOT/BRANCH/END tags (rigid receptor)
            receptor_lines = [l for l in pdbqt_str.splitlines()
                              if not l.startswith(("ROOT", "ENDROOT", "BRANCH",
                                                   "ENDBRANCH", "TORSDOF"))]
            with open(t_pdbqt, "w") as f:
                f.write("\n".join(receptor_lines) + "\n")

            mol_setups_m = preparator.prepare(monomer_mol)
            pdbqt_str_m = PDBQTWriterLegacy.write_string(mol_setups_m[0])[0]
            with open(m_pdbqt, "w") as f:
                f.write(pdbqt_str_m)

            # Parse PDBQT→RDKit index mapping from REMARK lines
            # REMARK SMILES IDX: "rdkit_idx pdbqt_pos" pairs (heavy atoms)
            # REMARK H PARENT: "parent_pdbqt_pos h_pdbqt_pos" (polar H)
            pdbqt_to_rdkit = {}
            h_parent_map = {}  # pdbqt_h_pos → pdbqt_parent_pos
            for line in pdbqt_str_m.splitlines():
                if line.startswith("REMARK SMILES IDX"):
                    tokens = line.split()[3:]
                    for j in range(0, len(tokens), 2):
                        rdkit_idx = int(tokens[j]) - 1  # 1-based → 0-based
                        pdbqt_pos = int(tokens[j + 1]) - 1
                        pdbqt_to_rdkit[pdbqt_pos] = rdkit_idx
                elif line.startswith("REMARK H PARENT"):
                    tokens = line.split()[3:]
                    for j in range(0, len(tokens), 2):
                        parent_pos = int(tokens[j]) - 1
                        h_pos = int(tokens[j + 1]) - 1
                        h_parent_map[h_pos] = parent_pos

            # Map polar H to RDKit: find H bonded to the parent atom
            for h_pdbqt, parent_pdbqt in h_parent_map.items():
                parent_rdkit = pdbqt_to_rdkit.get(parent_pdbqt)
                if parent_rdkit is not None:
                    parent_atom = monomer_mol.GetAtomWithIdx(parent_rdkit)
                    for neighbor in parent_atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 1 and neighbor.GetIdx() not in pdbqt_to_rdkit.values():
                            pdbqt_to_rdkit[h_pdbqt] = neighbor.GetIdx()
                            break

            # Run Vina
            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(t_pdbqt)
            v.set_ligand_from_file(m_pdbqt)
            v.compute_vina_maps(center=t_centroid.tolist(),
                                box_size=box_size.tolist())
            v.dock(exhaustiveness=64, n_poses=n_poses)

            # Extract poses: map Vina heavy+polar-H coords back to full RDKit mol
            energies = v.energies(n_poses=n_poses)
            all_coords = v.poses(n_poses=n_poses, coordinates_only=True)

            from rdkit.Geometry import Point3D

            poses = []
            for i in range(len(energies)):
                vina_coords = all_coords[i]  # (n_pdbqt_atoms, 3)

                if len(vina_coords) != len(pdbqt_to_rdkit):
                    continue  # unexpected size

                # Copy monomer mol and update mapped atom positions
                m_copy = Chem.RWMol(monomer_mol)
                conf = m_copy.GetConformer()
                for pdbqt_pos, rdkit_idx in pdbqt_to_rdkit.items():
                    x, y, z = vina_coords[pdbqt_pos]
                    conf.SetAtomPosition(rdkit_idx, Point3D(float(x), float(y), float(z)))

                # Re-optimize H positions (heavy atoms already placed by Vina)
                try:
                    from rdkit.Chem import AllChem as _AC
                    ff = _AC.MMFFGetMoleculeForceField(m_copy)
                    if ff is not None:
                        # Fix mapped atoms, only optimize unmapped H
                        for rdkit_idx in pdbqt_to_rdkit.values():
                            ff.AddFixedPoint(rdkit_idx)
                        ff.Minimize(maxIts=50)
                except Exception:
                    pass  # H positions stay at original relative positions

                _, m_pos_new = mol_to_arrays(m_copy)
                c_num = np.concatenate([t_num, m_num])
                c_pos = np.concatenate([t_pos, m_pos_new])
                poses.append((c_num, c_pos))

            if energies is not None and len(energies) > 0:
                logger.info(f"    Vina: {len(poses)} poses, best={energies[0][0]:+.1f} kcal/mol")
            return poses

    except Exception as e:
        logger.warning(f"    Vina docking failed: {e}")
        return []


def _arrays_to_mol(template_mol: Chem.Mol, monomer_mol: Chem.Mol,
                   complex_positions: np.ndarray) -> "Chem.Mol":
    """Convert optimized numpy positions back to an RDKit Mol for DFT handoff."""
    from rdkit.Geometry import Point3D
    if complex_positions is None:
        return None
    combo = Chem.CombineMols(template_mol, monomer_mol)
    combo = Chem.RWMol(combo)
    conf = combo.GetConformer()
    for i in range(combo.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(
            float(complex_positions[i, 0]),
            float(complex_positions[i, 1]),
            float(complex_positions[i, 2]),
        ))
    return combo.GetMol()


def _identify_binding_site(template_mol: Chem.Mol, complex_positions: np.ndarray,
                           n_template: int) -> dict:
    """Post-hoc binding site identification from optimized complex.

    Finds the template atom closest to the monomer → the binding site.
    Also checks if it's a donor/acceptor (H-bond capable).
    """
    if complex_positions is None:
        return {"atom_idx": -1, "symbol": "?", "type": "unknown"}

    t_pos = complex_positions[:n_template]
    m_pos = complex_positions[n_template:]
    m_centroid = m_pos.mean(axis=0)

    # Find closest template atom to monomer centroid
    dists = np.linalg.norm(t_pos - m_centroid, axis=1)
    closest_idx = int(np.argmin(dists))
    atom = template_mol.GetAtomWithIdx(closest_idx)
    symbol = atom.GetSymbol()

    # Check if H-bond capable
    donor_smarts = Chem.MolFromSmarts("[#7,#8;!H0]")
    acceptor_smarts = Chem.MolFromSmarts("[#7,#8;H0,$([#7,#8;-])]")
    is_donor = any(closest_idx in m for m in template_mol.GetSubstructMatches(donor_smarts))
    is_acceptor = any(closest_idx in m for m in template_mol.GetSubstructMatches(acceptor_smarts))

    if is_donor and is_acceptor:
        site_type = "donor+acceptor"
    elif is_donor:
        site_type = "donor"
    elif is_acceptor:
        site_type = "acceptor"
    else:
        site_type = "vdW/electrostatic"

    return {
        "atom_idx": closest_idx,
        "symbol": symbol,
        "type": site_type,
        "distance_to_monomer": round(float(dists[closest_idx]), 2),
    }


# ═══════════════════════════════════════════════════════════════════
#  Adaptive orientation count
# ═══════════════════════════════════════════════════════════════════

def _adaptive_n_orientations(template_mol: Chem.Mol, monomer_mol: Chem.Mol,
                              base_n: int = None) -> int:
    """Scale orientation count by combined molecular size.

    Larger molecules need more orientations to adequately sample
    the interaction surface. Base: 100 for ~20 heavy atoms (Phe-MAA),
    scales linearly with combined heavy atom count.
    """
    base_n = base_n or N_DOCK_ORIENTATIONS
    n_heavy_t = sum(1 for a in template_mol.GetAtoms() if a.GetAtomicNum() > 1)
    n_heavy_m = sum(1 for a in monomer_mol.GetAtoms() if a.GetAtomicNum() > 1)
    n_heavy = n_heavy_t + n_heavy_m
    # Reference: Phe(9) + MAA(5) = 14 heavy atoms → base_n
    scale = max(1.0, n_heavy / 14.0)
    adapted = int(base_n * scale)
    return min(adapted, 1000)  # Cap at 1000


# ═══════════════════════════════════════════════════════════════════
#  Main entry: compute_binding_energy
# ═══════════════════════════════════════════════════════════════════

def compute_binding_energy(monomer_name: str, monomer_smiles: str,
                           template_mol: Chem.Mol) -> dict:
    """Compute xTB binding energy for one monomer via QuantumDock approach."""
    try:
        return _compute_quantumdock(monomer_name, monomer_smiles, template_mol)
    except Exception as e:
        logger.warning(f"Failed for {monomer_name}: {e}")
        return {"name": monomer_name, "success": False, "error": str(e)}


def _compute_quantumdock(monomer_name: str, monomer_smiles: str,
                         template_mol: Chem.Mol) -> dict:
    """QuantumDock-style: surface orientations → SP screen → optimize.

    If ENSEMBLE_DOCKING is True in config, generates multiple template
    conformers and docks to each, taking the best binding energy.
    """
    # Ensemble docking: try multiple monomer conformers, take best
    try:
        from .config import ENSEMBLE_DOCKING
    except ImportError:
        ENSEMBLE_DOCKING = False

    if ENSEMBLE_DOCKING:
        conformers = smiles_to_mol3d_ensemble(monomer_smiles, n_confs=10, n_keep=3)
        best_result = None
        for ci, conf_mol in enumerate(conformers):
            try:
                result = _quantumdock_single(monomer_name, conf_mol, template_mol)
                if result.get("success") and (
                    best_result is None or result["dE_kcal"] < best_result["dE_kcal"]
                ):
                    best_result = result
                    best_result["ensemble_conformer"] = ci
            except Exception:
                pass
        if best_result:
            best_result["n_conformers_tested"] = len(conformers)
            return best_result

    monomer_mol = smiles_to_mol3d(monomer_smiles)
    return _quantumdock_single(monomer_name, monomer_mol, template_mol)


def _quantumdock_single(monomer_name: str, monomer_mol: Chem.Mol,
                         template_mol: Chem.Mol) -> dict:
    """Single-conformer QuantumDock computation."""

    # Adaptive orientation count based on molecular size
    n_orient = _adaptive_n_orientations(template_mol, monomer_mol)

    # Step 1a: ESP-guided surface orientations
    orientations = generate_docked_orientations(
        template_mol, monomer_mol,
        n_orientations=n_orient,
        surface_offset=DOCK_SURFACE_OFFSET,
    )
    n_esp = len(orientations)

    # Step 1b: AutoDock Vina docking poses (professional scoring function)
    vina_poses = _vina_dock(template_mol, monomer_mol, n_poses=30)
    orientations.extend(vina_poses)

    n_valid = len(orientations)
    logger.info(f"  [{monomer_name}] {n_esp} ESP + {len(vina_poses)} Vina = "
                f"{n_valid} total orientations")

    if n_valid == 0:
        return {"name": monomer_name, "success": False,
                "error": "All orientations clashed"}

    # Step 2: GFN2-xTB single-point screening
    top = screen_orientations_sp(
        template_mol, orientations, top_n=N_TOP_FOR_OPTIMIZATION
    )
    logger.info(f"  [{monomer_name}] Phase 1 SP → top {len(top)} "
                f"(best: {top[0][1]:+.3f} kcal/mol)")

    # Step 3: Phase 2 — focused refinement around best SP regions
    # Generate additional orientations near the best binding sites
    if len(top) >= 3:
        from scipy.spatial.transform import Rotation as Rot
        rng2 = np.random.RandomState(123)
        t_num, t_pos = mol_to_arrays(template_mol)
        m_num, m_pos = mol_to_arrays(monomer_mol)
        m_centered = m_pos - m_pos.mean(axis=0)
        n_refine = n_orient // 3  # ~1/3 of original budget for refinement
        refined = []

        for (c_num_top, c_pos_top), sp_e in top[:3]:  # Top 3 sites
            monomer_center = c_pos_top[len(t_num):].mean(axis=0)
            for _ in range(n_refine // 3):
                # Small perturbations: ±0.5A translation + ±10° rotation
                shift = rng2.randn(3) * 0.5
                rot = Rot.from_rotvec(rng2.randn(3) * 0.17)  # ~10°
                m_new = rot.apply(m_centered) + monomer_center + shift
                from scipy.spatial.distance import cdist as cdist2
                if cdist2(t_pos, m_new).min() < 1.2:
                    continue
                refined.append((np.concatenate([t_num, m_num]),
                                np.concatenate([t_pos, m_new])))

        if refined:
            logger.info(f"  [{monomer_name}] Phase 2 refine: {len(refined)} "
                        f"orientations near top 3 sites")
            top_refined = screen_orientations_sp(template_mol, refined,
                                                  top_n=N_TOP_FOR_OPTIMIZATION)
            # Merge with Phase 1 top, keep best overall
            all_top = top + top_refined
            all_top.sort(key=lambda x: x[1])
            top = all_top[:N_TOP_FOR_OPTIMIZATION]
            logger.info(f"  [{monomer_name}] After merge: best={top[0][1]:+.3f}")

    # Step 4: Full optimization on top candidates
    opt_result = optimize_top_candidates(template_mol, monomer_mol, top)
    de = opt_result["best_energy_kcal"]
    site = opt_result["binding_site"]
    # GAP 1 (partial): Boltzmann-weight the pose ensemble + site-heterogeneity
    try:
        from .chemistry_aware import ensemble_binding
        ensemble = ensemble_binding(opt_result.get("all_energies_kcal"))
    except Exception:
        ensemble = {}
    logger.info(f"  [{monomer_name}] Optimized → dE={de:+.3f} kcal/mol"
                + (f", ⟨dE⟩_Boltz={ensemble['boltzmann_mean_kcal']:+.3f}, "
                   f"n_eff={ensemble['n_effective_poses']}" if ensemble else "")
                + f", site: atom {site['atom_idx']} ({site['symbol']}, {site['type']})")

    # Save optimized complex coordinates for Stage 2 DFT handoff
    # (RDKit Mol is not JSON-serializable, so store as atom list)
    complex_mol = opt_result["best_complex_mol"]
    complex_coords = []
    if complex_mol is not None:
        conf = complex_mol.GetConformer()
        for i in range(complex_mol.GetNumAtoms()):
            sym = complex_mol.GetAtomWithIdx(i).GetSymbol()
            pos = conf.GetAtomPosition(i)
            complex_coords.append([sym, round(pos.x, 6), round(pos.y, 6), round(pos.z, 6)])

    return {
        "name": monomer_name,
        "smiles": monomer_smiles,   # stamp for the SMILES-change cache guard
        "dE_hartree": opt_result["best_energy_hartree"],
        "dE_kcal": de,
        "success": True,
        "search_mode": "quantumdock",
        "fast_screen_method": "GFN2-xTB_singlepoint",
        "n_orientations_generated": n_valid,
        "n_orientations_valid": n_valid,
        "n_top_optimized": len(top),
        "best_direction": f"site_{site['atom_idx']}_{site['type']}",
        "binding_site": site,
        "complex_coords": complex_coords,
        "ensemble": ensemble,     # GAP 1: Boltzmann pose ensemble + heterogeneity
    }



# ═══════════════════════════════════════════════════════════════════
#  Pipeline entry point
# ═══════════════════════════════════════════════════════════════════

def run_stage1(template_smiles: str = None,
               monomer_library: dict = None,
               output_dir: str = OUTPUT_DIRS["stage1"]) -> list:
    """Run xTB screening over all monomers.

    Returns list of monomer names with negative binding energy (dE < 0),
    sorted by binding energy (most negative first).
    """
    template_smiles = template_smiles or TEMPLATE_SMILES
    monomer_library = monomer_library or MONOMER_LIBRARY

    logger.info(f"Stage 1: Screening {len(monomer_library)} monomers "
                f"(mode={COMPLEX_SEARCH_MODE})")
    template_mol = smiles_to_mol3d(template_smiles)

    # ── Load existing results for skip logic ──
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    existing_results = []
    all_json = out_path / "stage1_all.json"
    if all_json.exists():
        with open(all_json) as f:
            existing_results = json.load(f)

    # SMILES-stamp guard: recompute a monomer whose cached entry was docked for a
    # different SMILES (e.g. APB para→meta). Legacy entries with no stamp are
    # trusted (skip) so the xTB cache isn't nuked wholesale — purge those by hand.
    _existing_by_name = {r["name"]: r for r in existing_results if r.get("success")}

    def _fresh(n, s):
        r = _existing_by_name.get(n)
        return r is not None and r.get("smiles") in (None, s)

    monomers_to_run = {n: s for n, s in monomer_library.items()
                       if not _fresh(n, s)}
    # Drop stale entries we are about to recompute (avoid duplicate names).
    existing_results = [r for r in existing_results
                        if r.get("name") not in monomers_to_run]
    skip_count = len(monomer_library) - len(monomers_to_run)
    if skip_count > 0:
        logger.info(f"  {skip_count} monomers already computed, skipping")
    logger.info(f"  {len(monomers_to_run)} monomers to compute")

    results = list(existing_results)
    if monomers_to_run:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(compute_binding_energy, name, smiles, template_mol): name
                for name, smiles in monomers_to_run.items()
            }
            for future in as_completed(futures):
                res = future.result()
                if res["success"]:
                    results.append(res)
                    logger.info(f"  {res['name']:>10s}: dE = {res['dE_kcal']:+.3f} kcal/mol")
                    # Save incrementally
                    with open(all_json, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    logger.warning(f"  {res['name']:>10s}: FAILED — {res.get('error', '?')}")

    results.sort(key=lambda r: r["dE_kcal"])

    # Filter: only monomers with negative binding energy (actual binding)
    # dE < 0 means the complex is more stable than isolated molecules
    bound = [r for r in results if r["dE_kcal"] < 0]
    unbound = [r for r in results if r["dE_kcal"] >= 0]
    if unbound:
        logger.info(f"  Filtered out {len(unbound)} non-binding monomers (dE >= 0): "
                    f"{[r['name'] for r in unbound]}")
    # Sort binders by xTB binding energy (most negative first)
    top_results = sorted(bound, key=lambda r: r["dE_kcal"])
    # Optional funnel cap: keep only the STAGE1_TOP_N strongest binders for Stage 2.
    # STAGE1_TOP_N=None (default) → pass ALL binders (no numeric funnel).
    if STAGE1_TOP_N and len(top_results) > STAGE1_TOP_N:
        logger.info(f"  Funnel cap: keeping top {STAGE1_TOP_N} of {len(top_results)} "
                    f"binders for Stage 2 (STAGE1_TOP_N)")
        top_results = top_results[:STAGE1_TOP_N]

    with open(all_json, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_path / "stage1_top.json", "w") as f:
        json.dump(top_results, f, indent=2)

    top_names = [r["name"] for r in top_results]
    logger.info(f"Stage 1 complete. {len(top_names)} binding monomers: {top_names}")
    return top_names


if __name__ == "__main__":
    run_stage1()

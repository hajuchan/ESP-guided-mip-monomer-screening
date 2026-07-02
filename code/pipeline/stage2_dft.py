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
    DFT_FUNCTIONAL_HBOND,
    DFT_FUNCTIONAL_DISP,
    HBOND_DOMINANCE_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage2] %(message)s")
logger = logging.getLogger(__name__)


def select_functional(template_smiles: str, monomer_smiles: str) -> str:
    """상호작용 유형에 따라 범함수를 자동 선택.

    H-bond donor/acceptor가 충분하면 ωB97XD (H-bond 정확),
    부족하면 ωB97M-V (분산력 정확).
    """
    from rdkit.Chem import Descriptors, Lipinski
    t_mol = Chem.MolFromSmiles(template_smiles)
    m_mol = Chem.MolFromSmiles(monomer_smiles)
    if t_mol is None or m_mol is None:
        return DFT_FUNCTIONAL

    # H-bond donor + acceptor 합산 (template + monomer)
    n_hbd = Lipinski.NumHDonors(t_mol) + Lipinski.NumHDonors(m_mol)
    n_hba = Lipinski.NumHAcceptors(t_mol) + Lipinski.NumHAcceptors(m_mol)
    n_hb_total = n_hbd + n_hba

    if n_hb_total >= HBOND_DOMINANCE_THRESHOLD:
        func = DFT_FUNCTIONAL_HBOND  # ωB97XD
        reason = f"H-bond 지배 (HBD={n_hbd}+HBA={n_hba}={n_hb_total})"
    else:
        func = DFT_FUNCTIONAL_DISP   # ωB97M-V
        reason = f"분산력 지배 (HBD={n_hbd}+HBA={n_hba}={n_hb_total})"

    logger.info(f"    범함수 선택: {func} — {reason}")
    return func


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


# ── Covalent bond detection ────────────────────────────────────────

# Typical covalent bond distance thresholds (Angstrom)
# If inter-fragment atom distance falls below these, a covalent bond likely formed
_COVALENT_THRESHOLDS = {
    ("B", "O"): 1.55,  ("B", "N"): 1.65,  # boronate ester / B-N dative
    ("C", "O"): 1.55,  ("C", "N"): 1.55,  # amide, ester
    ("C", "C"): 1.60,  ("C", "S"): 1.90,
    ("N", "H"): 1.15,  ("O", "H"): 1.10,
    ("Si", "O"): 1.75, ("P", "O"): 1.70,  # silane, phosphate
}
_DEFAULT_COVALENT = 1.60  # Å


def _check_covalent_bonds(complex_atom_str: str, n_template: int,
                          monomer_name: str) -> str:
    """Detect if covalent bonds formed between template and monomer fragments
    during DFT geometry optimization.

    Returns warning string if covalent bonds detected, empty string otherwise.
    Covalent bonds are not errors — they indicate covalent-MIP candidates
    (e.g., boronate ester MIP) that should be classified separately from
    non-covalent MIP monomers.
    """
    import numpy as np

    parts = [p.strip() for p in complex_atom_str.split(";") if p.strip()]
    if len(parts) < n_template + 1:
        return ""

    # Parse coordinates
    template_atoms, monomer_atoms = [], []
    for i, part in enumerate(parts):
        tokens = part.split()
        sym = tokens[0].replace("ghost-", "")
        pos = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        if i < n_template:
            template_atoms.append((sym, pos))
        else:
            monomer_atoms.append((sym, pos))

    # Check inter-fragment distances
    new_bonds = []
    for t_sym, t_pos in template_atoms:
        for m_sym, m_pos in monomer_atoms:
            dist = np.linalg.norm(t_pos - m_pos)
            pair = tuple(sorted([t_sym, m_sym]))
            threshold = _COVALENT_THRESHOLDS.get(pair, _DEFAULT_COVALENT)
            if dist < threshold:
                new_bonds.append(f"{t_sym}-{m_sym} ({dist:.2f}Å < {threshold}Å)")

    if new_bonds:
        bonds_str = ", ".join(new_bonds[:3])  # max 3 shown
        return (f"공유결합 형성 감지: {bonds_str}. "
                f"{monomer_name}은(는) 공유결합 MIP 후보일 수 있음 "
                f"(비공유결합 monomer와 직접 비교 주의)")
    return ""


# ── DFT Calculations ────────────────────────────────────────────────

def _dft_optimize(atom_str: str, basis: str, eps: float,
                  tmpdir: str, functional: str = None) -> str:
    """GPU-accelerated DFT geometry optimization with PCM solvation.

    Uses selected functional + PCM on GPU via gpu4pyscf + geomeTRIC.
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

    func = functional or DFT_FUNCTIONAL
    lib.param.TMPDIR = tmpdir

    # 2-level optimization: small basis + RI-J for speed, then large basis for SP
    mol = gto.M(atom=atom_str, basis=DFT_OPT_BASIS, verbose=0)
    mf = dft.RKS(mol).density_fit()  # RI-J density fitting (~3x faster)
    mf.xc = func
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
               tmpdir: str, use_gpu: bool = True,
               functional: str = None) -> float:
    """Run DFT/basis with PCM solvation. Returns energy in Hartree."""
    os.environ["PYSCF_TMPDIR"] = tmpdir

    from pyscf import gto, dft, solvent, lib
    lib.param.TMPDIR = tmpdir
    func = functional or DFT_FUNCTIONAL

    mol = gto.M(atom=atom_str, basis=basis, verbose=0)
    mf = dft.RKS(mol).density_fit()  # RI-J density fitting
    mf.xc = func

    # Apply D3BJ dispersion correction explicitly
    # (LibXC may not auto-activate ωB97XD's built-in D2)
    mf = _apply_d3bj(mf, mol)

    from pyscf.solvent import PCM
    mf = PCM(mf)
    mf.with_solvent.eps = eps
    # Solvent-specific probe radius (default=1.385Å is for water)
    # Acetonitrile: ~2.18Å based on molecular radius (Bondi vdW)
    _SOLVENT_PROBE_RADIUS = {35.69: 2.18, 78.39: 1.385}  # eps → probe(Å)
    probe = _SOLVENT_PROBE_RADIUS.get(eps, 1.385)
    try:
        mf.with_solvent.lebedev_order = 29
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.radii_table = None  # use default UFF radii
        # Note: PySCF PCM doesn't directly expose probe radius as a parameter.
        # The cavity is built from atom-specific UFF radii, not a solvent probe.
        # For acetonitrile, the main effect comes from eps (dielectric constant).
    except AttributeError:
        pass

    # GPU acceleration
    if use_gpu:
        mf = _try_gpu(mf)

    mf.kernel()
    return mf.e_tot


def dft_energy_ghost(real_atom_str: str, ghost_atom_str: str,
                     basis: str, eps: float, tmpdir: str,
                     use_gpu: bool = True,
                     functional: str = None) -> float:
    """DFT energy with ghost atoms for BSSE counterpoise correction.

    Boys & Bernardi, Mol. Phys. 1970.
    real_atom_str: atoms of the fragment being computed (at complex geometry)
    ghost_atom_str: atoms of the OTHER fragment (basis only, at complex geometry)
    """
    os.environ["PYSCF_TMPDIR"] = tmpdir

    from pyscf import gto, dft, solvent, lib
    lib.param.TMPDIR = tmpdir
    func = functional or DFT_FUNCTIONAL

    # Combine real atoms + ghost atoms (ghost provides basis functions only)
    combined_atom = real_atom_str + "; " + _make_ghost_string(ghost_atom_str)
    mol = gto.M(atom=combined_atom, basis=basis, verbose=0)

    mf = dft.RKS(mol).density_fit()  # RI-J
    mf.xc = func
    # Apply PCM for consistent solvation with complex energy calculation
    from pyscf.solvent import PCM as _PCM
    mf = _PCM(mf)
    mf.with_solvent.eps = eps
    try:
        mf.with_solvent.lebedev_order = 29
        mf.with_solvent.method = 'IEF-PCM'
    except AttributeError:
        pass
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
                        prebuilt_complex_mol: "Chem.Mol" = None,
                        functional_override: str = None) -> dict:
    """Compute DFT binding energy with BSSE for one (monomer, solvent) pair.

    If prebuilt_complex_mol is provided, use it instead of building from direction.
    Automatically selects ωB97XD (H-bond) or ωB97M-V (dispersion) based on
    H-bond donor/acceptor count of the template-monomer pair.
    If functional_override is provided, use that functional instead of auto-select.
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

            # Auto-select functional based on interaction type
            if functional_override:
                func = functional_override
            else:
                func = select_functional(template_smiles, monomer_smiles)

            template_atom = mol_to_pyscf_atom(template_mol)
            monomer_atom = mol_to_pyscf_atom(monomer_mol)
            complex_atom = mol_to_pyscf_atom(complex_mol)

            logger.info(f"  GPU DFT optimization ({func}+PCM)...")
            complex_atom = _dft_optimize(complex_atom, basis, eps, tmpdir,
                                         functional=func)

            # Check for covalent bond formation after optimization
            n_template = template_mol.GetNumAtoms()
            covalent_warning = _check_covalent_bonds(
                complex_atom, n_template, monomer_name
            )

            # Extract fragment coordinates from DFT-optimized complex for BSSE
            opt_parts = [p.strip() for p in complex_atom.split(";") if p.strip()]
            template_cp_str = "; ".join(opt_parts[:n_template])
            monomer_cp_str = "; ".join(opt_parts[n_template:])

            # Raw binding energy (no BSSE)
            e_complex = dft_energy(complex_atom, basis, eps, tmpdir,
                                   functional=func)
            e_template = dft_energy(template_atom, basis, eps, tmpdir,
                                    functional=func)
            e_monomer = dft_energy(monomer_atom, basis, eps, tmpdir,
                                   functional=func)
            raw_de = (e_complex - e_template - e_monomer) * HARTREE_TO_KCAL

            # BSSE counterpoise (DFT structure → DFT BSSE = consistent)
            logger.info(f"  BSSE counterpoise correction...")
            e_template_cp = dft_energy_ghost(
                template_cp_str, monomer_cp_str, basis, eps, tmpdir,
                functional=func
            )
            e_monomer_cp = dft_energy_ghost(
                monomer_cp_str, template_cp_str, basis, eps, tmpdir,
                functional=func
            )
            bsse_de = (e_complex - e_template_cp - e_monomer_cp) * HARTREE_TO_KCAL

            logger.info(f"  raw_dE={raw_de:+.3f}, bsse_dE={bsse_de:+.3f} kcal/mol")

            result = {
                "monomer": monomer_name,
                "solvent": solvent_name,
                "functional": func,
                "raw_dE_kcal": round(raw_de, 3),
                "bsse_dE_kcal": round(bsse_de, 3),
                "bsse_correction_kcal": round(raw_de - bsse_de, 3),
                "success": True,
            }
            if covalent_warning:
                result["covalent_warning"] = covalent_warning
                logger.warning(f"  ⚠ {monomer_name}: {covalent_warning}")
            return result
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

    # ── Load existing results for skip logic ──
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    existing_results = {}
    result_file = out_path / "stage2_dft.json"
    if result_file.exists():
        with open(result_file) as f:
            existing_results = json.load(f)

    # Determine which (monomer, solvent) pairs need computation
    skip_count = 0
    pairs_to_run = []
    for m_name, m_smiles in monomers_to_run.items():
        for s_name, eps in solvents.items():
            if (m_name in existing_results
                    and s_name in existing_results[m_name]
                    and existing_results[m_name][s_name].get("bsse_dE") is not None):
                skip_count += 1
                logger.info(f"  {m_name}/{s_name}: already computed, skipping")
            else:
                pairs_to_run.append((m_name, m_smiles, s_name, eps))

    n_total = len(monomers_to_run) * len(solvents)
    max_workers = N_GPU_WORKERS if USE_GPU else N_WORKERS
    logger.info(f"Stage 2: {len(pairs_to_run)} to compute, {skip_count} skipped "
                f"(total {n_total}, workers={max_workers}, GPU={USE_GPU})")

    results = dict(existing_results)  # Start with existing

    # ── Load optimized complex coordinates from Stage 1 ──
    complex_mol_map = {}
    if stage1_path.exists():
        with open(stage1_path) as f:
            stage1_data = json.load(f)
        for entry in stage1_data:
            coords = entry.get("complex_coords")
            if coords and entry.get("name"):
                # Rebuild RDKit Mol from stored coordinates
                from rdkit import Chem
                from rdkit.Chem import AllChem
                from rdkit.Geometry import Point3D
                template_mol_tmp = smiles_to_mol3d(template_smiles)
                monomer_mol_tmp = smiles_to_mol3d(entry["name"]
                    if entry["name"] not in monomer_library
                    else monomer_library[entry["name"]])
                # Build combined mol with Stage 1 optimized coordinates
                try:
                    combo = Chem.CombineMols(template_mol_tmp, monomer_mol_tmp)
                    combo = Chem.RWMol(combo)
                    conf = combo.GetConformer()
                    for i, (sym, x, y, z) in enumerate(coords):
                        if i < combo.GetNumAtoms():
                            conf.SetAtomPosition(i, Point3D(x, y, z))
                    complex_mol_map[entry["name"]] = combo.GetMol()
                    logger.info(f"  {entry['name']}: loaded Stage 1 complex ({len(coords)} atoms)")
                except Exception as e:
                    logger.warning(f"  {entry['name']}: failed to load Stage 1 complex: {e}")

    if pairs_to_run:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for m_name, m_smiles, s_name, eps in pairs_to_run:
                # Use prebuilt complex from Stage 1 if available
                prebuilt = complex_mol_map.get(m_name)
                if prebuilt is not None:
                    logger.info(f"  {m_name}/{s_name}: using Stage 1 optimized complex")

                fut = executor.submit(
                    compute_dft_binding,
                    m_name, m_smiles, template_smiles, s_name, eps,
                    "+x", output_dir,  # direction ignored when prebuilt is provided
                    prebuilt,
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
                    # Save incrementally after each pair
                    with open(result_file, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    logger.warning(f"  {m_name}/{s_name}: FAILED")

    # Final save
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Stage 2 complete. Results: {out_path / 'stage2_dft.json'}")
    return results


# ── Feature 4: ESP Map Visualization ─────────────────────────────────

def generate_esp_map(atom_str: str, basis: str, eps: float,
                     label: str, output_dir: str):
    """Generate 3D ESP isosurface map (Singh 2012 style).

    Computes electron density and ESP on a 3D grid, then renders an
    isosurface of the electron density colored by ESP values.
    Saves interactive HTML (plotly) and static PNG.

    Ref: Singh 2012 — ESP maps for template-monomer interaction analysis.
    """
    out_path = Path(output_dir)

    try:
        from pyscf import gto, dft, tools
        from pyscf.solvent import PCM
        import plotly.graph_objects as go

        mol = gto.M(atom=atom_str, basis=basis, verbose=0)
        mf = dft.RKS(mol)
        mf.xc = DFT_FUNCTIONAL
        mf = PCM(mf)
        mf.with_solvent.eps = eps
        try:
            mf = mf.to_gpu()
        except Exception:
            pass
        mf.kernel()

        dm = mf.make_rdm1()
        if hasattr(dm, 'get'):
            dm = dm.get()  # CuPy → NumPy

        # Generate electron density cube
        density_cube = str(out_path / f"_tmp_density_{label}.cube")
        tools.cubegen.density(mol, density_cube, dm, nx=60, ny=60, nz=60)

        # Generate ESP cube
        esp_cube = str(out_path / f"stage2_esp_{label}.cube")
        tools.cubegen.mep(mol, esp_cube, dm, nx=60, ny=60, nz=60)

        # Parse both cubes
        def _parse_cube(path):
            with open(path) as f:
                lines = f.readlines()
            n_atoms = abs(int(lines[2].split()[0]))
            origin = np.array([float(x) for x in lines[2].split()[1:4]])
            nx = int(lines[3].split()[0])
            ny = int(lines[4].split()[0])
            nz = int(lines[5].split()[0])
            axes = np.zeros((3, 3))
            for i in range(3):
                parts = lines[3 + i].split()
                axes[i] = [float(parts[j]) for j in range(1, 4)]
            # Atom positions
            atoms = []
            for i in range(n_atoms):
                parts = lines[6 + i].split()
                atoms.append({
                    'Z': int(float(parts[0])),
                    'pos': np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                })
            # Volumetric data
            data_start = 6 + n_atoms
            values = []
            for line in lines[data_start:]:
                values.extend([float(v) for v in line.split()])
            data = np.array(values).reshape(nx, ny, nz)
            return data, origin, axes, nx, ny, nz, atoms

        rho, origin, axes, nx, ny, nz, atoms = _parse_cube(density_cube)
        esp, _, _, _, _, _, _ = _parse_cube(esp_cube)

        bohr2ang = 0.529177

        # Marching cubes — smooth molecular surface (Singh 2012 style)
        # iso=0.002 gives the standard "molecular envelope"; 0.02 gives atom-blobs
        from skimage.measure import marching_cubes
        iso_val = 0.002
        if rho.max() < iso_val:
            iso_val = rho.max() * 0.05

        verts, faces, _, _ = marching_cubes(rho, iso_val)

        # Convert grid indices to Cartesian (Angstrom)
        cart_verts = np.zeros_like(verts)
        for i in range(3):
            cart_verts[:, i] = (origin[i] + verts[:, 0] * axes[0, i]
                                + verts[:, 1] * axes[1, i]
                                + verts[:, 2] * axes[2, i]) * bohr2ang

        # Interpolate ESP on isosurface vertices
        from scipy.interpolate import RegularGridInterpolator
        x_grid = np.arange(nx)
        y_grid = np.arange(ny)
        z_grid = np.arange(nz)
        esp_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), esp,
                                             bounds_error=False, fill_value=0)
        esp_on_surface = esp_interp(verts)

        # Clamp ESP for color mapping
        vmax = min(0.05, np.percentile(np.abs(esp_on_surface), 95))
        esp_clamped = np.clip(esp_on_surface, -vmax, vmax)

        # vdW radii (Bondi 1964) for atom marker sizes — Å
        vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52,
                     5: 1.92, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}

        # Singh 2012 style: smooth electron density isosurface colored by ESP
        # Jet-like colormap: red (negative) → orange → yellow → green → blue (positive)
        custom_colorscale = [
            [0.0, "red"],
            [0.25, "orange"],
            [0.45, "yellow"],
            [0.5, "green"],
            [0.55, "cyan"],
            [0.75, "dodgerblue"],
            [1.0, "blue"],
        ]

        # Build plotly figure with 3 traces:
        #   Trace 0: ESP-colored isosurface (variable opacity)
        #   Trace 1: Atom markers (fixed)
        #   Trace 2: Bond lines (fixed)
        fig = go.Figure()

        # Trace 0: Smooth electron density isosurface, colored by ESP
        fig.add_trace(go.Mesh3d(
            x=cart_verts[:, 0], y=cart_verts[:, 1], z=cart_verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=esp_clamped,
            colorscale=custom_colorscale,
            cmin=-vmax, cmax=vmax,
            opacity=0.6,
            colorbar=dict(title='ESP (a.u.)', len=0.5, thickness=12),
            name='ESP surface',
            showscale=True,
        ))

        # Trace 1: Atom markers
        elem_colors = {1: '#FFFFFF', 6: '#808080', 7: '#3050F8', 8: '#FF0D0D',
                       5: '#FFB5B5', 9: '#90E050', 15: '#FF8000',
                       16: '#FFFF30', 17: '#1FF01F'}
        atom_x, atom_y, atom_z = [], [], []
        atom_colors_l, atom_sizes, atom_text = [], [], []
        for atom in atoms:
            Z = atom['Z']
            pos = atom['pos'] * bohr2ang
            atom_x.append(pos[0]); atom_y.append(pos[1]); atom_z.append(pos[2])
            atom_colors_l.append(elem_colors.get(Z, '#FF00FF'))
            atom_sizes.append(vdw_radii.get(Z, 1.5) * 5)
            atom_text.append(f'Z={Z}')

        fig.add_trace(go.Scatter3d(
            x=atom_x, y=atom_y, z=atom_z,
            mode='markers',
            marker=dict(size=atom_sizes, color=atom_colors_l,
                        line=dict(width=1, color='black')),
            text=atom_text, showlegend=False, name='atoms', opacity=1.0,
        ))

        # Trace 2: Bonds (atom pairs within 1.8 Å)
        bond_x, bond_y, bond_z = [], [], []
        n_atoms_total = len(atoms)
        for i in range(n_atoms_total):
            pi = atoms[i]['pos'] * bohr2ang
            for j in range(i+1, n_atoms_total):
                pj = atoms[j]['pos'] * bohr2ang
                d = np.linalg.norm(pi - pj)
                if d < 1.8:
                    bond_x += [pi[0], pj[0], None]
                    bond_y += [pi[1], pj[1], None]
                    bond_z += [pi[2], pj[2], None]
        fig.add_trace(go.Scatter3d(
            x=bond_x, y=bond_y, z=bond_z,
            mode='lines',
            line=dict(color='#444444', width=4),
            showlegend=False, name='bonds', opacity=1.0,
        ))

        # Opacity slider — controls trace 0 (ESP)
        opacity_steps = [
            dict(args=["opacity", [op, 1.0, 1.0]],
                 label=f"{int(op*100)}%", method="restyle")
            for op in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ]

        fig.update_layout(
            title=label.replace("_", " ") + " ESP",
            scene=dict(
                xaxis=dict(visible=False, showbackground=False,
                           showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(visible=False, showbackground=False,
                           showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(visible=False, showbackground=False,
                           showgrid=False, zeroline=False, showticklabels=False),
                aspectmode='data',
                bgcolor='white',
            ),
            paper_bgcolor='white',
            width=1400, height=1200,
            margin=dict(l=10, r=10, t=40, b=10),
            sliders=[dict(
                active=4,
                currentvalue=dict(prefix="Surface Opacity: "),
                len=0.8, pad=dict(t=30), x=0.1,
                steps=opacity_steps,
            )],
        )

        # Save interactive HTML
        html_path = out_path / f"stage2_esp_{label}_3d.html"
        fig.write_html(str(html_path), include_plotlyjs='cdn')

        # Save static PNG
        png_path = out_path / f"stage2_esp_{label}.png"
        try:
            fig.write_image(str(png_path), scale=2)
        except Exception:
            # kaleido may not be available
            pass

        logger.info(f"  ESP 3D map saved: {html_path}")

        # Clean up temp density cube
        import os
        try:
            os.remove(density_cube)
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"  ESP map generation failed for {label}: {e}")
        import traceback
        traceback.print_exc()


def generate_esp_maps_for_binding(monomer_name: str, template_smiles: str,
                                  monomer_smiles: str, solvent_name: str,
                                  eps: float, output_dir: str,
                                  prebuilt_complex_mol=None):
    """Generate ESP maps for template, monomer, and complex."""
    if not USE_ESP_MAP:
        return

    template_mol = smiles_to_mol3d(template_smiles)
    monomer_mol = smiles_to_mol3d(monomer_smiles)
    if prebuilt_complex_mol is not None:
        complex_mol = prebuilt_complex_mol
    else:
        complex_mol = build_complex_mol(template_mol, monomer_mol)

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

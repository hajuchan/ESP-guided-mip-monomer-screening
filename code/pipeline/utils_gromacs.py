"""
GROMACS Utilities for Small-Molecule MIP Pipeline
=================================================
Adapted from Monomer_screening_in_Bio/utils_gromacs.py
Simplified for small-molecule template (aldehyde) + monomer systems.

Key differences from Bio pipeline:
- Template is small molecule (not peptide) → acpype parameterization
- tc-grps = System (no Protein/Non-Protein split)
- Simpler topology management
"""

import json
import logging
import os
import subprocess
import sys
import shutil
from pathlib import Path
from textwrap import dedent

logger = logging.getLogger(__name__)

# ── GROMACS binary ──
GMX_BIN = "gmx"  # Assumes 'source /usr/local/gromacs-gpu/bin/GMXRC' was run


def _tool_env():
    """Environment for external-tool subprocesses (acpype, gmx_MMPBSA) with the
    active conda env's bin/ guaranteed on PATH.

    acpype shells out to AmberTools (antechamber, sqm, tleap) and gmx_MMPBSA to
    sander/MMPBSA.py; if the launching process has a stripped PATH (nohup, cron,
    an IDE, a wrapper script) those binaries are invisible and EVERY molecule
    parameterization fails silently — exactly the '0 MD results' failure mode.
    Prepending sys.executable's dir (the env bin) makes the toolchain findable
    regardless of how the pipeline was launched. Ported from the working sibling
    (Monomer_screening_in_Bio/utils_gromacs.parameterize_monomer)."""
    env = os.environ.copy()
    conda_bin = str(Path(sys.executable).parent)
    if conda_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
    return env


def check_md_toolchain() -> dict:
    """Preflight: verify the small-molecule MD toolchain is reachable. Returns
    {tool: path-or-None}. A missing antechamber/sqm means acpype cannot assign
    charges and all parameterization will fail — better to report loudly than to
    emit 28 empty directories."""
    env_path = _tool_env().get("PATH", "")
    return {t: shutil.which(t, path=env_path)
            for t in ("antechamber", "sqm", "tleap", "acpype", GMX_BIN)}


def load_universe(work_dir, traj=None, prefer_tpr=True):
    """Create an MDAnalysis Universe, tolerant of GROMACS/MDAnalysis version
    skew. Newer GROMACS (2026 → tpx v138) writes .tpr files the installed
    MDAnalysis TPRParser cannot read (NotImplementedError), which otherwise
    aborts EVERY trajectory analysis. md.tpr is tried first (it carries
    charges/masses for better H-bond typing) but on ANY parse failure we fall
    back to a .gro structure, which MDAnalysis reads version-independently —
    sufficient for contact/distance/RDF/selection work on the existing xtc."""
    import MDAnalysis as mda
    work_dir = Path(work_dir)
    tops = []
    if prefer_tpr and (work_dir / "md.tpr").exists():
        tops.append(work_dir / "md.tpr")
    tops += [work_dir / g for g in ("md.gro", "npt.gro", "em.gro", "ions.gro")
             if (work_dir / g).exists()]
    last = None
    for t in tops:
        try:
            return mda.Universe(str(t), str(traj)) if traj else mda.Universe(str(t))
        except Exception as e:
            last = e
            logger.warning(f"  topology {t.name} unreadable by MDAnalysis "
                           f"({type(e).__name__}); falling back to next")
    raise (last or FileNotFoundError(f"no usable topology in {work_dir}"))

# ── MDP Templates ──

MDP_EM = dedent("""\
    ; Energy Minimization
    integrator  = steep
    emtol       = 1000.0
    emstep      = 0.01
    nsteps      = 50000
    nstlist     = 10
    cutoff-scheme = Verlet
    ns_type     = grid
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0
    pbc         = xyz
""")

MDP_NVT = dedent("""\
    ; NVT Equilibration
    integrator  = md
    nsteps      = {nsteps}
    dt          = {dt}
    nstxout-compressed = 5000
    nstenergy   = 5000
    nstlog      = 5000
    continuation = no
    constraint_algorithm = lincs
    constraints = h-bonds
    lincs_iter  = 1
    lincs_order = 4
    cutoff-scheme = Verlet
    ns_type     = grid
    nstlist     = 10
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0
    pbc         = xyz
    tcoupl      = V-rescale
    tc-grps     = System
    tau_t       = 0.1
    ref_t       = {temperature}
    pcoupl      = no
    gen_vel     = yes
    gen_temp    = {temperature}
    gen_seed    = -1
""")

MDP_NPT = dedent("""\
    ; NPT Equilibration
    integrator  = md
    nsteps      = {nsteps}
    dt          = {dt}
    nstxout-compressed = 5000
    nstenergy   = 5000
    nstlog      = 5000
    continuation = yes
    constraint_algorithm = lincs
    constraints = h-bonds
    lincs_iter  = 1
    lincs_order = 4
    cutoff-scheme = Verlet
    ns_type     = grid
    nstlist     = 10
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0
    pbc         = xyz
    tcoupl      = V-rescale
    tc-grps     = System
    tau_t       = 0.1
    ref_t       = {temperature}
    pcoupl      = Parrinello-Rahman
    pcoupltype  = isotropic
    tau_p       = 2.0
    ref_p       = 1.0
    compressibility = 4.5e-5
    refcoord_scaling = com
""")

MDP_PRODUCTION = dedent("""\
    ; Production MD
    integrator  = md
    nsteps      = {nsteps}
    dt          = {dt}
    nstxout-compressed = {nstxout}
    nstenergy   = 5000
    nstlog      = 5000
    continuation = yes
    constraint_algorithm = lincs
    constraints = h-bonds
    lincs_iter  = 1
    lincs_order = 4
    cutoff-scheme = Verlet
    ns_type     = grid
    nstlist     = 10
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0
    pbc         = xyz
    tcoupl      = V-rescale
    tc-grps     = System
    tau_t       = 0.1
    ref_t       = {temperature}
    pcoupl      = Parrinello-Rahman
    pcoupltype  = isotropic
    tau_p       = 2.0
    ref_p       = 1.0
    compressibility = 4.5e-5
""")


# ── GMX wrapper ──

def gmx(cmd_args: list, work_dir: Path, input_text: str = None,
         timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a gmx command. Always uses absolute paths."""
    work_dir = Path(work_dir).resolve()
    cmd = [GMX_BIN] + cmd_args
    logger.debug(f"  gmx: {' '.join(str(a) for a in cmd_args)} (cwd={work_dir})")

    result = subprocess.run(
        cmd, cwd=str(work_dir),
        input=input_text, capture_output=True, text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        logger.warning(f"  gmx stderr: {result.stderr[-500:]}")
    return result


# ── Molecule Parameterization ──

# ── PolCA Organosilane Force Field (Jorge et al., ACS Phys. Chem. Au 2021) ──
# GAFF2/acpype cannot type Si. PolCA supplies GROMACS-compatible organosilane
# LJ parameters, letting silane monomers (APTES, MPTMS, …) run MD instead of
# being dropped. The .itp/.gro are built directly: GAFF2 types for the organic
# part (Si→S proxy for Gasteiger charges) + PolCA LJ for Si. The embedded
# [ atomtypes ] is split out by _copy_and_split_itp; duplicate GAFF2 types
# across molecules become grompp warnings that -maxwarn absorbs, while the
# unique Si type is defined once. Ported from the sibling Bio pipeline.
_POLCA_SI_LJ = {
    "Si0": {"sigma": 0.580, "eps": 0.108},   # 4 alkyl
    "Si1": {"sigma": 0.551, "eps": 0.108},   # 3 alkyl, 1 O
    "Si2": {"sigma": 0.522, "eps": 0.108},   # 2 alkyl, 2 O
    "Si3": {"sigma": 0.493, "eps": 0.108},   # 1 alkyl, 3 O
    "Si4": {"sigma": 0.464, "eps": 0.108},   # 0 alkyl, 4 O (TEOS-like)
}


def _classify_si_type(smiles: str) -> str:
    """PolCA Si type from the number of O neighbours on the Si atom."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Si4"
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 14:
            n_o = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 8)
            return f"Si{n_o}" if f"Si{n_o}" in _POLCA_SI_LJ else "Si4"
    return "Si4"


def _generate_silane_itp(name: str, smiles: str, output_dir) -> dict:
    """Generate GROMACS .itp/.gro for a silane using PolCA Si + GAFF2 organic
    parameters (Jorge 2021). Returns {itp, gro, si_type, method}."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolTransforms

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.useRandomCoords = True; p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) != 0:
        return {"error": "3D embedding failed"}
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    # Charges: Si→S proxy for Gasteiger (Si unsupported), then force Si=+0.9
    rw = Chem.RWMol(mol)
    si_idx = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 14]
    for idx in si_idx:
        rw.GetAtomWithIdx(idx).SetAtomicNum(16)
    AllChem.ComputeGasteigerCharges(rw)
    charges = [float(rw.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
               for i in range(rw.GetNumAtoms())]
    for idx in si_idx:
        charges[idx] = 0.9
    # Enforce a neutral (integer) net charge. Forcing Si=+0.9 with no
    # redistribution leaves the molecule at ~+0.8 e; PME cannot neutralize a
    # fractional net charge (genion adds only integer ions) → spurious uniform
    # background and garbage electrostatics for every silane. Spread the residual
    # over all atoms and absorb 4-decimal rounding so the [atoms] block sums to 0.
    _n = len(charges)
    _resid = sum(charges) / _n if _n else 0.0
    charges = [round(c - _resid, 4) for c in charges]
    _drift = round(sum(charges), 4)
    if abs(_drift) >= 1e-4:
        _j = next((i for i in range(_n) if i not in si_idx), 0)
        charges[_j] = round(charges[_j] - _drift, 4)

    si_type = _classify_si_type(smiles)
    si_lj = _POLCA_SI_LJ[si_type]
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    etype = {6: "c3", 1: "h1", 8: "oh", 7: "n3", 14: si_type, 16: "ss", 5: "c3"}
    mmap = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999,
            14: 28.086, 16: 32.065, 5: 10.811}

    alines = []
    for i in range(n_atoms):
        a = mol.GetAtomWithIdx(i)
        e = a.GetAtomicNum()
        alines.append(
            f"    {i+1:5d} {etype.get(e,'c3'):>10s} 1    {name:>5s} "
            f"{a.GetSymbol()+str(i+1):>5s} {i+1:5d} {charges[i]:10.4f} "
            f"{mmap.get(e,12.011):10.4f}")

    itp_path = output_dir / f"{name}.itp"
    _gaff2_lj = {
        "c3": (0.33977, 0.45104), "h1": (0.24220, 0.08703),
        "oh": (0.32429, 0.38911), "ho": (0.05379, 0.01966),
        "n3": (0.33210, 0.41236), "hn": (0.11065, 0.04184),
        "os": (0.31561, 0.30376), "ha": (0.26255, 0.06736),
        "hc": (0.26002, 0.08703), "ss": (0.35636, 1.04600),
        "ca": (0.33152, 0.41338), "c1": (0.34790, 0.66777),
        "n1": (0.32735, 0.45940), "c2": (0.33152, 0.41338),
    }
    used_types = {etype.get(mol.GetAtomWithIdx(i).GetAtomicNum(), "c3")
                  for i in range(n_atoms)}
    at_lines = ["; name  bond_type  mass    charge  ptype  sigma       epsilon"]
    at_lines.append(f"  {si_type}  {si_type}  {28.086:.3f}  0.000  A  "
                    f"{si_lj['sigma']:.5e}  {si_lj['eps']:.5e}")
    _m = {"c3": 12.011, "h1": 1.008, "oh": 15.999, "ho": 1.008, "n3": 14.007,
          "hn": 1.008, "os": 15.999, "ha": 1.008, "hc": 1.008, "ss": 32.065,
          "ca": 12.011, "c1": 12.011, "n1": 14.007, "c2": 12.011}
    for t in sorted(used_types):
        if t != si_type and t in _gaff2_lj:
            s, e = _gaff2_lj[t]
            at_lines.append(f"  {t}  {t}  {_m.get(t,12.011):.3f}  0.000  A  {s:.5e}  {e:.5e}")
    atomtypes_section = "[ atomtypes ]\n" + "\n".join(at_lines) + "\n\n"

    _std_bond_len = {
        (6, 6): 0.1529, (6, 1): 0.1090, (6, 8): 0.1430, (6, 7): 0.1470,
        (6, 14): 0.1860, (8, 14): 0.1640, (8, 1): 0.0960, (7, 1): 0.1010,
        (14, 14): 0.2340, (6, 16): 0.1810, (16, 1): 0.1340,
    }
    blines = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx() + 1
        j = bond.GetEndAtomIdx() + 1
        e1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum()
        e2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum()
        dist = _std_bond_len.get((min(e1, e2), max(e1, e2)), 0.1500)
        blines.append(f"  {i:5d}  {j:5d}    1    {dist:.4f}  500000.0")

    aangle_lines = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neigh = [n.GetIdx() for n in atom.GetNeighbors()]
        for ni in range(len(neigh)):
            for nj in range(ni+1, len(neigh)):
                try:
                    angle = rdMolTransforms.GetAngleDeg(conf, neigh[ni], idx, neigh[nj])
                    aangle_lines.append(
                        f"  {neigh[ni]+1:5d}  {idx+1:5d}  {neigh[nj]+1:5d}    1    {angle:.2f}  500.0")
                except Exception:
                    pass

    # Proper dihedrals for rotatable (non-ring, non-aromatic) single bonds. The
    # hand-built topology previously had NONE, leaving the alkyl-silane backbone
    # with zero torsional barrier (unphysical free rotation). Add a generic 3-fold
    # sp3 term (~GAFF2 X-c3-c3-X, 0.65 kJ/mol) so backbone sampling is realistic.
    dlines = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetIsAromatic():
            continue
        b, c = bond.GetBeginAtom(), bond.GetEndAtom()
        for a in b.GetNeighbors():
            if a.GetIdx() == c.GetIdx():
                continue
            for d in c.GetNeighbors():
                if d.GetIdx() in (b.GetIdx(), a.GetIdx()):
                    continue
                dlines.append(f"  {a.GetIdx()+1:5d}  {b.GetIdx()+1:5d}  "
                              f"{c.GetIdx()+1:5d}  {d.GetIdx()+1:5d}    1    0.00  0.6500  3")

    bonds_section = "[ bonds ]\n" + "\n".join(blines) + "\n" if blines else ""
    angles_section = "\n[ angles ]\n" + "\n".join(aangle_lines) + "\n" if aangle_lines else ""
    dihedrals_section = "\n[ dihedrals ]\n" + "\n".join(dlines) + "\n" if dlines else ""

    itp_path.write_text(
        f"; {name} - PolCA Si (Jorge 2021) + GAFF2\n"
        f"{atomtypes_section}"
        f"[ moleculetype ]\n{name}    3\n\n[ atoms ]\n"
        + "\n".join(alines) + "\n\n" + bonds_section + angles_section + dihedrals_section,
        encoding="utf-8")

    gro_path = output_dir / f"{name}.gro"
    gl = [f"{name} silane", f" {n_atoms}"]
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        a = mol.GetAtomWithIdx(i)
        gl.append(f"{1:5d}{name:>5s}{a.GetSymbol()+str(i+1):>5s}{i+1:5d}"
                  f"{pos.x/10:8.3f}{pos.y/10:8.3f}{pos.z/10:8.3f}")
    gl.append("   5.00000   5.00000   5.00000")
    gro_path.write_text("\n".join(gl) + "\n", encoding="utf-8")

    logger.info(f"  {name}: PolCA topology ({si_type}, "
                f"sigma={si_lj['sigma']}, eps={si_lj['eps']})")
    return {"itp": str(itp_path), "gro": str(gro_path),
            "si_type": si_type, "method": "PolCA"}


def parameterize_small_molecule(smiles: str, name: str,
                                 output_dir: Path,
                                 charge_method: str = "bcc") -> dict:
    """Generate GROMACS topology for a small molecule using acpype (GAFF2).

    Returns dict with 'itp' and 'gro' paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Generate 3D structure
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    # Check for boron → use B→C substitution trick (Gerogiokas 2020)
    has_boron = any(a.GetAtomicNum() == 5 for a in mol.GetAtoms())
    boron_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 5]

    if has_boron:
        logger.info(f"  {name}: contains B — using B→C substitution for GAFF2 parameterization")
        return _parameterize_boron_molecule(mol, name, output_dir, charge_method, boron_indices)

    # Check for Si → PolCA organosilane force field (GAFF2 has no Si type)
    has_si = any(a.GetAtomicNum() == 14 for a in mol.GetAtoms())
    if has_si:
        logger.info(f"  {name}: contains Si → PolCA organosilane parameters (Jorge 2021)")
        return _generate_silane_itp(name, smiles, output_dir)

    # Save as PDB
    pdb_path = output_dir / f"{name}.pdb"
    Chem.MolToPDBFile(mol, str(pdb_path))

    # Run acpype for GAFF2 parameterization (via Python API to use conda env).
    # env=_tool_env() ensures AmberTools (antechamber/sqm) is on PATH — without
    # it acpype fails for EVERY molecule when the launcher has a stripped PATH.
    acpype_cmd = [sys.executable, "-c",
                  "from acpype.cli import init_main; init_main()"]
    abs_pdb = str(pdb_path.resolve())  # Must be absolute path for acpype
    abs_dir = str(output_dir.resolve())
    env = _tool_env()
    result = None
    try:
        result = subprocess.run(
            acpype_cmd + ["-i", abs_pdb, "-b", name,
                          "-c", charge_method, "-a", "gaff2", "-o", "gmx"],
            cwd=abs_dir, env=env,
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0 or "DOES NOT EXIST" in result.stdout:
            logger.warning(f"  acpype {charge_method} failed for {name}, trying gasteiger...")
            result = subprocess.run(
                acpype_cmd + ["-i", abs_pdb, "-b", name,
                              "-c", "gas", "-a", "gaff2", "-o", "gmx"],
                cwd=abs_dir, env=env,
                capture_output=True, text=True, timeout=600,
            )
    except Exception as e:
        logger.error(f"  acpype failed: {e}")
        return {"error": str(e)}

    # Find output files
    acpype_out = output_dir / f"{name}.acpype"
    if not acpype_out.exists():
        # Try alternative naming
        for d in output_dir.glob(f"{name}*acpype*"):
            if d.is_dir():
                acpype_out = d
                break

    itp_files = list(acpype_out.glob("*_GMX.itp")) + list(acpype_out.glob("*.itp"))
    gro_files = list(acpype_out.glob("*_GMX.gro")) + list(acpype_out.glob("*.gro"))

    # Filter out posre files
    itp_files = [f for f in itp_files if "posre" not in f.name.lower()]

    if not itp_files:
        # Surface acpype's own diagnostics — a bare "No ITP files" hides whether
        # antechamber/sqm was missing, timed out, or the charge step failed.
        tail = ""
        if result is not None:
            tail = (result.stderr or result.stdout or "")[-400:]
        if tail and ("antechamber" in tail.lower() or "not found" in tail.lower()
                     or "command not found" in tail.lower()):
            tail = ("[toolchain] AmberTools (antechamber/sqm) unreachable — "
                    "check `conda install -c conda-forge ambertools`. ") + tail
        return {"error": f"No ITP files found in {acpype_out}. acpype said: {tail}"}

    return {
        "itp": str(itp_files[0]),
        "gro": str(gro_files[0]) if gro_files else None,
        "name": name,
    }


# ── System Building ──


def build_multi_monomer_system(template_smiles, template_name,
                                monomer_dict, n_per_monomer,
                                work_dir, box_size=5.0,
                                temperature=298.15,
                                crosslinker_smiles=None,
                                crosslinker_name=None,
                                n_crosslinker=0,
                                solvent=None,
                                seed=42):
    """Build a pre-polymerization system with multiple monomer types.

    Args:
        monomer_dict: {name: smiles} for each monomer type to include
        n_per_monomer: number of copies per monomer type

    Reference: Bio project Phase 4 multi-monomer system building.
    """
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    names = list(monomer_dict.keys())
    logger.info(f"  Multi-monomer system: {template_name} + "
                f"{' + '.join(f'{n_per_monomer}×{n}' for n in names)}")

    # Parameterize template
    tmpl_param = parameterize_small_molecule(
        template_smiles, template_name, work_dir / "param_template")
    if "error" in tmpl_param:
        return {"error": f"Template parameterization failed: {tmpl_param['error']}"}

    tmpl_itp = _copy_and_split_itp(tmpl_param["itp"], work_dir, template_name,
                                   resname="TMP")

    # Unique, collision-free residue name per monomer type (M00, M01, …). NEVER
    # derive it from the name: m_name[:3] can collide with TMP/XLK or a solvent
    # resname (TOL/SOL/…), which would silently drop that monomer from the
    # 'not resname <solvent> XLK' analysis selection or merge two monomer types.
    mono_resnames = {n: f"M{i:02d}" for i, n in enumerate(monomer_dict)}

    # Parameterize each monomer type
    mono_params = {}
    for m_name, m_smiles in monomer_dict.items():
        p = parameterize_small_molecule(m_smiles, m_name, work_dir / f"param_{m_name}")
        if "error" in p:
            logger.warning(f"  {m_name} parameterization failed, skipping")
            continue
        _copy_and_split_itp(p["itp"], work_dir, m_name,
                            resname=mono_resnames[m_name])
        mono_params[m_name] = p

    if not mono_params:
        return {"error": "All monomer parameterizations failed"}

    # Parameterize cross-linker if provided
    xl_param = None
    if crosslinker_smiles and n_crosslinker > 0:
        xl_param = parameterize_small_molecule(
            crosslinker_smiles, crosslinker_name, work_dir / "param_crosslinker")
        if "error" not in xl_param:
            _copy_and_split_itp(xl_param["itp"], work_dir, crosslinker_name,
                                resname="XLK")
        else:
            xl_param = None

    # Build initial GRO: template + all monomers + cross-linker.
    # `seed` varies the random placement so independent replicas start from
    # different configurations (Stage 5 averages EBN/contact over replicas).
    import numpy as np
    center = box_size / 2.0
    rng = np.random.RandomState(seed)

    tmpl_lines = Path(tmpl_param["gro"]).read_text().strip().split("\n")
    tmpl_natoms = int(tmpl_lines[1].strip())
    tmpl_atoms = tmpl_lines[2:2+tmpl_natoms]

    tmpl_offset = _gro_center_offset(tmpl_atoms, center, center, center)
    all_atoms = list(_shift_gro_atoms(tmpl_atoms, *tmpl_offset, resnum=1, resname="TMP"))
    resnum = 2

    # Place each monomer type
    for m_name, p in mono_params.items():
        mono_lines = Path(p["gro"]).read_text().strip().split("\n")
        mono_natoms = int(mono_lines[1].strip())
        mono_atoms = mono_lines[2:2+mono_natoms]

        for i in range(n_per_monomer):
            while True:
                x = rng.uniform(0.5, box_size - 0.5)
                y = rng.uniform(0.5, box_size - 0.5)
                z = rng.uniform(0.5, box_size - 0.5)
                if np.sqrt((x-center)**2+(y-center)**2+(z-center)**2) > 1.0:
                    break
            offset = _gro_center_offset(mono_atoms, x, y, z)
            shifted = _shift_gro_atoms(mono_atoms, *offset, resnum=resnum,
                                        resname=mono_resnames[m_name])
            all_atoms.extend(shifted)
            resnum += 1

    # Place cross-linker
    if xl_param and n_crosslinker > 0:
        xl_lines = Path(xl_param["gro"]).read_text().strip().split("\n")
        xl_natoms = int(xl_lines[1].strip())
        xl_atoms = xl_lines[2:2+xl_natoms]
        for i in range(n_crosslinker):
            while True:
                x = rng.uniform(0.5, box_size - 0.5)
                y = rng.uniform(0.5, box_size - 0.5)
                z = rng.uniform(0.5, box_size - 0.5)
                if np.sqrt((x-center)**2+(y-center)**2+(z-center)**2) > 0.8:
                    break
            offset = _gro_center_offset(xl_atoms, x, y, z)
            shifted = _shift_gro_atoms(xl_atoms, *offset, resnum=resnum, resname="XLK")
            all_atoms.extend(shifted)
            resnum += 1

    # Write system GRO
    with open(work_dir / "system.gro", "w") as f:
        f.write("MIP multi-monomer pre-polymerization\n")
        f.write(f" {len(all_atoms)}\n")
        for line in all_atoms:
            f.write(line + "\n")
        f.write(f"   {box_size:.5f}   {box_size:.5f}   {box_size:.5f}\n")

    # Resolve + build the porogen solvent box BEFORE writing the topology, so its
    # ITP (atomtypes + moleculetype) is split into work_dir and can be #included
    # in the correct order (all [atomtypes] must precede any [moleculetype]).
    solv = _setup_md_solvent(work_dir, solvent=solvent, base_dir=work_dir)
    logger.info(f"  MD solvent: {solv['name']}")

    # Write topology with all monomer types
    atomtype_includes = ""
    mol_includes = f'#include "{template_name}.itp"\n'
    mol_section = f"{template_name}    1\n"

    all_names = [template_name] + list(mono_params.keys())
    if xl_param:
        all_names.append(crosslinker_name)
    if solv["resname"]:
        all_names.append(solv["resname"])

    for name in all_names:
        at_file = work_dir / f"{name}_atomtypes.itp"
        if at_file.exists():
            atomtype_includes += f'#include "{name}_atomtypes.itp"\n'

    for m_name in mono_params:
        mol_includes += f'#include "{m_name}.itp"\n'
        mol_section += f"{m_name}    {n_per_monomer}\n"
    if xl_param:
        mol_includes += f'#include "{crosslinker_name}.itp"\n'
        mol_section += f"{crosslinker_name}    {n_crosslinker}\n"
    if solv["resname"]:
        # moleculetype only — gmx solvate appends the porogen COUNT to [molecules]
        mol_includes += f'#include "{solv["resname"]}.itp"\n'

    top = f"""; MIP multi-monomer pre-polymerization topology
#include "amber99sb-ildn.ff/forcefield.itp"
{atomtype_includes}
{mol_includes}
#include "amber99sb-ildn.ff/tip3p.itp"
#include "amber99sb-ildn.ff/ions.itp"

[ system ]
MIP multi-monomer pre-polymerization

[ molecules ]
{mol_section}"""
    (work_dir / "topol.top").write_text(top)
    (work_dir / "em.mdp").write_text(MDP_EM)

    # Solvate (chosen porogen or water) + neutralize
    _solvate_and_ions(work_dir, solv)

    return {
        "gro": str(work_dir / "ions.gro"),
        "top": str(work_dir / "topol.top"),
        "monomer_names": list(mono_params.keys()),
        "n_per_monomer": n_per_monomer,
        # resname (M00/M01/…) → monomer name, so analyze_md's per-resname contact
        # can be attributed to each monomer for the mixture-based synthesis ratio.
        "resname_map": {mono_resnames[n]: n for n in mono_params},
    }

def build_mip_system(template_smiles: str, template_name: str,
                      monomer_smiles: str, monomer_name: str,
                      n_monomers: int, work_dir: Path,
                      box_size: float = 4.0,
                      temperature: float = 298.15,
                      crosslinker_smiles: str = None,
                      crosslinker_name: str = None,
                      n_crosslinker: int = 0,
                      solvent: str = None) -> dict:
    """Build a pre-polymerization system: template + N monomers + [cross-linker] + solvent.

    1. Parameterize template, monomer, [cross-linker] (acpype/GAFF2)
    2. Create box with template at center, monomers + cross-linker randomly placed
    3. Solvate with the chosen porogen (Stage-3 global porogen) or TIP3P water
    4. Add ions to neutralize

    Cross-linker inclusion: Ye et al. 2024 (Molecules)
    """
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    xl_str = f" + {n_crosslinker}× {crosslinker_name}" if crosslinker_smiles and n_crosslinker > 0 else ""
    logger.info(f"  Building system: {template_name} + {n_monomers}× {monomer_name}{xl_str}")

    # Parameterize
    tmpl_param = parameterize_small_molecule(
        template_smiles, template_name, work_dir / "param_template")
    mono_param = parameterize_small_molecule(
        monomer_smiles, monomer_name, work_dir / "param_monomer")

    if "error" in tmpl_param or "error" in mono_param:
        return {"error": f"Parameterization failed: {tmpl_param.get('error', '')} {mono_param.get('error', '')}"}

    # Parameterize cross-linker if provided
    xl_param = None
    if crosslinker_smiles and n_crosslinker > 0:
        xl_param = parameterize_small_molecule(
            crosslinker_smiles, crosslinker_name, work_dir / "param_crosslinker")
        if "error" in xl_param:
            logger.warning(f"  Cross-linker parameterization failed: {xl_param['error']}")
            xl_param = None

    # Copy ITP files to work_dir, splitting [ atomtypes ] into separate file,
    # relabelling acpype's 'UNL' residue to TMP/MON/XLK so resname-based
    # analysis selections (template/monomer/crosslinker separation) work.
    tmpl_itp = _copy_and_split_itp(tmpl_param["itp"], work_dir, template_name,
                                   resname="TMP")
    mono_itp = _copy_and_split_itp(mono_param["itp"], work_dir, monomer_name,
                                   resname="MON")
    if xl_param:
        _copy_and_split_itp(xl_param["itp"], work_dir, crosslinker_name,
                            resname="XLK")

    # Build combined GRO: template + monomers + cross-linker (random placement)
    _build_initial_gro(tmpl_param, mono_param, n_monomers, work_dir, box_size,
                       xl_param=xl_param, n_crosslinker=n_crosslinker)

    # Resolve + build the porogen box before writing topology (atomtypes ordering)
    solv = _setup_md_solvent(work_dir, solvent=solvent, base_dir=work_dir)
    logger.info(f"  MD solvent: {solv['name']}")

    # Write topology
    _write_topology(template_name, monomer_name, n_monomers, work_dir,
                    crosslinker_name=crosslinker_name if xl_param else None,
                    n_crosslinker=n_crosslinker if xl_param else 0,
                    solvent_resname=solv["resname"])

    # Solvate (chosen porogen or water) + neutralize
    _solvate_and_ions(work_dir, solv)

    return {
        "gro": str(work_dir / "ions.gro"),
        "top": str(work_dir / "topol.top"),
        "template_itp": str(tmpl_itp),
        "monomer_itp": str(mono_itp),
        "n_template_atoms": _count_atoms_in_gro(tmpl_param.get("gro", "")),
    }


def _read_global_porogen(base_dir):
    """Walk up from base_dir to find Stage 3's global_porogen.json → porogen name."""
    if not base_dir:
        return None
    p = Path(base_dir).resolve()
    for anc in [p, *p.parents]:
        gp = anc / "stage3" / "global_porogen.json"
        if gp.exists():
            try:
                return json.loads(gp.read_text()).get("porogen")
            except Exception:
                return None
    return None


def _resolve_md_solvent(solvent=None, base_dir=None):
    """Resolve the porogen NAME to solvate the MD box with.
    Priority: explicit `solvent` arg → config MD_SOLVENT → 'Water'.
    'auto' reads the Stage-3 global porogen (walking up from base_dir)."""
    from .config import MD_SOLVENT
    name = solvent if solvent is not None else MD_SOLVENT
    if isinstance(name, str) and name.lower() == "auto":
        name = _read_global_porogen(base_dir) or "Water"
    if not name:
        name = "Water"
    alias = {"water": "Water", "h2o": "Water", "tip3p": "Water", "spc": "Water",
             "acetonitrile": "Acetonitrile", "acn": "Acetonitrile",
             "chloroform": "Chloroform", "toluene": "Toluene",
             "methanol": "Methanol", "meoh": "Methanol",
             "dmf": "DMF", "dmso": "DMSO"}
    return alias.get(str(name).lower(), name)


def _prepare_solvent_box(work_dir, solvent_name):
    """Parameterize the porogen, split its ITP into work_dir (so the main
    topology can #include it), and pack a density-correct cubic box for gmx
    solvate. Returns (box_gro, resname, atoms_per_mol). (None, None, 0) signals
    'use the built-in spc216 water box'. Never raises."""
    from .config import SOLVENT_PROPERTIES
    props = SOLVENT_PROPERTIES.get(solvent_name)
    if not props or props.get("resname") == "SOL":
        return None, None, 0                          # water → spc216
    resn = props["resname"]
    work_dir = Path(work_dir)
    sdir = work_dir / f"solvent_{resn}"
    sdir.mkdir(parents=True, exist_ok=True)
    box_gro = sdir / "solvent_box.gro"
    try:
        param = parameterize_small_molecule(props["smiles"], resn, sdir / "param")
        if "error" in param:
            logger.warning(f"  Porogen {solvent_name} parameterization failed; "
                           f"MD will fall back to TIP3P water")
            return None, None, 0
        # Make the porogen ITP (moleculetype + atomtypes) available to topol.top,
        # relabelling acpype's 'UNL' residue to `resn`. CRITICAL: the moleculetype
        # is named `resn`, and gmx solvate appends the box's RESIDUE name to
        # [molecules] — so the box residue MUST also be `resn`, else grompp gets
        # an orphan 'UNL <n>' line referencing an undefined moleculetype and the
        # whole MD build fails (silently, since gmx() only warns).
        _copy_and_split_itp(param["itp"], work_dir, resn, resname=resn)
        apm = _count_atoms_in_gro(param.get("gro", "")) or 0
        if not box_gro.exists():
            edge = 3.0                                # nm
            v_cm3 = (edge * 1e-7) ** 3
            n = int(props["density"] * v_cm3 * 6.02214076e23 / props["mw"])
            n = max(30, n)
            gmx(["insert-molecules", "-ci", param["gro"], "-nmol", str(n),
                 "-box", str(edge), str(edge), str(edge),
                 "-o", "solvent_box.gro"], sdir)
            if box_gro.exists():
                _relabel_gro_residue(box_gro, resn)   # UNL → resn (match moltype)
        if box_gro.exists():
            logger.info(f"  Built {solvent_name} solvent box "
                        f"(resname {resn}, {apm} atoms/mol) for MD")
            return box_gro, resn, apm
    except Exception as e:
        logger.warning(f"  Porogen {solvent_name} box build failed ({e}); "
                       f"MD will fall back to TIP3P water")
    return None, None, 0


def _setup_md_solvent(work_dir, solvent=None, base_dir=None):
    """Resolve porogen + build its box. Returns
    {name, resname, box, atoms_per_mol}; resname None → spc216 water."""
    name = _resolve_md_solvent(solvent, base_dir if base_dir is not None else work_dir)
    box, resn, apm = _prepare_solvent_box(work_dir, name)
    if resn is None and name != "Water":
        logger.info(f"  Solvent '{name}' unavailable → TIP3P water")
        name = "Water"
    return {"name": name, "resname": resn, "box": box, "atoms_per_mol": apm}


def _ensure_solvent_count(work_dir, resname, atoms_per_mol):
    """Guard: gmx solvate -p should append the porogen molecule count to
    [ molecules ]; if it did not, count residues from solvated.gro and add it."""
    import re
    topol = Path(work_dir) / "topol.top"
    text = topol.read_text()
    if re.search(rf"(?m)^\s*{re.escape(resname)}\s+\d+\s*$", text):
        return
    gro = Path(work_dir) / "solvated.gro"
    if not gro.exists() or atoms_per_mol <= 0:
        return
    lines = gro.read_text().split("\n")
    try:
        natoms = int(lines[1].strip())
    except (IndexError, ValueError):
        return
    n_atoms = sum(1 for ln in lines[2:2 + natoms] if ln[5:10].strip() == resname)
    n_mol = n_atoms // atoms_per_mol
    if n_mol > 0:
        topol.write_text(text.rstrip() + f"\n{resname}    {n_mol}\n")
        logger.info(f"  [guard] Added {n_mol}× {resname} to [ molecules ]")


def _make_solvent_index(work_dir, resname):
    """Write genion.ndx with a 'SOL' group = the porogen residues, so genion can
    embed neutralizing ions in a NON-water solvent (which has no built-in 'SOL'
    group). Returns the index path, or None on failure."""
    try:
        import MDAnalysis as mda
        from MDAnalysis.selections.gromacs import SelectionWriter
        gro = Path(work_dir) / "solvated.gro"
        u = mda.Universe(str(gro))
        sel = u.select_atoms(f"resname {resname}")
        if len(sel) == 0:
            return None
        ndx = Path(work_dir) / "genion.ndx"
        with SelectionWriter(str(ndx), mode="w") as w:
            w.write(u.atoms, name="System")
            w.write(sel, name="SOL")
        return ndx
    except Exception as e:
        logger.warning(f"  genion index build failed ({e})")
        return None


def _solvate_and_ions(work_dir, solv):
    """gmx solvate (porogen box or spc216 water) + genion neutralize. For a
    non-water porogen, genion needs an explicit 'SOL' index group; if the system
    is already neutral (the usual case for acpype-built neutral molecules),
    genion adds nothing and we fall back to the un-neutralized solvated box."""
    import shutil as _shutil
    work_dir = Path(work_dir)
    porogen = bool(solv.get("box") and solv.get("resname"))
    if porogen:
        gmx(["solvate", "-cp", "system.gro", "-cs", str(solv["box"]),
             "-o", "solvated.gro", "-p", "topol.top"], work_dir)
        _ensure_solvent_count(work_dir, solv["resname"], solv.get("atoms_per_mol", 0))
    else:
        gmx(["solvate", "-cp", "system.gro", "-cs", "spc216.gro",
             "-o", "solvated.gro", "-p", "topol.top"], work_dir)

    gmx(["grompp", "-f", "em.mdp", "-c", "solvated.gro",
         "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "50"], work_dir)

    cmd = ["genion", "-s", "ions.tpr", "-o", "ions.gro",
           "-p", "topol.top", "-pname", "NA", "-nname", "CL", "-neutral"]
    if porogen:
        ndx = _make_solvent_index(work_dir, solv["resname"])
        if ndx:
            cmd += ["-n", str(ndx)]
    gmx(cmd, work_dir, input_text="SOL\n")

    # Robustness: if genion produced no output (neutral system, or no SOL group
    # in a porogen box), use the solvated box directly so the build never dies.
    if not (work_dir / "ions.gro").exists():
        logger.info("  genion added no ions (system neutral); using solvated box")
        _shutil.copy2(str(work_dir / "solvated.gro"), str(work_dir / "ions.gro"))


def _build_initial_gro(tmpl_param, mono_param, n_monomers, work_dir, box_size,
                       xl_param=None, n_crosslinker=0):
    """Create initial GRO with template at center, monomers + cross-linker randomly placed."""
    import numpy as np

    # Read template GRO
    tmpl_lines = Path(tmpl_param["gro"]).read_text().strip().split("\n")
    tmpl_natoms = int(tmpl_lines[1].strip())
    tmpl_atoms = tmpl_lines[2:2+tmpl_natoms]

    # Read monomer GRO
    mono_lines = Path(mono_param["gro"]).read_text().strip().split("\n")
    mono_natoms = int(mono_lines[1].strip())
    mono_atoms = mono_lines[2:2+mono_natoms]

    # Read cross-linker GRO if provided
    xl_atoms = None
    if xl_param and n_crosslinker > 0:
        xl_lines = Path(xl_param["gro"]).read_text().strip().split("\n")
        xl_natoms = int(xl_lines[1].strip())
        xl_atoms = xl_lines[2:2+xl_natoms]

    # Place template at center
    center = box_size / 2.0
    tmpl_offset = _gro_center_offset(tmpl_atoms, center, center, center)
    shifted_tmpl = _shift_gro_atoms(tmpl_atoms, *tmpl_offset, resnum=1, resname="TMP")

    # Place monomers randomly around template
    all_atoms = list(shifted_tmpl)
    rng = np.random.RandomState(42)
    resnum = 2

    for i in range(n_monomers):
        while True:
            x = rng.uniform(0.5, box_size - 0.5)
            y = rng.uniform(0.5, box_size - 0.5)
            z = rng.uniform(0.5, box_size - 0.5)
            dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
            if dist > 1.0:
                break
        offset = _gro_center_offset(mono_atoms, x, y, z)
        shifted_mono = _shift_gro_atoms(mono_atoms, *offset,
                                         resnum=resnum, resname="MON")
        all_atoms.extend(shifted_mono)
        resnum += 1

    # Place cross-linker molecules (Ye 2024)
    if xl_atoms and n_crosslinker > 0:
        for i in range(n_crosslinker):
            while True:
                x = rng.uniform(0.5, box_size - 0.5)
                y = rng.uniform(0.5, box_size - 0.5)
                z = rng.uniform(0.5, box_size - 0.5)
                dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                if dist > 0.8:
                    break
            offset = _gro_center_offset(xl_atoms, x, y, z)
            shifted_xl = _shift_gro_atoms(xl_atoms, *offset,
                                           resnum=resnum, resname="XLK")
            all_atoms.extend(shifted_xl)
            resnum += 1

    # Write system GRO
    total = len(all_atoms)
    with open(work_dir / "system.gro", "w") as f:
        f.write("MIP pre-polymerization system\n")
        f.write(f" {total}\n")
        for line in all_atoms:
            f.write(line + "\n")
        f.write(f"   {box_size:.5f}   {box_size:.5f}   {box_size:.5f}\n")

    # Write EM MDP
    (work_dir / "em.mdp").write_text(MDP_EM)


def _write_topology(template_name, monomer_name, n_monomers, work_dir,
                     crosslinker_name=None, n_crosslinker=0,
                     solvent_resname=None):
    """Write GROMACS topology file.

    atomtypes must come before moleculetype, so they are included
    right after the forcefield.itp. When `solvent_resname` is a porogen (not
    water), its atomtypes + moleculetype are included too; gmx solvate appends
    the porogen molecule COUNT to [ molecules ].
    """
    work_dir = Path(work_dir)

    # Check which atomtypes files exist
    mol_names = [template_name, monomer_name]
    if crosslinker_name:
        mol_names.append(crosslinker_name)
    if solvent_resname:
        mol_names.append(solvent_resname)

    atomtype_includes = ""
    for name in mol_names:
        at_file = work_dir / f"{name}_atomtypes.itp"
        if at_file.exists():
            atomtype_includes += f'#include "{name}_atomtypes.itp"\n'

    mol_includes = f'#include "{template_name}.itp"\n#include "{monomer_name}.itp"\n'
    if crosslinker_name:
        mol_includes += f'#include "{crosslinker_name}.itp"\n'
    if solvent_resname:
        mol_includes += f'#include "{solvent_resname}.itp"\n'

    mol_section = f"{template_name}    1\n{monomer_name}    {n_monomers}\n"
    if crosslinker_name and n_crosslinker > 0:
        mol_section += f"{crosslinker_name}    {n_crosslinker}\n"

    top = f"""; MIP pre-polymerization topology
#include "amber99sb-ildn.ff/forcefield.itp"

; Atom types from acpype (must come before moleculetype)
{atomtype_includes}
; Molecule definitions
{mol_includes}
; Solvent
#include "amber99sb-ildn.ff/tip3p.itp"
#include "amber99sb-ildn.ff/ions.itp"

[ system ]
MIP pre-polymerization

[ molecules ]
{mol_section}"""
    (work_dir / "topol.top").write_text(top)


# ── MD Pipeline ──

def run_md_pipeline(work_dir: Path, time_ns: float = 50.0,
                     temperature: float = 298.15,
                     gpu_id: str = "0") -> dict:
    """Run full MD pipeline: EM → NVT → NPT → Production.

    Features (matching Bio pipeline):
    - Checkpoint resume: if md.cpt exists, resumes from last checkpoint
    - GPU optimization: -nb gpu -pme gpu -bonded gpu -update gpu
    - Skip completed stages (em.gro exists → skip EM, etc.)
    - Real-time progress via -v flag on production
    - Reduced trajectory for analysis (stride 100)
    """
    work_dir = Path(work_dir).resolve()
    dt = 0.002  # ps (2 fs)

    # ── Energy minimization ──
    if not (work_dir / "em.gro").exists():
        logger.info("  EM...")
        gmx(["grompp", "-f", "em.mdp", "-c", "ions.gro",
             "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "50"], work_dir)
        gmx(["mdrun", "-deffnm", "em", "-nb", "gpu"], work_dir, timeout=300)
    else:
        logger.info("  EM: FOUND (skipping)")

    if not (work_dir / "em.gro").exists():
        return {"error": "EM failed — em.gro not created"}

    # ── NVT equilibration (100ps) ──
    if not (work_dir / "nvt.gro").exists():
        logger.info("  NVT (100ps)...")
        nvt_mdp = MDP_NVT.format(nsteps=50000, dt=dt, temperature=temperature)
        (work_dir / "nvt.mdp").write_text(nvt_mdp)
        gmx(["grompp", "-f", "nvt.mdp", "-c", "em.gro", "-r", "em.gro",
             "-p", "topol.top", "-o", "nvt.tpr", "-maxwarn", "50"], work_dir)
        gmx(["mdrun", "-deffnm", "nvt", "-nb", "gpu"], work_dir, timeout=600)
    else:
        logger.info("  NVT: FOUND (skipping)")

    # ── NPT equilibration (100ps) ──
    if not (work_dir / "npt.gro").exists():
        logger.info("  NPT (100ps)...")
        npt_mdp = MDP_NPT.format(nsteps=50000, dt=dt, temperature=temperature)
        (work_dir / "npt.mdp").write_text(npt_mdp)
        gmx(["grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro",
             "-t", "nvt.cpt", "-p", "topol.top", "-o", "npt.tpr", "-maxwarn", "50"], work_dir)
        gmx(["mdrun", "-deffnm", "npt", "-nb", "gpu"], work_dir, timeout=600)
    else:
        logger.info("  NPT: FOUND (skipping)")

    # ── Production MD ──
    nsteps = int(time_ns * 1e6 / (dt * 1000))
    nstxout = max(5000, nsteps // 1000)

    # Generate tpr if not exists
    if not (work_dir / "md.tpr").exists():
        prod_mdp = MDP_PRODUCTION.format(
            nsteps=nsteps, dt=dt, nstxout=nstxout, temperature=temperature)
        (work_dir / "md.mdp").write_text(prod_mdp)
        gmx(["grompp", "-f", "md.mdp", "-c", "npt.gro",
             "-t", "npt.cpt", "-p", "topol.top", "-o", "md.tpr",
             "-maxwarn", "50"], work_dir)

    # Build mdrun command with GPU optimization
    md_cmd = ["mdrun", "-deffnm", "md", "-v"]

    # Resume from checkpoint if available (interrupted run)
    md_cpt = work_dir / "md.cpt"
    if md_cpt.exists():
        md_cmd.extend(["-cpi", "md.cpt", "-append"])
        logger.info(f"  Production: RESUMING from checkpoint ({time_ns}ns)")
    else:
        logger.info(f"  Production ({time_ns}ns, {nsteps} steps)...")

    # Check if already completed
    md_xtc = work_dir / "md.xtc"
    md_log = work_dir / "md.log"
    if md_xtc.exists() and md_log.exists():
        # Check if finished by looking for "Finished" in log
        try:
            log_tail = md_log.read_text()[-500:]
            if "Finished" in log_tail or "Performance" in log_tail:
                logger.info("  Production: COMPLETED (skipping)")
                return {
                    "traj": str(md_xtc),
                    "top": str(work_dir / "npt.gro"),
                    "tpr": str(work_dir / "md.tpr"),
                }
        except Exception:
            pass

    # GPU flags (all-GPU offloading)
    md_cmd.extend([
        "-nb", "gpu",
        "-pme", "gpu",
        "-bonded", "gpu",
        "-update", "gpu",
        "-gpu_id", gpu_id,
    ])

    # Run production with real-time output (-v) via subprocess
    full_cmd = [GMX_BIN] + md_cmd
    logger.info(f"  CMD: {' '.join(full_cmd)}")
    subprocess.run(full_cmd, cwd=str(work_dir),
                   timeout=int(time_ns * 3600))  # timeout = time_ns hours

    # ── Create reduced trajectory for analysis ──
    reduced = work_dir / "md_reduced.xtc"
    if not reduced.exists() and md_xtc.exists():
        logger.info("  Creating reduced trajectory (stride 100)...")
        gmx(["trjconv", "-f", "md.xtc", "-s", "md.tpr",
             "-o", "md_reduced.xtc", "-skip", "100"],
            work_dir, input_text="0\n", timeout=300)

    return {
        "traj": str(md_xtc),
        "top": str(work_dir / "npt.gro"),
        "tpr": str(work_dir / "md.tpr"),
        "reduced_traj": str(reduced) if reduced.exists() else None,
    }


# ── Boron Parameterization (Gerogiokas 2020) ──

# Literature B parameters for boronic acid (Gerogiokas et al., Molecules 2020)
# DOI: 10.3390/molecules25092196
BORON_VDW = {"sigma": 0.398, "epsilon": 0.1422}  # nm, kJ/mol (UFF-derived)
BORON_BONDS = {
    "B-O": {"r0": 0.1365, "k": 217568},   # nm, kJ/mol/nm² (520 kcal/mol/Å²)
    "B-C": {"r0": 0.1566, "k": 167360},   # nm, kJ/mol/nm² (400 kcal/mol/Å²)
    "B-N": {"r0": 0.1540, "k": 175728},   # nm, kJ/mol/nm²
}
BORON_ANGLES = {
    "C-B-O": {"theta0": 117.8, "k": 292.88},   # deg, kJ/mol/rad²
    "O-B-O": {"theta0": 112.5, "k": 334.72},   # deg, kJ/mol/rad²
    "C-B-N": {"theta0": 120.0, "k": 292.88},
    "N-B-O": {"theta0": 115.0, "k": 292.88},
}


def _parameterize_boron_molecule(mol, name, output_dir, charge_method, boron_indices):
    """Parameterize boron-containing molecule using B→C substitution trick.

    Method (Gerogiokas et al., Molecules 2020):
    1. Replace B with C in SMILES/PDB → antechamber can process
    2. Run acpype on substituted molecule → get GAFF2 parameters
    3. Replace C parameters back with literature B parameters
    4. Fix atom types and vdW for B atoms

    Returns dict with 'itp' and 'gro' paths.
    """
    import sys
    from rdkit import Chem
    from rdkit.Chem import AllChem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create B→C substituted molecule for acpype
    rw_mol = Chem.RWMol(mol)
    for idx in boron_indices:
        rw_mol.GetAtomWithIdx(idx).SetAtomicNum(6)  # B(5) → C(6)
    sub_mol = rw_mol.GetMol()

    # Re-embed (C has different geometry than B)
    AllChem.EmbedMolecule(Chem.AddHs(sub_mol), AllChem.ETKDGv3())

    sub_pdb = output_dir / f"{name}_sub.pdb"
    Chem.MolToPDBFile(sub_mol, str(sub_pdb))

    # Also save original PDB with B for GRO coordinate reference
    orig_pdb = output_dir / f"{name}.pdb"
    Chem.MolToPDBFile(mol, str(orig_pdb))

    # Step 2: Run acpype on substituted molecule
    acpype_cmd = [sys.executable, "-c",
                  "from acpype.cli import init_main; init_main()"]
    abs_pdb = str(sub_pdb.resolve())
    abs_dir = str(output_dir.resolve())

    try:
        result = subprocess.run(
            acpype_cmd + ["-i", abs_pdb, "-b", name,
                          "-c", "gas", "-a", "gaff2", "-o", "gmx"],
            cwd=abs_dir, env=_tool_env(),
            capture_output=True, text=True, timeout=600)
    except Exception as e:
        return {"error": f"acpype failed for B→C substitution: {e}"}

    acpype_out = output_dir / f"{name}.acpype"
    if not acpype_out.exists():
        for d in output_dir.glob(f"{name}*acpype*"):
            if d.is_dir():
                acpype_out = d; break

    itp_files = [f for f in acpype_out.glob("*_GMX.itp") if "posre" not in f.name.lower()]
    gro_files = list(acpype_out.glob("*_GMX.gro"))

    if not itp_files:
        return {"error": f"acpype B→C substitution failed for {name}"}

    # Step 3: Fix ITP — replace C atom types with B parameters
    itp_path = itp_files[0]
    _fix_boron_itp(itp_path, mol, boron_indices)

    # Step 4: Fix GRO — replace C element with B
    if gro_files:
        _fix_boron_gro(gro_files[0], mol, boron_indices)

    logger.info(f"  {name}: B parameterization complete (B→C substitution + literature params)")

    return {
        "itp": str(itp_files[0]),
        "gro": str(gro_files[0]) if gro_files else None,
        "name": name,
    }


def _fix_boron_itp(itp_path, mol, boron_indices):
    """Fix the B->C-substituted ITP: restore the boron atomtype (LJ), boron MASS,
    and the literature B-O / B-C / B-N bond and X-B-X angle parameters. Previously
    only the vdW atomtype was patched — BORON_BONDS/BORON_ANGLES and the boron
    mass were never applied, so the boronic group kept carbon geometry and mass."""
    content = Path(itp_path).read_text()
    lines = content.split("\n")
    new_lines = []
    in_atomtypes = in_atoms = in_bonds = in_angles = False
    b_set_1based = {i + 1 for i in boron_indices}

    def _elem(one_based):
        try:
            return mol.GetAtomWithIdx(one_based - 1).GetSymbol()
        except Exception:
            return "?"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[ atomtypes ]"):
            in_atomtypes, in_atoms, in_bonds, in_angles = True, False, False, False
        elif stripped.startswith("[ atoms ]"):
            in_atomtypes, in_atoms, in_bonds, in_angles = False, True, False, False
        elif stripped.startswith("[ bonds ]"):
            in_atomtypes, in_atoms, in_bonds, in_angles = False, False, True, False
        elif stripped.startswith("[ angles ]"):
            in_atomtypes, in_atoms, in_bonds, in_angles = False, False, False, True
        elif stripped.startswith("["):
            in_atomtypes = in_atoms = in_bonds = in_angles = False

        # Add the boron atomtype once (LJ + correct mass 10.811)
        if in_atomtypes and stripped and not stripped.startswith((";", "[")):
            if "b_boron" not in content:
                new_lines.append(line)
                new_lines.append(
                    f" b_boron  b_boron  10.81100  0.00000   A "
                    f"  {BORON_VDW['sigma']:.5e}   {BORON_VDW['epsilon']:.5e} ; Boron (UFF)")
                content += "b_boron"
                continue

        # [atoms]: set boron atomtype + boron mass (col 8), keep proxy charge
        elif in_atoms and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            if len(parts) >= 8:
                try:
                    if int(parts[0]) in b_set_1based:
                        parts[1] = "b_boron"     # atomtype
                        parts[7] = "10.811"       # mass (was carbon 12.011)
                        line = "  ".join(parts)
                except (ValueError, IndexError):
                    pass

        # [bonds]: rewrite r0/k for any B-O / B-C / B-N bond (literature)
        elif in_bonds and stripped and not stripped.startswith((";", "[")):
            parts = stripped.split()
            if len(parts) >= 5:
                try:
                    i, j = int(parts[0]), int(parts[1])
                    if i in b_set_1based or j in b_set_1based:
                        other = j if i in b_set_1based else i
                        key = {"O": "B-O", "C": "B-C", "N": "B-N"}.get(_elem(other))
                        if key in BORON_BONDS:
                            bp = BORON_BONDS[key]
                            line = (f"  {i:5d}  {j:5d}    1    "
                                    f"{bp['r0']:.4f}  {bp['k']:.1f} ; {key}")
                except (ValueError, IndexError):
                    pass

        # [angles]: rewrite theta0/k for X-B-X angles where boron is central
        elif in_angles and stripped and not stripped.startswith((";", "[")):
            parts = stripped.split()
            if len(parts) >= 6:
                try:
                    ai, aj, ak = int(parts[0]), int(parts[1]), int(parts[2])
                    if aj in b_set_1based:
                        outers = tuple(sorted([_elem(ai), _elem(ak)]))
                        akey = {("C", "O"): "C-B-O", ("O", "O"): "O-B-O",
                                ("C", "N"): "C-B-N", ("N", "O"): "N-B-O"}.get(outers)
                        if akey in BORON_ANGLES:
                            ap = BORON_ANGLES[akey]
                            line = (f"  {ai:5d}  {aj:5d}  {ak:5d}    1    "
                                    f"{ap['theta0']:.2f}  {ap['k']:.2f} ; {akey}")
                except (ValueError, IndexError):
                    pass

        new_lines.append(line)

    Path(itp_path).write_text("\n".join(new_lines))


def _fix_boron_gro(gro_path, mol, boron_indices):
    """Fix GRO: update element names for B atoms."""
    lines = Path(gro_path).read_text().strip().split("\n")
    b_set = set(boron_indices)

    for i in range(2, len(lines) - 1):  # skip header, natoms, box
        atom_idx = i - 2  # 0-based
        if atom_idx in b_set:
            # Replace atom name field (columns 10-15) with " B  "
            line = lines[i]
            if len(line) >= 44:
                lines[i] = line[:10] + "   B " + line[15:]

    Path(gro_path).write_text("\n".join(lines) + "\n")


# ── ITP Processing ──

def _relabel_itp_residue(lines, resname):
    """Rewrite the [ atoms ] residue column (field 4) of an ITP to `resname`.
    acpype labels EVERY molecule's residue 'UNL', so template/monomer/crosslinker/
    porogen are indistinguishable by resname in the .tpr and trajectory — which
    silently breaks every `resname`-based analysis selection (crosslinker
    exclusion, solvent exclusion, VIP). Grompp matches [molecules] by moleculetype
    NAME (untouched here), so relabelling the residue is safe."""
    out, in_atoms = [], False
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            in_atoms = s.startswith("[ atoms ]")
            out.append(line)
            continue
        if in_atoms and s and not s.startswith(";"):
            code, sep, comment = line.partition(";")
            parts = code.split()
            if len(parts) >= 8:
                parts[3] = resname                      # residue column
                out.append("  ".join(parts) + (f" ;{comment}" if sep else ""))
                continue
        out.append(line)
    return out


def _relabel_gro_residue(gro_path, resname):
    """Rewrite the residue-name field (cols 5:10) of every atom in a .gro."""
    p = Path(gro_path)
    lines = p.read_text().split("\n")
    if len(lines) < 3:
        return
    try:
        n = int(lines[1].strip())
    except (ValueError, IndexError):
        return
    out = lines[:2]
    for ln in lines[2:2 + n]:
        out.append((ln[:5] + f"{resname:<5s}"[:5] + ln[10:]) if len(ln) >= 10 else ln)
    out += lines[2 + n:]
    p.write_text("\n".join(out))


def _copy_and_split_itp(src_itp: str, work_dir: Path, name: str,
                        resname: str = None) -> str:
    """Copy ITP file, splitting [ atomtypes ] into a separate _atomtypes.itp.

    GROMACS requires [ atomtypes ] before [ moleculetype ] in the topology.
    acpype puts both in one ITP, causing 'Invalid order for directive' error.
    Solution: split atomtypes into {name}_atomtypes.itp, included before the main ITP.

    If `resname` is given, the [ atoms ] residue column is relabelled from
    acpype's 'UNL' to it, so downstream `resname`-based selections work.
    """
    src = Path(src_itp)
    content = src.read_text()
    lines = content.split("\n")

    atomtypes_lines = []
    molecule_lines = []
    in_atomtypes = False
    past_atomtypes = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[ atomtypes ]"):
            in_atomtypes = True
            atomtypes_lines.append(line)
            continue
        if in_atomtypes:
            if stripped.startswith("[") and "atomtypes" not in stripped:
                in_atomtypes = False
                past_atomtypes = True
                molecule_lines.append(line)
            else:
                atomtypes_lines.append(line)
        else:
            molecule_lines.append(line)

    # Write atomtypes file
    if atomtypes_lines:
        at_path = work_dir / f"{name}_atomtypes.itp"
        at_path.write_text("\n".join(atomtypes_lines) + "\n")

    # Relabel residue (acpype 'UNL' → intended resname) for analysis selections
    if resname:
        molecule_lines = _relabel_itp_residue(molecule_lines, resname)

    # Write molecule ITP (without atomtypes)
    itp_path = work_dir / f"{name}.itp"
    itp_path.write_text("\n".join(molecule_lines) + "\n")

    return str(itp_path)


# ── Analysis ──

def _solvent_resnames():
    """All resnames that count as SOLVENT/ions in a trajectory — water, ions, and
    every porogen in SOLVENT_PROPERTIES. Used to build 'not resname ...' monomer
    selections so the porogen never leaks into monomer/template statistics."""
    base = ["SOL", "WAT", "NA", "CL", "Na+", "Cl-", "K", "K+"]
    try:
        from .config import SOLVENT_PROPERTIES
        base += [v["resname"] for v in SOLVENT_PROPERTIES.values()]
    except Exception:
        base += ["ACN", "CHL", "TOL", "MOH", "DMF", "DSO"]
    seen, out = set(), []
    for r in base:
        if r not in seen:
            seen.add(r); out.append(r)
    return " ".join(out)


def run_mmpbsa(work_dir: Path, interval: int = 10, igb: int = 5,
               start_frac: float = 0.5, timeout: int = 3600) -> dict:
    """MM/GBSA binding free energy of the template to the monomer/cross-linker
    assembly from a Stage 5 MD trajectory (gmx_MMPBSA). Intermediate between the
    single-pose enthalpy and a full-mixture free energy (audit GAP #1): real ΔG
    (MM energy + GB solvation) averaged over the equilibrium ensemble.

    Needs md.tpr / md.xtc / topol.top in work_dir and gmx_MMPBSA + mpi4py.
    receptor = monomers+crosslinker, ligand = template (first non-solvent resid).
    Returns {dG_bind_kcal, ...} or {error}. Never raises.
    """
    import shutil
    import re
    work_dir = Path(work_dir).resolve()
    tpr, xtc, top = work_dir / "md.tpr", work_dir / "md.xtc", work_dir / "topol.top"
    if not (tpr.exists() and xtc.exists() and top.exists()):
        return {"error": "missing md.tpr/md.xtc/topol.top"}
    if shutil.which("gmx_MMPBSA") is None:
        return {"error": "gmx_MMPBSA not installed"}
    try:
        import MDAnalysis as mda
        from MDAnalysis.selections.gromacs import SelectionWriter
        u = load_universe(Path(tpr).parent, xtc)
        non_sol = u.select_atoms(f"not resname {_solvent_resnames()}")
        if len(non_sol.residues) < 2:
            return {"error": "need template + >=1 monomer for MM/PBSA"}
        frid = non_sol.residues[0].resid
        lig = u.select_atoms(f"resid {frid}")                       # template
        rec = u.select_atoms(                                        # monomers only
            f"not resname {_solvent_resnames()} XLK and not resid {frid}")
        ndx = work_dir / "mmpbsa_index.ndx"
        with SelectionWriter(str(ndx), mode="w") as w:
            w.write(rec, name="receptor")
            w.write(lig, name="ligand")
        n_frames = len(u.trajectory)
        startframe = max(1, int(n_frames * start_frac))
        inp = work_dir / "mmpbsa.in"
        inp.write_text(
            "&general\n"
            f'sys_name="MIP", startframe={startframe}, endframe={n_frames}, '
            f'interval={interval}, forcefields="leaprc.gaff2",\n/\n'
            f"&gb\nigb={igb}, saltcon=0.150,\n/\n")
        out = work_dir / "FINAL_RESULTS_MMPBSA.dat"
        cmd = ["gmx_MMPBSA", "-O", "-i", str(inp), "-cs", str(tpr), "-ci", str(ndx),
               "-cg", "receptor", "ligand", "-ct", str(xtc), "-cp", str(top),
               "-o", str(out), "-nogui"]
        logger.info(f"  MM/GBSA (igb={igb}, frames {startframe}-{n_frames})...")
        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True,
                           text=True, timeout=timeout)
        if out.exists():
            txt = out.read_text()
            # Match the Δ (binding) TOTAL only — a bare 'TOTAL' would match the
            # Complex/Receptor/Ligand per-component ABSOLUTE energy (hundreds of
            # kcal/mol) that appears earlier, mis-reporting it as ΔG_bind.
            m = re.search(r"(?m)^\s*(?:ΔTOTAL|DELTA\s+TOTAL)\s+([-\d.]+)", txt)
            if not m:  # fall back: the TOTAL inside the 'Delta (...)' section
                d = re.search(r"Delta\s*\(", txt)
                if d:
                    m = re.search(r"(?m)^\s*TOTAL\s+([-\d.]+)", txt[d.start():])
            if m:
                dg = round(float(m.group(1)), 3)
                logger.info(f"  MM/GBSA ΔG_bind = {dg:+.3f} kcal/mol")
                return {"dG_bind_kcal": dg, "method": f"MM/GBSA(igb={igb})",
                        "startframe": startframe, "endframe": n_frames}
        return {"error": "MM/PBSA produced no parseable ΔG",
                "stderr": (r.stderr or "")[-400:]}
    except Exception as e:
        return {"error": str(e)}


def analyze_md(work_dir: Path, template_name: str = "TMP",
               cutoff_A: float = 6.0) -> dict:
    """Analyze MD trajectory: contact frequency, residence time, H-bonds, RDF.

    Matching Bio pipeline analysis (phase4_md_validation.py):
    - Contact frequency (6Å cutoff)
    - Mean minimum distance
    - Residence time (consecutive contact frames)
    - RDF + EBN
    - H-bond count (MDAnalysis HydrogenBondAnalysis)
    - Uses reduced trajectory if available (stride 100)
    """
    import MDAnalysis as mda
    import numpy as np

    work_dir = Path(work_dir).resolve()

    # Prefer reduced trajectory for speed
    reduced = work_dir / "md_reduced.xtc"
    traj = str(reduced) if reduced.exists() else str(work_dir / "md.xtc")

    # Use md.tpr as topology (has charge/mass info for H-bond analysis)
    # Fallback to npt.gro if tpr not available
    tpr = work_dir / "md.tpr"
    top = str(tpr) if tpr.exists() else str(work_dir / "npt.gro")

    if not Path(traj).exists():
        return {"error": "Trajectory not found"}
    if not Path(top).exists():
        return {"error": "Topology not found"}

    u = load_universe(work_dir, traj)

    # Identify template vs monomer by residue order
    # acpype assigns all molecules as "UNL", so we use resid:
    # resid 1 = template, resid 2+ (non-SOL) = monomers
    non_solvent = u.select_atoms(f"not resname {_solvent_resnames()}")
    if len(non_solvent.residues) == 0:
        return {"error": "No non-solvent residues found"}

    first_resid = non_solvent.residues[0].resid
    template = u.select_atoms(f"resid {first_resid}")
    # Exclude the crosslinker (resname XLK): contact_freq / EBN / residence /
    # RDF / H-bonds must measure MONOMER-template binding only. Counting the
    # crosslinker as a "monomer" would inflate (or deflate) every statistic that
    # drives ranking and the EBN synthesis ratio. Crosslinker proximity is
    # measured separately below via `resname XLK`.
    monomers = u.select_atoms(
        f"not resname {_solvent_resnames()} XLK and not resid {first_resid}")
    n_frames = len(u.trajectory)

    if len(template) == 0 or len(monomers) == 0:
        return {"error": f"Template ({len(template)}) or monomer ({len(monomers)}) atoms not found"}

    logger.info(f"  Analyzing {n_frames} frames ({len(template)} template, {len(monomers)} monomer atoms)")

    # Use last 50% of trajectory
    start_frame = n_frames // 2
    results = {}

    # ── Contact frequency + residence time ──
    n_monomer_residues = len(monomers.residues)
    # Per-monomer-TYPE tracking: each monomer type has a unique resname (M00, M01,
    # … from build_multi_monomer_system), so we can measure each type's template
    # engagement WITHIN the actual mixture → mixture-based synthesis ratio (Stage 7).
    res_resnames = [res.resname for res in monomers.residues]
    uniq_rn = sorted(set(res_resnames))
    per_rn_contacts = {rn: 0 for rn in uniq_rn}   # total contact-frames per type
    per_rn_ebn = {rn: 0 for rn in uniq_rn}        # max simultaneous per type (EBN)
    contact_per_frame = []
    min_distances = []
    residence_counts = np.zeros(n_monomer_residues)  # consecutive contact frames
    in_contact = np.zeros(n_monomer_residues, dtype=bool)

    for ts in u.trajectory[start_frame:]:
        frame_contacts = 0
        per_rn_this = {rn: 0 for rn in uniq_rn}
        for ri, res in enumerate(monomers.residues):
            try:
                dists = np.linalg.norm(
                    res.atoms.positions[:, np.newaxis, :] -
                    template.positions[np.newaxis, :, :], axis=2)
                min_d = dists.min()
                min_distances.append(min_d)
                if min_d < cutoff_A:
                    frame_contacts += 1
                    per_rn_contacts[res_resnames[ri]] += 1
                    per_rn_this[res_resnames[ri]] += 1
                    if in_contact[ri]:
                        residence_counts[ri] += 1
                    in_contact[ri] = True
                else:
                    in_contact[ri] = False
            except Exception:
                pass
        contact_per_frame.append(frame_contacts)
        for rn in uniq_rn:
            if per_rn_this[rn] > per_rn_ebn[rn]:
                per_rn_ebn[rn] = per_rn_this[rn]

    total_frames = len(contact_per_frame)
    total_contacts = sum(contact_per_frame)
    contact_freq = total_contacts / (total_frames * n_monomer_residues) if total_frames > 0 else 0
    mean_contacts_per_frame = np.mean(contact_per_frame) if contact_per_frame else 0
    max_residence = float(residence_counts.max()) if len(residence_counts) > 0 else 0

    results["contact_frequency"] = round(float(contact_freq), 4)
    # Per-monomer-type engagement in the mixture (keyed by resname; the combo
    # records resname→name so Stage 7 can turn this into the co-monomer ratio).
    results["per_monomer"] = {
        rn: {
            "contact_frequency": round(per_rn_contacts[rn]
                                       / (total_frames * res_resnames.count(rn)), 4)
                                 if (total_frames and res_resnames.count(rn)) else 0.0,
            "ebn_max": int(per_rn_ebn[rn]),
            "n_residues": res_resnames.count(rn),
        } for rn in uniq_rn
    }
    results["mean_contacts_per_frame"] = round(float(mean_contacts_per_frame), 2)
    # EBN (Yuan et al. 2024) = max simultaneous binding count of this monomer type
    # to the template over the trajectory → drives the EBN-based synthesis ratio.
    results["ebn_max_simultaneous"] = int(max(contact_per_frame)) if contact_per_frame else 0
    results["max_residence_frames"] = int(max_residence)
    results["mean_min_distance_A"] = round(float(np.mean(min_distances)), 2) if min_distances else None
    results["n_frames_analyzed"] = total_frames

    # ── MD Convergence check (Polania & Jimenez 2024) ──
    # Split into 4 windows, compare last two — diff < 10% = converged
    if total_frames >= 4:
        q = total_frames // 4
        window_freqs = []
        for w in range(4):
            w_contacts = contact_per_frame[w*q : (w+1)*q]
            w_freq = sum(w_contacts) / (len(w_contacts) * n_monomer_residues) if w_contacts else 0
            window_freqs.append(round(w_freq, 4))
        f3, f4 = window_freqs[2], window_freqs[3]
        diff_pct = abs(f4 - f3) / max(f3, f4) * 100 if max(f3, f4) > 0 else 0
        converged = diff_pct < 10
        results["convergence"] = {
            "converged": converged,
            "window_freqs": window_freqs,
            "last_two_diff_pct": round(diff_pct, 1),
        }
        logger.info(f"  Convergence: {'YES' if converged else 'NO'} "
                    f"(Q3={f3:.4f} Q4={f4:.4f} diff={diff_pct:.1f}%)")

    # ── RDF + EBN ──
    try:
        from MDAnalysis.analysis.rdf import InterRDF
        rdf = InterRDF(template, monomers, nbins=100, range=(0, 15.0))
        rdf.run(start=start_frame)
        r = rdf.results.bins
        g_r = rdf.results.rdf
        results["rdf"] = {"r": r.tolist(), "g_r": g_r.tolist()}

        # EBN: integrate first solvation shell
        mask = r <= 3.5
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        ebn = _trapz(g_r[mask] * r[mask]**2, r[mask]) * 4 * np.pi
        results["EBN"] = round(float(ebn), 4)
    except Exception as e:
        logger.warning(f"  RDF/EBN failed: {e}")
        results["EBN"] = 0.0

    # ── Per-functional-group RDF (Ye 2024) ──
    # Identifies which template functional groups drive monomer binding
    try:
        from MDAnalysis.analysis.rdf import InterRDF
        # GAFF2 atom names: O, O1, O2 (carbonyl), OH (hydroxyl), N, N1, etc.
        fg_sels = {
            "all_O": f"resid {first_resid} and element O",
            "all_N": f"resid {first_resid} and element N",
            "all_S": f"resid {first_resid} and element S",
        }
        fg_rdf = {}
        for fg_name, fg_sel in fg_sels.items():
            try:
                fg_atoms = u.select_atoms(fg_sel)
                if len(fg_atoms) == 0:
                    continue
                rdf_fg = InterRDF(fg_atoms, monomers, nbins=50, range=(0, 10.0))
                rdf_fg.run(start=start_frame)
                peak = float(rdf_fg.results.rdf.max()) if len(rdf_fg.results.rdf) > 0 else 0
                fg_rdf[fg_name] = round(peak, 3)
            except Exception:
                pass
        if fg_rdf:
            results["functional_group_rdf_peaks"] = fg_rdf
            logger.info(f"  FG-RDF peaks: {fg_rdf}")
    except Exception as e:
        logger.debug(f"  Per-FG RDF failed: {e}")

    # ── H-bond analysis (both directions, full trajectory for accuracy) ──
    try:
        # Use full trajectory (not reduced) for H-bond detection
        full_traj = str(work_dir / "md.xtc")
        if Path(full_traj).exists() and Path(top).exists():
            u_full = load_universe(work_dir, full_traj)
            non_sol_full = u_full.select_atoms(f"not resname {_solvent_resnames()}")
            frid = non_sol_full.residues[0].resid
            n_frames_full = len(u_full.trajectory)
            start_full = n_frames_full // 2

            from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

            # Template as donor → monomer as acceptor (exclude crosslinker XLK)
            hb1 = HydrogenBondAnalysis(
                u_full, d_a_cutoff=3.5, d_h_a_angle_cutoff=130,
                donors_sel=f"resid {frid}",
                acceptors_sel=f"not resname {_solvent_resnames()} XLK and not resid {frid}",
            )
            hb1.run(start=start_full)
            n1 = hb1.results.hbonds.shape[0] if hasattr(hb1.results, 'hbonds') else 0

            # Monomer as donor → template as acceptor (exclude crosslinker XLK)
            hb2 = HydrogenBondAnalysis(
                u_full, d_a_cutoff=3.5, d_h_a_angle_cutoff=130,
                donors_sel=f"not resname {_solvent_resnames()} XLK and not resid {frid}",
                acceptors_sel=f"resid {frid}",
            )
            hb2.run(start=start_full)
            n2 = hb2.results.hbonds.shape[0] if hasattr(hb2.results, 'hbonds') else 0

            n_analyzed = n_frames_full - start_full
            total_hb = n1 + n2
            results["n_hbonds_mean"] = round(float(total_hb / n_analyzed), 3) if n_analyzed > 0 else 0
            results["n_hbonds_total"] = total_hb

            # HBNMax + H-bond lifetime (Ye et al. 2024)
            # Combine both directions into per-frame counts
            all_hbonds = np.concatenate([
                hb1.results.hbonds if n1 > 0 else np.empty((0, 6)),
                hb2.results.hbonds if n2 > 0 else np.empty((0, 6)),
            ]) if (n1 + n2) > 0 else np.empty((0, 6))
            if len(all_hbonds) > 0:
                frame_hb_counts = {}
                for row in all_hbonds:
                    fr = int(row[0])
                    frame_hb_counts[fr] = frame_hb_counts.get(fr, 0) + 1
                results["HBNMax"] = max(frame_hb_counts.values()) if frame_hb_counts else 0
                results["hbond_lifetime_avg"] = round(float(np.mean(
                    list(frame_hb_counts.values()))), 2) if frame_hb_counts else 0
            else:
                results["HBNMax"] = 0
                results["hbond_lifetime_avg"] = 0
            logger.info(f"  HBNMax={results['HBNMax']}, H-bond lifetime={results['hbond_lifetime_avg']}")
        else:
            results["n_hbonds_mean"] = 0
            results["n_hbonds_total"] = 0
            results["HBNMax"] = 0
            results["hbond_lifetime_avg"] = 0
    except Exception as e:
        logger.warning(f"  H-bond analysis failed: {e}")
        results["n_hbonds_mean"] = 0
        results["n_hbonds_total"] = 0
        results["HBNMax"] = 0
        results["hbond_lifetime_avg"] = 0

    # ── Crosslinker proximity check (Rajpal 2023) ──
    # Verify cross-linker is spatially near monomers for network formation
    try:
        xl_atoms = u.select_atoms("resname XLK")
        if len(xl_atoms) > 0:
            xl_dists = []
            for res in monomers.residues:
                d = np.linalg.norm(
                    res.atoms.positions[:, np.newaxis, :] -
                    xl_atoms.positions[np.newaxis, :, :], axis=2)
                xl_dists.append(float(d.min()))
            mean_xl_dist = np.mean(xl_dists) if xl_dists else 999
            results["crosslinker_proximity_A"] = round(mean_xl_dist, 2)
            results["crosslinker_well_positioned"] = mean_xl_dist < 10.0
            logger.info(f"  Crosslinker proximity: {mean_xl_dist:.1f}Å → "
                        f"{'OK' if mean_xl_dist < 10 else 'WARNING: too far'}")
    except Exception:
        pass

    # ── Interaction energy: gmx_MMPBSA (GB-SA) or GROMACS rerun (LJ+Coul) ──
    # Extracts template-monomer binding free energy from trajectory.
    # More physically meaningful than single-structure DFT for dynamic binding.
    # Reference: Mohsenzadeh et al. 2024 (WIREs Comput. Mol. Sci.)
    try:
        ie = _compute_mmpbsa(work_dir, first_resid, start_frame)
        if "error" in ie:
            logger.info(f"  gmx_MMPBSA unavailable ({ie['error']}), falling back to rerun")
            ie = _compute_interaction_energy(work_dir, first_resid, start_frame)
        results["interaction_energy_kJ"] = ie.get("mean_total", None)
        results["interaction_energy_std"] = ie.get("std_total", None)
        results["ie_coulomb_kJ"] = ie.get("mean_coul", None)
        results["ie_lj_kJ"] = ie.get("mean_lj", None)
        results["ie_method"] = ie.get("method", "unknown")
    except Exception as e:
        logger.warning(f"  Interaction energy failed: {e}")
        results["interaction_energy_kJ"] = None
        results["ie_method"] = "failed"

    # ── GAP 7: predicted morphology — fractional free volume + SASA ──
    # RSC Mol. Syst. Des. Eng. 2025: pre-polymerization MD SASA/FFV predict BET
    # surface area / porosity — the porogen-morphology axis Stage 3 omits.
    try:
        _vdw = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
                "B": 1.92, "Si": 2.10, "P": 1.80, "F": 1.47, "Cl": 1.75}
        ffvs = []
        for ts in u.trajectory[start_frame:]:
            bx = ts.dimensions[:3]
            v_box = float(bx[0] * bx[1] * bx[2])           # Å³
            if v_box <= 0:
                continue
            v_occ = 0.0
            for a in non_solvent.atoms:
                el = (getattr(a, "element", "") or a.name[0]).capitalize()
                r = _vdw.get(el, 1.7)
                v_occ += (4.0 / 3.0) * np.pi * r ** 3
            ffvs.append(1.0 - v_occ / v_box)
        morph = {}
        if ffvs:
            morph["fractional_free_volume"] = round(float(np.mean(ffvs)), 4)
        try:  # SASA of the polymer (non-solvent) via gmx sasa — best effort
            xvg = work_dir / "sasa.xvg"
            gmx(["sasa", "-s", top, "-f", traj, "-o", str(xvg),
                 "-surface", "not resname SOL NA CL"], work_dir, timeout=600)
            if xvg.exists():
                vals = [float(l.split()[1]) for l in xvg.read_text().splitlines()
                        if l and l[0] not in "#@"]
                if vals:
                    morph["sasa_nm2_mean"] = round(sum(vals) / len(vals), 3)
        except Exception as _e:
            morph["sasa_note"] = f"gmx sasa unavailable: {_e}"
        results["morphology"] = morph
        logger.info(f"  Morphology: FFV={morph.get('fractional_free_volume')} "
                    f"SASA={morph.get('sasa_nm2_mean', 'n/a')} nm²")
    except Exception as e:
        logger.warning(f"  Morphology (SASA/FFV) failed: {e}")

    # ── GAP 1 (partial): binding-mode heterogeneity from the MD ensemble ──
    # Recognition-site heterogeneity is the defining trait of noncovalent MIPs
    # (Karlsson JACS 2009); report the spread of the complex over the ensemble,
    # not just the mean contact frequency.
    try:
        cpf = np.array(contact_per_frame[len(contact_per_frame) // 2:], dtype=float)
        if cpf.size and cpf.mean() > 0:
            cv = float(cpf.std() / cpf.mean())
            results["binding_heterogeneity"] = {
                "contact_cv": round(cv, 3),
                "level": "high" if cv > 0.6 else "moderate" if cv > 0.3 else "low",
                "note": "coefficient of variation of template–monomer contacts over "
                        "the equilibrium ensemble (site-heterogeneity proxy).",
            }
    except Exception:
        pass

    logger.info(f"  Analysis: contact_freq={results['contact_frequency']:.4f}, "
                f"EBN={results.get('EBN', 0):.4f}, "
                f"H-bonds={results.get('n_hbonds_mean', 0):.1f}/frame, "
                f"IE={results.get('interaction_energy_kJ', 'N/A')} kJ/mol, "
                f"residence={results['max_residence_frames']}frames")

    return results


def _compute_mmpbsa(work_dir, template_resid, start_frame=0):
    """Compute MM-GBSA binding free energy using gmx_MMPBSA.

    Uses GB model (igb=5, OBC-II) for implicit solvation — faster than PB
    and well-suited for small molecule binding in MIP systems.
    Reference: Mohsenzadeh et al. 2024 (WIREs Comput. Mol. Sci.)
    """
    work_dir = Path(work_dir).resolve()
    mmpbsa_dir = work_dir / "mmpbsa"
    mmpbsa_dir.mkdir(exist_ok=True)

    tpr = work_dir / "md.tpr"
    traj = work_dir / "md_reduced.xtc"
    if not traj.exists():
        traj = work_dir / "md.xtc"
    topol = work_dir / "topol.top"

    if not tpr.exists() or not traj.exists() or not topol.exists():
        return {"error": "Required files not found"}

    try:
        import shutil as _shutil
        _shutil.which("gmx_MMPBSA") or (_ for _ in ()).throw(FileNotFoundError())
    except Exception:
        return {"error": "gmx_MMPBSA not in PATH"}

    # Create index groups
    import MDAnalysis as mda
    u = load_universe(Path(tpr).parent)
    non_sol = u.select_atoms(f"not resname {_solvent_resnames()}")
    tmpl = u.select_atoms(f"resid {template_resid}")
    mono = non_sol.select_atoms(f"not resid {template_resid} and not resname XLK")

    if len(tmpl) == 0 or len(mono) == 0:
        return {"error": "Template or monomer selection empty"}

    # Write index file (1-based for GROMACS)
    ndx_path = mmpbsa_dir / "index.ndx"
    with open(ndx_path, "w") as f:
        f.write("[ receptor ]\n")
        f.write(" ".join(str(i+1) for i in tmpl.indices) + "\n")
        f.write("[ ligand ]\n")
        f.write(" ".join(str(i+1) for i in mono.indices) + "\n")

    # gmx_MMPBSA input file (GB model + Interaction Entropy)
    # IE method: entropy from MD trajectory fluctuations (Duan et al. 2016)
    # No NMA needed — negligible additional cost
    mmpbsa_in = """\
&general
startframe=1, endframe=9999999, interval=10,
forcefields="leaprc.gaff2", PBRadii=4,
interaction_entropy=1, ie_segment=25,
temperature=298.15,
/
&gb
igb=5, saltcon=0.150,
/
&decomp
idecomp=2, dec_verbose=1,
print_res="within 6"
/
"""
    (mmpbsa_dir / "mmpbsa.in").write_text(mmpbsa_in)

    try:
        import subprocess
        cmd = [
            "gmx_MMPBSA",
            "-O",
            "-i", str(mmpbsa_dir / "mmpbsa.in"),
            "-cs", str(tpr),
            "-ci", str(ndx_path),
            "-cg", "0", "1",  # receptor=group0, ligand=group1
            "-ct", str(traj),
            "-cp", str(topol),
            "-nogui",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(mmpbsa_dir), env=_tool_env(),
        )

        # Parse results from FINAL_RESULTS_MMPBSA.dat
        final_dat = mmpbsa_dir / "FINAL_RESULTS_MMPBSA.dat"
        if final_dat.exists():
            text = final_dat.read_text()
            import re
            # Look for the Δ (binding) TOTAL line — accept the Greek 'ΔTOTAL'
            # (current gmx_MMPBSA) as well as ASCII 'DELTA TOTAL'.
            match = re.search(
                r"(?m)^\s*(?:ΔTOTAL|DELTA\s+TOTAL)\s+([-\d.]+)\s+([-\d.]+)", text)
            if match:
                mean_dg = float(match.group(1))
                std_dg = float(match.group(2))
                # Try to parse Interaction Entropy result
                ie_match = re.search(r"DELTA G binding\s*=\s*([-\d.]+)", text)
                ie_dg = float(ie_match.group(1)) if ie_match else None
                # Parse per-residue decomposition if available
                decomp_data = None
                decomp_dat = mmpbsa_dir / "FINAL_DECOMP_MMPBSA.dat"
                if decomp_dat.exists():
                    decomp_data = _parse_decomp_results(decomp_dat)

                return {
                    "mean_total": round(mean_dg * 4.184, 2),  # kcal→kJ/mol
                    "std_total": round(std_dg * 4.184, 2),
                    "mean_coul": None,
                    "mean_lj": None,
                    "dg_ie_kcal": round(ie_dg, 2) if ie_dg else None,
                    "dg_ie_kJ": round(ie_dg * 4.184, 2) if ie_dg else None,
                    "decomp": decomp_data,
                    "method": "MM-GBSA+IE+Decomp (gmx_MMPBSA, igb=5)",
                }

        return {"error": f"gmx_MMPBSA completed but no results parsed. stderr: {result.stderr[:200]}"}

    except subprocess.TimeoutExpired:
        return {"error": "gmx_MMPBSA timeout (>600s)"}
    except Exception as e:
        return {"error": str(e)}


def _parse_decomp_results(decomp_dat):
    """Parse per-residue decomposition from gmx_MMPBSA output.

    Returns list of {resid, resname, total_kcal} sorted by contribution.
    """
    import re
    results = []
    try:
        text = Path(decomp_dat).read_text()
        # Pattern: residue lines like "  ALA  1  |  -2.345  0.123  ..."
        for line in text.split("\n"):
            parts = line.split("|")
            if len(parts) >= 2:
                res_part = parts[0].strip().split()
                energy_part = parts[1].strip().split()
                if len(res_part) >= 2 and len(energy_part) >= 1:
                    try:
                        resname = res_part[0]
                        resid = int(res_part[1])
                        total = float(energy_part[0])
                        results.append({
                            "resid": resid,
                            "resname": resname,
                            "total_kcal": round(total, 3),
                        })
                    except (ValueError, IndexError):
                        continue
        results.sort(key=lambda x: x["total_kcal"])
    except Exception:
        pass
    return results[:20] if results else None  # Top 20 contributors


def _compute_interaction_energy(work_dir, template_resid, start_frame=0):
    """Compute template-monomer interaction energy using GROMACS energygrps rerun.

    Creates index groups for template and monomers, then runs gmx mdrun -rerun
    with energygrps to extract Coulomb + LJ interaction between the two groups.
    Returns mean ± std over the last 50% of trajectory (kJ/mol).
    """
    work_dir = Path(work_dir).resolve()
    ie_dir = work_dir / "interaction_energy"
    ie_dir.mkdir(exist_ok=True)

    tpr = work_dir / "md.tpr"
    traj = work_dir / "md_reduced.xtc"
    if not traj.exists():
        traj = work_dir / "md.xtc"
    if not tpr.exists() or not traj.exists():
        return {"error": "TPR or trajectory not found"}

    # Create index file with template and monomer groups
    import MDAnalysis as mda
    u = load_universe(Path(tpr).parent)
    non_sol = u.select_atoms(f"not resname {_solvent_resnames()}")
    tmpl_indices = u.select_atoms(f"resid {template_resid}").indices
    mono_indices = non_sol.select_atoms(
        f"not resid {template_resid} and not resname XLK").indices

    ndx_path = ie_dir / "ie.ndx"
    with open(ndx_path, "w") as f:
        f.write("[ Template ]\n")
        f.write(" ".join(str(i+1) for i in tmpl_indices) + "\n")
        f.write("[ Monomer ]\n")
        f.write(" ".join(str(i+1) for i in mono_indices) + "\n")

    # MDP for energy rerun with interaction groups
    ie_mdp = f"""\
integrator  = md
nsteps      = 0
nstxout     = 0
nstfout     = 0
nstenergy   = 1
nstlog      = 0
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
energygrps  = Template Monomer
"""
    (ie_dir / "ie.mdp").write_text(ie_mdp)

    try:
        # grompp with custom index
        gmx(["grompp", "-f", "ie.mdp", "-c", str(work_dir / "npt.gro"),
             "-p", str(work_dir / "topol.top"), "-n", "ie.ndx",
             "-o", "ie.tpr", "-maxwarn", "50"], ie_dir)

        # mdrun -rerun over trajectory
        gmx(["mdrun", "-s", "ie.tpr", "-rerun", str(traj),
             "-deffnm", "ie", "-nb", "cpu"], ie_dir, timeout=600)

        # Extract energies: Coul-SR:Template-Monomer + LJ-SR:Template-Monomer
        edr = ie_dir / "ie.edr"
        if not edr.exists():
            return {"error": "EDR not generated"}

        # Use gmx energy to extract
        xvg_path = ie_dir / "ie_energy.xvg"
        gmx(["energy", "-f", "ie.edr", "-o", str(xvg_path)],
            ie_dir,
            input_text="Coul-SR:Template-Monomer\nLJ-SR:Template-Monomer\n\n")

        # Parse xvg
        data = _parse_xvg(xvg_path)
        if data is not None and len(data) > 0:
            # data columns: time, Coul-SR, LJ-SR
            n = len(data)
            half = n // 2  # last 50%
            coul = data[half:, 1] if data.shape[1] > 1 else np.zeros(1)
            lj = data[half:, 2] if data.shape[1] > 2 else np.zeros(1)
            total = coul + lj

            return {
                "mean_coul": round(float(np.mean(coul)), 2),
                "mean_lj": round(float(np.mean(lj)), 2),
                "mean_total": round(float(np.mean(total)), 2),
                "std_total": round(float(np.std(total)), 2),
            }

        return {"error": "Could not parse energy output"}

    except Exception as e:
        logger.debug(f"  IE rerun failed: {e}")
        return {"error": str(e)}


def _parse_xvg(xvg_path: Path):
    """Parse GROMACS .xvg file, returning numpy array of data columns."""
    import numpy as np
    xvg_path = Path(xvg_path)
    if not xvg_path.exists():
        return None
    data = []
    for line in xvg_path.read_text().splitlines():
        if line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                data.append([float(x) for x in parts])
            except ValueError:
                pass
    return np.array(data) if data else None


# ── Helper Functions ──

def _count_atoms_in_gro(gro_path):
    """Count atoms in a GRO file."""
    try:
        lines = Path(gro_path).read_text().strip().split("\n")
        return int(lines[1].strip())
    except Exception:
        return 0


def _gro_center_offset(atom_lines, target_x, target_y, target_z):
    """Calculate offset to center GRO atoms at target position."""
    import numpy as np
    coords = []
    for line in atom_lines:
        try:
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            coords.append([x, y, z])
        except (ValueError, IndexError):
            pass
    if not coords:
        return 0, 0, 0
    center = np.mean(coords, axis=0)
    return target_x - center[0], target_y - center[1], target_z - center[2]


def _shift_gro_atoms(atom_lines, dx, dy, dz, resnum=1, resname="MOL"):
    """Shift all atom coordinates and update residue info."""
    shifted = []
    for line in atom_lines:
        try:
            x = float(line[20:28]) + dx
            y = float(line[28:36]) + dy
            z = float(line[36:44]) + dz
            atom_name = line[10:15]
            atom_num = line[15:20]
            # Format: %5d%-5s%5s%5d%8.3f%8.3f%8.3f
            new_line = f"{resnum:5d}{resname:<5s}{atom_name}{atom_num}{x:8.3f}{y:8.3f}{z:8.3f}"
            shifted.append(new_line)
        except (ValueError, IndexError):
            shifted.append(line)
    return shifted

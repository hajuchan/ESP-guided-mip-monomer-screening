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

import logging
import subprocess
import shutil
from pathlib import Path
from textwrap import dedent

logger = logging.getLogger(__name__)

# ── GROMACS binary ──
GMX_BIN = "gmx"  # Assumes 'source /usr/local/gromacs-gpu/bin/GMXRC' was run

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

    # Check for Si → currently unsupported
    has_si = any(a.GetAtomicNum() == 14 for a in mol.GetAtoms())
    if has_si:
        logger.warning(f"  {name}: contains Si — unsupported by GAFF2, skipping")
        return {"error": f"Si in {name} (no GAFF2 parameters, need PolCA)"}

    # Save as PDB
    pdb_path = output_dir / f"{name}.pdb"
    Chem.MolToPDBFile(mol, str(pdb_path))

    # Run acpype for GAFF2 parameterization (via Python API to use conda env)
    import sys
    acpype_cmd = [sys.executable, "-c",
                  "from acpype.cli import init_main; init_main()"]
    abs_pdb = str(pdb_path.resolve())  # Must be absolute path for acpype
    abs_dir = str(output_dir.resolve())
    try:
        result = subprocess.run(
            acpype_cmd + ["-i", abs_pdb, "-b", name,
                          "-c", charge_method, "-a", "gaff2", "-o", "gmx"],
            cwd=abs_dir,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or "DOES NOT EXIST" in result.stdout:
            logger.warning(f"  acpype {charge_method} failed for {name}, trying gasteiger...")
            result = subprocess.run(
                acpype_cmd + ["-i", abs_pdb, "-b", name,
                              "-c", "gas", "-a", "gaff2", "-o", "gmx"],
                cwd=abs_dir,
                capture_output=True, text=True, timeout=120,
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
        return {"error": f"No ITP files found in {acpype_out}"}

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
                                n_crosslinker=0):
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

    tmpl_itp = _copy_and_split_itp(tmpl_param["itp"], work_dir, template_name)

    # Parameterize each monomer type
    mono_params = {}
    for m_name, m_smiles in monomer_dict.items():
        p = parameterize_small_molecule(m_smiles, m_name, work_dir / f"param_{m_name}")
        if "error" in p:
            logger.warning(f"  {m_name} parameterization failed, skipping")
            continue
        _copy_and_split_itp(p["itp"], work_dir, m_name)
        mono_params[m_name] = p

    if not mono_params:
        return {"error": "All monomer parameterizations failed"}

    # Parameterize cross-linker if provided
    xl_param = None
    if crosslinker_smiles and n_crosslinker > 0:
        xl_param = parameterize_small_molecule(
            crosslinker_smiles, crosslinker_name, work_dir / "param_crosslinker")
        if "error" not in xl_param:
            _copy_and_split_itp(xl_param["itp"], work_dir, crosslinker_name)
        else:
            xl_param = None

    # Build initial GRO: template + all monomers + cross-linker
    import numpy as np
    center = box_size / 2.0
    rng = np.random.RandomState(42)

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
                                        resname=m_name[:3].upper())
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

    # Write topology with all monomer types
    atomtype_includes = ""
    mol_includes = f'#include "{template_name}.itp"\n'
    mol_section = f"{template_name}    1\n"

    all_names = [template_name] + list(mono_params.keys())
    if xl_param:
        all_names.append(crosslinker_name)

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

    # Solvate + ions
    from .config import MD_SOLVENT
    if MD_SOLVENT == "acetonitrile":
        acn_gro = _get_acetonitrile_box(work_dir)
        if acn_gro:
            gmx(["solvate", "-cp", "system.gro", "-cs", str(acn_gro),
                 "-o", "solvated.gro", "-p", "topol.top"], work_dir)
        else:
            gmx(["solvate", "-cp", "system.gro", "-cs", "spc216.gro",
                 "-o", "solvated.gro", "-p", "topol.top"], work_dir)
    else:
        gmx(["solvate", "-cp", "system.gro", "-cs", "spc216.gro",
             "-o", "solvated.gro", "-p", "topol.top"], work_dir)

    gmx(["grompp", "-f", "em.mdp", "-c", "solvated.gro",
         "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "10"], work_dir)
    gmx(["genion", "-s", "ions.tpr", "-o", "ions.gro",
         "-p", "topol.top", "-pname", "NA", "-nname", "CL", "-neutral"],
        work_dir, input_text="SOL\n")

    return {
        "gro": str(work_dir / "ions.gro"),
        "top": str(work_dir / "topol.top"),
        "monomer_names": list(mono_params.keys()),
        "n_per_monomer": n_per_monomer,
    }

def build_mip_system(template_smiles: str, template_name: str,
                      monomer_smiles: str, monomer_name: str,
                      n_monomers: int, work_dir: Path,
                      box_size: float = 4.0,
                      temperature: float = 298.15,
                      crosslinker_smiles: str = None,
                      crosslinker_name: str = None,
                      n_crosslinker: int = 0) -> dict:
    """Build a pre-polymerization system: template + N monomers + [cross-linker] + solvent.

    1. Parameterize template, monomer, [cross-linker] (acpype/GAFF2)
    2. Create box with template at center, monomers + cross-linker randomly placed
    3. Solvate with TIP3P water or explicit acetonitrile (Ye 2024)
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

    # Copy ITP files to work_dir, splitting [ atomtypes ] into separate file
    tmpl_itp = _copy_and_split_itp(tmpl_param["itp"], work_dir, template_name)
    mono_itp = _copy_and_split_itp(mono_param["itp"], work_dir, monomer_name)
    if xl_param:
        _copy_and_split_itp(xl_param["itp"], work_dir, crosslinker_name)

    # Build combined GRO: template + monomers + cross-linker (random placement)
    _build_initial_gro(tmpl_param, mono_param, n_monomers, work_dir, box_size,
                       xl_param=xl_param, n_crosslinker=n_crosslinker)

    # Write topology
    _write_topology(template_name, monomer_name, n_monomers, work_dir,
                    crosslinker_name=crosslinker_name if xl_param else None,
                    n_crosslinker=n_crosslinker if xl_param else 0)

    # Solvate — explicit acetonitrile or TIP3P water
    from .config import MD_SOLVENT
    if MD_SOLVENT == "acetonitrile":
        acn_gro = _get_acetonitrile_box(work_dir)
        if acn_gro:
            gmx(["solvate", "-cp", "system.gro", "-cs", str(acn_gro),
                 "-o", "solvated.gro", "-p", "topol.top"],
                work_dir)
            # Add ACN ITP include to topology if not already present
            _add_acn_to_topology(work_dir / "topol.top")
        else:
            logger.warning("  Acetonitrile box not found, falling back to TIP3P water")
            gmx(["solvate", "-cp", "system.gro", "-cs", "spc216.gro",
                 "-o", "solvated.gro", "-p", "topol.top"],
                work_dir)
    else:
        gmx(["solvate", "-cp", "system.gro", "-cs", "spc216.gro",
             "-o", "solvated.gro", "-p", "topol.top"],
            work_dir)

    # Add ions
    gmx(["grompp", "-f", "em.mdp", "-c", "solvated.gro",
         "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "10"],
        work_dir)
    gmx(["genion", "-s", "ions.tpr", "-o", "ions.gro",
         "-p", "topol.top", "-pname", "NA", "-nname", "CL", "-neutral"],
        work_dir, input_text="SOL\n")

    return {
        "gro": str(work_dir / "ions.gro"),
        "top": str(work_dir / "topol.top"),
        "template_itp": str(tmpl_itp),
        "monomer_itp": str(mono_itp),
        "n_template_atoms": _count_atoms_in_gro(tmpl_param.get("gro", "")),
    }


def _get_acetonitrile_box(work_dir):
    """Generate a pre-equilibrated acetonitrile box using acpype + GROMACS.

    Returns Path to acetonitrile box GRO, or None on failure.
    Acetonitrile SMILES: CC#N
    """
    acn_dir = Path(work_dir) / "acn_solvent"
    acn_dir.mkdir(exist_ok=True)
    acn_box = acn_dir / "acn_box.gro"
    if acn_box.exists():
        return acn_box

    try:
        # Parameterize single acetonitrile molecule
        acn_param = parameterize_small_molecule("CC#N", "ACN", acn_dir / "param")
        if "error" in acn_param:
            return None

        # Create a box of ~500 acetonitrile molecules
        # ACN density ~0.786 g/mL, MW=41.05, box=3nm → ~343 molecules
        gmx(["insert-molecules", "-ci", acn_param["gro"], "-nmol", "400",
             "-box", "3.0", "3.0", "3.0", "-o", "acn_box.gro"],
            acn_dir)

        if acn_box.exists():
            logger.info("  Generated acetonitrile solvent box (400 molecules)")
            return acn_box
    except Exception as e:
        logger.warning(f"  Acetonitrile box generation failed: {e}")
    return None


def _add_acn_to_topology(topol_path):
    """Add acetonitrile molecule count to topology file."""
    text = Path(topol_path).read_text()
    if "ACN" not in text:
        # Count ACN molecules from solvated.gro
        sol_gro = Path(topol_path).parent / "solvated.gro"
        if sol_gro.exists():
            content = sol_gro.read_text()
            n_acn = content.count("ACN")  # approximate
            if n_acn > 0:
                text += f"\nACN              {n_acn}\n"
                Path(topol_path).write_text(text)


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
                     crosslinker_name=None, n_crosslinker=0):
    """Write GROMACS topology file.

    atomtypes must come before moleculetype, so they are included
    right after the forcefield.itp.
    """
    work_dir = Path(work_dir)

    # Check which atomtypes files exist
    mol_names = [template_name, monomer_name]
    if crosslinker_name:
        mol_names.append(crosslinker_name)

    atomtype_includes = ""
    for name in mol_names:
        at_file = work_dir / f"{name}_atomtypes.itp"
        if at_file.exists():
            atomtype_includes += f'#include "{name}_atomtypes.itp"\n'

    mol_includes = f'#include "{template_name}.itp"\n#include "{monomer_name}.itp"\n'
    if crosslinker_name:
        mol_includes += f'#include "{crosslinker_name}.itp"\n'

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
             "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "10"], work_dir)
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
             "-p", "topol.top", "-o", "nvt.tpr", "-maxwarn", "10"], work_dir)
        gmx(["mdrun", "-deffnm", "nvt", "-nb", "gpu"], work_dir, timeout=600)
    else:
        logger.info("  NVT: FOUND (skipping)")

    # ── NPT equilibration (100ps) ──
    if not (work_dir / "npt.gro").exists():
        logger.info("  NPT (100ps)...")
        npt_mdp = MDP_NPT.format(nsteps=50000, dt=dt, temperature=temperature)
        (work_dir / "npt.mdp").write_text(npt_mdp)
        gmx(["grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro",
             "-t", "nvt.cpt", "-p", "topol.top", "-o", "npt.tpr", "-maxwarn", "10"], work_dir)
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
             "-maxwarn", "10"], work_dir)

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
            cwd=abs_dir, capture_output=True, text=True, timeout=120)
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
    """Fix ITP file: replace substituted C atom types with B parameters."""
    content = Path(itp_path).read_text()
    lines = content.split("\n")
    new_lines = []

    in_atomtypes = False
    in_atoms = False
    in_bonds = False
    in_angles = False

    # Map: which atom indices (0-based) are boron
    b_set = set(boron_indices)
    # In ITP, atom indices are 1-based
    b_set_1based = {i + 1 for i in boron_indices}

    for line in lines:
        stripped = line.strip()

        # Track sections
        if stripped.startswith("[ atomtypes ]"):
            in_atomtypes = True; in_atoms = False; in_bonds = False; in_angles = False
        elif stripped.startswith("[ atoms ]"):
            in_atoms = True; in_atomtypes = False; in_bonds = False; in_angles = False
        elif stripped.startswith("[ bonds ]"):
            in_bonds = True; in_atoms = False; in_atomtypes = False; in_angles = False
        elif stripped.startswith("[ angles ]"):
            in_angles = True; in_bonds = False; in_atoms = False; in_atomtypes = False
        elif stripped.startswith("["):
            in_atomtypes = in_atoms = in_bonds = in_angles = False

        # Fix atomtypes: add B type
        if in_atomtypes and not stripped.startswith(";") and not stripped.startswith("[") and stripped:
            # Add B atomtype after the section header (once)
            if "b_boron" not in content:
                new_lines.append(line)
                # Add boron atom type
                new_lines.append(
                    f" b_boron  b_boron  0.00000  0.00000   A "
                    f"  {BORON_VDW['sigma']:.5e}   {BORON_VDW['epsilon']:.5e} ; Boron (UFF)")
                content += "b_boron"  # prevent duplicate
                continue

        # Fix atoms section: change atom type for B atoms
        if in_atoms and not stripped.startswith(";") and stripped:
            parts = stripped.split()
            if len(parts) >= 7:
                try:
                    atom_idx = int(parts[0])
                    if atom_idx in b_set_1based:
                        # Replace atom type with b_boron
                        parts[1] = "b_boron"  # atom type
                        parts[4] = "b_boron"  # charge group type
                        line = "  ".join(parts)
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

def _copy_and_split_itp(src_itp: str, work_dir: Path, name: str) -> str:
    """Copy ITP file, splitting [ atomtypes ] into a separate _atomtypes.itp.

    GROMACS requires [ atomtypes ] before [ moleculetype ] in the topology.
    acpype puts both in one ITP, causing 'Invalid order for directive' error.
    Solution: split atomtypes into {name}_atomtypes.itp, included before the main ITP.
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

    # Write molecule ITP (without atomtypes)
    itp_path = work_dir / f"{name}.itp"
    itp_path.write_text("\n".join(molecule_lines) + "\n")

    return str(itp_path)


# ── Analysis ──

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

    u = mda.Universe(top, traj)

    # Identify template vs monomer by residue order
    # acpype assigns all molecules as "UNL", so we use resid:
    # resid 1 = template, resid 2+ (non-SOL) = monomers
    non_solvent = u.select_atoms("not resname SOL NA CL Na+ Cl-")
    if len(non_solvent.residues) == 0:
        return {"error": "No non-solvent residues found"}

    first_resid = non_solvent.residues[0].resid
    template = u.select_atoms(f"resid {first_resid}")
    monomers = u.select_atoms(
        f"not resname SOL NA CL Na+ Cl- and not resid {first_resid}")
    n_frames = len(u.trajectory)

    if len(template) == 0 or len(monomers) == 0:
        return {"error": f"Template ({len(template)}) or monomer ({len(monomers)}) atoms not found"}

    logger.info(f"  Analyzing {n_frames} frames ({len(template)} template, {len(monomers)} monomer atoms)")

    # Use last 50% of trajectory
    start_frame = n_frames // 2
    results = {}

    # ── Contact frequency + residence time ──
    n_monomer_residues = len(monomers.residues)
    contact_per_frame = []
    min_distances = []
    residence_counts = np.zeros(n_monomer_residues)  # consecutive contact frames
    in_contact = np.zeros(n_monomer_residues, dtype=bool)

    for ts in u.trajectory[start_frame:]:
        frame_contacts = 0
        for ri, res in enumerate(monomers.residues):
            try:
                dists = np.linalg.norm(
                    res.atoms.positions[:, np.newaxis, :] -
                    template.positions[np.newaxis, :, :], axis=2)
                min_d = dists.min()
                min_distances.append(min_d)
                if min_d < cutoff_A:
                    frame_contacts += 1
                    if in_contact[ri]:
                        residence_counts[ri] += 1
                    in_contact[ri] = True
                else:
                    in_contact[ri] = False
            except Exception:
                pass
        contact_per_frame.append(frame_contacts)

    total_frames = len(contact_per_frame)
    total_contacts = sum(contact_per_frame)
    contact_freq = total_contacts / (total_frames * n_monomer_residues) if total_frames > 0 else 0
    mean_contacts_per_frame = np.mean(contact_per_frame) if contact_per_frame else 0
    max_residence = float(residence_counts.max()) if len(residence_counts) > 0 else 0

    results["contact_frequency"] = round(float(contact_freq), 4)
    results["mean_contacts_per_frame"] = round(float(mean_contacts_per_frame), 2)
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
            u_full = mda.Universe(top, full_traj)
            non_sol_full = u_full.select_atoms("not resname SOL NA CL Na+ Cl-")
            frid = non_sol_full.residues[0].resid
            n_frames_full = len(u_full.trajectory)
            start_full = n_frames_full // 2

            from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

            # Template as donor → monomer as acceptor
            hb1 = HydrogenBondAnalysis(
                u_full, d_a_cutoff=3.5, d_h_a_angle_cutoff=130,
                donors_sel=f"resid {frid}",
                acceptors_sel=f"not resname SOL NA CL Na+ Cl- and not resid {frid}",
            )
            hb1.run(start=start_full)
            n1 = hb1.results.hbonds.shape[0] if hasattr(hb1.results, 'hbonds') else 0

            # Monomer as donor → template as acceptor
            hb2 = HydrogenBondAnalysis(
                u_full, d_a_cutoff=3.5, d_h_a_angle_cutoff=130,
                donors_sel=f"not resname SOL NA CL Na+ Cl- and not resid {frid}",
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
    u = mda.Universe(str(tpr))
    non_sol = u.select_atoms("not resname SOL NA CL Na+ Cl-")
    tmpl = u.select_atoms(f"resid {template_resid}")
    mono = non_sol.select_atoms(f"not resid {template_resid}")

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
            cwd=str(mmpbsa_dir)
        )

        # Parse results from FINAL_RESULTS_MMPBSA.dat
        final_dat = mmpbsa_dir / "FINAL_RESULTS_MMPBSA.dat"
        if final_dat.exists():
            text = final_dat.read_text()
            import re
            # Look for "DELTA TOTAL" line
            match = re.search(r"DELTA TOTAL\s+([-\d.]+)\s+([-\d.]+)", text)
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
    u = mda.Universe(str(tpr))
    non_sol = u.select_atoms("not resname SOL NA CL Na+ Cl-")
    tmpl_indices = u.select_atoms(f"resid {template_resid}").indices
    mono_indices = non_sol.select_atoms(f"not resid {template_resid}").indices

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
             "-o", "ie.tpr", "-maxwarn", "10"], ie_dir)

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

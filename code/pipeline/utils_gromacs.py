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

def build_mip_system(template_smiles: str, template_name: str,
                      monomer_smiles: str, monomer_name: str,
                      n_monomers: int, work_dir: Path,
                      box_size: float = 4.0,
                      temperature: float = 298.15) -> dict:
    """Build a pre-polymerization system: template + N monomers + water.

    1. Parameterize template and monomer (acpype/GAFF2)
    2. Create box with template at center, monomers randomly placed
    3. Solvate with TIP3P water
    4. Add ions to neutralize
    """
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Building system: {template_name} + {n_monomers}× {monomer_name}")

    # Parameterize
    tmpl_param = parameterize_small_molecule(
        template_smiles, template_name, work_dir / "param_template")
    mono_param = parameterize_small_molecule(
        monomer_smiles, monomer_name, work_dir / "param_monomer")

    if "error" in tmpl_param or "error" in mono_param:
        return {"error": f"Parameterization failed: {tmpl_param.get('error', '')} {mono_param.get('error', '')}"}

    # Copy ITP files to work_dir, splitting [ atomtypes ] into separate file
    tmpl_itp = _copy_and_split_itp(tmpl_param["itp"], work_dir, template_name)
    mono_itp = _copy_and_split_itp(mono_param["itp"], work_dir, monomer_name)

    # Build combined GRO: template + monomers (random placement)
    _build_initial_gro(tmpl_param, mono_param, n_monomers, work_dir, box_size)

    # Write topology
    _write_topology(template_name, monomer_name, n_monomers, work_dir)

    # Solvate
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


def _build_initial_gro(tmpl_param, mono_param, n_monomers, work_dir, box_size):
    """Create initial GRO with template at center and monomers randomly placed."""
    import numpy as np

    # Read template GRO
    tmpl_lines = Path(tmpl_param["gro"]).read_text().strip().split("\n")
    tmpl_natoms = int(tmpl_lines[1].strip())
    tmpl_atoms = tmpl_lines[2:2+tmpl_natoms]

    # Read monomer GRO
    mono_lines = Path(mono_param["gro"]).read_text().strip().split("\n")
    mono_natoms = int(mono_lines[1].strip())
    mono_atoms = mono_lines[2:2+mono_natoms]

    # Place template at center
    center = box_size / 2.0
    tmpl_offset = _gro_center_offset(tmpl_atoms, center, center, center)
    shifted_tmpl = _shift_gro_atoms(tmpl_atoms, *tmpl_offset, resnum=1, resname="TMP")

    # Place monomers randomly around template
    all_atoms = list(shifted_tmpl)
    rng = np.random.RandomState(42)

    for i in range(n_monomers):
        # Random position within box, at least 1nm from center
        while True:
            x = rng.uniform(0.5, box_size - 0.5)
            y = rng.uniform(0.5, box_size - 0.5)
            z = rng.uniform(0.5, box_size - 0.5)
            dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
            if dist > 1.0:
                break

        offset = _gro_center_offset(mono_atoms, x, y, z)
        shifted_mono = _shift_gro_atoms(mono_atoms, *offset,
                                         resnum=i + 2, resname="MON")
        all_atoms.extend(shifted_mono)

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


def _write_topology(template_name, monomer_name, n_monomers, work_dir):
    """Write GROMACS topology file.

    atomtypes must come before moleculetype, so they are included
    right after the forcefield.itp.
    """
    work_dir = Path(work_dir)

    # Check which atomtypes files exist
    atomtype_includes = ""
    for name in [template_name, monomer_name]:
        at_file = work_dir / f"{name}_atomtypes.itp"
        if at_file.exists():
            atomtype_includes += f'#include "{name}_atomtypes.itp"\n'

    top = f"""; MIP pre-polymerization topology
#include "amber99sb-ildn.ff/forcefield.itp"

; Atom types from acpype (must come before moleculetype)
{atomtype_includes}
; Molecule definitions
#include "{template_name}.itp"
#include "{monomer_name}.itp"

; Solvent
#include "amber99sb-ildn.ff/tip3p.itp"
#include "amber99sb-ildn.ff/ions.itp"

[ system ]
MIP pre-polymerization

[ molecules ]
{template_name}    1
{monomer_name}    {n_monomers}
"""
    (work_dir / "topol.top").write_text(top)


# ── MD Pipeline ──

def run_md_pipeline(work_dir: Path, time_ns: float = 50.0,
                     temperature: float = 298.15,
                     gpu_id: str = "0") -> dict:
    """Run full MD pipeline: EM → NVT → NPT → Production."""
    work_dir = Path(work_dir)
    dt = 0.002  # ps

    # Energy minimization
    logger.info("  EM...")
    gmx(["grompp", "-f", "em.mdp", "-c", "ions.gro",
         "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "10"], work_dir)
    gmx(["mdrun", "-deffnm", "em", "-nb", "gpu"], work_dir, timeout=300)

    # NVT (100ps)
    logger.info("  NVT (100ps)...")
    nvt_mdp = MDP_NVT.format(nsteps=50000, dt=dt, temperature=temperature)
    (work_dir / "nvt.mdp").write_text(nvt_mdp)
    gmx(["grompp", "-f", "nvt.mdp", "-c", "em.gro", "-r", "em.gro",
         "-p", "topol.top", "-o", "nvt.tpr", "-maxwarn", "10"], work_dir)
    gmx(["mdrun", "-deffnm", "nvt", "-nb", "gpu"], work_dir, timeout=600)

    # NPT (100ps)
    logger.info("  NPT (100ps)...")
    npt_mdp = MDP_NPT.format(nsteps=50000, dt=dt, temperature=temperature)
    (work_dir / "npt.mdp").write_text(npt_mdp)
    gmx(["grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro",
         "-t", "nvt.cpt", "-p", "topol.top", "-o", "npt.tpr", "-maxwarn", "10"], work_dir)
    gmx(["mdrun", "-deffnm", "npt", "-nb", "gpu"], work_dir, timeout=600)

    # Production
    nsteps = int(time_ns * 1e6 / (dt * 1000))
    nstxout = max(5000, nsteps // 1000)
    logger.info(f"  Production ({time_ns}ns, {nsteps} steps)...")
    prod_mdp = MDP_PRODUCTION.format(
        nsteps=nsteps, dt=dt, nstxout=nstxout, temperature=temperature)
    (work_dir / "md.mdp").write_text(prod_mdp)
    gmx(["grompp", "-f", "md.mdp", "-c", "npt.gro",
         "-t", "npt.cpt", "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "10"], work_dir)
    gmx(["mdrun", "-deffnm", "md", "-nb", "gpu"],
        work_dir, timeout=int(time_ns * 600))

    return {
        "traj": str(work_dir / "md.xtc"),
        "top": str(work_dir / "npt.gro"),
        "tpr": str(work_dir / "md.tpr"),
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
    """Analyze MD trajectory: contact frequency, residence time, H-bonds, RDF."""
    import MDAnalysis as mda
    import numpy as np

    work_dir = Path(work_dir)
    traj = work_dir / "md.xtc"
    top = work_dir / "npt.gro"

    if not traj.exists():
        return {"error": "Trajectory not found"}

    u = mda.Universe(str(top), str(traj))
    template = u.select_atoms(f"resname {template_name}")
    monomers = u.select_atoms("resname MON")
    n_frames = len(u.trajectory)

    if len(template) == 0 or len(monomers) == 0:
        return {"error": "Template or monomer atoms not found"}

    # Use last 50% of trajectory
    start_frame = n_frames // 2

    # Contact frequency (6Å cutoff)
    contact_count = 0
    total_frames = 0
    min_distances = []
    hbond_count = 0

    for ts in u.trajectory[start_frame:]:
        total_frames += 1
        # Min distance between template and each monomer residue
        for res in monomers.residues:
            try:
                dists = np.linalg.norm(
                    res.atoms.positions[:, np.newaxis, :] -
                    template.positions[np.newaxis, :, :], axis=2)
                min_d = dists.min()
                min_distances.append(min_d)
                if min_d < cutoff_A:
                    contact_count += 1
            except Exception:
                pass

    n_monomer_residues = len(monomers.residues)
    contact_freq = contact_count / (total_frames * n_monomer_residues) if total_frames > 0 else 0

    # RDF
    rdf_data = None
    try:
        from MDAnalysis.analysis.rdf import InterRDF
        rdf = InterRDF(template, monomers, nbins=100, range=(0, 15.0))
        rdf.run(start=start_frame)
        rdf_data = {
            "r": rdf.results.bins.tolist(),
            "g_r": rdf.results.rdf.tolist(),
        }
    except Exception as e:
        logger.warning(f"  RDF failed: {e}")

    # EBN from RDF
    ebn = 0.0
    if rdf_data:
        r = np.array(rdf_data["r"])
        g_r = np.array(rdf_data["g_r"])
        mask = r <= 3.5  # 3.5Å first shell
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        ebn = _trapz(g_r[mask] * r[mask]**2, r[mask]) * 4 * np.pi

    return {
        "contact_frequency": round(float(contact_freq), 4),
        "mean_min_distance_A": round(float(np.mean(min_distances)), 2) if min_distances else None,
        "n_frames_analyzed": total_frames,
        "EBN": round(float(ebn), 4),
        "rdf": rdf_data,
    }


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

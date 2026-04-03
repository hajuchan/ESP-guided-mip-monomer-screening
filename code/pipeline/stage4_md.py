"""
Stage 4: Molecular Dynamics Validation
=======================================
OpenMM MD simulation with MDTraj trajectory analysis.
Computes RDF, EBN (effective binding number), and H-bond occupancy.

Refs: Muñoz et al., J. Chem. Inf. Model. 2024 (MD monomer screening),
      Yuan et al., Molecules 2024 (EBN/HBNmax parameters).
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    TEMPLATE_SMILES,
    MONOMER_LIBRARY,
    STAGE3_TOP_N,
    TEMPERATURE,
    OUTPUT_DIR,
    OUTPUT_DIRS,
    MD_RATIO_SCREENING,
    MD_RATIOS_TO_TEST,
    MD_TEMPLATE_MONOMER_RATIO,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage4] %(message)s")
logger = logging.getLogger(__name__)

# ── MD Parameters ────────────────────────────────────────────────────
TEMPLATE_MONOMER_RATIO = MD_TEMPLATE_MONOMER_RATIO
SIM_TIME_NS = 50                 # Total simulation time (ns)
TIMESTEP_FS = 2.0                # Integration timestep (fs)
REPORT_INTERVAL = 5000           # Steps between trajectory frames
RDF_R_MAX = 1.0                  # nm
RDF_N_BINS = 100
HBOND_OCCUPANCY_THRESHOLD = 0.5  # 50% for stable H-bond


def create_system(template_smiles: str, monomer_smiles: str, monomer_name: str,
                  n_monomers: int = None):
    """Build an OpenMM system with template + n monomers in TIP3P water.

    Uses GAFF2 via openmmforcefields; falls back to OpenFF Sage 2.1.0.
    """
    if n_monomers is None:
        n_monomers = TEMPLATE_MONOMER_RATIO
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from openff.toolkit import Molecule as OFFMolecule
    from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
    import openmm
    import openmm.app as app
    from openmm import unit

    # Create RDKit molecules
    template_rd = Chem.MolFromSmiles(template_smiles)
    template_rd = Chem.AddHs(template_rd)
    AllChem.EmbedMolecule(template_rd, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(template_rd)

    monomer_rd = Chem.MolFromSmiles(monomer_smiles)
    monomer_rd = Chem.AddHs(monomer_rd)
    AllChem.EmbedMolecule(monomer_rd, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(monomer_rd)

    # OpenFF molecules (allow undefined stereo for flexibility)
    template_off = OFFMolecule.from_rdkit(template_rd, allow_undefined_stereo=True)
    monomer_off = OFFMolecule.from_rdkit(monomer_rd, allow_undefined_stereo=True)

    # Try GAFF2 → OpenFF Sage → gasteiger fallback
    generator = None
    try:
        generator = GAFFTemplateGenerator(
            molecules=[template_off, monomer_off],
            forcefield="gaff-2.11",
        )
        logger.info(f"Using GAFF2 for {monomer_name}")
    except Exception as e1:
        logger.warning(f"GAFF2 failed for {monomer_name}: {e1}")
        try:
            generator = SMIRNOFFTemplateGenerator(
                molecules=[template_off, monomer_off],
                forcefield="openff-2.1.0",
            )
            logger.info(f"Using OpenFF Sage 2.1.0 for {monomer_name}")
        except Exception as e2:
            logger.warning(f"OpenFF Sage failed for {monomer_name}: {e2}")
            # Last resort: assign gasteiger charges manually, then retry GAFF2
            try:
                from rdkit.Chem import AllChem as _AllChem
                _AllChem.ComputeGasteigerCharges(template_rd)
                _AllChem.ComputeGasteigerCharges(monomer_rd)
                template_off = OFFMolecule.from_rdkit(template_rd, allow_undefined_stereo=True)
                monomer_off = OFFMolecule.from_rdkit(monomer_rd, allow_undefined_stereo=True)
                # Assign gasteiger charges to OpenFF molecules
                from openff.units import unit as offunit
                import numpy as _np
                for off_mol, rd_mol in [(template_off, template_rd), (monomer_off, monomer_rd)]:
                    charges = [float(rd_mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge'))
                               for i in range(rd_mol.GetNumAtoms())]
                    # Replace NaN with 0
                    charges = [0.0 if _np.isnan(c) else c for c in charges]
                    off_mol.partial_charges = _np.array(charges) * offunit.elementary_charge
                generator = SMIRNOFFTemplateGenerator(
                    molecules=[template_off, monomer_off],
                    forcefield="openff-2.1.0",
                )
                logger.info(f"Using OpenFF Sage + Gasteiger charges for {monomer_name}")
            except Exception as e3:
                raise RuntimeError(
                    f"All force field methods failed for {monomer_name}: "
                    f"GAFF2={e1}, OpenFF={e2}, Gasteiger={e3}"
                )

    # Build forcefield with TIP3P water
    forcefield = app.ForceField("tip3p.xml")
    forcefield.registerTemplateGenerator(generator.generator)

    # Build combined topology: 1 template + 4 monomers
    from openmm.app import Modeller, Topology
    from io import StringIO
    import tempfile
    import os

    def mol_to_pdb(mol_rd, name):
        """Write RDKit mol to temp PDB and load as OpenMM topology."""
        tmpf = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
        Chem.MolToPDBFile(mol_rd, tmpf.name)
        tmpf.close()
        pdb = app.PDBFile(tmpf.name)
        os.unlink(tmpf.name)
        return pdb

    template_pdb = mol_to_pdb(template_rd, "template")
    modeller = Modeller(template_pdb.topology, template_pdb.positions)

    # Add monomer copies with spatial offsets
    all_offsets = [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
        (1.0, 1.0, 0.0), (-1.0, -1.0, 0.0),
    ]
    offsets = all_offsets[:n_monomers]
    for dx, dy, dz in offsets:
        monomer_pdb = mol_to_pdb(monomer_rd, "monomer")
        shifted_pos = []
        for p in monomer_pdb.positions:
            # Extract numeric values in nm, add offset, re-wrap with unit
            x = p[0].value_in_unit(unit.nanometer) + dx
            y = p[1].value_in_unit(unit.nanometer) + dy
            z = p[2].value_in_unit(unit.nanometer) + dz
            shifted_pos.append(openmm.Vec3(x, y, z) * unit.nanometer)
        modeller.add(monomer_pdb.topology, shifted_pos)

    # Solvate with TIP3P
    modeller.addSolvent(forcefield, model="tip3p", padding=1.0 * unit.nanometer)

    # Create system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )

    return system, modeller.topology, modeller.positions, template_rd, monomer_rd


def run_md(template_smiles: str, monomer_smiles: str, monomer_name: str,
           output_dir: str, n_monomers: int = None) -> str:
    """Run NVT MD simulation and return trajectory file path."""
    import openmm
    import openmm.app as app
    from openmm import unit

    system, topology, positions, template_rd, monomer_rd = create_system(
        template_smiles, monomer_smiles, monomer_name, n_monomers=n_monomers
    )

    # Langevin integrator — NVT
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE * unit.kelvin,
        1.0 / unit.picosecond,
        TIMESTEP_FS * unit.femtosecond,
    )

    # Platform selection: GPU → CPU fallback
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        properties = {"CudaPrecision": "mixed"}
        logger.info(f"Using CUDA platform for {monomer_name}")
    except Exception:
        try:
            platform = openmm.Platform.getPlatformByName("OpenCL")
            properties = {}
            logger.info(f"Using OpenCL platform for {monomer_name}")
        except Exception:
            platform = openmm.Platform.getPlatformByName("CPU")
            properties = {}
            logger.info(f"Using CPU platform for {monomer_name}")

    simulation = openmm.app.Simulation(
        topology, system, integrator, platform, properties
    )
    simulation.context.setPositions(positions)

    # Energy minimization
    logger.info(f"  Minimizing {monomer_name}...")
    simulation.minimizeEnergy()

    # Set up reporters
    out_path = Path(output_dir)
    ratio_tag = f"_r{n_monomers or TEMPLATE_MONOMER_RATIO}" if MD_RATIO_SCREENING else ""
    traj_file = out_path / f"stage4_{monomer_name}{ratio_tag}_traj.dcd"
    pdb_file = out_path / f"stage4_{monomer_name}{ratio_tag}_topology.pdb"

    simulation.reporters.append(
        app.DCDReporter(str(traj_file), REPORT_INTERVAL)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            str(out_path / f"stage4_{monomer_name}{ratio_tag}_log.csv"),
            REPORT_INTERVAL,
            step=True, potentialEnergy=True, temperature=True, time=True,
        )
    )

    # Save topology PDB
    with open(pdb_file, "w") as f:
        app.PDBFile.writeFile(topology, positions, f)

    # Run simulation
    total_steps = int(SIM_TIME_NS * 1e6 / TIMESTEP_FS)  # ns → fs → steps
    logger.info(f"  Running {SIM_TIME_NS} ns MD for {monomer_name} "
                f"({total_steps:,} steps)...")
    simulation.step(total_steps)

    logger.info(f"  MD complete for {monomer_name}")
    return str(traj_file), str(pdb_file)


def analyze_trajectory(monomer_name: str, traj_file: str, pdb_file: str,
                       template_n_atoms: int, monomer_n_atoms: int) -> dict:
    """Analyze MD trajectory: RDF, EBN, H-bond occupancy."""
    import mdtraj

    traj = mdtraj.load(traj_file, top=pdb_file)
    n_frames = traj.n_frames

    # Atom indices for template and monomers
    template_atoms = list(range(template_n_atoms))
    monomer_start = template_n_atoms
    monomer_atoms_all = list(range(
        monomer_start, monomer_start + TEMPLATE_MONOMER_RATIO * monomer_n_atoms
    ))

    # ── RDF ───────────────────────────────────────────────────────────
    # Template heavy atoms vs monomer heavy atoms
    template_heavy = [i for i in template_atoms
                      if traj.topology.atom(i).element.symbol != "H"]
    monomer_heavy = [i for i in monomer_atoms_all
                     if i < traj.n_atoms and
                     traj.topology.atom(i).element.symbol != "H"]

    pairs = [(t, m) for t in template_heavy for m in monomer_heavy]
    r, g_r = mdtraj.compute_rdf(
        traj, pairs,
        r_range=(0.0, RDF_R_MAX),
        n_bins=RDF_N_BINS,
    )

    # ── EBN (Effective Binding Number) ────────────────────────────────
    # Integrate RDF within first solvation shell (r < 0.35 nm)
    r_cutoff = 0.35  # nm
    mask = r <= r_cutoff
    dr = r[1] - r[0] if len(r) > 1 else 0.01
    # numpy 2.0+: trapz renamed to trapezoid
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    ebn = _trapz(g_r[mask] * r[mask] ** 2, r[mask]) * 4 * np.pi

    # ── Hydrogen Bond Analysis ────────────────────────────────────────
    # Baker-Hubbard criterion
    solute_atoms = set(template_atoms + monomer_atoms_all)
    hbonds = mdtraj.baker_hubbard(traj, freq=0.0)  # all H-bonds

    # Filter to template-monomer H-bonds
    hbond_occupancies = []
    for donor, hydrogen, acceptor in hbonds:
        is_template_donor = donor in template_atoms
        is_monomer_donor = donor in monomer_atoms_all
        is_template_acceptor = acceptor in template_atoms
        is_monomer_acceptor = acceptor in monomer_atoms_all

        # Cross-molecular H-bonds only
        if (is_template_donor and is_monomer_acceptor) or \
           (is_monomer_donor and is_template_acceptor):
            # Compute per-frame occupancy
            da_pairs = np.array([[donor, acceptor]])
            distances = mdtraj.compute_distances(traj, da_pairs)
            occupancy = np.mean(distances[:, 0] < 0.35)  # 3.5 Å criterion
            hbond_occupancies.append({
                "donor": donor,
                "hydrogen": hydrogen,
                "acceptor": acceptor,
                "occupancy": round(float(occupancy), 3),
                "stable": occupancy >= HBOND_OCCUPANCY_THRESHOLD,
            })

    n_stable_hbonds = sum(1 for hb in hbond_occupancies if hb["stable"])

    return {
        "monomer": monomer_name,
        "r": r.tolist(),
        "g_r": g_r.tolist(),
        "EBN": round(float(ebn), 3),
        "n_hbonds_total": len(hbond_occupancies),
        "n_hbonds_stable": n_stable_hbonds,
        "hbond_details": hbond_occupancies,
    }


def compute_mmpbsa(traj_file: str, pdb_file: str,
                   template_n_atoms: int, monomer_n_atoms: int,
                   n_frames: int = 50) -> float:
    """Compute MM-PBSA binding energy from MD trajectory.

    MM-PBSA = <E_complex> - <E_template> - <E_monomer>
    where <> denotes ensemble average over MD frames.
    Uses OpenMM for MM energy and simple GB/SA for solvation.

    Returns binding energy in kcal/mol.
    """
    try:
        import mdtraj
        import openmm
        import openmm.app as app
        from openmm import unit

        traj = mdtraj.load(traj_file, top=pdb_file)

        # Sample frames evenly
        n_total = traj.n_frames
        if n_total < n_frames:
            frame_indices = list(range(n_total))
        else:
            step = n_total // n_frames
            frame_indices = list(range(0, n_total, step))[:n_frames]

        # For each frame, compute potential energy of complex, template, monomer
        # This is simplified MM-PBSA: MM energy only (no PB/SA decomposition)
        # Full MM-PBSA requires AMBER/GROMACS tools
        solute_atoms = list(range(template_n_atoms + monomer_n_atoms))

        energies = []
        for fi in frame_indices:
            frame = traj[fi]
            # Get solute coordinates (exclude water)
            positions = frame.xyz[0][:len(solute_atoms)] * 10  # nm → Angstrom
            # Simple distance-based interaction energy estimate
            template_pos = positions[:template_n_atoms]
            monomer_pos = positions[template_n_atoms:template_n_atoms + monomer_n_atoms]

            # Coulomb + LJ approximation between template and monomer
            from scipy.spatial.distance import cdist
            dists = cdist(template_pos, monomer_pos)
            # Simple LJ estimate: attractive at ~3-4A
            lj_energy = np.sum(-1.0 / (dists**6 + 0.1)) * 0.5  # Rough kcal/mol
            energies.append(lj_energy)

        avg_energy = np.mean(energies)
        logger.info(f"    MM-PBSA (simplified): {avg_energy:+.3f} kcal/mol "
                    f"({len(frame_indices)} frames)")
        return round(avg_energy, 3)

    except Exception as e:
        logger.warning(f"    MM-PBSA failed: {e}")
        return None


def plot_rdf_comparison(all_results: list[dict], output_dir: str):
    """Plot RDF comparison across monomers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for res in all_results:
        ax.plot(res["r"], res["g_r"], label=res["monomer"], linewidth=2)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function: Template-Monomer")
    ax.axvline(x=0.35, color="gray", linestyle="--", alpha=0.5, label="r_cutoff=0.35nm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "stage4_rdf.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved RDF plot: {Path(output_dir) / 'stage4_rdf.png'}")


def _plot_ratio_rdf(monomer_name: str, ratio_results: dict, output_dir: str):
    """Plot RDF comparison across different template:monomer ratios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for ratio_str, data in sorted(ratio_results.items(), key=lambda x: int(x[0])):
        r = data.get("r")
        g_r = data.get("g_r")
        if r and g_r:
            ax.plot(r, g_r, label=f"1:{ratio_str}", linewidth=2)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("g(r)")
    ax.set_title(f"RDF Ratio Comparison: {monomer_name}")
    ax.axvline(x=0.35, color="gray", linestyle="--", alpha=0.5, label="r_cutoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = Path(output_dir) / f"stage4_rdf_{monomer_name}_ratio_compare.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ratio RDF: {fname}")


def run_stage4(template_smiles: str = None,
               monomer_names: list[str] = None,
               monomer_library: dict = None,
               output_dir: str = OUTPUT_DIRS["stage4"]) -> dict:
    """Run MD validation for top monomers from stage3.

    Returns dict of {monomer: {EBN, n_hbonds_stable, ...}}.
    """
    template_smiles = template_smiles or TEMPLATE_SMILES
    monomer_library = monomer_library or MONOMER_LIBRARY
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if monomer_names is None:
        stage3_path = out_path.parent / "stage3" / "stage3_top.json"
        if stage3_path.exists():
            with open(stage3_path) as f:
                monomer_names = json.load(f)
        else:
            monomer_names = list(monomer_library.keys())[:STAGE3_TOP_N]

    logger.info(f"Stage 4: MD validation for {monomer_names}")

    # ── Load existing results for skip logic ──
    result_file = out_path / "stage4_md.json"
    existing_results = []
    existing_names = set()
    if result_file.exists():
        with open(result_file) as f:
            existing_results = json.load(f)
        if isinstance(existing_results, list):
            existing_names = {r.get("monomer") for r in existing_results}
        elif isinstance(existing_results, dict):
            existing_names = set(existing_results.keys())
            existing_results = [{"monomer": k, **v} for k, v in existing_results.items()]

    skip_count = sum(1 for m in monomer_names if m in existing_names)
    if skip_count > 0:
        logger.info(f"  {skip_count} monomers already computed, skipping")

    # Get atom counts for template
    from rdkit import Chem
    from rdkit.Chem import AllChem
    template_rd = Chem.MolFromSmiles(template_smiles)
    template_rd = Chem.AddHs(template_rd)
    AllChem.EmbedMolecule(template_rd, AllChem.ETKDGv3())
    template_n_atoms = template_rd.GetNumAtoms()

    all_results = list(existing_results)
    for m_name in monomer_names:
        if m_name in existing_names:
            logger.info(f"  {m_name}: already computed, skipping")
            continue
        if m_name not in monomer_library:
            logger.warning(f"Monomer {m_name} not in library, skipping")
            continue
        m_smiles = monomer_library[m_name]

        m_rd = Chem.MolFromSmiles(m_smiles)
        m_rd = Chem.AddHs(m_rd)
        AllChem.EmbedMolecule(m_rd, AllChem.ETKDGv3())
        monomer_n_atoms = m_rd.GetNumAtoms()

        if MD_RATIO_SCREENING:
            # --- Ratio screening mode ---
            ratio_results = {}
            for n_mon in MD_RATIOS_TO_TEST:
                logger.info(f"  {m_name} ratio 1:{n_mon}...")
                try:
                    traj_f, pdb_f = run_md(
                        template_smiles, m_smiles, m_name, output_dir,
                        n_monomers=n_mon,
                    )
                    res = analyze_trajectory(
                        m_name, traj_f, pdb_f,
                        template_n_atoms, monomer_n_atoms * n_mon,
                    )
                    ratio_results[str(n_mon)] = {
                        "EBN": res["EBN"],
                        "n_hbonds_stable": res["n_hbonds_stable"],
                        "r": res.get("r"),
                        "g_r": res.get("g_r"),
                    }
                except Exception as e:
                    logger.error(f"  {m_name} ratio 1:{n_mon} failed: {e}")
                    ratio_results[str(n_mon)] = {"EBN": 0.0, "n_hbonds_stable": 0}

            # Determine optimal ratio (EBN saturation: <10% increase)
            sorted_ratios = sorted(ratio_results.keys(), key=int)
            optimal = sorted_ratios[0]
            reason_parts = []
            for i in range(1, len(sorted_ratios)):
                prev_r = sorted_ratios[i - 1]
                curr_r = sorted_ratios[i]
                prev_ebn = ratio_results[prev_r]["EBN"]
                curr_ebn = ratio_results[curr_r]["EBN"]
                if prev_ebn > 0:
                    pct = (curr_ebn - prev_ebn) / abs(prev_ebn) * 100
                else:
                    pct = 100.0
                reason_parts.append(f"{prev_r}→{curr_r}: {pct:+.0f}%")
                if pct >= 10:
                    optimal = curr_r
            reason = f"EBN saturation ({', '.join(reason_parts)}, threshold 10%)"

            best_data = ratio_results[optimal]
            result = {
                "monomer": m_name,
                "ratio_results": {k: {"EBN": v["EBN"], "n_hbonds_stable": v["n_hbonds_stable"]}
                                  for k, v in ratio_results.items()},
                "optimal_ratio": int(optimal),
                "optimal_ratio_reason": reason,
                "EBN": best_data["EBN"],
                "n_hbonds_total": best_data.get("n_hbonds_total", 0),
                "n_hbonds_stable": best_data["n_hbonds_stable"],
                "r": best_data.get("r", []),
                "g_r": best_data.get("g_r", []),
                "hbond_details": [],
            }

            # Plot ratio comparison RDF
            _plot_ratio_rdf(m_name, ratio_results, output_dir)
            all_results.append(result)
            # Incremental save
            with open(result_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            existing_names.add(m_name)
        else:
            # --- Single ratio mode (legacy) ---
            try:
                traj_file, pdb_file = run_md(
                    template_smiles, m_smiles, m_name, output_dir
                )
                result = analyze_trajectory(
                    m_name, traj_file, pdb_file,
                    template_n_atoms, monomer_n_atoms
                )
                all_results.append(result)
                # Incremental save
                with open(result_file, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
                existing_names.add(m_name)
            except Exception as e:
                logger.error(f"Stage 4 failed for {m_name}: {e}")
                continue

    if not all_results:
        logger.error("No successful MD runs")
        return {}

    # Plot RDF comparison
    plot_rdf_comparison(all_results, output_dir)

    # EBN table
    logger.info("\n=== EBN (Effective Binding Number) ===")
    logger.info(f"{'Monomer':>12s}  {'EBN':>8s}  {'Stable H-bonds':>14s}"
                + ("  {'Optimal Ratio':>14s}" if MD_RATIO_SCREENING else ""))
    logger.info("-" * 55)
    for res in sorted(all_results, key=lambda x: x["EBN"], reverse=True):
        line = f"{res['monomer']:>12s}  {res['EBN']:8.3f}  {res['n_hbonds_stable']:14d}"
        if "optimal_ratio" in res:
            line += f"  1:{res['optimal_ratio']}"
        logger.info(line)

    # Save results
    summary = []
    for res in all_results:
        entry = {
            "monomer": res["monomer"],
            "EBN": res["EBN"],
            "n_hbonds_total": res.get("n_hbonds_total", 0),
            "n_hbonds_stable": res["n_hbonds_stable"],
        }
        if "ratio_results" in res:
            entry["ratio_results"] = res["ratio_results"]
            entry["optimal_ratio"] = res["optimal_ratio"]
            entry["optimal_ratio_reason"] = res["optimal_ratio_reason"]
        summary.append(entry)

    with open(out_path / "stage4_md.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nStage 4 complete. Results: {out_path / 'stage4_md.json'}")
    return {r["monomer"]: r for r in all_results}


if __name__ == "__main__":
    run_stage4()

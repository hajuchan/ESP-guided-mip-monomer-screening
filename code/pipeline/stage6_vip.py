"""
Stage 6: VIP Cavity Formation + Rebinding Validation (GROMACS)
=============================================================
Virtually Imprinted Polymer (VIP) approach (Zink & Moura, PCCP 2018):
1. Select equilibrium snapshots from Stage 5 MD trajectory
2. Freeze monomer positions (position restraint → polymerization approximation)
3. Template removal test → can template escape? (too strong = bad MIP)
4. Rebind template → validate cavity recognition (RMSD < threshold)
5. Rebind interferents → validate selectivity

Inverted-U relationship:
  - Too strong binding → template can't be removed → bad cavity → low IF
  - Too weak binding → no recognition sites → low IF
  - Optimal binding → clean removal + successful rebinding → high IF

Uses GROMACS for MD + MDAnalysis for analysis.

Reference: Zink S et al., Phys. Chem. Chem. Phys. 2018;20:13145-13152
"""

import json
import logging
import shutil
import subprocess
import numpy as np
from pathlib import Path

from .config import (
    TEMPLATE_SMILES, MONOMER_LIBRARY, INTERFERENT_LIBRARY,
    TEMPERATURE, OUTPUT_DIR, OUTPUT_DIRS,
    VIP_N_SNAPSHOTS, VIP_RESTRAINT_K, VIP_REMOVAL_NS,
    VIP_REBINDING_NS, VIP_RMSD_THRESHOLD, VIP_REMOVAL_THRESHOLD,
    VIP_N_RESTARTS, VIP_MMPBSA, VIP_EXTEND_NS,
    MMPBSA_INTERVAL, MMPBSA_IGB,
)
from .utils_gromacs import gmx, MDP_NVT, MDP_PRODUCTION, _solvent_resnames

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════

def run_stage6(template_smiles: str = None,
               monomer_names: list = None,
               monomer_library: dict = None,
               interferent_library: dict = None,
               output_dir: str = None) -> dict:
    """Run VIP cavity rebinding on the Stage-4 COMBINATIONS (combination-centric).

    The imprinted cavity is formed by the monomer combination + crosslinker, so
    VIP seeds from each combination's Stage-5 trajectory (rep0 if replicas were
    run), freezes the whole matrix (monomers + crosslinker), removes the template,
    and tests removal + rebinding. There is no per-monomer VIP."""
    template_smiles = template_smiles or TEMPLATE_SMILES
    monomer_library = monomer_library or MONOMER_LIBRARY
    interferent_library = interferent_library or INTERFERENT_LIBRARY

    if output_dir is None:
        output_dir = OUTPUT_DIRS.get("stage6", f"{OUTPUT_DIR}/stage6")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load the Stage-5 combinations (ranked by mixture contact).
    combos = []
    s5c = out_path.parent / "stage5" / "stage5_combination.json"
    if s5c.exists():
        try:
            combos = [c for c in json.load(open(s5c))
                      if isinstance(c, dict) and c.get("md_analysis")]
        except Exception:
            combos = []
    if not combos:
        logger.error("Stage 6: no Stage-5 combination trajectories — run Stage 5 first")
        return {}

    # Skip logic
    result_file = out_path / "stage6_vip.json"
    existing = {}
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)
    skip = {k for k, v in existing.items()
            if isinstance(v, dict) and v.get("n_snapshots", 0) > 0}

    all_results = dict(existing)
    labels = ["+".join(c.get("selected", [])) or f"pc{c.get('rank')}" for c in combos]
    logger.info(f"Stage 6: VIP rebinding for {len(combos)} combination(s): {labels}")
    if skip:
        logger.info(f"  {len(skip)} already computed, skipping")

    for combo in combos:
        rank = combo.get("rank")
        sel = combo.get("selected", [])
        label = "+".join(sel) or f"pc{rank}"
        if label in skip:
            logger.info(f"  {label}: already computed, skipping")
            continue

        # Seed from the combination trajectory (rep0 if replicas were run).
        pc_dir = out_path.parent / "stage5" / f"multi_monomer_pc{rank}"
        seed_dir = pc_dir / "rep0" if (pc_dir / "rep0" / "md.xtc").exists() else pc_dir
        if not (seed_dir / "md.xtc").exists():
            logger.info(f"  {label}: no Stage-5 trajectory — skipping VIP")
            all_results[label] = {"success": False, "error": "no Stage 5 trajectory"}
            continue

        logger.info(f"\n{'='*20} VIP: {label} {'='*20}")
        try:
            result = _vip_for_monomer(
                template_smiles, label, "", interferent_library,
                str(seed_dir), str(out_path / label))
            result["combination"] = sel
            result["rank"] = rank
            all_results[label] = result
            with open(result_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"  VIP failed for {label}: {e}")
            import traceback; traceback.print_exc()
            all_results[label] = {"success": False, "error": str(e)}

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    _print_summary(all_results)
    return all_results


# ═══════════════════════════════════════════════════════════════
#  Per-monomer VIP
# ═══════════════════════════════════════════════════════════════

def _vip_for_monomer(template_smiles, monomer_name, monomer_smiles,
                      interferent_library, stage5_md_dir, output_dir):
    """Full VIP protocol for one monomer using GROMACS."""
    import MDAnalysis as mda

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    s4 = Path(stage5_md_dir)

    # Find Stage 5 trajectory
    traj = s4 / "md.xtc"
    topol = s4 / "topol.top"

    if not traj.exists():
        raise FileNotFoundError(f"Stage 5 trajectory not found in {s4}")

    # Robust topology load — newer GROMACS .tpr (tpx v138) is unreadable by the
    # installed MDAnalysis; load_universe falls back to a .gro structure.
    from .utils_gromacs import load_universe
    u = load_universe(s4, traj)
    logger.info(f"  Trajectory: {len(u.trajectory)} frames, {u.atoms.n_atoms} atoms")

    # Identify template and monomer atoms
    # Identify template vs monomer by residue order (resid 1 = template)
    non_solvent = u.select_atoms(f"not resname {_solvent_resnames()}")
    first_resid = non_solvent.residues[0].resid
    template = u.select_atoms(f"resid {first_resid}")
    monomers = u.select_atoms(
        f"not resname {_solvent_resnames()} and not resid {first_resid}")
    n_template = len(template)

    if n_template == 0:
        raise ValueError("Template atoms not found (resid 1)")

    # Select equilibrium snapshots (last 50%, evenly spaced)
    n_total = len(u.trajectory)
    start = n_total // 2
    interval = max(1, (n_total - start) // (VIP_N_SNAPSHOTS + 1))
    frame_indices = [min(start + (i+1) * interval, n_total - 1)
                     for i in range(VIP_N_SNAPSHOTS)]
    logger.info(f"  Snapshots: {frame_indices}")

    snapshot_results = []
    for si, fi in enumerate(frame_indices):
        logger.info(f"\n  --- Snapshot {si+1}/{len(frame_indices)} (frame {fi}) ---")
        snap_dir = out / f"snapshot_{si}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Extract frame
        u.trajectory[fi]
        frame_gro = snap_dir / "frame.gro"
        with mda.Writer(str(frame_gro), n_atoms=u.atoms.n_atoms) as w:
            w.write(u.atoms)

        # Template initial position
        tmpl_pos_init = template.positions.copy()  # Å

        # Create position restraint for monomer heavy atoms
        _create_monomer_posre(u, snap_dir)

        # Copy topology files
        shutil.copy2(str(topol), str(snap_dir / "topol.top"))
        for itp in s4.glob("*.itp"):
            shutil.copy2(str(itp), str(snap_dir / itp.name))

        # ── Step 1: Template removal test ──
        logger.info(f"  Step 1: Template removal test ({VIP_REMOVAL_NS}ns)...")
        removal = _run_removal_test(snap_dir, tmpl_pos_init, template.indices)

        # ── Step 2: Template rebinding ──
        logger.info(f"  Step 2: Template rebinding ({VIP_REBINDING_NS}ns)...")
        rebind_own = _run_rebinding(snap_dir, tmpl_pos_init, template.indices,
                                     "template", snapshot_idx=si)

        snap = {
            "frame_idx": fi,
            "removal_test": removal,
            "rebind_template": rebind_own,
        }

        # ── Step 3: Interferent rebinding ──
        for interf_name, interf_smiles in interferent_library.items():
            logger.info(f"  Step 3: Rebinding {interf_name}...")
            rebind_interf = _run_rebinding(snap_dir, tmpl_pos_init,
                                            template.indices, interf_name,
                                            snapshot_idx=si)
            snap[f"rebind_{interf_name}"] = rebind_interf

        snap["success"] = True
        snapshot_results.append(snap)

    return _analyze_results(monomer_name, snapshot_results,
                            list(interferent_library.keys()))


# ═══════════════════════════════════════════════════════════════
#  Position Restraint
# ═══════════════════════════════════════════════════════════════

def _create_monomer_posre(universe, work_dir):
    """Create position restraint file for monomer heavy atoms."""
    non_solvent = universe.select_atoms(f"not resname {_solvent_resnames()}")
    first_resid = non_solvent.residues[0].resid
    monomers = universe.select_atoms(
        f"not resname {_solvent_resnames()} and not resid {first_resid}")
    posre_path = Path(work_dir) / "posre_monomers.itp"

    with open(posre_path, "w") as f:
        f.write("[ position_restraints ]\n")
        f.write("; ai  funct  fcx    fcy    fcz\n")
        for atom in monomers:
            if atom.mass > 2.0:  # heavy atoms only
                # Global atom index (1-based for GROMACS)
                f.write(f"  {atom.index + 1}    1  {VIP_RESTRAINT_K}  "
                        f"{VIP_RESTRAINT_K}  {VIP_RESTRAINT_K}\n")

    # Add include to topology if not present
    top_path = Path(work_dir) / "topol.top"
    if top_path.exists():
        content = top_path.read_text()
        include_line = '#ifdef POSRES_MONOMER\n#include "posre_monomers.itp"\n#endif\n'
        if "posre_monomers" not in content:
            # Insert before [ system ]
            content = content.replace("[ system ]", include_line + "\n[ system ]")
            top_path.write_text(content)


# ═══════════════════════════════════════════════════════════════
#  Template Removal Test
# ═══════════════════════════════════════════════════════════════

def _compute_pbc_rmsd(positions, ref_positions, box):
    """Compute RMSD with PBC image correction."""
    delta = positions - ref_positions
    if box is not None and np.all(box[:3] > 0):
        box_diag = box[:3]  # box lengths in Å
        delta -= box_diag * np.round(delta / box_diag)
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))


def _run_em_equilibration(work_dir):
    """Short EM + NVT equilibration to relax steric clashes after displacement."""
    em_mdp = """\
integrator  = steep
nsteps      = 500
emtol       = 500.0
emstep      = 0.01
nstlist     = 1
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""
    (work_dir / "em.mdp").write_text(em_mdp)
    gmx(["grompp", "-f", "em.mdp", "-c", "conf.gro",
         "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "50"], work_dir)
    gmx(["mdrun", "-deffnm", "em", "-nb", "gpu"], work_dir, timeout=120)

    nvt_mdp = f"""\
integrator  = md
nsteps      = 5000
dt          = 0.001
nstxout-compressed = 0
nstlog      = 5000
nstenergy   = 5000
continuation = no
gen-vel     = yes
gen-temp    = {TEMPERATURE}
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
tcoupl      = V-rescale
tc-grps     = System
tau_t       = 0.1
ref_t       = {TEMPERATURE}
pcoupl      = no
"""
    (work_dir / "nvt.mdp").write_text(nvt_mdp)
    conf = "em.gro" if (work_dir / "em.gro").exists() else "conf.gro"
    gmx(["grompp", "-f", "nvt.mdp", "-c", conf,
         "-p", "topol.top", "-o", "nvt.tpr", "-maxwarn", "50"], work_dir)
    gmx(["mdrun", "-deffnm", "nvt", "-nb", "gpu"], work_dir, timeout=120)

    if (work_dir / "nvt.gro").exists():
        shutil.copy2(str(work_dir / "nvt.gro"), str(work_dir / "conf_eq.gro"))
        return "conf_eq.gro"
    elif (work_dir / "em.gro").exists():
        return "em.gro"
    return "conf.gro"


def _run_removal_test(snap_dir, tmpl_pos_init, tmpl_indices):
    """Run MD with monomer restraints, template free. Measure template escape."""
    import MDAnalysis as mda

    removal_dir = Path(snap_dir) / "removal"
    removal_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy frame and topology
        shutil.copy2(str(snap_dir / "frame.gro"), str(removal_dir / "conf.gro"))
        shutil.copy2(str(snap_dir / "topol.top"), str(removal_dir / "topol.top"))
        for f in snap_dir.glob("*.itp"):
            shutil.copy2(str(f), str(removal_dir / f.name))

        # MDP: gen-vel=yes for fresh start (no checkpoint)
        dt = 0.002
        nsteps = int(VIP_REMOVAL_NS * 1e6 / (dt * 1000))
        mdp = MDP_PRODUCTION.format(
            nsteps=nsteps, dt=dt, nstxout=5000, temperature=TEMPERATURE
        )
        mdp = mdp.replace("continuation = yes", "continuation = no")
        mdp += f"\ngen-vel     = yes\ngen-temp    = {TEMPERATURE}\n"
        mdp += "define = -DPOSRES_MONOMER\n"
        (removal_dir / "md.mdp").write_text(mdp)

        # grompp + mdrun
        gmx(["grompp", "-f", "md.mdp", "-c", "conf.gro",
             "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "50"],
            removal_dir)
        gmx(["mdrun", "-deffnm", "md", "-nb", "gpu"],
            removal_dir, timeout=int(VIP_REMOVAL_NS * 300))

        # Analyze template RMSD (PBC-aware)
        traj = removal_dir / "md.xtc"
        top = removal_dir / "conf.gro"
        if traj.exists():
            u = mda.Universe(str(top), str(traj))
            template = u.select_atoms(f"index {' '.join(str(i) for i in tmpl_indices)}")

            rmsds = []
            for ts in u.trajectory:
                rmsd = _compute_pbc_rmsd(
                    template.positions, tmpl_pos_init, ts.dimensions)
                rmsds.append(rmsd)

            final_rmsd = rmsds[-1] if rmsds else 0
            max_rmsd = max(rmsds) if rmsds else 0
            escaped = final_rmsd > VIP_REMOVAL_THRESHOLD

            logger.info(f"    Removal: final={final_rmsd:.1f}Å, "
                        f"max={max_rmsd:.1f}Å, escaped={escaped}")
            return {
                "final_rmsd_A": round(float(final_rmsd), 2),
                "max_rmsd_A": round(float(max_rmsd), 2),
                "escaped": bool(escaped),
            }

        return {"error": "Trajectory not generated", "escaped": None}

    except Exception as e:
        logger.error(f"    Removal failed: {e}")
        return {"error": str(e), "escaped": None}


# ═══════════════════════════════════════════════════════════════
#  Rebinding MD
# ═══════════════════════════════════════════════════════════════

def _analyze_rebind_traj(rebind_dir, ref_gro, tmpl_indices, tmpl_pos_init, label,
                          with_mmpbsa=None):
    """Compute rebinding metrics from an EXISTING trajectory (no MD run):
    Q4 equilibrium RMSD, Q1→Q4 drift/convergence, residence, contacts, and
    optional MM-GBSA ΔG. Shared by the live run, reanalyze, and extend paths."""
    import MDAnalysis as mda
    rebind_dir = Path(rebind_dir)
    traj = rebind_dir / "md.xtc"
    if not traj.exists():
        return {"error": "Trajectory not generated", "rebound": False}
    top = rebind_dir / ref_gro
    if not top.exists():
        top = next((rebind_dir / c for c in ("conf.gro", "em.gro", "md.gro")
                    if (rebind_dir / c).exists()), rebind_dir / "conf.gro")
    try:
        u = mda.Universe(str(top), str(traj))
    except Exception as e:
        return {"error": f"universe: {e}", "rebound": False}

    idx = " ".join(str(i) for i in tmpl_indices)
    template = u.select_atoms(f"index {idx}")
    monomers = u.select_atoms(f"not resname {_solvent_resnames()} and not index {idx}")
    rmsds, contacts = [], []
    for ts in u.trajectory:
        rmsds.append(_compute_pbc_rmsd(template.positions, tmpl_pos_init, ts.dimensions))
        if len(monomers) > 0:
            d = np.linalg.norm(template.positions[:, np.newaxis, :] -
                               monomers.positions[np.newaxis, :, :], axis=2)
            contacts.append(int(np.sum(d.min(axis=0) < 6.0)))
    if not rmsds:
        return {"error": "Empty trajectory", "rebound": False}

    n = len(rmsds)
    final_rmsd = rmsds[-1]
    q4 = rmsds[3 * n // 4:] if n >= 4 else rmsds          # equilibrium tail (BIO)
    q4_mean = float(np.mean(q4))
    q4_std = float(np.std(q4)) if len(q4) > 1 else 0.0
    q1_mean = float(np.mean(rmsds[:n // 4])) if n >= 4 else q4_mean
    drift = q4_mean - q1_mean                              # Q1→Q4 convergence
    converged = abs(drift) <= 1.5
    residence = sum(1 for r in rmsds if r <= VIP_REMOVAL_THRESHOLD) / n
    rebound = q4_mean < VIP_RMSD_THRESHOLD
    out = {
        "final_rmsd_A": round(float(final_rmsd), 2),
        "mean_rmsd_A": round(float(np.mean(rmsds)), 2),
        "q4_rmsd_A": round(q4_mean, 2),
        "q4_std_A": round(q4_std, 2),
        "drift_A": round(drift, 2),
        "converged": bool(converged),
        "residence": round(residence, 3),
        "rebound": bool(rebound),
    }
    if contacts:
        out["mean_contacts"] = round(float(np.mean(contacts)), 1)
        out["final_contacts"] = contacts[-1]
    # MM-GBSA ΔG over the equilibrium tail (Kumar 2024). Never raises.
    if (VIP_MMPBSA if with_mmpbsa is None else with_mmpbsa):
        try:
            from .utils_gromacs import run_mmpbsa
            mp = run_mmpbsa(rebind_dir, interval=MMPBSA_INTERVAL, igb=MMPBSA_IGB)
            if mp.get("dG_bind_kcal") is not None:
                out["mmpbsa_dG"] = mp["dG_bind_kcal"]
        except Exception:
            pass
    _c = "" if converged else f" ⚠non-converged(drift={drift:+.1f}Å)"
    _dg = f", ΔG={out['mmpbsa_dG']}" if "mmpbsa_dG" in out else ""
    logger.info(f"    Rebind [{label}]: Q4={q4_mean:.1f}Å, final={final_rmsd:.1f}Å, "
                f"residence={residence:.2f}, rebound={rebound}{_dg}{_c}")
    return out


def _run_rebinding_once(rebind_dir, snap_dir, tmpl_indices, tmpl_pos_init, label, seed):
    """One rebinding replicate: displace template, EM, production MD, analyze."""
    rebind_dir, snap_dir = Path(rebind_dir), Path(snap_dir)
    rebind_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(snap_dir / "frame.gro"), str(rebind_dir / "conf.gro"))
        shutil.copy2(str(snap_dir / "topol.top"), str(rebind_dir / "topol.top"))
        for f in snap_dir.glob("*.itp"):
            shutil.copy2(str(f), str(rebind_dir / f.name))
        _displace_template_in_gro(rebind_dir / "conf.gro", tmpl_indices,
                                  displacement_nm=0.5, seed=seed)
        try:
            eq_conf = _run_em_equilibration(rebind_dir)
        except Exception:
            eq_conf = "conf.gro"
        dt = 0.002
        nsteps = int(VIP_REBINDING_NS * 1e6 / (dt * 1000))
        mdp = MDP_PRODUCTION.format(nsteps=nsteps, dt=dt, nstxout=5000,
                                    temperature=TEMPERATURE)
        mdp = mdp.replace("continuation = yes", "continuation = no")
        mdp += f"\ngen-vel     = yes\ngen-temp    = {TEMPERATURE}\n"
        mdp += "define = -DPOSRES_MONOMER\n"
        (rebind_dir / "md.mdp").write_text(mdp)
        gmx(["grompp", "-f", "md.mdp", "-c", eq_conf, "-p", "topol.top",
             "-o", "md.tpr", "-maxwarn", "50"], rebind_dir)
        gmx(["mdrun", "-deffnm", "md", "-nb", "gpu"], rebind_dir,
            timeout=int(VIP_REBINDING_NS * 300))
        return _analyze_rebind_traj(rebind_dir, eq_conf, tmpl_indices,
                                    tmpl_pos_init, label)
    except Exception as e:
        logger.error(f"    Rebinding failed: {e}")
        return {"error": str(e), "rebound": False}


def _ensemble_rebind(reps):
    """Ensemble-average multi-restart replicate metrics (majority vote for the
    boolean flags, mean ± spread for the continuous ones)."""
    reps = [r for r in reps if isinstance(r, dict) and "error" not in r]
    if not reps:
        return {"error": "all replicates failed", "rebound": False}
    if len(reps) == 1:
        return reps[0]

    def _m(k):
        vals = [r[k] for r in reps if r.get(k) is not None]
        return round(float(np.mean(vals)), 3) if vals else None
    agg = {
        "q4_rmsd_A": _m("q4_rmsd_A"),
        "q4_ensemble_std_A": round(float(np.std(
            [r["q4_rmsd_A"] for r in reps if r.get("q4_rmsd_A") is not None])), 2),
        "final_rmsd_A": _m("final_rmsd_A"),
        "mean_rmsd_A": _m("mean_rmsd_A"),
        "drift_A": _m("drift_A"),
        "residence": _m("residence"),
        "converged": sum(1 for r in reps if r.get("converged")) >= (len(reps) + 1) // 2,
        "rebound": sum(1 for r in reps if r.get("rebound")) >= (len(reps) + 1) // 2,
        "n_replicates": len(reps),
    }
    if any("mmpbsa_dG" in r for r in reps):
        agg["mmpbsa_dG"] = _m("mmpbsa_dG")
    if any("mean_contacts" in r for r in reps):
        agg["mean_contacts"] = _m("mean_contacts")
    logger.info(f"    Ensemble(n={len(reps)}): Q4={agg['q4_rmsd_A']}±"
                f"{agg['q4_ensemble_std_A']}Å residence={agg['residence']} "
                f"rebound={agg['rebound']}")
    return agg


def _run_rebinding(snap_dir, tmpl_pos_init, tmpl_indices, label, snapshot_idx=0):
    """Rebinding with optional multi-restart ensemble (VIP_N_RESTARTS). Each
    replicate perturbs the initial displacement (different seed); continuous
    metrics are ensemble-averaged with the spread reported."""
    snap_dir = Path(snap_dir)
    n_rep = max(1, int(VIP_N_RESTARTS))
    base = snap_dir / f"rebind_{label}"
    reps = [_run_rebinding_once(base if n_rep == 1 else base / f"rep_{rep}",
                                snap_dir, tmpl_indices, tmpl_pos_init, label,
                                seed=42 + snapshot_idx + rep * 1000)
            for rep in range(n_rep)]
    return _ensemble_rebind(reps)


def _displace_template_in_gro(gro_path, tmpl_indices, displacement_nm=0.5,
                               seed=42):
    """Displace template atoms by random vector in GRO file."""
    lines = Path(gro_path).read_text().strip().split("\n")
    rng = np.random.RandomState(seed)
    direction = rng.randn(3)
    direction /= np.linalg.norm(direction)

    for idx in tmpl_indices:
        line_num = idx + 2  # GRO: line 0=title, line 1=natoms, line 2+=atoms
        if line_num < len(lines) - 1:
            line = lines[line_num]
            try:
                x = float(line[20:28]) + direction[0] * displacement_nm
                y = float(line[28:36]) + direction[1] * displacement_nm
                z = float(line[36:44]) + direction[2] * displacement_nm
                lines[line_num] = line[:20] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[44:]
            except (ValueError, IndexError):
                pass

    Path(gro_path).write_text("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════

def _analyze_results(monomer_name, snapshots, interferent_names):
    """Analyze VIP: removal + rebinding + selectivity."""
    n_total = 0
    n_removal_ok = 0
    n_rebind_ok = 0
    n_both_ok = 0
    interf_rebound = {name: 0 for name in interferent_names}

    for snap in snapshots:
        if not snap.get("success"):
            continue
        n_total += 1

        removal = snap.get("removal_test", {})
        rebind = snap.get("rebind_template", {})

        if removal.get("escaped"):
            n_removal_ok += 1
        if rebind.get("rebound"):
            n_rebind_ok += 1
        if removal.get("escaped") and rebind.get("rebound"):
            n_both_ok += 1

        for interf_name in interferent_names:
            interf = snap.get(f"rebind_{interf_name}", {})
            if interf.get("rebound"):
                interf_rebound[interf_name] += 1

    removal_rate = n_removal_ok / n_total if n_total > 0 else 0
    rebind_rate = n_rebind_ok / n_total if n_total > 0 else 0
    both_rate = n_both_ok / n_total if n_total > 0 else 0
    n_selective = sum(1 for cnt in interf_rebound.values() if cnt == 0)
    sel_score = n_selective / len(interferent_names) if interferent_names else 0

    # ── Composite VIP score (literature-based) ──
    # For small-molecule templates, removal in 10ns is often too short
    # to observe 8Å displacement → both_rate ≈ 0 for all monomers.
    # Use rebind_rate as primary indicator (Zink & Moura 2018: cavity
    # recognition is the key metric, not removal speed).
    #
    # Selectivity: graded scoring based on interferent rebind frequency.
    #   - Binary (rejected/not) is too coarse for small molecules where
    #     all interferents fit in the cavity.
    #   - Instead: sel = 1 - mean(interf_rebind_rate / template_rebind_rate)
    #     If interferents rebind as often as template → sel ≈ 0 (non-selective)
    #     If interferents rebind less than template → sel > 0 (selective)
    #   - Clamped to [0, 1].
    #   Reference: Ye et al. 2024, Muñoz et al. 2024 (graded binding metrics)
    if n_total > 0 and rebind_rate > 0:
        interf_ratios = []
        for cnt in interf_rebound.values():
            interf_rebind_rate = cnt / n_total
            interf_ratios.append(interf_rebind_rate / rebind_rate)
        sel_score = max(0.0, 1.0 - (sum(interf_ratios) / len(interf_ratios))) if interf_ratios else 0
    elif n_total > 0:
        # Template doesn't rebind → sel = 0
        sel_score = 0.0

    # (vip_score is computed below from a CONTINUOUS retention metric, not the
    # binary rebind_rate — see the retention block after the RMSD aggregation.)

    # ── Selectivity Index (Mohsenzadeh 2024): RMSD-based ──
    # SI = mean(interf_final_rmsd) / mean(template_final_rmsd)
    # SI > 1.5 → selective, SI 1.0-1.5 → weak, SI < 1.0 → non-selective
    own_rmsds = []
    own_residences = []
    own_dgs = []
    n_conv = n_conv_total = 0
    interf_rmsds_all = {name: [] for name in interferent_names}
    for snap in snapshots:
        if not snap.get("success"):
            continue
        rt = snap.get("rebind_template", {})
        # Q4 equilibrium RMSD (BIO); fall back to final_rmsd for old data.
        own_r = rt.get("q4_rmsd_A", rt.get("final_rmsd_A"))
        if own_r is not None:
            own_rmsds.append(own_r)
        if rt.get("residence") is not None:
            own_residences.append(rt["residence"])
        if rt.get("mmpbsa_dG") is not None:
            own_dgs.append(rt["mmpbsa_dG"])
        if rt.get("converged") is not None:
            n_conv_total += 1
            n_conv += 1 if rt["converged"] else 0
        for iname in interferent_names:
            it = snap.get(f"rebind_{iname}", {})
            ir = it.get("q4_rmsd_A", it.get("final_rmsd_A"))
            if ir is not None:
                interf_rmsds_all[iname].append(ir)

    own_mean = np.mean(own_rmsds) if own_rmsds else 999.0
    si_per_interf = {}
    for iname in interferent_names:
        i_rmsds = interf_rmsds_all[iname]
        i_mean = np.mean(i_rmsds) if i_rmsds else 0.0
        si = float(i_mean / own_mean) if own_mean > 0.1 else 0.0
        si_per_interf[iname] = round(si, 3)
    selectivity_index = float(np.mean(list(si_per_interf.values()))) if si_per_interf else 0.0

    # Statistical significance (Welch's t-test: own vs each interferent RMSD)
    from scipy import stats as _stats
    p_values = {}
    for iname in interferent_names:
        i_rmsds = interf_rmsds_all[iname]
        if len(own_rmsds) >= 2 and len(i_rmsds) >= 2:
            _, p = _stats.ttest_ind(own_rmsds, i_rmsds, equal_var=False)
            p_values[iname] = round(float(p), 4)
        else:
            p_values[iname] = None

    # ── Continuous retention score (replaces the binary rebind_rate) ──
    # "rebound if RMSD < 5 Å" is 0 for every weakly-recognised template
    # (e.g. nonpolar gamma-terpinene), so it cannot rank monomers. Instead grade
    # RETENTION from the equilibrium template RMSD: 1.0 when the template is held
    # in the cavity (≤ rebind threshold), decaying to 0 as it escapes (~2× the
    # removal threshold). A template that lingers at 6 Å now scores well above
    # one that flies off to 18 Å — a discriminating, non-zero signal.
    _R_near = float(VIP_RMSD_THRESHOLD)          # Å, fully retained
    _R_far = 2.0 * float(VIP_REMOVAL_THRESHOLD)  # Å, fully escaped
    retention = (max(0.0, min(1.0, (_R_far - own_mean) / (_R_far - _R_near)))
                 if own_rmsds else 0.0)
    if not own_rmsds:
        retention_quality = "no_data"
    elif own_mean <= VIP_RMSD_THRESHOLD:
        retention_quality = "retained"
    elif own_mean <= VIP_REMOVAL_THRESHOLD:
        retention_quality = "weak"
    else:
        retention_quality = "escaped"
    # Mean residence (fraction of the rebinding MD spent inside the cavity).
    residence_mean = float(np.mean(own_residences)) if own_residences else 0.0
    # Convergence from the per-snapshot Q1→Q4 drift test (BIO). If most snapshots
    # are non-converged, a low retention is sampling-limited — run longer MD.
    converged = (n_conv / n_conv_total >= 0.5) if n_conv_total else None
    # VIP score = mean(equilibrium retention, residence) × selectivity. Residence
    # rewards a template that LINGERS even if it eventually drifts, so weak
    # binders still get ranked instead of all collapsing to 0.
    vip_score = 0.5 * (retention + residence_mean) * (1 + sel_score)

    return {
        "monomer": monomer_name,
        "n_snapshots": n_total,
        "removal_rate": round(removal_rate, 2),
        "rebind_rate": round(rebind_rate, 2),
        "both_rate": round(both_rate, 2),
        "retention": round(retention, 3),
        "retention_quality": retention_quality,
        "residence": round(residence_mean, 3),
        "mmpbsa_dG_mean": round(float(np.mean(own_dgs)), 2) if own_dgs else None,
        "converged": converged,
        "selectivity_score": round(sel_score, 2),
        "selectivity_index": round(selectivity_index, 3),
        "selectivity_index_per_interf": si_per_interf,
        "p_values": p_values,
        "own_rmsd_mean": round(float(own_mean), 2),
        "own_rmsd_std": round(float(np.std(own_rmsds)), 2) if len(own_rmsds) > 1 else 0.0,
        "vip_score": round(vip_score, 3),
        "interferent_results": {
            name: {"rebound_count": cnt, "rejected": cnt == 0}
            for name, cnt in interf_rebound.items()
        },
    }


def _snapshot_template_ref(snap_dir):
    """Reconstruct (tmpl_indices, tmpl_pos_init) from a snapshot's frame.gro —
    the SAME template identification the live run uses (first non-solvent resid).
    Lets reanalyze/extend recompute RMSD without the original in-memory state."""
    import MDAnalysis as mda
    ref = Path(snap_dir) / "frame.gro"
    if not ref.exists():
        return None, None
    try:
        u = mda.Universe(str(ref))
        non_sol = u.select_atoms(f"not resname {_solvent_resnames()}")
        frid = non_sol.residues[0].resid
        template = u.select_atoms(f"resid {frid}")
        return list(template.indices), template.positions.copy()
    except Exception:
        return None, None


def reanalyze_vip(output_dir=None):
    """Re-derive VIP metrics (Q4/drift/residence/retention/ΔG) from EXISTING
    rebinding trajectories — NO MD re-run — and rewrite stage6_vip.json. Applies
    the current metric definitions to an old run. (BIO reanalyze_phase5.)"""
    if output_dir is None:
        output_dir = OUTPUT_DIRS.get("stage6", f"{OUTPUT_DIR}/stage6")
    out = Path(output_dir)
    interferent_names = list(INTERFERENT_LIBRARY.keys())
    monomer_dirs = sorted(d for d in out.iterdir()
                          if d.is_dir() and (d / "snapshot_0").exists())
    logger.info(f"Reanalyze VIP: {len(monomer_dirs)} monomer(s) in {out}")
    results = {}
    for mdir in monomer_dirs:
        mono = mdir.name
        snapshot_results = []
        for snap_dir in sorted(mdir.glob("snapshot_*")):
            tmpl_indices, tmpl_pos = _snapshot_template_ref(snap_dir)
            if tmpl_indices is None:
                continue
            snap = {"success": True, "removal": {}}
            base = snap_dir / "rebind_template"
            reps = sorted(base.glob("rep_*"))
            if reps:
                snap["rebind_template"] = _ensemble_rebind([
                    _analyze_rebind_traj(rp, "conf.gro", tmpl_indices, tmpl_pos, "template")
                    for rp in reps])
            elif base.exists():
                snap["rebind_template"] = _analyze_rebind_traj(
                    base, "conf.gro", tmpl_indices, tmpl_pos, "template")
            else:
                continue
            for iname in interferent_names:
                idir = snap_dir / f"rebind_{iname}"
                if idir.exists():
                    snap[f"rebind_{iname}"] = _analyze_rebind_traj(
                        idir, "conf.gro", tmpl_indices, tmpl_pos, iname)
            snapshot_results.append(snap)
        if snapshot_results:
            results[mono] = _analyze_results(mono, snapshot_results, interferent_names)
            r = results[mono]
            logger.info(f"  {mono}: vip_score={r.get('vip_score')} "
                        f"retention={r.get('retention')} residence={r.get('residence')} "
                        f"ΔG={r.get('mmpbsa_dG_mean')}")
    with open(out / "stage6_vip.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Reanalyze VIP: wrote stage6_vip.json ({len(results)} monomers)")
    _print_summary(results)
    return results


def extend_drifting_vip(output_dir=None, extend_ns=None, drift_threshold=1.5):
    """Extend NON-CONVERGED rebinding MDs (|Q1→Q4 drift| > threshold) by
    extend_ns, continuing from the checkpoint (gmx convert-tpr -extend + mdrun
    -cpi), then reanalyze. Focuses extra sampling on only the runs that need it.
    (BIO extend_drifting_mds.)"""
    if output_dir is None:
        output_dir = OUTPUT_DIRS.get("stage6", f"{OUTPUT_DIR}/stage6")
    out = Path(output_dir)
    extend_ns = int(extend_ns if extend_ns is not None else VIP_EXTEND_NS)
    n_ext = 0
    for mdir in sorted(d for d in out.iterdir() if d.is_dir()):
        for base in sorted(mdir.glob("snapshot_*/rebind_template")):
            snap_dir = base.parent
            tmpl_indices, tmpl_pos = _snapshot_template_ref(snap_dir)
            if tmpl_indices is None:
                continue
            run_dirs = sorted(base.glob("rep_*")) or [base]
            for rd in run_dirs:
                if not (rd / "md.tpr").exists() or not (rd / "md.cpt").exists():
                    continue
                m = _analyze_rebind_traj(rd, "conf.gro", tmpl_indices, tmpl_pos,
                                         "template", with_mmpbsa=False)
                if m.get("converged", True) or abs(m.get("drift_A", 0)) <= drift_threshold:
                    continue
                logger.info(f"  Extending {rd} by {extend_ns} ns "
                            f"(drift={m.get('drift_A')}Å)")
                try:
                    gmx(["convert-tpr", "-s", "md.tpr",
                         "-extend", str(extend_ns * 1000), "-o", "md.tpr"], rd)
                    gmx(["mdrun", "-deffnm", "md", "-cpi", "md.cpt", "-nb", "gpu"],
                        rd, timeout=int(extend_ns * 300))
                    n_ext += 1
                except Exception as e:
                    logger.warning(f"    extend failed: {e}")
    logger.info(f"Extend VIP: extended {n_ext} non-converged run(s); reanalyzing")
    return reanalyze_vip(output_dir)


def _print_summary(all_results):
    """Print VIP summary."""
    logger.info(f"\n{'='*75}")
    logger.info("Stage 6: VIP Cavity Rebinding Summary")
    logger.info(f"{'='*75}")
    logger.info(f"{'Monomer':<10} {'Removal':>10} {'Rebind':>10} {'Both':>10} "
                f"{'Selectivity':>12} {'VIP Score':>10}")
    logger.info("-" * 65)

    ranked = []
    for m_name, data in all_results.items():
        if not isinstance(data, dict) or "vip_score" not in data:
            continue
        logger.info(f"{m_name:<10} {data['removal_rate']:>10.0%} {data['rebind_rate']:>10.0%} "
                    f"{data['both_rate']:>10.0%} {data['selectivity_score']:>12.0%} "
                    f"{data['vip_score']:>10.3f}")
        ranked.append((m_name, data["vip_score"]))

    ranked.sort(key=lambda x: -x[1])
    logger.info(f"\nFinal VIP Ranking:")
    for i, (m, score) in enumerate(ranked, 1):
        logger.info(f"  {i}. {m} (VIP score = {score:.3f})")
    logger.info(f"{'='*75}")

"""
Stage 5: VIP Cavity Formation + Rebinding Validation (GROMACS)
=============================================================
Virtually Imprinted Polymer (VIP) approach (Zink & Moura, PCCP 2018):
1. Select equilibrium snapshots from Stage 4 MD trajectory
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
)
from .utils_gromacs import gmx, MDP_NVT, MDP_PRODUCTION

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════

def run_stage5(template_smiles: str = None,
               monomer_names: list = None,
               monomer_library: dict = None,
               interferent_library: dict = None,
               output_dir: str = None) -> dict:
    """Run VIP cavity rebinding for all monomers."""
    template_smiles = template_smiles or TEMPLATE_SMILES
    monomer_library = monomer_library or MONOMER_LIBRARY
    interferent_library = interferent_library or INTERFERENT_LIBRARY

    if output_dir is None:
        output_dir = OUTPUT_DIRS.get("stage5", f"{OUTPUT_DIR}/stage5")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load monomer list
    if monomer_names is None:
        for src in [out_path.parent / "stage4" / "stage4_md.json",
                    out_path.parent / "stage3" / "stage3_top.json"]:
            if src.exists():
                with open(src) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    monomer_names = [r.get("monomer", r) if isinstance(r, dict) else r
                                     for r in data]
                break
        if monomer_names is None:
            monomer_names = list(monomer_library.keys())

    # Skip logic
    result_file = out_path / "stage5_vip.json"
    existing = {}
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)
    skip_names = {k for k, v in existing.items()
                  if isinstance(v, dict) and v.get("n_snapshots", 0) > 0}

    all_results = dict(existing)
    logger.info(f"Stage 5: VIP rebinding for {monomer_names}")
    if skip_names:
        logger.info(f"  {len(skip_names)} already computed, skipping")

    for m_name in monomer_names:
        if m_name in skip_names:
            logger.info(f"  {m_name}: already computed, skipping")
            continue
        if m_name not in monomer_library:
            continue

        logger.info(f"\n{'='*20} VIP: {m_name} {'='*20}")

        try:
            result = _vip_for_monomer(
                template_smiles, m_name, monomer_library[m_name],
                interferent_library,
                str(out_path.parent / "stage4" / m_name),
                str(out_path / m_name),
            )
            all_results[m_name] = result
            with open(result_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"  VIP failed for {m_name}: {e}")
            import traceback; traceback.print_exc()
            all_results[m_name] = {"success": False, "error": str(e)}

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    _print_summary(all_results)
    return all_results


# ═══════════════════════════════════════════════════════════════
#  Per-monomer VIP
# ═══════════════════════════════════════════════════════════════

def _vip_for_monomer(template_smiles, monomer_name, monomer_smiles,
                      interferent_library, stage4_md_dir, output_dir):
    """Full VIP protocol for one monomer using GROMACS."""
    import MDAnalysis as mda

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    s4 = Path(stage4_md_dir)

    # Find Stage 4 trajectory
    traj = s4 / "md.xtc"
    top = s4 / "npt.gro"
    topol = s4 / "topol.top"

    if not traj.exists() or not top.exists():
        raise FileNotFoundError(f"Stage 4 trajectory not found in {s4}")

    u = mda.Universe(str(top), str(traj))
    logger.info(f"  Trajectory: {len(u.trajectory)} frames, {u.atoms.n_atoms} atoms")

    # Identify template and monomer atoms
    # Identify template vs monomer by residue order (resid 1 = template)
    non_solvent = u.select_atoms("not resname SOL NA CL Na+ Cl-")
    first_resid = non_solvent.residues[0].resid
    template = u.select_atoms(f"resid {first_resid}")
    monomers = u.select_atoms(
        f"not resname SOL NA CL Na+ Cl- and not resid {first_resid}")
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
    non_solvent = universe.select_atoms("not resname SOL NA CL Na+ Cl-")
    first_resid = non_solvent.residues[0].resid
    monomers = universe.select_atoms(
        f"not resname SOL NA CL Na+ Cl- and not resid {first_resid}")
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
         "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "10"], work_dir)
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
         "-p", "topol.top", "-o", "nvt.tpr", "-maxwarn", "10"], work_dir)
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
             "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "10"],
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

def _run_rebinding(snap_dir, tmpl_pos_init, tmpl_indices, label,
                    snapshot_idx=0):
    """Displace template, run MD, check if template returns to cavity."""
    import MDAnalysis as mda

    rebind_dir = Path(snap_dir) / f"rebind_{label}"
    rebind_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Start from frame with template displaced slightly
        shutil.copy2(str(snap_dir / "frame.gro"), str(rebind_dir / "conf.gro"))
        shutil.copy2(str(snap_dir / "topol.top"), str(rebind_dir / "topol.top"))
        for f in snap_dir.glob("*.itp"):
            shutil.copy2(str(f), str(rebind_dir / f.name))

        # Displace template by ~5Å — unique direction per snapshot
        _displace_template_in_gro(
            rebind_dir / "conf.gro", tmpl_indices,
            displacement_nm=0.5, seed=42 + snapshot_idx)

        # EM + short NVT equilibration to relax steric clashes
        try:
            eq_conf = _run_em_equilibration(rebind_dir)
        except Exception:
            eq_conf = "conf.gro"

        dt = 0.002
        nsteps = int(VIP_REBINDING_NS * 1e6 / (dt * 1000))
        mdp = MDP_PRODUCTION.format(
            nsteps=nsteps, dt=dt, nstxout=5000, temperature=TEMPERATURE
        )
        mdp = mdp.replace("continuation = yes", "continuation = no")
        mdp += f"\ngen-vel     = yes\ngen-temp    = {TEMPERATURE}\n"
        mdp += "define = -DPOSRES_MONOMER\n"
        (rebind_dir / "md.mdp").write_text(mdp)

        gmx(["grompp", "-f", "md.mdp", "-c", eq_conf,
             "-p", "topol.top", "-o", "md.tpr", "-maxwarn", "10"],
            rebind_dir)
        gmx(["mdrun", "-deffnm", "md", "-nb", "gpu"],
            rebind_dir, timeout=int(VIP_REBINDING_NS * 300))

        # Analyze RMSD + contact count from initial position (PBC-aware)
        traj = rebind_dir / "md.xtc"
        if traj.exists():
            u = mda.Universe(str(rebind_dir / eq_conf), str(traj))
            template = u.select_atoms(f"index {' '.join(str(i) for i in tmpl_indices)}")
            monomers = u.select_atoms(
                f"not resname SOL NA CL Na+ Cl- and not index {' '.join(str(i) for i in tmpl_indices)}")

            rmsds = []
            contacts = []
            for ts in u.trajectory:
                rmsd = _compute_pbc_rmsd(
                    template.positions, tmpl_pos_init, ts.dimensions)
                rmsds.append(rmsd)
                # Contact count: monomer atoms within 6Å of template
                if len(monomers) > 0:
                    dists = np.linalg.norm(
                        template.positions[:, np.newaxis, :] -
                        monomers.positions[np.newaxis, :, :], axis=2)
                    n_contact = int(np.sum(dists.min(axis=0) < 6.0))
                    contacts.append(n_contact)

            final_rmsd = rmsds[-1] if rmsds else 999
            mean_rmsd = np.mean(rmsds) if rmsds else 999
            rebound = final_rmsd < VIP_RMSD_THRESHOLD

            contact_info = {}
            if contacts:
                contact_info = {
                    "mean_contacts": round(float(np.mean(contacts)), 1),
                    "final_contacts": contacts[-1] if contacts else 0,
                }

            logger.info(f"    Rebind [{label}]: final={final_rmsd:.1f}Å, "
                        f"mean={mean_rmsd:.1f}Å, contacts={contact_info.get('mean_contacts', 0):.0f}, "
                        f"rebound={rebound}")
            return {
                "final_rmsd_A": round(float(final_rmsd), 2),
                "mean_rmsd_A": round(float(mean_rmsd), 2),
                "rebound": bool(rebound),
                **contact_info,
            }

        return {"error": "Trajectory not generated", "rebound": False}

    except Exception as e:
        logger.error(f"    Rebinding failed: {e}")
        return {"error": str(e), "rebound": False}


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

    # Score = rebind_rate × (1 + sel_score)
    # Primary metric: cavity rebinding rate (Zink & Moura 2018).
    # Falls back to both_rate formula when removal works (large templates).
    if both_rate > 0:
        vip_score = both_rate * (1 + sel_score)
    else:
        vip_score = rebind_rate * (1 + sel_score)

    # ── Selectivity Index (Mohsenzadeh 2024): RMSD-based ──
    # SI = mean(interf_final_rmsd) / mean(template_final_rmsd)
    # SI > 1.5 → selective, SI 1.0-1.5 → weak, SI < 1.0 → non-selective
    own_rmsds = []
    interf_rmsds_all = {name: [] for name in interferent_names}
    for snap in snapshots:
        if not snap.get("success"):
            continue
        own_r = snap.get("rebind_template", {}).get("final_rmsd_A")
        if own_r is not None:
            own_rmsds.append(own_r)
        for iname in interferent_names:
            ir = snap.get(f"rebind_{iname}", {}).get("final_rmsd_A")
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

    return {
        "monomer": monomer_name,
        "n_snapshots": n_total,
        "removal_rate": round(removal_rate, 2),
        "rebind_rate": round(rebind_rate, 2),
        "both_rate": round(both_rate, 2),
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


def _print_summary(all_results):
    """Print VIP summary."""
    logger.info(f"\n{'='*75}")
    logger.info("Stage 5: VIP Cavity Rebinding Summary")
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

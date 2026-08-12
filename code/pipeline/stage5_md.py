"""
Stage 5: Pre-polymerization MD Simulation (GROMACS)
===================================================
Simulates template + monomers in explicit solvent to evaluate
dynamic binding behavior before polymerization.

Analysis:
- Contact frequency (6Å cutoff)
- RDF (Radial Distribution Function)
- EBN (Effective Binding Number)
- Mean minimum distance
- Synthesis ratio recommendation (contact freq inverse)

Adapted from Monomer_screening_in_Bio/phase4_md_validation.py
Uses GROMACS (system install) + MDAnalysis for analysis.

Ref: Muñoz et al., J. Chem. Inf. Model. 2024 (MD-based monomer selection)
     Ye et al., Molecules 2024 (EBN/HBNmax parameters)
"""

import json
import logging
import numpy as np
from pathlib import Path

from .config import (
    TEMPLATE_SMILES, TEMPLATE_NAME, MONOMER_LIBRARY,
    TEMPERATURE, OUTPUT_DIR, OUTPUT_DIRS,
    MD_RATIO_SCREENING, MD_RATIOS_TO_TEST, MD_TEMPLATE_MONOMER_RATIO,
    MD_TIME_NS, MD_CONTACT_CUTOFF, MD_BOX_SIZE,
    MD_INCLUDE_CROSSLINKER, MD_CROSSLINKER_RATIO, MD_MULTI_MONOMER,
    CROSSLINKER_LIBRARY, MMSD_MD_TOP_N, MD_COMBO_N_PER_MONOMER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage5] %(message)s")
logger = logging.getLogger(__name__)


def _classify_poly(smi):
    """vinyl / silane / oxidative from SMILES (mirrors stage4/stage7)."""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return "unknown"
        if m.HasSubstructMatch(Chem.MolFromSmarts("[Si]")):
            return "silane"
        if m.HasSubstructMatch(Chem.MolFromSmarts("[C;!a]=[C;!a]")):
            return "vinyl"
        return "oxidative"
    except Exception:
        return "unknown"


def _compatible_crosslinker(m_smiles):
    """Chemistry-matched crosslinker for a monomer's MD box: silane monomer →
    silane crosslinker (TEOS/TMOS), vinyl → vinyl (EGDMA), oxidative → none.
    Using EGDMA for every monomer put a vinyl crosslinker in silane/oxidative
    boxes (a non-physical matrix). Returns (name, smiles, n_copies)."""
    need = _classify_poly(m_smiles)
    if need == "oxidative":
        return None, None, 0
    for name, smi in CROSSLINKER_LIBRARY.items():
        if _classify_poly(smi) == need:
            return name, smi, MD_CROSSLINKER_RATIO
    n0 = next(iter(CROSSLINKER_LIBRARY.items()))  # fallback: first in library
    return n0[0], n0[1], MD_CROSSLINKER_RATIO


def run_stage5(template_smiles: str = None,
               monomer_names: list = None,
               monomer_library: dict = None,
               output_dir: str = None) -> dict:
    """Run GROMACS pre-polymerization MD for all monomers.

    Returns {monomer: {contact_freq, EBN, mean_dist, ...}}
    """
    from .utils_gromacs import (build_mip_system, run_md_pipeline, analyze_md,
                                check_md_toolchain)

    # Preflight: acpype needs AmberTools (antechamber/sqm) reachable. If it is
    # not, EVERY parameterization fails and the stage produces zero results —
    # report that loudly up front rather than after 28 silent failures.
    tools = check_md_toolchain()
    missing = [t for t in ("antechamber", "sqm") if not tools.get(t)]
    if missing:
        logger.error(f"  ⚠ MD toolchain incomplete — missing {missing} on PATH. "
                     f"acpype cannot assign GAFF2 charges; every monomer will "
                     f"fail. Fix: conda install -c conda-forge ambertools")
    else:
        logger.info(f"  MD toolchain OK: antechamber={tools['antechamber']}")

    template_smiles = template_smiles or TEMPLATE_SMILES
    template_name = TEMPLATE_NAME if hasattr(__import__('pipeline.config', fromlist=['TEMPLATE_NAME']), 'TEMPLATE_NAME') else "TMP"
    monomer_library = monomer_library or MONOMER_LIBRARY

    if output_dir is None:
        output_dir = OUTPUT_DIRS.get("stage5", f"{OUTPUT_DIR}/stage5")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ══ Combination-centric Stage 5 ══════════════════════════════════════
    # The imprinted cavity is the monomer COMBINATION + crosslinker, so Stage 5
    # runs MD ONLY on Stage 4's top combinations — there is no per-monomer single
    # MD. Each combination is simulated as MD_COMBO_N_REPLICAS independent boxes
    # (different placement); per-monomer EBN / contact are AVERAGED over the
    # replicas (robust, and avoids the over-crowding that inflates EBN).
    from . import config as _cfg
    md_top_n = getattr(_cfg, "MMSD_MD_TOP_N", 3)
    n_reps = max(1, getattr(_cfg, "MD_COMBO_N_REPLICAS", 3))
    n_per = MD_COMBO_N_PER_MONOMER

    # Stage 4 combinations (or run MMSD here as a fallback).
    mmsd_json = out_path.parent / "stage4" / "mmsd_results.json"
    mres = None
    if mmsd_json.exists():
        try:
            mres = json.loads(mmsd_json.read_text())
        except Exception:
            mres = None
    if mres is None and getattr(_cfg, "MMSD_ENABLE", True):
        try:
            from .stage4_mmsd import run_mmsd
            mres = run_mmsd(template_smiles=template_smiles,
                            output_dir=str(out_path.parent / "stage4"))
        except Exception as e:
            logger.warning(f"  MMSD search failed ({e})")

    combos = []
    for pc in ((mres or {}).get("top_pcs") or [])[:md_top_n]:
        sel = [m for m in pc.get("functional_monomers", []) if m in monomer_library]
        if sel:
            combos.append({"selected": sel,
                           "score": pc.get("bo_objective") or 0.0,
                           "method": f"MMSD ({(mres or {}).get('optimizer')})",
                           "crosslinker": pc.get("crosslinker"),
                           "mmsd": pc})
    result_file = out_path / "stage5_combination.json"
    if not combos:
        logger.error("Stage 5: no Stage-4 combinations to validate — run Stage 4 first")
        with open(result_file, "w") as f:
            json.dump([], f)
        return []

    # Resume: reuse a combination already fully computed.
    existing = []
    if result_file.exists():
        try:
            existing = json.load(open(result_file))
        except Exception:
            existing = []
    done = {tuple(sorted(c.get("selected", []))): c for c in existing
            if isinstance(c, dict) and c.get("md_analysis")}

    from .utils_gromacs import (build_multi_monomer_system, run_md_pipeline,
                                analyze_md)
    logger.info(f"Stage 5: combination MD — {len(combos)} combination(s) × "
                f"{n_reps} replica(s), {n_per} copies/monomer type")

    for rank, combo in enumerate(combos, 1):
        combo["rank"] = rank
        key = tuple(sorted(combo["selected"]))
        pc_dir = out_path / f"multi_monomer_pc{rank}"
        traj0 = (pc_dir / "rep0" / "md.xtc") if n_reps > 1 else (pc_dir / "md.xtc")
        if key in done and traj0.exists():
            for k in ("md_analysis", "resname_map", "mmpbsa", "replicas", "ebn_censored"):
                if k in done[key]:
                    combo[k] = done[key][k]
            logger.info(f"  [PC{rank}] {combo['selected']}: already computed, skipping")
            continue

        combo_smiles = {n: monomer_library[n] for n in combo["selected"]
                        if n in monomer_library}
        combo_box = MD_BOX_SIZE + 1.0 + 0.5 * max(0, len(combo_smiles) - 2)
        # Crosslinker: MMSD sets crosslinker=None for self-crosslinking (oxidative)
        # systems — respect that (no EGDMA in an incompatible matrix); only fall
        # back to a default when a NAME was given that isn't in the library.
        _xl = combo.get("crosslinker")
        if MD_INCLUDE_CROSSLINKER and _xl is not None:
            xl_name = (_xl if _xl in CROSSLINKER_LIBRARY
                       else list(CROSSLINKER_LIBRARY.keys())[0])
            xl_smiles = CROSSLINKER_LIBRARY[xl_name]
            n_xl = MD_CROSSLINKER_RATIO
        else:
            xl_name = xl_smiles = None
            n_xl = 0
        logger.info(f"  [PC{rank}] {combo['selected']} + {xl_name or 'no'} xl "
                    f"(score={combo['score']:.3f})")

        rep_analyses, rmap = [], {}
        for rep in range(n_reps):
            rep_dir = pc_dir if n_reps == 1 else pc_dir / f"rep{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            try:
                sys_info = build_multi_monomer_system(
                    template_smiles, "TMP", combo_smiles,
                    n_per_monomer=n_per, work_dir=rep_dir / "build",
                    box_size=combo_box, crosslinker_smiles=xl_smiles,
                    crosslinker_name=xl_name, n_crosslinker=n_xl, seed=42 + rep)
                if "error" in sys_info:
                    logger.warning(f"  [PC{rank} rep{rep}] build failed: "
                                   f"{sys_info['error']}")
                    continue
                import shutil as _shutil
                for f in (rep_dir / "build").glob("*"):
                    if f.is_file():
                        _shutil.copy2(str(f), str(rep_dir / f.name))
                run_md_pipeline(rep_dir, time_ns=MD_TIME_NS, temperature=TEMPERATURE)
                a = analyze_md(rep_dir, template_name="TMP", cutoff_A=MD_CONTACT_CUTOFF)
                rep_analyses.append(a)
                rmap = sys_info.get("resname_map", rmap)
                logger.info(f"  [PC{rank} rep{rep}] contact="
                            f"{a.get('contact_frequency', 0):.4f}")
                if getattr(_cfg, "MD_MMPBSA", False) and rep == 0:
                    from .utils_gromacs import run_mmpbsa
                    combo["mmpbsa"] = run_mmpbsa(
                        rep_dir, interval=getattr(_cfg, "MMPBSA_INTERVAL", 10),
                        igb=getattr(_cfg, "MMPBSA_IGB", 5))
            except Exception as e:
                logger.warning(f"  [PC{rank} rep{rep}] MD failed: {e}")

        if not rep_analyses:
            combo["md_error"] = "all replicas failed"
        else:
            agg, censored = _aggregate_replicas(rep_analyses, n_per)
            combo["md_analysis"] = agg
            combo["resname_map"] = rmap
            combo["replicas"] = len(rep_analyses)
            combo["ebn_censored"] = censored
            if censored:
                logger.warning(f"  [PC{rank}] EBN hit the copy cap ({n_per}) for "
                               f"{censored} — raise MD_COMBO_N_PER_MONOMER.")
            logger.info(f"  [PC{rank}] {len(rep_analyses)} replica(s) → contact="
                        f"{agg.get('contact_frequency', 0):.4f}")
        with open(result_file, "w") as f:
            json.dump(combos, f, indent=2, default=str)

    combos.sort(key=lambda c: c.get("md_analysis", {}).get("contact_frequency", -1.0),
                reverse=True)
    with open(result_file, "w") as f:
        json.dump(combos, f, indent=2, default=str)
    logger.info(f"  Saved {len(combos)} MD-validated combination(s) "
                f"→ stage5_combination.json")
    return combos


def _aggregate_replicas(analyses, n_per):
    """Average key metrics across independent replicas of a combination MD.
    Returns (aggregated_analysis, censored_resnames). EBN censoring = a monomer
    type whose MEAN EBN reaches the per-type copy cap n_per (the measurement is
    supply-limited → the true EBN may be higher; raise MD_COMBO_N_PER_MONOMER)."""
    import numpy as _np
    agg = dict(analyses[0])                       # keep rdf/hbond/etc. from rep 0
    for k in ("contact_frequency", "EBN", "ebn_max_simultaneous",
              "mean_min_distance_A", "n_hbonds_mean"):
        vals = [a.get(k) for a in analyses if isinstance(a.get(k), (int, float))]
        if vals:
            agg[k] = round(float(_np.mean(vals)), 4)

    per_keys = set()
    for a in analyses:
        per_keys |= set((a.get("per_monomer") or {}).keys())
    per, censored = {}, []
    for rn in per_keys:
        cfs = [(a.get("per_monomer") or {}).get(rn, {}).get("contact_frequency")
               for a in analyses]
        ebns = [(a.get("per_monomer") or {}).get(rn, {}).get("ebn_max")
                for a in analyses]
        cfs = [x for x in cfs if isinstance(x, (int, float))]
        ebns = [x for x in ebns if isinstance(x, (int, float))]
        mean_ebn = round(float(_np.mean(ebns)), 3) if ebns else 0
        per[rn] = {
            "contact_frequency": round(float(_np.mean(cfs)), 4) if cfs else 0.0,
            "ebn_max": mean_ebn,
            "ebn_replicas": ebns,
        }
        if ebns and mean_ebn >= n_per - 0.5:      # reached the supply cap
            censored.append(rn)
    agg["per_monomer"] = per
    agg["n_replicas"] = len(analyses)
    return agg, censored


def _optimize_combination(all_results, monomer_library, out_path):
    """Find optimal monomer combination based on Stage 5 MD metrics.

    Uses Stage 2 binding energy + Stage 5 contact frequency/EBN.
    Greedy forward selection with binding site diversity bonus.
    """
    from .config import MD_MULTI_MONOMER_TOP_N
    import json

    n_select = MD_MULTI_MONOMER_TOP_N

    # Load Stage 2 binding energies
    s2_path = out_path.parent / "stage2" / "stage2_dft.json"
    s2 = {}
    if s2_path.exists():
        with open(s2_path) as f:
            s2_data = json.load(f)
        for m, solvents in s2_data.items():
            for solv, vals in solvents.items():
                if isinstance(vals, dict) and "bsse_dE" in vals:
                    s2[m] = abs(vals["bsse_dE"])
                    break

    # Load Stage 1 binding sites
    s1_path = out_path.parent / "stage1" / "stage1_all.json"
    sites = {}
    if s1_path.exists():
        with open(s1_path) as f:
            for entry in json.load(f):
                sites[entry.get("name")] = entry.get("binding_site", {}).get("atom_idx")

    # Build per-monomer score from Stage 5 results
    mono_data = {}
    for r in all_results:
        m = r["monomer"]
        mono_data[m] = {
            "contact": r.get("contact_frequency", 0),
            "ebn": r.get("EBN", 0),
            "hbonds": r.get("n_hbonds_mean", 0),
            "be": s2.get(m, 10),
            "site": sites.get(m),
        }

    ranked = sorted(mono_data.keys(),
                    key=lambda m: mono_data[m]["contact"], reverse=True)
    if len(ranked) <= n_select:
        return {"selected": ranked, "score": 0, "method": "all_monomers"}

    # Greedy forward selection
    be_median = np.median([d["be"] for d in mono_data.values()])
    selected = []
    remaining = list(ranked)

    for step in range(n_select):
        best_m, best_score = None, -float("inf")
        for c in remaining:
            trial = selected + [c]
            # Weighted score: contact + EBN + inverted-U BE
            score = sum(
                mono_data[m]["contact"] * 2 +
                min(mono_data[m]["ebn"] / 100, 1) +
                np.exp(-0.5 * ((mono_data[m]["be"] - be_median) / 3)**2)
                for m in trial
            )
            # Site diversity bonus
            trial_sites = [mono_data[m]["site"] for m in trial if mono_data[m]["site"] is not None]
            diversity = len(set(trial_sites)) / len(trial) if trial_sites else 1.0
            score *= (1 + 0.5 * diversity)
            if score > best_score:
                best_score, best_m = score, c
        if best_m:
            selected.append(best_m)
            remaining.remove(best_m)

    return {
        "selected": selected,
        "score": round(best_score, 3),
        "method": "greedy_forward_selection",
        "metrics": {m: mono_data[m] for m in selected},
    }


def _recommend_ratios(results):
    """Recommend synthesis ratios. Default EBN-based (Yuan 2024): more
    simultaneous binding sites → more copies; optional contact-freq inverse."""
    from .config import MD_RATIO_METHOD
    if MD_RATIO_METHOD == "ebn":
        ebn = {r["monomer"]: r.get("ebn_max", 0)
               for r in results if r.get("ebn_max", 0) > 0}
        if ebn:
            min_e = min(ebn.values())
            ratios = {m: max(1, round(v / min_e)) for m, v in ebn.items()}
            logger.info("\n  Recommended synthesis ratios (EBN direct, Yuan 2024):")
            for m, ratio in sorted(ratios.items(), key=lambda x: -x[1]):
                logger.info(f"    {m}: 1:{ratio} (EBN={ebn[m]})")
            return
    contacts = {r["monomer"]: r["contact_frequency"]
                for r in results if r.get("contact_frequency", 0) > 0}
    if not contacts:
        return
    max_contact = max(contacts.values())
    ratios = {m: round(max_contact / c, 1) for m, c in contacts.items()}
    logger.info("\n  Recommended synthesis ratios (contact freq inverse):")
    for m, ratio in sorted(ratios.items(), key=lambda x: -x[1]):
        logger.info(f"    {m}: 1:{ratio:.1f} (contact_freq={contacts[m]:.4f})")


def _print_summary(results):
    """Print Stage 5 summary."""
    logger.info(f"\n{'='*60}")
    logger.info("Stage 5: Pre-polymerization MD Summary")
    logger.info(f"{'='*60}")
    logger.info(f"{'Monomer':<10} {'Contact Freq':>12} {'EBN':>8} {'Mean Dist(Å)':>12}")
    logger.info("-" * 45)

    for r in sorted(results, key=lambda x: -x.get("contact_frequency", 0)):
        m = r.get("monomer", "?")
        cf = r.get("contact_frequency", 0)
        ebn = r.get("EBN", 0)
        md = r.get("mean_min_distance_A", "N/A")
        logger.info(f"{m:<10} {cf:>12.4f} {ebn:>8.4f} {md!s:>12}")

    logger.info(f"{'='*60}")


if __name__ == "__main__":
    run_stage5()

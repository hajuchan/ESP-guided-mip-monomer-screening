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
    MD_INCLUDE_CROSSLINKER, MD_CROSSLINKER_RATIO,
    CROSSLINKER_LIBRARY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage5] %(message)s")
logger = logging.getLogger(__name__)


def run_stage5(template_smiles: str = None,
               monomer_names: list = None,
               monomer_library: dict = None,
               output_dir: str = None) -> dict:
    """Run GROMACS pre-polymerization MD for all monomers.

    Returns {monomer: {contact_freq, EBN, mean_dist, ...}}
    """
    from .utils_gromacs import build_mip_system, run_md_pipeline, analyze_md

    template_smiles = template_smiles or TEMPLATE_SMILES
    template_name = TEMPLATE_NAME if hasattr(__import__('pipeline.config', fromlist=['TEMPLATE_NAME']), 'TEMPLATE_NAME') else "TMP"
    monomer_library = monomer_library or MONOMER_LIBRARY

    if output_dir is None:
        output_dir = OUTPUT_DIRS.get("stage5", f"{OUTPUT_DIR}/stage5")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load monomer list
    if monomer_names is None:
        for src in [out_path.parent / "stage3" / "stage3_top.json",
                    out_path.parent / "stage1" / "stage1_top.json"]:
            if src.exists():
                with open(src) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    monomer_names = [r.get("name", r) if isinstance(r, dict) else r
                                     for r in data]
                break
        if monomer_names is None:
            monomer_names = list(monomer_library.keys())

    # Skip logic
    result_file = out_path / "stage5_md.json"
    existing = []
    existing_names = set()
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing_names = {r.get("monomer") for r in existing}

    if existing_names:
        logger.info(f"  {len(existing_names)} already computed, skipping")

    all_results = list(existing)

    logger.info(f"Stage 5: GROMACS MD for {monomer_names}")

    for m_name in monomer_names:
        if m_name in existing_names:
            logger.info(f"  {m_name}: already computed, skipping")
            continue
        if m_name not in monomer_library:
            logger.warning(f"  {m_name}: not in library, skipping")
            continue

        m_smiles = monomer_library[m_name]
        ratios = MD_RATIOS_TO_TEST if MD_RATIO_SCREENING else [MD_TEMPLATE_MONOMER_RATIO]
        n_mono = ratios[-1]  # Use highest ratio for analysis

        logger.info(f"\n{'='*20} MD: {m_name} (1:{n_mono}) {'='*20}")

        md_dir = out_path / m_name
        md_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build system
            sys_info = build_mip_system(
                template_smiles, "TMP",
                m_smiles, m_name,
                n_monomers=n_mono,
                work_dir=md_dir / "build",
                box_size=MD_BOX_SIZE,
                temperature=TEMPERATURE,
                crosslinker_smiles=list(CROSSLINKER_LIBRARY.values())[0] if MD_INCLUDE_CROSSLINKER else None,
                crosslinker_name=list(CROSSLINKER_LIBRARY.keys())[0] if MD_INCLUDE_CROSSLINKER else None,
                n_crosslinker=MD_CROSSLINKER_RATIO if MD_INCLUDE_CROSSLINKER else 0,
            )

            if "error" in sys_info:
                logger.error(f"  System build failed: {sys_info['error']}")
                continue

            # Copy files to MD directory
            import shutil
            for f in (md_dir / "build").glob("*"):
                if f.is_file():
                    shutil.copy2(str(f), str(md_dir / f.name))

            # Run MD
            md_result = run_md_pipeline(
                md_dir, time_ns=MD_TIME_NS,
                temperature=TEMPERATURE,
            )

            # Analyze
            analysis = analyze_md(md_dir, template_name="TMP",
                                   cutoff_A=MD_CONTACT_CUTOFF)

            result = {
                "monomer": m_name,
                "n_monomers": n_mono,
                "md_time_ns": MD_TIME_NS,
                "contact_frequency": analysis.get("contact_frequency", 0),
                "EBN": analysis.get("EBN", 0),
                "mean_min_distance_A": analysis.get("mean_min_distance_A"),
                "n_frames_analyzed": analysis.get("n_frames_analyzed", 0),
                "n_hbonds_mean": analysis.get("n_hbonds_mean", 0),
                "interaction_energy_kJ": analysis.get("interaction_energy_kJ"),
                "interaction_energy_std": analysis.get("interaction_energy_std"),
                "ie_method": analysis.get("ie_method", "N/A"),
                "success": True,
            }

            all_results.append(result)
            existing_names.add(m_name)

            # Incremental save
            with open(result_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            logger.info(f"  {m_name}: contact_freq={result['contact_frequency']:.4f}, "
                        f"EBN={result['EBN']:.4f}")

        except Exception as e:
            logger.error(f"  Stage 5 failed for {m_name}: {e}")
            import traceback; traceback.print_exc()

    # Final save
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Synthesis ratio recommendation
    _recommend_ratios(all_results)
    _print_summary(all_results)

    # ── Multi-monomer combination optimization + MD ──
    from .config import (MD_MULTI_MONOMER, MD_MULTI_MONOMER_TOP_N,
                         MD_INCLUDE_CROSSLINKER, MD_CROSSLINKER_RATIO)
    if MD_MULTI_MONOMER and len(all_results) >= 2:
        logger.info("\n--- Multi-monomer combination optimization ---")
        try:
            from . import config as _cfg
            combo = None
            # Prefer the combination already chosen by Stage 4 (MMSD)
            mmsd_json = out_path.parent / "stage4" / "mmsd_results.json"
            mres = None
            if mmsd_json.exists():
                try:
                    mres = json.loads(mmsd_json.read_text())
                except Exception:
                    mres = None
            # If Stage 4 (MMSD) was not run, fall back to running it here
            if mres is None and getattr(_cfg, "MMSD_ENABLE", True):
                try:
                    from .stage4_mmsd import run_mmsd
                    mres = run_mmsd(template_smiles=template_smiles,
                                    output_dir=str(out_path.parent / "stage4"))
                except Exception as e:
                    logger.warning(f"  MMSD search failed ({e}); "
                                   f"using greedy metric selection")
            top = (mres or {}).get("top_pcs") or []
            if top:
                pc = top[0]
                combo = {
                    "selected": [m for m in pc.get("functional_monomers", [])
                                 if m in monomer_library],
                    "score": pc.get("bo_objective") or 0.0,
                    "method": f"MMSD ({(mres or {}).get('optimizer')})",
                    "crosslinker": pc.get("crosslinker"),
                    "mmsd": pc,
                }
            if not (combo and combo.get("selected")):
                combo = _optimize_combination(all_results, monomer_library, out_path)
            if combo and combo.get("selected"):
                combo_names = combo["selected"]
                combo_smiles = {n: monomer_library[n] for n in combo_names if n in monomer_library}

                logger.info(f"  Optimal combination: {combo_names} (score={combo['score']:.3f})")
                with open(out_path / "stage5_combination.json", "w") as f:
                    json.dump(combo, f, indent=2, default=str)

                # Run multi-monomer MD
                from .utils_gromacs import build_multi_monomer_system, run_md_pipeline, analyze_md
                mm_dir = out_path / "multi_monomer"
                mm_dir.mkdir(exist_ok=True)

                # Prefer the crosslinker chosen by MMSD, else first in library
                _mmsd_xl = combo.get("crosslinker")
                if MD_INCLUDE_CROSSLINKER:
                    xl_name = (_mmsd_xl if _mmsd_xl in CROSSLINKER_LIBRARY
                               else list(CROSSLINKER_LIBRARY.keys())[0])
                    xl_smiles = CROSSLINKER_LIBRARY[xl_name]
                else:
                    xl_name = xl_smiles = None

                sys_info = build_multi_monomer_system(
                    template_smiles, "TMP", combo_smiles,
                    n_per_monomer=2, work_dir=mm_dir / "build",
                    box_size=MD_BOX_SIZE + 1.0,  # larger box for more molecules
                    crosslinker_smiles=xl_smiles,
                    crosslinker_name=xl_name,
                    n_crosslinker=MD_CROSSLINKER_RATIO if MD_INCLUDE_CROSSLINKER else 0,
                )
                if "error" not in sys_info:
                    import shutil as _shutil
                    for f in (mm_dir / "build").glob("*"):
                        if f.is_file():
                            _shutil.copy2(str(f), str(mm_dir / f.name))
                    run_md_pipeline(mm_dir, time_ns=MD_TIME_NS, temperature=TEMPERATURE)
                    mm_analysis = analyze_md(mm_dir, template_name="TMP",
                                             cutoff_A=MD_CONTACT_CUTOFF)
                    combo["md_analysis"] = mm_analysis
                    with open(out_path / "stage5_combination.json", "w") as f:
                        json.dump(combo, f, indent=2, default=str)
                    logger.info(f"  Multi-monomer MD complete: "
                                f"contact={mm_analysis.get('contact_frequency', 0):.4f}")
        except Exception as e:
            logger.warning(f"  Multi-monomer optimization failed: {e}")
            import traceback; traceback.print_exc()

    return all_results


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
    """Recommend synthesis ratios based on contact frequency inverse."""
    contacts = {r["monomer"]: r["contact_frequency"]
                for r in results if r.get("contact_frequency", 0) > 0}
    if not contacts:
        return

    # Inverse contact frequency → weak binders need more monomer
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

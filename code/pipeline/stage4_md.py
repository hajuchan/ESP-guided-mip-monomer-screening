"""
Stage 4: Pre-polymerization MD Simulation (GROMACS)
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
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage4] %(message)s")
logger = logging.getLogger(__name__)

# ── Stage 4 Parameters ──
MD_TIME_NS = 50              # Production MD time
MD_N_MONOMERS = 4            # Default template:monomer ratio
MD_CONTACT_CUTOFF = 6.0      # Å, contact frequency cutoff
MD_BOX_SIZE = 4.0            # nm, initial box size


def run_stage4(template_smiles: str = None,
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
        output_dir = OUTPUT_DIRS.get("stage4", f"{OUTPUT_DIR}/stage4")
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
    result_file = out_path / "stage4_md.json"
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

    logger.info(f"Stage 4: GROMACS MD for {monomer_names}")

    for m_name in monomer_names:
        if m_name in existing_names:
            logger.info(f"  {m_name}: already computed, skipping")
            continue
        if m_name not in monomer_library:
            logger.warning(f"  {m_name}: not in library, skipping")
            continue

        m_smiles = monomer_library[m_name]
        n_mono = MD_TEMPLATE_MONOMER_RATIO if not MD_RATIO_SCREENING else MD_RATIOS_TO_TEST[-1]

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
                "n_hbonds_stable": 0,  # TODO: implement H-bond analysis
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
            logger.error(f"  Stage 4 failed for {m_name}: {e}")
            import traceback; traceback.print_exc()

    # Final save
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Synthesis ratio recommendation
    _recommend_ratios(all_results)
    _print_summary(all_results)

    return all_results


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
    """Print Stage 4 summary."""
    logger.info(f"\n{'='*60}")
    logger.info("Stage 4: Pre-polymerization MD Summary")
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
    run_stage4()

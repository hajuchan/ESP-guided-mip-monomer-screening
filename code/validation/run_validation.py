"""
MIP Pipeline Validation Runner
===============================
전체 검증 흐름:
  1. --compute: monomer-template DFT (reference_energies.json)
  2. --selectivity: monomer-interferent DFT (selectivity_data.json)
  3. --md: Stage 4 MD 시뮬레이션 (stage4_md.json)
  4. --stage all: 전체 검증 (Stage 2 + 3 + 4 순위 vs 실험 IF)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from pipeline.config import (
    OUTPUT_DIR, TEMPLATE_SMILES, MONOMER_LIBRARY,
    INTERFERENT_LIBRARY, KB_KCAL, TEMPERATURE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Validation] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIP Pipeline Validation Suite"
    )
    parser.add_argument(
        "--compute", action="store_true",
        help="Step 1: Compute monomer-template DFT (reference_energies.json)"
    )
    parser.add_argument(
        "--selectivity", action="store_true",
        help="Step 2: Compute monomer-interferent DFT (selectivity_data.json)"
    )
    parser.add_argument(
        "--md", action="store_true",
        help="Step 3: Run Stage 4 MD simulations (stage4_md.json)"
    )
    parser.add_argument(
        "--all-compute", action="store_true",
        help="Run steps 1+2+3 sequentially"
    )
    parser.add_argument(
        "--rank", action="store_true",
        help="Compute and display Stage 2/3/4 rankings vs experimental IF"
    )
    parser.add_argument(
        "--load-only", action="store_true",
        help="Use existing results only (no new computation)"
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["numerical", "ranking", "selectivity", "new_features", "all"],
        help="Which validation to run"
    )
    parser.add_argument(
        "--feature", type=str, default="all",
        help="Feature validation: A-H or 'all'"
    )
    parser.add_argument(
        "--md-monomers", type=str, nargs="+",
        default=["OPD", "MAA", "4VB", "APB", "ACM", "PYR"],
        help="Monomers for MD simulation"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
    )
    return parser.parse_args()


# ── Step 1: Compute monomer-template DFT ──

def step_compute_references(output_dir):
    """Compute monomer-template binding energies."""
    logger.info("=" * 60)
    logger.info("Step 1: Computing monomer-template DFT (reference_energies.json)")
    logger.info("=" * 60)
    from .compute_reference import compute_all_references
    compute_all_references(output_dir=output_dir)


# ── Step 2: Compute monomer-interferent DFT ──

def step_compute_selectivity(output_dir):
    """Compute monomer-interferent binding energies for Stage 3."""
    logger.info("=" * 60)
    logger.info("Step 2: Computing monomer-interferent DFT (selectivity_data.json)")
    logger.info("=" * 60)
    from .compute_selectivity import compute_selectivity_data
    compute_selectivity_data(output_dir=output_dir)


# ── Step 3: Run MD simulations ──

def step_run_md(output_dir, monomer_names):
    """Run Stage 4 MD for specified monomers."""
    logger.info("=" * 60)
    logger.info(f"Step 3: Running Stage 4 MD for {monomer_names}")
    logger.info("=" * 60)
    from pipeline.stage4_md import run_stage4
    results = run_stage4(
        template_smiles=TEMPLATE_SMILES,
        monomer_names=monomer_names,
        monomer_library=MONOMER_LIBRARY,
    )
    return results


# ── Rankings: Stage 2 vs Stage 3 vs Stage 4 vs IF ──

def compute_all_rankings(output_dir):
    """Load all results and compute Stage 2/3/4 rankings vs experimental IF."""
    val_dir = Path(output_dir) / "validation"

    # Experimental IF (Mukasa 2023)
    exp_if = {"OPD": 3.2, "MAA": 2.8, "4VB": 2.4, "APB": 2.0, "ACM": 1.5, "PYR": 1.2}
    monomers = list(exp_if.keys())

    results = {"exp_if": exp_if}

    # ── Stage 2: monomer-template binding energy ──
    ref_path = val_dir / "reference_energies.json"
    if ref_path.exists():
        with open(ref_path) as f:
            ref = json.load(f)
        mukasa = ref.get("mukasa2023", {})

        stage2 = {}
        for m in monomers:
            key = f"phe_{m}"
            if key in mukasa and mukasa[key].get("bsse_dE") is not None:
                stage2[m] = mukasa[key]["bsse_dE"]
        results["stage2"] = stage2
    else:
        logger.warning("reference_energies.json not found — Stage 2 ranking unavailable")

    # ── Stage 3: selectivity (monomer-interferent) ──
    sel_path = val_dir / "selectivity_data.json"
    if sel_path.exists():
        with open(sel_path) as f:
            sel_data = json.load(f)

        interferents = list(INTERFERENT_LIBRARY.keys())

        for func_key, func_label in [("wb97xd", "ωB97XD"), ("wb97m_v", "ωB97M-V")]:
            func_data = sel_data.get(func_key, {})
            if not func_data:
                continue

            stage3 = {}
            for m in monomers:
                e_template = results.get("stage2", {}).get(m)
                if e_template is None:
                    continue

                interf_energies = []
                for interf in interferents:
                    key = f"{m}_{interf}"
                    entry = func_data.get(key, {})
                    e_interf = entry.get("bsse_dE")
                    if e_interf is not None:
                        interf_energies.append(e_interf)

                if interf_energies:
                    avg_interf = np.mean(interf_energies)
                    ddE = e_template - avg_interf
                    stage3[m] = {"ddE": ddE, "avg_interf": avg_interf}

            results[f"stage3_{func_key}"] = stage3
    else:
        logger.warning("selectivity_data.json not found — Stage 3 ranking unavailable")

    # ── Stage 4: MD (EBN, H-bond) ──
    # Check multiple possible locations
    md_path = None
    for candidate in [
        Path(output_dir) / "stage4" / "stage4_md.json",
        Path(output_dir) / "stage4_md.json",
    ]:
        if candidate.exists():
            md_path = candidate
            break

    if md_path:
        with open(md_path) as f:
            md_data = json.load(f)

        stage4 = {}
        if isinstance(md_data, list):
            for entry in md_data:
                m = entry.get("monomer")
                if m in exp_if:
                    stage4[m] = {
                        "EBN": entry.get("EBN", 0),
                        "n_hbonds_stable": entry.get("n_hbonds_stable", 0),
                    }
        elif isinstance(md_data, dict):
            for m, entry in md_data.items():
                if m in exp_if:
                    if isinstance(entry, dict):
                        stage4[m] = {
                            "EBN": entry.get("EBN", 0),
                            "n_hbonds_stable": entry.get("n_hbonds_stable", 0),
                        }
        results["stage4"] = stage4
    else:
        logger.warning("stage4_md.json not found — Stage 4 ranking unavailable")

    return results


def print_ranking_report(results):
    """Print comprehensive ranking comparison table."""
    from scipy.stats import spearmanr

    exp_if = results["exp_if"]
    monomers = sorted(exp_if.keys(), key=lambda m: -exp_if[m])

    print()
    print("=" * 90)
    print("  Phenylalanine MIP: Stage 2 → Stage 3 → Stage 4 vs 실험 IF")
    print("=" * 90)
    print()

    # ── Header ──
    header = f"{'Monomer':<8} {'IF(실험)':>8}"
    stages = []
    if "stage2" in results:
        header += f" {'Stage2 bsse':>12} {'S2순위':>6}"
        stages.append("stage2")
    for func in ["wb97xd", "wb97m_v"]:
        key = f"stage3_{func}"
        if key in results:
            label = "S3(XD)" if func == "wb97xd" else "S3(MV)"
            header += f" {label+' ΔΔE':>12} {label+'순위':>8}"
            stages.append(key)
    if "stage4" in results:
        header += f" {'S4 EBN':>8} {'S4 Hb':>6} {'S4순위':>6}"
        stages.append("stage4")
    print(header)
    print("-" * len(header))

    # ── Compute rankings ──
    rankings = {}

    if "stage2" in results:
        s2 = results["stage2"]
        s2_sorted = sorted(s2.items(), key=lambda x: x[1])
        rankings["stage2"] = {m: i + 1 for i, (m, _) in enumerate(s2_sorted)}

    for func in ["wb97xd", "wb97m_v"]:
        key = f"stage3_{func}"
        if key in results:
            s3 = results[key]
            s3_sorted = sorted(s3.items(), key=lambda x: x[1]["ddE"])
            rankings[key] = {m: i + 1 for i, (m, _) in enumerate(s3_sorted)}

    if "stage4" in results:
        s4 = results["stage4"]
        s4_sorted = sorted(s4.items(), key=lambda x: -x[1]["EBN"])
        rankings["stage4"] = {m: i + 1 for i, (m, _) in enumerate(s4_sorted)}

    # ── Print rows ──
    for m in monomers:
        row = f"{m:<8} {exp_if[m]:>8.1f}"

        if "stage2" in results:
            e = results["stage2"].get(m)
            r = rankings.get("stage2", {}).get(m, "-")
            row += f" {e:>+12.3f} {r:>6}위" if e else f" {'N/A':>12} {'?':>6}"

        for func in ["wb97xd", "wb97m_v"]:
            key = f"stage3_{func}"
            if key in results:
                s3 = results.get(key, {}).get(m, {})
                ddE = s3.get("ddE")
                r = rankings.get(key, {}).get(m, "-")
                row += f" {ddE:>+12.3f} {r:>8}위" if ddE else f" {'N/A':>12} {'?':>8}"

        if "stage4" in results:
            s4 = results.get("stage4", {}).get(m, {})
            ebn = s4.get("EBN", 0)
            hb = s4.get("n_hbonds_stable", 0)
            r = rankings.get("stage4", {}).get(m, "-")
            if ebn:
                row += f" {ebn:>8.3f} {hb:>6} {r:>6}위"
            else:
                row += f" {'N/A':>8} {'?':>6} {'?':>6}"

        print(row)

    # ── Spearman correlations ──
    print()
    print("-" * 60)
    print(f"{'지표':<20} {'Spearman ρ':>12} {'OPD 위치':>10} {'PYR 위치':>10}")
    print("-" * 60)

    if_vals = [exp_if[m] for m in monomers]

    for stage_key, label in [
        ("stage2", "Stage 2 (결합에너지)"),
        ("stage3_wb97xd", "Stage 3 (ωB97XD)"),
        ("stage3_wb97m_v", "Stage 3 (ωB97M-V)"),
        ("stage4", "Stage 4 (EBN)"),
    ]:
        if stage_key not in rankings:
            continue

        rank_dict = rankings[stage_key]
        if len(rank_dict) < 3:
            continue

        if stage_key == "stage2":
            vals = [-results["stage2"].get(m, 0) for m in monomers]
        elif stage_key.startswith("stage3"):
            vals = [-results[stage_key].get(m, {}).get("ddE", 0) for m in monomers]
        elif stage_key == "stage4":
            vals = [results["stage4"].get(m, {}).get("EBN", 0) for m in monomers]

        valid = [(v, i) for v, i in zip(vals, if_vals) if v != 0]
        if len(valid) >= 3:
            rho, _ = spearmanr([v[0] for v in valid], [v[1] for v in valid])
        else:
            rho = float("nan")

        opd_r = rank_dict.get("OPD", "?")
        pyr_r = rank_dict.get("PYR", "?")
        print(f"{label:<20} {rho:>+12.3f} {str(opd_r) + '위':>10} {str(pyr_r) + '위':>10}")

    print()

    return rankings


# ── Legacy validations ──

def run_legacy_validations(output_dir: str) -> dict:
    """Run original pipeline validations (numerical, ranking, selectivity)."""
    results = {}

    logger.info("\n--- Validation 1: Numerical Reproduction ---")
    try:
        from .validate_numerical import validate_numerical
        results["numerical"] = validate_numerical(output_dir)
    except Exception as e:
        logger.error(f"Numerical validation error: {e}")
        results["numerical"] = {"overall": "FAIL", "error": str(e)}

    logger.info("\n--- Validation 2: IF Ranking Correlation ---")
    try:
        from .validate_ranking import validate_ranking
        results["ranking"] = validate_ranking(output_dir)
    except Exception as e:
        logger.error(f"Ranking validation error: {e}")
        results["ranking"] = {"overall": "FAIL", "error": str(e)}

    logger.info("\n--- Validation 3: Selectivity Direction ---")
    try:
        from .validate_selectivity import validate_selectivity
        results["selectivity"] = validate_selectivity(output_dir)
    except Exception as e:
        logger.error(f"Selectivity validation error: {e}")
        results["selectivity"] = {"overall": "FAIL", "error": str(e)}

    return results


def run_feature_validations(output_dir: str, features: list = None) -> dict:
    """Run new feature validations (A-H)."""
    logger.info("\n--- Validation 4: New Feature Checks ---")
    try:
        from .validate_new_features import run_all_feature_validations
        return run_all_feature_validations(output_dir, features)
    except Exception as e:
        logger.error(f"Feature validation error: {e}")
        return {"error": {"overall": "FAIL", "error": str(e)}}


def print_final_report(legacy: dict, features: dict, output_dir: str) -> dict:
    """Print final validation report and save to JSON."""
    n_pass = 0
    n_total = 0

    print(f"\n\u2554{'=' * 58}\u2557")
    print(f"\u2551  MIP Pipeline Validation Report                          \u2551")
    print(f"\u2560{'=' * 58}\u2563")

    print(f"\u2551  [Legacy Pipeline Validation]                            \u2551")
    legacy_labels = {
        "numerical": "Numerical reproduction",
        "ranking": "IF ranking correlation",
        "selectivity": "Selectivity direction",
    }
    for key, label in legacy_labels.items():
        res = legacy.get(key, {})
        status = res.get("overall", "SKIP")
        n_total += 1
        if status in ("PASS", "SKIP"):
            n_pass += 1

        detail = ""
        if key == "numerical":
            np_ = res.get("n_pass", 0)
            nf = res.get("n_fail", 0)
            bias = res.get("systematic_bias_mean", 0)
            std = res.get("systematic_bias_std", 0)
            detail = f"{np_}/{np_ + nf}  bias={bias:+.2f}\u00b1{std:.2f}"
        elif key == "ranking":
            rh = res.get("heptachlor", {}).get("rho", "?")
            rd = res.get("DDT", {}).get("rho", "?")
            detail = f"\u03c1={rh}(Hept), {rd}(DDT)"
        elif key == "selectivity":
            orank = res.get("OPD_rank", "?")
            prank = res.get("PYR_rank", "?")
            detail = f"OPD=#{orank}, PYR=#{prank}"

        line = f"  {label:<28s} {status:<6s} {detail}"
        print(f"\u2551{line:<58s}\u2551")

    print(f"\u2560{'=' * 58}\u2563")
    print(f"\u2551  [New Feature Validation]                                \u2551")

    feature_labels = {
        "A_multidirectional": "A. Multi-directional search",
        "B_solvent_strategy": "B. Solvent strategy",
        "C_ratio_screening": "C. Ratio screening",
        "D_esp_map": "D. ESP map",
        "E_crosslinker": "E. Cross-linker",
        "F_report": "F. HTML report",
        "G_interferent_suggestion": "G. Interferent suggestion",
        "H_if_prediction": "H. IF prediction",
    }

    for key, label in feature_labels.items():
        res = features.get(key, {})
        status = res.get("overall", "SKIP")
        n_total += 1
        if status in ("PASS", "SKIP"):
            n_pass += 1

        line = f"  {label:<28s} {status:<6s}"
        print(f"\u2551{line:<58s}\u2551")

    overall = "PASS" if n_pass == n_total else "FAIL"
    print(f"\u2560{'=' * 58}\u2563")
    line = f"  Overall: {overall} ({n_pass}/{n_total})"
    print(f"\u2551{line:<58s}\u2551")
    print(f"\u255a{'=' * 58}\u255d")

    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "v6 (adaptive functional + Stage 3 selectivity)",
        "legacy_validation": legacy,
        "feature_validation": features,
        "overall": overall,
        "n_pass": n_pass,
        "n_total": n_total,
    }

    report_path = Path(output_dir) / "validation" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {report_path}")

    return report


def main():
    args = parse_args()
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Computation steps ──
    if args.all_compute:
        args.compute = True
        args.selectivity = True
        args.md = True

    if args.compute:
        step_compute_references(out_dir)

    if args.selectivity:
        step_compute_selectivity(out_dir)

    if args.md:
        step_run_md(out_dir, args.md_monomers)

    # ── Ranking comparison ──
    if args.rank or args.stage == "all":
        logger.info("\n" + "=" * 60)
        logger.info("Stage 2 → Stage 3 → Stage 4 Ranking vs Experimental IF")
        logger.info("=" * 60)
        ranking_results = compute_all_rankings(out_dir)
        print_ranking_report(ranking_results)

    # ── Validation checks ──
    legacy_results = {}
    feature_results = {}

    if args.stage in ("all", "numerical", "ranking", "selectivity"):
        if args.stage == "all":
            legacy_results = run_legacy_validations(out_dir)
        else:
            if args.stage == "numerical":
                from .validate_numerical import validate_numerical
                legacy_results["numerical"] = validate_numerical(out_dir)
            elif args.stage == "ranking":
                from .validate_ranking import validate_ranking
                legacy_results["ranking"] = validate_ranking(out_dir)
            elif args.stage == "selectivity":
                from .validate_selectivity import validate_selectivity
                legacy_results["selectivity"] = validate_selectivity(out_dir)

    if args.stage in ("all", "new_features"):
        features = None if args.feature == "all" else [
            f.strip().upper() for f in args.feature.split(",")
        ]
        feature_results = run_feature_validations(out_dir, features)

    # ── Final report ──
    if legacy_results or feature_results:
        report = print_final_report(legacy_results, feature_results, out_dir)

        if report["overall"] == "FAIL":
            logger.info("\nRunning failure diagnostics...")
            try:
                from .diagnose_failure import diagnose_failures
                diagnostics = diagnose_failures(out_dir)
                if diagnostics:
                    print("\n--- Diagnostics ---")
                    for d in diagnostics:
                        print(f"  {d}")
            except Exception as e:
                logger.warning(f"Diagnostics failed: {e}")

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

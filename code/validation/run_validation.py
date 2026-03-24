"""
MIP Pipeline Validation Runner
===============================
Orchestrates all validation steps and generates final report.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from pipeline.config import OUTPUT_DIR

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
        help="Compute reference energies before validation"
    )
    parser.add_argument(
        "--load-only", action="store_true",
        help="Use existing reference_energies.json only"
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
        "--output-dir", type=str, default=OUTPUT_DIR,
    )
    return parser.parse_args()


def run_legacy_validations(output_dir: str) -> dict:
    """Run original pipeline validations (numerical, ranking, selectivity)."""
    results = {}

    # Numerical validation
    logger.info("\n--- Validation 1: Numerical Reproduction ---")
    try:
        from .validate_numerical import validate_numerical
        results["numerical"] = validate_numerical(output_dir)
    except Exception as e:
        logger.error(f"Numerical validation error: {e}")
        results["numerical"] = {"overall": "FAIL", "error": str(e)}

    # Ranking validation
    logger.info("\n--- Validation 2: IF Ranking Correlation ---")
    try:
        from .validate_ranking import validate_ranking
        results["ranking"] = validate_ranking(output_dir)
    except Exception as e:
        logger.error(f"Ranking validation error: {e}")
        results["ranking"] = {"overall": "FAIL", "error": str(e)}

    # Selectivity validation
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

    border = "=" * 58

    print(f"\n\u2554{'=' * 58}\u2557")
    print(f"\u2551  MIP Pipeline Validation Report                          \u2551")
    print(f"\u2560{'=' * 58}\u2563")

    # Legacy validations
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
        if status == "SKIP":
            detail = "(not computed)"
        elif key == "numerical":
            np_ = res.get("n_pass", 0)
            nf = res.get("n_fail", 0)
            bias = res.get("systematic_bias_mean", 0)
            std = res.get("systematic_bias_std", 0)
            detail = f"{np_}/{np_+nf}  bias={bias:+.2f}\u00b1{std:.2f}"
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

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "Modified pipeline (Features 1-8)",
        "legacy_validation": legacy,
        "feature_validation": features,
        "overall": overall,
        "n_pass": n_pass,
        "n_total": n_total,
    }

    report_path = Path(output_dir) / "validation" / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {report_path}")

    return report


def main():
    args = parse_args()
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1: Compute references if requested
    if args.compute:
        logger.info("Computing reference energies...")
        from .compute_reference import compute_all_references
        compute_all_references(output_dir=out_dir)

    # Step 2: Run validations
    legacy_results = {}
    feature_results = {}

    if args.stage in ("all", "numerical", "ranking", "selectivity"):
        if args.stage == "all":
            legacy_results = run_legacy_validations(out_dir)
        else:
            # Run single legacy validation
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

    # Step 3: Final report
    report = print_final_report(legacy_results, feature_results, out_dir)

    # Step 4: Diagnose failures
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
    logger.info(f"\nTotal validation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

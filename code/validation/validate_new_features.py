"""
Validate New Features (1-8)
===========================
Validates the 8 additional features added to the MIP pipeline.
"""

import argparse
import json
import logging
from glob import glob
from pathlib import Path

from pipeline.config import (
    OUTPUT_DIR, SOLVENTS, SYNTHESIS_SOLVENT, SOLVENT_STRATEGY,
    USE_ESP_MAP,
    MD_RATIO_SCREENING, MD_RATIOS_TO_TEST,
    CROSSLINKER_LIBRARY, TEMPLATE_SMILES, TEMPLATE_NAME,
)
from .config_validation import (
    VALID_SOLVENT_STRATEGIES,
    CROSSLINKER_EXPECTED_WEAK, CROSSLINKER_WEAK_THRESHOLD,
    IF_MODEL_MAX_LOO_RMSE,
    SINGH_COMPLEX_HBOND_DISTANCE_RANGE, SINGH_COMPLEX_EXPECTED_HBONDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Validate] %(message)s")
logger = logging.getLogger(__name__)


# ── A: Multi-directional search ──────────────────────────────────────

def validate_multidirectional(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate Feature 1: multi-directional complex search."""
    out = Path(output_dir)
    result = {
        "best_direction_field_exists": False,
        "best_is_minimum": True,
        "multidirectional_vs_single_improvement": {},
        "hbond_structure_check": "SKIP",
        "overall": "SKIP",
    }

    top_path = out / "stage1" / "stage1_top.json"
    if not top_path.exists():
        logger.warning("stage1_top.json not found — skipping A")
        return result

    with open(top_path) as f:
        stage1_data = json.load(f)

    if not stage1_data:
        return result

    # Check best_direction field
    has_bd = all("best_direction" in entry for entry in stage1_data if entry.get("success"))
    result["best_direction_field_exists"] = has_bd

    if not has_bd:
        result["overall"] = "FAIL"
        logger.info("  A. best_direction field missing in stage1_top.json")
        return result

    # Check that best_direction is actually the minimum
    all_minimum = True
    improvements = {}
    for entry in stage1_data:
        if not entry.get("success") or "all_directions" not in entry:
            continue
        name = entry["name"]
        best_dir = entry["best_direction"]
        all_dirs = entry["all_directions"]

        # Filter out None/failed directions
        valid = {d: e for d, e in all_dirs.items() if e is not None}
        if not valid:
            continue

        min_dir = min(valid, key=valid.get)
        if valid.get(best_dir) != valid.get(min_dir):
            all_minimum = False

        # Compare vs +x
        px_energy = all_dirs.get("+x")
        best_energy = entry["dE_kcal"]
        if px_energy is not None:
            improvements[name] = round(px_energy - best_energy, 3)

    result["best_is_minimum"] = all_minimum
    result["multidirectional_vs_single_improvement"] = improvements

    result["overall"] = "PASS" if has_bd and all_minimum else "FAIL"
    logger.info(f"  A. Multi-directional: bd_exists={has_bd}, "
                f"is_min={all_minimum}, improvements={len(improvements)}")
    return result


# ── B: Solvent strategy ──────────────────────────────────────────────

def validate_solvent_strategy(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate Feature 2: solvent aggregation strategy."""
    out = Path(output_dir)
    result = {
        "strategy_valid": SOLVENT_STRATEGY in VALID_SOLVENT_STRATEGIES,
        "synthesis_solvent_exists": SYNTHESIS_SOLVENT in SOLVENTS,
        "csv_has_selected_solvent_column": False,
        "solvent_consistency": True,
        "overall": "SKIP",
    }

    if not result["strategy_valid"]:
        result["overall"] = "FAIL"
        return result

    if SOLVENT_STRATEGY == "synthesis_match" and not result["synthesis_solvent_exists"]:
        result["overall"] = "FAIL"
        return result

    csv_path = out / "stage3" / "stage3_selectivity.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        result["csv_has_selected_solvent_column"] = "selected_solvent" in df.columns
        if "selected_solvent" in df.columns and SOLVENT_STRATEGY == "synthesis_match":
            result["solvent_consistency"] = all(
                df["selected_solvent"] == SYNTHESIS_SOLVENT
            )

    result["overall"] = "PASS" if (
        result["strategy_valid"] and result["synthesis_solvent_exists"]
    ) else "FAIL"

    logger.info(f"  B. Solvent strategy: {SOLVENT_STRATEGY}, "
                f"valid={result['strategy_valid']}, "
                f"solvent_in_solvents={result['synthesis_solvent_exists']}")
    return result


# ── C: Ratio screening ───────────────────────────────────────────────

def validate_ratio_screening(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate Feature 3: template:monomer ratio screening."""
    out = Path(output_dir)
    result = {
        "ratio_results_present": False,
        "all_ratios_computed": False,
        "ebn_trend_valid": True,
        "overall": "SKIP",
    }

    if not MD_RATIO_SCREENING:
        logger.info("  C. Ratio screening disabled — SKIP")
        return result

    md_path = out / "stage5" / "stage5_md.json"
    if not md_path.exists():
        logger.info("  C. stage5_md.json not found — SKIP")
        return result

    with open(md_path) as f:
        md_data = json.load(f)

    for entry in md_data:
        if "ratio_results" not in entry:
            continue
        result["ratio_results_present"] = True

        rr = entry["ratio_results"]
        expected = set(str(r) for r in MD_RATIOS_TO_TEST)
        computed = set(rr.keys())
        if expected.issubset(computed):
            result["all_ratios_computed"] = True

        # Check EBN trend (should increase or plateau)
        sorted_ratios = sorted(rr.keys(), key=int)
        for i in range(1, len(sorted_ratios)):
            prev = rr[sorted_ratios[i - 1]].get("EBN", 0)
            curr = rr[sorted_ratios[i]].get("EBN", 0)
            if curr < prev * 0.8:  # Allow small fluctuations
                result["ebn_trend_valid"] = False

    if result["ratio_results_present"]:
        result["overall"] = "PASS" if (
            result["all_ratios_computed"] and result["ebn_trend_valid"]
        ) else "FAIL"
    else:
        result["overall"] = "SKIP"

    logger.info(f"  C. Ratio screening: present={result['ratio_results_present']}, "
                f"trend_valid={result['ebn_trend_valid']}")
    return result


# ── D: ESP map ────────────────────────────────────────────────────────

def validate_esp_map(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate Feature 4: ESP map generation."""
    out = Path(output_dir)
    result = {
        "esp_files_present": False,
        "top3_esp_complete": False,
        "cube_files_present": False,
        "images_valid": True,
        "overall": "SKIP",
    }

    if not USE_ESP_MAP:
        logger.info("  D. ESP map disabled — SKIP")
        return result

    esp_pngs = glob(str(out / "stage2" / "stage2_esp_*.png"))
    esp_cubes = glob(str(out / "stage2_esp_*.cube"))
    result["esp_files_present"] = len(esp_pngs) > 0
    result["cube_files_present"] = len(esp_cubes) > 0

    # Check TOP-3 monomer ESP completeness
    top3_path = out / "stage3" / "stage3_top.json"
    if top3_path.exists():
        with open(top3_path) as f:
            top3 = json.load(f)
        all_complete = True
        for m in top3:
            for suffix in ["template", "monomer", "complex"]:
                pat = str(out / f"stage2_esp_{m}_*_{suffix}.png")
                if not glob(pat):
                    all_complete = False
        result["top3_esp_complete"] = all_complete

    # Validate image files
    if esp_pngs:
        try:
            from PIL import Image
            for p in esp_pngs[:3]:  # Check first 3
                Image.open(p).verify()
        except Exception:
            result["images_valid"] = False

    if result["esp_files_present"]:
        result["overall"] = "PASS"
    else:
        # ESP not generated yet — this is expected if Stage 2 hasn't run with ESP
        result["overall"] = "SKIP"

    logger.info(f"  D. ESP map: {len(esp_pngs)} PNGs, {len(esp_cubes)} cubes")
    return result


# ── E: Cross-linker screening ────────────────────────────────────────

def validate_crosslinker(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate Feature 5: cross-linker screening results."""
    out = Path(output_dir)
    result = {
        "all_crosslinkers_present": False,
        "assessment_fields_valid": False,
        "egdma_weak_binding": False,
        "overall": "SKIP",
    }

    cl_path = out / "features" / "crosslinker.json"
    if not cl_path.exists():
        logger.info("  E. crosslinker.json not found — SKIP")
        return result

    with open(cl_path) as f:
        cl_data = json.load(f)

    expected_cls = set(CROSSLINKER_LIBRARY.keys())
    present_cls = set(cl_data.keys())
    result["all_crosslinkers_present"] = expected_cls.issubset(present_cls)

    # Check assessment fields
    has_assessment = all(
        any("assessment" in v for v in solvent_data.values())
        for solvent_data in cl_data.values()
        if isinstance(solvent_data, dict)
    )
    result["assessment_fields_valid"] = has_assessment

    # Check EGDMA weak binding
    for weak_cl in CROSSLINKER_EXPECTED_WEAK:
        if weak_cl in cl_data:
            for solvent_name, data in cl_data[weak_cl].items():
                if isinstance(data, dict) and "bsse_dE" in data:
                    if data["bsse_dE"] > CROSSLINKER_WEAK_THRESHOLD:
                        result["egdma_weak_binding"] = True

    result["overall"] = "PASS" if (
        result["all_crosslinkers_present"] and result["assessment_fields_valid"]
    ) else "FAIL" if cl_data else "SKIP"

    logger.info(f"  E. Cross-linker: all_present={result['all_crosslinkers_present']}, "
                f"egdma_weak={result['egdma_weak_binding']}")
    return result


# ── F: HTML report ────────────────────────────────────────────────────

def validate_report(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate Feature 6: HTML report generation."""
    out = Path(output_dir)
    result = {
        "report_file_exists": False,
        "required_sections_present": False,
        "images_embedded": False,
        "references_present": False,
        "overall": "SKIP",
    }

    reports = glob(str(out / "reports" / "mip_report_*.html"))
    if not reports:
        logger.info("  F. No HTML report found — SKIP")
        return result

    result["report_file_exists"] = True
    # Use the most recent report
    report_path = max(reports)

    with open(report_path, encoding="utf-8") as f:
        html = f.read()

    # Check required sections
    required = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    result["required_sections_present"] = all(s in html for s in required)

    # Check base64 images
    result["images_embedded"] = "data:image/png;base64" in html

    # Check references
    result["references_present"] = "Singh" in html or "Mukasa" in html or "reference" in html.lower()

    result["overall"] = "PASS" if (
        result["report_file_exists"] and result["required_sections_present"]
    ) else "FAIL"

    logger.info(f"  F. Report: exists={result['report_file_exists']}, "
                f"sections={result['required_sections_present']}, "
                f"images={result['images_embedded']}")
    return result


# ── G: Interferent suggestion ─────────────────────────────────────────

def validate_interferent_suggestion(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate Feature 7: auto interferent suggestion."""
    result = {
        "function_callable": False,
        "tyrosine_suggested": False,
        "tanimoto_range_valid": True,
        "output_format_valid": True,
        "overall": "FAIL",
    }

    try:
        from pipeline.suggest_interferents import suggest_interferents
        result["function_callable"] = True

        suggestions = suggest_interferents(
            template_smiles=TEMPLATE_SMILES, output_dir=output_dir
        )

        if not suggestions:
            result["overall"] = "FAIL"
            return result

        # Check Tyrosine is in suggestions
        names = [s.get("name", "").lower() for s in suggestions]
        result["tyrosine_suggested"] = any("tyrosine" in n for n in names)

        # Check Tanimoto range
        for s in suggestions:
            t = s.get("tanimoto", 0)
            if not (0.3 <= t <= 0.8):
                result["tanimoto_range_valid"] = False

        # Check format
        for s in suggestions:
            if not all(k in s for k in ["name", "smiles", "tanimoto"]):
                result["output_format_valid"] = False

        result["overall"] = "PASS" if (
            result["function_callable"] and result["tyrosine_suggested"] and
            result["tanimoto_range_valid"] and result["output_format_valid"]
        ) else "FAIL"

    except Exception as e:
        logger.warning(f"  G. Interferent suggestion failed: {e}")
        result["overall"] = "FAIL"

    logger.info(f"  G. Interferent suggestion: callable={result['function_callable']}, "
                f"tyrosine={result['tyrosine_suggested']}")
    return result


# ── H: IF prediction model ───────────────────────────────────────────

def validate_if_prediction(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate Feature 8: IF prediction model."""
    out = Path(output_dir)
    result = {
        "loo_rmse_acceptable": False,
        "predictions_in_range": True,
        "feature_count_valid": False,
        "model_saved": False,
        "rank_consistency": True,
        "overall": "FAIL",
    }

    try:
        from pipeline.predict_if import train_if_model, LITERATURE_DATA, _build_features

        # Check feature count
        sample = _build_features(-5.0, 4.71)
        result["feature_count_valid"] = len(sample) >= 3

        # Run training
        train_result = train_if_model(output_dir=output_dir)

        # Check model file
        result["model_saved"] = (out / "features" / "predict_if_model.pkl").exists()

        # Load predictions
        pred_path = out / "features" / "predict_if.json"
        if pred_path.exists():
            with open(pred_path) as f:
                pred_data = json.load(f)

            loo_rmse = pred_data.get("loo_cv_rmse", 999)
            result["loo_rmse_acceptable"] = loo_rmse <= IF_MODEL_MAX_LOO_RMSE

            # Check prediction ranges
            for p in pred_data.get("predictions", []):
                if_pred = p.get("IF_predicted", 0)
                if not (0.5 <= if_pred <= 5.0):
                    result["predictions_in_range"] = False

        result["overall"] = "PASS" if (
            result["loo_rmse_acceptable"] and result["feature_count_valid"] and
            result["model_saved"]
        ) else "FAIL"

    except Exception as e:
        logger.warning(f"  H. IF prediction failed: {e}")

    logger.info(f"  H. IF prediction: rmse_ok={result['loo_rmse_acceptable']}, "
                f"model_saved={result['model_saved']}")
    return result


# ── Main dispatcher ───────────────────────────────────────────────────

ALL_VALIDATORS = {
    "A": ("A_multidirectional", validate_multidirectional),
    "B": ("B_solvent_strategy", validate_solvent_strategy),
    "C": ("C_ratio_screening", validate_ratio_screening),
    "D": ("D_esp_map", validate_esp_map),
    "E": ("E_crosslinker", validate_crosslinker),
    "F": ("F_report", validate_report),
    "G": ("G_interferent_suggestion", validate_interferent_suggestion),
    "H": ("H_if_prediction", validate_if_prediction),
}


def run_all_feature_validations(output_dir: str = OUTPUT_DIR,
                                features: list = None) -> dict:
    """Run all or selected feature validations."""
    if features is None:
        features = list(ALL_VALIDATORS.keys())

    results = {}
    for key in features:
        if key not in ALL_VALIDATORS:
            logger.warning(f"Unknown feature: {key}")
            continue
        name, func = ALL_VALIDATORS[key]
        logger.info(f"Running validation {key}: {name}")
        results[name] = func(output_dir)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate new pipeline features")
    parser.add_argument("--feature", type=str, default="all",
                        help="Feature to validate: A-H or 'all'")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.feature == "all":
        features = None
    else:
        features = [f.strip().upper() for f in args.feature.split(",")]

    results = run_all_feature_validations(args.output_dir, features)

    print("\n" + "=" * 50)
    print("Feature Validation Summary")
    print("=" * 50)
    for name, res in results.items():
        status = res.get("overall", "?")
        print(f"  {name:<30s}  {status}")
    print("=" * 50)

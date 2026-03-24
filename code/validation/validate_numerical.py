"""
Numerical Validation: DFT binding energies vs Singh 2012 literature
====================================================================
Compares BSSE-corrected binding energies from our DFT calculations
against Table 1 of Singh et al. (2012) DOI: 10.2174/157341112803216807.
"""

import json
import logging
from pathlib import Path

from pipeline.config import OUTPUT_DIR
from .config_validation import SINGH_2012_BINDING_ENERGIES, TOLERANCE_KCAL

logger = logging.getLogger(__name__)


def validate_numerical(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate DFT binding energies against Singh 2012 reference values.

    Parameters
    ----------
    output_dir : str
        Directory containing results/reference_energies.json.

    Returns
    -------
    dict
        Validation summary with overall status, counts, bias stats,
        and per-pair details.
    """
    ref_path = Path(output_dir) / "validation" / "reference_energies.json"

    if not ref_path.exists():
        logger.warning("reference_energies.json not found at %s", ref_path)
        return {"overall": "SKIP", "reason": "reference_energies.json not found"}

    with open(ref_path, "r") as f:
        data = json.load(f)

    singh_data = data.get("singh2012", {})

    per_pair = {}
    n_pass = 0
    n_fail = 0
    n_skip = 0
    diffs_signed = []  # computed - reference (for bias calculation)

    for (template, monomer), reference in SINGH_2012_BINDING_ENERGIES.items():
        key = f"{template}_{monomer}"
        entry = singh_data.get(key, {})
        computed = entry.get("bsse_dE")

        if computed is None:
            status = "SKIP"
            diff = None
            n_skip += 1
        else:
            diff = abs(computed - reference)
            if diff <= TOLERANCE_KCAL:
                status = "PASS"
                n_pass += 1
            else:
                status = "FAIL"
                n_fail += 1
            diffs_signed.append(computed - reference)

        per_pair[key] = {
            "status": status,
            "diff": diff,
            "ref": reference,
            "computed": computed,
        }

    # Systematic bias
    if diffs_signed:
        bias_mean = sum(diffs_signed) / len(diffs_signed)
        variance = sum((d - bias_mean) ** 2 for d in diffs_signed) / len(diffs_signed)
        bias_std = variance ** 0.5
    else:
        bias_mean = 0.0
        bias_std = 0.0

    overall = "PASS" if n_fail == 0 and n_skip < len(SINGH_2012_BINDING_ENERGIES) else "FAIL"

    # Print formatted table
    header = f"{'Template':<14} {'Monomer':<10} {'Ref':>8} {'Calc':>8} {'D':>8}   {'Status'}"
    print(header)
    print("-" * len(header))
    for (template, monomer), reference in SINGH_2012_BINDING_ENERGIES.items():
        key = f"{template}_{monomer}"
        info = per_pair[key]
        comp_str = f"{info['computed']:8.2f}" if info["computed"] is not None else "     N/A"
        diff_str = f"{info['diff']:8.2f}" if info["diff"] is not None else "     N/A"
        print(f"{template:<14} {monomer:<10} {reference:8.2f} {comp_str} {diff_str}   {info['status']}")
    print()
    print(f"Systematic bias: {bias_mean:+.3f} +/- {bias_std:.3f} kcal/mol")
    print(f"Result: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP  =>  {overall}")

    return {
        "overall": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "systematic_bias_mean": bias_mean,
        "systematic_bias_std": bias_std,
        "per_pair": per_pair,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = validate_numerical()
    print()
    print(json.dumps(results, indent=2))

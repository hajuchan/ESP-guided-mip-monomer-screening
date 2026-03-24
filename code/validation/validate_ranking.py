"""
Validate IF-energy ranking correlation using Spearman rho against Singh 2012.

Compares computed BSSE-corrected binding energies with experimental
imprinting factors (IF) from Singh et al. (2012) for heptachlor and DDT
template-monomer systems.
"""

import json
import logging
from pathlib import Path

from scipy.stats import spearmanr

from pipeline.config import OUTPUT_DIR
from .config_validation import SINGH_2012_IF, SPEARMAN_THRESHOLD, SPEARMAN_DDT_MIN

logger = logging.getLogger(__name__)


def _rank_string(monomers, values):
    """Build a rank string like 'MAA>4VP>Sty' from monomers sorted by descending value."""
    abbrev = {"MAA": "MAA", "4VP": "4VP", "styrene": "Sty"}
    paired = sorted(zip(values, monomers), reverse=True)
    return ">".join(abbrev.get(m, m) for _, m in paired)


def validate_ranking(output_dir=OUTPUT_DIR) -> dict:
    """Validate IF-energy ranking correlation for Singh 2012 systems.

    Returns
    -------
    dict
        Per-template results (rho, p_value, status, computed_rank, IF_rank)
        and an overall status ("PASS", "WARNING", "FAIL", or "SKIP").
    """
    ref_path = Path(output_dir) / "validation" / "reference_energies.json"
    if not ref_path.exists():
        logger.warning("reference_energies.json not found at %s — skipping ranking validation", ref_path)
        return {"overall": "SKIP"}

    with open(ref_path) as f:
        ref_data = json.load(f)

    singh = ref_data.get("singh2012", {})
    monomers = ["MAA", "4VP", "styrene"]
    templates = ["heptachlor", "DDT"]
    thresholds = {
        "heptachlor": SPEARMAN_THRESHOLD,
        "DDT": SPEARMAN_DDT_MIN,
    }

    results = {}

    for template in templates:
        tdata = singh.get(template, {})

        # Collect valid pairs
        valid_monomers = []
        e_vals = []
        if_vals = []

        for mon in monomers:
            bsse_dE = tdata.get(mon, {}).get("bsse_dE", None)
            if bsse_dE is None:
                logger.warning("Missing bsse_dE for %s–%s, skipping pair", template, mon)
                continue
            valid_monomers.append(mon)
            e_vals.append(-bsse_dE)  # sign flip: more negative energy → higher value
            if_vals.append(SINGH_2012_IF[template][mon])

        if len(valid_monomers) < 3:
            logger.warning("Not enough data for %s (need 3, got %d)", template, len(valid_monomers))
            results[template] = {
                "rho": None, "p_value": None, "status": "SKIP",
                "computed_rank": "N/A", "IF_rank": "N/A",
            }
            continue

        rho, p_value = spearmanr(e_vals, if_vals)

        computed_rank = _rank_string(valid_monomers, e_vals)
        if_rank = _rank_string(valid_monomers, if_vals)

        threshold = thresholds[template]
        if rho >= threshold:
            status = "PASS" if template == "heptachlor" else "PASS(relaxed)"
        else:
            status = "FAIL"

        results[template] = {
            "rho": float(rho),
            "p_value": float(p_value),
            "status": status,
            "computed_rank": computed_rank,
            "IF_rank": if_rank,
        }

    # ── Print formatted table ──────────────────────────────────────────
    header = f"{'Template':<14} {'Spearman ρ':>10} {'p-value':>9} {'Calc rank':<16} {'IF rank':<16} {'Status'}"
    print(header)
    print("-" * len(header))

    for template in templates:
        r = results.get(template)
        if r is None or r["rho"] is None:
            print(f"{template:<14} {'N/A':>10} {'N/A':>9} {'N/A':<16} {'N/A':<16} SKIP")
        else:
            print(
                f"{template:<14} {r['rho']:>10.3f} {r['p_value']:>9.3f} "
                f"{r['computed_rank']:<16} {r['IF_rank']:<16} {r['status']}"
            )

    # ── Overall status ─────────────────────────────────────────────────
    statuses = [r["status"] for r in results.values()]

    if any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif any(s == "PASS(relaxed)" for s in statuses):
        overall = "WARNING"
    else:
        overall = "PASS"

    print(f"\nOverall ranking validation: {overall}")

    results["overall"] = overall
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = validate_ranking()
    if result["overall"] == "SKIP":
        print("Ranking validation skipped — no reference_energies.json found.")

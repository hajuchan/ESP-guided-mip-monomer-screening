"""
Validate selectivity ranking against Mukasa 2023 expectations.
==============================================================
Loads BSSE-corrected binding energies from reference_energies.json,
computes selectivity scores S = exp(delta_E / kT) for each monomer,
and checks whether the ranking matches literature expectations.
"""

import json
import logging
from math import exp, log
from pathlib import Path

from pipeline.config import OUTPUT_DIR, KB_KCAL, TEMPERATURE
from .config_validation import MUKASA_2023_EXPECTED_TOP3, MUKASA_2023_EXPECTED_BOTTOM2

logger = logging.getLogger(__name__)


def validate_selectivity(output_dir: str = OUTPUT_DIR) -> dict:
    """Validate monomer selectivity ranking against Mukasa 2023.

    Parameters
    ----------
    output_dir : str
        Directory containing results/reference_energies.json.

    Returns
    -------
    dict
        Validation results including ranks, pass/fail status, and full ranking.
    """
    ref_path = Path(output_dir) / "validation" / "reference_energies.json"

    # ── Guard: missing file or empty section ───────────────────────────
    if not ref_path.exists():
        logger.warning("reference_energies.json not found at %s", ref_path)
        return {"overall": "SKIP"}

    with open(ref_path, "r") as f:
        data = json.load(f)

    mukasa = data.get("mukasa2023")
    if not mukasa:
        logger.warning("mukasa2023 section missing or empty in reference_energies.json")
        return {"overall": "SKIP"}

    # ── Separate template-monomer vs interferent pairs ─────────────────
    interferent_names = {"tyrosine", "leucine", "dopamine"}
    template_pairs = {}   # monomer_name -> bsse_dE
    interferent_pairs = {}  # interferent_name -> bsse_dE

    for key, entry in mukasa.items():
        # Keys expected as "phe_OPD", "phe_tyrosine", etc.
        if not key.startswith("phe_"):
            continue
        suffix = key[len("phe_"):]
        bsse_dE = entry.get("bsse_dE") if isinstance(entry, dict) else entry

        if bsse_dE is None:
            logger.info("Skipping %s — bsse_dE is null", key)
            continue

        if suffix.lower() in interferent_names:
            interferent_pairs[suffix.lower()] = float(bsse_dE)
        else:
            template_pairs[suffix] = float(bsse_dE)

    if not template_pairs:
        logger.warning("No valid template-monomer pairs found")
        return {"overall": "SKIP"}

    # ── Average interferent binding energy ─────────────────────────────
    if interferent_pairs:
        E_interferent_avg = sum(interferent_pairs.values()) / len(interferent_pairs)
    else:
        logger.warning("No interferent data; delta_E will equal E_template")
        E_interferent_avg = 0.0

    # ── Compute selectivity for each monomer ───────────────────────────
    kT = KB_KCAL * TEMPERATURE
    monomer_data = []

    for monomer, E_template in template_pairs.items():
        delta_E = E_template - E_interferent_avg
        S = exp(delta_E / kT)
        log_S = log(S)
        monomer_data.append({
            "monomer": monomer,
            "bsse_dE": E_template,
            "delta_E": delta_E,
            "log_S": log_S,
        })

    # Rank by log_S: more negative = more selective = rank 1
    monomer_data.sort(key=lambda d: d["log_S"])
    for rank, entry in enumerate(monomer_data, start=1):
        entry["rank"] = rank

    total = len(monomer_data)
    full_ranking = [d["monomer"] for d in monomer_data]

    # ── Look up OPD and PYR ranks ──────────────────────────────────────
    rank_map = {d["monomer"]: d["rank"] for d in monomer_data}
    OPD_rank = rank_map.get("OPD")
    PYR_rank = rank_map.get("PYR")

    # ── Validation checks ──────────────────────────────────────────────
    OPD_check = (OPD_rank is not None) and (OPD_rank <= 3)
    PYR_check = (PYR_rank is not None) and (PYR_rank >= total - 1)
    direction_check = (
        OPD_rank is not None
        and PYR_rank is not None
        and OPD_rank < PYR_rank
    )

    overall = "PASS" if (OPD_check and PYR_check and direction_check) else "FAIL"

    # ── Print summary table ────────────────────────────────────────────
    def _status(entry):
        m = entry["monomer"]
        r = entry["rank"]
        parts = []
        if m == "OPD":
            parts.append("PASS(top3)" if OPD_check else "FAIL(top3)")
        if m == "PYR":
            parts.append("PASS(bottom2)" if PYR_check else "FAIL(bottom2)")
        return ", ".join(parts) if parts else ""

    header = f"{'Monomer':<12} {'bsse_dE':>10} {'delta_E':>10} {'log(S)':>10} {'Rank':>6}   Status"
    print(header)
    print("-" * len(header))
    for entry in monomer_data:
        status = _status(entry)
        print(
            f"{entry['monomer']:<12} {entry['bsse_dE']:>10.3f} "
            f"{entry['delta_E']:>10.3f} {entry['log_S']:>10.4f} "
            f"{entry['rank']:>6}   {status}"
        )
    print(f"\nOverall: {overall}")

    return {
        "OPD_rank": OPD_rank,
        "PYR_rank": PYR_rank,
        "direction_correct": direction_check,
        "overall": overall,
        "full_ranking": full_ranking,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = validate_selectivity()
    print(f"\nResult: {json.dumps(result, indent=2)}")

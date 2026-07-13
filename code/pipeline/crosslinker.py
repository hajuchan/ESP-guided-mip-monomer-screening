"""
Stage 2 – Feature 5: Cross-linker Screening
============================================
Evaluate cross-linker candidates by computing DFT binding energy with the
template molecule.  A good cross-linker should NOT bind strongly to the
template (weak interaction preferred) so that it serves only as a structural
scaffold without competing with the functional monomer.

Uses the same DFT methodology as stage2_dft (B3LYP-D3BJ/6-311+G* + ddCOSMO
+ counterpoise BSSE correction).
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import (
    TEMPLATE_SMILES,
    CROSSLINKER_LIBRARY,
    SOLVENTS,
    N_GPU_WORKERS,
    N_WORKERS,
    USE_GPU,
    CROSSLINKER_THRESHOLD,
    OUTPUT_DIR,
    OUTPUT_DIRS,
)
from .stage2_dft import compute_dft_binding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CrossLinker] %(message)s")
logger = logging.getLogger(__name__)


# ── Main screening routine ──────────────────────────────────────────

def run_crosslinker_screening(
    template_smiles: str = None,
    crosslinker_library: dict = None,
    solvents: dict = None,
    output_dir: str = OUTPUT_DIR,
) -> dict:
    """Screen cross-linker candidates against the template molecule.

    For each (crosslinker x solvent) pair the DFT binding energy is computed
    via ``stage2_dft.compute_dft_binding``.  Results are assessed as follows:

    * **GOOD** – ``bsse_dE > CROSSLINKER_THRESHOLD`` (weak interaction)
    * **WARNING** – ``bsse_dE <= CROSSLINKER_THRESHOLD`` (too strong)

    Returns
    -------
    dict
        Nested dict ``{crosslinker: {solvent: {raw_dE, bsse_dE, assessment, reason}}}``
    """
    template_smiles = template_smiles or TEMPLATE_SMILES
    crosslinker_library = crosslinker_library or CROSSLINKER_LIBRARY
    solvents = solvents or SOLVENTS

    n_tasks = len(crosslinker_library) * len(solvents)
    max_workers = N_GPU_WORKERS if USE_GPU else N_WORKERS

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "crosslinker.json"

    # ── Resume: reuse already-computed (crosslinker, solvent) pairs ──
    # Each pair is an expensive DFT job (~7 min); recomputing a completed
    # screening every run wasted hours. Load the cache and only compute the
    # missing pairs (failures aren't cached, so they retry — e.g. TRIM OOM).
    results: dict = {}
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    def _cached(cl, sol):
        return isinstance(results.get(cl), dict) and sol in results[cl]

    pending = [(cl, smi, sol, eps)
               for cl, smi in crosslinker_library.items()
               for sol, eps in solvents.items()
               if not _cached(cl, sol)]
    n_cached = n_tasks - len(pending)

    logger.info(
        f"Cross-linker screening: {len(crosslinker_library)} cross-linkers "
        f"x {len(solvents)} solvents = {n_tasks} DFT jobs "
        f"({n_cached} cached, {len(pending)} to compute; "
        f"workers={max_workers}, GPU={USE_GPU})"
    )
    if not pending:
        logger.info("  all pairs cached — skipping DFT")

    # ── Parallel DFT calculations (only the missing pairs) ──────────
    raw_results = {}  # (cl_name, sol_name) -> dft result dict
    if pending:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for cl_name, cl_smiles, sol_name, eps in pending:
                fut = executor.submit(
                    compute_dft_binding,
                    cl_name, cl_smiles, template_smiles, sol_name, eps,
                )
                futures[fut] = (cl_name, sol_name)

            for future in as_completed(futures):
                cl_name, sol_name = futures[future]
                res = future.result()
                raw_results[(cl_name, sol_name)] = res
                if res["success"]:
                    logger.info(
                        f"  {cl_name:>10s}/{sol_name:<15s}: "
                        f"raw={res['raw_dE_kcal']:+.3f}, "
                        f"bsse={res['bsse_dE_kcal']:+.3f} kcal/mol"
                    )
                else:
                    logger.warning(f"  {cl_name}/{sol_name}: FAILED – {res.get('error', 'unknown')}")

    # ── Assess NEW results and merge into the cached set ────────────
    for (cl_name, sol_name), res in raw_results.items():
        if not res["success"]:
            continue
        bsse_dE = res["bsse_dE_kcal"]
        raw_dE = res["raw_dE_kcal"]

        if bsse_dE > CROSSLINKER_THRESHOLD:
            assessment = "GOOD"
            reason = "template과 약한 상호작용"
        else:
            assessment = "WARNING"
            reason = (
                f"template과 강한 결합 ({bsse_dE:+.3f} kcal/mol "
                f"<= threshold {CROSSLINKER_THRESHOLD})"
            )

        results.setdefault(cl_name, {})[sol_name] = {
            "raw_dE": round(raw_dE, 3),
            "bsse_dE": round(bsse_dE, 3),
            "assessment": assessment,
            "reason": reason,
        }

    # ── Save merged JSON ────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {json_path}")

    # ── Console summary ─────────────────────────────────────────────
    _print_summary(results)

    # ── Recommend best cross-linker (weakest binding) ───────────────
    _recommend_best(results)

    return results


def _print_summary(results: dict) -> None:
    """Print results grouped by assessment (GOOD / WARNING)."""
    good_entries = []
    warning_entries = []

    for cl_name, solvents_data in results.items():
        for sol_name, data in solvents_data.items():
            entry = (cl_name, sol_name, data)
            if data["assessment"] == "GOOD":
                good_entries.append(entry)
            else:
                warning_entries.append(entry)

    logger.info("=" * 60)
    logger.info("Cross-linker Screening Summary")
    logger.info("=" * 60)

    logger.info(f"\n  GOOD ({len(good_entries)} entries):")
    for cl, sol, d in sorted(good_entries):
        logger.info(f"    {cl:>10s} / {sol:<15s}  bsse_dE={d['bsse_dE']:+.3f} kcal/mol")

    logger.info(f"\n  WARNING ({len(warning_entries)} entries):")
    for cl, sol, d in sorted(warning_entries):
        logger.info(f"    {cl:>10s} / {sol:<15s}  bsse_dE={d['bsse_dE']:+.3f} kcal/mol")

    logger.info("=" * 60)


def _recommend_best(results: dict) -> None:
    """Recommend the cross-linker with the weakest average template binding."""
    if not results:
        logger.warning("No successful results – cannot recommend a cross-linker.")
        return

    avg_binding: dict[str, float] = {}
    for cl_name, solvents_data in results.items():
        energies = [d["bsse_dE"] for d in solvents_data.values()]
        if energies:
            avg_binding[cl_name] = sum(energies) / len(energies)

    if not avg_binding:
        return

    # Weakest binding = highest (least negative) average bsse_dE
    best_cl = max(avg_binding, key=avg_binding.get)
    logger.info(
        f"Recommendation: {best_cl} "
        f"(avg bsse_dE = {avg_binding[best_cl]:+.3f} kcal/mol, "
        f"weakest template binding)"
    )


if __name__ == "__main__":
    run_crosslinker_screening()

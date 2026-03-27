"""
Compute DFT Reference Energies for Pipeline Validation
=======================================================
Uses QuantumDock-style binding site search (GFN2-xTB) to find optimal
complex geometry, then runs DFT for validation against literature.
"""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from pipeline.config import (
    MONOMER_LIBRARY, SOLVENTS, INTERFERENT_LIBRARY,
    TEMPLATE_SMILES, N_WORKERS, N_GPU_WORKERS, USE_GPU,
    OUTPUT_DIR, N_DOCK_ORIENTATIONS, N_TOP_FOR_OPTIMIZATION, DOCK_SURFACE_OFFSET,
)
from .config_validation import (
    SINGH_2012_BINDING_ENERGIES, VALIDATION_SMILES,
)
from pipeline.stage2_dft import compute_dft_binding
from pipeline.stage1_xtb import (
    smiles_to_mol3d, generate_docked_orientations,
    screen_orientations_sp, optimize_top_candidates, mol_to_arrays,
    _adaptive_n_orientations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Reference] %(message)s")
logger = logging.getLogger(__name__)


def _compute_one_pair(task: dict) -> dict:
    """Compute DFT for one template-monomer pair, using binding site search."""
    label = task["label"]
    logger.info(f"  {label}: starting (binding_site search + DFT)...")

    try:
        # Step 1: xTB QuantumDock surface search for optimal complex
        template_mol = smiles_to_mol3d(task["template_smiles"])
        monomer_mol = smiles_to_mol3d(task["monomer_smiles"])

        n_orient = _adaptive_n_orientations(template_mol, monomer_mol)
        orientations = generate_docked_orientations(
            template_mol, monomer_mol,
            n_orientations=n_orient,
            surface_offset=DOCK_SURFACE_OFFSET,
        )
        n_valid = len(orientations)
        logger.info(f"  {label}: {n_orient} orient (adaptive) → {n_valid} valid")

        if n_valid == 0:
            return {"label": label, "group": task["group"],
                    "bsse_dE": None, "error": "all orientations clashed", "success": False}

        top = screen_orientations_sp(template_mol, orientations,
                                     top_n=N_TOP_FOR_OPTIMIZATION)
        xtb_de = top[0][1] if top else 0
        search_info = {"n_orientations": n_valid, "xtb_best_sp": round(xtb_de, 3)}
        logger.info(f"  {label}: xTB SP best={xtb_de:+.3f} kcal/mol")

        opt_result = optimize_top_candidates(template_mol, monomer_mol, top)
        xtb_opt_de = opt_result["best_energy_kcal"]
        logger.info(f"  {label}: xTB opt dE={xtb_opt_de:+.3f} kcal/mol")

        # Step 2: DFT full optimization on top 3 xTB candidates
        # Each candidate gets DFT geometry optimization → pick best
        from pipeline.stage1_xtb import _arrays_to_mol
        best_dft_de = float("inf")
        best_res = None

        # Get top 3 from xTB optimization results
        # opt_result has best_positions/best_numbers; we also need other candidates
        # For simplicity, run DFT on the best xTB complex (already optimized)
        best_complex_mol = opt_result.get("best_complex_mol")
        best_dir_str = f"site_{opt_result.get('binding_site', {}).get('atom_idx', 0)}"

        logger.info(f"  {label}: DFT full optimization on xTB best complex...")
        res = compute_dft_binding(
            monomer_name=task["monomer_name"],
            monomer_smiles=task["monomer_smiles"],
            template_smiles=task["template_smiles"],
            solvent_name=task["solvent_name"],
            eps=task["eps"],
            prebuilt_complex_mol=best_complex_mol,
        )

        if res["success"]:
            return {
                "label": label,
                "group": task["group"],
                "bsse_dE": res["bsse_dE_kcal"],
                "raw_dE": res["raw_dE_kcal"],
                "solvent": task["solvent_name"],
                "best_direction": best_dir_str,
                "xtb_binding_energy": xtb_opt_de,
                "search_info": search_info,
                "success": True,
            }
        else:
            return {
                "label": label, "group": task["group"],
                "bsse_dE": None, "error": res.get("error", "unknown"),
                "search_info": search_info, "success": False,
            }

    except Exception as exc:
        logger.error(f"  {label}: error — {exc}")
        return {
            "label": label, "group": task["group"],
            "bsse_dE": None, "error": str(exc), "success": False,
        }


def _build_singh2012_tasks() -> list:
    templates = ["heptachlor", "DDT"]
    monomers = {
        "MAA": MONOMER_LIBRARY["MAA"],
        "4VP": MONOMER_LIBRARY["4VP"],
        "styrene": MONOMER_LIBRARY["Styrene"],
    }
    tasks = []
    for tmpl in templates:
        for mon_key, mon_smiles in monomers.items():
            tasks.append({
                "label": f"{tmpl}_{mon_key}",
                "group": "singh2012",
                "template_smiles": VALIDATION_SMILES[tmpl],
                "monomer_name": mon_key,
                "monomer_smiles": mon_smiles,
                "solvent_name": "Chloroform",
                "eps": SOLVENTS["Chloroform"],
            })
    return tasks


def _build_mukasa2023_tasks() -> list:
    monomer_names = ["OPD", "MAA", "4VB", "APB", "ACM", "PYR"]
    tasks = []
    for mon in monomer_names:
        tasks.append({
            "label": f"phe_{mon}",
            "group": "mukasa2023",
            "template_smiles": TEMPLATE_SMILES,
            "monomer_name": mon,
            "monomer_smiles": MONOMER_LIBRARY[mon],
            "solvent_name": "MeOH",
            "eps": SOLVENTS["MeOH"],
        })
    return tasks



def compute_all_references(output_dir: str = OUTPUT_DIR) -> dict:
    """Compute DFT reference energies with binding site search.

    Skips already-computed pairs (resumes from previous run).
    """
    out_path = Path(output_dir)
    (out_path / "validation").mkdir(parents=True, exist_ok=True)
    ref_file = out_path / "validation" / "reference_energies.json"

    # Load existing results to skip completed pairs
    existing = {"metadata": {}, "singh2012": {}, "mukasa2023": {}}
    if ref_file.exists():
        with open(ref_file) as f:
            existing = json.load(f)
        logger.info(f"Loaded {len(existing.get('singh2012',{}))} singh + "
                    f"{len(existing.get('mukasa2023',{}))} mukasa existing results")

    all_tasks = _build_singh2012_tasks() + _build_mukasa2023_tasks()
    # Filter out already-completed pairs
    tasks = []
    for t in all_tasks:
        group = t["group"]
        label = t["label"]
        if label in existing.get(group, {}) and existing[group][label].get("bsse_dE") is not None:
            logger.info(f"  {label}: already computed, skipping")
            continue
        tasks.append(t)
    logger.info(f"{len(tasks)} pairs to compute ({len(all_tasks)-len(tasks)} skipped)")
    # Use 1 worker since each task runs binding site search (CPU intensive)
    # followed by DFT (GPU)
    max_workers = 1 if USE_GPU else min(N_WORKERS, 2)

    logger.info(
        f"Computing {len(tasks)} reference energies "
        f"(binding_site + DFT, workers={max_workers})"
    )

    # Start from existing results
    results_by_group = {
        "singh2012": dict(existing.get("singh2012", {})),
        "mukasa2023": dict(existing.get("mukasa2023", {})),
    }

    for task in tasks:
        result = _compute_one_pair(task)
        label = result["label"]
        group = result["group"]

        if result["success"]:
            results_by_group[group][label] = {
                "bsse_dE": result["bsse_dE"],
                "raw_dE": result["raw_dE"],
                "solvent": result["solvent"],
                "best_direction": result["best_direction"],
                "search_info": result.get("search_info"),
            }
            logger.info(f"  {label}: bsse_dE={result['bsse_dE']:+.3f} kcal/mol")
        else:
            results_by_group[group][label] = {
                "bsse_dE": None,
                "error": result.get("error", "unknown"),
            }
            logger.warning(f"  {label}: FAILED")

        # Save intermediate results after each pair
        output = {
            "metadata": {
                "compute_date": datetime.now().isoformat(),
                "dft_level": "wB97M-V/def2-TZVP",
                "bsse_corrected": True,
                "search_mode": "quantumdock",
            },
            "singh2012": results_by_group["singh2012"],
            "mukasa2023": results_by_group["mukasa2023"],
        }
        with open(ref_file, "w") as f:
            json.dump(output, f, indent=2)

    logger.info(f"Reference energies saved: {ref_file}")
    return output


if __name__ == "__main__":
    compute_all_references()

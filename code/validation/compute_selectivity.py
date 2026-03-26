"""
Compute Stage 3 Selectivity Data for Validation
================================================
Calculates monomer-interferent binding energies needed for proper
selectivity ranking. Uses the same DFT protocol as compute_reference.py.

Stage 3 formula:
  ΔΔE = ΔE(monomer-template) - ΔE(monomer-interferent)
  S = exp(ΔΔE / kB·T)

This script computes ΔE(monomer-interferent) for all combinations.
"""

import json
import logging
from datetime import datetime
from math import log, exp
from pathlib import Path

import numpy as np

from pipeline.config import (
    MONOMER_LIBRARY, INTERFERENT_LIBRARY, SOLVENTS,
    TEMPLATE_SMILES, OUTPUT_DIR, KB_KCAL, TEMPERATURE,
    DFT_FUNCTIONAL,
)
from pipeline.stage2_dft import compute_dft_binding
from pipeline.stage1_xtb import (
    smiles_to_mol3d, generate_docked_orientations,
    screen_orientations_sp, optimize_top_candidates,
    _adaptive_n_orientations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Selectivity] %(message)s")
logger = logging.getLogger(__name__)


def _compute_monomer_interferent(monomer_name, monomer_smiles,
                                  interferent_name, interferent_smiles,
                                  solvent_name, eps, functional_override=None):
    """Compute binding energy between a monomer and an interferent.

    Same protocol as template-monomer: xTB docking → DFT optimization → SP + BSSE.
    The interferent acts as the "template" in this calculation.
    """
    label = f"{monomer_name}-{interferent_name}"
    logger.info(f"  {label}: starting ({functional_override or DFT_FUNCTIONAL})...")

    try:
        interf_mol = smiles_to_mol3d(interferent_smiles)
        mono_mol = smiles_to_mol3d(monomer_smiles)

        n_orient = _adaptive_n_orientations(interf_mol, mono_mol)
        orientations = generate_docked_orientations(
            interf_mol, mono_mol, n_orientations=n_orient,
        )
        n_valid = len(orientations)

        if n_valid == 0:
            return {"label": label, "bsse_dE": None, "error": "all clashed"}

        top = screen_orientations_sp(interf_mol, orientations, top_n=10)
        opt = optimize_top_candidates(interf_mol, mono_mol, top)
        complex_mol = opt.get("best_complex_mol")

        res = compute_dft_binding(
            monomer_name=monomer_name,
            monomer_smiles=monomer_smiles,
            template_smiles=interferent_smiles,
            solvent_name=solvent_name,
            eps=eps,
            prebuilt_complex_mol=complex_mol,
            functional_override=functional_override,
        )

        if res["success"]:
            logger.info(f"  {label}: bsse_dE={res['bsse_dE_kcal']:+.3f} kcal/mol")
            return {
                "label": label,
                "bsse_dE": res["bsse_dE_kcal"],
                "raw_dE": res["raw_dE_kcal"],
                "functional": functional_override or DFT_FUNCTIONAL,
                "success": True,
            }
        else:
            return {"label": label, "bsse_dE": None, "error": res.get("error")}

    except Exception as exc:
        logger.error(f"  {label}: {exc}")
        return {"label": label, "bsse_dE": None, "error": str(exc)}


def compute_selectivity_data(output_dir=OUTPUT_DIR):
    """Compute all monomer-interferent binding energies for Stage 3 selectivity.

    Computes for both ωB97XD and ωB97M-V functionals.
    Results saved to results/validation/selectivity_data.json
    """
    out_path = Path(output_dir) / "validation"
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / "selectivity_data.json"

    # Load existing results
    existing = {}
    if result_file.exists():
        with open(result_file) as f:
            existing = json.load(f)

    monomers = {
        "OPD": MONOMER_LIBRARY["OPD"],
        "MAA": MONOMER_LIBRARY["MAA"],
        "4VB": MONOMER_LIBRARY["4VB"],
        "APB": MONOMER_LIBRARY["APB"],
        "ACM": MONOMER_LIBRARY["ACM"],
        "PYR": MONOMER_LIBRARY["PYR"],
    }

    interferents = {
        "Tyrosine": INTERFERENT_LIBRARY["Tyrosine"],
        "Leucine": INTERFERENT_LIBRARY["Leucine"],
        "Dopamine": INTERFERENT_LIBRARY["Dopamine"],
    }

    functionals = ["wb97xd", "wb97m-v"]
    solvent_name = "MeOH"
    eps = SOLVENTS["MeOH"]

    total = len(monomers) * len(interferents) * len(functionals)
    done = 0

    for func in functionals:
        func_key = func.replace("-", "_")
        if func_key not in existing:
            existing[func_key] = {}

        for m_name, m_smiles in monomers.items():
            for i_name, i_smiles in interferents.items():
                key = f"{m_name}_{i_name}"
                done += 1

                if key in existing[func_key] and existing[func_key][key].get("bsse_dE") is not None:
                    logger.info(f"  [{done}/{total}] {func} {key}: already computed, skipping")
                    continue

                logger.info(f"  [{done}/{total}] {func} {key}")
                result = _compute_monomer_interferent(
                    m_name, m_smiles, i_name, i_smiles,
                    solvent_name, eps,
                    functional_override=func,
                )
                existing[func_key][key] = result

                # Save after each computation
                with open(result_file, "w") as f:
                    json.dump(existing, f, indent=2)

    logger.info(f"All {total} pairs completed. Saved to {result_file}")
    return existing


def compute_rankings(output_dir=OUTPUT_DIR):
    """Compute Stage 2 (binding energy) and Stage 3 (selectivity) rankings.

    Reads reference_energies.json (template-monomer) and
    selectivity_data.json (monomer-interferent) to produce final rankings.
    """
    out_path = Path(output_dir) / "validation"

    # Load template-monomer binding energies
    ref_file = out_path / "reference_energies.json"
    if not ref_file.exists():
        logger.error("reference_energies.json not found. Run compute_all_references first.")
        return

    with open(ref_file) as f:
        ref = json.load(f)

    # Load monomer-interferent binding energies
    sel_file = out_path / "selectivity_data.json"
    if not sel_file.exists():
        logger.error("selectivity_data.json not found. Run compute_selectivity_data first.")
        return

    with open(sel_file) as f:
        sel = json.load(f)

    mukasa = ref.get("mukasa2023", {})
    kbt = KB_KCAL * TEMPERATURE

    monomers = ["OPD", "MAA", "4VB", "APB", "ACM", "PYR"]
    interferents = ["Tyrosine", "Leucine", "Dopamine"]
    exp_if = {"OPD": 3.2, "MAA": 2.8, "4VB": 2.4, "APB": 2.0, "ACM": 1.5, "PYR": 1.2}

    results = {}

    for func in ["wb97xd", "wb97m_v"]:
        func_display = "ωB97XD" if "xd" in func else "ωB97M-V"
        func_key = func

        print(f"\n{'='*70}")
        print(f"  {func_display} — Stage 2 (결합에너지) vs Stage 3 (선택도)")
        print(f"{'='*70}")

        # Stage 2: template-monomer binding energy
        # reference_energies.json is from the CURRENT functional (last run)
        # We need both functionals' data — check if we have it
        # For now use the current reference_energies.json (which is ωB97M-V)
        # and note that ωB97XD results are from previous runs

        # Stage 2 binding energies
        stage2 = {}
        for m in monomers:
            key = f"phe_{m}"
            if key in mukasa and mukasa[key].get("bsse_dE") is not None:
                stage2[m] = mukasa[key]["bsse_dE"]

        # Stage 3: selectivity using monomer-interferent data
        stage3 = {}
        sel_data = sel.get(func_key, {})

        print(f"\n  Stage 3 선택도 계산:")
        print(f"  ΔΔE = ΔE(monomer-Phe) - ΔE(monomer-interferent)")
        print()
        print(f"  {'Monomer':<8} {'ΔE(m-Phe)':>10}", end="")
        for interf in interferents:
            print(f" {'ΔE(m-'+interf[:3]+')':>12}", end="")
        print(f" {'avg ΔΔE':>10} {'log(S)':>10}")
        print(f"  {'-'*80}")

        for m in monomers:
            e_template = stage2.get(m)
            if e_template is None:
                continue

            ddE_list = []
            interf_energies = []
            for interf in interferents:
                key = f"{m}_{interf}"
                data = sel_data.get(key, {})
                e_interf = data.get("bsse_dE")
                interf_energies.append(e_interf)
                if e_interf is not None:
                    ddE = e_template - e_interf
                    ddE_list.append(ddE)

            if ddE_list:
                avg_ddE = np.mean(ddE_list)
                avg_logS = avg_ddE / kbt
            else:
                avg_ddE = None
                avg_logS = None

            stage3[m] = {"avg_ddE": avg_ddE, "avg_logS": avg_logS}

            print(f"  {m:<8} {e_template:>+10.3f}", end="")
            for e in interf_energies:
                if e is not None:
                    print(f" {e:>+12.3f}", end="")
                else:
                    print(f" {'N/A':>12}", end="")
            if avg_ddE is not None:
                print(f" {avg_ddE:>+10.3f} {avg_logS:>+10.2f}")
            else:
                print(f" {'N/A':>10} {'N/A':>10}")

        # Rankings
        s2_sorted = sorted(stage2.items(), key=lambda x: x[1])
        s3_sorted = sorted(
            [(m, v) for m, v in stage3.items() if v["avg_ddE"] is not None],
            key=lambda x: x[1]["avg_ddE"]
        )

        print(f"\n  순위 비교:")
        print(f"  {'순위':<4} {'실험 IF':>15} {'Stage2(결합에너지)':>22} {'Stage3(선택도)':>20}")
        exp_sorted = sorted(exp_if.items(), key=lambda x: -x[1])
        for i in range(min(6, len(s2_sorted))):
            e = exp_sorted[i] if i < len(exp_sorted) else ("-", 0)
            s2 = s2_sorted[i][0] if i < len(s2_sorted) else "-"
            s3 = s3_sorted[i][0] if i < len(s3_sorted) else "-"
            print(f"  {i+1:<4} {e[0]+'('+str(e[1])+')':>15} {s2:>22} {s3:>20}")

        s2_names = [x[0] for x in s2_sorted]
        s3_names = [x[0] for x in s3_sorted]
        if "OPD" in s2_names and "PYR" in s2_names:
            print(f"\n  Stage 2: OPD={s2_names.index('OPD')+1}위, PYR={s2_names.index('PYR')+1}위")
        if "OPD" in s3_names and "PYR" in s3_names:
            print(f"  Stage 3: OPD={s3_names.index('OPD')+1}위, PYR={s3_names.index('PYR')+1}위")

        results[func_key] = {
            "stage2": {m: e for m, e in stage2.items()},
            "stage3": {m: v for m, v in stage3.items()},
            "stage2_ranking": [x[0] for x in s2_sorted],
            "stage3_ranking": [x[0] for x in s3_sorted],
        }

    # Save rankings
    ranking_file = out_path / "ranking_comparison.json"
    with open(ranking_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "exp_if": exp_if,
            "results": results,
        }, f, indent=2)
    logger.info(f"Rankings saved to {ranking_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true",
                        help="Compute monomer-interferent binding energies")
    parser.add_argument("--rank", action="store_true",
                        help="Compute and display rankings (requires prior --compute)")
    parser.add_argument("--all", action="store_true",
                        help="Compute + rank")
    args = parser.parse_args()

    if args.all or args.compute:
        compute_selectivity_data()
    if args.all or args.rank:
        compute_rankings()
    if not (args.all or args.compute or args.rank):
        print("Usage:")
        print("  python -m validation.compute_selectivity --compute   # Run DFT calculations")
        print("  python -m validation.compute_selectivity --rank      # Show rankings")
        print("  python -m validation.compute_selectivity --all       # Both")

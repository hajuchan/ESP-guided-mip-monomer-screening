"""
Stage 3: Selectivity Calculation and Ranking
=============================================
Computes selectivity scores based on differential binding energies
between template and interferents.

Ref: Mukasa et al., Adv. Mater. 2023 — S ∝ exp(ΔE / kB·T)
"""

import json
import logging
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    TEMPLATE_SMILES,
    MONOMER_LIBRARY,
    INTERFERENT_LIBRARY,
    SOLVENTS,
    STAGE3_TOP_N,
    N_WORKERS,
    KB_KCAL,
    TEMPERATURE,
    HARTREE_TO_KCAL,
    OUTPUT_DIR,
    OUTPUT_DIRS,
    SOLVENT_STRATEGY,
    SYNTHESIS_SOLVENT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage3] %(message)s")
logger = logging.getLogger(__name__)


def aggregate_solvent_energy(solvent_data: dict, strategy: str = None) -> tuple:
    """Select a single bsse_dE value from multi-solvent DFT results.

    Args:
        solvent_data: {solvent_name: {"bsse_dE": float, ...}}
        strategy: "synthesis_match", "minimum", "average", "worst"

    Returns:
        (selected_bsse_dE, selected_solvent_name)
    """
    strategy = strategy or SOLVENT_STRATEGY
    energies = {s: v["bsse_dE"] for s, v in solvent_data.items()}

    if not energies:
        return 0.0, "N/A"

    if strategy == "synthesis_match":
        solvent = SYNTHESIS_SOLVENT
        if solvent in energies:
            return energies[solvent], solvent
        # Fallback to first available solvent
        first = next(iter(energies))
        logger.warning(f"Synthesis solvent '{solvent}' not found, using '{first}'")
        return energies[first], first

    elif strategy == "minimum":
        best_s = min(energies, key=energies.get)
        return energies[best_s], best_s

    elif strategy == "average":
        avg = sum(energies.values()) / len(energies)
        return round(avg, 3), "average"

    elif strategy == "worst":
        worst_s = max(energies, key=energies.get)
        return energies[worst_s], worst_s

    else:
        raise ValueError(f"Unknown solvent strategy: {strategy}")


def compute_interferent_binding(template_smiles: str,
                                monomer_names: list[str],
                                monomer_library: dict,
                                interferent_library: dict,
                                solvents: dict,
                                output_dir: str) -> dict:
    """Compute DFT binding energies for interferent-monomer pairs.

    Uses the same DFT protocol as stage2.
    Returns {interferent: {monomer: {solvent: bsse_dE}}}
    """
    from .stage2_dft import compute_dft_binding
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # ── Load existing results for skip logic ──
    cache_path = Path(output_dir) / "stage3_interferent_dft.json"
    results = {}
    if cache_path.exists():
        with open(cache_path) as f:
            results = json.load(f)

    tasks = []
    skip_count = 0
    for interf_name, interf_smiles in interferent_library.items():
        for m_name in monomer_names:
            m_smiles = monomer_library[m_name]
            for s_name, eps in solvents.items():
                if (interf_name in results
                        and m_name in results[interf_name]
                        and s_name in results[interf_name][m_name]):
                    skip_count += 1
                else:
                    tasks.append((interf_name, m_name, m_smiles, interf_smiles, s_name, eps))

    total = len(interferent_library) * len(monomer_names) * len(solvents)
    logger.info(f"Computing {len(tasks)} interferent-monomer DFT jobs "
                f"({skip_count} skipped, {total} total)")

    if tasks:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {}
            for interf_name, m_name, m_smiles, interf_smiles, s_name, eps in tasks:
                fut = executor.submit(
                    compute_dft_binding,
                    m_name, m_smiles, interf_smiles, s_name, eps
                )
                futures[fut] = (interf_name, m_name, s_name)

            for future in as_completed(futures):
                interf_name, m_name, s_name = futures[future]
                res = future.result()
                if res["success"]:
                    results.setdefault(interf_name, {}).setdefault(m_name, {})[s_name] = \
                        res["bsse_dE_kcal"]
                    logger.info(
                        f"  {interf_name}/{m_name}/{s_name}: "
                        f"bsse={res['bsse_dE_kcal']:+.3f} kcal/mol"
                    )
                    # Incremental save
                    with open(cache_path, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    logger.warning(f"  {interf_name}/{m_name}/{s_name}: FAILED")

    # Final save
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def compute_selectivity(template_dft: dict, interferent_dft: dict,
                        interferent_library: dict,
                        solvents: dict) -> pd.DataFrame:
    """Compute selectivity scores using the configured solvent strategy.

    S = exp(ΔE / (kB·T)) where ΔE = E_template - E_interferent
    Final score = mean of log(S) over all interferents.
    """
    kbt = KB_KCAL * TEMPERATURE  # kcal/mol
    logger.info(f"[Solvent strategy: {SOLVENT_STRATEGY}"
                f"{' → ' + SYNTHESIS_SOLVENT if SOLVENT_STRATEGY == 'synthesis_match' else ''}]")

    rows = []
    for m_name, solvent_data in template_dft.items():
        # Use solvent strategy to pick the representative energy
        e_template, selected_solvent = aggregate_solvent_energy(solvent_data)
        logger.info(f"  {m_name}: {selected_solvent} bsse_dE = {e_template:+.3f} kcal/mol")

        log_s_values = []
        for interf_name in interferent_library:
            # Get interferent energy using same strategy
            interf_solvent_data = interferent_dft.get(interf_name, {}).get(m_name, {})
            if not interf_solvent_data:
                logger.warning(f"Missing interferent data: {interf_name}/{m_name}")
                continue
            # For interferent, pick same solvent if synthesis_match, else apply strategy
            if SOLVENT_STRATEGY == "synthesis_match" and selected_solvent in interf_solvent_data:
                e_interf = interf_solvent_data[selected_solvent]
            elif isinstance(next(iter(interf_solvent_data.values())), dict):
                e_interf, _ = aggregate_solvent_energy(interf_solvent_data)
            else:
                # Legacy format: {solvent: bsse_dE_value}
                e_interf = interf_solvent_data.get(selected_solvent,
                           next(iter(interf_solvent_data.values())))

            delta_e = e_template - e_interf
            s_value = exp(delta_e / kbt)
            log_s = log(s_value) if s_value > 0 else 0.0
            log_s_values.append(log_s)

        avg_log_s = np.mean(log_s_values) if log_s_values else 0.0
        rows.append({
            "monomer": m_name,
            "selected_solvent": selected_solvent,
            "solvent_strategy": SOLVENT_STRATEGY,
            "bsse_dE_template": round(e_template, 3),
            "avg_log_S": round(avg_log_s, 3),
            "n_interferents": len(log_s_values),
        })

    df = pd.DataFrame(rows)
    return df


def plot_results(df: pd.DataFrame, output_dir: str):
    """Generate binding energy vs selectivity scatter plot and bar chart."""
    out_path = Path(output_dir)

    # ── Scatter: binding energy vs avg selectivity ───────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    solvent_col = "selected_solvent" if "selected_solvent" in df.columns else "solvent"
    for solvent in df[solvent_col].unique():
        sub = df[df[solvent_col] == solvent]
        ax.scatter(sub["bsse_dE_template"], sub["avg_log_S"],
                   label=solvent, s=80, alpha=0.7)
        for _, row in sub.iterrows():
            ax.annotate(row["monomer"], (row["bsse_dE_template"], row["avg_log_S"]),
                        fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("BSSE-corrected binding energy (kcal/mol)")
    ax.set_ylabel("Average log(S) — selectivity")
    ax.set_title("Binding Energy vs Selectivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "stage3_scatter.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved scatter plot: {out_path / 'stage3_scatter.png'}")

    # ── Bar chart: average selectivity per monomer ───────────────────
    avg_by_monomer = df.groupby("monomer")["avg_log_S"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_by_monomer.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_ylabel("Average log(S)")
    ax.set_title("Monomer Selectivity Ranking (averaged over solvents)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "stage3_bar.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved bar chart: {out_path / 'stage3_bar.png'}")


def run_stage3(template_smiles: str = None,
               monomer_library: dict = None,
               interferent_library: dict = None,
               solvents: dict = None,
               output_dir: str = OUTPUT_DIRS["stage3"]) -> list[str]:
    """Run selectivity analysis.

    Returns top-N monomer names by average selectivity.
    """
    template_smiles = template_smiles or TEMPLATE_SMILES
    monomer_library = monomer_library or MONOMER_LIBRARY
    interferent_library = interferent_library or INTERFERENT_LIBRARY
    solvents = solvents or SOLVENTS
    out_path = Path(output_dir)

    # Load stage2 template-monomer DFT results
    stage2_path = out_path.parent / "stage2" / "stage2_dft.json"
    if not stage2_path.exists():
        raise FileNotFoundError(f"Stage 2 results not found: {stage2_path}")
    with open(stage2_path) as f:
        template_dft = json.load(f)

    monomer_names = list(template_dft.keys())
    logger.info(f"Stage 3: Selectivity for {len(monomer_names)} monomers, "
                f"{len(interferent_library)} interferents")

    # Compute interferent binding energies (with skip logic for completed pairs)
    # Always call compute_interferent_binding — it loads cache internally
    # and only computes missing pairs
    interferent_dft = compute_interferent_binding(
        template_smiles, monomer_names, monomer_library,
        interferent_library, solvents, output_dir
        )

    # Compute selectivity
    df = compute_selectivity(template_dft, interferent_dft,
                             interferent_library, solvents)

    # Save CSV
    csv_path = out_path / "stage3_selectivity.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved: {csv_path}")

    # Plot
    plot_results(df, output_dir)

    # Rank monomers by average selectivity across solvents
    ranking = (df.groupby("monomer")["avg_log_S"]
               .mean()
               .sort_values(ascending=False))
    top_names = ranking.head(STAGE3_TOP_N).index.tolist()

    logger.info("Selectivity ranking:")
    for i, (name, score) in enumerate(ranking.items(), 1):
        marker = " <<<" if name in top_names else ""
        logger.info(f"  {i}. {name:>10s}: avg_log(S) = {score:+.3f}{marker}")

    # Save top monomers
    with open(out_path / "stage3_top.json", "w") as f:
        json.dump(top_names, f, indent=2)

    return top_names


if __name__ == "__main__":
    run_stage3()

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
    N_WORKERS,
    KB_KCAL,
    TEMPERATURE,
    HARTREE_TO_KCAL,
    OUTPUT_DIR,
    OUTPUT_DIRS,
    SOLVENT_STRATEGY,
    SYNTHESIS_SOLVENT,
    STAGE3_TOP_N,
    HBOND_DOMINANCE_THRESHOLD,
    CAVITY_CORRECTION,
    CAVITY_ALPHA,
    CAVITY_BETA,
    compute_molecular_volume,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Stage3] %(message)s")
logger = logging.getLogger(__name__)


# ── Global porogen selection (single optimal solvent, fixed system-wide) ──
# Literature-grounded (van Wissen 2025; Vasapollo 2011; Del Sole 2009;
# Suryana 2021; Liu 2021; Rosengren 2009): a MIP polymerises in ONE porogen,
# so the solvent is a formulation-level (global) choice — NOT per monomer.
# Because Stage 2 evaluates template+monomer+complex in the SAME PCM dielectric,
# the strongest (most-negative) in-solvent BSSE binding coincides with the
# least-screened, lowest-ε aprotic porogen — the two literature camps converge.
try:
    from .config import POROGEN_TIE_TAU          # kcal/mol tie band (DFT+PCM noise)
except ImportError:
    POROGEN_TIE_TAU = 1.0
try:
    from .config import POROGEN_LOW_EPS_WARN     # ε below which polar templates get a solubility flag
except ImportError:
    POROGEN_LOW_EPS_WARN = 3.0

# Protic solvents disrupt template–monomer H-bonds → hard-gated for H-bond MIPs.
_PROTIC_SOLVENTS = {
    "water", "h2o", "methanol", "meoh", "ethanol", "etoh", "n-propanol",
    "propanol", "1-propanol", "2-propanol", "isopropanol", "ipa",
    "n-butanol", "butanol", "formic acid", "acetic acid", "formamide",
    "ethylene glycol", "glycerol",
}


def _is_protic(solvent_name: str) -> bool:
    return solvent_name.strip().lower() in _PROTIC_SOLVENTS


def select_global_porogen(template_dft: dict, solvents: dict,
                          template_smiles: str = None, k: int = None) -> tuple:
    """Pick ONE global porogen from the per-solvent × per-monomer BSSE matrix.

    Algorithm (literature-grounded — see refs above):
      0. matrix hygiene: solvents present for the candidate monomers
      1. regime: H-bond-driven vs dispersion (template + best monomer, HBD+HBA)
      2. T_k = top-k monomers by solvent-averaged binding (k=STAGE3_TOP_N)
      3. Score(s) = mean over T_k of E[m][s]  (more negative = stronger)
      4. argmin Score; if H-bond-driven, HARD-GATE out protic solvents
      5. dielectric tie-breaker: within Score-tie band pick LOWEST ε
      6. solubility sanity flag for very low-ε porogen + polar template
      7. k=1 cross-check (single best complex)

    Returns (porogen_name, info_dict).
    """
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    template_smiles = template_smiles or TEMPLATE_SMILES
    k = k or STAGE3_TOP_N

    # STEP 0: build E[m][s] matrix from bsse_dE
    monomers = list(template_dft.keys())
    E = {}
    for m in monomers:
        E[m] = {}
        for s, v in template_dft[m].items():
            if isinstance(v, dict) and v.get("bsse_dE") is not None:
                try:
                    E[m][s] = float(v["bsse_dE"])
                except (TypeError, ValueError):
                    pass
    monomers = [m for m in monomers if E[m]]
    if not monomers:
        return (SYNTHESIS_SOLVENT, {"error": "no binding-energy matrix"})
    # solvents present for every candidate monomer (comparable set); else union
    solvent_names = [s for s in solvents if all(s in E[m] for m in monomers)]
    if not solvent_names:
        alls = set()
        for m in monomers:
            alls.update(E[m].keys())
        solvent_names = sorted(alls)
    if not solvent_names:
        return (SYNTHESIS_SOLVENT, {"error": "no solvent data"})

    # STEP 2: T_k by solvent-averaged binding (avoids circularity)
    Ebar = {m: sum(E[m][s] for s in solvent_names if s in E[m])
               / len([s for s in solvent_names if s in E[m]])
            for m in monomers if any(s in E[m] for s in solvent_names)}
    ranked = sorted(Ebar, key=Ebar.get)          # most negative first
    T_k = ranked[:max(1, k)]
    m1 = ranked[0]

    # STEP 3: Score(s) = top-k mean; Score1 = k=1 cross-check
    Score = {}
    for s in solvent_names:
        vals = [E[m][s] for m in T_k if s in E[m]]
        if vals:
            Score[s] = sum(vals) / len(vals)
    Score1 = {s: E[m1][s] for s in solvent_names if s in E[m1]}
    if not Score:
        return (solvent_names[0], {"error": "no scores", "porogen": solvent_names[0]})

    # STEP 1: regime (template + best monomer H-bond count)
    n_hb = 0
    t_mol = Chem.MolFromSmiles(template_smiles) if template_smiles else None
    if t_mol is not None:
        n_hb += Lipinski.NumHDonors(t_mol) + Lipinski.NumHAcceptors(t_mol)
    m1_smiles = MONOMER_LIBRARY.get(m1)
    m1_mol = Chem.MolFromSmiles(m1_smiles) if m1_smiles else None
    if m1_mol is not None:
        n_hb += Lipinski.NumHDonors(m1_mol) + Lipinski.NumHAcceptors(m1_mol)
    hbond_driven = n_hb >= HBOND_DOMINANCE_THRESHOLD

    # STEP 4: primary pick + protic hard gate
    candidates = list(Score.keys())
    s_raw = min(candidates, key=Score.get)
    gate_applied = False
    if hbond_driven:
        aprotic = [s for s in candidates if not _is_protic(s)]
        if aprotic and _is_protic(s_raw):
            candidates = aprotic
            gate_applied = True
    s_gated = min(candidates, key=Score.get)

    # STEP 5: dielectric tie-breaker (within noise band, lowest ε wins)
    min_score = Score[s_gated]
    band = [s for s in candidates if Score[s] - min_score <= POROGEN_TIE_TAU]
    s_star = min(band, key=lambda s: solvents.get(s, float("inf")))

    # STEP 6: solubility sanity flag
    eps_star = solvents.get(s_star)
    warning = None
    template_polar = (t_mol is not None and
                      (Lipinski.NumHDonors(t_mol) + Lipinski.NumHAcceptors(t_mol)) >= 2)
    if eps_star is not None and eps_star < POROGEN_LOW_EPS_WARN and template_polar:
        higher = sorted((s for s in candidates
                         if solvents.get(s, 0) >= POROGEN_LOW_EPS_WARN),
                        key=Score.get)
        alt = higher[0] if higher else None
        warning = (f"porogen '{s_star}' (ε={eps_star}) is very low-polarity but the "
                   f"template looks polar/ionizable — verify solubility"
                   + (f"; practical fallback: '{alt}' (ε={solvents.get(alt)})" if alt else ""))

    # STEP 7: k=1 cross-check
    s_k1 = min(Score1, key=Score1.get) if Score1 else s_star

    info = {
        "porogen": s_star,
        "epsilon": eps_star,
        "regime": "hbond" if hbond_driven else "dispersion",
        "protic_gate_applied": gate_applied,
        "T_k": T_k,
        "m1_single_best_complex": m1,
        "score_table_topk_mean": {s: round(Score[s], 3) for s in Score},
        "score_table_k1": {s: round(Score1[s], 3) for s in Score1},
        "tie_band": band,
        "tie_tau_kcal": POROGEN_TIE_TAU,
        "k1_solvent": s_k1,
        "k1_agrees_with_topk": s_k1 == s_star,
        "solubility_warning": warning,
        "k": k,
        "method": ("top-k mean in-solvent BSSE binding, argmin + protic H-bond gate "
                   "+ low-ε tie-breaker (van Wissen 2025; Suryana 2021; Liu 2021; "
                   "Del Sole 2009)"),
    }
    return s_star, info


def aggregate_solvent_energy(solvent_data: dict, strategy: str = None,
                             forced_solvent: str = None) -> tuple:
    """Select a single bsse_dE value from multi-solvent DFT results.

    Args:
        solvent_data: {solvent_name: {"bsse_dE": float, ...}}
        strategy: "global_optimal", "synthesis_match", "minimum", "average", "worst"
        forced_solvent: for "global_optimal"/"synthesis_match", use this solvent
            (the globally-selected porogen) instead of SYNTHESIS_SOLVENT.

    Returns:
        (selected_bsse_dE, selected_solvent_name)
    """
    strategy = strategy or SOLVENT_STRATEGY
    energies = {s: v["bsse_dE"] for s, v in solvent_data.items()}

    if not energies:
        return 0.0, "N/A"

    if strategy in ("synthesis_match", "global_optimal"):
        solvent = forced_solvent or (SYNTHESIS_SOLVENT
                                     if strategy == "synthesis_match" else None)
        if solvent and solvent in energies:
            return energies[solvent], solvent
        # Fallback to first available solvent
        first = next(iter(energies))
        logger.warning(f"Porogen '{solvent}' not found, using '{first}'")
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
    from .stage1_xtb import (
        smiles_to_mol3d, generate_docked_orientations,
        screen_orientations_sp, optimize_top_candidates,
        _adaptive_n_orientations,
    )
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
        for interf_name, m_name, m_smiles, interf_smiles, s_name, eps in tasks:
            try:
                # Same protocol as Stage 2: xTB docking → DFT
                # Interferent acts as "template" in docking
                interf_mol = smiles_to_mol3d(interf_smiles)
                mono_mol = smiles_to_mol3d(m_smiles)

                n_orient = _adaptive_n_orientations(interf_mol, mono_mol)
                orientations = generate_docked_orientations(
                    interf_mol, mono_mol, n_orientations=n_orient,
                )
                if len(orientations) == 0:
                    logger.warning(f"  {interf_name}/{m_name}: all orientations clashed")
                    continue

                top = screen_orientations_sp(interf_mol, orientations, top_n=10)
                opt = optimize_top_candidates(interf_mol, mono_mol, top)
                complex_mol = opt.get("best_complex_mol")

                res = compute_dft_binding(
                    m_name, m_smiles, interf_smiles, s_name, eps,
                    prebuilt_complex_mol=complex_mol,
                )

                if res["success"]:
                    results.setdefault(interf_name, {}).setdefault(m_name, {})[s_name] = \
                        res["bsse_dE_kcal"]
                    logger.info(
                        f"  {interf_name}/{m_name}/{s_name}: "
                        f"bsse={res['bsse_dE_kcal']:+.3f} kcal/mol"
                    )
                else:
                    logger.warning(f"  {interf_name}/{m_name}/{s_name}: DFT FAILED")
            except Exception as e:
                logger.warning(f"  {interf_name}/{m_name}/{s_name}: FAILED — {e}")

            # Incremental save after each pair
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def compute_selectivity(template_dft: dict, interferent_dft: dict,
                        interferent_library: dict,
                        solvents: dict,
                        template_smiles: str = None,
                        forced_solvent: str = None) -> pd.DataFrame:
    """Compute selectivity scores with cavity shape correction.

    Mukasa 2023 convention: E = |binding energy| (positive, larger = stronger)
      ΔE = E_Tar - E_Int_eff > 0 → selective for template
      S = exp(ΔE / kT), higher S → more selective

    Cavity correction (when CAVITY_CORRECTION=True):
      V_ratio = V_interferent / V_template
      f_cavity = V_ratio^β  (filling factor, 0~1)
      E_Int_eff = E_Int * f_cavity  (attenuated interferent binding)
      ΔE = E_Tar - E_Int_eff + α * max(V_template - V_interferent, 0)
    """
    kbt = KB_KCAL * TEMPERATURE  # kcal/mol
    template_smiles = template_smiles or TEMPLATE_SMILES

    _shown_solvent = (forced_solvent if SOLVENT_STRATEGY == "global_optimal"
                      else SYNTHESIS_SOLVENT if SOLVENT_STRATEGY == "synthesis_match"
                      else None)
    logger.info(f"[Solvent strategy: {SOLVENT_STRATEGY}"
                f"{' → ' + _shown_solvent if _shown_solvent else ''}]")
    logger.info(f"[Cavity correction: {CAVITY_CORRECTION}"
                f"{f', α={CAVITY_ALPHA}, β={CAVITY_BETA}' if CAVITY_CORRECTION else ''}]")

    # Compute molecular volumes for cavity correction
    vol_template = None
    vol_interferents = {}
    if CAVITY_CORRECTION:
        try:
            vol_template = compute_molecular_volume(template_smiles)
            logger.info(f"  Template volume: {vol_template:.1f} Å³")
            for interf_name, interf_smiles in interferent_library.items():
                vol_interferents[interf_name] = compute_molecular_volume(interf_smiles)
                logger.info(f"  {interf_name} volume: {vol_interferents[interf_name]:.1f} Å³"
                            f" (ratio={vol_interferents[interf_name]/vol_template:.2f})")
        except Exception as e:
            logger.warning(f"  Volume computation failed: {e}, disabling cavity correction")
            vol_template = None

    rows = []
    for m_name, solvent_data in template_dft.items():
        # Use solvent strategy to pick the representative energy
        e_template, selected_solvent = aggregate_solvent_energy(
            solvent_data, forced_solvent=forced_solvent)

        # Convert to absolute value (Mukasa convention: positive = stronger)
        E_Tar = abs(e_template)
        logger.info(f"  {m_name}: {selected_solvent} |E_Tar| = {E_Tar:.3f} kcal/mol")

        log_s_values = []
        for interf_name in interferent_library:
            # Get interferent energy using same strategy
            interf_solvent_data = interferent_dft.get(interf_name, {}).get(m_name, {})
            if not interf_solvent_data:
                logger.warning(f"Missing interferent data: {interf_name}/{m_name}")
                continue
            # For interferent, use the SAME (global/synthesis) solvent as the template
            if (SOLVENT_STRATEGY in ("synthesis_match", "global_optimal")
                    and selected_solvent in interf_solvent_data):
                e_interf = interf_solvent_data[selected_solvent]
            elif isinstance(next(iter(interf_solvent_data.values())), dict):
                e_interf, _ = aggregate_solvent_energy(
                    interf_solvent_data, forced_solvent=forced_solvent)
            else:
                e_interf = interf_solvent_data.get(selected_solvent,
                           next(iter(interf_solvent_data.values())))

            E_Int = abs(e_interf)

            # Cavity correction
            if CAVITY_CORRECTION and vol_template and interf_name in vol_interferents:
                v_interf = vol_interferents[interf_name]
                v_ratio = min(v_interf / vol_template, 1.0)
                f_cavity = v_ratio ** CAVITY_BETA
                E_Int_eff = E_Int * f_cavity
                delta_v = max(vol_template - v_interf, 0)
                delta_e = E_Tar - E_Int_eff + CAVITY_ALPHA * delta_v
            else:
                delta_e = E_Tar - E_Int

            s_value = exp(delta_e / kbt)
            log_s = log(s_value) if s_value > 0 else 0.0
            log_s_values.append(log_s)

        avg_log_s = np.mean(log_s_values) if log_s_values else 0.0
        rows.append({
            "monomer": m_name,
            "selected_solvent": selected_solvent,
            "solvent_strategy": SOLVENT_STRATEGY,
            "bsse_dE_template": round(e_template, 3),
            "E_Tar": round(E_Tar, 3),
            "avg_log_S": round(avg_log_s, 3),
            "n_interferents": len(log_s_values),
            "cavity_corrected": CAVITY_CORRECTION and vol_template is not None,
        })

    df = pd.DataFrame(rows)
    return df


def plot_results(df: pd.DataFrame, output_dir: str):
    """Generate binding energy vs selectivity scatter plot and bar chart."""
    out_path = Path(output_dir)

    # ── Scatter: binding strength vs selectivity (upper-right = best) ──
    fig, ax = plt.subplots(figsize=(10, 7))
    solvent_col = "selected_solvent" if "selected_solvent" in df.columns else "solvent"
    # Use |bsse_dE| so higher = stronger binding (x-axis right = stronger)
    for solvent in df[solvent_col].unique():
        sub = df[df[solvent_col] == solvent]
        x_vals = sub["bsse_dE_template"].abs()
        ax.scatter(x_vals, sub["avg_log_S"],
                   label=solvent, s=80, alpha=0.7)
        for _, row in sub.iterrows():
            ax.annotate(row["monomer"],
                        (abs(row["bsse_dE_template"]), row["avg_log_S"]),
                        fontsize=9, fontweight='bold', ha="left", va="bottom")
    ax.set_xlabel("|Binding Energy| (kcal/mol) — stronger →", fontsize=12)
    ax.set_ylabel("avg log(S) — more selective →", fontsize=12)
    ax.set_title("Binding Strength vs Selectivity (upper-right = best monomer)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "stage3_scatter.png", dpi=300)
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

    # ── Global porogen selection: ONE optimal solvent, fixed system-wide ──
    global_porogen = None
    if SOLVENT_STRATEGY == "global_optimal":
        global_porogen, porogen_info = select_global_porogen(
            template_dft, solvents, template_smiles)
        logger.info(f"[Global porogen] {global_porogen} "
                    f"(ε={porogen_info.get('epsilon')}, regime={porogen_info.get('regime')})"
                    f" — top-{porogen_info.get('k')} monomers {porogen_info.get('T_k')}")
        logger.info(f"  Score (top-k mean ΔE, kcal/mol): {porogen_info.get('score_table_topk_mean')}")
        if porogen_info.get("protic_gate_applied"):
            logger.info("  protic solvent rejected (H-bond MIP gate)")
        if not porogen_info.get("k1_agrees_with_topk"):
            logger.info(f"  note: single-best-complex prefers "
                        f"{porogen_info.get('k1_solvent')}")
        if porogen_info.get("solubility_warning"):
            logger.warning(f"  {porogen_info['solubility_warning']}")
        with open(out_path / "global_porogen.json", "w") as f:
            json.dump(porogen_info, f, indent=2, default=str)

    if interferent_library:
        # Compute interferent binding energies (with skip logic for completed pairs)
        # Always call compute_interferent_binding — it loads cache internally
        # and only computes missing pairs
        interferent_dft = compute_interferent_binding(
            template_smiles, monomer_names, monomer_library,
            interferent_library, solvents, output_dir
            )

        # Compute selectivity
        df = compute_selectivity(template_dft, interferent_dft,
                                 interferent_library, solvents,
                                 template_smiles=template_smiles,
                                 forced_solvent=global_porogen)

        # Save CSV (sorted by avg_log_S descending — most selective first)
        df = df.sort_values("avg_log_S", ascending=False).reset_index(drop=True)
        csv_path = out_path / "stage3_selectivity.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved: {csv_path}")

        # Plot
        plot_results(df, output_dir)

        # Rank monomers by average selectivity (no filtering — all pass to Stage 4)
        ranking = (df.groupby("monomer")["avg_log_S"]
                   .mean()
                   .sort_values(ascending=False))
        top_names = ranking.index.tolist()  # All monomers, ranked

        logger.info("Selectivity ranking (all passed to Stage 4):")
        for i, (name, score) in enumerate(ranking.items(), 1):
            logger.info(f"  {i}. {name:>10s}: avg_log(S) = {score:+.3f}")
    else:
        # No-interferent mode: skip selectivity, rank by DFT binding energy.
        be = {}
        if global_porogen:
            logger.info(f"No interferents — ranking by binding energy in the global "
                        f"porogen '{global_porogen}'.")
            for m, solvs in template_dft.items():
                v = solvs.get(global_porogen)
                if isinstance(v, dict) and v.get("bsse_dE") is not None:
                    be[m] = float(v["bsse_dE"])
                else:  # monomer missing the porogen → fall back to its mean
                    vals = [x["bsse_dE"] for x in solvs.values()
                            if isinstance(x, dict) and "bsse_dE" in x]
                    if vals:
                        be[m] = sum(vals) / len(vals)
        else:
            logger.info("No interferents configured — skipping selectivity; "
                        "ranking monomers by |DFT binding energy| (mean over solvents).")
            for m, solvs in template_dft.items():
                vals = [v["bsse_dE"] for v in solvs.values()
                        if isinstance(v, dict) and "bsse_dE" in v]
                if vals:
                    be[m] = sum(vals) / len(vals)
        # Most negative (strongest) binding first
        top_names = sorted(be, key=lambda m: be.get(m, 0.0))
        logger.info("Binding-energy ranking (all passed to Stage 4):")
        for i, name in enumerate(top_names, 1):
            logger.info(f"  {i}. {name:>10s}: BE = {be.get(name, float('nan')):+.3f} kcal/mol")

    # Save all ranked monomers (Stage 4 receives all)
    with open(out_path / "stage3_top.json", "w") as f:
        json.dump(top_names, f, indent=2)

    # Cross-linker recommendation (integrated from crosslinker.py)
    from .config import CROSSLINKER_LIBRARY, CROSSLINKER_SCREENING, CROSSLINKER_THRESHOLD
    if CROSSLINKER_SCREENING and CROSSLINKER_LIBRARY:
        logger.info("\n--- Cross-linker screening ---")
        try:
            from .crosslinker import run_crosslinker_screening
            cl_results = run_crosslinker_screening(
                template_smiles=template_smiles,
                output_dir=str(out_path))
            # Recommend best cross-linker (weakest template binding)
            if cl_results:
                best_cl = None
                best_be = float("-inf")
                for cl_name, solvs in cl_results.items():
                    for s_name, vals in solvs.items():
                        if isinstance(vals, dict):
                            be = vals.get("bsse_dE", vals.get("raw_dE", -999))
                            if be > best_be:
                                best_be = be
                                best_cl = cl_name
                if best_cl:
                    logger.info(f"  Recommended cross-linker: {best_cl} "
                                f"(template binding = {best_be:+.3f} kcal/mol, weakest)")
                    with open(out_path / "stage3_crosslinker.json", "w") as f:
                        json.dump({"recommended": best_cl,
                                   "template_binding": best_be}, f, indent=2)
        except Exception as e:
            logger.warning(f"  Cross-linker screening failed: {e}")

    return top_names


# Note: multi-monomer combination search is Stage 4 (stage4_mmsd.py); the
# MD-metric fallback (_optimize_combination) lives in stage5_md.py
# where Stage 4 MD metrics (contact freq, EBN) are available.


def _legacy_optimize_monomer_combination(template_dft, interferent_dft,
                                  interferent_library, template_smiles,
                                  monomer_library, ranked_names,
                                  n_select=3, output_dir=None):
    """Find the optimal combination of n_select monomers for MIP synthesis.

    Strategy (adapted from Bio Phase 3 MMSD for small-molecule MIP):
    1. Score each monomer by: selectivity × binding_diversity
    2. Greedy forward selection: add monomer that maximizes combo score
    3. Interference check: penalize monomers binding to the same site

    Combo score = sum(individual selectivity) × diversity_bonus × (1 - interference_penalty)

    Reference: Bio project Phase 3 MMSD (greedy forward selection + swap refinement)
    """
    from itertools import combinations

    if len(ranked_names) <= n_select:
        return {"selected": ranked_names, "score": 0, "method": "all_monomers"}

    # Collect per-monomer metrics
    mono_scores = {}
    mono_sites = {}
    for m in ranked_names:
        m_data = template_dft.get(m, {})
        # Get binding energy and site info
        be = None
        for solv, vals in m_data.items():
            if isinstance(vals, dict) and "bsse_dE" in vals:
                be = abs(vals["bsse_dE"])
                break
        if be is None:
            continue

        # Get binding site from Stage 1 (if available)
        s1_path = Path(output_dir).parent / "stage1" / "stage1_all.json" if output_dir else None
        site_idx = None
        if s1_path and s1_path.exists():
            with open(s1_path) as f:
                s1_data = json.load(f)
            for entry in s1_data:
                if entry.get("name") == m:
                    site_idx = entry.get("binding_site", {}).get("atom_idx")
                    break

        mono_scores[m] = be
        mono_sites[m] = site_idx

    if not mono_scores:
        return None

    # Greedy forward selection
    selected = []
    remaining = list(mono_scores.keys())

    for step in range(min(n_select, len(remaining))):
        best_m = None
        best_score = -float("inf")

        for candidate in remaining:
            trial = selected + [candidate]

            # 1. Sum of binding energies (moderate is better — inverted U)
            be_values = [mono_scores[m] for m in trial]
            be_median = np.median(list(mono_scores.values()))
            be_score = sum(np.exp(-0.5 * ((be - be_median) / 3.0)**2) for be in be_values)

            # 2. Site diversity bonus: different binding sites → better cavity
            sites = [mono_sites.get(m) for m in trial if mono_sites.get(m) is not None]
            n_unique = len(set(sites)) if sites else len(trial)
            diversity = n_unique / len(trial)

            # 3. Interference penalty: same site → competition
            interference = 0
            if len(sites) > 1:
                from collections import Counter
                site_counts = Counter(sites)
                interference = sum(c - 1 for c in site_counts.values()) / len(sites)

            combo_score = be_score * (1 + 0.5 * diversity) * (1 - 0.3 * interference)

            if combo_score > best_score:
                best_score = combo_score
                best_m = candidate

        if best_m:
            selected.append(best_m)
            remaining.remove(best_m)
            logger.info(f"  Step {step+1}: + {best_m} (score={best_score:.3f})")

    # Swap refinement: try replacing each position
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            current = selected[i]
            for candidate in remaining:
                trial = selected[:i] + [candidate] + selected[i+1:]
                be_values = [mono_scores[m] for m in trial]
                be_median = np.median(list(mono_scores.values()))
                be_score = sum(np.exp(-0.5 * ((be - be_median) / 3.0)**2) for be in be_values)
                sites = [mono_sites.get(m) for m in trial if mono_sites.get(m) is not None]
                n_unique = len(set(sites)) if sites else len(trial)
                diversity = n_unique / len(trial)
                from collections import Counter
                interference = 0
                if len(sites) > 1:
                    site_counts = Counter(sites)
                    interference = sum(c - 1 for c in site_counts.values()) / len(sites)
                trial_score = be_score * (1 + 0.5 * diversity) * (1 - 0.3 * interference)
                if trial_score > best_score * 1.01:  # 1% improvement threshold
                    selected[i] = candidate
                    remaining.append(current)
                    remaining.remove(candidate)
                    current = candidate
                    best_score = trial_score
                    improved = True
                    logger.info(f"  Swap: {selected} (score={best_score:.3f})")

    return {
        "selected": selected,
        "score": round(best_score, 3),
        "method": "greedy_forward_selection_with_swap",
        "binding_sites": {m: mono_sites.get(m) for m in selected},
        "binding_energies": {m: round(mono_scores.get(m, 0), 2) for m in selected},
    }


if __name__ == "__main__":
    run_stage3()

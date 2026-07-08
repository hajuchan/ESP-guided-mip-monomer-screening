"""
MMSD: Multi-Monomer Simultaneous Docking — combination search
==============================================================
Greedy Forward Selection + Swap Refinement + MMSD, with optional
NSGA-II / Bayesian optimizers.

Ported from ``Monomer_screening_in_Bio/code/pipeline/phase3_mmsd.py``
(Rajpal et al., Sci. Rep. 2024 — MMSD protocol) and adapted to this
project's *template-centered* GFN2-xTB docking engine.

Backend difference vs the Bio project
-------------------------------------
The Bio pipeline docks monomers sequentially into a **protein epitope**
with AutoDock4 GA. This project has no protein — the "receptor" is the
small-molecule **template**. So "sequential simultaneous docking" is
reinterpreted as: dock monomer 1 onto the template surface, xTB-optimise,
then dock monomer 2 onto (template+monomer1), and so on. Each step reuses
``stage1_xtb.optimize_top_candidates`` whose returned ``best_energy_kcal``
is exactly the incremental binding energy of adding that monomer to the
current complex.

Key concept (identical to Bio):
    delta_sum = mmsd_sum - smd_sum
      mmsd_sum < smd_sum  → synergy      (cooperative binding)
      mmsd_sum > smd_sum  → interference (steric clash)

Objective (size-normalised, identical to Bio):
    bo_objective = mmsd_per_monomer + BO_INTERFERENCE_PENALTY * max(0, delta_sum)

References
    Rajpal et al., Sci. Rep. 2024 — MMSD protocol
    Deb 2002 — NSGA-II ; Frazier 2018 / Garcia-Ortegon 2022 — Bayesian opt
"""

import json
import logging
from itertools import combinations  # noqa: F401  (kept for parity / future use)
from pathlib import Path

import numpy as np

from . import config as cfg
from .config import MONOMER_LIBRARY, CROSSLINKER_LIBRARY, TEMPLATE_SMILES

logger = logging.getLogger(__name__)


# ── Tunables (config.py overrides these via getattr) ───────────────────
def _c(name, default):
    return getattr(cfg, name, default)


MMSD_OPTIMIZER          = _c("MMSD_OPTIMIZER", "nsga2")      # nsga2|bayesian|greedy
MMSD_MIN_COMBO_SIZE     = _c("MMSD_MIN_COMBO_SIZE", 2)
MMSD_MAX_COMBO_SIZE     = _c("MMSD_MAX_COMBO_SIZE", 4)
MMSD_TOP_PC             = _c("MMSD_TOP_PC", 3)
MMSD_CANDIDATE_POOL     = _c("MMSD_CANDIDATE_POOL", 8)       # restrict search space
MMSD_N_ORIENTATIONS     = _c("MMSD_N_ORIENTATIONS", 60)      # per sequential step
MMSD_TOP_FOR_OPT        = _c("MMSD_TOP_FOR_OPT", 5)          # xTB-opt per step
MMSD_SURFACE_OFFSET     = _c("MMSD_SURFACE_OFFSET", 2.5)
BO_INTERFERENCE_PENALTY = _c("BO_INTERFERENCE_PENALTY", 0.3)
BAYESIAN_N_CALLS        = _c("BAYESIAN_N_CALLS", 30)
BAYESIAN_ACQUISITION    = _c("BAYESIAN_ACQUISITION", "EI")
NSGA_POP_SIZE           = _c("NSGA_POP_SIZE", 16)
NSGA_N_GEN              = _c("NSGA_N_GEN", 10)
MMSD_INCLUDE_CROSSLINKER            = _c("MMSD_INCLUDE_CROSSLINKER", True)
MMSD_ENFORCE_POLY_COMPAT            = _c("MMSD_ENFORCE_POLYMERIZATION_COMPATIBILITY", True)
MMSD_REQUIRE_CHEMISTRY_DIVERSITY    = _c("MMSD_REQUIRE_CHEMISTRY_DIVERSITY", True)
MMSD_MIN_CHEMISTRY_CLASSES          = _c("MMSD_MIN_CHEMISTRY_CLASSES", 2)
MMSD_MAX_PER_CLASS_COUNT            = _c("MMSD_MAX_PER_CLASS_COUNT", 2)
MMSD_CHEMISTRY_ENTROPY_WEIGHT       = _c("MMSD_CHEMISTRY_ENTROPY_WEIGHT", 0.3)


# ── Monomer chemistry metadata (class + polymerization) ────────────────
# Overridable via config.MONOMER_META. All library monomers polymerise by
# free-radical vinyl chemistry, so polymerization compatibility is trivially
# satisfied here; the meaningful diversity axis is the functional class.
_DEFAULT_MONOMER_META = {
    "MAA":     {"class": "acidic",      "polymerization": "vinyl"},
    "AA":      {"class": "acidic",      "polymerization": "vinyl"},
    "ITA":     {"class": "acidic",      "polymerization": "vinyl"},
    "4VBA":    {"class": "acidic",      "polymerization": "vinyl"},
    "4VP":     {"class": "basic",       "polymerization": "vinyl"},
    "2VP":     {"class": "basic",       "polymerization": "vinyl"},
    "VIM":     {"class": "basic",       "polymerization": "vinyl"},
    "OPD":     {"class": "basic",       "polymerization": "vinyl"},
    "APB":     {"class": "boronic",     "polymerization": "vinyl"},
    "4VB":     {"class": "boronic",     "polymerization": "vinyl"},
    "MAAD":    {"class": "neutral",     "polymerization": "vinyl"},
    "ACM":     {"class": "neutral",     "polymerization": "vinyl"},
    "NIPAM":   {"class": "neutral",     "polymerization": "vinyl"},
    "NVP":     {"class": "neutral",     "polymerization": "vinyl"},
    "PYR":     {"class": "neutral",     "polymerization": "vinyl"},
    "Styrene": {"class": "hydrophobic", "polymerization": "vinyl"},
}
MONOMER_META = {**_DEFAULT_MONOMER_META, **_c("MONOMER_META", {})}


def _meta(name):
    return MONOMER_META.get(name, {"class": "unknown", "polymerization": "vinyl"})


# ── Compatibility & diversity (ported from Bio utils_analysis) ─────────

def is_polymerization_compatible(monomers):
    """Return (ok, ptype, conflicts). One-pot copolymerisation requires a
    single polymerization chemistry across all monomers."""
    ptypes = {_meta(m)["polymerization"] for m in monomers}
    ptypes.discard("surface")  # surface-only monomers don't drive chemistry
    if len(ptypes) <= 1:
        return True, (next(iter(ptypes)) if ptypes else "vinyl"), []
    return False, None, [f"mixed polymerization chemistries: {sorted(ptypes)}"]


def check_chemistry_diversity(monomers):
    """Rule 1: >= MMSD_MIN_CHEMISTRY_CLASSES distinct classes.
    Rule 2: no single class exceeds MMSD_MAX_PER_CLASS_COUNT.
    Returns dict with validity, classes, entropy, normalized_entropy."""
    classes = [_meta(m)["class"] for m in monomers]
    uniq = sorted(set(classes))
    counts = {c: classes.count(c) for c in uniq}

    valid = (len(uniq) >= MMSD_MIN_CHEMISTRY_CLASSES and
             all(v <= MMSD_MAX_PER_CLASS_COUNT for v in counts.values()))

    # Shannon entropy of class distribution
    n = len(classes)
    probs = [v / n for v in counts.values()] if n else []
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    max_entropy = np.log(len(uniq)) if len(uniq) > 1 else 1.0
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    return {
        "valid": valid, "classes": uniq, "counts": counts,
        "entropy": float(entropy), "normalized_entropy": norm_entropy,
    }


def _get_compatible_crosslinkers(monomers):
    """Crosslinkers whose polymerization matches the functional set.
    All library monomers/crosslinkers are vinyl here, so returns all XLs."""
    ok, ptype, _ = is_polymerization_compatible(monomers)
    if not ok:
        return []
    return list(CROSSLINKER_LIBRARY.keys())


def _synthesizability_score(monomers):
    """Mean synthetic accessibility (higher = easier). Uses RDKit sascorer
    when available (SA in [1,10]; we return 10 - SA so higher is better),
    else a neutral constant. Range ~[0, 9]."""
    try:
        from rdkit import Chem
        from rdkit.Chem import RDConfig
        import os as _os
        import sys as _sys
        _sys.path.append(_os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer  # type: ignore
        vals = []
        for m in monomers:
            smi = MONOMER_LIBRARY.get(m) or CROSSLINKER_LIBRARY.get(m)
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                vals.append(10.0 - sascorer.calculateScore(mol))
        return float(np.mean(vals)) if vals else 5.0
    except Exception:
        return 5.0  # neutral when sascorer unavailable


# ── Lightweight xTB docking (no DFT charges — tractable for search) ────

def _simple_orientations(current_mol, monomer_mol, n_orient, offset, seed=42):
    """Generate clash-free surface orientations of monomer around current_mol.

    Unlike stage1's DFT-ESP orientation generator, this uses geometric surface
    placement + random rotations only — appropriate for a *combination search*
    where hundreds of complexes are evaluated. Final geometry accuracy comes
    from the subsequent xTB optimisation.
    """
    from scipy.spatial.transform import Rotation
    from scipy.spatial.distance import cdist
    from .stage1_xtb import mol_to_arrays, generate_surface_points

    t_num, t_pos = mol_to_arrays(current_mol)
    m_num, m_pos = mol_to_arrays(monomer_mol)
    m_centered = m_pos - m_pos.mean(axis=0)
    t_centroid = t_pos.mean(axis=0)

    surface_pts = generate_surface_points(current_mol)
    rng = np.random.RandomState(seed)
    offsets = [max(1.5, offset - 1.0), offset, offset + 1.0]
    n_per = max(1, n_orient // len(offsets))
    valid = []

    for off in offsets:
        idxs = rng.choice(len(surface_pts), n_per, replace=True)
        for idx in idxs:
            pt = surface_pts[idx]
            direction = pt - t_centroid
            norm = np.linalg.norm(direction)
            direction = (direction / norm) if norm > 1e-6 else np.array([1., 0., 0.])
            target = pt + direction * off
            rot = Rotation.from_rotvec(rng.randn(3) * np.pi)  # random orientation
            m_new = rot.apply(m_centered) + target
            if cdist(t_pos, m_new).min() < 1.2:
                continue
            valid.append((np.concatenate([t_num, m_num]),
                          np.concatenate([t_pos, m_new])))
    return valid


def _dock_step(current_mol, monomer_name, monomer_smiles, seed=42):
    """Dock one monomer onto current_mol and xTB-optimise.

    Returns (incremental_binding_energy_kcal, new_complex_mol) or (None, None).
    """
    from .stage1_xtb import (smiles_to_mol3d, screen_orientations_sp,
                             optimize_top_candidates)

    m_mol = smiles_to_mol3d(monomer_smiles)
    orientations = _simple_orientations(
        current_mol, m_mol, MMSD_N_ORIENTATIONS, MMSD_SURFACE_OFFSET, seed=seed)
    if not orientations:
        return None, None
    top = screen_orientations_sp(current_mol, orientations,
                                 top_n=MMSD_TOP_FOR_OPT)
    if not top:
        return None, None
    opt = optimize_top_candidates(current_mol, m_mol, top)
    be = opt.get("best_energy_kcal")
    complex_mol = opt.get("best_complex_mol")
    if be is None or not np.isfinite(be) or complex_mol is None:
        return None, None
    return float(be), complex_mol


# ── MMSD sequential evaluation of ONE composition ──────────────────────

def _run_single_mmsd(pc_id, functional_monomers, template_mol,
                     smd_be, seed=42):
    """Sequentially dock each functional monomer onto the template, then pick
    the best compatible crosslinker at the final step. Computes MMSD metrics.
    """
    monomer_energies = {}
    current = template_mol

    for step, m_name in enumerate(functional_monomers):
        smi = MONOMER_LIBRARY.get(m_name)
        if smi is None:
            logger.warning(f"  {pc_id}: {m_name} not in MONOMER_LIBRARY — excluded")
            return _excluded(pc_id, functional_monomers, monomer_energies)
        be, current = _dock_step(current, m_name, smi, seed=seed + step)
        if be is None:
            logger.warning(f"  {pc_id} step {step}: {m_name} docking failed — excluded")
            return _excluded(pc_id, functional_monomers, monomer_energies)
        monomer_energies[m_name] = round(be, 3)
        logger.info(f"    {pc_id} step {step}: +{m_name} → ΔE={be:+.2f} kcal/mol")

    best_xl, best_xl_be = None, 0.0
    xl_comparison = {}
    if MMSD_INCLUDE_CROSSLINKER:
        compatible = _get_compatible_crosslinkers(functional_monomers)
        for xl in compatible:
            be, _ = _dock_step(current, xl, CROSSLINKER_LIBRARY[xl],
                               seed=seed + 100)
            xl_comparison[xl] = round(be, 3) if be is not None else None
            if be is not None and (best_xl is None or be < best_xl_be):
                best_xl, best_xl_be = xl, be
        if best_xl is None:
            logger.warning(f"  {pc_id}: all crosslinkers failed — excluded")
            return _excluded(pc_id, functional_monomers, monomer_energies,
                             xl_comparison)

    all_monomers = list(functional_monomers) + ([best_xl] if best_xl else [])
    if best_xl:
        monomer_energies[best_xl] = round(best_xl_be, 3)

    mmsd_sum = sum(monomer_energies[m] for m in functional_monomers) + \
        (best_xl_be if best_xl else 0.0)
    smd_sum = sum(smd_be.get(m, 0.0) for m in all_monomers
                  if smd_be.get(m) is not None)
    delta_sum = mmsd_sum - smd_sum

    n_mono = len(all_monomers)
    mmsd_per_monomer = mmsd_sum / n_mono if n_mono else 0.0
    bo_objective = mmsd_per_monomer + BO_INTERFERENCE_PENALTY * max(0.0, delta_sum)

    return {
        "pc_id": pc_id,
        "monomers": all_monomers,
        "functional_monomers": list(functional_monomers),
        "n_functional": len(functional_monomers),
        "crosslinker": best_xl,
        "crosslinker_comparison": xl_comparison,
        "monomer_energies": monomer_energies,
        "mmsd_sum": round(mmsd_sum, 3),
        "smd_sum": round(smd_sum, 3),
        "delta_sum": round(delta_sum, 3),
        "mmsd_per_monomer": round(mmsd_per_monomer, 3),
        "bo_objective": round(bo_objective, 3),
        "synergy": delta_sum < 0,
        "excluded": False,
    }


def _excluded(pc_id, functional_monomers, monomer_energies, xl_comparison=None):
    return {
        "pc_id": pc_id, "monomers": list(functional_monomers),
        "functional_monomers": list(functional_monomers),
        "monomer_energies": monomer_energies, "crosslinker": None,
        "crosslinker_comparison": xl_comparison or {},
        "mmsd_sum": None, "smd_sum": None, "delta_sum": None,
        "mmsd_per_monomer": None, "bo_objective": None,
        "synergy": False, "excluded": True,
    }


# ── Optimizer 1: Greedy Forward Selection + Swap Refinement ────────────

def _run_greedy_mmsd(template_mol, available, smd_be, seed=42):
    """Forward selection (add best monomer while avg improves) + swap
    refinement (replace each slot if objective improves)."""
    all_results = []
    cache = {}

    def evaluate(trial):
        key = tuple(sorted(trial))
        if key in cache:
            return cache[key]
        if MMSD_ENFORCE_POLY_COMPAT and not is_polymerization_compatible(trial)[0]:
            cache[key] = None
            return None
        r = _run_single_mmsd(f"FWD_{'_'.join(trial)}", trial, template_mol,
                             smd_be, seed=seed)
        all_results.append(r)
        cache[key] = r
        return r

    # sort candidates by single-monomer BE (best first)
    smd_sorted = sorted(available, key=lambda m: smd_be.get(m, 0.0))
    logger.info(f"[greedy] pool ranked by SMD BE: {smd_sorted}")

    selected, best_avg, best_obj, best_result = [], float("inf"), float("inf"), None

    # ── Phase A: forward selection ──
    for round_k in range(MMSD_MAX_COMBO_SIZE):
        cands = [m for m in smd_sorted if m not in selected]
        if not cands:
            break
        rb_cand, rb_obj, rb_res = None, float("inf"), None
        for c in cands:
            r = evaluate(selected + [c])
            if not r or r.get("bo_objective") is None:
                continue
            if r["bo_objective"] < rb_obj:
                rb_obj, rb_cand, rb_res = r["bo_objective"], c, r
        if rb_cand is None:
            break
        new_avg = rb_res["mmsd_per_monomer"]
        if len(selected) >= MMSD_MIN_COMBO_SIZE and new_avg >= best_avg:
            logger.info(f"[greedy] STOP at {len(selected)}: adding {rb_cand} "
                        f"worsens avg ({new_avg:.3f} vs {best_avg:.3f})")
            break
        selected.append(rb_cand)
        best_avg, best_obj, best_result = new_avg, rb_obj, rb_res
        logger.info(f"[greedy] + {rb_cand} → {selected} "
                    f"(avg={best_avg:.3f}, obj={best_obj:.3f})")

    if not selected:
        logger.warning("[greedy] forward selection found no valid combination")
        return all_results

    # ── Phase B: swap refinement ──
    improved = True
    while improved:
        improved = False
        for i, cur in enumerate(selected):
            for alt in [m for m in smd_sorted if m not in selected]:
                r = evaluate(selected[:i] + [alt] + selected[i + 1:])
                if r and r.get("bo_objective") is not None and r["bo_objective"] < best_obj:
                    logger.info(f"[greedy] SWAP {cur}→{alt}: "
                                f"obj {best_obj:.3f}→{r['bo_objective']:.3f}")
                    selected[i], best_obj, best_result = alt, r["bo_objective"], r
                    improved = True
                    break
            if improved:
                break

    logger.info(f"[greedy] result: {selected} + "
                f"{best_result.get('crosslinker') if best_result else '?'} "
                f"(obj={best_obj:.3f})")
    return all_results


# ── Optimizer 2: Bayesian Optimization (skopt) ─────────────────────────

def _run_bayesian_mmsd(template_mol, available, smd_be, seed=42):
    try:
        from skopt import gp_minimize
        from skopt.space import Categorical
    except ImportError:
        logger.warning("[bayesian] skopt missing — falling back to greedy")
        return None

    choices = available + ["NONE"]
    space = [Categorical(choices, name=f"pos_{i}")
             for i in range(MMSD_MAX_COMBO_SIZE)]
    all_results, cache = [], {}

    def objective(raw):
        combo = []
        for m in raw:
            if m != "NONE" and m not in combo:
                combo.append(m)
        if len(combo) < MMSD_MIN_COMBO_SIZE:
            return 0.0
        if MMSD_ENFORCE_POLY_COMPAT and not is_polymerization_compatible(combo)[0]:
            return 0.0
        key = tuple(sorted(combo))
        if key in cache:
            return cache[key]
        r = _run_single_mmsd(f"BO_{len(all_results)+1}_{combo[0]}", combo,
                             template_mol, smd_be, seed=seed)
        all_results.append(r)
        obj = r.get("bo_objective") if r and r.get("bo_objective") is not None else 0.0
        cache[key] = obj
        logger.info(f"[bayesian] {len(all_results)}/{BAYESIAN_N_CALLS} "
                    f"{combo} → obj={obj:.3f}")
        return obj

    logger.info(f"[bayesian] GP optimisation, n_calls={BAYESIAN_N_CALLS}")
    gp_minimize(objective, space, n_calls=BAYESIAN_N_CALLS,
                acq_func=BAYESIAN_ACQUISITION, random_state=seed,
                n_initial_points=min(8, BAYESIAN_N_CALLS))
    return all_results


# ── Optimizer 3: NSGA-II multi-objective (pymoo) ───────────────────────

def _run_nsga2_mmsd(template_mol, available, smd_be, seed=42):
    """3 minimised objectives:
       1. affinity        = mmsd_per_monomer     (more negative better)
       2. interference    = delta_sum − entropy bonus  (synergy + diversity)
       3. synthesizability = −synth_score/10     (higher score better)
    Returns (all_results, pareto_front)."""
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.pntx import PointCrossover
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.repair.rounding import RoundingRepair
        from pymoo.optimize import minimize
    except ImportError:
        logger.warning("[nsga2] pymoo missing — falling back")
        return None, None

    pool = available + ["NONE"]
    n_choices = len(pool)
    all_results, cache = [], {}

    def decode(x):
        combo = []
        for idx in x:
            m = pool[int(idx)]
            if m != "NONE" and m not in combo:
                combo.append(m)
        return combo

    class Prob(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=MMSD_MAX_COMBO_SIZE, n_obj=3,
                             xl=0, xu=n_choices - 1, vtype=int)

        def _evaluate(self, x, out, *a, **k):
            combo = decode(x)
            BAD = [1e6, 1e6, 1e6]
            if len(combo) < MMSD_MIN_COMBO_SIZE:
                out["F"] = BAD
                return
            if MMSD_ENFORCE_POLY_COMPAT and not is_polymerization_compatible(combo)[0]:
                out["F"] = BAD
                return
            if MMSD_REQUIRE_CHEMISTRY_DIVERSITY and not check_chemistry_diversity(combo)["valid"]:
                out["F"] = BAD
                return
            key = tuple(sorted(combo))
            if key in cache:
                out["F"] = cache[key]
                return
            r = _run_single_mmsd(f"NSGA_{len(all_results)+1}_{combo[0]}", combo,
                                 template_mol, smd_be, seed=seed)
            if not r or r.get("bo_objective") is None:
                cache[key] = BAD
                out["F"] = BAD
                return

            div = check_chemistry_diversity(combo)
            ent_bonus = MMSD_CHEMISTRY_ENTROPY_WEIGHT * div["normalized_entropy"]
            f = [
                r["mmsd_per_monomer"],                       # affinity
                r["delta_sum"] - ent_bonus,                  # interference − diversity
                -_synthesizability_score(r["monomers"]) / 10.0,   # synthesizability
            ]
            r["objectives"] = f
            r["chemistry"] = div
            all_results.append(r)
            cache[key] = f
            out["F"] = f
            logger.info(f"[nsga2] {len(all_results)} {combo} → "
                        f"aff={f[0]:.2f} interf={f[1]:.2f} synth={-f[2]*10:.1f}")

    logger.info(f"[nsga2] pop={NSGA_POP_SIZE} gen={NSGA_N_GEN} "
                f"(≤ {NSGA_POP_SIZE*NSGA_N_GEN} evals)")
    res = minimize(
        Prob(),
        NSGA2(pop_size=NSGA_POP_SIZE, sampling=IntegerRandomSampling(),
              crossover=PointCrossover(n_points=2),
              mutation=PolynomialMutation(prob=0.15, eta=15, vtype=int,
                                          repair=RoundingRepair()),
              eliminate_duplicates=True),
        termination=("n_gen", NSGA_N_GEN), seed=seed, verbose=False)

    pareto = []
    X = np.atleast_2d(res.X) if res.X is not None else []
    for x in X:
        combo = decode(x)
        if len(combo) < MMSD_MIN_COMBO_SIZE:
            continue
        match = next((r for r in all_results
                      if tuple(sorted(r["functional_monomers"])) == tuple(sorted(combo))), None)
        if match:
            pareto.append(match)
    pareto.sort(key=lambda r: r.get("bo_objective", 0.0))
    logger.info(f"[nsga2] Pareto front: {len(pareto)} non-dominated solutions")
    return all_results, pareto


# ── Entry point ────────────────────────────────────────────────────────

def _load_smd_be(output_dir):
    """Single-monomer binding energies (SMD baseline). Prefer Stage 1 xTB
    dE (same engine as MMSD), fall back to Stage 2 DFT bsse_dE."""
    base = Path(output_dir).parent
    smd = {}
    s1 = base / "stage1" / "stage1_all.json"
    if s1.exists():
        for e in json.loads(s1.read_text()):
            if isinstance(e, dict) and e.get("success") and e.get("dE_kcal") is not None:
                smd[e["name"]] = float(e["dE_kcal"])
    if not smd:
        s2 = base / "stage2" / "stage2_dft.json"
        if s2.exists():
            for m, solvs in json.loads(s2.read_text()).items():
                vals = [v["bsse_dE"] for v in solvs.values()
                        if isinstance(v, dict) and "bsse_dE" in v]
                if vals:
                    smd[m] = float(np.mean(vals))
    return smd


def _load_stage3_ranking(output_dir):
    """Stage 3 porogen-fixed monomer ranking (stage3_top.json), if present."""
    p = Path(output_dir).parent / "stage3" / "stage3_top.json"
    if p.exists():
        try:
            names = json.loads(p.read_text())
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass
    return None


def run_mmsd(template_smiles=None, output_dir=None, optimizer=None, force=False):
    """Run MMSD combination search. Returns a result dict with top polymer
    compositions (``top_pcs``) and the flat evaluation list.

    If a valid ``mmsd_results.json`` already exists and ``force`` is False,
    it is reused (the search is expensive and has no internal item-level skip).
    """
    from .stage1_xtb import smiles_to_mol3d

    template_smiles = template_smiles or TEMPLATE_SMILES
    out = Path(output_dir) if output_dir else Path(cfg.OUTPUT_DIRS["stage4"])
    out.mkdir(parents=True, exist_ok=True)
    optimizer = optimizer or MMSD_OPTIMIZER

    # Resume: reuse a previous complete result unless forced.
    result_path = out / "mmsd_results.json"
    if not force and result_path.exists():
        try:
            cached = json.loads(result_path.read_text())
            if cached.get("top_pcs"):
                logger.info(f"MMSD: reusing existing {result_path.name} "
                            f"({len(cached['top_pcs'])} PCs) — use --force to recompute")
                return cached
        except Exception as e:
            logger.warning(f"MMSD: could not read cached results ({e}); recomputing")

    smd_be = _load_smd_be(out)
    if not smd_be:
        logger.error("MMSD: no single-monomer binding energies found "
                     "(run Stage 1/2 first)")
        return {"error": "no SMD baseline"}

    # candidate pool: prefer Stage 3 porogen-fixed ranking, else strongest SMD binders
    ranked3 = _load_stage3_ranking(out)
    if ranked3:
        pool = [m for m in ranked3
                if m in MONOMER_LIBRARY and m in smd_be][:MMSD_CANDIDATE_POOL]
        pool_src = "Stage 3 ranking"
    else:
        pool = [m for m in sorted(smd_be, key=lambda m: smd_be[m])
                if m in MONOMER_LIBRARY][:MMSD_CANDIDATE_POOL]
        pool_src = "SMD binding"
    logger.info(f"MMSD candidate pool ({len(pool)}, from {pool_src}): {pool}")

    template_mol = smiles_to_mol3d(template_smiles)

    # optimizer with auto-fallback
    order = {"nsga2": ["nsga2", "bayesian", "greedy"],
             "bayesian": ["bayesian", "greedy"],
             "greedy": ["greedy"]}.get(optimizer, ["greedy"])

    all_results, pareto, used = [], None, None
    for opt in order:
        if opt == "nsga2":
            res = _run_nsga2_mmsd(template_mol, pool, smd_be)
            if res[0] is not None:
                all_results, pareto, used = res[0], res[1], "nsga2"
                break
        elif opt == "bayesian":
            res = _run_bayesian_mmsd(template_mol, pool, smd_be)
            if res is not None:
                all_results, used = res, "bayesian"
                break
        else:
            all_results = _run_greedy_mmsd(template_mol, pool, smd_be)
            used = "greedy"
            break

    valid = [r for r in all_results
             if not r.get("excluded") and r.get("bo_objective") is not None]
    valid.sort(key=lambda r: r["bo_objective"])

    if pareto:
        top_pcs = pareto[:MMSD_TOP_PC]
    else:
        top_pcs = valid[:MMSD_TOP_PC]

    result = {
        "optimizer": used,
        "template_smiles": template_smiles,
        "candidate_pool": pool,
        "n_evaluations": len(all_results),
        "top_pcs": top_pcs,
        "pareto_front": pareto,
        "all_results": all_results,
        "selected": top_pcs[0]["monomers"] if top_pcs else [],
        "score": top_pcs[0]["bo_objective"] if top_pcs else None,
    }
    (out / "mmsd_results.json").write_text(
        json.dumps(result, indent=2, default=str))
    _plot(valid, out)

    logger.info(f"\nMMSD top {len(top_pcs)} compositions (optimizer={used}):")
    for i, pc in enumerate(top_pcs, 1):
        logger.info(f"  {i}. {pc['monomers']} obj={pc.get('bo_objective')} "
                    f"mmsd_sum={pc.get('mmsd_sum')} Δ={pc.get('delta_sum')} "
                    f"synergy={pc.get('synergy')}")
    return result


def _plot(valid, out):
    if not valid:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vals = [r["mmsd_sum"] for r in valid]
        best = np.minimum.accumulate(vals)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(vals) + 1), vals, "o", alpha=0.5, label="MMSD sum")
        ax.plot(range(1, len(best) + 1), best, "-r", lw=2, label="best so far")
        ax.set_xlabel("evaluation")
        ax.set_ylabel("MMSD sum (kcal/mol)")
        ax.set_title("MMSD combination search")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "mmsd_convergence.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.debug(f"MMSD plot skipped: {e}")


def run_stage4(template_smiles=None, output_dir=None, optimizer=None, force=False):
    """Stage 4 entry point: MMSD multi-monomer combination search.

    Thin wrapper over run_mmsd() so the pipeline orchestrator can call it
    like any other stage.
    """
    return run_mmsd(template_smiles=template_smiles, output_dir=output_dir,
                    optimizer=optimizer, force=force)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [MMSD] %(message)s")
    run_stage4()

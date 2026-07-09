"""
chemistry_aware.py — literature-driven MIP design corrections
=============================================================
Adds the highest-impact considerations the deep-research audit found MISSING
from the binding-energy-only pipeline. Each function cites the finding it fixes.

GAP 4+6  estimate_kass / recommend_stoichiometry
         Association-constant-driven stoichiometry (Wulff & Knorr, Bioseparation
         2001): Kass > ~900 M⁻¹ → 1:1 stoichiometric (all sites in cavity);
         below → excess monomer needed (non-specific sites). Replaces the fixed
         1:4 ratio. Also: excess FM raises non-specific binding (Nicholls 2021).

GAP 3    dimerization_energy
         Functional-monomer self-association (Zhang/Shimizu, Macromolecules 2010;
         Karlsson JACS 2009): MAA dimerises → less free monomer, but this
         "very efficiently suppresses background sites". Reported as a metric.

GAP 2    detect_ionizable / ionized_variants / charged_binding_compare
         Protonation/ionization state (ACS Appl. Polym. Mater. 2020: perrhenate
         binds PROTONATED 4-VP; IJMS 2014). Neutral-SMILES can misrank acid–base
         systems. We enumerate microstates and compare neutral vs ion-pair xTB
         binding, flagging when the charged form dominates.

GAP 8    bleeding_risk
         Incomplete template removal & bleeding (IJMS 2011): too-strong /
         inaccessible complexes leach residual template → quantification error;
         recommend dummy template.
"""

import logging
import math

logger = logging.getLogger(__name__)

# Physical constants (kcal/mol units)
_R = 1.987204e-3          # gas constant, kcal/(mol·K)
_T = 298.15               # K
# Rough translational/rotational entropy penalty of bimolecular association at
# ~1 M standard state (kcal/mol). DFT/xTB give ~ΔE (≈ΔH); real ΔG = ΔH − TΔS.
# This makes Kass a realistic proxy rather than an enthalpy-only overestimate.
_ASSOCIATION_ENTROPY_PENALTY = 8.0


# ── GAP 1 (partial): pose-ensemble free-energy proxy ──────────────────

def ensemble_binding(energies_kcal, T: float = _T) -> dict:
    """Boltzmann-weight the binding energies of the WHOLE pose ensemble instead
    of taking the single lowest-energy pose. Recognition is an ensemble / free-
    energy phenomenon whose site heterogeneity comes from this diversity
    (Nicholls JMR 1998; Karlsson JACS 2009; Nicholls Polymers 2021).

    Returns the best pose, a Boltzmann-weighted mean (free-energy-like), and
    n_effective_poses (participation ratio) as a binding-site heterogeneity proxy.
    """
    es = [float(e) for e in (energies_kcal or [])
          if e is not None and math.isfinite(float(e))]
    if not es:
        return {}
    e_min = min(es)
    ws = [math.exp(-(e - e_min) / (_R * T)) for e in es]   # relative to best
    Z = sum(ws)
    p = [w / Z for w in ws]
    weighted = sum(pi * e for pi, e in zip(p, es))
    n_eff = 1.0 / sum(pi * pi for pi in p)                 # thermally accessible poses
    return {
        "best_pose_kcal": round(e_min, 3),
        "boltzmann_mean_kcal": round(weighted, 3),
        "ensemble_spread_kcal": round(max(es) - e_min, 3),
        "n_effective_poses": round(n_eff, 2),
        "n_poses": len(es),
        "site_heterogeneity": ("high" if n_eff >= 3 else
                               "moderate" if n_eff >= 1.5 else "low"),
        "note": ("Boltzmann-weighted over the pose ensemble (Karlsson JACS 2009); "
                 "n_eff ≈ thermally accessible poses = binding-site heterogeneity "
                 "proxy. Full ensemble free energy needs many-molecule MD (GAP #1)."),
    }


# ── GAP 4+6: association constant → stoichiometry ──────────────────────

def estimate_kass(binding_kcal: float, entropy_penalty: float = None,
                  T: float = _T) -> dict:
    """Estimate the template–monomer association constant Kass (M⁻¹) from the
    computed complex binding energy and decide the imprinting stoichiometry.

    ΔG ≈ ΔE_binding + entropy_penalty ; Kass = exp(−ΔG/RT).
    Wulff & Knorr 2001: Kass > 900 M⁻¹ → 1:1 stoichiometric imprinting is
    feasible (every binding site inside the cavity, minimal heterogeneity);
    otherwise conventional excess-monomer imprinting (non-specific sites).
    """
    if entropy_penalty is None:
        entropy_penalty = _ASSOCIATION_ENTROPY_PENALTY
    dG = float(binding_kcal) + entropy_penalty          # kcal/mol
    if dG < 0:
        kass = math.exp(-dG / (_R * T))
    else:
        kass = math.exp(-dG / (_R * T))                 # <1, weak
    stoichiometric = kass > 900.0
    return {
        "binding_kcal": round(float(binding_kcal), 3),
        "assumed_entropy_penalty_kcal": entropy_penalty,
        "dG_kcal": round(dG, 3),
        "Kass_per_M": round(kass, 1),
        "log10_Kass": round(math.log10(kass), 2) if kass > 0 else None,
        "regime": "stoichiometric_1to1" if stoichiometric else "excess_monomer",
        "recommended_T_M_ratio": 1 if stoichiometric else 4,
        "rationale": (
            "Kass > 900 M⁻¹ → 1:1 stoichiometric: all recognition sites inside "
            "the cavity, low heterogeneity (Wulff & Knorr 2001)."
            if stoichiometric else
            "Kass ≤ 900 M⁻¹ → excess monomer needed; expect functional groups "
            "outside the cavity and non-specific sites (Nicholls 2021)."
        ),
        "caveat": ("Kass from ΔE + fixed entropy penalty is an order-of-magnitude "
                   "proxy (true ΔG needs ensemble free energy — audit GAP #1)."),
    }


# ── GAP 3: functional-monomer self-association (dimerisation) ──────────

def dimerization_energy(smiles: str, n_orientations: int = 24,
                        top_for_opt: int = 3) -> dict:
    """GFN2-xTB dimerisation energy ΔE_dim = E(dimer) − 2·E(monomer) (kcal/mol).
    Negative → self-associates (removes free monomer, but suppresses background
    sites — Zhang/Shimizu 2010). Reuses the MMSD template-centred docking with
    the monomer docked onto itself.
    """
    try:
        from .stage1_xtb import (smiles_to_mol3d, screen_orientations_sp,
                                 optimize_top_candidates)
        from .stage4_mmsd import _simple_orientations
    except Exception as e:
        return {"error": f"import failed: {e}"}
    try:
        mol = smiles_to_mol3d(smiles)
        orientations = _simple_orientations(mol, mol, n_orientations, 2.5, seed=7)
        if not orientations:
            return {"dimerization_kcal": None, "note": "no clash-free dimer pose"}
        top = screen_orientations_sp(mol, orientations, top_n=top_for_opt)
        if not top:
            return {"dimerization_kcal": None, "note": "SP screen empty"}
        opt = optimize_top_candidates(mol, mol, top)
        de = opt.get("best_energy_kcal")
        if de is None:
            return {"dimerization_kcal": None, "note": "optimisation failed"}
        return {
            "dimerization_kcal": round(float(de), 3),
            "self_associates": de < -1.0,
            "note": ("self-association lowers free-monomer conc. but suppresses "
                     "non-specific background sites (Zhang/Shimizu 2010)"
                     if de < -1.0 else "weak self-association"),
        }
    except Exception as e:
        return {"error": str(e)}


# ── GAP 2: ionization / protonation microstates ───────────────────────

def detect_ionizable(smiles: str) -> list:
    """Detect ionizable groups relevant to acid–base MIP recognition."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    patterns = {
        "carboxyl":      "[CX3](=O)[OX2H1]",
        "pyridine_N":    "[n;X2;H0]",
        "imidazole_N":   "[n;X2;H0]1cncc1",
        "primary_amine": "[NX3;H2;!$(NC=O);!$(N=*)]",
        "secondary_amine": "[NX3;H1;!$(NC=O);!$(N=*);!$(n)]",
        "boronic_acid":  "[BX3]([OX2H])[OX2H]",
        "sulfonic_acid": "[SX4](=O)(=O)[OX2H]",
    }
    found = []
    for name, smarts in patterns.items():
        q = Chem.MolFromSmarts(smarts)
        if q is not None and m.HasSubstructMatch(q):
            found.append(name)
    return found


def ionized_variants(smiles: str) -> list:
    """Return the physiologically relevant ionized microstate(s) as
    [(label, smiles, total_charge)]. Acids → conjugate base; N-bases → conjugate
    acid. Neutral form is always the pipeline default; this exposes the ion-pair
    state for GAP-2 comparison."""
    from rdkit import Chem
    out = []
    m0 = Chem.MolFromSmiles(smiles)
    if m0 is None:
        return out

    def _emit(label, mol, charge):
        try:
            Chem.SanitizeMol(mol)
            out.append((label, Chem.MolToSmiles(mol), charge))
        except Exception:
            pass

    # Deprotonate a carboxylic / sulfonic acid → anion
    for smarts, lbl in [("[CX3](=O)[OX2H1]", "carboxylate"),
                        ("[SX4](=O)(=O)[OX2H1]", "sulfonate")]:
        q = Chem.MolFromSmarts(smarts)
        m = Chem.MolFromSmiles(smiles)
        if q is not None and m is not None and m.HasSubstructMatch(q):
            mol = Chem.RWMol(m)
            match = m.GetSubstructMatch(q)
            o_idx = match[-1]                       # the -OH oxygen
            a = mol.GetAtomWithIdx(o_idx)
            a.SetFormalCharge(-1)
            a.SetNumExplicitHs(0)
            _emit(f"{lbl}(-1)", mol, -1)
            break

    # Protonate an aromatic pyridine/imidazole N or aliphatic amine → cation
    for smarts, lbl in [("[n;X2;H0]", "N-protonated(+1)"),
                        ("[NX3;H2;!$(NC=O);!$(N=*)]", "ammonium(+1)"),
                        ("[NX3;H1;!$(NC=O);!$(N=*);!$(n)]", "ammonium(+1)")]:
        q = Chem.MolFromSmarts(smarts)
        m = Chem.MolFromSmiles(smiles)
        if q is not None and m is not None and m.HasSubstructMatch(q):
            mol = Chem.RWMol(m)
            n_idx = m.GetSubstructMatch(q)[0]
            a = mol.GetAtomWithIdx(n_idx)
            a.SetFormalCharge(+1)
            a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
            _emit(lbl, mol, +1)
            break
    return out


def speciation_flags(template_smiles: str, monomer_smiles: str) -> dict:
    """GAP 2 summary for one template–monomer pair: which partners are ionizable
    and whether an ion-pair recognition mode should be considered (acid template
    + base monomer, or vice versa) — the case where neutral SMILES misranks."""
    t_ion = detect_ionizable(template_smiles)
    m_ion = detect_ionizable(monomer_smiles)
    t_acid = any(g in t_ion for g in ("carboxyl", "sulfonic_acid", "boronic_acid"))
    t_base = any(g in t_ion for g in ("pyridine_N", "imidazole_N",
                                      "primary_amine", "secondary_amine"))
    m_acid = any(g in m_ion for g in ("carboxyl", "sulfonic_acid", "boronic_acid"))
    m_base = any(g in m_ion for g in ("pyridine_N", "imidazole_N",
                                      "primary_amine", "secondary_amine"))
    ion_pair = (t_acid and m_base) or (t_base and m_acid)
    return {
        "template_ionizable": t_ion,
        "monomer_ionizable": m_ion,
        "ion_pair_recognition_likely": ion_pair,
        "monomer_ionized_variants": ionized_variants(monomer_smiles),
        "template_ionized_variants": ionized_variants(template_smiles),
        "warning": ("Acid–base pair: neutral-SMILES binding may misrank this "
                    "system — screen the ion-pair (charged) microstate "
                    "(ACS Appl. Polym. Mater. 2020)." if ion_pair else None),
    }


# ── GAP 8: template removal / bleeding risk ───────────────────────────

def bleeding_risk(removal_success_rate: float = None,
                  binding_kcal: float = None) -> dict:
    """Estimate template-bleeding risk. High binding strength and/or poor
    geometric removability → residual template that later leaches (IJMS 2011).
    removal_success_rate: fraction of VIP snapshots where template escaped
    (Stage 6); binding_kcal: template–monomer complex energy."""
    reasons = []
    score = 0.0
    if removal_success_rate is not None:
        if removal_success_rate < 0.34:
            score += 0.5
            reasons.append(f"low geometric removability ({removal_success_rate:.0%})")
    if binding_kcal is not None and binding_kcal < -15.0:
        score += 0.5
        reasons.append(f"very strong complex ({binding_kcal:.1f} kcal/mol) resists extraction")
    level = "high" if score >= 0.75 else "medium" if score >= 0.4 else "low"
    return {
        "bleeding_risk": level,
        "score": round(score, 2),
        "reasons": reasons,
        "recommendation": (
            "Consider a structural-analog DUMMY TEMPLATE and validate template "
            "removal by wash-cycle monitoring (IJMS 2011)."
            if level != "low" else
            "Standard exhaustive washing (Soxhlet) expected sufficient; still "
            "verify no residual-template bleeding during rebinding."),
    }

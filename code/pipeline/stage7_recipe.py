"""
Stage 7: Synthesis Recipe Generation
=====================================
Generates synthesis protocol based on pipeline results:
- Monomer ranking from Stage 6 VIP (or Stage 5 MD if VIP unavailable)
- Cross-linker recommendation from Stage 3
- Synthesis ratio from Stage 5 contact frequency
- Global porogen from Stage 3 (solvent memory)
- Protocol type: free-radical polymerization (default for organic monomers)

Output: JSON recipe + human-readable protocol
"""

import json
import logging
from pathlib import Path

from .config import (
    TEMPLATE_SMILES, TEMPLATE_NAME, MONOMER_LIBRARY,
    CROSSLINKER_LIBRARY, SOLVENTS, SYNTHESIS_SOLVENT,
    OUTPUT_DIR, OUTPUT_DIRS,
)

logger = logging.getLogger(__name__)


# Recipe protocol chemistry → the crosslinker polymerization type it REQUIRES.
# One-pot MIPs use a single polymerization chemistry (Liu 2017): a free-radical
# (vinyl) network cannot incorporate a silane sol-gel crosslinker and vice-versa.
_CHEM_TO_XL_POLY = {"free-radical": "vinyl", "sol-gel": "silane", "oxidative": None}
_CHEM_DEFAULT_XL = {"free-radical": "EGDMA", "sol-gel": "TEOS", "oxidative": None}


def _xl_polymerization(name):
    """Classify a crosslinker's polymerization chemistry from its SMILES,
    mirroring stage4_mmsd._detect_polymerization (Si→silane, non-aromatic
    C=C→vinyl, else oxidative)."""
    smi = CROSSLINKER_LIBRARY.get(name, "")
    if not smi:
        return "unknown"
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return "unknown"
        if m.HasSubstructMatch(Chem.MolFromSmarts("[Si]")):
            return "silane"
        if m.HasSubstructMatch(Chem.MolFromSmarts("[C;!a]=[C;!a]")):
            return "vinyl"
        return "oxidative"
    except Exception:
        return "unknown"


def _select_crosslinker(base, chemistry, mmsd_xl):
    """Pick a crosslinker whose chemistry matches the winning polymerization
    route. Priority:
      1. the crosslinker MMSD co-selected with the combo (matched by construction);
      2. the Stage-3 screening recommendation, IF compatible;
      3. the weakest-binding COMPATIBLE crosslinker from the full screening;
      4. a sensible per-chemistry default.
    Oxidative polymers self-crosslink → returns (None, reason)."""
    need = _CHEM_TO_XL_POLY.get(chemistry, "vinyl")
    if need is None:                       # oxidative → no molecular crosslinker
        return None, "oxidative self-crosslinking (no molecular crosslinker)"

    # 1) MMSD's own choice (already chemistry-matched with the combo)
    if mmsd_xl and _xl_polymerization(mmsd_xl) == need:
        return mmsd_xl, "MMSD combo (chemistry-matched)"

    # 2) Stage-3 recommendation, only if compatible with the winning chemistry
    rec = None
    cl_path = base / "stage3" / "stage3_crosslinker.json"
    if cl_path.exists():
        try:
            rec = json.load(open(cl_path)).get("recommended")
        except Exception:
            rec = None
    if rec and _xl_polymerization(rec) == need:
        return rec, "Stage 3 screening (compatible)"

    # 3) weakest-binding COMPATIBLE crosslinker from the full screening
    #    (least-negative avg dE = least template competition, chemistry-filtered)
    feat = base / "features" / "crosslinker.json"
    if feat.exists():
        try:
            data = json.load(open(feat))
            best, best_be = None, float("-inf")
            for name, solvs in data.items():
                if _xl_polymerization(name) != need:
                    continue
                bes = [v.get("bsse_dE", v.get("raw_dE")) for v in solvs.values()
                       if isinstance(v, dict) and
                       (v.get("bsse_dE") is not None or v.get("raw_dE") is not None)]
                if not bes:
                    continue
                avg = sum(bes) / len(bes)
                if avg > best_be:
                    best_be, best = avg, name
            if best:
                return best, "weakest-binding compatible (Stage 3 full screening)"
        except Exception:
            pass

    # 4) per-chemistry default
    return _CHEM_DEFAULT_XL.get(chemistry, "EGDMA"), "per-chemistry default"


def run_stage7(output_dir: str = None) -> dict:
    """Generate synthesis recipe from pipeline results."""
    if output_dir is None:
        output_dir = OUTPUT_DIRS.get("reports", f"{OUTPUT_DIR}/reports")
    base = Path(output_dir).parent
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Prefer the globally-selected porogen from Stage 3 (solvent memory);
    # fall back to the configured synthesis solvent.
    solvent = SYNTHESIS_SOLVENT
    porogen_path = base / "stage3" / "global_porogen.json"
    if porogen_path.exists():
        try:
            with open(porogen_path) as f:
                _p = json.load(f).get("porogen")
            if _p:
                solvent = _p
        except Exception:
            pass

    recipe = {
        "template": TEMPLATE_NAME,
        "template_smiles": TEMPLATE_SMILES,
        "solvent": solvent,
    }

    # ── Load VIP results (Stage 6) or MD (Stage 5) ──
    vip_path = base / "stage6" / "stage6_vip.json"
    s4_path = base / "stage5" / "stage5_md.json"

    # Rank monomers, cascading through the richest available source.
    monomer_ranking, src = [], None
    if vip_path.exists():
        vip = json.load(open(vip_path))
        ranked = [(m, d.get("vip_score", 0)) for m, d in vip.items()
                  if isinstance(d, dict) and d.get("vip_score", 0) > 0]
        ranked.sort(key=lambda x: -x[1])
        if ranked:
            monomer_ranking, src = ranked, "Stage 6 VIP"
    if not monomer_ranking and s4_path.exists():
        s4 = json.load(open(s4_path))
        if isinstance(s4, list):
            ranked = [(r["monomer"], r.get("contact_frequency", 0))
                      for r in s4 if r.get("contact_frequency", 0) > 0]
            ranked.sort(key=lambda x: -x[1])
            if ranked:
                monomer_ranking, src = ranked, "Stage 5 MD (contact frequency)"
    if not monomer_ranking:
        s3_path = base / "stage3" / "stage3_top.json"
        if s3_path.exists():
            names = json.load(open(s3_path))
            monomer_ranking = [(n, 0) for n in names]
            src = "Stage 3 selectivity"
    recipe["ranking_source"] = src

    if not monomer_ranking:
        logger.warning("No monomer ranking data found")
        return {"error": "No results to generate recipe"}

    # Top 3 monomers
    top3 = monomer_ranking[:3]
    recipe["top3_monomers"] = [
        {"rank": i+1, "name": m, "score": round(s, 3),
         "smiles": MONOMER_LIBRARY.get(m, "")}
        for i, (m, s) in enumerate(top3)
    ]

    # ── Polymerization chemistry (winning MMSD combo) ──
    # Determined here so the crosslinker can be kept chemically compatible.
    chemistry = "free-radical"
    mmsd_xl = None
    mmsd_path = base / "stage4" / "mmsd_results.json"
    if mmsd_path.exists():
        try:
            _top = (json.load(open(mmsd_path)).get("top_pcs") or [])
            if _top:
                chemistry = _top[0].get("synthesis_method") or "free-radical"
                mmsd_xl = _top[0].get("crosslinker")
        except Exception:
            pass
    recipe["polymerization"] = chemistry

    # ── Cross-linker (must match the polymerization chemistry) ──
    xl_name, xl_reason = _select_crosslinker(base, chemistry, mmsd_xl)
    recipe["crosslinker"] = xl_name
    recipe["crosslinker_smiles"] = (
        CROSSLINKER_LIBRARY.get(xl_name, "") if xl_name else "")
    recipe["crosslinker_source"] = xl_reason

    # ── Synthesis ratio ──
    # EBN-based (Yuan et al. 2024): more simultaneous binding sites (EBN↑) →
    # more monomer copies (direct proportion). Optional: inverse contact freq.
    from .config import MD_RATIO_METHOD
    if s4_path.exists():
        s4 = json.load(open(s4_path))
        if isinstance(s4, list):
            if MD_RATIO_METHOD == "ebn":
                ebn = {r["monomer"]: r.get("ebn_max", 0)
                       for r in s4 if r.get("ebn_max", 0) > 0}
                if ebn:
                    min_e = min(ebn.values())
                    recipe["synthesis_ratios"] = {m: max(1, round(v / min_e))
                                                  for m, v in ebn.items()}
                    recipe["ratio_method"] = "EBN direct (Yuan 2024)"
            if "synthesis_ratios" not in recipe:      # contact-inverse (option/fallback)
                contacts = {r["monomer"]: r.get("contact_frequency", 0)
                            for r in s4 if r.get("contact_frequency", 0) > 0}
                if contacts:
                    max_c = max(contacts.values())
                    recipe["synthesis_ratios"] = {m: round(max_c / c, 1)
                                                  for m, c in contacts.items()}
                    recipe["ratio_method"] = "inverse contact frequency"

    # Default ratio if not computed
    if "synthesis_ratios" not in recipe:
        recipe["synthesis_ratios"] = {m: 4.0 for m, _ in top3}
        recipe["ratio_method"] = "default (1:4)"

    # ── Protocol (chemistry-specific) ──
    best_monomer = top3[0][0] if top3 else "MAA"
    ratio = recipe["synthesis_ratios"].get(best_monomer, 4.0)
    xl = recipe["crosslinker"]

    # ── Chemistry-aware corrections (deep-research audit GAP 2/3/4+6/8) ──
    from . import chemistry_aware as _ca
    best_smi = MONOMER_LIBRARY.get(best_monomer, "")

    # Best-monomer template binding energy (Stage 2 DFT, in the chosen porogen)
    best_be = None
    s2dft = base / "stage2" / "stage2_dft.json"
    if s2dft.exists():
        try:
            s2 = json.load(open(s2dft)).get(best_monomer, {})
            if solvent in s2 and isinstance(s2[solvent], dict):
                best_be = s2[solvent].get("bsse_dE")
            else:
                vals = [v.get("bsse_dE") for v in s2.values()
                        if isinstance(v, dict) and v.get("bsse_dE") is not None]
                best_be = sum(vals) / len(vals) if vals else None
        except Exception:
            pass

    # GAP 4+6: association-constant-driven stoichiometry (overrides fixed 1:4)
    if best_be is not None:
        stoich = _ca.estimate_kass(best_be)
        recipe["stoichiometry"] = stoich
        ratio = float(stoich["recommended_T_M_ratio"])

    # GAP 2: ionization / ion-pair speciation
    recipe["speciation"] = _ca.speciation_flags(TEMPLATE_SMILES, best_smi)

    # GAP 3: functional-monomer self-association (xTB dimerisation)
    try:
        recipe["self_association"] = _ca.dimerization_energy(best_smi)
    except Exception as e:
        recipe["self_association"] = {"error": str(e)}

    # GAP 8: template bleeding / removability risk
    removal_rate = None
    vip_p = base / "stage6" / "stage6_vip.json"
    if vip_p.exists():
        try:
            v = json.load(open(vip_p)).get(best_monomer, {})
            removal_rate = (v.get("removal_rate")
                            or v.get("removal_success_rate"))
        except Exception:
            pass
    recipe["bleeding"] = _ca.bleeding_risk(removal_rate, best_be)

    # GAP 1 (partial): pose-ensemble heterogeneity of the best monomer (Stage 1)
    s1p = base / "stage1" / "stage1_all.json"
    if s1p.exists():
        try:
            for _e in json.load(open(s1p)):
                if _e.get("name") == best_monomer and _e.get("ensemble"):
                    recipe["ensemble"] = _e["ensemble"]
                    break
        except Exception:
            pass

    # GAP 1 (intermediate): MM/GBSA binding free energy from Stage 5 MD (best combo)
    s5c = base / "stage5" / "stage5_combination.json"
    if s5c.exists():
        try:
            _combos = json.load(open(s5c))
            if isinstance(_combos, list) and _combos:
                _mp = _combos[0].get("mmpbsa") or {}
                if _mp.get("dG_bind_kcal") is not None:
                    recipe["mmpbsa_dG"] = _mp
        except Exception:
            pass

    # GAP 5: crosslinker is an ACTIVE template-binding species, not inert filler
    # (Shoravi/Olsson IJMS 2014 — EGDMA often binds the template more than the FM).
    _mmsd_p = base / "stage4" / "mmsd_results.json"
    if _mmsd_p.exists():
        try:
            _pc = (json.load(open(_mmsd_p)).get("top_pcs") or [{}])[0]
            _me = _pc.get("monomer_energies") or {}
            _xl = _pc.get("crosslinker")
            _xl_be = _me.get(_xl)
            _fm_bes = [_me[m] for m in (_pc.get("functional_monomers") or [])
                       if m in _me]
            if _xl_be is not None and _fm_bes:
                recipe["crosslinker_binding"] = {
                    "crosslinker": _xl,
                    "xl_template_kcal": _xl_be,
                    "strongest_monomer_kcal": min(_fm_bes),
                    "xl_dominates": _xl_be <= min(_fm_bes),
                }
        except Exception:
            pass

    # Polymerization chemistry + crosslinker already resolved above.
    xl_disp = (f"{xl} ({CROSSLINKER_LIBRARY.get(xl, '')})" if xl
               else "none (self-crosslinking)")
    xl_ratio_disp = f"{ratio * 5:.0f}" if xl else "0 (self-crosslinking)"

    header = f"""=== MIP Synthesis Protocol ({chemistry}) ===

Template: {TEMPLATE_NAME} ({TEMPLATE_SMILES})
Monomer:  {best_monomer} ({MONOMER_LIBRARY.get(best_monomer, '')})
Cross-linker: {xl_disp}
Solvent:  {solvent}

Molar ratio (Template : Monomer : Cross-linker):
  1 : {ratio:.0f} : {xl_ratio_disp}
"""

    if chemistry.startswith("sol-gel"):
        steps = f"""
Protocol (sol-gel / silane condensation):
1. Dissolve {TEMPLATE_NAME} (1 mmol) and {best_monomer} ({ratio:.0f} mmol) in ethanol (10 mL)
2. Stir 30 min at RT → template–silane pre-complex
3. Add {xl} ({ratio * 5:.0f} mmol, e.g. TEOS), then H₂O ({ratio * 10:.0f} mmol) + HCl (cat., pH ≈ 3)
4. Stir 2 h at RT (hydrolysis) → age/gel 24–72 h at 40°C (condensation)
5. Dry to xerogel under vacuum at 60°C
6. Remove template by washing with MeOH/AcOH (9:1), repeat until absent
7. Dry under vacuum at 60°C"""
        removal = "MeOH/AcOH washing"
    elif chemistry.startswith("oxidative"):
        steps = f"""
Protocol (oxidative / chemical polymerization):
1. Dissolve {TEMPLATE_NAME} (1 mmol) and {best_monomer} ({ratio:.0f} mmol) in 0.1 M aqueous acid (10 mL)
2. Stir 30 min at 0–4°C → template–monomer pre-complex
3. Add oxidant FeCl₃ or (NH₄)₂S₂O₈ ({ratio * 2:.0f} mmol) dropwise at 0–4°C
   (or electropolymerize by CV, 0–0.9 V)
4. Polymerize 4–24 h at 0–4°C
   NOTE: polypyrrole/polyaniline self-crosslink — no molecular cross-linker used
5. Collect by filtration/centrifugation
6. Remove template by washing with water/MeOH (+ mild acid) until absent
7. Dry under vacuum at 40°C"""
        removal = "water/MeOH washing"
    else:  # free-radical (vinyl)
        chemistry = "free-radical"
        steps = f"""
Protocol (free-radical polymerization):
1. Dissolve {TEMPLATE_NAME} (1 mmol) in {solvent} (10 mL)
2. Add {best_monomer} ({ratio:.0f} mmol) and stir for 30 min at RT
   → Pre-polymerization complex formation
3. Add {xl} ({ratio * 5:.0f} mmol) and AIBN (0.1 mmol) as initiator
4. Purge with N₂ for 10 min
5. Heat to 60°C for 24 h (free-radical polymerization)
6. Remove template by Soxhlet extraction with MeOH/AcOH (9:1) for 48 h
7. Dry under vacuum at 40°C for 12 h"""
        removal = "Soxhlet MeOH/AcOH"

    # ── Design considerations (chemistry-aware corrections) ──
    _stoich = recipe.get("stoichiometry") or {}
    _spec = recipe.get("speciation") or {}
    _self = recipe.get("self_association") or {}
    _bleed = recipe.get("bleeding") or {}
    notes = ["\nDesign considerations (literature-grounded):"]
    _ens = recipe.get("ensemble") or {}
    if _ens:
        notes.append(
            f"  - Pose ensemble: best {_ens.get('best_pose_kcal')} vs Boltzmann-mean "
            f"{_ens.get('boltzmann_mean_kcal')} kcal/mol, n_eff={_ens.get('n_effective_poses')} "
            f"→ site heterogeneity {_ens.get('site_heterogeneity')}. Recognition is an "
            f"ensemble/free-energy phenomenon, not a single pose (Karlsson 2009).")
    _mmp = recipe.get("mmpbsa_dG") or {}
    if _mmp.get("dG_bind_kcal") is not None:
        notes.append(
            f"  - MM/GBSA binding free energy (Stage 5 MD ensemble): "
            f"ΔG_bind = {_mmp.get('dG_bind_kcal')} kcal/mol ({_mmp.get('method')}) — "
            f"real free energy over the equilibrium ensemble, not single-pose enthalpy.")
    if _stoich:
        notes.append(
            f"  - Stoichiometry: Kass≈{_stoich.get('Kass_per_M')} M⁻¹ → "
            f"{_stoich.get('regime')} → T:monomer = 1:{ratio:.0f}. "
            f"{_stoich.get('rationale')}")
    if _spec.get("ion_pair_recognition_likely"):
        notes.append(
            f"  - ⚠ Ionization: acid–base pair (template {_spec.get('template_ionizable')}, "
            f"monomer {_spec.get('monomer_ionizable')}). Recognition is likely via the "
            f"ION-PAIR — verify the charged microstate; neutral-form ranking may mislead.")
    if _self.get("dimerization_kcal") is not None:
        notes.append(
            f"  - Self-association: {best_monomer} dimerisation ΔE="
            f"{_self.get('dimerization_kcal')} kcal/mol "
            f"({'self-associates — lowers free monomer but suppresses non-specific sites' if _self.get('self_associates') else 'weak'}).")
    if _bleed.get("bleeding_risk"):
        notes.append(
            f"  - Template bleeding risk: {_bleed.get('bleeding_risk')}. "
            f"{_bleed.get('recommendation')}")
    _xlb = recipe.get("crosslinker_binding") or {}
    if _xlb:
        _msg = (f"crosslinker {_xlb.get('crosslinker')} binds the template "
                f"{_xlb.get('xl_template_kcal')} kcal/mol vs strongest monomer "
                f"{_xlb.get('strongest_monomer_kcal')}")
        if _xlb.get("xl_dominates"):
            notes.append(
                f"  - ⚠ Cross-linker is an active recognition species: {_msg} — "
                f"it DOMINATES template binding; cross-linker choice co-determines "
                f"selectivity (Shoravi/Olsson 2014). Consider screening EGDMA vs TRIM.")
        else:
            notes.append(f"  - Cross-linker binding accounted for: {_msg}.")

    protocol = header + steps + f"""

NIP (Non-Imprinted Polymer) control:
  Same protocol without template.

Quality control:
  - FT-IR: confirm template removal ({removal})
  - Rebinding test: incubate MIP/NIP with {TEMPLATE_NAME} solution
  - Calculate IF = Q_MIP / Q_NIP (target: IF > 2)
""" + "\n".join(notes) + "\n"

    recipe["protocol"] = protocol.strip()

    # Save
    recipe_path = out / "synthesis_recipe.json"
    with open(recipe_path, "w") as f:
        json.dump(recipe, f, indent=2, default=str)

    protocol_path = out / "synthesis_protocol.txt"
    with open(protocol_path, "w") as f:
        f.write(protocol)

    logger.info(f"\n{protocol}")
    logger.info(f"Recipe saved: {recipe_path}")
    logger.info(f"Protocol saved: {protocol_path}")

    return recipe


if __name__ == "__main__":
    run_stage7()

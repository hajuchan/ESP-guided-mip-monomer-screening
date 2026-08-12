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
# monomer polymerization type each protocol chemistry requires
_CHEM_TO_MONO_POLY = {"free-radical": "vinyl", "sol-gel": "silane", "oxidative": "oxidative"}


def _canon_chemistry(chem):
    """Normalize Stage-4's verbose `synthesis_method` (e.g.
    'oxidative/electro-polymerization') to the canonical protocol key used by
    _CHEM_TO_XL_POLY / _CHEM_TO_MONO_POLY / the radical gate. Without this,
    oxidative combos fall through to the default 'vinyl' path and get a
    contradictory EGDMA crosslinker (stage4 writes the long form; these dicts
    key on the short form)."""
    c = str(chem or "").lower()
    if c.startswith("oxidative") or "electro" in c:
        return "oxidative"
    if "sol-gel" in c or "silane" in c:
        return "sol-gel"
    if "radical" in c or "vinyl" in c:
        return "free-radical"
    return chem or "free-radical"


def _classify_polymerization(smi):
    """Si→silane, non-aromatic C=C→vinyl, else oxidative
    (mirrors stage4_mmsd._detect_polymerization)."""
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


def _xl_polymerization(name):
    return _classify_polymerization(CROSSLINKER_LIBRARY.get(name, ""))


def _mono_polymerization(name):
    return _classify_polymerization(MONOMER_LIBRARY.get(name, ""))


# Radical monomer/crosslinker families for reactivity-ratio matching. A
# crosslinker copolymerises efficiently only with monomers of a compatible
# reactive group (methacrylate↔dimethacrylate EGDMA/TRIM; styrenic↔DVB;
# acrylamide↔bisacrylamide BAM). N-vinyl monomers (NVP/VIM) have r≈0 and
# incorporate poorly with any of these. Order matters (methacrylate before
# acrylate). Ref: Odian, *Principles of Polymerization* (reactivity ratios).
_RADICAL_FAMILY_SMARTS = [
    ("methacrylate", "[CH2]=[CX3]([CH3])[CX3]=O"),
    ("styrenic",     "[CX3]=[CX3]c"),
    ("acrylamide",   "[CH2]=[CH][CX3](=O)[NX3]"),
    ("acrylate",     "[CH2]=[CH][CX3](=O)[OX1,OX2]"),
    ("n-vinyl",      "[CH2]=[CH][#7]"),   # vinyl on N (NVP; incl. aromatic n: VIM)
]


def _radical_family(smi):
    """Radical copolymerization family of a monomer/crosslinker SMILES (or None)."""
    if not smi:
        return None
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        for fam, sm in _RADICAL_FAMILY_SMARTS:
            patt = Chem.MolFromSmarts(sm)
            if patt is not None and m.HasSubstructMatch(patt):
                return fam
    except Exception:
        return None
    return None


def _dft_bindings(base, solvent):
    """{monomer: template binding ΔE (kcal/mol) in the porogen} from Stage 2 DFT.
    Falls back to a monomer's mean over solvents when the porogen entry is
    missing. Used to weight the MD ranking by real affinity so a non-binder that
    merely lingers near the template (high contact frequency, ~0 DFT binding)
    cannot outrank a genuine strong binder."""
    out = {}
    p = base / "stage2" / "stage2_dft.json"
    if not p.exists():
        return out
    try:
        s2 = json.load(open(p))
    except Exception:
        return out
    for m, solvs in (s2.items() if isinstance(s2, dict) else []):
        if not isinstance(solvs, dict):
            continue
        v = solvs.get(solvent)
        be = v.get("bsse_dE") if isinstance(v, dict) else None
        if be is None:
            vals = [x.get("bsse_dE") for x in solvs.values()
                    if isinstance(x, dict) and x.get("bsse_dE") is not None]
            be = sum(vals) / len(vals) if vals else None
        if be is not None:
            out[m] = float(be)
    return out


def _wash_solvent(template_smiles, template_name):
    """Choose a template-removal wash matched to the TEMPLATE chemistry, and
    never the template itself. A fixed MeOH/AcOH wash (a) fails to elute a
    nonpolar terpene template (MeOH barely dissolves it) and (b) is self-defeating
    when the template IS acetic acid (the wash reintroduces the analyte, so the
    FT-IR removal QC can never pass). Returns (wash, note)."""
    from . import chemistry_aware as _ca
    try:
        ion = _ca.detect_ionizable(template_smiles) or []
    except Exception:
        ion = []
    name = (template_name or "").lower()
    smi = template_smiles or ""
    is_acid = any(g in ion for g in ("carboxyl", "sulfonic_acid", "boronic_acid"))
    is_base = any(g in ion for g in ("primary_amine", "secondary_amine",
                                     "pyridine_N", "imidazole_N"))
    template_is_acetic = ("acetic" in name) or smi in ("CC(=O)O", "CC(O)=O", "OC(C)=O")
    polar = any(c in smi for c in ("O", "N", "S", "n", "o", "s"))
    if is_acid:
        acid = "formic acid" if template_is_acetic else "acetic acid"
        note = (f"acidic template — protic MeOH/{acid} competes the H-bonds"
                + ("; AcOH avoided (it is the template)" if template_is_acetic else ""))
        return f"MeOH/{acid} (9:1)", note
    if is_base:
        return "MeOH + 1% NH₃(aq)", "basic template — mildly basic MeOH wash breaks ionic H-bonds"
    if polar:
        return "MeOH (Soxhlet)", "polar template — hot MeOH disrupts the H-bond complex"
    return ("acetone or ethanol (Soxhlet), or the polymerization porogen",
            "nonpolar template — MeOH extracts poorly; use acetone/ethanol or the "
            "porogen to dissolve and elute the template")


def _combination_ratio(base, method):
    """Mixture-based synthesis ratio from the TOP combination MD: each monomer's
    per-type EBN (max simultaneous template binding) measured WHILE the monomers
    are co-present, mapped resname→name via the combo's resname_map. This is the
    Yuan 2024 EBN ratio, but read from the ACTUAL mixture (competition included)
    instead of each monomer in isolation. If EBN is degenerate across types (all
    equal — e.g. every type hit the copy cap), fall back to the continuous
    per-type contact frequency so the ratio still discriminates.

    method 'ebn'/other      → more sites ⇒ more copies (Yuan 2024, direct).
    method 'contact_inverse'→ weaker ⇒ more copies (compensation).
    Returns ({name: ratio}, method_str) or (None, None) when there is no genuine
    multi-monomer combination MD to read (→ caller falls back to single-monomer)."""
    p = base / "stage5" / "stage5_combination.json"
    if not p.exists():
        return None, None
    try:
        combos = json.load(open(p))
    except Exception:
        return None, None
    if not isinstance(combos, list) or not combos:
        return None, None
    per = (combos[0].get("md_analysis") or {}).get("per_monomer") or {}
    rmap = combos[0].get("resname_map") or {}
    ebn, cf = {}, {}
    for rn, d in per.items():
        name = rmap.get(rn, rn)
        d = d or {}
        ebn[name] = int(d.get("ebn_max", 0) or 0)
        cf[name] = float(d.get("contact_frequency", 0.0) or 0.0)
    names = [n for n in ebn if ebn[n] > 0 or cf[n] > 0]
    if len(names) < 2:
        return None, None               # not a real multi-monomer mixture
    # Prefer EBN; use contact only if EBN can't tell the types apart.
    use_ebn = len({ebn[n] for n in names}) > 1 and all(ebn[n] > 0 for n in names)
    sig = {n: (ebn[n] if use_ebn else cf[n]) for n in names}
    basis = "EBN" if use_ebn else "contact"
    if method == "contact_inverse":
        mx = max(sig.values())
        return ({n: round(mx / v, 1) for n, v in sig.items()},
                f"inverse mixture {basis} (combination MD)")
    mn = min(sig.values())
    return ({n: max(1, round(v / mn)) for n, v in sig.items()},
            f"direct mixture {basis} (combination MD, Yuan 2024 EBN)")


def _select_crosslinker(base, chemistry, mmsd_xl=None, best_monomer=None):
    """Pick the most INERT crosslinker compatible with the winning chemistry AND
    reactivity-matched to the best monomer.

    A crosslinker is used in large excess (~80 %, T:XL ≈ 1:20); if it binds the
    template it seeds NON-SPECIFIC sites throughout the whole matrix (not just
    the imprinted cavity), raising NIP binding and LOWERING the imprinting factor
    (Shoravi/Olsson 2014). So the criterion is the WEAKEST template binding among
    chemistry-compatible crosslinkers — deliberately NOT MMSD's pick. Among those,
    prefer the crosslinker in the SAME radical family as the monomer (reactivity
    ratio: methacrylate↔EGDMA/TRIM, styrenic↔DVB, acrylamide↔BAM) so it actually
    copolymerises. Oxidative polymers self-crosslink → returns (None, reason)."""
    need = _CHEM_TO_XL_POLY.get(chemistry, "vinyl")
    if need is None:                       # oxidative → no molecular crosslinker
        return None, "oxidative self-crosslinking (no molecular crosslinker)"

    fam = _radical_family(MONOMER_LIBRARY.get(best_monomer, "")) if best_monomer else None
    warn = ""
    if fam == "n-vinyl":
        warn = (f"; NOTE: {best_monomer} is N-vinyl (r≈0) — it copolymerises poorly "
                f"with (meth)acrylate/styrenic crosslinkers; expect low incorporation, "
                f"consider a reactivity-matched divinyl or a redox/low-T initiator")

    # 1) compatible crosslinkers from the DFT screening, ranked by inertness
    feat = base / "features" / "crosslinker.json"
    if feat.exists():
        try:
            data = json.load(open(feat))
            cands = {}                      # name -> avg template ΔE (higher = more inert)
            for name, solvs in data.items():
                if _xl_polymerization(name) != need:
                    continue
                bes = [v.get("bsse_dE", v.get("raw_dE")) for v in solvs.values()
                       if isinstance(v, dict) and
                       (v.get("bsse_dE") is not None or v.get("raw_dE") is not None)]
                if bes:
                    cands[name] = sum(bes) / len(bes)
            if cands:
                # prefer the same reactive family; within the pool, most inert wins
                same = {n: v for n, v in cands.items()
                        if fam and _radical_family(CROSSLINKER_LIBRARY.get(n, "")) == fam}
                pool = same or cands
                best = max(pool, key=pool.get)
                match = (f", reactivity-matched to the {fam} monomer" if same else
                         (f" (no {fam}-family crosslinker in library; reactivity ratio "
                          f"may be suboptimal)" if fam else ""))
                return best, (f"most inert compatible{match} — weakest template binding "
                              f"(avg ΔE={pool[best]:+.3f} kcal/mol){warn}")
        except Exception:
            pass

    # 2) Stage-3 recommendation (also weakest-binding based), if compatible
    cl_path = base / "stage3" / "stage3_crosslinker.json"
    if cl_path.exists():
        try:
            rec = json.load(open(cl_path)).get("recommended")
            if rec and _xl_polymerization(rec) == need:
                return rec, f"Stage 3 screening (weakest-binding, compatible){warn}"
        except Exception:
            pass

    # 3) per-chemistry default (standard inert crosslinker)
    return (_CHEM_DEFAULT_XL.get(chemistry, "EGDMA"),
            f"per-chemistry default (standard inert crosslinker){warn}")


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

    # ── Combination-centric ranking ──
    # The recipe centers on the best COMBINATION (Stage-5 combination MD, sorted by
    # mixture contact; Stage-6 VIP validates its cavity). best_monomer = the
    # combination's DOMINANT monomer (highest mixture contact); the co-monomer feed
    # comes from the mixture ratio. Falls back to the Stage-3 per-monomer ranking.
    vip_path = base / "stage6" / "stage6_vip.json"
    s4_path = base / "stage5" / "stage5_md.json"   # legacy single-monomer (usually absent)

    combos = []
    s5c_path = base / "stage5" / "stage5_combination.json"
    if s5c_path.exists():
        try:
            combos = [c for c in json.load(open(s5c_path))
                      if isinstance(c, dict) and c.get("md_analysis")]
        except Exception:
            combos = []
    best_combo = combos[0] if combos else None     # stage5 sorts by mixture contact

    monomer_ranking, src = [], None
    if best_combo:
        per = (best_combo.get("md_analysis") or {}).get("per_monomer") or {}
        rmap = best_combo.get("resname_map") or {}
        ranked = sorted(((rmap.get(rn, rn), (d or {}).get("contact_frequency", 0.0))
                         for rn, d in per.items()), key=lambda x: -x[1])
        monomer_ranking = [(m, s) for m, s in ranked if m in MONOMER_LIBRARY]
        src = "Stage 5 combination MD (mixture contact)"
        recipe["recommended_combination"] = best_combo.get("selected", [])
        _lbl = ("+".join(best_combo.get("selected", []))
                or f"pc{best_combo.get('rank')}")
        if vip_path.exists():                       # VIP validation of the cavity
            try:
                _v = json.load(open(vip_path)).get(_lbl, {})
                if isinstance(_v, dict) and _v.get("vip_score") is not None:
                    recipe["combination_vip"] = {
                        "label": _lbl, "vip_score": _v.get("vip_score"),
                        "removal_rate": (_v.get("removal_rate")
                                         or _v.get("removal_success_rate"))}
                    src = "Stage 5 combination MD + Stage 6 VIP"
            except Exception:
                pass
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

    # ── Polymerization chemistry (winning MMSD combo) ──
    # Determined BEFORE monomer selection so the whole recipe (monomer +
    # protocol + crosslinker) is ONE self-consistent chemistry. Otherwise the
    # MD contact-frequency ranking can surface e.g. pyrrole (oxidative) as the
    # top binder while the protocol stays free-radical → a chemically impossible
    # recipe (pyrrole + AIBN).
    chemistry = "free-radical"
    mmsd_xl = None
    mmsd_path = base / "stage4" / "mmsd_results.json"
    if mmsd_path.exists():
        try:
            _t = (json.load(open(mmsd_path)).get("top_pcs") or [])
            if _t:
                chemistry = _t[0].get("synthesis_method") or "free-radical"
                mmsd_xl = _t[0].get("crosslinker")
        except Exception:
            pass
    # Normalize to the canonical key so _CHEM_TO_XL_POLY / _CHEM_TO_MONO_POLY /
    # the radical gate all match (stage4 stamps the verbose form).
    chemistry = _canon_chemistry(chemistry)

    # Keep only monomers whose polymerization matches the winning chemistry, so
    # best_monomer agrees with the protocol and crosslinker.
    _need_poly = _CHEM_TO_MONO_POLY.get(chemistry)
    if _need_poly:
        _consistent = [(m, s) for (m, s) in monomer_ranking
                       if _mono_polymerization(m) == _need_poly]
        if _consistent:
            n_drop = len(monomer_ranking) - len(_consistent)
            if n_drop:
                recipe["ranking_note"] = (
                    f"filtered to {_need_poly} monomers for consistency with the "
                    f"{chemistry} protocol ({n_drop} incompatible dropped)")
                logger.info(f"  Ranking filtered to {_need_poly}: "
                            f"{n_drop} chemistry-incompatible monomers dropped")
            monomer_ranking = _consistent

    # Top 3 monomers (chemistry-consistent)
    top3 = monomer_ranking[:3]
    recipe["top3_monomers"] = [
        {"rank": i+1, "name": m, "score": round(s, 3),
         "smiles": MONOMER_LIBRARY.get(m, "")}
        for i, (m, s) in enumerate(top3)
    ]

    recipe["polymerization"] = chemistry
    best_monomer = top3[0][0] if top3 else "MAA"

    # ── Cross-linker (chemistry-compatible + reactivity-matched to best monomer) ──
    xl_name, xl_reason = _select_crosslinker(base, chemistry, mmsd_xl,
                                             best_monomer=best_monomer)
    recipe["crosslinker"] = xl_name
    recipe["crosslinker_smiles"] = (
        CROSSLINKER_LIBRARY.get(xl_name, "") if xl_name else "")
    recipe["crosslinker_source"] = xl_reason

    # ── Synthesis ratio ──
    # PREFER the mixture-based ratio: each monomer's template engagement measured
    # in Stage 4's actual combination (combination MD, monomers co-present), so
    # competition between co-monomers is reflected. Fall back to single-monomer
    # EBN (Yuan 2024 — each monomer in isolation), then a fixed 1:4 default.
    from .config import MD_RATIO_METHOD
    combo_ratios, combo_method = _combination_ratio(base, MD_RATIO_METHOD)
    if combo_ratios:
        recipe["synthesis_ratios"] = combo_ratios
        recipe["ratio_method"] = combo_method
        recipe["ratio_basis"] = "combination (mixture)"
    elif s4_path.exists():
        s4 = json.load(open(s4_path))
        if isinstance(s4, list):
            if MD_RATIO_METHOD == "ebn":
                ebn = {r["monomer"]: r.get("ebn_max", 0)
                       for r in s4 if r.get("ebn_max", 0) > 0}
                if ebn:
                    min_e = min(ebn.values())
                    recipe["synthesis_ratios"] = {m: max(1, round(v / min_e))
                                                  for m, v in ebn.items()}
                    recipe["ratio_method"] = "EBN direct (Yuan 2024, single-monomer)"
                    recipe["ratio_basis"] = "single-monomer"
            if "synthesis_ratios" not in recipe:      # contact-inverse (option/fallback)
                contacts = {r["monomer"]: r.get("contact_frequency", 0)
                            for r in s4 if r.get("contact_frequency", 0) > 0}
                if contacts:
                    max_c = max(contacts.values())
                    recipe["synthesis_ratios"] = {m: round(max_c / c, 1)
                                                  for m, c in contacts.items()}
                    recipe["ratio_method"] = "inverse contact frequency (single-monomer)"
                    recipe["ratio_basis"] = "single-monomer"

    # Default ratio if not computed
    if "synthesis_ratios" not in recipe:
        recipe["synthesis_ratios"] = {m: 4.0 for m, _ in top3}
        recipe["ratio_method"] = "default (1:4)"
        recipe["ratio_basis"] = "default"

    # Co-monomer feed string when the mixture ratio spans ≥2 monomers, so the
    # multi-monomer ratio is actually visible (protocol prose is single-monomer).
    if len(recipe["synthesis_ratios"]) >= 2:
        _co = sorted(recipe["synthesis_ratios"].items(), key=lambda x: -x[1])
        recipe["comonomer_ratio"] = ("1 (template) : "
                                     + " : ".join(f"{r} {m}" for m, r in _co))

    # ── Protocol (chemistry-specific) ── (best_monomer set above)
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

    # Boronic-acid recognition is pH-gated: the monomer binds a cis-diol only as
    # the tetrahedral BORONATE anion (above the boronic pKa), so a boronic-acid
    # monomer MIP must be prepared/rebound in a basic buffer — otherwise the
    # covalent diol recognition it was chosen for never forms.
    try:
        _mono_ion = _ca.detect_ionizable(best_smi) or []
        _tmpl_ion = _ca.detect_ionizable(TEMPLATE_SMILES) or []
    except Exception:
        _mono_ion, _tmpl_ion = [], []
    if "boronic_acid" in _mono_ion:
        from rdkit import Chem as _Chem
        _tm = _Chem.MolFromSmiles(TEMPLATE_SMILES) if TEMPLATE_SMILES else None
        _has_diol = bool(_tm is not None and
                         _tm.HasSubstructMatch(_Chem.MolFromSmarts("[CX4;!$(C=O)][OX2H].[CX4;!$(C=O)][OX2H]")))
        recipe["boronic_recognition"] = {
            "monomer_is_boronic": True,
            "template_has_diol": _has_diol,
            "condition": "prepare and rebind in a basic buffer (pH ≈ pKa_B + 1, "
                         "typically pH 8–10) so the boronate anion forms",
            "note": ("covalent cis-diol recognition — matched to this target"
                     if _has_diol else
                     "boronic monomer chosen but the template shows no cis-diol; "
                     "confirm the intended (diol/anion) recognition mode"),
        }
        recipe.setdefault("warnings", []).append(
            "Boronic-acid monomer: use a basic buffer (pH ~8–10) for imprinting "
            "and rebinding, else the boronate–diol complex does not form.")
        logger.info("  Boronic-acid monomer → basic-pH condition flagged")

    # GAP 5b: template radical-chemistry compatibility. A radical-scavenging
    # template (γ-terpinene's bis-allylic 1,4-diene, phenols, thiols, quinones,
    # anilines) will NOT cure a free-radical (vinyl) MIP — it consumes the
    # propagating radical. Gate the recommended chemistry so the recipe is
    # actually synthesizable (matches the wet-lab γ-terpinene vs methyl-benzoate
    # result: the former fails a MAA/EGDMA radical MIP, the latter works).
    radical_compat = _ca.template_radical_compatibility(TEMPLATE_SMILES)
    recipe["template_radical_compatibility"] = radical_compat
    recipe.setdefault("warnings", [])
    if chemistry == "free-radical" and radical_compat.get("severity") == "high":
        _mtf = ", ".join(m["motif"] for m in radical_compat["motifs"])
        warn = (f"⚠ SYNTHESIS FEASIBILITY: template '{TEMPLATE_NAME}' carries a "
                f"radical-interfering motif ({_mtf}); the recommended FREE-RADICAL "
                f"protocol will be RETARDED or fail to cure. "
                f"{radical_compat['recommendation']}")
        recipe["warnings"].append(warn)
        recipe["synthesis_feasibility"] = "at-risk (radical inhibition)"
        logger.warning(f"  {warn}")
    elif chemistry == "free-radical" and radical_compat.get("severity") == "medium":
        recipe["warnings"].append(
            f"Template '{TEMPLATE_NAME}' may retard free-radical cure: "
            f"{radical_compat['recommendation']}")
        recipe["synthesis_feasibility"] = "caution"
        logger.warning(f"  Template may retard free-radical cure ({TEMPLATE_NAME})")
    else:
        recipe.setdefault("synthesis_feasibility", "ok")

    # GAP 3: functional-monomer self-association (xTB dimerisation)
    try:
        recipe["self_association"] = _ca.dimerization_energy(best_smi)
    except Exception as e:
        recipe["self_association"] = {"error": str(e)}

    # GAP 8: template bleeding / removability risk — from the winning
    # COMBINATION's VIP (stage6 is keyed by combination label, not monomer).
    removal_rate = None
    vip_p = base / "stage6" / "stage6_vip.json"
    if vip_p.exists() and best_combo is not None:
        try:
            _lbl2 = ("+".join(best_combo.get("selected", []))
                     or f"pc{best_combo.get('rank')}")
            v = json.load(open(vip_p)).get(_lbl2, {})
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

    # GAP 5: verify the SELECTED crosslinker is actually inert. We now pick the
    # weakest-binding compatible crosslinker, so this should CONFIRM it does not
    # compete with the monomer (Shoravi/Olsson 2014). Compares the recipe's
    # crosslinker DFT binding (in the porogen) with the best monomer's.
    if xl:
        try:
            _feat = json.load(open(base / "features" / "crosslinker.json"))
            _xls = _feat.get(xl, {}) or {}
            _xl_be = (_xls.get(solvent, {}) or {}).get("bsse_dE")
            if _xl_be is None:
                _vals = [v.get("bsse_dE") for v in _xls.values()
                         if isinstance(v, dict) and v.get("bsse_dE") is not None]
                _xl_be = sum(_vals) / len(_vals) if _vals else None
            if _xl_be is not None and best_be is not None:
                recipe["crosslinker_binding"] = {
                    "crosslinker": xl,
                    "xl_template_kcal": round(_xl_be, 3),
                    "best_monomer_kcal": round(best_be, 3),
                    "xl_dominates": _xl_be <= best_be,
                }
        except Exception:
            pass

    # Polymerization chemistry + crosslinker already resolved above.
    xl_disp = (f"{xl} ({CROSSLINKER_LIBRARY.get(xl, '')})" if xl
               else "none (self-crosslinking)")
    xl_ratio_disp = f"{ratio * 5:.0f}" if xl else "0 (self-crosslinking)"

    # Molar-ratio line: show the full co-monomer feed (mixture-derived) when the
    # recipe spans ≥2 monomers, else the single Template:Monomer:XL line.
    _co = recipe.get("comonomer_ratio")
    if _co:
        ratio_line = (f"Molar ratio (mixture-derived, {recipe.get('ratio_method','')}):\n"
                      f"  {_co} : {xl_ratio_disp} {xl or ''}")
    else:
        ratio_line = ("Molar ratio (Template : Monomer : Cross-linker):\n"
                      f"  1 : {ratio:.0f} : {xl_ratio_disp}")

    header = f"""=== MIP Synthesis Protocol ({chemistry}) ===

Template: {TEMPLATE_NAME} ({TEMPLATE_SMILES})
Monomer:  {best_monomer} ({MONOMER_LIBRARY.get(best_monomer, '')})
Cross-linker: {xl_disp}
Solvent:  {solvent}

{ratio_line}
"""

    # Template-removal wash matched to the TEMPLATE chemistry (never the template)
    wash, wash_note = _wash_solvent(TEMPLATE_SMILES, TEMPLATE_NAME)
    recipe["template_removal"] = {"wash": wash, "rationale": wash_note}

    if chemistry.startswith("sol-gel"):
        steps = f"""
Protocol (sol-gel / silane condensation):
1. Dissolve {TEMPLATE_NAME} (1 mmol) and {best_monomer} ({ratio:.0f} mmol) in ethanol (10 mL)
2. Stir 30 min at RT → template–silane pre-complex
3. Add {xl} ({ratio * 5:.0f} mmol, e.g. TEOS), then H₂O ({ratio * 10:.0f} mmol) + HCl (cat., pH ≈ 3)
4. Stir 2 h at RT (hydrolysis) → age/gel 24–72 h at 40°C (condensation)
5. Dry to xerogel under vacuum at 60°C
6. Remove template by washing with {wash}, repeat until absent ({wash_note})
7. Dry under vacuum at 60°C"""
        removal = f"{wash} washing"
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
6. Remove template by washing with {wash} until absent ({wash_note})
7. Dry under vacuum at 40°C"""
        removal = f"{wash} washing"
    else:  # free-radical (vinyl)
        chemistry = "free-radical"
        steps = f"""
Protocol (free-radical polymerization):
1. Dissolve {TEMPLATE_NAME} (1 mmol) in {solvent} (10 mL)
2. Add {best_monomer} ({ratio:.0f} mmol) and stir for 30 min at RT
   → Pre-polymerization complex formation
3. Add {xl} ({ratio * 5:.0f} mmol) and AIBN (0.1 mmol) as initiator
   (thermal; for a volatile/heat-sensitive template use a photo-initiator,
    e.g. 0.05 mmol DMPA/Irgacure, and cure under UV at 0–10°C instead)
4. Purge with N₂ for 10 min in a sealed vessel (exclude O₂ — it inhibits cure)
5. Heat to 60°C for 24 h (free-radical polymerization)
6. Remove template by Soxhlet extraction with {wash} for 48 h ({wash_note})
7. Dry under vacuum at 40°C for 12 h"""
        removal = f"Soxhlet {wash}"

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
                f"{_xlb.get('xl_template_kcal')} kcal/mol vs best monomer "
                f"{_xlb.get('best_monomer_kcal')}")
        if _xlb.get("xl_dominates"):
            notes.append(
                f"  - ⚠ Cross-linker is an active recognition species: {_msg} — "
                f"it competes with/dominates the monomer; a strongly-binding "
                f"crosslinker seeds non-specific sites and lowers IF "
                f"(Shoravi/Olsson 2014).")
        else:
            notes.append(
                f"  - Cross-linker is inert (as intended): {_msg} — it binds the "
                f"template far more weakly than the monomer, so it contributes "
                f"structure without seeding non-specific sites (Shoravi/Olsson 2014).")

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

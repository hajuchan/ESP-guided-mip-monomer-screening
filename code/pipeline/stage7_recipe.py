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

    monomer_ranking = []
    if vip_path.exists():
        with open(vip_path) as f:
            vip = json.load(f)
        # Rank by VIP score
        ranked = [(m, d.get("vip_score", 0)) for m, d in vip.items()
                  if isinstance(d, dict) and d.get("vip_score", 0) > 0]
        ranked.sort(key=lambda x: -x[1])
        monomer_ranking = ranked
        recipe["ranking_source"] = "Stage 5 VIP"
    elif s4_path.exists():
        with open(s4_path) as f:
            s4 = json.load(f)
        if isinstance(s4, list):
            ranked = [(r["monomer"], r.get("contact_frequency", 0))
                      for r in s4 if r.get("contact_frequency", 0) > 0]
            ranked.sort(key=lambda x: -x[1])
            monomer_ranking = ranked
        recipe["ranking_source"] = "Stage 4 MD (contact frequency)"
    else:
        # Fallback: Stage 3
        s3_path = base / "stage3" / "stage3_top.json"
        if s3_path.exists():
            with open(s3_path) as f:
                names = json.load(f)
            monomer_ranking = [(n, 0) for n in names]
        recipe["ranking_source"] = "Stage 3 selectivity"

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

    # ── Cross-linker ──
    cl_path = base / "stage3" / "stage3_crosslinker.json"
    if cl_path.exists():
        with open(cl_path) as f:
            cl = json.load(f)
        recipe["crosslinker"] = cl.get("recommended", "EGDMA")
        recipe["crosslinker_smiles"] = CROSSLINKER_LIBRARY.get(
            recipe["crosslinker"], "")
    else:
        recipe["crosslinker"] = "EGDMA"  # default
        recipe["crosslinker_smiles"] = CROSSLINKER_LIBRARY.get("EGDMA", "")

    # ── Synthesis ratio ──
    if s4_path.exists():
        with open(s4_path) as f:
            s4 = json.load(f)
        if isinstance(s4, list):
            contacts = {r["monomer"]: r.get("contact_frequency", 0)
                        for r in s4 if r.get("contact_frequency", 0) > 0}
            if contacts:
                max_c = max(contacts.values())
                ratios = {m: round(max_c / c, 1) for m, c in contacts.items()}
                recipe["synthesis_ratios"] = ratios

    # Default ratio if not computed
    if "synthesis_ratios" not in recipe:
        recipe["synthesis_ratios"] = {m: 4.0 for m, _ in top3}

    # ── Protocol (chemistry-specific) ──
    best_monomer = top3[0][0] if top3 else "MAA"
    ratio = recipe["synthesis_ratios"].get(best_monomer, 4.0)
    xl = recipe["crosslinker"]

    # Polymerization chemistry from the winning MMSD combo (Stage 4).
    chemistry = "free-radical"
    mmsd_path = base / "stage4" / "mmsd_results.json"
    if mmsd_path.exists():
        try:
            _top = (json.load(open(mmsd_path)).get("top_pcs") or [])
            if _top:
                chemistry = _top[0].get("synthesis_method") or "free-radical"
        except Exception:
            pass
    recipe["polymerization"] = chemistry

    header = f"""=== MIP Synthesis Protocol ({chemistry}) ===

Template: {TEMPLATE_NAME} ({TEMPLATE_SMILES})
Monomer:  {best_monomer} ({MONOMER_LIBRARY.get(best_monomer, '')})
Cross-linker: {xl} ({CROSSLINKER_LIBRARY.get(xl, '')})
Solvent:  {solvent}

Molar ratio (Template : Monomer : Cross-linker):
  1 : {ratio:.0f} : {ratio * 5:.0f}
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

    protocol = header + steps + f"""

NIP (Non-Imprinted Polymer) control:
  Same protocol without template.

Quality control:
  - FT-IR: confirm template removal ({removal})
  - Rebinding test: incubate MIP/NIP with {TEMPLATE_NAME} solution
  - Calculate IF = Q_MIP / Q_NIP (target: IF > 2)
"""

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

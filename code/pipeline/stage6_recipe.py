"""
Stage 6: Synthesis Recipe Generation
=====================================
Generates synthesis protocol based on pipeline results:
- Monomer ranking from Stage 5 VIP (or Stage 4 if VIP unavailable)
- Cross-linker recommendation from Stage 3
- Synthesis ratio from Stage 4 contact frequency
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


def run_stage6(output_dir: str = None) -> dict:
    """Generate synthesis recipe from pipeline results."""
    if output_dir is None:
        output_dir = OUTPUT_DIRS.get("reports", f"{OUTPUT_DIR}/reports")
    base = Path(output_dir).parent
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    recipe = {
        "template": TEMPLATE_NAME,
        "template_smiles": TEMPLATE_SMILES,
        "solvent": SYNTHESIS_SOLVENT,
    }

    # ── Load VIP results (Stage 5) or Stage 4 ──
    vip_path = base / "stage5" / "stage5_vip.json"
    s4_path = base / "stage4" / "stage4_md.json"

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

    # ── Protocol ──
    best_monomer = top3[0][0] if top3 else "MAA"
    ratio = recipe["synthesis_ratios"].get(best_monomer, 4.0)
    xl = recipe["crosslinker"]

    protocol = f"""
=== MIP Synthesis Protocol ===

Template: {TEMPLATE_NAME} ({TEMPLATE_SMILES})
Monomer:  {best_monomer} ({MONOMER_LIBRARY.get(best_monomer, '')})
Cross-linker: {xl} ({CROSSLINKER_LIBRARY.get(xl, '')})
Solvent:  {SYNTHESIS_SOLVENT}

Molar ratio (Template : Monomer : Cross-linker):
  1 : {ratio:.0f} : {ratio * 5:.0f}

Protocol:
1. Dissolve {TEMPLATE_NAME} (1 mmol) in {SYNTHESIS_SOLVENT} (10 mL)
2. Add {best_monomer} ({ratio:.0f} mmol) and stir for 30 min at RT
   → Pre-polymerization complex formation
3. Add {xl} ({ratio * 5:.0f} mmol) and AIBN (0.1 mmol) as initiator
4. Purge with N₂ for 10 min
5. Heat to 60°C for 24 h (free-radical polymerization)
6. Remove template by Soxhlet extraction with MeOH/AcOH (9:1) for 48 h
7. Dry under vacuum at 40°C for 12 h

NIP (Non-Imprinted Polymer) control:
  Same protocol without template (step 1: solvent only)

Quality control:
  - FT-IR: confirm template removal (C=O stretch at 1720 cm⁻¹ absent)
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
    run_stage6()

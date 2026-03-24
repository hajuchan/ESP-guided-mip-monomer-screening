"""
Feature 7 — Auto Interferent Suggestion
========================================
Suggests potential interferent molecules for MIP selectivity testing
by combining:
  1. Structural similarity search (RDKit, local)
  2. PubChem similar-compound lookup (optional, network)

Usage
-----
    python suggest_interferents.py
    # or from another module:
    from suggest_interferents import suggest_interferents
    suggestions = suggest_interferents()
"""

import json
import os
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from .config import TEMPLATE_SMILES, TEMPLATE_NAME, OUTPUT_DIR, INTERFERENT_LIBRARY

# ── Built-in molecule database ───────────────────────────────────────
BUILTIN_MOLECULES = {
    "Phenylalanine": "N[C@@H](Cc1ccccc1)C(=O)O",
    "Tyrosine": "NC(Cc1ccc(O)cc1)C(=O)O",
    "Tryptophan": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "Histidine": "NC(Cc1cnc[nH]1)C(=O)O",
    "Leucine": "CC(C)CC(N)C(=O)O",
    "Isoleucine": "CC[C@H](C)[C@@H](N)C(=O)O",
    "Valine": "CC(C)[C@@H](N)C(=O)O",
    "Alanine": "C[C@@H](N)C(=O)O",
    "Glycine": "NCC(=O)O",
    "Serine": "NC(CO)C(=O)O",
    "Threonine": "C[C@@H](O)[C@H](N)C(=O)O",
    "Methionine": "CSCC[C@@H](N)C(=O)O",
    "Dopamine": "NCCc1ccc(O)c(O)c1",
    "Epinephrine": "CNC[C@@H](O)c1ccc(O)c(O)c1",
    "Serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "Norepinephrine": "NCC(O)c1ccc(O)c(O)c1",
    "Heptachlor": "ClC1=C(Cl)C2(Cl)C3C=CC1(Cl)C3(Cl)C2(Cl)Cl",
    "DDT": "ClC(Cl)(Cl)c1ccc(cc1)C(c1ccc(Cl)cc1)c1ccc(Cl)cc1",
    "Atrazine": "CCNc1nc(Cl)nc(NC(C)C)n1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Naproxen": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}


# ── Helpers ──────────────────────────────────────────────────────────

def _morgan_fp(mol):
    """Return a Morgan fingerprint (radius=2, 2048 bits)."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _canonical(smiles: str) -> str:
    """Return canonical SMILES string for reliable comparison."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


# ── Method 1: Structure similarity (local) ───────────────────────────

def _local_similarity_search(template_smiles: str) -> list[dict]:
    """
    Compare the template against every molecule in BUILTIN_MOLECULES
    using Tanimoto similarity on Morgan fingerprints.

    Returns entries with 0.3 <= similarity <= 0.8, sorted descending.
    """
    tmpl_mol = Chem.MolFromSmiles(template_smiles)
    if tmpl_mol is None:
        print(f"[suggest] WARNING: Could not parse template SMILES: {template_smiles}")
        return []

    tmpl_fp = _morgan_fp(tmpl_mol)
    tmpl_canon = _canonical(template_smiles)

    results = []
    for name, smi in BUILTIN_MOLECULES.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # Exclude exact template match
        if _canonical(smi) == tmpl_canon:
            continue
        fp = _morgan_fp(mol)
        sim = DataStructs.TanimotoSimilarity(tmpl_fp, fp)
        if 0.3 <= sim <= 0.8:
            results.append({
                "name": name,
                "smiles": smi,
                "tanimoto": round(sim, 4),
                "source": "builtin",
            })

    results.sort(key=lambda d: d["tanimoto"], reverse=True)
    return results


# ── Method 2: PubChem API (optional) ─────────────────────────────────

def _pubchem_search(template_smiles: str) -> list[dict]:
    """
    Query PubChem for structurally similar compounds.

    Returns a list of dicts with SMILES, CID, IUPACName, MolecularWeight.
    Silently returns [] on any failure (missing library, network, timeout).
    """
    try:
        import pubchempy as pcp
    except ImportError:
        return []

    try:
        compounds = pcp.get_compounds(
            template_smiles,
            namespace="smiles",
            searchtype="similarity",
            Threshold=40,
            MaxRecords=20,
            timeout=10,
        )
    except Exception:
        return []

    tmpl_canon = _canonical(template_smiles)
    tmpl_mol = Chem.MolFromSmiles(template_smiles)
    tmpl_fp = _morgan_fp(tmpl_mol) if tmpl_mol else None

    results = []
    for c in compounds:
        smi = c.canonical_smiles
        if smi is None:
            continue
        if _canonical(smi) == tmpl_canon:
            continue

        # Compute local Tanimoto for consistent scoring
        tanimoto = None
        if tmpl_fp is not None:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                tanimoto = round(
                    DataStructs.TanimotoSimilarity(tmpl_fp, _morgan_fp(mol)), 4
                )

        results.append({
            "name": c.iupac_name or f"CID-{c.cid}",
            "smiles": smi,
            "cid": c.cid,
            "iupac_name": c.iupac_name,
            "molecular_weight": c.molecular_weight,
            "tanimoto": tanimoto,
            "source": "pubchem",
        })

    results.sort(key=lambda d: (d["tanimoto"] or 0), reverse=True)
    return results


# ── Main entry point ─────────────────────────────────────────────────

def suggest_interferents(
    template_smiles: str | None = None,
    output_dir: str = OUTPUT_DIR,
) -> list[dict]:
    """
    Suggest potential interferent molecules for MIP selectivity testing.

    Parameters
    ----------
    template_smiles : str or None
        SMILES of the template molecule. Defaults to config.TEMPLATE_SMILES.
    output_dir : str
        Directory for saving the JSON output.

    Returns
    -------
    list[dict]
        Suggested interferents with name, SMILES, Tanimoto score, and source.
    """
    if template_smiles is None:
        template_smiles = TEMPLATE_SMILES

    print("=" * 60)
    print("  Feature 7: Auto Interferent Suggestion")
    print("=" * 60)
    print(f"  Template : {TEMPLATE_NAME}")
    print(f"  SMILES   : {template_smiles}")
    print("-" * 60)

    # --- Method 1: local similarity ---
    print("\n[1/2] Searching built-in molecule database ...")
    local_hits = _local_similarity_search(template_smiles)
    print(f"      Found {len(local_hits)} candidates (0.3 <= Tanimoto <= 0.8)")

    # --- Method 2: PubChem ---
    print("[2/2] Querying PubChem for similar compounds ...")
    pubchem_hits = _pubchem_search(template_smiles)
    if pubchem_hits:
        print(f"      Found {len(pubchem_hits)} candidates from PubChem")
    else:
        print("      PubChem unavailable or returned no results — skipped")

    # --- Merge (local first, then PubChem that are not duplicates) ---
    seen_canon = {_canonical(h["smiles"]) for h in local_hits}
    merged = list(local_hits)
    for h in pubchem_hits:
        canon = _canonical(h["smiles"])
        if canon not in seen_canon:
            seen_canon.add(canon)
            merged.append(h)

    # --- Console output ---
    print("\n" + "=" * 60)
    print(f"  Suggested interferents ({len(merged)} total)")
    print("=" * 60)
    for i, entry in enumerate(merged, 1):
        sim_str = f"{entry['tanimoto']:.4f}" if entry["tanimoto"] is not None else "N/A"
        src = entry["source"]
        print(f"  {i:>3}. [{src:<7}] {entry['name']:<25} Tanimoto={sim_str}")
        print(f"       SMILES: {entry['smiles']}")

    # --- config.py snippet ---
    print("\n" + "-" * 60)
    print("  Copy-paste snippet for config.py INTERFERENT_LIBRARY:")
    print("-" * 60)
    print("INTERFERENT_LIBRARY = {")
    # Include existing interferents
    for name, smi in INTERFERENT_LIBRARY.items():
        print(f'    "{name}": "{smi}",')
    # Append new suggestions not already in the library
    existing_canon = {_canonical(s) for s in INTERFERENT_LIBRARY.values()}
    for entry in merged:
        if _canonical(entry["smiles"]) not in existing_canon:
            print(f'    "{entry["name"]}": "{entry["smiles"]}",')
    print("}")

    # --- Save JSON ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "suggested_interferents.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)
    print(f"\n  Saved -> {out_path}")
    print("=" * 60)

    return merged


if __name__ == "__main__":
    suggest_interferents()

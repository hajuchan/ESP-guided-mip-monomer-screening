"""
Validation Configuration
========================
Reference values from literature for pipeline validation.
Imports physical constants and settings from config.py — no duplication.
"""

from pipeline.config import (
    HARTREE_TO_KCAL, KB_KCAL, TEMPERATURE, OUTPUT_DIR,
    MONOMER_LIBRARY, SOLVENTS, INTERFERENT_LIBRARY,
    SOLVENT_STRATEGY, SYNTHESIS_SOLVENT,
    USE_ESP_MAP,
    MD_RATIO_SCREENING, MD_RATIOS_TO_TEST,
    CROSSLINKER_LIBRARY, CROSSLINKER_THRESHOLD,
)

# ── Singh 2012 기준값 ────────────────────────────────────────────────
# DOI: 10.2174/157341112803216807
# Table 1: BSSE-corrected binding energies (kcal/mol)
SINGH_2012_BINDING_ENERGIES = {
    ("heptachlor", "MAA"):     -8.11,
    ("heptachlor", "4VP"):     -7.76,
    ("heptachlor", "styrene"): -4.13,
    ("DDT",        "MAA"):     -8.99,
    ("DDT",        "4VP"):     -7.86,
    ("DDT",        "styrene"): -8.28,
}

# Table 1: Experimental Imprinting Factors
SINGH_2012_IF = {
    "heptachlor": {"MAA": 1.92, "4VP": 1.45, "styrene": 0.99},
    "DDT":        {"MAA": 1.43, "4VP": 1.65, "styrene": 1.21},
}

# ── Mukasa 2023 기준값 ───────────────────────────────────────────────
# DOI: 10.1002/adma.202212161
# Figure 2f, 2g
MUKASA_2023_EXPECTED_TOP3    = ["OPD", "MAA", "4VB"]
MUKASA_2023_EXPECTED_BOTTOM2 = ["PYR", "ANI"]

# ── 허용 오차 ────────────────────────────────────────────────────────
TOLERANCE_KCAL     = 0.5   # 수치 재현 허용 오차 (kcal/mol)
SPEARMAN_THRESHOLD = 0.8   # heptachlor 합격 기준
SPEARMAN_DDT_MIN   = 0.5   # DDT fair correlation (Singh 2012)

# ── 검증용 SMILES (검증 전용 분자) ────────────────────────────────────
VALIDATION_SMILES = {
    "heptachlor": "ClC1=C(Cl)C2(Cl)C3C=CC1(Cl)C3(Cl)C2(Cl)Cl",
    "DDT":        "ClC(Cl)(Cl)c1ccc(cc1)C(c1ccc(Cl)cc1)c1ccc(Cl)cc1",
}
# MAA, 4VP, Styrene etc. are imported from config.MONOMER_LIBRARY

# ── Singh 2012 복합체 구조 기준 ──────────────────────────────────────
SINGH_COMPLEX_HBOND_DISTANCE_RANGE = (2.8, 3.6)  # Angstrom
SINGH_COMPLEX_EXPECTED_HBONDS = 2

# ── IF 예측 모델 검증 기준 ────────────────────────────────────────────
IF_MODEL_MAX_LOO_RMSE = 0.5

# ── 용매 전략 검증 기준 ──────────────────────────────────────────────
VALID_SOLVENT_STRATEGIES = ["synthesis_match", "minimum", "average", "worst"]

# ── Cross-linker 검증 기준 ───────────────────────────────────────────
CROSSLINKER_EXPECTED_WEAK = ["EGDMA"]
CROSSLINKER_WEAK_THRESHOLD = -1.0  # kcal/mol

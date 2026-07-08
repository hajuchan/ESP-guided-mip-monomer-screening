"""
MIP Screening Pipeline Configuration
=====================================
Template, monomer, interferent, and solvent settings for
computational screening of functional monomers.
"""

# ── Template ─────────────────────────────────────────────────────────
# Candidate screening targets (this run does one at a time).
TEMPLATES = {
    "Gamma-terpinene": "CC1=CCC(=CC1)C(C)C",   # C10H16 monoterpene
    "Acetic Acid":     "CC(=O)O",              # C2H4O2
    "Methyl Benzoate": "COC(=O)c1ccccc1",      # C8H8O2
}
# Active target for THIS run. Stage modules do `from .config import
# TEMPLATE_SMILES / TEMPLATE_NAME`, so these two names must exist.
# To screen another target, change the key below (or pass --template SMILES).
TEMPLATE_NAME = "Gamma-terpinene"
TEMPLATE_SMILES = TEMPLATES[TEMPLATE_NAME]

# ── Monomer Library ──────────────────────────────────────────────────
# {name: SMILES}  — covers common MIP functional monomers
MONOMER_LIBRARY = {
    "MAA":     "CC(=C)C(=O)O",           # Methacrylic acid
    "MAAD":    "CC(=C)C(=O)N",           # Methacrylamide
    "4VP":     "C(=C)c1ccncc1",          # 4-Vinylpyridine
    "OPD":     "Nc1ccccc1N",             # o-Phenylenediamine
    "ACM":     "C=CC(N)=O",              # Acrylamide
    "PYR":     "c1cc[nH]c1",             # Pyrrole
    "4VB":     "C=Cc1ccc(B(O)O)cc1",     # 4-Vinylphenylboronic acid
    "APB":     "Nc1ccc(cc1)B(O)O",       # 3-Aminophenylboronic acid
    "Styrene": "C=Cc1ccccc1",            # Styrene
    "AA":      "C=CC(=O)O",              # Acrylic acid
    "NVP":     "C=CN1CCCC1=O",           # N-Vinylpyrrolidone
    # ── Extended screening candidates ────────────────────────────────
    "ITA":     "C=C(CC(=O)O)C(=O)O",     # Itaconic acid (acidic)
    "2VP":     "C=Cc1ccccn1",            # 2-Vinylpyridine (basic)
    "VIM":     "C=Cn1ccnc1",             # 1-Vinylimidazole (basic)
    "NIPAM":   "C=CC(=O)NC(C)C",         # N-Isopropylacrylamide (neutral)
    "4VBA":    "C=Cc1ccc(C(=O)O)cc1",    # 4-Vinylbenzoic acid (hydrophobic π-π / acidic)
    # ── Hydrophobic / π-rich vinyls (nonpolar templates, e.g. terpene) ──
    "BMA":     "CCCCOC(=O)C(=C)C",         # n-Butyl methacrylate (hydrophobic)
    "LMA":     "CCCCCCCCCCCCOC(=O)C(=C)C", # Lauryl methacrylate (strong hydrophobic)
    "tBAm":    "CC(C)(C)NC(=O)C=C",        # N-tert-Butylacrylamide (hydrophobic + amide)
    "2VN":     "C=Cc1ccc2ccccc2c1",        # 2-Vinylnaphthalene (aromatic π-stack)
    "HEMA":    "CC(=C)C(=O)OCCO",          # 2-Hydroxyethyl methacrylate (H-bond OH)
    "DMAEMA":  "CC(=C)C(=O)OCCN(C)C",      # 2-(Dimethylamino)ethyl methacrylate (cationic anchor)
    # ── Silane monomers (sol-gel; separate chemistry, auto-detected) ──
    "APTES":   "NCCCO[Si](OCC)(OCC)OCC",   # 3-Aminopropyltriethoxysilane (amine anchor)
    "MPTMS":   "SCCCO[Si](OC)(OC)OC",      # 3-Mercaptopropyltrimethoxysilane (thiol)
    "PTES":    "CCO[Si](OCC)(OCC)c1ccccc1",# Phenyltriethoxysilane (π-stack)
    "IBTES":   "CC(C)CO[Si](OCC)(OCC)OCC", # Isobutyltriethoxysilane (hydrophobic)
    "UPTMS":   "O=C(N)NCCCO[Si](OC)(OC)OC",# 3-Ureidopropyltrimethoxysilane (multi H-bond)
    "CETES":   "N#CCCCO[Si](OCC)(OCC)OCC", # 3-Cyanopropyltriethoxysilane (H-bond acceptor)
}

# ── Interferent Library ──────────────────────────────────────────────
# Disabled for this screening (no selectivity comparison this round).
# When empty, Stage 3 skips selectivity and ranks by |DFT binding energy|.
# Re-enable by adding {name: SMILES} entries.
INTERFERENT_LIBRARY = {
    # "Acetic Acid":    "CC(=O)O",
    # "Ethanol":      "CCO",
    # "Acetone":     "CC(=O)C",
}

# ── Solvents (name → dielectric constant ε) ─────────────────────────
# Values from PySCF ddCOSMO / standard references  [Lipparini2013]
SOLVENTS = {
    "Chloroform":   4.71,
    "Acetonitrile": 35.69,
    # "MeOH":         32.61,
    "Toluene":      2.38,
}

# ── Pipeline Parameters ─────────────────────────────────────────────
# Stage 1 funnel cap: keep only the top-N strongest xTB binders for Stage 2 DFT.
# None = pass ALL binders (dE<0) — appropriate for a small curated library.
# Set to an int (e.g. 20) to save DFT/MMSD cost on a large library.
STAGE1_TOP_N = None
# Stage 3 global-porogen scoring uses the top-k monomers by binding (T_k). Not a
# hard funnel — all monomers still pass to Stage 4 (MMSD searches over them).
STAGE3_TOP_N = 3
N_WORKERS    = 8       # CPU parallel processes
N_GPU_WORKERS = 1      # GPU parallel processes (limited by VRAM)
USE_GPU      = True    # Use GPU acceleration when available
from pathlib import Path as _Path
OUTPUT_DIR   = str(_Path(__file__).resolve().parent.parent.parent / "results")

# ── Results Subdirectories ─────────────────────────────────────────
OUTPUT_DIRS = {
    "stage1":     f"{OUTPUT_DIR}/stage1",
    "stage2":     f"{OUTPUT_DIR}/stage2",
    "stage3":     f"{OUTPUT_DIR}/stage3",
    "stage4":     f"{OUTPUT_DIR}/stage4",   # MMSD combination search
    "stage5":     f"{OUTPUT_DIR}/stage5",   # pre-polymerization MD
    "stage6":     f"{OUTPUT_DIR}/stage6",   # VIP cavity rebinding
    "stage7":     f"{OUTPUT_DIR}/stage7",   # synthesis recipe
    "features":   f"{OUTPUT_DIR}/features",
    "validation": f"{OUTPUT_DIR}/validation",
    "reports":    f"{OUTPUT_DIR}/reports",
}


def get_output_path(stage_key: str) -> "Path":
    """Return Path for a stage output directory, creating it if needed."""
    from pathlib import Path
    p = Path(OUTPUT_DIRS[stage_key])
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Physical Constants ───────────────────────────────────────────────
HARTREE_TO_KCAL = 627.509  # 1 Hartree in kcal/mol
KB_KCAL = 0.001987204      # Boltzmann constant in kcal/(mol·K)
TEMPERATURE = 298.15        # K

# ── 추가 기능 설정 ──────────────────────────────────────────────────

# Feature 1: QuantumDock 분자 도킹 (Mukasa et al. 2023)
# 분자 표면 전체에서 orientation 생성 + GFN2-xTB SP 스크리닝
ENSEMBLE_DOCKING = True      # True: dock multiple monomer conformers, take best
N_DOCK_ORIENTATIONS = 200    # monomer당 기본 docking 방향 수 (적응형으로 자동 스케일)
N_TOP_FOR_OPTIMIZATION = 10  # SP 스크리닝 후 full optimization 대상 수
DOCK_SURFACE_OFFSET = 2.5    # vdW 표면에서 monomer 중심까지 거리 (Angstrom)
COMPLEX_SEARCH_MODE = "quantumdock"  # 내부 참조용

# DFT Method — 상호작용 유형에 따라 자동 선택
# H-bond 지배 → ωB97XD (D2 경험적 분산): H-bond 기술 우수 (Heptachlor ρ=1.000)
# 분산력 지배 → ωB97M-V (VV10 nonlocal 분산): 분산력 기술 우수 (DDT ρ=1.000)
# 판별 기준: template+monomer의 H-bond donor/acceptor 수
DFT_FUNCTIONAL_HBOND = "wb97xd"     # H-bond 지배 시스템용
DFT_FUNCTIONAL_DISP  = "wb97m-v"    # 분산력 지배 시스템용
DFT_FUNCTIONAL       = "wb97xd"     # 기본값 (fallback)
# H-bond donor/acceptor가 이 수 이상이면 H-bond 지배로 판별
HBOND_DOMINANCE_THRESHOLD = 2
DFT_OPT_BASIS  = "def2-svp"    # geometry optimization용 (빠름)
DFT_SP_BASIS   = "def2-tzvp"   # single-point energy용 (정확)

# DFT geometry optimization 여부
# False (기본) — Stage 1 xTB geometry에 DFT SP만 (DFT//xTB, 표준·빠름).
#   비공유 복합체에서 xTB geometry 오차는 상호작용에너지 ~0.5-1.5 kcal/mol 수준이며
#   계통적이라 스크리닝 랭킹은 보존됨 (Bannwarth 2019; Grimme CREST/CENSO).
# True  — 복합체를 DFT로 재최적화 후 SP (정밀하나 pair당 수 분~수십 분).
DFT_GEOMETRY_OPT = False

# DFT partial relaxation for large molecules (DFT_GEOMETRY_OPT=True 시에만)
DFT_RELAX_HEAVY_THRESHOLD = 25  # 이 이상의 heavy atom 수에서 DFT relaxation 수행
DFT_RELAX_STEPS = 5              # partial optimization 스텝 수

# Feature 2: 용매(porogen) 선택 전략
SOLVENT_STRATEGY = "global_optimal"
# "global_optimal"  — 시스템 전역에서 최적 porogen 1개를 자동 선택 후 고정 (권장)
#                     top-k(=STAGE3_TOP_N) 평균 결합에너지 argmin + protic 게이트
#                     + 저유전율 tie-breaker. MIP는 한 porogen으로 중합하므로
#                     monomer별이 아닌 시스템 단위 선택이 맞음.
#                     [van Wissen 2025; Vasapollo 2011; Suryana 2021; Liu 2021;
#                      Del Sole 2009; Rosengren 2009]
# "synthesis_match" — SYNTHESIS_SOLVENT 용매의 결합에너지 사용 (Mukasa 2023)
# "minimum"         — (비권장, 물리적으로 부적절) monomer별 최강 결합 용매
# "average"         — 전 용매 평균
# "worst"           — 가장 약한 결합에너지 (보수적 평가)
SYNTHESIS_SOLVENT = "Acetonitrile"   # global_optimal 실패/미설정 시 fallback

# global_optimal tunables
POROGEN_TIE_TAU = 1.0        # kcal/mol — Score tie band (DFT+PCM 오차 범위)
POROGEN_LOW_EPS_WARN = 3.0   # ε 이 값 미만 + 극성 template면 용해도 경고

# Feature: Cavity Shape Selectivity (Stage 3)
# MIP cavity의 3D 형상 선택성을 반영하는 보정
# interferent가 template보다 작으면 cavity에 안정적으로 들어가지 못함
CAVITY_CORRECTION = True
CAVITY_ALPHA = 0.10    # additive penalty: kcal/(mol·Å³), vdW 에너지 밀도 기반
CAVITY_BETA = 0.5      # multiplicative exponent for volume ratio


def compute_molecular_volume(smiles: str) -> float:
    """Compute 3D molecular volume (Å³) from SMILES using RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    return AllChem.ComputeMolVolume(mol)


# Feature 3: Template:Monomer 비율 스크리닝
MD_RATIO_SCREENING = True      # True면 비율별 스크리닝
MD_RATIOS_TO_TEST = [1, 2, 4]   # 테스트할 비율 목록
MD_TEMPLATE_MONOMER_RATIO = 4   # 고정 비율 (MD_RATIO_SCREENING=False 시)

# ── Stage 4: MMSD combination search ──
MMSD_ENABLE = True
MMSD_OPTIMIZER = "nsga2"       # "nsga2" | "bayesian" | "greedy" (자동 fallback)
# Funnel: Stage 2 DFT ranking → 중합 호환 monomer만 → 상위 N을 MMSD로.
MMSD_CANDIDATE_POOL = 10       # DFT 랭킹 상위 N (None이면 전체)
MMSD_POLYMERIZATION = None     # None = 모든 중합 방식(vinyl/silane/oxidative) 경쟁,
#                                MMSD가 조합을 단일 화학으로만 구성 → 이긴 화학 채택.
#                                특정 방식으로 강제하려면 "vinyl" 등 지정.
#                                (중합 타입은 SMILES에서 자동 판별)
MMSD_BE_THRESHOLD = None       # 최소 결합 강도(kcal/mol). None=제한 없음.
#                                (예: -2.0이면 그보다 약한 monomer 제외 — 약결합 template엔 주의)
MMSD_MIN_COMBO_SIZE = 2        # 조합 최소 functional monomer 수
MMSD_MAX_COMBO_SIZE = 4        # 조합 최대 functional monomer 수
MMSD_TOP_PC = 3                # MMSD가 산출하는 상위 조합(PC) 수
MMSD_MD_TOP_N = 3              # Stage 5에서 MD로 검증할 상위 조합 수

# ── Stage 5: MD Parameters ──
MD_TIME_NS = 50              # Production MD time (ns)
MD_CONTACT_CUTOFF = 6.0      # Å, contact frequency cutoff
MD_BOX_SIZE = 4.0            # nm, initial box size
MD_TEMPERATURE = 298.15      # K, simulation temperature
MD_SOLVENT = "water"         # "water" (TIP3P) or "acetonitrile" (GAFF2 explicit)
MD_INCLUDE_CROSSLINKER =True   # True: add cross-linker to MD system (Ye 2024)
MD_CROSSLINKER_RATIO = 20       # cross-linker:template molar ratio
MD_MULTI_MONOMER = True         # True: include all top monomers in one simulation
MD_MULTI_MONOMER_TOP_N = 3       # Number of top monomers to include in multi-monomer MD

# ── Stage 6: VIP Parameters ──
VIP_N_SNAPSHOTS = 5              # Equilibrium snapshots (evenly spaced)
VIP_RESTRAINT_K = 1000           # kJ/mol/nm² position restraint
VIP_REMOVAL_NS = 10             # Template removal test (ns)
VIP_REBINDING_NS = 10           # Rebinding MD (ns)
VIP_RMSD_THRESHOLD = 5.0        # Å, rebinding success
VIP_REMOVAL_THRESHOLD = 8.0     # Å, template escaped (removal OK)

# Feature 4: ESP 맵 시각화
USE_ESP_MAP = True  # True면 Stage 2에서 ESP 맵 생성

# Feature 5: Cross-linker 스크리닝
CROSSLINKER_LIBRARY = {
    # vinyl (free-radical) crosslinkers
    "EGDMA": "C=C(C)C(=O)OCCOC(=O)C(=C)C",
    "DVB":   "C=Cc1ccccc1C=C",
    "TRIM":  "C=C(C)C(=O)OCC(CC)(COC(=O)C(=C)C)OC(=O)C(=C)C",
    "BAM":   "C=CC(=O)NCCNC(=O)C=C",
    # silane (sol-gel) crosslinkers — used only for silane monomer combos
    "TEOS":  "CCO[Si](OCC)(OCC)OCC",       # Tetraethyl orthosilicate
    "TMOS":  "CO[Si](OC)(OC)OC",           # Tetramethyl orthosilicate
}
CROSSLINKER_SCREENING = True  # True면 cross-linker 후보 스크리닝
CROSSLINKER_THRESHOLD = -1.0  # kcal/mol — 이보다 강하면 부적합 경고

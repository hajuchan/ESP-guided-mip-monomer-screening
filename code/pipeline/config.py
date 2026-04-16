"""
MIP Screening Pipeline Configuration
=====================================
Template, monomer, interferent, and solvent settings for
computational screening of functional monomers.
"""

# ── Template ─────────────────────────────────────────────────────────
TEMPLATE_NAME = "Hexanal" 
TEMPLATE_SMILES = "CCCCCC=O"  

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
}

# ── Interferent Library ──────────────────────────────────────────────
INTERFERENT_LIBRARY = {
    "Acetic Acid":    "CC(=O)O",
    "Ethanol":      "CCO",
    "Acetone":     "CC(=O)C",
}

# ── Solvents (name → dielectric constant ε) ─────────────────────────
# Values from PySCF ddCOSMO / standard references  [Lipparini2013]
SOLVENTS = {
    "Acetonitrile": 35.69,
}

# ── Pipeline Parameters ─────────────────────────────────────────────
STAGE1_TOP_N = 7       # Number of monomers passed from xTB screening
STAGE3_TOP_N = 3       # Number of monomers sent to MD verification
N_WORKERS    = 11      # CPU parallel processes
N_GPU_WORKERS = 1      # GPU parallel processes (limited by VRAM)
USE_GPU      = True    # Use GPU acceleration when available
from pathlib import Path as _Path
OUTPUT_DIR   = str(_Path(__file__).resolve().parent.parent.parent / "results")

# ── Results Subdirectories ─────────────────────────────────────────
OUTPUT_DIRS = {
    "stage1":     f"{OUTPUT_DIR}/stage1",
    "stage2":     f"{OUTPUT_DIR}/stage2",
    "stage3":     f"{OUTPUT_DIR}/stage3",
    "stage4":     f"{OUTPUT_DIR}/stage4",
    "stage5":     f"{OUTPUT_DIR}/stage5",
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
ENSEMBLE_DOCKING = False      # True: dock multiple monomer conformers, take best
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

# DFT partial relaxation for large molecules
DFT_RELAX_HEAVY_THRESHOLD = 25  # 이 이상의 heavy atom 수에서 DFT relaxation 수행
DFT_RELAX_STEPS = 5              # partial optimization 스텝 수

# Feature 2: 용매 선택 전략
SOLVENT_STRATEGY = "synthesis_match"
# "synthesis_match" — SYNTHESIS_SOLVENT 용매의 결합에너지 사용 (Mukasa 2023)
# "minimum"         — 가장 음수인 결합에너지 (최대 결합력 기준)
# "average"         — 전 용매 평균
# "worst"           — 가장 약한 결합에너지 (보수적 평가)
SYNTHESIS_SOLVENT = "Acetonitrile"

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
MD_RATIO_SCREENING = False      # True면 비율별 스크리닝
MD_RATIOS_TO_TEST = [1, 2, 4]   # 테스트할 비율 목록
MD_TEMPLATE_MONOMER_RATIO = 4   # 고정 비율 (MD_RATIO_SCREENING=False 시)

# ── Stage 4: MD Parameters ──
MD_TIME_NS = 50              # Production MD time (ns)
MD_CONTACT_CUTOFF = 6.0      # Å, contact frequency cutoff
MD_BOX_SIZE = 4.0            # nm, initial box size
MD_TEMPERATURE = 298.15      # K, simulation temperature
MD_SOLVENT = "water"         # "water" (TIP3P) or "acetonitrile" (GAFF2 explicit)
MD_INCLUDE_CROSSLINKER = False   # True: add cross-linker to MD system (Ye 2024)
MD_CROSSLINKER_RATIO = 20       # cross-linker:template molar ratio
MD_MULTI_MONOMER = False         # True: include all top monomers in one simulation
MD_MULTI_MONOMER_TOP_N = 3       # Number of top monomers to include in multi-monomer MD

# ── Stage 5: VIP Parameters ──
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
    "EGDMA": "C=C(C)C(=O)OCCOC(=O)C(=C)C",
    "DVB":   "C=Cc1ccccc1C=C",
    "TRIM":  "C=C(C)C(=O)OCC(CC)(COC(=O)C(=C)C)OC(=O)C(=C)C",
    "BAM":   "C=CC(=O)NCCNC(=O)C=C",
}
CROSSLINKER_SCREENING = False
CROSSLINKER_THRESHOLD = -1.0  # kcal/mol — 이보다 강하면 부적합 경고

# ESP-guided MIP Monomer Screening Pipeline

Molecularly Imprinted Polymer(MIP) 합성을 위한 최적 functional monomer를 계산화학적으로 스크리닝하는 6-Stage 파이프라인.

---

## 파이프라인 전체 구조

```
Stage 1: ESP-guided 분자 표면 도킹 + GFN2-xTB 스크리닝
         ├── DFT ESP 전하 계산 (GPU, B3LYP/def2-SVP)
         ├── ESP-guided vdW 표면 도킹 (적응형 ~200-400개)
         ├── AutoDock Vina 도킹 (exhaustiveness=64)
         ├── GFN2-xTB SP 스크리닝 → 상위 10개 선별
         └── xTB full optimization → dE < 0 필터 (결합하는 것만 통과)
    ↓ N개 (자동 결정)
Stage 2: DFT 정밀 결합에너지 (GPU)
         ├── GPU DFT geometry optimization (def2-SVP + RI-J + geomeTRIC)
         ├── 적응형 범함수: H-bond 지배 → ωB97XD / 분산력 지배 → ωB97M-V
         ├── SP energy: def2-TZVP + PCM 용매
         └── BSSE counterpoise (gas-phase ghost atom)
    ↓ N개 (필터 없음)
Stage 3: 선택도 평가 + Cross-linker 추천
         ├── monomer-interferent xTB 도킹 → DFT 결합에너지
         ├── Cavity shape correction (분자 볼륨 기반)
         ├── S = exp(ΔE / kT), Mukasa 2023 공식
         └── Cross-linker DFT 스크리닝 → 최적 cross-linker 추천
    ↓ N개 (필터 없음, 순위 참고)
Stage 4: Pre-polymerization MD (GROMACS, GPU)
         ├── GAFF2 parameterization (acpype) + 보론 B→C 치환
         ├── Template + N×monomer + TIP3P water
         ├── EM → NVT → NPT → 50ns Production MD
         ├── Contact frequency, RDF, EBN 분석
         └── 합성 비율 자동 결정 (contact freq 역비례)
    ↓ N개 (필터 없음)
Stage 5: VIP Cavity Rebinding (GROMACS)
         ├── 균등 간격 5개 snapshot 선택 (cherry-picking 방지)
         ├── Monomer position restraint (1000 kJ/mol/nm²) → 중합 근사
         ├── Template removal test → 이탈하면 "제거 가능" (적정 결합 = 좋은 MIP)
         ├── Template rebinding MD → RMSD < 5Å = cavity 인식 성공
         ├── Interferent rebinding → own만 성공이면 selective
         └── VIP score = both_rate × (1 + selectivity)
    ↓ VIP 순위
Stage 6: 합성 레시피 자동 생성
         ├── Top 3 monomer + cross-linker + 비율
         └── 합성 프로토콜 (free-radical polymerization)
```

---

## 각 Stage 상세

### Stage 1: ESP-guided 분자 표면 도킹 + GFN2-xTB 스크리닝

**파일**: `code/pipeline/stage1_xtb.py`

Template과 monomer의 최적 결합 배향을 탐색한다. Mukasa et al. (2023)의 다단계 스크리닝 전략을 기반으로, DFT ESP 전하 기반 배향 생성과 AutoDock Vina 도킹을 결합했다.

**ESP(Electrostatic Potential)**: 분자 주변 공간의 전하 분포 지도. 양전하(파란색) 영역에는 monomer의 음전하 부분을, 음전하(빨간색) 영역에는 양전하 부분을 마주보게 배치하여 수소결합과 정전기 상호작용을 최대화한다.

**프로토콜**:
1. DFT 레벨 Mulliken 전하 계산 (B3LYP/def2-SVP, GPU ~7초/분자)
2. vdW 표면 위에 ESP-guided orientation 생성 (분자 크기 적응형)
3. AutoDock Vina 도킹 (exhaustiveness=64) — 소수성/형태 적합성 보완
4. GFN2-xTB SP screening → 상위 10개 선별
5. xTB full optimization (L-BFGS-B) → 최적 결합에너지 + 좌표
6. **필터: dE < 0 (결합에너지 음수)인 monomer만 통과** — 개수 자동 결정

**핵심 기술**:
- Fibonacci sphere sampling으로 vdW 표면점 생성
- meeko 0.7+ API로 PDBQT 생성 (REMARK IDX 매핑으로 좌표 복원)
- 최적화된 복합체 좌표를 `complex_coords`로 JSON 저장 → Stage 2 전달

### Stage 2: DFT 정밀 결합에너지 계산

**파일**: `code/pipeline/stage2_dft.py`

Stage 1에서 찾은 최적 복합체 구조를 DFT 레벨에서 정밀 계산한다.

**적응형 범함수 선택** (v6):
- H-bond 지배 시스템 (D+A ≥ 2): ωB97XD — Heptachlor ρ=1.000
- 분산력 지배 시스템 (D+A < 2): ωB97M-V (VV10 nonlocal) — DDT ρ=1.000
- RDKit `Lipinski.NumHDonors()` + `NumHAcceptors()`로 자동 판별

**2단계 기저함수**:
- Geometry optimization: def2-SVP + RI-J (density fitting, ~3배 가속)
- Single-point energy: def2-TZVP (정확)

**BSSE 보정**: Boys-Bernardi counterpoise를 **gas-phase에서** 수행 (PCM cavity 왜곡 방지).

**xTB→DFT 좌표 전달**: Stage 1의 `complex_coords`를 읽어 `prebuilt_complex_mol`로 DFT에 전달. 방향 문자열이 아닌 실제 좌표를 전달하여 PES 불일치 방지.

### Stage 3: 선택도 평가 + Cross-linker 추천

**파일**: `code/pipeline/stage3_selectivity.py`

각 monomer가 template에 얼마나 **선택적으로** 결합하는지 평가한다. 단순 결합 강도가 아닌, interferent 대비 선택도를 계산한다.

**선택도 공식** (Mukasa 2023):
```
ΔE = |E(monomer-template)| - |E(monomer-interferent)|
S = exp(ΔE / kBT)
양수 = template에 선택적
```

**Cavity shape correction**: interferent가 template보다 작으면 MIP cavity에 안정적으로 들어가지 못함을 반영.
```
V_ratio = V_interferent / V_template
f_cavity = V_ratio^β (β=0.5)
E_Int_eff = E_Int × f_cavity + α × max(V_template - V_interferent, 0)
```

**Cross-linker 자동 추천**: Stage 3 실행 시 cross-linker DFT 스크리닝을 함께 수행. Template과 가장 약하게 결합하는 cross-linker를 추천 (좋은 cross-linker = template과 경쟁 안 함).

**필터링 없음**: 모든 monomer를 Stage 4로 전달. 선택도는 참고 지표로만 사용.

### Stage 4: Pre-polymerization MD (GROMACS)

**파일**: `code/pipeline/stage4_md.py`, `code/pipeline/utils_gromacs.py`

Template + monomer의 동적 결합 행동을 MD 시뮬레이션으로 평가한다.

**시스템 구성**:
- GAFF2 force field (acpype parameterization)
- 보론산(B) 분자: B→C 치환 + 문헌 B 파라미터 (Gerogiokas 2020)
- Template 1개 + Monomer 4개 + TIP3P 수상자
- 50ns NVT production MD (GROMACS GPU)

**분석**:
- Contact frequency (6Å cutoff): monomer가 template 근처에 머무는 빈도
- RDF (Radial Distribution Function): 거리별 monomer 밀도
- EBN (Effective Binding Number): 첫 번째 solvation shell 내 coordination number

**합성 비율 자동 결정**: Contact frequency의 역비례 — 약한 결합 monomer를 더 많이 넣어 균등한 cavity 형성.

### Stage 5: VIP Cavity Rebinding (GROMACS)

**파일**: `code/pipeline/stage5_vip.py`

Virtually Imprinted Polymer (VIP) 방식으로 실제 MIP cavity 형성과 rebinding을 시뮬레이션한다 (Zink & Moura, PCCP 2018).

**Inverted-U 관계 해결**:
- 너무 강한 결합 → template 제거 불가 → bad cavity → 낮은 IF
- 너무 약한 결합 → 인식점 없음 → 낮은 IF
- **적당한 결합 → 깨끗한 제거 + 성공적 rebinding → 높은 IF**

**프로토콜**:
1. Stage 4 trajectory 후반 50%에서 **균등 간격 5개 snapshot** 선택
2. Monomer position restraint (1000 kJ/mol/nm²) → 중합 근사
3. **Template removal test** (10ns): template이 이탈하면 제거 가능 (moderate binding = good)
4. **Rebinding MD** (10ns): template RMSD < 5Å → cavity 인식 성공
5. **Selectivity**: interferent rebinding → own만 성공이면 selective
6. **VIP score = both_rate × (1 + selectivity)** → 최종 순위

### Stage 6: 합성 레시피 자동 생성

**파일**: `code/pipeline/stage6_recipe.py`

Stage 5 VIP 순위, Stage 4 합성 비율, Stage 3 cross-linker 추천을 종합하여 합성 프로토콜을 자동 생성한다.

**출력**:
- `synthesis_recipe.json`: Top 3 monomer + cross-linker + 비율
- `synthesis_protocol.txt`: 단계별 합성 프로토콜 (free-radical polymerization)

---

## 추가 기능

| 기능 | 파일 | 설명 |
|------|------|------|
| ESP 맵 시각화 | `stage2_dft.py` | 3D vdW 표면 ESP (plotly interactive HTML + PNG) |
| 자동 Interferent 제안 | `suggest_interferents.py` | Tanimoto 유사도 + PubChem API로 후보 추천 |
| IF 예측 모델 | `predict_if.py` | 문헌 데이터 기반 Ridge/RF 회귀 (LOO-CV) |
| HTML 리포트 | `generate_report.py` | 전체 결과 통합 HTML (base64 이미지 embed) |

---

## 실행 방법

```bash
conda activate MIPscreen
source /usr/local/gromacs-gpu/bin/GMXRC
cd MIP_simulation

# config.py에서 TEMPLATE_SMILES, MONOMER_LIBRARY 설정 후:

# 전체 파이프라인 (Stage 1→2→3→4→5→6)
python run_pipeline.py --stage all --output-dir results/hexanal

# 개별 Stage
python run_pipeline.py --stage 1 --output-dir results/hexanal
python run_pipeline.py --stage 4 --output-dir results/hexanal
python run_pipeline.py --stage 5 --output-dir results/hexanal

# 추가 기능
python run_pipeline.py --crosslinker --output-dir results/hexanal
python run_pipeline.py --suggest-interferents
python run_pipeline.py --report --output-dir results/hexanal
```

### 디렉토리 구조

```
MIP_simulation/
├── run_pipeline.py              # 엔트리포인트
├── run_validation.py            # 검증 실행
├── run_selectivity.py           # 선택도 검증 (범함수 비교)
├── environment.yml              # conda 환경
├── code/pipeline/
│   ├── config.py                # 전역 설정
│   ├── run_pipeline.py          # Stage 오케스트레이터
│   ├── stage1_xtb.py            # ESP 도킹 + xTB 스크리닝
│   ├── stage2_dft.py            # DFT 결합에너지
│   ├── stage3_selectivity.py    # 선택도 + cross-linker
│   ├── stage4_md.py             # GROMACS pre-polymerization MD
│   ├── stage5_vip.py            # VIP cavity rebinding
│   ├── stage6_recipe.py         # 합성 레시피
│   ├── utils_gromacs.py         # GROMACS 유틸리티
│   ├── crosslinker.py           # Cross-linker DFT
│   ├── generate_report.py       # HTML 리포트
│   ├── suggest_interferents.py  # Interferent 자동 제안
│   └── predict_if.py            # IF 예측 모델
├── code/validation/
│   ├── run_validation.py        # 검증 오케스트레이터
│   ├── compute_reference.py     # 문헌 기준 DFT 계산
│   ├── compute_selectivity.py   # 범함수별 선택도 비교
│   ├── config_validation.py     # 검증 기준값
│   └── validate_*.py            # 각종 검증
└── results/
    ├── hexanal/stage1~6/        # Hexanal 결과
    ├── nonanal/stage1~6/        # Nonanal 결과
    └── validation/              # 검증 결과
```

---

## 핵심 파라미터

| Stage | 파라미터 | 값 | 근거 |
|-------|---------|-----|------|
| 1 | ESP 전하 | B3LYP/def2-SVP Mulliken (GPU) | DFT 레벨 정확도, ~7초/분자 |
| 1 | Vina exhaustiveness | 64 | MIP 소분자 복합체용 (기본 8의 8배) |
| 1 | 필터 기준 | dE < 0 | 결합하는 monomer만 통과 (개수 자동) |
| 2 | 범함수 | ωB97XD / ωB97M-V (적응형) | H-bond/분산력 시스템별 최적 |
| 2 | 기저함수 | def2-SVP (opt) / def2-TZVP (SP) | 2단계 기저 (속도+정확도) |
| 2 | 용매 모델 | PCM (IEF-PCM) | GPU gradient 지원 (ddCOSMO 불가) |
| 3 | Cavity α | 0.10 kcal/(mol·Å³) | vdW 에너지 밀도 기반 물리 상수 |
| 3 | Cavity β | 0.5 | 비선형 cavity filling (표면적 ∝ V^(2/3)) |
| 4 | Force field | GAFF2 (acpype) | 소분자 표준 |
| 4 | 보론 파라미터 | B→C 치환 + 문헌값 | Gerogiokas 2020 |
| 4 | MD 시간 | 50 ns | 평형 도달 |
| 5 | Snapshot | 균등 간격 5개 | Cherry-picking 방지 (Zink 2018) |
| 5 | Position restraint | 1000 kJ/mol/nm² | 중합 근사 |
| 5 | Rebinding 기준 | RMSD < 5 Å | Cavity 인식 성공 |
| 5 | Removal 기준 | RMSD > 8 Å | Template 이탈 (제거 가능) |

---

## 사용된 계산화학 방법론

| 방법 | 소프트웨어 | 근거 논문 | Stage |
|------|-----------|----------|-------|
| GFN2-xTB semiempirical | tblite [6] | Bannwarth et al. 2019 | 1 |
| DFT ωB97XD (H-bond) | gpu4pyscf [12] + PySCF [11] | Chai & Head-Gordon 2008 | 2 |
| DFT ωB97M-V (dispersion) | gpu4pyscf [12] + PySCF [11] | Mardirossian & Head-Gordon 2017 [7] | 2 |
| def2-SVP / def2-TZVP 기저 | PySCF [11] | Weigend & Ahlrichs 2005 | 2 |
| RI-J density fitting | gpu4pyscf [12] | — | 2 |
| PCM implicit solvation | PySCF [11] | — | 2 |
| BSSE counterpoise (gas-phase) | PySCF [11] | Boys & Bernardi 1970 [8] | 2 |
| geomeTRIC optimizer | geometric [15] | Wang & Song 2016 | 2 |
| AutoDock Vina docking | vina + meeko [14] | Trott & Olson 2010 | 1 |
| Selectivity S ∝ exp(ΔE/kT) | numpy | Mukasa et al. 2023 [1] | 3 |
| Cavity shape correction | RDKit (ComputeMolVolume) | 본 연구 (vdW 에너지 밀도 기반) | 3 |
| GAFF2 parameterization | acpype [18] + AmberTools | Wang et al. 2004 | 4 |
| Boron B→C substitution | acpype [18] + custom frcmod | Gerogiokas et al. 2020 [10] | 4 |
| Pre-polymerization MD | GROMACS [13] GPU | Muñoz et al. 2024 [4] | 4 |
| Contact frequency / EBN | MDAnalysis [16] | Ye et al. 2024 [5] | 4 |
| VIP cavity rebinding | GROMACS [13] + MDAnalysis [16] | Zink & Moura 2018 [3] | 5 |

---

## 설치

```bash
# 환경 생성
conda env create -f environment.yml
conda activate MIPscreen

# GROMACS (시스템 설치, 별도)
source /usr/local/gromacs-gpu/bin/GMXRC

# 확인
python -c "from tblite.interface import Calculator; print('tblite OK')"
python -c "import pyscf; print('pyscf OK')"
python -c "import MDAnalysis; print('MDAnalysis OK')"
gmx --version
```

### 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | CUDA 지원 NVIDIA | RTX 4070 Ti+ (12GB+ VRAM) |
| CPU | 8코어 | 16코어 |
| RAM | 16GB | 32GB |
| GROMACS | 2023+ (GPU build) | 2025.2 |

---

## 참고 논문

### 방법론 핵심 논문

| # | 저자 | 저널 | 년도 | DOI | 파이프라인 적용 |
|---|------|------|------|-----|---------------|
| 1 | Mukasa et al. | *Adv. Mater.* | 2023 | 10.1002/adma.202212161 | Stage 3 선택도 공식 S ∝ exp(ΔE/kBT), 다단계 스크리닝 전략 |
| 2 | Singh et al. | *Curr. Anal. Chem.* | 2012 | 10.2174/157341112803216807 | DFT MIP 스크리닝 원형, BSSE 보정, 실험 IF 검증 데이터 |
| 3 | Zink & Moura | *Phys. Chem. Chem. Phys.* | 2018 | 10.1039/c7cp08284c | Stage 5 VIP cavity rebinding (position restraint, template removal, rebinding MD) |
| 4 | Muñoz et al. | *J. Chem. Inf. Model.* | 2024 | 10.1021/acs.jcim.4c00775 | Stage 4 pre-polymerization MD, contact frequency, monomer 선별 |
| 5 | Ye et al. | *Molecules* | 2024 | 10.3390/molecules29174236 | Stage 4 EBN/HBNmax 정량 파라미터, H-bond 점유율 분석 |

### 계산화학 방법론

| # | 저자 | 저널 | 년도 | DOI | 적용 |
|---|------|------|------|-----|------|
| 6 | Bannwarth et al. | *J. Chem. Theory Comput.* | 2019 | 10.1021/acs.jctc.8b01176 | GFN2-xTB (Stage 1 fast screening) |
| 7 | Mardirossian & Head-Gordon | *J. Chem. Phys.* | 2017 | 10.1063/1.4986508 | ωB97M-V 범함수 (분산력 지배 시스템) |
| 8 | Boys & Bernardi | *Mol. Phys.* | 1970 | 10.1080/00268977000101561 | BSSE counterpoise 보정 (gas-phase ghost atom) |
| 9 | Bursch et al. | *Angew. Chem. Int. Ed.* | 2022 | 10.1002/anie.202205735 | DFT best-practice: def2-TZVP 기저, 적응형 범함수 근거 |
| 10 | Gerogiokas et al. | *Molecules* | 2020 | 10.3390/molecules25092196 | 보론산 GAFF2 파라미터화 (B→C 치환 + 문헌 파라미터) |

### 소프트웨어

| # | 소프트웨어 | 저자 | 저널 | 년도 | DOI |
|---|-----------|------|------|------|-----|
| 11 | PySCF | Sun et al. | *J. Chem. Phys.* | 2020 | 10.1063/5.0006074 |
| 12 | GPU4PySCF | Wu et al. | *arXiv* | 2024 | 10.48550/arXiv.2404.09452 |
| 13 | GROMACS | Abraham et al. | *SoftwareX* | 2015 | 10.1016/j.softx.2015.06.001 |
| 14 | AutoDock Vina | Trott & Olson | *J. Comput. Chem.* | 2010 | 10.1002/jcc.21334 |
| 15 | geomeTRIC | Wang & Song | *J. Chem. Phys.* | 2016 | 10.1063/1.4952956 |
| 16 | MDAnalysis | Michaud-Agrawal et al. | *J. Comput. Chem.* | 2011 | 10.1002/jcc.21787 |
| 17 | RDKit | Landrum | Open-source | 2006– | rdkit.org |
| 18 | acpype | Sousa da Silva & Vranken | *BMC Res. Notes* | 2012 | 10.1186/1756-0500-5-367 |

# ESP-guided MIP Monomer Screening Pipeline

Molecularly Imprinted Polymer(MIP) 합성을 위한 최적 functional monomer를 계산화학적으로 스크리닝하는 7-Stage 파이프라인.

> **주요 업데이트 (2026)**: 다중 target 자동 실행(`--all-templates`, `MIP_TEMPLATE`) · CPU 워커 자동 사이징(코어 80% + RAM 상한, `MIP_WORKERS`) · VIP 전면 개편(50ns/10snap, Q4 평형 RMSD + 수렴 진단, residence+retention 연속 점수, `--reanalyze-vip`/`--extend-vip`, multi-restart, MM-GBSA) · Stage 7 결합-가중 랭킹 + 화학 일관성 필터 + inert 가교제 선택 · 실란 PolCA MD 지원 · GROMACS 2026.3 대응(tpx→.gro 폴백, maxwarn 50) · DFT GPU-OOM→CPU 자동 폴백.

---

## 파이프라인 전체 구조

```
Stage 1: ESP-guided 분자 표면 도킹 + GFN2-xTB 스크리닝
         ├── DFT ESP 전하 계산 (GPU, B3LYP/def2-SVP)
         ├── Functional group focused sampling (N,O,S 주변 5× 가중치)
         ├── ESP-guided vdW 표면 도킹 (적응형 ~200개)
         ├── AutoDock Vina 도킹 (exhaustiveness=64)
         ├── GFN2-xTB SP 스크리닝 → 상위 10개 선별
         └── xTB full optimization → dE < 0 필터 (결합하는 것만 통과)
    ↓ N개 (자동 결정)
Stage 2: DFT 정밀 결합에너지 (GPU)
         ├── SP-only (기본): Stage 1 xTB geometry에 DFT SP (DFT//xTB, 빠름)
         │         └ DFT_GEOMETRY_OPT=True면 GPU DFT 재최적화(def2-SVP+RI-J+geomeTRIC)
         ├── 적응형 범함수: H-bond 지배 → ωB97XD / 분산력 지배 → ωB97M-V
         ├── SP energy: def2-TZVP + PCM 용매 (분산 내장, 외부 D3 미사용)
         └── BSSE counterpoise (gas-phase ghost atom)
    ↓ N개 (필터 없음)
Stage 3: 전역 Porogen 선택 + 선택도 평가 + Cross-linker 추천
         ├── Global porogen 자동 선택 (top-k 평균 결합에너지 argmin
         │         + protic H-bond 게이트 + 저유전율 tie-break) → 시스템 전역 고정
         ├── (interferent 있음) monomer-interferent DFT 결합에너지 → S = exp(ΔE/kT)
         ├── (interferent 없음) porogen 내 |결합에너지| 랭킹으로 대체
         ├── Cavity shape correction (분자 볼륨 기반)
         └── Cross-linker DFT 스크리닝 → 최적 cross-linker 추천
    ↓ N개 (필터 없음, 순위 참고)
Stage 4: MMSD 다중 monomer 조합 탐색
         ├── 후보 pool = Stage 3 DFT 랭킹 상위 N(=10)
         ├── 중합 방식 경쟁: vinyl/silane/oxidative (조합은 단일 화학, Liu 2017)
         ├── template 중심 순차 xTB 도킹 (greedy / Bayesian / NSGA-II)
         ├── synergy 판정: delta = mmsd_sum − smd_sum (<0 협동)
         ├── chemistry diversity 필터 + 화학별 crosslinker (vinyl→EGDMA, silane→TEOS)
         └── 상위 3개 조합(PC) → stage4/mmsd_results.json
    ↓ 상위 3개 조합 (Stage 5 MD로)
Stage 5: Pre-polymerization MD (GROMACS, GPU)
         ├── GAFF2 parameterization (acpype) + 보론 B→C 치환 + 실란 PolCA (Si, Jorge 2021)
         ├── Template + N×monomer + cross-linker + 용매 (MMSD 상위 3조합 각각 MD 검증)
         ├── EM → NVT → NPT → 50ns Production MD
         ├── Contact frequency, RDF, EBN, H-bond + SASA/FFV(형태학) + MM-GBSA 분석
         └── 합성 비율 자동 결정 (EBN 정비례, Yuan 2024)
    ↓ N개 (필터 없음)
Stage 6: VIP Cavity Rebinding (GROMACS) — BIO phase5 방식
         ├── 균등 간격 10개 snapshot 선택 (cherry-picking 방지)
         ├── Monomer position restraint (1000 kJ/mol/nm²) → 중합 근사
         ├── Template removal test (10ns) → RMSD > 8Å = 이탈 성공
         ├── Template rebinding MD (50ns) → Q4 평형 RMSD + Q1→Q4 수렴 진단
         ├── Residence(잔류시간) + 연속 retention 점수 → 약결합도 순위화
         ├── 옵션: multi-restart 앙상블 · MM-GBSA ΔG · --reanalyze-vip/--extend-vip
         └── VIP score = ½(retention + residence) × (1 + selectivity)
    ↓ VIP 순위
Stage 7: 합성 레시피 자동 생성
         ├── 모노머 랭킹 = |DFT 결합|×contact (결합-가중) → 승리 화학으로 필터(일관성)
         ├── Cross-linker = 화학-호환 중 가장 inert(약결합, 예: EGDMA) → 비특이 자리 최소화
         ├── Top 조합 + porogen + 화학량론(Kass 기반 1:1/과잉)
         ├── Chemistry-aware 보정: 이온화·자기회합·crosslinker 활성종·bleeding (문헌 감사)
         └── 화학별 프로토콜 (free-radical / sol-gel / oxidative) + Design considerations
```

---

## 각 Stage 상세

### Stage 1: ESP-guided 분자 표면 도킹 + GFN2-xTB 스크리닝

**파일**: `code/pipeline/stage1_xtb.py`

Template과 monomer의 최적 결합 배향을 탐색한다. Mukasa et al. (2023)의 다단계 스크리닝 전략을 기반으로, DFT ESP 전하 기반 배향 생성과 AutoDock Vina 도킹을 결합했다.

**ESP(Electrostatic Potential)**: 분자 주변 공간의 전하 분포 지도. 양전하(파란색) 영역에는 monomer의 음전하 부분을, 음전하(빨간색) 영역에는 양전하 부분을 마주보게 배치하여 수소결합과 정전기 상호작용을 최대화한다.

**프로토콜**:
1. DFT 레벨 Mulliken 전하 계산 (B3LYP/def2-SVP, GPU ~7초/분자)
2. **Functional group focused sampling**: 헤테로원자(N, O, S) 및 인접 원자 주변 표면점에 5× 가중치 부여 → C=O, OH, NH₂ 등 반응성 기능기 주변에 도킹 배향 집중 (~47% 배향이 기능기 5Å 이내)
3. AutoDock Vina 도킹 (exhaustiveness=64) — 소수성/형태 적합성 보완
4. GFN2-xTB SP screening → 상위 10개 선별
5. xTB full optimization (L-BFGS-B) → 최적 결합에너지 + 좌표
6. **필터: dE < 0 (결합에너지 음수)인 monomer만 통과** — 개수 자동 결정

**핵심 기술**:
- Fibonacci sphere sampling으로 vdW 표면점 생성
- **Functional group focused docking**: `functional_group_atoms` (Z ∈ {7,8,16} + neighbors)에 대해 가장 가까운 표면점 가중치 5배 증가. 균일 표면 샘플링의 편향 문제 해결 (긴 알킬 체인보다 C=O 등 반응 부위에 집중)
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

**GPU OOM 폴백**: 큰 분자(예: TRIM 가교제)가 GPU VRAM을 초과하면(`cudaError`) 동일 functional/basis로 **CPU에서 자동 재계산**한다 — 에너지 동일, 느리지만 완주. Stage 3 가교제 스크리닝에도 적용.

**xTB→DFT 좌표 전달**: Stage 1의 `complex_coords`를 읽어 `prebuilt_complex_mol`로 DFT에 전달. 방향 문자열이 아닌 실제 좌표를 전달하여 PES 불일치 방지.

**ESP 시각화**: 3D vdW 표면 ESP 맵 (plotly interactive HTML + publication-quality PNG). 복합체 상태의 ESP도 생성하여 결합 부위 확인 가능.

### Stage 3: 전역 Porogen 선택 + 선택도 평가 + Cross-linker 추천

**파일**: `code/pipeline/stage3_selectivity.py`

각 monomer가 template에 얼마나 **선택적으로** 결합하는지 평가한다. 단순 결합 강도가 아닌, interferent 대비 선택도를 계산한다.

**전역 Porogen 자동 선택** (`select_global_porogen`, `SOLVENT_STRATEGY="global_optimal"`):

MIP는 **하나의 porogen**으로 중합되므로, 용매는 monomer별이 아니라 **시스템 전역에서 1개를 선택해 고정**해야 한다 (문헌 검증: van Wissen 2025; Vasapollo 2011). Stage 2가 template·monomer·complex를 **동일 PCM 유전율**에서 계산하므로, "가장 강한 결합 용매 선택"(Suryana 2021)과 "solvation 교란 최소화"(Liu 2021)가 동일한 저유전율 aprotic 답으로 수렴한다.

알고리즘 (`per-solvent × per-monomer` BSSE 행렬 → porogen 1개):
1. **T_k** = 용매평균 결합 상위 k(=`STAGE3_TOP_N`) monomer
2. **Score(s)** = T_k 평균 결합에너지 → argmin (가장 음수)
3. **Protic 하드 게이트** — H-bond MIP면 양성자성 용매(MeOH/물) 거부 (H-bond 파괴; Del Sole 2009)
4. **유전율 tie-breaker** — Score 오차밴드(τ=1.0 kcal/mol) 내 **최저 ε** 선택 (H-bond 보존)
5. **용해도 sanity** — 저유전율 porogen + 극성 template면 경고 + 고ε aprotic fallback 제시

선택된 porogen은 `stage3/global_porogen.json`에 기록되어 **Stage 6 레시피**까지 일관 적용된다 (solvent memory).

**선택도 공식** (Mukasa 2023, interferent가 있을 때):
```
ΔE = |E(monomer-template)| - |E(monomer-interferent)|   (전역 porogen 내에서)
S = exp(ΔE / kBT)
양수 = template에 선택적
```

**Interferent 없을 때**: 선택도 계산을 건너뛰고 **porogen 내 |결합에너지| 랭킹**으로 대체한다 (`INTERFERENT_LIBRARY = {}`).

**Cavity shape correction**: interferent가 template보다 작으면 MIP cavity에 안정적으로 들어가지 못함을 반영.
```
V_ratio = V_interferent / V_template
f_cavity = V_ratio^β (β=0.5)
E_Int_eff = E_Int × f_cavity + α × max(V_template - V_interferent, 0)
```

**Cross-linker 자동 추천**: Stage 3 실행 시 cross-linker DFT 스크리닝을 함께 수행. Template과 가장 약하게 결합하는 cross-linker를 추천 (좋은 cross-linker = template과 경쟁 안 함). **항목별 캐시**: 완료된 (가교제×용매) 쌍은 skip, 실패분(예: TRIM OOM)만 재시도. 결과는 `features/crosslinker.json`에 공유 저장되어 Stage 7이 화학-호환 중 가장 inert한 것을 최종 선택.

**필터링 없음**: 모든 monomer를 Stage 4로 전달. 선택도는 참고 지표로만 사용.

### Stage 4: MMSD 다중 monomer 조합 탐색

**파일**: `code/pipeline/stage4_mmsd.py`

단일 monomer 순위를 넘어 **최적 monomer 조합**을 탐색한다. `Monomer_screening_in_Bio`의 MMSD (Rajpal 2024)를 이 프로젝트의 **template 중심 xTB 도킹**으로 이식한 것으로, 단백질 대신 template 소분자에 monomer를 하나씩 순차로 붙여 xTB 최적화한다.

Stage 3에서 porogen을 고정하고 monomer 랭킹을 확정한 **뒤** 그 안에서 조합을 최적화하므로, 순서상 selectivity(3) → MMSD(4)가 맞다. (형제 프로젝트 `Monomer_screening_in_Bio`와 동일 철학: 넓은 pool → MMSD → 최적 조합 → MD)

- **후보 pool (funnel)**: Stage 3 DFT 랭킹 → **중합 호환(free-radical=vinyl) monomer만** + **DFT 상위 `MMSD_CANDIDATE_POOL`(=10)**. C=C 없는 monomer(PYR/OPD/APB 등 산화중합)는 SMILES 자동판별로 제외 → Stage 2 DFT가 "누가 MMSD로 갈지" 결정. (Liu 2017: 다른 중합 타입 불혼합)
- **탐색 엔진**: NSGA-II(기본, pymoo) / Bayesian(skopt) / greedy forward+swap — 자동 fallback
- **순차 MMSD**: template에 monomer를 하나씩 붙여 xTB 최적화 → 복합체 결합에너지 누적
- **Synergy 지표**: `delta = mmsd_sum − smd_sum` (<0 = 협동결합, >0 = 입체 간섭)
- **목적함수**: `mmsd_per_monomer + 0.3·max(0, delta)` (크기 정규화)
- **필터**: polymerization 호환성 + chemistry diversity(class ≥2, class당 ≤2) + 최적 crosslinker 순차 선택
- 결과는 `stage4/mmsd_results.json`에 **상위 조합(top PC) `MMSD_TOP_PC`개**로 저장 → Stage 5 MD로 전달
- 설정: `MMSD_ENABLE`, `MMSD_OPTIMIZER`, `MMSD_CANDIDATE_POOL`, `MMSD_TOP_PC`, `MMSD_MD_TOP_N` (config.py)

### Stage 5: Pre-polymerization MD (GROMACS)

**파일**: `code/pipeline/stage5_md.py`, `code/pipeline/utils_gromacs.py`

Template + monomer의 동적 결합 행동을 MD 시뮬레이션으로 평가한다. 개별 monomer MD에 더해, Stage 4 MMSD의 **상위 `MMSD_MD_TOP_N`(기본 3)개 조합**을 각각 multi-monomer MD로 검증하고 contact frequency로 재순위한다 (`multi_monomer_pc1..pc3/`, `stage5_combination.json`).

**시스템 구성**:
- GAFF2 force field (acpype Python API로 parameterization)
- 보론산(B) 분자: B→C 치환 + 문헌 B 파라미터 (Gerogiokas 2020)
- **실란(Si) monomer: PolCA organosilane force field** (Jorge 2021) — GAFF2엔 Si 없어 예전엔 skip했으나, 이제 Si0~Si4 LJ 파라미터로 MD 실행 (`_generate_silane_itp`)
- Template 1개 + Monomer 4개 + cross-linker + 용매 (`MD_SOLVENT`: acetonitrile 또는 TIP3P water)
- EM → NVT (1ns) → NPT (1ns) → 50ns Production MD (GROMACS GPU)
- Checkpoint resume 지원 (`-cpi md.cpt -append`); 완료된 궤적은 skip, 분석만 재실행
- **MM-GBSA ΔG** (gmx_MMPBSA, 옵션): 단일 pose 엔탈피와 전체 혼합물 자유에너지 사이의 앙상블 ΔG (`MD_MMPBSA`)

**GROMACS 2026 대응**: 최신 GROMACS(2026.3)는 `.tpr`을 tpx v138로 저장하는데 설치된 MDAnalysis가 못 읽음 → `load_universe()`가 자동으로 `.gro`로 폴백(버전 무관). grompp 경고 다수(atomtype 재정의) → `-maxwarn 50`으로 흡수.

**분석** (MDAnalysis):
- Contact frequency (6Å cutoff): monomer가 template 근처에 머무는 빈도
- RDF (Radial Distribution Function): 거리별 monomer 밀도
- EBN (Effective Binding Number): 첫 번째 solvation shell 내 coordination number
- **H-bond 분석**: bidirectional HydrogenBondAnalysis (TPR topology, 전체 trajectory)
- Template/monomer 식별: resid 기반 (acpype "UNL" resname 문제 회피)

**합성 비율 자동 결정** (`MD_RATIO_METHOD="ebn"`, 기본): **EBN(최대 동시 결합 수) 정비례** — 표적에 자리가 많은(EBN↑) monomer를 그만큼 더 많이 넣는다 (Yuan et al. 2024, 형제 프로젝트와 동일). `"contact_inverse"`로 바꾸면 약결합 monomer를 더 넣는 기존 방식(보상).

### Stage 6: VIP Cavity Rebinding (GROMACS)

**파일**: `code/pipeline/stage6_vip.py`

Virtually Imprinted Polymer (VIP) 방식으로 실제 MIP cavity 형성과 rebinding을 시뮬레이션한다 (Zink & Moura, PCCP 2018). Stage 5 MD trajectory에서 snapshot을 선택한다.

**프로토콜** (형제 `Monomer_screening_in_Bio/phase5_rebinding.py` 이식):
1. Stage 5 MD trajectory 후반 50%에서 **균등 간격 10개 snapshot** 선택
2. Monomer position restraint (1000 kJ/mol/nm²) → 중합 근사
3. **Template removal test** (10ns): template RMSD > 8Å → 이탈 성공 (제거 가능)
4. **Rebinding MD** (50ns): template 5Å 변위 후 release → 아래 지표로 평가
5. **Interferent rebinding**: 각 interferent에 대해 동일 test

**분석 지표 (BIO phase5 방식)**:
- **Q4 평형 RMSD**: 마지막 25% 궤적 평균 — 느린 binding/unbinding kinetics에 강건 (단일 final frame 대신)
- **Q1→Q4 수렴 진단**: drift > 1.5Å면 "미수렴" 경고 → rebind 실패가 진짜인지 샘플링 부족인지 판별
- **Residence(잔류시간)**: cavity(≤removal 임계) 내 머문 프레임 비율 — 결국 이탈해도 "얼마나 버텼나"로 약결합 monomer 구별
- **연속 retention 점수**: Q4 RMSD를 [0,1]로 등급화 (retained ≤5Å → 1, escaped ~16Å → 0)
- (옵션) **MM-GBSA ΔG** (rebinding별 결합자유에너지, `VIP_MMPBSA`), **multi-restart 앙상블** (`VIP_N_RESTARTS`회 perturbed 평균)

**VIP Scoring** (연속·잔류 기반):
```
VIP score = ½ × (retention + residence) × (1 + selectivity)
```
- 이진 rebind(0/1)은 약결합 template에서 전부 0이라 순위 불가 → **연속 retention + residence**로 대체 (BIO)
- residence 덕에 다 이탈해도 "오래 버틴 monomer"가 위로 옴
- selectivity: graded — `1 - mean(interf/template Q4-RMSD 비)` (interferent 있을 때)

**재분석 / 연장 (MD 재실행 없이)**:
- `--reanalyze-vip`: 기존 궤적에서 Q4/residence/retention/ΔG 재계산 → 새 지표를 옛 run에 즉시 적용 (수 초)
- `--extend-vip`: 미수렴(drift>1.5Å) rebinding MD만 checkpoint에서 연장 후 재분석

**설계 근거**: 원래 score는 `rebind_rate(0/1)` 기반이었으나 소분자·무극성 template에서는 10ns 내 되붙기(diffusion-limited)가 샘플링 불가 → 전부 rebind=0으로 순위가 안 나옴. Zink 2018은 17β-estradiol(272Å³)을 썼으나 gamma-terpinene 같은 무극성체는 인식이 약해 shape/dispersion 위주라 더 그렇다. 그래서 (1) 샘플링 강화(50ns/10snap), (2) 수렴 진단, (3) 연속 retention+residence로 재설계했다. 무극성 template의 인식은 cavity shape이 주도하는데 파이프라인이 이 축을 못 잡으므로, 최종 IF는 실험으로 검증할 것을 권장한다.

**구현 세부사항**:
- Template/monomer 식별: resid 기반 (첫 번째 non-solvent residue = template)
- acpype가 모든 분자에 "UNL" resname을 부여하므로 resname 대신 resid로 구분

### Stage 7: 합성 레시피 자동 생성

**파일**: `code/pipeline/stage7_recipe.py`, `code/pipeline/chemistry_aware.py`

Stage 6 VIP 순위, Stage 5 합성 비율, Stage 4 MMSD 조합, Stage 3 cross-linker/porogen 추천을 종합하여 합성 프로토콜을 자동 생성한다. 중합 방식(free-radical/sol-gel/oxidative)에 맞는 프로토콜을 출력한다.

**모노머 랭킹 (결합-가중)**: 랭킹 소스는 VIP(6) → MD(5) → Stage 3 순으로 캐스케이드. MD 단계는 **`|DFT 결합에너지| × contact_freq`**로 순위를 매긴다 — contact_freq 단독은 희박한 프레임에서 노이즈가 커서, 실제로 안 붙는데 얼쩡거리기만 하는 비결합체(예: VIM, DFT≈0)가 강결합체(NVP) 위로 올라가는 문제가 있어 DFT 친화도로 가중한다.

**화학 일관성 필터**: best_monomer는 **승리 중합 화학(MMSD 조합)에 맞는 monomer로 필터**된다. 안 그러면 MD contact 랭킹이 피롤(산화중합)을 1위로 올려도 프로토콜은 free-radical+AIBN이라 **화학적으로 불가능한 레시피**가 나온다. 필터로 monomer·프로토콜·가교제가 한 화학으로 일관된다.

**Cross-linker = 가장 inert(약결합)**: 가교제는 ~80% 대량이라 template에 결합하면 매트릭스 전체에 비특이 자리를 만들어 IF를 낮춘다 (Shoravi/Olsson 2014). 따라서 **화학-호환 가교제 중 template에 가장 약하게 결합하는 것**(DFT 스크리닝, 예: vinyl → EGDMA)을 고른다 — MMSD가 total-binding 최적화로 고른 강결합 가교제가 아니라.

**Chemistry-aware 보정** (`chemistry_aware.py`, 문헌 감사 반영 — 검증된 8 gap 중 6개):
- **화학량론 (GAP 4+6)**: 결합에너지 → Kass 추정 → Kass>900 M⁻¹면 1:1 화학량론, 아니면 과잉 monomer. 고정 1:4 대체 (Wulff & Knorr 2001).
- **이온화 (GAP 2)**: 이온화기 검출 + ion-pair microstate. acid–base 쌍이면 "중성 SMILES 랭킹 오류 가능" 경고 (ACS Appl. Polym. Mater. 2020).
- **자기회합 (GAP 3)**: monomer 이합체화 ΔG (xTB) — 유리 monomer↓이나 비특이 자리 억제 (Zhang/Shimizu 2010).
- **Crosslinker 활성종 (GAP 5)**: crosslinker–template 결합이 monomer보다 강하면 경고 (Shoravi/Olsson 2014).
- **Bleeding (GAP 8)**: 제거 난이도 + 강결합 → 잔류 template 유출 위험 + dummy template 권고 (IJMS 2011).

모든 보정은 레시피의 "Design considerations"에 인용과 함께 기재된다.
(Stage 5 MD는 형태학 지표 **SASA/FFV**(GAP 7, RSC 2025)도 산출.)

**출력**:
- `synthesis_recipe.json`: Top 3 monomer + cross-linker + 비율 + stoichiometry/speciation/self_association/bleeding
- 프로토콜 텍스트: 화학별 단계 + Design considerations

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
conda activate MIPscreen              # 또는 사용 중인 env
source /usr/local/gromacs-gpu/bin/GMXRC
cd MIP_simulation

# 전체 파이프라인 (Stage 1→2→3→4→5→6→7)
# 결과는 실행 위치 무관하게 results/<TEMPLATE_NAME>/stage1..7 에 저장.
# 이미 완료된 항목(monomer/solvent/조합)은 item-level로 자동 skip → 중단 후 이어서 진행.
python run_pipeline.py --stage all

# 개별 Stage (1 xTB, 2 DFT, 3 porogen+선택도, 4 MMSD, 5 MD, 6 VIP, 7 recipe)
python run_pipeline.py --stage 5      # MD
python run_pipeline.py --stage 6      # VIP
python run_pipeline.py --stage 7      # 레시피

# ── 다중 Target ──
# config.TEMPLATES 전부 순차 실행 (target별 subprocess 격리 → results/<name>/ 분리)
python run_pipeline.py --all-templates --stage all
# 특정 target만 env로 선택 (config 안 고치고)
MIP_TEMPLATE="Methyl Benzoate" python run_pipeline.py --stage all

# ── 자원 튜닝 (환경변수) ──
# 워커 수는 기본 자동(코어 80%, 가용 RAM 상한). OOM 나면 낮추기:
MIP_WORKERS=2 python run_pipeline.py --all-templates --stage all
#   MIP_WORKERS=<n> 직접 지정 / MIP_RAM_PER_WORKER_GB=2.0 RAM 예산 / MIP_GPU_WORKERS=<n>

# ── VIP 재분석/연장 (MD 재실행 없이) ──
python run_pipeline.py --reanalyze-vip     # 기존 궤적 → Q4/residence/retention/ΔG 재계산
python run_pipeline.py --extend-vip         # 미수렴 rebinding MD만 연장 후 재분석

# ── 강제 재계산 ── (--force는 Stage 4 MMSD만; 다른 stage는 해당 출력 폴더 삭제)
python run_pipeline.py --stage 4 --force

# ── 추가 기능 ──
python run_pipeline.py --crosslinker           # 가교제 스크리닝 (항목별 캐시)
python run_pipeline.py --report                # HTML 리포트
python run_pipeline.py --suggest-interferents  # / --auto-interferents
python run_pipeline.py --predict-if
python run_pipeline.py --update-if-model --monomer MAA --experimental-if 15

# 장시간 실행은 백그라운드 권장 (fresh target의 Stage 1 xTB는 수 시간~하루)
nohup python run_pipeline.py --all-templates --stage all > all.log 2>&1 &
```

**환경변수 요약**: `MIP_TEMPLATE`(활성 target) · `MIP_WORKERS`(CPU 워커, 0/미설정=자동) · `MIP_GPU_WORKERS` · `MIP_RAM_PER_WORKER_GB`(워커당 RAM 예산). 전체 플래그는 루트 `run_pipeline.py` docstring 또는 `--help` 참조.

### 디렉토리 구조

```
MIP_simulation/
├── run_pipeline.py              # 엔트리포인트
├── run_validation.py            # 검증 실행
├── run_selectivity.py           # 선택도 검증 (범함수 비교)
├── environment.yml              # conda 환경 (MIPscreen)
├── code/pipeline/
│   ├── config.py                # 전역 설정
│   ├── run_pipeline.py          # Stage 오케스트레이터
│   ├── stage1_xtb.py            # ESP 도킹 + xTB 스크리닝
│   ├── stage2_dft.py            # DFT 결합에너지 + ESP 시각화
│   ├── stage3_selectivity.py    # 전역 porogen 선택 + 선택도 + cross-linker
│   ├── stage4_mmsd.py           # MMSD 다중 monomer 조합 탐색 (greedy/BO/NSGA-II)
│   ├── stage5_md.py             # GROMACS pre-polymerization MD (MMSD 조합 사용)
│   ├── stage6_vip.py            # VIP cavity rebinding
│   ├── stage7_recipe.py         # 합성 레시피
│   ├── utils_gromacs.py         # GROMACS 유틸리티 (parameterization, MD, 분석)
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
    ├── g_terpinene/stage1~7/    # Gamma-terpinene 결과
    ├── acetic_acid/stage1~7/    # Acetic acid 결과
    ├── methyl_benzoate/stage1~7/# Methyl benzoate 결과
    └── validation/              # 검증 결과
```

---

## 현재 설정 (config.py)

| 항목 | 값 |
|------|-----|
| Template 후보 (`TEMPLATES`) | Gamma-terpinene (`CC1=CCC(=CC1)C(C)C`), Acetic Acid (`CC(=O)O`), Methyl Benzoate (`COC(=O)c1ccccc1`) |
| 활성 Template | `TEMPLATE_NAME` 로 선택 (기본 Gamma-terpinene) |
| Monomers (28) | vinyl 19 (MAA·AA·ITA·4VBA·MAAD·ACM·NIPAM·4VP·2VP·VIM·NVP·Styrene·2VN·4VB·BMA·LMA·tBAm·HEMA·DMAEMA), silane 6 (APTES·MPTMS·PTES·IBTES·UPTMS·CETES), oxidative 3 (PYR·OPD·APB) |
| 중합 방식 | `MMSD_POLYMERIZATION=None` — vinyl/silane/oxidative **경쟁**, 조합은 단일 화학, 이긴 화학 채택 (Stage 7이 화학별 프로토콜 출력) |
| Interferents | 없음 (이번 스크리닝 비활성 — Stage 3는 결합에너지 랭킹으로 대체) |
| Solvents (porogen 후보) | Chloroform (ε=4.71), Acetonitrile (ε=35.69), Toluene (ε=2.38) |
| Porogen 전략 | `global_optimal` — 전역 최적 porogen 1개 자동 선택·고정 |
| Cross-linker | vinyl: EGDMA·DVB·TRIM·BAM / silane: TEOS·TMOS — 레시피는 **화학-호환 중 가장 inert(약결합)** 자동 선택 |
| Workers (CPU) | **자동** — 코어의 80%, 가용 RAM 상한(워커당 ~2GB). `MIP_WORKERS`로 오버라이드 |
| Workers (GPU) | 1 (VRAM 제한). `MIP_GPU_WORKERS`로 오버라이드 |
| VIP 샘플링 | 10 snapshot × 50ns rebinding (`VIP_N_SNAPSHOTS`, `VIP_REBINDING_NS`); 옵션 `VIP_N_RESTARTS`·`VIP_MMPBSA` |

---

## 핵심 파라미터

| Stage | 파라미터 | 값 | 근거 |
|-------|---------|-----|------|
| 1 | ESP 전하 | B3LYP/def2-SVP Mulliken (GPU) | DFT 레벨 정확도, ~7초/분자 |
| 1 | Functional group boost | 5× (N, O, S + neighbors) | 반응 부위 집중 탐색, 균일 샘플링 편향 해소 |
| 1 | Vina exhaustiveness | 64 | MIP 소분자 복합체용 (기본 8의 8배) |
| 1 | 필터 기준 | dE < 0 | 결합하는 monomer만 통과 (개수 자동) |
| 2 | 범함수 | ωB97XD / ωB97M-V (적응형) | H-bond/분산력 시스템별 최적 |
| 2 | 기저함수 | def2-SVP (opt) / def2-TZVP (SP) | 2단계 기저 (속도+정확도) |
| 2 | 용매 모델 | PCM (IEF-PCM) | GPU gradient 지원 (ddCOSMO 불가) |
| 3 | Porogen 전략 | `global_optimal` (top-k 평균 결합 argmin) | 단일 porogen 중합 (van Wissen 2025) |
| 3 | Porogen tie band (τ) | 1.0 kcal/mol → 최저 ε | DFT+PCM 오차 내 H-bond 보존 |
| 3 | Cavity α | 0.10 kcal/(mol·Å³) | vdW 에너지 밀도 기반 물리 상수 |
| 3 | Cavity β | 0.5 | 비선형 cavity filling (표면적 ∝ V^(2/3)) |
| 4 | MMSD optimizer | NSGA-II / Bayesian / greedy | 조합 탐색 (Rajpal 2024) |
| 4 | MMSD 목적함수 | mmsd_per_monomer + 0.3·max(0,Δ) | synergy/interference 크기 정규화 |
| 5 | Force field | GAFF2 (acpype Python API) | 소분자 표준 |
| 5 | 보론 파라미터 | B→C 치환 + 문헌값 | Gerogiokas 2020 |
| 5 | MD 시간 | 50 ns | 평형 도달 |
| 5 | Template:monomer 비율 | 1:4 | 고정 |
| 6 | Snapshot | 균등 간격 10개 | 통계력 (BIO phase5) |
| 6 | Rebinding MD | 50 ns + Q4 평형 RMSD | 느린 kinetics 대응 + Q1→Q4 수렴 진단 |
| 6 | Position restraint | 1000 kJ/mol/nm² | 중합 근사 |
| 6 | Rebinding 기준 | Q4 RMSD < 5 Å | Cavity 인식 성공 |
| 6 | Removal 기준 | RMSD > 8 Å | Template 이탈 (제거 가능) |
| 6 | Scoring | ½(retention + residence) × (1 + selectivity) | 연속·잔류 기반 (약결합도 순위화) |

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
| Global porogen 선택 | numpy + RDKit | van Wissen 2025 [19], Suryana 2021 [20], Liu 2021 [21], Del Sole 2009 [22] | 3 |
| Cavity shape correction | RDKit (ComputeMolVolume) | 본 연구 (vdW 에너지 밀도 기반) | 3 |
| MMSD 조합 탐색 (greedy/BO/NSGA-II) | pymoo + skopt + tblite | Rajpal et al. 2024 [23], Deb 2002 | 4 |
| GAFF2 parameterization | acpype [18] + AmberTools | Wang et al. 2004 | 5 |
| Boron B→C substitution | acpype [18] + custom frcmod | Gerogiokas et al. 2020 [10] | 5 |
| Pre-polymerization MD | GROMACS [13] GPU | Muñoz et al. 2024 [4] | 5 |
| Contact frequency / EBN | MDAnalysis [16] | Ye et al. 2024 [5] | 5 |
| H-bond analysis | MDAnalysis [16] (HydrogenBondAnalysis) | — | 5 |
| VIP cavity rebinding | GROMACS [13] + MDAnalysis [16] | Zink & Moura 2018 [3] | 6 |

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
python -c "import gpu4pyscf; print('gpu4pyscf OK')"
python -c "import MDAnalysis; print('MDAnalysis OK')"
python -c "import openmm; print('OpenMM OK')"
python -c "from vina import Vina; print('Vina OK')"
python -c "import pymoo, skopt; print('MMSD optimizers OK')"   # NSGA-II / Bayesian
gmx --version
```

### 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | CUDA 지원 NVIDIA | RTX 4070 Ti+ (12GB+ VRAM) |
| CPU | 8코어 | 16코어 |
| RAM | 16GB | 32GB |
| GROMACS | 2023+ (GPU build) | 2025.2 (2026.3 호환: tpx→.gro 폴백·maxwarn 자동) |
| Python | 3.11 | 3.11 (tblite 호환) |

---

## 알려진 이슈 및 해결책

| 이슈 | 원인 | 해결 |
|------|------|------|
| meeko `restraints` AttributeError | meeko 0.7+ API 변경 | `prepare()` return value 사용 |
| Vina PDBQT 원자 불일치 | `merge_these_atom_types` 기본값 | REMARK IDX 매핑으로 좌표 복원 |
| acpype "UNL" resname | acpype가 모든 분자에 UNL 부여 | resid 기반 template/monomer 식별 |
| GROMACS `[ atomtypes ]` 순서 에러 | ITP에 atomtypes 포함 시 directive 순서 오류 | `_copy_and_split_itp()`로 분리 |
| ddCOSMO GPU gradient 미지원 | gpu4pyscf 제한 | PCM (IEF-PCM)으로 대체 |
| 보론산 GAFF2 파라미터 부재 | GAFF2에 B 파라미터 없음 | B→C 치환 + 문헌 파라미터 (Gerogiokas 2020) |
| H-bond 분석 0건 | 축약 trajectory + 단방향 분석 | 전체 trajectory + bidirectional + TPR topology |
| VIP score 전부 0 | 약결합 template은 10ns 내 재결합(diffusion-limited) 샘플링 불가 → 이진 rebind 전부 0 | 연속 **retention + residence** 점수로 재설계 + 50ns/10snap + Q4 평형·수렴 진단 (BIO phase5) |
| 실란(Si) monomer MD 불가 | GAFF2에 Si 파라미터 없음 | **PolCA organosilane** force field (Jorge 2021, `_generate_silane_itp`) |
| MDAnalysis "tpx version 138" | GROMACS 2026.3 .tpr가 최신 tpx → 설치된 MDAnalysis 미지원 | `load_universe()`가 .gro topology로 자동 폴백 (버전 무관) |
| grompp "Too many warnings (12)" | GROMACS 2026.3이 atomtype 재정의 경고 다수 | 모든 grompp `-maxwarn 50` (무해 경고 흡수) |
| DFT GPU OOM (큰 분자, TRIM 등) | def2-TZVP가 GPU VRAM 초과 (cudaError) | **CPU로 자동 폴백** (동일 basis/에너지, 느림) |
| acpype 전멸 (antechamber 못 찾음) | launcher가 PATH 벗겨 AmberTools 안 보임 | subprocess에 conda bin PATH 주입(`_tool_env`) + preflight 체크 |
| 전체 target `rc=-9` (SIGKILL) | 시스템 RAM 고갈 (고아 프로세스 / 워커 과다) | 워커 자동 사이징(RAM 상한) + `pkill` 정리, `MIP_WORKERS`로 하향 |

---

## 참고 논문

### 방법론 핵심 논문

| # | 저자 | 저널 | 년도 | DOI | 파이프라인 적용 |
|---|------|------|------|-----|---------------|
| 1 | Mukasa et al. | *Adv. Mater.* | 2023 | 10.1002/adma.202212161 | Stage 3 선택도 공식 S ∝ exp(ΔE/kBT), 다단계 스크리닝 전략 |
| 2 | Singh et al. | *Curr. Anal. Chem.* | 2012 | 10.2174/157341112803216807 | DFT MIP 스크리닝 원형, BSSE 보정, 실험 IF 검증 데이터 |
| 3 | Zink & Moura | *Phys. Chem. Chem. Phys.* | 2018 | 10.1039/c7cp08284c | Stage 6 VIP cavity rebinding (position restraint, template removal, rebinding MD) |
| 4 | Muñoz et al. | *J. Chem. Inf. Model.* | 2024 | 10.1021/acs.jcim.4c00775 | Stage 5 pre-polymerization MD, contact frequency, monomer 선별 |
| 5 | Ye et al. | *Molecules* | 2024 | 10.3390/molecules29174236 | Stage 5 EBN/HBNmax 정량 파라미터, H-bond 점유율 분석 |

### 계산화학 방법론

| # | 저자 | 저널 | 년도 | DOI | 적용 |
|---|------|------|------|-----|------|
| 6 | Bannwarth et al. | *J. Chem. Theory Comput.* | 2019 | 10.1021/acs.jctc.8b01176 | GFN2-xTB (Stage 1 fast screening) |
| 7 | Mardirossian & Head-Gordon | *J. Chem. Phys.* | 2017 | 10.1063/1.4986508 | ωB97M-V 범함수 (분산력 지배 시스템) |
| 8 | Boys & Bernardi | *Mol. Phys.* | 1970 | 10.1080/00268977000101561 | BSSE counterpoise 보정 (gas-phase ghost atom) |
| 9 | Bursch et al. | *Angew. Chem. Int. Ed.* | 2022 | 10.1002/anie.202205735 | DFT best-practice: def2-TZVP 기저, 적응형 범함수 근거 |
| 10 | Gerogiokas et al. | *Molecules* | 2020 | 10.3390/molecules25092196 | 보론산 GAFF2 파라미터화 (B→C 치환 + 문헌 파라미터) |

### Porogen 선택 & MMSD 조합 탐색 (2026 추가)

리서치 워크플로우로 조사·교차검증한 문헌 (14/14 주장 검증 통과).

| # | 저자 | 저널 | 년도 | DOI / URL | 파이프라인 적용 |
|---|------|------|------|-----------|---------------|
| 19 | van Wissen et al. | *Polymers* | 2025 | PMC12030623 | Stage 3 porogen 선택 원리 (proticity 우선, 저유전율 aprotic이 H-bond 보존, dissolve-all 원칙, solvent memory, 단일 formulation 결정) |
| 20 | Suryana et al. | *Molecules* | 2021 | 10.3390/molecules26071891 | DFT+PCM 표준 워크플로우: 최강(가장 음수) 결합 용매 선택 (maximize-interaction 진영) |
| 21 | Liu et al. | *Polymers* | 2021 | 10.3390/polym13162657 | solvation 교란 최소화 기준 (두 진영이 동일 저유전율 답으로 수렴하는 근거) |
| 22 | Del Sole et al. | *Molecules* | 2009 | 10.3390/molecules14072632 | 실험+DFT 직접 증거: nicotinamide/MAA가 chloroform은 선택적, acetonitrile은 결합 無 |
| 23 | (MMSD, Refaat et al.) | *Sci. Rep.* | 2024 | 10.1038/s41598-024-73114-3 | Stage 4 MMSD 조합 탐색 원형 (`Monomer_screening_in_Bio/phase3_mmsd.py`에서 이식) |
| 24 | Rosengren et al. | *Biosens. Bioelectron.* | 2009 | 10.1016/j.bios.2009.06.042 | 유전상수만으로 불충분 — proticity/H-bond 능력은 별개 축 (PCA) |

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

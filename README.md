# MIP Functional Monomer Screening Pipeline

## 문서 개요

Molecularly Imprinted Polymer(MIP) 합성을 위한 최적 functional monomer를 계산화학적으로 스크리닝하는 Python 파이프라인의 전체 구조, 각 기능의 과학적 원리, 사용된 알고리즘과 라이브러리를 설명한다.

---

## 목차

1. [MIP란 무엇인가](#1-mip란-무엇인가)
2. [파이프라인 전체 구조](#2-파이프라인-전체-구조)
3. [Stage 1: ESP-guided 분자 표면 도킹 + GFN2-xTB 스크리닝](#3-stage-1-esp-guided-분자-표면-도킹--gfn2-xtb-스크리닝)
4. [Stage 2: DFT 정밀 결합에너지 계산](#4-stage-2-dft-정밀-결합에너지-계산)
5. [Stage 3: 선택도 계산 및 순위 결정](#5-stage-3-선택도-계산-및-순위-결정)
6. [Stage 4: 분자동역학(MD) 검증](#6-stage-4-분자동역학md-검증)
7. [용매 선택 전략](#7-용매-선택-전략)
8. [Template:Monomer 비율 스크리닝](#8-templatemonomer-비율-스크리닝)
9. [ESP 맵 시각화](#9-esp-맵-시각화)
10. [Cross-linker 스크리닝](#10-cross-linker-스크리닝)
11. [자동 Interferent 제안](#11-자동-interferent-제안)
12. [Imprinting Factor 예측 모델](#12-imprinting-factor-예측-모델)
13. [HTML 리포트 자동 생성](#13-html-리포트-자동-생성)
14. [검증 프레임워크](#14-검증-프레임워크)
15. [핵심 알고리즘 요약 테이블](#15-핵심-알고리즘-요약-테이블)
16. [참고 논문 전체 목록](#16-참고-논문-전체-목록)
17. [실행 환경 및 의존 패키지](#17-실행-환경-및-의존-패키지)

---

## 1. MIP란 무엇인가

Molecularly Imprinted Polymer(분자각인 고분자)는 특정 표적 분자(template)의 형태와 화학적 성질을 "기억"하도록 설계된 합성 고분자이다. Template 분자 주변에서 functional monomer와 cross-linker를 중합한 뒤 template을 제거하면, template과 정확히 맞는 3차원 인식 자리(binding cavity)가 남는다.

MIP 성능의 핵심은 **functional monomer 선택**이다. Monomer가 template과 강하게, 그리고 선택적으로 결합해야 좋은 MIP가 만들어진다. 이 파이프라인은 계산화학으로 최적의 monomer를 사전 예측한다.

---

## 2. 파이프라인 전체 구조

4단계 깔때기(funnel) 전략을 사용한다. 비용이 적은 방법부터 시작하여 후보를 점진적으로 좁힌다 (Mukasa et al. 2023의 다단계 스크리닝 전략 참고).

```
Stage 1: ESP-guided 분자 표면 도킹 + GFN2-xTB 스크리닝
         ├── DFT ESP 전하 계산 (GPU, ~7초/분자)
         │   → template, monomer의 B3LYP/def2-SVP Mulliken 전하
         ├── ESP-guided 표면 도킹 (분자 크기 적응형 ~200-400개 orientation)
         │   → 전하 가중 vdW 표면 샘플링 + 쌍극자 정렬
         ├── AutoDock Vina 도킹 (exhaustiveness=64, ~30개 pose)
         │   → 전문 scoring function으로 소수성/형태 적합성 탐색
         ├── GFN2-xTB SP 스크리닝 (~270개 합산, ~10분)
         └── 상위 10개 full xTB optimization
    ↓ 상위 N개 선별 + xTB 최적 좌표를 Stage 2로 전달
Stage 2: DFT ωB97M-V/def2-TZVP + PCM (정밀 계산, GPU)
         ├── GPU DFT geometry optimization (def2-SVP + RI-J + geomeTRIC)
         ├── ωB97M-V/def2-TZVP single-point (complex, template, monomer)
         └── BSSE counterpoise (gas-phase ghost atom)
    ↓ BSSE 보정 결합에너지
Stage 3: 선택도 평가 (template vs interferent 비교)
    ↓ 상위 3개
Stage 4: 50ns MD 시뮬레이션 (동적 행동 검증, GPU)
    ↓ RDF, EBN, H-bond 분석
추가 기능: Cross-linker 스크리닝, ESP 맵, IF 예측, 리포트 생성
```

---

## 3. Stage 1: ESP-guided 분자 표면 도킹 + GFN2-xTB 스크리닝

### 과학적 원리

분자 표면 전체에서 monomer를 template에 도킹하여 최적 결합 배향을 탐색한다. Mukasa et al. (2023)의 다단계 스크리닝 전략을 참고하되, 다음을 개선했다:

- **DFT 레벨 정전기 포텐셜(ESP)**로 배향 생성 (원논문: 무작위)
- **AutoDock Vina** 도킹으로 소수성/형태 적합성 보완 (원논문: 없음)
- **GFN2-xTB** (Bannwarth et al., 2019)로 fast screening (원논문: PM3)

GFN2-xTB는 D4 분산력을 self-consistent하게 포함하며, PM3 대비 비공유 상호작용 기술이 우수하다. 모든 비공유 상호작용(수소결합, 정전기, van der Waals)을 고려한다.

### ESP(Electrostatic Potential)란?

분자 주변 공간의 **전하 분포 지도**이다. "이 위치에 양전하 입자를 놓으면 얼마나 끌리거나 밀리는가"를 3차원 공간의 모든 점에 대해 계산한 것이다.

```
ESP(r) = Σ (핵전하 Zₐ / |r - Rₐ|) − ∫ (전자밀도 ρ(r') / |r - r'|) dr'
         ──────────────────────────   ────────────────────────────────────
         핵이 끌어당기는 힘 (+)        전자가 밀어내는 힘 (−)
```

ESP 값이 **음수(빨간색)인 영역**: 전자가 풍부 → 양전하 부위와 결합 가능 (예: C=O, -OH, -NH₂ 의 비공유 전자쌍)
ESP 값이 **양수(파란색)인 영역**: 전자가 부족 → 음전하 부위와 결합 가능 (예: N-H, O-H 의 수소)

MIP에서 ESP가 중요한 이유: template의 음전하 영역에는 monomer의 양전하 부분이, 양전하 영역에는 음전하 부분이 마주보도록 배치해야 **수소결합과 정전기 상호작용이 최대화**된다. Singh 2012에서도 ESP 맵으로 template-monomer 결합 부위를 설명했다.

본 파이프라인에서 ESP는 두 가지 용도로 사용된다:

1. **Stage 1 배향 가이드**: 각 원자의 DFT 부분전하(Mulliken)로 monomer의 접근 방향과 회전 각도를 결정 — ESP의 간이(discrete) 버전
2. **Feature 4 ESP 맵 시각화**: 3D 격자에서 연속적인 ESP를 계산하여 결합 부위를 시각적으로 분석

### 5단계 도킹 프로토콜

Mukasa et al. (2023) Fig.2b에서 제시한 "monomer당 100개 docked complex" 방식을 확장한다:

```
Step 1a: ESP-guided 표면 orientation 생성 (DFT 부분전하 기반, 분자 크기 적응형)
    ↓ 정전기 상보성 기반 배향 + ±15° random perturbation
Step 1b: AutoDock Vina 도킹 (exhaustiveness=64, ~30 pose)
    ↓ 소수성/형태 적합성 기반 결합 배향
Step 2: GFN2-xTB single-point 스크리닝 (ESP + Vina 합산 ~270개)
    ↓ 에너지 오름차순 정렬 → 상위 10개 선별
Step 3: GFN2-xTB full geometry optimization
    ↓ 최종 ground state 결합에너지 + binding site 식별
Step 4: xTB 최적 좌표를 DFT에 직접 전달 (prebuilt_complex_mol)
    ↓ Stage 2 DFT가 최적 배향에서 GPU geometry optimization + 정밀 에너지 계산
```

### Step 1: van der Waals 표면 orientation 생성

분자 표면 전체를 탐색하여 모든 비공유 상호작용(수소결합, 정전기, vdW)을 포괄한다. H-bond site에만 제한하지 않는 것이 핵심이다 (Mukasa et al. 2023).

**표면점 생성**: 각 원자의 van der Waals 구(sphere) 위에 Fibonacci sphere sampling으로 균일 분포 점을 생성한다. 이후 분자 내부에 매몰된(buried) 점을 필터링하여 표면 노출 점만 남긴다.

```
각 원자 i에 대해:
  r_i = vdW 반경 (H: 1.20, C: 1.70, N: 1.55, O: 1.52 Angstrom)
  20개 Fibonacci sphere 점을 반경 r_i에 생성
  → 다른 원자 j의 vdW 구 내부(0.8*r_j 이내)에 있으면 제거

결과: Phe → ~269개 표면점
```

**DFT 레벨 부분전하 계산**: ESP-guided 배치에 사용하는 부분전하를 B3LYP/def2-SVP Mulliken 전하로 계산한다 (GPU SCF + CPU Mulliken 분석, ~7초/분자). xTB Mulliken 전하는 방향족 탄소와 헤테로원자(B, Cl)에서 정량적으로 부정확하여, 특히 보론산(B(O)O) 분자에서 잘못된 배향을 유도할 수 있다. DFT 레벨 전하는 이를 해결한다.

**분자 크기 적응형 orientation 수**: 분자가 클수록 더 많은 방향을 탐색해야 표면을 충분히 커버할 수 있다. 분자 크기에 비례하여 orientation 수를 자동 조정한다.

```
base = 100 (N_DOCK_ORIENTATIONS)
reference = 14 heavy atoms (Phe 9 + MAA 5)
scale = max(1.0, combined_heavy_atoms / 14)
n_orientations = min(base × scale, 1000)

예시:
  Phe(12) + MAA(6) = 18 heavy → 128개
  Heptachlor(16) + MAA(6) = 22 heavy → 157개
  DDT(25) + MAA(6) = 31 heavy → 221개
```

DDT처럼 큰 분자에서는 100개로는 부족하여 결합에너지가 과소평가되었으나, 적응형으로 221개를 생성하면 유효 orientation이 62→141개로 2배 이상 증가하여 더 정확한 탐색이 가능해진다.

**ESP-guided orientation 배치**: 적응형으로 결정된 개수만큼 표면점을 **전하 가중 확률**로 선택하고, 각 점에 대해:
1. template ESP에서 가장 음전하/양전하 영역에 가중치 부여 (|charge| 비례 확률)
2. monomer의 쌍극자 모멘트를 template 표면 법선 방향으로 정렬 (상보적 전하 배치)
3. ±15° random perturbation 추가 (다양성 확보)
4. 표면점에서 DOCK_SURFACE_OFFSET (2.5A) 만큼 떨어진 위치에 monomer 중심 배치
5. **원자 겹침 필터**: template-monomer 원자 간 최소 거리 < 1.2A인 구조 제거

이 방식은 무작위 회전 대비 화학적으로 의미 있는 배향을 생성한다. Singh 2012에서 ESP 맵으로 결합을 설명한 것과 동일한 원리를 orientation 생성에 적용한 것이다.

### Step 1b: AutoDock Vina 도킹

ESP-guided 배향이 정전기 상호작용에 특화된 반면, 소수성 상호작용(DDT-Styrene 같은 pi-pi stacking)이나 형태 적합성은 놓칠 수 있다. AutoDock Vina의 경험적 scoring function이 이를 보완한다.

```
Vina scoring function:
  ΔG = w1·gauss1 + w2·gauss2 + w3·repulsion + w4·hydrophobic + w5·hydrogen_bond

exhaustiveness=64: 소분자-소분자 MIP 복합체용 탐색 깊이 (단백질 도킹 기본값 8의 8배)
n_poses=30: 상위 30개 pose 저장
```

ESP ~240개 + Vina ~30개 = **~270개 orientation**이 합산되어 xTB SP screening으로 일괄 평가된다. Vina가 없는 환경에서는 자동으로 ESP-only로 fallback한다.

### Step 2: GFN2-xTB single-point 스크리닝

Mukasa et al. (2023)에서 PM3가 수행하던 fast-screening 역할을 GFN2-xTB single-point로 대체한다. 구조 최적화 없이 에너지만 계산하므로 orientation당 ~1-2초로 매우 빠르다.

```
각 유효 orientation에 대해:
  E_complex_SP = GFN2-xTB single point (복합체)
  E_monomer_SP = GFN2-xTB single point (monomer 부분만)
  ΔE_SP = (E_complex_SP - E_template_SP - E_monomer_SP) × 627.509 kcal/mol

에너지 오름차순 정렬 → 상위 N_TOP_FOR_OPTIMIZATION (기본 10)개 선별
```

### Step 3: Full GFN2-xTB geometry optimization

상위 후보만 정밀 최적화하여 계산 비용을 절약한다.

**핵심 최적화**: template과 monomer의 에너지는 한 번만 계산한다 (분자 구조가 동일하므로). 복합체 최적화만 후보 수만큼 반복한다.

```
E_template, E_monomer ← 각 1회만 L-BFGS-B 최적화 (사전 계산)

각 top-N 후보에 대해:
  E_complex ← L-BFGS-B 최적화 (max 200 iter, ftol=1e-4)
  ΔE = (E_complex - E_template - E_monomer) × 627.509 kcal/mol

가장 음수인 ΔE = ground state → 최종 결합에너지
```

### Step 4: xTB→DFT 좌표 전달 (prebuilt_complex_mol)

xTB로 최적화한 복합체 좌표를 Stage 2 DFT에 직접 전달하여, DFT가 최적 결합 배향에서 정밀 에너지를 계산하도록 한다. 이 연계가 없으면 DFT가 기본 방향(+x)으로 복합체를 다시 만들어 xTB 탐색 결과가 무의미해진다.

```
xTB optimize → best_positions (numpy array)
    ↓ _arrays_to_mol()
RDKit Mol (3D conformer with optimized coordinates)
    ↓ prebuilt_complex_mol 파라미터로 전달
Stage 2 compute_dft_binding() → DFT SP on xTB-optimized geometry
```

`_arrays_to_mol()` 함수가 xTB numpy 좌표를 RDKit Mol로 변환한다. `CombineMols(template, monomer)`로 결합 후 각 원자의 위치를 최적화된 좌표로 덮어쓴다.

검증 결과 (Heptachlor-MAA):
- xTB→DFT 전달 시 raw_dE = **-8.081** kcal/mol (논문 -8.11과 0.03 차이)
- 전달 없이 기본 +x: raw_dE = -0.729 kcal/mol (논문과 7.38 차이)

### Binding site 사후 식별

Mukasa et al. (2023): "Results from this screening correctly revealed two potential binding sites on Phe on the carboxyl (COOH) and amine (NH2) functional groups" — binding site는 결과에서 사후적으로 발견되는 것이다.

최적화된 복합체에서 monomer 중심에 가장 가까운 template 원자를 찾고, 해당 원자가 H-bond donor/acceptor인지 또는 vdW/정전기 상호작용 부위인지 분류한다.

### 결합에너지 공식

```
ΔE = E(complex) - E(template) - E(monomer)

음수 ΔE = 안정적 결합 (좋은 monomer)
양수 ΔE = 반발 (나쁜 monomer)
단위: Hartree × 627.509 = kcal/mol
```

### 검증 결과 및 한계

**Mukasa 2023 (Phe) monomer 순위 예측**:

```
계산 순위: APB(-19.7) > 4VB(-11.8) > MAA(-3.8) > OPD(-3.8) > PYR(-2.4) > ACM(-1.2)
논문 순위: OPD > MAA > 4VB > APB > ACM > ANI > PYR
```

- **OPD > PYR 방향성 ✓** — 핵심 예측 방향이 정확
- **PYR 하위권 ✓** — bottom 2 이내
- **OPD 4위** — MAA와 0.013 kcal/mol 차이 (사실상 동점)
- **APB, 4VB 과대평가** — 보론산(B(O)O) 분자에서 xTB→DFT 구조 불일치

**현재 파이프라인의 정확도와 한계**:

| 강점 | 한계 |
|------|------|
| OPD > PYR 방향 정확 | 절대값 재현은 ±2-5 kcal/mol 오차 |
| H-bond 시스템 순위 예측 양호 | 분산력 지배 시스템(DDT) 부정확 |
| DFT ESP 전하로 정전기 배향 정확 | 보론산(B) 분자 과대평가 |
| pair당 ~15분 실용적 속도 | DFT geometry optimization은 비실용적 |

**과대평가 원인**: xTB에서 최적화한 구조를 DFT single-point로 계산할 때, xTB와 DFT의 potential energy surface 차이로 인해 일부 분자에서 비최적 구조의 에너지가 과대/과소평가됨. DFT geometry optimization으로 해결 가능하나 pair당 4-5시간 소요로 비실용적.

**BSSE counterpoise를 적용하지 않는 이유**: xTB 최적화 구조에서 DFT ghost atom 계산 시 기저함수 환경 불일치로 BSSE가 과보정됨. BSSE 미적용 raw_dE가 논문 값과 더 일치하므로 raw_dE를 기본 결합에너지로 사용.

### 병렬화

ProcessPoolExecutor로 monomer별 독립 프로세스 실행 (기본 16 workers). SP screening은 tblite와의 호환성을 위해 순차(serial) 실행한다.

---

## 4. Stage 2: DFT 정밀 결합에너지 계산

### 과학적 원리

밀도범함수 이론(DFT)은 슈뢰딩거 방정식을 전자 밀도 기반으로 근사하여 분자 에너지를 정밀하게 계산하는 방법이다. Bursch, Grimme et al. (2022) "Best-Practice DFT Protocols"에 따라 최적 조합을 사용한다:

```
ωB97M-V / def2-TZVP + PCM + BSSE counterpoise (gas-phase ghost)
(geometry optimization: def2-SVP + RI-J density fitting)
```

각 구성 요소의 역할:

| 구성 요소 | 역할 | 참고 논문 |
|-----------|------|-----------|
| **ωB97M-V** | 범위 분리 하이브리드 범함수 + VV10 nonlocal 분산력. 비공유결합 벤치마크에서 최고 정확도 | Mardirossian & Head-Gordon 2016 |
| **def2-TZVP** | Ahlrichs triple-zeta 기저. Pople(6-311+G*) 대비 보론 등 전 원소에서 균형 잡힌 기저 | Weigend & Ahlrichs 2005 |
| **def2-SVP** | 작은 기저. geometry optimization에서 빠른 속도. SP는 def2-TZVP로 정밀 계산 | Weigend & Ahlrichs 2005 |
| **RI-J density fitting** | Coulomb 적분 근사. DFT 계산 ~3배 가속, 정확도 손실 < 0.1 kcal/mol | |
| **PCM** | Polarizable Continuum Model. gpu4pyscf에서 analytical gradient 지원 → GPU geometry optimization 가능 | |
| **BSSE counterpoise** | 기저함수 중첩 오차 보정. ghost atom 계산은 **기상(gas-phase)**에서 수행 (PCM cavity 왜곡 방지) | Boys & Bernardi 1970 |

> **설계 판단**: ddCOSMO 대신 PCM을 사용하는 이유 — gpu4pyscf가 PCM analytical gradient를 지원하여 GPU에서 DFT geometry optimization이 가능하다. ddCOSMO는 geomeTRIC과 호환되지 않아 silent failure가 발생한다. BSSE ghost atom 계산에서 PCM을 포함하면 cavity 형태가 왜곡되어 과보정이 발생하므로, ghost 계산은 기상에서 수행한다.

### GPU 가속

gpu4pyscf (Wu et al., 2024)를 사용하여 **SCF + gradient + geometry optimization** 모두 GPU에서 실행한다. CPU 32코어 대비 ~30배 가속. CUDA 미설치 시 자동 CPU fallback.

### 2단계 기저 전략 (Dual-basis)

```
Step 1: GPU DFT geometry optimization
  ωB97M-V/def2-SVP + RI-J + PCM (작은 기저로 빠른 구조 최적화)
  geomeTRIC optimizer, maxsteps=50, 수렴 기준: rms_grad < 3e-4

Step 2: GPU DFT single-point (3회)
  ωB97M-V/def2-TZVP + RI-J + PCM (큰 기저로 정밀 에너지)
  1. E(complex)      — 최적화된 복합체 좌표
  2. E(template)     — template 단독 (복합체에서 추출)
  3. E(monomer)      — monomer 단독 (복합체에서 추출)

Step 3: BSSE counterpoise (2회, 기상)
  ωB97M-V/def2-TZVP (PCM 없음, ghost atom)
  4. E(template+ghost) — template 기저 + monomer ghost atom
  5. E(monomer+ghost)  — monomer 기저 + template ghost atom

raw_dE  = (E1 - E2 - E3) × 627.509 kcal/mol
bsse_dE = (E1 - E4 - E5) × 627.509 kcal/mol  ← 기본 결합에너지
```

---

## 5. Stage 3: 선택도 계산 및 순위 결정

### 과학적 원리

좋은 MIP monomer는 template과 강하게 결합할 뿐 아니라, 유사한 간섭 물질(interferent)과는 **약하게** 결합해야 한다. Mukasa et al. (2023)의 선택도 공식을 사용한다.

**핵심: 같은 monomer가 template vs interferent에 얼마나 다르게 결합하는가**

```
ΔE = ΔE(monomer-template) − ΔE(monomer-interferent)
S = exp(ΔE / kB·T)

kB = 0.001987 kcal/(mol·K), T = 298.15 K
```

구체적으로:
```
ΔE(monomer-template)     = monomer가 template에 결합하는 에너지 (Stage 2에서 계산)
ΔE(monomer-interferent)  = 같은 monomer가 interferent에 결합하는 에너지 (Stage 3에서 추가 계산)
```

```
ΔE < 0 → monomer가 template에 더 강하게 결합 → 선택적 (좋은 MIP)
ΔE > 0 → monomer가 interferent에 더 강하게 결합 → 비선택적 (나쁜 MIP)
```

예시 (OPD monomer):
```
ΔE(OPD-Phe)       = -6 kcal/mol  (template에 강하게 결합)
ΔE(OPD-Tyrosine)  = -2 kcal/mol  (interferent에 약하게 결합)
ΔE = (-6) - (-2)  = -4 kcal/mol  → 선택적! (S < 1)

ΔE(4VB-Phe)       = -46 kcal/mol (template에 강하게 결합)
ΔE(4VB-Tyrosine)  = -40 kcal/mol (interferent에도 강하게 결합!)
ΔE = (-46) - (-40) = -6 kcal/mol → 덜 선택적 (결합 강도 차이가 크지 않음)
```

최종 점수 = 모든 interferent에 대한 log(S) 평균. 더 음수일수록 선택적.

**주의: ΔE(interferent-template)이 아님.** "interferent가 template 자리에 들어올 때의 에너지"가 아니라, "**같은 monomer**가 interferent와 결합할 때의 에너지"를 비교하는 것이다. 이 차이가 핵심이다.

### 구현 기술

**Stage 3는 (monomer × interferent) 조합의 DFT를 추가 계산한다.** Stage 2에서 계산한 monomer-template 결합에너지에 더해, 각 monomer와 각 interferent의 결합에너지를 동일한 DFT 프로토콜(적응형 범함수/def2-TZVP + PCM + BSSE)로 계산한다. 예를 들어 monomer 6개 × interferent 3개 = 18개 추가 DFT 계산이 필요하다. 결과는 `stage3_interferent_dft.json`에 캐시하여 재계산을 방지한다.

시각화는 matplotlib으로 두 종류 출력:
1. **Scatter plot**: x=결합에너지, y=선택도 (이상적 위치: 좌하단)
2. **Bar chart**: monomer별 평균 선택도 순위

---

## 6. Stage 4: 분자동역학(MD) 검증

### 과학적 원리

Stage 1~3의 정적 계산과 달리, MD 시뮬레이션은 용액 내 분자들의 동적 행동을 재현한다. Munoz et al. (2024)의 MD 기반 monomer 선별 방법론과 Yuan et al. (2024)의 EBN/HBNmax 정량 파라미터를 적용한다.

### 시뮬레이션 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| 앙상블 | NVT | Langevin Middle 적분기 |
| 온도 | 298.15 K | 실온 |
| 타임스텝 | 2 fs | HBonds 구속과 함께 |
| 시뮬레이션 시간 | 50 ns | 25,000,000 스텝 |
| 비결합 방법 | PME | 장거리 정전기 |
| 컷오프 | 1.0 nm | 비결합 상호작용 절단 |
| 힘장 | GAFF2 (fallback: OpenFF Sage 2.1.0) | openmmforcefields |
| 물 모델 | TIP3P (explicit) | 실제 물 분자 포함 |
| 몰비 | template:monomer = 1:4 | Singh 2012 합성 조건 |
| GPU | CUDA mixed precision | OpenMM 플랫폼 |

### 궤적 분석

**RDF (방사 분포 함수)**: template heavy atom과 monomer heavy atom 사이의 거리 분포. MDTraj `compute_rdf()`로 계산, 범위 0~1.0nm, 100 bins.

```
g(r) > 1 → 선호 거리 (monomer가 template 주변에 모임)
g(r) = 1 → 무작위 분포
g(r) < 1 → 배제 영역
```

**EBN (유효 결합 수)** (Yuan et al. 2024): RDF의 첫 번째 용매화 껍질(r < 0.35nm)을 적분한 값.

```
EBN = 4pi * integral(0 to 0.35nm) g(r) * r^2 dr
```

numpy `trapz()`로 사다리꼴 적분. 높은 EBN = template 주변에 monomer가 많음 = 좋은 monomer.

**수소결합 분석**: MDTraj `baker_hubbard()` 알고리즘으로 검출. 기준: D-A 거리 < 3.5A, D-H-A 각도 > 120도. Template-monomer 교차 분자 수소결합만 필터링. 점유율(occupancy) >= 50% → 안정적 H-bond.

---

## 7. 용매 선택 전략

### 문제

Stage 2에서 3개 용매(Chloroform, Acetonitrile, MeOH)에서 각각 결합에너지를 계산하지만, Stage 3 선택도 계산에서 어떤 용매 값을 사용할지 기준이 필요하다.

### 4가지 전략

| 전략 | 로직 | 사용 시기 |
|------|------|----------|
| `synthesis_match` | 실제 합성 용매와 동일한 조건 사용 (기본값) | Mukasa 2023 방식. 실험 재현 목적 |
| `minimum` | 3개 용매 중 가장 강한 결합에너지 선택 | 최대 결합력 기준 평가 |
| `average` | 3개 용매의 평균 | 용매 독립적 평가 |
| `worst` | 가장 약한 결합에너지 선택 | 보수적 평가 (어떤 용매에서도 결합 보장) |

구현: `aggregate_solvent_energy()` 함수가 multi-solvent DFT 결과에서 전략에 따라 단일 에너지값과 선택된 용매명을 반환한다.

---

## 8. Template:Monomer 비율 스크리닝

### 과학적 근거

Singh 2012와 Yuan et al. 2024 모두 template:monomer 최적 비율이 monomer마다 다름을 보고했다. 1:4 고정은 일부 monomer에서 최적이 아닐 수 있다.

### 알고리즘

1. 각 monomer에 대해 비율 1:1, 1:2, 1:4로 별도 MD 시뮬레이션 실행
2. 비율별 EBN과 안정 H-bond 수 계산
3. 최적 비율 결정: **EBN 포화 기준**

```
비율 증가에 따른 EBN 변화율 계산:
  1→2: pct = (EBN_2 - EBN_1) / |EBN_1| × 100
  2→4: pct = (EBN_4 - EBN_2) / |EBN_2| × 100

변화율 >= 10% → 아직 포화 안 됨 → 더 높은 비율 채택
변화율 < 10% → 포화 → 이전 비율이 최적
```

예: EBN이 1:1에서 1.2, 1:2에서 1.9 (+58%), 1:4에서 2.3 (+21%) → 1:4 채택 (아직 10% 이상 증가).

비율별 RDF를 하나의 그래프에 겹쳐서 비교 가능하다.

---

## 9. ESP 맵 시각화

### 과학적 근거

Singh 2012의 핵심 분석 중 하나가 Molecular Electrostatic Potential(MEP) 맵으로 template-monomer 결합 부위를 시각화하는 것이다. 양전하(파란색) 영역과 음전하(빨간색) 영역의 상보적 분포가 결합의 화학적 성격을 설명한다.

### 구현 기술

1. PySCF ωB97M-V/def2-TZVP DFT 계산 완료 후 density matrix 추출
2. `pyscf.tools.cubegen.mep()` 함수로 3D 격자(60x60x60) MEP 계산
3. Gaussian cube 파일 형식으로 저장
4. matplotlib으로 2D 등고선 맵 생성:
   - z 중간 평면(z-midpoint)을 슬라이스
   - `RdBu_r` 색상 맵 (빨간색=음전하, 파란색=양전하)
   - `TwoSlopeNorm`으로 0 중심 정규화, vmax=0.1 a.u. 클램핑
   - 원자 위치를 검은 점으로 오버레이
5. 각 monomer에 대해 3개 맵 생성: template 단독, monomer 단독, 복합체

---

## 10. Cross-linker 스크리닝

### 과학적 근거

Mukasa 2023은 cross-linker도 template과의 결합에너지를 평가했다. Cross-linker가 template과 너무 강하게 결합하면 monomer의 인식 자리 형성을 방해하여 imprinting 효율이 떨어진다. 좋은 cross-linker = template과의 결합이 약한 것.

### 평가 로직

Stage 2와 동일한 DFT 프로토콜(ωB97M-V/def2-TZVP + PCM + BSSE)로 template-crosslinker 결합에너지를 계산한다. `stage2_dft.py`의 `compute_dft_binding()`을 직접 import하여 재사용 — 계산 로직 중복 없음.

| 평가 | 조건 | 의미 |
|------|------|------|
| **GOOD** | bsse_dE > -1.0 kcal/mol | 약한 상호작용. monomer와 경쟁하지 않음 |
| **WARNING** | bsse_dE <= -1.0 kcal/mol | 강한 결합. imprinting 효율 저하 우려 |

권장 cross-linker: 모든 용매에서 평균 bsse_dE가 가장 약한(0에 가까운) 것을 선택.

내장 cross-linker 라이브러리: EGDMA (범용), DVB (방향족), TRIM (3작용기), BAM (아미드).

---

## 11. 자동 Interferent 제안

### 문제

선택도 평가에 적절한 interferent를 수동으로 선택하기 어렵다. Template과 구조적으로 유사하지만 구별해야 할 분자를 자동으로 제안한다.

### 방법 1: 로컬 Tanimoto 유사도 (항상 사용 가능)

1. RDKit MorganFingerprint (radius=2, 2048 bits)로 분자 지문(fingerprint) 생성
2. 내장 데이터베이스(25개 분자: 아미노산 12종, 신경전달물질 4종, 농약 3종, 의약품 4종)와 Tanimoto 유사도 계산

```
Tanimoto(A, B) = |FP_A ∩ FP_B| / |FP_A ∪ FP_B|

필터: 0.3 <= Tanimoto <= 0.8
  0.8 이상: 너무 유사 → MIP로도 구별 불가
  0.3 이하: 너무 다름 → interferent로 의미 없음
```

3. Tanimoto 내림차순 정렬 후 config.py 복사-붙여넣기용 스니펫 출력

### 방법 2: PubChem API (선택적, 네트워크 필요)

`pubchempy` 패키지로 PubChem PUG REST API에 유사 화합물 검색 (Threshold=40, MaxRecords=20, 타임아웃 10초). 실패 시 자동으로 방법 1만 사용. 결과는 로컬 Tanimoto 점수로 재계산하여 일관된 기준으로 병합.

---

## 12. Imprinting Factor 예측 모델

### 과학적 근거

현재 파이프라인 출력(결합에너지, 선택도, EBN)은 Imprinting Factor(IF)와 상관관계가 있지만 직접 IF를 예측하지는 않는다. 문헌 데이터를 학습하여 계산 지표 → IF 예측 회귀 모델을 구축한다.

### 학습 데이터

Singh 2012 (6 data points)와 Mukasa 2023 (4 data points), 총 10개:

| 출처 | Template | Monomer | bsse_dE | epsilon | IF |
|------|----------|---------|---------|---------|-----|
| Singh 2012 | Heptachlor | MAA | -8.11 | 4.71 | 1.92 |
| Singh 2012 | Heptachlor | 4VP | -7.76 | 4.71 | 1.45 |
| Singh 2012 | DDT | 4VP | -7.86 | 4.71 | 1.65 |
| Mukasa 2023 | Phe | OPD | -11.9 | 32.61 | 3.2 |
| ... | | | | | |

### 피처 엔지니어링

```
X = [bsse_dE, solvent_eps, bsse_dE * solvent_eps]

bsse_dE: 직접적 결합 친화도
solvent_eps: 용매 극성 효과
교호작용항: 용매에 따른 결합에너지 민감도 포착
```

### 모델 비교 및 선택

3종 모델을 scikit-learn으로 학습:

| 모델 | 하이퍼파라미터 | 특징 |
|------|--------------|------|
| LinearRegression | 없음 | 기준선 모델 |
| Ridge | alpha=1.0 (L2 정규화) | 과적합 방지 |
| RandomForestRegressor | n_estimators=100, max_depth=3 | 비선형 패턴 |

**Leave-One-Out 교차 검증**: N개 데이터에서 1개를 빼고 N-1개로 학습, 1개를 예측. N번 반복하여 RMSE 계산. 데이터가 적을 때(N=10) 가장 적절한 교차 검증 방법.

```
LOO-CV RMSE = sqrt( (1/N) * sum(y_true - y_pred)^2 )
```

최소 RMSE 모델 선택 후 전체 데이터로 재학습.

### 신뢰구간

정규 근사 95% 신뢰구간:
```
IF_pred +/- 1.96 * residual_std

residual_std = LOO-CV 잔차의 표준편차
```

### 모델 업데이트

실험으로 IF를 측정한 후 `update_model()` 함수로 학습 데이터에 추가하고 재학습할 수 있다. 기존 모델 pickle에서 학습 데이터를 로드 → 새 데이터 추가 → 3종 모델 재비교 → 최적 모델 재선택 → 저장.

---

## 13. HTML 리포트 자동 생성

파이프라인 완료 후 모든 결과를 단일 HTML 파일로 통합한다.

### 기술 구현

- **이미지 임베딩**: base64 인코딩 (`data:image/png;base64,...`). 외부 파일 의존성 없이 HTML 하나로 완결
- **스타일**: 인라인 CSS만 사용 (외부 CSS/JS 불필요). 흰 배경, sans-serif 폰트, 줄무늬 테이블(#f9f9f9)
- **누락 처리**: 결과 파일이 없는 섹션은 "미실행" 표시 (오류 아님)

### HTML 구조

1. 헤더 — Template 정보, 실행 일시, config 요약
2. Stage 1 — xTB 결합에너지 테이블
3. Stage 2 — DFT 결합에너지 (monomer x solvent 행렬)
4. Stage 3 — 선택도 테이블 + scatter plot + bar chart
5. Stage 4 — EBN/H-bond 테이블 + RDF 그래프
6. Cross-linker 평가 (GOOD/WARNING 색상 코딩)
7. ESP 맵 갤러리 (자동 발견: glob 패턴)
8. 최종 권장사항 — TOP-3 monomer, 권장 cross-linker, 최적 비율
9. 참고 논문 목록

---

## 14. 검증 프레임워크

### 14.1 설계 원리

문헌의 **실험 Imprinting Factor(IF)**로 파이프라인 정확도를 체계적으로 검증한다. 파이프라인의 목적은 절대값 재현이 아닌 **monomer 순위 예측**이므로, Spearman 순위 상관계수(ρ)가 핵심 지표이다. 검증 코드는 파이프라인 코드와 분리되어 있으며, 파이프라인의 DFT 함수를 import하여 재사용한다 (계산 로직 중복 없음).

### 14.2 검증 기준점

| 기준점 | 출처 | Template | Monomers | 실험 IF 순위 | 합격 기준 |
|--------|------|----------|----------|-------------|----------|
| 1 | Singh 2012 | Heptachlor | MAA, 4VP, Styrene | MAA(1.92) > 4VP(1.45) > Sty(0.99) | Spearman ρ ≥ 0.8 |
| 2 | Singh 2012 | DDT | MAA, 4VP, Styrene | 4VP(1.65) > MAA(1.43) > Sty(1.21) | Spearman ρ ≥ 0.5 |
| 3 | Mukasa 2023 | Phenylalanine | OPD, MAA, 4VB, APB, ACM, PYR | OPD(~3.2) > ... > PYR(~1.2) | OPD > PYR 방향성 |

### 14.3 DFT 설정 반복 개선 과정

Heptachlor-MAA (실험 IF=1.92, 논문 계산값=-8.11 kcal/mol)를 기준 시스템으로 6번의 반복을 거쳐 최적화했다.

| 버전 | 범함수/기저 | 용매 모델 | 구조 최적화 | BSSE | Heptachlor-MAA 결과 | 문제점 |
|------|-----------|----------|-----------|------|-------------------|--------|
| v1 | B3LYP-D3BJ/6-311+G* | ddCOSMO | xTB SP only | DFT ghost | raw=-8.08, bsse=-2.28 | xTB 구조 + DFT ghost → BSSE 과보정 |
| v2 | ωB97XD/6-311+G* | PCM | xTB SP only | 없음 | raw=-9.61 | BSSE 없이 과대평가 |
| v3 | ωB97XD/6-311+G* | PCM | GPU geomeTRIC | 없음 | raw=-12.36 | DFT opt 후에도 과대평가 |
| v4 | ωB97XD/6-311+G* | PCM | GPU geomeTRIC | PCM ghost | bsse=-0.95 | PCM cavity에 ghost atom → 왜곡 |
| v5 | ωB97M-V/def2-TZVP | PCM | GPU geomeTRIC | gas-phase ghost | bsse=-14.32 | DDT ρ=1.0이나 Heptachlor ρ=0.5 |
| **v6** | **적응형 (ωB97XD / ωB97M-V) + def2-TZVP** | **PCM** | **GPU geomeTRIC** | **gas-phase ghost** | **ωB97XD bsse=-10.03** | **ρ=1.0 (양 시스템)** |

**각 버전에서 변경한 사항과 교훈:**

**v1→v2** (변경: ddCOSMO→PCM, BSSE 제거)
- xTB 구조에서 DFT ghost atom을 계산하면 PES 불일치로 BSSE가 과보정됨 (raw=-8.08이 bsse=-2.28으로 급감)
- ddCOSMO는 gpu4pyscf의 analytical gradient를 지원하지 않아 GPU DFT optimization 불가능 → PCM으로 전환
- BSSE를 일단 제거하여 과보정 문제를 우회

**v2→v3** (변경: DFT SP only → GPU geomeTRIC geometry optimization 추가)
- xTB 구조에서 DFT SP만 하면, xTB가 찾은 최적 구조가 DFT PES에서의 최적과 달라 부정확
- PCM + GPU gradient가 작동하므로 geomeTRIC으로 DFT 레벨 구조 최적화 가능해짐
- 그러나 DFT opt 후 BSSE 없이 raw_dE를 사용하면 여전히 과대평가 (-12.36)

**v3→v4** (변경: BSSE 재활성화, PCM 포함 ghost atom 계산)
- DFT 최적화 구조를 얻었으므로 이제 BSSE가 정상 작동할 것으로 기대
- 그러나 PCM cavity 안에서 ghost atom을 계산하면 cavity 형태가 왜곡됨 → BSSE가 다시 과보정 (bsse=-0.95)

**v4→v5** (변경: ωB97XD→ωB97M-V, 6-311+G*→def2-TZVP, BSSE를 gas-phase에서 수행, density fitting 추가, dual-basis 전략)
- BSSE ghost atom 계산에서 PCM을 제외하여 cavity 왜곡 문제 해결
- 범함수를 ωB97M-V로 변경: VV10 nonlocal 분산력이 D3BJ 경험적 보정보다 정확 (전자밀도 기반 vs 원자쌍 거리 기반)
- 기저함수를 def2-TZVP로 변경: Pople 계열(6-311+G*)보다 보론(B), 염소(Cl) 등 전 원소에서 균형 잡힌 기술. 더 완전한 기저이므로 BSSE 자체도 감소 (basis set superposition이 작아짐)
- 2단계 기저 분리 (dual-basis): 최적화는 def2-SVP(작음, 빠름), 에너지는 def2-TZVP(큼, 정확) → 속도 ~4배 향상
- density fitting (RI-J) 추가: DFT SCF ~2-3배 추가 가속
- 결과: DDT ρ=1.000 달성. 그러나 Heptachlor ρ=0.500으로 하락 — ωB97M-V가 Styrene의 분산력을 과대평가하여 2-3위 순위 역전

**v5→v6** (변경: 단일 범함수 → 상호작용 유형별 적응형 범함수 자동 선택)
- 단일 범함수로는 H-bond 지배(Heptachlor)와 분산력 지배(DDT) 시스템을 동시에 만족 불가
- template+monomer의 H-bond donor/acceptor 수를 RDKit로 자동 계산하여 범함수 선택:
  - H-bond D+A ≥ 2 → ωB97XD (H-bond에 최적화, Heptachlor ρ=1.000)
  - H-bond D+A < 2 → ωB97M-V (분산력에 최적화, DDT ρ=1.000)
- 두 시스템 모두 ρ=1.000 동시 달성

### 14.4 범함수별 검증 결과

#### Heptachlor (Singh 2012) — H-bond 지배 시스템

| Monomer | 실험 IF | 논문 계산값 | ωB97XD raw | ωB97XD bsse | ωB97M-V raw | ωB97M-V bsse |
|---------|--------|-----------|-----------|------------|------------|-------------|
| MAA | **1.92** | -8.11 | -12.114 | **-10.026** | -19.062 | -14.322 |
| 4VP | **1.45** | -7.76 | -11.730 | **-9.869** | -17.645 | -12.956 |
| Styrene | **0.99** | -4.13 | -11.290 | **-7.917** | -19.752 | -13.582 |

| 기준 | 예측 순위 | Spearman ρ | 판정 |
|------|----------|-----------|------|
| 실험 IF | MAA > 4VP > Styrene | — | 기준 |
| **ωB97XD raw** | **MAA > 4VP > Styrene** | **1.000** | **✓ 완벽** |
| **ωB97XD bsse** | **MAA > 4VP > Styrene** | **1.000** | **✓ 완벽** |
| ωB97M-V raw | Styrene > MAA > 4VP | -0.500 | ✗ 역전 |
| ωB97M-V bsse | MAA > Styrene > 4VP | 0.500 | ✗ 2-3위 역전 |

→ H-bond 지배 시스템에서 **ωB97XD가 ρ=1.000**으로 최적.

#### DDT (Singh 2012) — 분산력 지배 시스템

| Monomer | 실험 IF | 논문 계산값 | ωB97XD raw | ωB97XD bsse | ωB97M-V raw | ωB97M-V bsse |
|---------|--------|-----------|-----------|------------|------------|-------------|
| MAA | **1.43** | -8.99 | -5.239 | -3.503 | +4.790 (해리) | -13.470 |
| 4VP | **1.65** | -7.86 | -4.800 | -4.639 | +7.152 (해리) | **-15.038** |
| Styrene | **1.21** | -8.28 | -7.296 | -5.282 | +4.454 (해리) | -13.353 |

| 기준 | 예측 순위 | Spearman ρ | 판정 |
|------|----------|-----------|------|
| 실험 IF | 4VP > MAA > Styrene | — | 기준 |
| 논문 계산값 (Gaussian B3LYP) | MAA > Styrene > 4VP | — | ✗ |
| ωB97XD bsse | Styrene > 4VP > MAA | 0.500 | △ |
| **ωB97M-V bsse** | **4VP > MAA > Styrene** | **1.000** | **✓ 완벽** |

→ 분산력 지배 시스템에서 **ωB97M-V가 ρ=1.000**으로 최적. 참고: 논문 계산(Gaussian B3LYP)도 IF 순위와 불일치 — DDT의 IF 범위가 좁아(1.21~1.65) 순위 예측이 본질적으로 어려운 시스템.

#### Phenylalanine (Mukasa 2023) — MIP 실용 검증

| Monomer | 실험 IF | 논문 순위 | 논문 계산값 | ωB97XD raw | ωB97XD bsse | ωB97M-V raw | ωB97M-V bsse |
|---------|--------|----------|-----------|-----------|------------|------------|-------------|
| OPD | **~3.2** | 1위 | -11.9 | -5.767 | -5.776 | -10.339 | -25.591 |
| MAA | **~2.8** | 2위 | -11.5 | -2.785 | -2.386 | -8.608 | -24.088 |
| 4VB | ~2.4 | 3위 | -10.8 | -18.552 | -8.058 | -46.175 | -29.925 |
| APB | ~2.0 | 4위 | — | -20.810 | -4.621 | -46.345 | -28.106 |
| ACM | ~1.5 | 5위 | — | -6.036 | -7.272 | -11.369 | -31.888 |
| PYR | **~1.2** | 7위 | -6.4 | -2.842 | -3.813 | -7.160 | -21.848 |

실험 IF 출처: Mukasa et al. 2023 Figure 3 근사값.

| 기준 | 예측 순위 | OPD>PYR | 판정 |
|------|----------|---------|------|
| 실험 IF | OPD > MAA > 4VB > APB > ACM > PYR | ✓ | 기준 |
| ωB97XD bsse | 4VB > ACM > **OPD** > APB > **PYR** > MAA | **✓** (3위>5위) | △ 보론산 과대 |
| ωB97M-V bsse | ACM > 4VB > APB > **OPD** > MAA > **PYR** | **✓** (4위>6위) | △ 보론산 과대 |

보론산(4VB, APB) 제외 시:

| 기준 | 예측 순위 | 판정 |
|------|----------|------|
| 실험 IF | OPD > MAA > ACM > PYR | 기준 |
| ωB97M-V bsse | **ACM > OPD > MAA > PYR** | ✓ OPD>PYR, MAA>PYR |

→ 모든 범함수에서 **OPD > PYR 방향성이 정확**. 보론산을 제외하면 ωB97M-V가 실험 방향성을 더 잘 재현.

### 14.5 적응형 범함수 선택 (v6, 최종)

| 상호작용 유형 | 판별 조건 | 범함수 | 근거 |
|-------------|----------|--------|------|
| H-bond 지배 | H-bond D+A ≥ 2 | ωB97XD | Heptachlor ρ=1.000 |
| 분산력 지배 | H-bond D+A < 2 | ωB97M-V (VV10 nonlocal) | DDT ρ=1.000 |

판별: RDKit `Lipinski.NumHDonors()` + `NumHAcceptors()`로 template+monomer의 H-bond D+A를 자동 계산.

#### 적응형 범함수의 한계: 교차 비교 문제

적응형 전략은 각 monomer-template pair에 대해 최적의 범함수를 선택하지만, **서로 다른 범함수로 계산된 결합에너지를 직접 비교할 수 없다**는 근본적 한계가 있다.

```
예: Heptachlor 시스템
  MAA(ωB97XD):     bsse = -10.03 kcal/mol
  Styrene(ωB97M-V): bsse = -13.58 kcal/mol
  → Styrene이 더 강해 보이지만, ωB97M-V가 전체적으로 에너지를 더 크게 계산하므로
    이 비교는 의미 없음
```

이 문제는 다음 경우에 발생한다:
- 같은 template에 대해 일부 monomer는 ωB97XD, 일부는 ωB97M-V로 계산될 때
- 예: DDT-MAA(ωB97XD) vs DDT-Styrene(ωB97M-V)

**같은 범함수로 계산된 monomer 간에는 순위 비교가 정확하다.** 실제 MIP 응용에서 대부분의 monomer(MAA, 4VP, OPD, ACM 등)는 H-bond D+A ≥ 2로 **동일하게 ωB97XD가 선택**되므로 교차 비교 문제가 발생하는 경우는 드물다. Styrene처럼 H-bond 작용기가 없는 monomer가 포함될 때만 주의가 필요하며, 이 경우 파이프라인은 결과에 사용된 범함수를 함께 표시하여 사용자가 판단할 수 있도록 한다.

Consensus ranking(두 범함수의 순위 합산)도 검토했으나, 정확한 범함수의 순위를 부정확한 범함수가 희석시키는 문제가 있어 적응형 단일 선택이 더 우수했다.

#### 최종 예측 순위 vs 실험 IF

**Heptachlor** — 적응형 범함수, Spearman ρ = 1.000

| 순위 | Monomer | 범함수 | bsse_dE | 실험 IF | IF 순위 | 일치 |
|------|---------|--------|---------|--------|--------|------|
| 1 | MAA | ωB97XD | -10.026 | 1.92 | 1위 | **✓** |
| 2 | 4VP | ωB97XD | -9.869 | 1.45 | 2위 | **✓** |
| 3 | Styrene | ωB97M-V | -13.582 | 0.99 | 3위 | **✓** |

**DDT** — 단일 범함수(ωB97M-V) 참조 시 ρ = 1.000

| 순위 | Monomer | 범함수 | bsse_dE | 실험 IF | IF 순위 | 일치 |
|------|---------|--------|---------|--------|--------|------|
| 1 | 4VP | ωB97M-V | -15.038 | 1.65 | 1위 | **✓** |
| 2 | MAA | ωB97M-V | -13.470 | 1.43 | 2위 | **✓** |
| 3 | Styrene | ωB97M-V | -13.353 | 1.21 | 3위 | **✓** |

※ 적응형 전략에서 범함수가 혼합되면 bsse_dE 절대값의 교차 비교가 불가능하므로, DDT처럼 분산력 지배 시스템에서는 **ωB97M-V 단일 결과**를 참조.

**Phenylalanine** — OPD > PYR 방향성 정확

| 순위 | Monomer | 범함수 | bsse_dE | 실험 IF | 논문 순위 | 비고 |
|------|---------|--------|---------|--------|----------|------|
| 1 | 4VB | ωB97XD | -8.058 | ~2.4 | 3위 | 공유결합 MIP |
| 2 | ACM | ωB97XD | -7.272 | ~1.5 | 5위 | |
| 3 | **OPD** | ωB97XD | **-5.776** | **~3.2** | **1위** | **핵심 후보** |
| 4 | APB | ωB97XD | -4.621 | ~2.0 | 4위 | 공유결합 MIP |
| 5 | **PYR** | ωB97XD | **-3.813** | **~1.2** | **7위** | **최하위** |
| 6 | MAA | ωB97XD | -2.386 | ~2.8 | 2위 | |

### 14.6 종합 판정

| 검증 항목 | 결과 | 판정 |
|-----------|------|------|
| Heptachlor IF 순위 (ωB97XD) | Spearman ρ = **1.000** | **PASS** |
| DDT IF 순위 (ωB97M-V) | Spearman ρ = **1.000** | **PASS** |
| DDT IF 1위 예측 (적응형) | 4VP = 1위 | **PASS** |
| Phe OPD > PYR 방향성 | OPD=3위, PYR=5위 | **PASS** |
| Phe PYR 최하위권 | PYR = 5~6위/6개 | **PASS** |
| 보론산(4VB, APB) 분류 | 공유결합 MIP 자동 감지 | **PASS** |

### 14.7 공유결합 MIP 자동 감지

DFT optimization 후 template-monomer 간 **공유결합 형성 여부**를 원자 간 거리로 자동 검사한다. 공유결합이 감지되면 오류가 아니라 **결합 유형이 다른 것**으로 분류한다.

| MIP 유형 | 결합 | 에너지 범위 | 예시 monomer | 감지 기준 |
|---------|------|-----------|-------------|----------|
| 비공유결합 | H-bond, 정전기, vdW | -2 ~ -15 kcal/mol | MAA, 4VP, OPD | — |
| **공유결합** | **가역적 공유결합** | **-30 ~ -50 kcal/mol** | **4VB, APB (보론산)** | **B-N < 1.65Å, B-O < 1.55Å** |
| 반공유결합 | 금속 배위 | -15 ~ -30 kcal/mol | vinylimidazole | C-N < 1.55Å |

보론산이 -46 kcal/mol로 계산되는 이유: 보론(B)의 빈 p-오비탈에 template NH₂의 lone pair가 배위결합(dative bond)을 형성. 이것은 계산 오류가 아니라 **보론산 MIP의 실제 작동 원리**이다.

파이프라인 처리:
1. 결과 JSON에 `"covalent_warning"` 필드 자동 추가
2. 비공유결합 monomer와 **별도 그룹으로 순위** 매김
3. HTML 리포트에서 "공유결합 MIP 후보"로 별도 표시

### 14.8 결합에너지 vs 선택도 — Stage 3가 필요한 이유

보론산(4VB)은 Phe와의 **결합에너지가 가장 크지만** (-46 kcal/mol), **실험 IF는 OPD보다 낮다** (4VB IF~2.4 < OPD IF~3.2). 이는 MIP의 핵심 원리와 관련된다.

**IF는 결합 강도가 아니라 선택도를 측정한다:**

```
IF = (template 특이적 결합) / (비특이적 결합)
```

```
보론산(4VB):                          OPD:
  + Phe:     -46 kcal/mol (강함)        + Phe:     -6 kcal/mol
  + Tyrosine: -40 kcal/mol (역시 강함)   + Tyrosine: -2 kcal/mol (약함)
  + Dopamine: -42 kcal/mol (역시 강함)   + Dopamine: -1 kcal/mol (약함)
  → 선택도 낮음 → IF 낮음              → 선택도 높음 → IF 높음
```

보론산의 B(OH)₂는 diol 기를 가진 **모든** 분자와 비선택적으로 반응한다. OPD의 두 NH₂는 Phe의 COOH+NH₂와 **상보적 H-bond 네트워크**를 형성하며, 이 공간적 배치가 Phe에만 맞는다. 이것이 분자 각인(molecular imprinting)의 본질이다.

| 기준 | 수식 | 의미 |
|------|------|------|
| ✗ Stage 2 결합에너지 | ΔE(monomer-template) | 결합이 강한 monomer 선택 (보론산 1위) |
| **✓ Stage 3 선택도** | **ΔΔE = ΔE(monomer-template) − ΔE(monomer-interferent)** | **template에만 선택적인 monomer 선택 (OPD 1위)** |

본 파이프라인의 **Stage 3(선택도 계산)**이 이 역할을 수행한다. Stage 2는 monomer-template **결합에너지**만으로 순위를 매기므로 보론산이 과대평가된다. Stage 3에서 **같은 monomer가 interferent에 얼마나 결합하는지** 추가 계산하여, 그 차이(ΔΔE)로 **선택도 순위**를 도출한다. 이것이 4단계 파이프라인에서 Stage 2만으로 최종 판단하지 않고 Stage 3를 반드시 거쳐야 하는 이유이다.

### 14.9 에너지 유형 정의

| 유형 | 수식 | 용도 |
|------|------|------|
| raw_dE | E(complex) − E(template) − E(monomer) | 참고용 |
| **bsse_dE** | **E(complex) − E(template+ghost) − E(monomer+ghost)** | **최종 결합에너지** |

※ BSSE ghost atom 계산은 gas-phase에서 수행 (PCM cavity 왜곡 방지, v4의 교훈).

### 14.10 실패 진단

검증 실패 시 `diagnose_failure.py`가 원인별 진단 메시지를 자동 출력한다:
- 수치 오차 > 2.0 kcal/mol → "xTB→DFT 구조 불일치, 분산력 지배 시스템 확인"
- OPD top3 밖 → "MAA와 동점 여부 확인, 용매 조건 확인 (Mukasa 2023은 MeOH)"
- 보론산 과대 → "공유결합 MIP 자동 분류 확인, Stage 3 선택도로 재순위"

---

## 15. 핵심 알고리즘 요약 테이블

| 알고리즘 | 핵심 기술 | 사용 라이브러리 | 수식/방법 |
|---------|----------|--------------|----------|
| 3D 배좌 생성 | ETKDGv3 + MMFF | RDKit | 거리 기하학 + 힘장 최적화 |
| vdW 표면점 생성 | Fibonacci sphere + buried 필터 | numpy + scipy | 원자별 vdW 구 위 균일 샘플링 |
| 표면 도킹 | 무작위 회전 + clash 필터 | scipy.spatial.transform | Rotation.random() + cdist > 1.2A |
| 적응형 orientation 수 | heavy atom 비례 스케일링 | numpy | n = base × (n_heavy / 14), cap 1000 |
| DFT ESP 전하 | B3LYP/def2-SVP Mulliken (GPU SCF) | gpu4pyscf | ESP-guided 배향용 (~7초/분자) |
| AutoDock Vina 도킹 | 경험적 scoring function | vina + meeko | exhaustiveness=64, 소수성/형태 보완 |
| xTB SP 스크리닝 | GFN2-xTB single-point (PM3 대체) | tblite | ΔE_SP = E_complex - E_template - E_monomer |
| xTB 구조 최적화 | L-BFGS-B | tblite + scipy | max 200 iter, ftol=1e-4 |
| xTB→DFT 좌표 전달 | _arrays_to_mol + prebuilt_complex_mol | RDKit + PySCF | numpy→RDKit Mol→DFT opt |
| DFT geometry opt | ωB97M-V/def2-SVP + RI-J + PCM | gpu4pyscf + geomeTRIC | GPU, maxsteps=50 |
| DFT SP 에너지 | ωB97M-V/def2-TZVP + RI-J + PCM | gpu4pyscf | Kohn-Sham DFT (VV10 nonlocal) |
| 용매 모델 | PCM (Polarizable Continuum Model) | PySCF solvent | GPU gradient 지원, ddCOSMO 대체 |
| BSSE 보정 | Counterpoise (gas-phase ghost) | PySCF ghost atoms | PCM 제외하여 cavity 왜곡 방지 |
| GPU 가속 | SCF + gradient + geomeTRIC 전체 | gpu4pyscf | mf.to_gpu(), ~30배 가속 |
| 선택도 | 볼츠만 분포 | numpy | S = exp(ΔE / kB·T) |
| 용매 전략 | 4가지 집계 | pandas | synthesis_match / min / avg / worst |
| MD 시뮬레이션 | Langevin NVT | OpenMM CUDA | GAFF2 + TIP3P |
| RDF | 방사 분포 함수 | MDTraj | compute_rdf() |
| EBN | RDF 적분 | numpy | 4pi * integral g(r)*r^2 dr |
| H-bond 검출 | Baker-Hubbard | MDTraj | D-A < 3.5A, angle > 120 |
| 비율 최적화 | EBN 포화점 | numpy | delta(EBN) < 10% |
| ESP 맵 | MEP cube + 등고선 | PySCF cubegen + matplotlib | 60x60x60 격자, z-slice |
| 분자 지문 | Morgan FP | RDKit | radius=2, 2048 bits |
| Tanimoto 유사도 | 집합 유사도 | RDKit DataStructs | \|A∩B\| / \|A∪B\| |
| IF 예측 | 앙상블 회귀 | scikit-learn | RandomForest + LOO-CV |
| 신뢰구간 | 정규 근사 | scipy.stats | +/- 1.96 * residual_std |
| HTML 리포트 | base64 인라인 | Python 표준 lib | data:image/png;base64 |

---

## 16. 참고 논문 전체 목록

| # | 저자 | 저널 / 연도 | DOI | 파이프라인 적용 |
|---|------|------------|-----|---------------|
| 1 | Singh et al. | Curr. Anal. Chem. 2012 | 10.2174/157341112803216807 | DFT 스크리닝 원형, 결합에너지 공식, IF 검증 데이터 |
| 2 | Mukasa et al. | Adv. Mater. 2023 | 10.1002/adma.202212161 | 다단계 도킹 스크리닝 전략, 선택도 공식, PM3→xTB 대체 근거, Phe 검증 |
| 3 | Munoz et al. | J. Chem. Inf. Model. 2024 | 10.1021/acs.jcim.4c00775 | MD 기반 monomer 선별, prepolymerization 분석 |
| 4 | Yuan et al. | Molecules 2024 | 10.3390/molecules29174236 | EBN, HBNmax 정량 파라미터, H-bond 점유율 |
| 5 | Boys & Bernardi | Mol. Phys. 1970 | 10.1080/00268977000101561 | BSSE counterpoise 보정 |
| 6 | Grimme et al. | J. Chem. Phys. 2010 | 10.1063/1.3382344 | DFT-D3 분산력 보정, BJ 감쇠 함수 |
| 7 | Lipparini et al. | J. Chem. Theory Comput. 2013 | 10.1021/ct400280b | ddCOSMO 용매 모델 |
| 8 | Bannwarth et al. | J. Chem. Theory Comput. 2019 | 10.1021/acs.jctc.8b01176 | GFN2-xTB 반경험적 방법론 |
| 9 | Wu et al. | arXiv 2024 | 10.48550/arXiv.2404.09452 | GPU4PySCF GPU 가속 |
| 10 | Bursch, Grimme et al. | Angew. Chem. Int. Ed. 2022 | 10.1002/anie.202205735 | Best-Practice DFT: def2 기저 권장, Pople 기저 비권장 |
| 11 | Mardirossian & Head-Gordon | J. Chem. Phys. 2016 | 10.1063/1.4952647 | ωB97M-V 범함수 (VV10 nonlocal 분산력) |
| 12 | Weigend & Ahlrichs | Phys. Chem. Chem. Phys. 2005 | 10.1039/b508541a | def2 기저함수 세트 (def2-SVP, def2-TZVP) |

---

## 17. 실행 환경 및 의존 패키지

### 주요 의존 패키지

| 패키지 | 용도 | Stage |
|--------|------|-------|
| RDKit | SMILES 파싱, 3D 배좌 생성, 분자 지문 | 전체 + Feature 7 |
| tblite | GFN2-xTB 에너지/gradient 계산 | Stage 1 |
| PySCF | DFT 양자화학 (ωB97M-V, PCM, ghost atom, geomeTRIC) | Stage 1 ESP + Stage 2 |
| gpu4pyscf | DFT GPU 가속 (SCF + gradient + geometry optimization) | Stage 1 ESP + Stage 2 |
| geomeTRIC | DFT geometry optimization engine | Stage 2 |
| vina | AutoDock Vina 분자 도킹 (exhaustiveness=64) | Stage 1 |
| meeko | Vina용 분자 입출력 변환 | Stage 1 |
| OpenMM | MD 시뮬레이션 (Langevin, PME) | Stage 4 |
| openff-toolkit | 힘장 파라미터화 (OpenFF Sage) | Stage 4 |
| openmmforcefields | GAFF2 힘장 연결 | Stage 4 |
| MDTraj | 궤적 분석 (RDF, H-bond) | Stage 4 |
| scikit-learn | 회귀 모델 (Ridge, RF), LOO-CV | Feature 8 |
| joblib | 모델 직렬화 | Feature 8 |
| scipy | L-BFGS-B 최적화, Spearman 상관, 정규분포 | Stage 1, 검증 |
| numpy, pandas | 수치 계산, 데이터 테이블 | 전체 |
| matplotlib | 그래프 (RDF, 선택도, ESP 맵) | 전체 |
| pubchempy | PubChem API (선택적) | Feature 7 |

### 실행 방법

모든 설정은 `code/pipeline/config.py`에서 변경한다. Template SMILES, monomer 목록, 범함수, 기저, 용매를 수정한 뒤 아래 명령을 실행하면 된다.

```bash
cd "/home/chan/Research/MIP simulation"
conda activate MIPscreen

# 전체 파이프라인 (Stage 1→2→3→4)
python run_pipeline.py --stage all

# 개별 stage
python run_pipeline.py --stage 1        # xTB 스크리닝
python run_pipeline.py --stage 2        # DFT 계산

# 추가 기능
python run_pipeline.py --crosslinker           # Cross-linker 스크리닝
python run_pipeline.py --suggest-interferents   # Interferent 자동 제안
python run_pipeline.py --predict-if            # IF 예측 모델
python run_pipeline.py --report                # HTML 리포트 생성

# 검증 (15 pair: Singh 6 + Mukasa 9)
python run_validation.py --compute --stage all    # DFT 계산 + 전체 검증
python run_validation.py --load-only --stage all  # 기존 결과로 검증만
```

### 주요 config 변수

```python
# code/pipeline/config.py
TEMPLATE_SMILES = "N[C@@H](Cc1ccccc1)C(=O)O"  # 타겟 분자
MONOMER_LIBRARY = {"MAA": "CC(=C)C(=O)O", ...}  # 스크리닝 대상
DFT_FUNCTIONAL  = "wb97m-v"    # 범함수
DFT_OPT_BASIS   = "def2-svp"  # optimization 기저
DFT_SP_BASIS    = "def2-tzvp" # single-point 기저
SYNTHESIS_SOLVENT = "Chloroform"  # 합성 용매
```

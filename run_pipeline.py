#!/usr/bin/env python
"""
MIP Screening Pipeline — 프로젝트 루트 실행 파일
================================================
config-driven 7-stage 워크플로. 루트에서 `python run_pipeline.py`로 실행하며
결과는 results/<TEMPLATE_NAME>/stage1..7/ 에 저장됩니다 (실행 위치 무관).

준비:
    conda activate MIPscreen        # 또는 사용 중인 env
    cd MIP_simulation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 실행 (--stage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python run_pipeline.py                    # 전체 (1→2→3→4→5→6→7) [기본]
    python run_pipeline.py --stage 1          # GFN2-xTB 도킹 + 포즈 앙상블
    python run_pipeline.py --stage 2          # DFT 단일점(ωB97X-D/M-V+PCM+BSSE//xTB)
    python run_pipeline.py --stage 3          # 전역 porogen 선택 + 랭킹 + 가교제 스크리닝
    python run_pipeline.py --stage 4          # MMSD 다중-모노머 조합 탐색
    python run_pipeline.py --stage 5          # pre-polymerization MD (GROMACS)
    python run_pipeline.py --stage 6          # VIP cavity rebinding
    python run_pipeline.py --stage 7          # 합성 레시피 생성
  * 각 stage는 item-level(모노머/용매별)로 resume — 이미 계산된 건 건너뜀.
  * stage 5(MD)도 완료된 궤적은 skip하고 분석만 다시 함.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
다중 TARGET (--all-templates / MIP_TEMPLATE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python run_pipeline.py --all-templates                 # config.TEMPLATES 전부 순차
    python run_pipeline.py --all-templates --stage 5       # 특정 stage만 전 target
    MIP_TEMPLATE="Methyl Benzoate" python run_pipeline.py  # 단일 target을 env로 선택
  * --all-templates는 target별 subprocess로 격리 → results/<TEMPLATE_NAME>/ 분리 저장.
  * config.TEMPLATE_NAME 기본값을 바꿔도 됨 (또는 위 env). --output-dir와 병용 금지.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIP 재분석/연장 (BIO phase5) — MD 재실행 없이 지표 개선
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python run_pipeline.py --reanalyze-vip    # 기존 stage6 궤적 → Q4/residence/retention/ΔG 재계산 (수 초)
    python run_pipeline.py --extend-vip        # 미수렴(drift>1.5Å) rebinding MD만 연장 후 재분석

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
부가 기능
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python run_pipeline.py --crosslinker            # 가교제 스크리닝만 (항목별 캐시)
    python run_pipeline.py --report                 # 기존 결과로 HTML 리포트
    python run_pipeline.py --suggest-interferents   # interferent 제안
    python run_pipeline.py --auto-interferents      # 제안 + config 자동 반영
    python run_pipeline.py --predict-if             # imprinting factor 예측
    python run_pipeline.py --update-if-model --monomer NVP --experimental-if 3.2
                                                    # 실측 IF로 예측모델 갱신

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
공통 옵션
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    --template "<SMILES>"    config 대신 임의 SMILES로 실행
    --output-dir <경로>      결과 경로 지정 (기본: results/<TEMPLATE_NAME>)
    --force                  Stage 4 MMSD를 캐시 무시하고 재계산
                             (다른 stage는 해당 출력 폴더를 지워 재실행)
    -h / --help              전체 플래그 도움말

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
설정 (code/pipeline/config.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TEMPLATES / TEMPLATE_NAME     타겟 분자(들). env MIP_TEMPLATE로 오버라이드
    MONOMER_LIBRARY               스크리닝 monomer (vinyl/silane/oxidative 자동감지)
    CROSSLINKER_LIBRARY           가교제 (화학-호환 자동 필터, 가장 inert 선택)
    SOLVENTS / SOLVENT_STRATEGY   용매(ε) / 전역 porogen 선택
    DFT_FUNCTIONAL, DFT_SP_BASIS  DFT 범함수/기저 (SP-only//xTB)
    DFT_GEOMETRY_OPT              True면 DFT 구조최적화(기본 False=SP)
    MMSD_*                        조합 탐색(greedy/BO/NSGA-II) 파라미터
    MD_TIME_NS, MD_MMPBSA         Stage 5 MD 길이 / MM-GBSA
    VIP_N_SNAPSHOTS(10), VIP_REBINDING_NS(50)   VIP 샘플링
    VIP_N_RESTARTS, VIP_MMPBSA, VIP_EXTEND_NS   VIP 부가(멀티리스타트/ΔG/연장)
    MD_RATIO_METHOD               합성 비율 산정("ebn")

출력: results/<TEMPLATE_NAME>/{stage1..7, features, reports}/
      reports/synthesis_protocol.txt · synthesis_recipe.json · mip_report_*.html
"""
import sys
from pathlib import Path

# code/ 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent / "code"))

from pipeline.run_pipeline import main
main()

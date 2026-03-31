#!/usr/bin/env python
"""
MIP Pipeline Validation — 프로젝트 루트 실행 파일
=================================================
사용법:
    conda activate MIPscreen
    cd "MIP simulation"

    # 전체 계산 + 검증 (처음 실행 시)
    python run_validation.py --all-compute --stage all

    # 개별 단계 실행
    python run_validation.py --compute         # Step 1: monomer-template DFT
    python run_validation.py --selectivity     # Step 2: monomer-interferent DFT (Stage 3)
    python run_validation.py --md              # Step 3: Stage 4 MD 시뮬레이션

    # 순위 비교만 (이미 계산된 결과 사용)
    python run_validation.py --rank            # Stage 2/3/4 순위 vs 실험 IF

    # 기존 결과로 검증만 실행
    python run_validation.py --load-only --stage all

    # 특정 monomer만 MD 실행
    python run_validation.py --md --md-monomers OPD MAA PYR

계산 단계:
    Step 1 (--compute):     monomer-template DFT → reference_energies.json
    Step 2 (--selectivity): monomer-interferent DFT → selectivity_data.json
    Step 3 (--md):          50ns MD × 6 monomers → stage4_md.json

검증 대상:
    Singh 2012 (6):   Heptachlor × (MAA, 4VP, Styrene) + DDT × (MAA, 4VP, Styrene)
    Mukasa 2023 (6):  Phe × (OPD, MAA, 4VB, APB, ACM, PYR)
    Interferents (5): Tyrosine, Valine, Leucine, Isoleucine, Dopamine

설정:
    모든 설정은 code/pipeline/config.py 에서 변경합니다.
"""
import sys
from pathlib import Path

# code/ 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent / "code"))

from validation.run_validation import main
main()

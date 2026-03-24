#!/usr/bin/env python
"""
MIP Pipeline Validation — 프로젝트 루트 실행 파일
=================================================
사용법:
    conda activate MIPscreen
    cd "/home/chan/Research/MIP simulation"

    # 문헌 기준값 DFT 계산 + 전체 검증
    python run_validation.py --compute --stage all

    # 기존 결과로 검증만 실행
    python run_validation.py --load-only --stage all

    # 문헌 기준값 DFT 계산만 (19 pair: Singh 6 + Mukasa 9 + Gallic Acid 4)
    python run_validation.py --compute

    # 특정 검증만 실행
    python run_validation.py --load-only --stage numerical   # 수치 재현성
    python run_validation.py --load-only --stage ranking     # IF 순위 상관
    python run_validation.py --load-only --stage selectivity # 선택도 방향
    python run_validation.py --load-only --stage new_features # 추가 기능 검증

검증 대상 (19 pair):
    Singh 2012 (6):  Heptachlor × (MAA, 4VP, Styrene) + DDT × (MAA, 4VP, Styrene)
    Mukasa 2023 (9): Phe × (OPD, MAA, 4VB, APB, ACM, PYR) + interferents (3)
    Pardeshi 2012 (4): Gallic Acid × (AA, AAm, 4VP, MMA)

설정:
    모든 설정은 code/pipeline/config.py 에서 변경합니다.
    검증 기준값은 code/validation/config_validation.py 에 정의되어 있습니다.
"""
import sys
from pathlib import Path

# code/ 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent / "code"))

from validation.run_validation import main
main()

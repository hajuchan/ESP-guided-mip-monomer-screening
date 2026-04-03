#!/usr/bin/env python
"""
MIP Screening Pipeline — 프로젝트 루트 실행 파일
================================================
사용법:
    conda activate MIPscreen
    cd MIP_simulation

    # 전체 파이프라인 실행 (Stage 1→2→3→4)
    python run_pipeline.py

    # 특정 스테이지만 실행
    python run_pipeline.py --stage 1        # xTB 스크리닝
    python run_pipeline.py --stage 2        # DFT 계산
    python run_pipeline.py --stage 3        # 선택도 분석
    python run_pipeline.py --stage 4        # MD 시뮬레이션

    # 추가 기능
    python run_pipeline.py --crosslinker           # Cross-linker 스크리닝
    python run_pipeline.py --suggest-interferents   # Interferent 자동 제안
    python run_pipeline.py --predict-if            # IF 예측 모델
    python run_pipeline.py --report                # HTML 리포트 생성

설정:
    모든 설정은 code/pipeline/config.py 에서 변경합니다.
    - TEMPLATE_SMILES: 타겟 분자
    - MONOMER_LIBRARY: 스크리닝할 monomer 목록
    - DFT_FUNCTIONAL: DFT 범함수 (기본: wb97m-v)
    - DFT_SP_BASIS: SP 기저함수 (기본: def2-tzvp)
    - SOLVENTS: 용매 및 유전상수
"""
import sys
from pathlib import Path

# code/ 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent / "code"))

from pipeline.run_pipeline import main
main()

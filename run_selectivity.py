#!/usr/bin/env python
"""
MIP Stage 3 Selectivity 계산 — 프로젝트 루트 실행 파일
=====================================================
monomer-interferent 결합에너지를 계산하고, 범함수별 Stage 3 선택도 순위를 도출한다.

사용법:
    conda activate MIPscreen
    cd "/home/chan/Research/MIP simulation"

    # monomer-interferent DFT 계산 (60개: 6 monomer × 5 interferent × 2 범함수)
    python run_selectivity.py --compute

    # 계산 완료 후 순위 비교 (Stage 2 결합에너지 vs Stage 3 선택도)
    python run_selectivity.py --rank

    # 계산 + 순위 한번에
    python run_selectivity.py --all

계산 내용:
    Stage 3 선택도 공식:
        ΔΔE = ΔE(monomer-template) - ΔE(monomer-interferent)
        S = exp(ΔΔE / kB·T)

    필요한 데이터:
        ΔE(monomer-template)     — run_validation.py --compute로 이미 계산됨
        ΔE(monomer-interferent)  — 이 스크립트가 계산

    조합:
        Monomer 6개:      OPD, MAA, 4VB, APB, ACM, PYR
        Interferent 5개:  Tyrosine, Valine, Leucine, Isoleucine, Dopamine
        범함수 2종:       ωB97XD, ωB97M-V
        → 총 60개 DFT 계산

    중간 중단 후 재실행하면 완료된 pair는 자동으로 건너뜁니다.

결과 파일:
    results/validation/selectivity_data.json    — monomer-interferent 결합에너지
    results/validation/ranking_comparison.json  — Stage 2 vs Stage 3 순위 비교

사전 조건:
    run_validation.py --compute가 먼저 완료되어야 합니다.
    (reference_energies.json에 monomer-template 결합에너지가 필요)

설정:
    모든 설정은 code/pipeline/config.py 에서 변경합니다.
"""
import sys
from pathlib import Path

# code/ 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent / "code"))

from validation.compute_selectivity import compute_selectivity_data, compute_rankings
import argparse

parser = argparse.ArgumentParser(
    description="MIP Stage 3 Selectivity: monomer-interferent DFT 계산 및 순위 비교"
)
parser.add_argument("--compute", action="store_true",
                    help="monomer-interferent 결합에너지 DFT 계산 (60개)")
parser.add_argument("--rank", action="store_true",
                    help="Stage 2 vs Stage 3 순위 비교 출력")
parser.add_argument("--all", action="store_true",
                    help="계산 + 순위 비교 한번에 실행")
parser.add_argument("--interferents", nargs="+", default=None,
                    help="특정 interferent만 계산 (예: --interferents Valine Isoleucine)")
args = parser.parse_args()

if args.all or args.compute:
    compute_selectivity_data(interferent_filter=args.interferents)
if args.all or args.rank:
    compute_rankings()
if not (args.all or args.compute or args.rank):
    parser.print_help()

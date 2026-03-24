"""
Diagnostic messages for failed validations
===========================================
Reads validation_report.json and produces actionable diagnostics
for each failed check.
"""

import json
import logging
from pathlib import Path

from pipeline.config import OUTPUT_DIR
from .config_validation import VALID_SOLVENT_STRATEGIES

logger = logging.getLogger(__name__)


def diagnose_failures(output_dir: str = OUTPUT_DIR) -> list[str]:
    """Analyse validation_report.json and return diagnostic strings."""

    report_path = Path(output_dir) / "validation" / "validation_report.json"

    if not report_path.exists():
        msg = "검증 결과 없음. run_validation.py를 먼저 실행하세요."
        print(msg)
        return [msg]

    with open(report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)

    diagnostics: list[str] = []

    # Merge legacy and feature validations into flat dict
    all_validations = {}
    all_validations.update(report.get("legacy_validation", {}))
    all_validations.update(report.get("feature_validation", {}))

    for name, entry in all_validations.items():
        if not isinstance(entry, dict):
            continue
        overall = entry.get("overall", "PASS")
        if overall in ("PASS", "SKIP"):
            continue

        # ── Legacy pipeline failures ────────────────────────────────
        if name == "numerical":
            diffs = entry.get("diffs", {})
            max_diff = max((abs(v) for v in diffs.values()), default=0.0)

            if max_diff > 2.0:
                diagnostics.extend([
                    f"[{name}] BSSE ghost atom 구현 오류 가능성",
                    f"[{name}] PySCF ghost atom 키워드 확인",
                    f"[{name}] 복합체 구조 local minimum 가능",
                ])
            elif 0.5 <= max_diff <= 2.0:
                diagnostics.extend([
                    f"[{name}] D3BJ damping 파라미터 차이 (Gaussian IOP vs dftd3)",
                    f"[{name}] 허용 범위이나 dftd3 BJ 파라미터 확인 권장",
                ])

        elif name == "ranking":
            diagnostics.extend([
                f"[{name}] 복합체 탐색 방향 수 확인 (COMPLEX_SEARCH_MODE)",
                f"[{name}] xTB geometry optimization 수렴 확인",
            ])

        elif name == "selectivity":
            diagnostics.extend([
                f"[{name}] 용매 확인: Mukasa 2023은 MeOH(32.61)",
                f"[{name}] SOLVENT_STRATEGY/SYNTHESIS_SOLVENT 확인",
                f"[{name}] stage2_dft.json 용매 결과 확인",
            ])

        # ── Feature validation failures (A–H) ──────────────────────
        elif name == "A_multidirectional":
            diagnostics.extend([
                f"[{name}] stage1_top.json에 best_direction 필드 없음",
                f"[{name}] COMPLEX_SEARCH_MODE 값 확인",
            ])

        elif name == "B_solvent_strategy":
            diagnostics.extend([
                f"[{name}] SOLVENT_STRATEGY 값 확인",
                f"[{name}] 허용값: {VALID_SOLVENT_STRATEGIES}",
                f"[{name}] SYNTHESIS_SOLVENT가 SOLVENTS에 없음",
            ])

        elif name == "C_ratio_screening":
            diagnostics.extend([
                f"[{name}] stage4_md.json에 ratio_results 없음",
                f"[{name}] MD_RATIO_SCREENING=True 확인",
            ])

        elif name == "D_esp_map":
            diagnostics.extend([
                f"[{name}] stage2_esp_*.png 없음",
                f"[{name}] USE_ESP_MAP=True 확인",
                f"[{name}] PySCF cubegen import 오류",
            ])

        elif name == "E_crosslinker":
            diagnostics.extend([
                f"[{name}] stage2_crosslinker.json 없음",
                f"[{name}] run_pipeline.py --crosslinker 실행",
                f"[{name}] EGDMA WARNING 시 template 상호작용 확인",
            ])

        elif name == "F_report":
            diagnostics.extend([
                f"[{name}] mip_report_*.html 없음",
                f"[{name}] run_pipeline.py --report 실행",
                f"[{name}] generate_report.py import 오류",
            ])

        elif name == "G_interferent_suggestion":
            diagnostics.extend([
                f"[{name}] suggest_interferents() import 실패",
                f"[{name}] BUILTIN_MOLECULES 확인",
                f"[{name}] RDKit MorganFingerprint 오류",
            ])

        elif name == "H_if_prediction":
            diagnostics.extend([
                f"[{name}] LOO-RMSE > 0.5",
                f"[{name}] LITERATURE_DATA 부족",
                f"[{name}] Ridge alpha 조정",
                f"[{name}] --update-if-model 권장",
            ])

        else:
            diagnostics.append(f"[{name}] 검증 실패 (overall={overall})")

    # ── Print with clear formatting ─────────────────────────────────
    if diagnostics:
        print("=" * 60)
        print("  진단 결과 (Diagnostic Messages)")
        print("=" * 60)
        for i, d in enumerate(diagnostics, 1):
            print(f"  {i}. {d}")
        print("=" * 60)
    else:
        print("모든 검증 통과 — 진단 메시지 없음.")

    return diagnostics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = diagnose_failures()
    if results:
        logger.info("총 %d건의 진단 메시지 생성", len(results))

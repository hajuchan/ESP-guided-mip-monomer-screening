"""
MIP Screening Pipeline — Main Entry Point
==========================================
Multi-stage computational screening of functional monomers
for Molecularly Imprinted Polymer (MIP) synthesis.

Stage 1: GFN2-xTB fast screening
Stage 2: DFT B3LYP-D3BJ/6-311+G* + ddCOSMO + BSSE
Stage 3: Selectivity calculation and ranking
Stage 4: OpenMM MD validation (RDF, EBN, H-bond)

Additional features:
  --crosslinker          Cross-linker screening
  --suggest-interferents Auto interferent suggestion
  --predict-if           IF prediction model
  --report               Generate HTML report
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running directly as a script (python run_pipeline.py …) in addition to
# `python -m pipeline.run_pipeline`. A bare script run has no package context, so
# the relative imports below (from .config import …) would raise "attempted
# relative import with no known parent package". Registering __package__ and
# putting the package's parent dir on sys.path makes both invocations work.
if __package__ in (None, ""):
    import os as _os
    sys.path.insert(
        0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "pipeline"

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIP Functional Monomer Screening Pipeline"
    )
    parser.add_argument(
        "--template", type=str, default=None,
        help="Template SMILES string (default: from config.py)"
    )
    parser.add_argument(
        "--all-templates", action="store_true",
        help="Run the whole pipeline for EVERY target in config.TEMPLATES, "
             "each in its own subprocess → results/<TEMPLATE_NAME>/. "
             "Respects --stage; do not combine with --output-dir."
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["1", "2", "3", "4", "5", "6", "7", "all"],
        help="Which stage to run (default: all)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute Stage 4 MMSD even if mmsd_results.json exists. "
             "Other stages already resume at the item level (per monomer / "
             "solvent pair); delete a stage's output to fully redo it."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=("Output directory. Default: <results>/<TEMPLATE_NAME> "
              "(stage1..stage7 nested inside).")
    )
    # Feature 5: Cross-linker screening
    parser.add_argument(
        "--crosslinker", action="store_true",
        help="Run cross-linker screening only"
    )
    # Feature 6: Report generation
    parser.add_argument(
        "--report", action="store_true",
        help="Generate HTML report from existing results"
    )
    # Feature 7: Interferent suggestion
    parser.add_argument(
        "--suggest-interferents", action="store_true",
        help="Suggest interferents for the template"
    )
    parser.add_argument(
        "--auto-interferents", action="store_true",
        help="Suggest and auto-apply interferents to config"
    )
    # Feature 8: IF prediction
    parser.add_argument(
        "--predict-if", action="store_true",
        help="Run imprinting factor prediction"
    )
    parser.add_argument(
        "--update-if-model", action="store_true",
        help="Update IF model with experimental data"
    )
    parser.add_argument(
        "--monomer", type=str, default=None,
        help="Monomer name for --update-if-model"
    )
    parser.add_argument(
        "--experimental-if", type=float, default=None,
        help="Experimental IF value for --update-if-model"
    )
    return parser.parse_args()


def _stage_dir(root_dir: str, stage_key: str) -> str:
    """Return stage-specific output subdirectory, creating it if needed."""
    from pathlib import Path
    p = Path(root_dir) / stage_key
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def run_stage(stage_num: int, template_smiles: str, output_dir: str,
              prev_results: dict, force: bool = False) -> dict:
    """Run a single pipeline stage and return its results.

    Stages resume at the item level (each skips already-computed monomers /
    solvent pairs). ``force=True`` bypasses reuse where a stage supports it
    (currently Stage 4 MMSD).
    """
    t0 = time.time()

    if stage_num == 1:
        from .stage1_xtb import run_stage1
        top_monomers = run_stage1(
            template_smiles=template_smiles,
            output_dir=_stage_dir(output_dir, "stage1"),
        )
        result = {"top_monomers": top_monomers}

    elif stage_num == 2:
        from .stage2_dft import run_stage2
        monomer_names = prev_results.get("top_monomers")
        dft_results = run_stage2(
            template_smiles=template_smiles,
            monomer_names=monomer_names,
            output_dir=_stage_dir(output_dir, "stage2"),
        )
        result = {"dft_results": dft_results}

    elif stage_num == 3:
        from .stage3_selectivity import run_stage3
        top_monomers = run_stage3(
            template_smiles=template_smiles,
            output_dir=_stage_dir(output_dir, "stage3"),
        )
        result = {"top_selective": top_monomers}

    elif stage_num == 4:
        from .stage4_mmsd import run_stage4
        mmsd_result = run_stage4(
            template_smiles=template_smiles,
            output_dir=_stage_dir(output_dir, "stage4"),
            force=force,
        )
        result = {"mmsd": mmsd_result}

    elif stage_num == 5:
        from .stage5_md import run_stage5
        monomer_names = prev_results.get("top_selective")
        md_results = run_stage5(
            template_smiles=template_smiles,
            monomer_names=monomer_names,
            output_dir=_stage_dir(output_dir, "stage5"),
        )
        result = {"md_results": md_results}

    elif stage_num == 6:
        from .stage6_vip import run_stage6
        vip_results = run_stage6(
            template_smiles=template_smiles,
            output_dir=_stage_dir(output_dir, "stage6"),
        )
        result = {"vip_results": vip_results}

    elif stage_num == 7:
        from .stage7_recipe import run_stage7
        recipe = run_stage7(output_dir=_stage_dir(output_dir, "reports"))
        result = {"recipe": recipe}

    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 1)
    logger.info(f"Stage {stage_num} completed in {elapsed:.1f}s")
    return result


def print_summary(output_dir: str):
    """Print final summary with recommended monomers."""
    out = Path(output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY — MIP Monomer Screening Results")
    logger.info("=" * 60)

    dft_data = {}
    selectivity_data = {}
    md_data = {}

    s2 = out / "stage2" / "stage2_dft.json"
    if s2.exists():
        with open(s2) as f:
            dft_data = json.load(f)

    s3csv = out / "stage3" / "stage3_selectivity.csv"
    if s3csv.exists():
        import pandas as pd
        df = pd.read_csv(s3csv)
        selectivity_data = (
            df.groupby("monomer")["avg_log_S"].mean()
            .to_dict()
        )

    s4 = out / "stage5" / "stage5_md.json"
    if s4.exists():
        with open(s4) as f:
            md_list = json.load(f)
        md_data = {r["monomer"]: r for r in md_list}

    s3top = out / "stage3" / "stage3_top.json"
    if s3top.exists():
        with open(s3top) as f:
            top_names = json.load(f)
    elif dft_data:
        avg_be = {}
        for m, solvs in dft_data.items():
            energies = [v["bsse_dE"] for v in solvs.values()]
            avg_be[m] = sum(energies) / len(energies) if energies else 0
        top_names = sorted(avg_be, key=avg_be.get)[:3]
    else:
        top_names = []

    logger.info(f"\nRecommended TOP-{len(top_names)} Monomers:")
    header = f"{'Rank':<6}{'Monomer':<12}{'Avg BE (kcal/mol)':<20}{'Avg log(S)':<14}{'EBN':<10}{'H-bonds':<10}"
    if md_data and any("optimal_ratio" in v for v in md_data.values()):
        header += f"{'Ratio':<8}"
    logger.info(header)
    logger.info("-" * len(header))

    for i, name in enumerate(top_names, 1):
        if name in dft_data:
            be_vals = [v["bsse_dE"] for v in dft_data[name].values()]
            avg_be = sum(be_vals) / len(be_vals) if be_vals else float("nan")
        else:
            avg_be = float("nan")

        avg_sel = selectivity_data.get(name, float("nan"))
        ebn = md_data.get(name, {}).get("EBN", float("nan"))
        n_hb = md_data.get(name, {}).get("n_hbonds_mean", md_data.get(name, {}).get("n_hbonds_stable", "N/A"))
        opt_ratio = md_data.get(name, {}).get("optimal_ratio", "")

        line = f"{i:<6}{name:<12}{avg_be:<+20.3f}{avg_sel:<14.3f}{ebn:<10.3f}{str(n_hb):<10}"
        if opt_ratio:
            line += f"1:{opt_ratio}"
        logger.info(line)

    logger.info("=" * 60)


def main():
    args = parse_args()

    # ── Multi-target driver ──────────────────────────────────────────
    # Run the full pipeline for every entry in config.TEMPLATES, each in its own
    # subprocess. Subprocess isolation is required because TEMPLATE_NAME/SMILES
    # are module-level config constants imported across the codebase — a fresh
    # process re-imports config with MIP_TEMPLATE set to the right target.
    if args.all_templates:
        import os as _os
        import subprocess as _sp
        from .config import TEMPLATES
        passthrough = [a for a in sys.argv[1:] if a != "--all-templates"]
        summary = {}
        for name in TEMPLATES:
            print("\n" + "█" * 60)
            print(f"█  TARGET: {name}  → results/{name.replace(' ', '_')}/")
            print("█" * 60, flush=True)
            env = {**_os.environ, "MIP_TEMPLATE": name}
            proc = _sp.run([sys.executable, _os.path.abspath(__file__)] + passthrough,
                           env=env)
            summary[name] = proc.returncode
        print("\n=== Multi-target summary ===")
        for name, code in summary.items():
            print(f"  {name:20s} {'OK' if code == 0 else f'FAILED (rc={code})'}")
        sys.exit(0 if all(c == 0 for c in summary.values()) else 1)

    # Default output dir = <results>/<TEMPLATE_NAME>, with stage1..stage7 nested.
    if args.output_dir:
        out_dir = args.output_dir
    else:
        from .config import OUTPUT_DIR as _ROOT, TEMPLATE_NAME
        safe_name = TEMPLATE_NAME.strip().replace(" ", "_").replace("/", "_")
        out_dir = str(Path(_ROOT) / safe_name)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "reports").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Pipeline] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(Path(out_dir) / "reports" / "pipeline.log")),
        ],
    )

    from .config import TEMPLATE_SMILES
    template = args.template or TEMPLATE_SMILES

    # ── Feature-only modes ───────────────────────────────────────────
    if args.crosslinker:
        from .crosslinker import run_crosslinker_screening
        run_crosslinker_screening(template_smiles=template,
                                  output_dir=_stage_dir(out_dir, "features"))
        return

    if args.suggest_interferents or args.auto_interferents:
        from .suggest_interferents import suggest_interferents
        suggest_interferents(template_smiles=template,
                             output_dir=_stage_dir(out_dir, "features"))
        return

    if args.predict_if:
        from .predict_if import train_if_model
        train_if_model(output_dir=_stage_dir(out_dir, "features"))
        return

    if args.update_if_model:
        if not args.monomer or args.experimental_if is None:
            logger.error("--update-if-model requires --monomer and --experimental-if")
            return
        from .predict_if import update_model
        update_model(args.monomer, args.experimental_if,
                     output_dir=_stage_dir(out_dir, "features"))
        return

    if args.report:
        from .generate_report import generate_report
        path = generate_report(output_dir=out_dir)
        logger.info(f"Report generated: {path}")
        return

    # ── Main pipeline ────────────────────────────────────────────────
    logger.info(f"Template SMILES: {template}")
    logger.info(f"Output directory: {out_dir}")

    t_total = time.time()
    timings = {}
    prev_results = {}

    if args.stage == "all":
        stages = [1, 2, 3, 4, 5, 6, 7]
    else:
        stages = [int(args.stage)]

    for stage_num in stages:
        logger.info(f"\n{'='*20} STAGE {stage_num} {'='*20}")
        # Each stage resumes at the item level (per monomer / solvent pair);
        # already-computed work is skipped, remaining/new work is done.
        result = run_stage(stage_num, template, out_dir, prev_results,
                           force=args.force)
        timings[f"stage{stage_num}"] = result["elapsed_s"]
        prev_results.update(result)

    total_time = time.time() - t_total

    # Timing summary
    logger.info(f"\n--- Timing Summary ---")
    for stage_name, t in timings.items():
        logger.info(f"  {stage_name}: {t:.1f}s")
    logger.info(f"  Total: {total_time:.1f}s")

    # Cross-linker screening (if enabled in config)
    from .config import CROSSLINKER_SCREENING
    if args.stage == "all" and CROSSLINKER_SCREENING:
        logger.info(f"\n{'='*20} CROSS-LINKER {'='*20}")
        try:
            from .crosslinker import run_crosslinker_screening
            run_crosslinker_screening(template_smiles=template,
                                      output_dir=_stage_dir(out_dir, "features"))
        except Exception as e:
            logger.warning(f"Cross-linker screening failed: {e}")

    # Final summary
    print_summary(out_dir)

    # Auto-generate report
    if args.stage == "all":
        try:
            from .generate_report import generate_report
            path = generate_report(output_dir=out_dir)
            logger.info(f"Report: {path}")
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")


if __name__ == "__main__":
    main()

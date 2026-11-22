from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STAGES = {
    "prepare": [["prepare_data.py"]],
    "baseline": [["run_baseline.py"]],
    "advanced": [["run_advanced_baselines.py"]],
    "feature_ablation": [["run_feature_ablation.py"]],
    "measurement_sensitivity": [["run_repeated_measurement_sensitivity.py"]],
    "interaction_ablation": [["run_interaction_feature_ablation.py"]],
    "feature_importance": [["run_feature_importance.py"]],
    "feature_selection": [["run_fold_feature_selection.py"]],
    "seed_ensemble": [["run_seed_ensemble.py"]],
    "enhanced": [["run_enhanced_models.py"]],
    "extra_splits": [["make_extra_splits.py"]],
    "split_stability": [["run_split_stability.py"]],
    "error_analysis": [["run_error_analysis.py"]],
    "solvent_ranking": [["run_solvent_ranking_validation.py"]],
    "nested_tuning": [["run_nested_tuning.py"]],
    "robustness": [["run_robustness_experiments.py"]],
    "conformal": [["run_conformal.py"]],
    "report_assets": [["build_report_assets.py"]],
    "package": [["package_code.py"]],
    "mlp": [["run_mlp_experiments.py"]],
    "finalize": [
        ["summarize_ensemble.py"],
        ["train_final.py"],
        ["make_figures.py"],
    ],
    "check": [["self_check.py"], ["-m", "pytest"]],
}

PIPELINES = {
    "core": [
        "prepare",
        "baseline",
        "advanced",
        "feature_ablation",
        "interaction_ablation",
        "measurement_sensitivity",
        "enhanced",
        "extra_splits",
        "split_stability",
        "robustness",
        "conformal",
        "finalize",
        "error_analysis",
        "solvent_ranking",
        "feature_importance",
        "seed_ensemble",
        "report_assets",
        "package",
        "check",
    ],
    "research": [
        "prepare",
        "baseline",
        "advanced",
        "feature_ablation",
        "interaction_ablation",
        "measurement_sensitivity",
        "enhanced",
        "extra_splits",
        "split_stability",
        "robustness",
        "conformal",
        "mlp",
        "finalize",
        "error_analysis",
        "solvent_ranking",
        "nested_tuning",
        "feature_importance",
        "feature_selection",
        "seed_ensemble",
        "report_assets",
        "package",
        "check",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one project stage without leaving PyCharm."
    )
    parser.add_argument(
        "target",
        choices=[*STAGES, *PIPELINES],
        help="Stage or named pipeline to execute",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = PIPELINES.get(args.target, [args.target])
    for stage in stages:
        print(f"\n=== Running {stage} ===", flush=True)
        for command_args in STAGES[stage]:
            command = [sys.executable, *command_args]
            subprocess.run(command, cwd=ROOT, check=True)
    print("\nPipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()

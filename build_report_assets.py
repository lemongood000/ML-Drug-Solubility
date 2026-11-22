from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "report_assets"


def markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    headers = [str(column) for column in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(RESULTS_DIR / "baseline_summary.csv")
    advanced = pd.read_csv(RESULTS_DIR / "advanced_baseline_summary.csv")
    enhanced = pd.read_csv(RESULTS_DIR / "enhanced_model_summary.csv")
    nested = pd.read_csv(RESULTS_DIR / "nested_tuning_summary.csv")
    splits = pd.read_csv(RESULTS_DIR / "split_stability_summary.csv")
    ablation = pd.read_csv(RESULTS_DIR / "feature_ablation_summary.csv")
    robustness = pd.read_csv(RESULTS_DIR / "robustness_summary.csv")
    repeated = pd.read_csv(RESULTS_DIR / "repeated_measurement_sensitivity.csv")
    interaction_ablation = pd.read_csv(RESULTS_DIR / "interaction_feature_ablation.csv")
    importance = pd.read_csv(RESULTS_DIR / "feature_importance.csv")
    feature_selection = pd.read_csv(RESULTS_DIR / "fold_feature_selection_summary.csv")
    seed_ensemble = pd.read_csv(RESULTS_DIR / "seed_ensemble_summary.csv")
    conformal_metrics = pd.read_csv(RESULTS_DIR / "conformal_metrics.csv")
    ranking_summary = json.loads(
        (RESULTS_DIR / "ranking_validation" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    error_summary = json.loads(
        (RESULTS_DIR / "error_analysis" / "report.json").read_text(encoding="utf-8")
    )

    model_rows = []
    for source_name, frame in (
        ("baseline", baseline),
        ("advanced", advanced),
        ("enhanced", enhanced),
    ):
        for row in frame.to_dict("records"):
            model_rows.append(
                {
                    "family": source_name,
                    "model": row["model"],
                    "rmse_mean": row["rmse_mean"],
                    "rmse_std": row["rmse_std"],
                    "mae_mean": row["mae_mean"],
                    "r2_mean": row["r2_mean"],
                }
            )
    model_comparison = pd.DataFrame(model_rows).sort_values("rmse_mean")
    model_comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    splits.to_csv(OUTPUT_DIR / "split_stability.csv", index=False)
    ablation.to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)
    robustness.to_csv(OUTPUT_DIR / "robustness.csv", index=False)
    repeated.to_csv(OUTPUT_DIR / "repeated_measurement_sensitivity.csv", index=False)
    interaction_ablation.to_csv(
        OUTPUT_DIR / "interaction_feature_ablation.csv", index=False
    )
    importance.head(50).to_csv(OUTPUT_DIR / "feature_importance_top50.csv", index=False)
    feature_selection.to_csv(OUTPUT_DIR / "fold_feature_selection.csv", index=False)
    seed_ensemble.to_csv(OUTPUT_DIR / "seed_ensemble.csv", index=False)
    conformal_metrics.to_csv(OUTPUT_DIR / "conformal_by_fold.csv", index=False)

    sections = [
        "# Report Tables",
        "## Model comparison",
        markdown_table(
            model_comparison[
                ["family", "model", "rmse_mean", "rmse_std", "mae_mean", "r2_mean"]
            ]
        ),
        "## Split stability and leakage diagnostics",
        markdown_table(
            splits[
                [
                    "split_kind",
                    "split_seed",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "r2_mean",
                ]
            ]
        ),
        "## Feature-block ablation",
        markdown_table(
            ablation[
                [
                    "feature_set",
                    "feature_count",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "r2_mean",
                ]
            ]
        ),
        "## Incremental chemistry features beyond MACCS",
        markdown_table(
            interaction_ablation[
                [
                    "feature_set",
                    "feature_count",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "r2_mean",
                ]
            ]
        ),
        "## Repeated-measurement aggregation sensitivity",
        markdown_table(
            repeated[
                [
                    "aggregation",
                    "rows",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "r2_mean",
                ]
            ]
        ),
        "## Top fold-wise XGBoost gain features",
        markdown_table(importance.head(30)),
        "## Fold-internal feature-selection comparison",
        markdown_table(feature_selection),
        "## Multi-seed XGBoost comparison",
        markdown_table(seed_ensemble),
        "## Negative robustness results",
        markdown_table(
            robustness[
                [
                    "config",
                    "rmse_mean",
                    "rmse_std",
                    "mae_mean",
                    "r2_mean",
                    "conflict_rmse_mean",
                ]
            ]
        ),
        "## Nested tuning estimate",
        markdown_table(nested),
        "## Conformal interval coverage by fold",
        markdown_table(
            conformal_metrics[
                [
                    "fold",
                    "rmse",
                    "coverage_80",
                    "coverage_90",
                    "mean_width_80",
                    "mean_width_90",
                ]
            ]
        ),
        "## Solvent-ranking validation at 25 C",
        "\n".join(f"- {key}: {value}" for key, value in ranking_summary.items()),
        "## Error and applicability-domain findings",
        "\n".join(f"- {key}: {value}" for key, value in error_summary.items()),
    ]
    (OUTPUT_DIR / "tables.md").write_text("\n\n".join(sections), encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "best_single_models": model_comparison.head(5).to_dict("records"),
                "split_stability": splits.to_dict("records"),
                "nested_tuning": nested.to_dict("records"),
                "repeated_measurement_sensitivity": repeated.to_dict("records"),
                "interaction_feature_ablation": (
                    interaction_ablation.to_dict("records")
                ),
                "top_feature_importance": importance.head(30).to_dict("records"),
                "fold_feature_selection": feature_selection.to_dict("records"),
                "seed_ensemble": seed_ensemble.to_dict("records"),
                "solvent_ranking": ranking_summary,
                "error_analysis": error_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    figure_names = sorted(path.name for path in FIGURES_DIR.glob("*.png"))
    (OUTPUT_DIR / "figure_index.md").write_text(
        "# Figure Index\n\n" + "\n".join(f"- {name}" for name in figure_names),
        encoding="utf-8",
    )
    print(f"Report tables written to {OUTPUT_DIR}")
    print(f"Integrated figures: {len(figure_names)}")


if __name__ == "__main__":
    main()

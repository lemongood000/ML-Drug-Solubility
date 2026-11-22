from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import regression_metrics
from src.oof import latest_oof_run, write_oof_artifact
from src.provenance import build_manifest, sha256_file


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def finite_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    quantile_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, quantile_level, method="higher"))


def main() -> None:
    source_run = latest_oof_run(RESULTS_DIR / "oof", "enhanced-main")
    predictions = pd.read_csv(source_run / "predictions.csv", encoding="utf-8-sig")
    model_columns = [
        "hist_gradient_boosting",
        "xgboost",
        "random_forest",
    ]
    pivot = predictions.pivot(
        index=["row_id", "fold", "y_true"],
        columns="model",
        values="y_pred",
    ).reset_index()
    missing_models = set(model_columns).difference(pivot.columns)
    if missing_models:
        raise ValueError(
            f"Enhanced OOF run is missing models: {sorted(missing_models)}"
        )
    pivot["y_pred"] = pivot[model_columns].mean(axis=1)

    fold_rows = []
    prediction_frames = []
    for outer_fold in sorted(pivot["fold"].unique()):
        calibration_mask = pivot["fold"] != outer_fold
        test_mask = pivot["fold"] == outer_fold
        scores = np.abs(
            pivot.loc[calibration_mask, "y_true"].to_numpy()
            - pivot.loc[calibration_mask, "y_pred"].to_numpy()
        )
        center = pivot.loc[test_mask, "y_pred"].to_numpy()
        radius_80 = finite_quantile(scores, 0.80)
        radius_90 = finite_quantile(scores, 0.90)
        y_test = pivot.loc[test_mask, "y_true"].to_numpy()
        metrics = regression_metrics(y_test, center)
        fold_rows.append(
            {
                "fold": int(outer_fold),
                **metrics,
                "coverage_80": float(np.mean(np.abs(y_test - center) <= radius_80)),
                "coverage_90": float(np.mean(np.abs(y_test - center) <= radius_90)),
                "mean_width_80": 2 * radius_80,
                "mean_width_90": 2 * radius_90,
                "radius_80": radius_80,
                "radius_90": radius_90,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_id": pivot.loc[test_mask, "row_id"].to_numpy(),
                    "fold": int(outer_fold),
                    "y_true": y_test,
                    "y_pred": center,
                    "lower_80": center - radius_80,
                    "upper_80": center + radius_80,
                    "lower_90": center - radius_90,
                    "upper_90": center + radius_90,
                }
            )
        )
        print(
            f"outer {outer_fold}: RMSE={metrics['rmse']:.4f} "
            f"coverage80={fold_rows[-1]['coverage_80']:.3f} "
            f"coverage90={fold_rows[-1]['coverage_90']:.3f}",
            flush=True,
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(RESULTS_DIR / "conformal_metrics.csv", index=False)
    all_scores = np.abs(pivot["y_true"].to_numpy() - pivot["y_pred"].to_numpy())
    calibration = {
        "method": "grouped_cross_fitted_empirical_ensemble_interval",
        "calibration_scores": len(all_scores),
        "source_oof_run": source_run.name,
        "radius_80": finite_quantile(all_scores, 0.80),
        "radius_90": finite_quantile(all_scores, 0.90),
        "mean_coverage_80": float(fold_metrics["coverage_80"].mean()),
        "mean_coverage_90": float(fold_metrics["coverage_90"].mean()),
        "mean_width_80": float(fold_metrics["mean_width_80"].mean()),
        "mean_width_90": float(fold_metrics["mean_width_90"].mean()),
    }
    (RESULTS_DIR / "conformal_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    oof = pd.concat(prediction_frames, ignore_index=True)
    oof.to_csv(RESULTS_DIR / "conformal_oof.csv", index=False)
    manifest = build_manifest(
        {
            "experiment": "grouped_cross_fitted_empirical_interval",
            "clean_sha256": sha256_file(RESULTS_DIR / "clean.csv"),
            "folds_sha256": sha256_file(RESULTS_DIR / "folds.json"),
            "source_oof_manifest": source_run.name,
            "feature_config": "all_enhanced_features_with_maccs",
            "include_maccs": True,
            "coverage_levels": [0.8, 0.9],
        }
    )
    write_oof_artifact(
        RESULTS_DIR / "oof",
        "empirical-interval",
        manifest,
        {"cross_fitted_ensemble": oof[["row_id", "fold", "y_true", "y_pred"]]},
        seed=42,
    )
    print("\nEmpirical interval calibration")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()

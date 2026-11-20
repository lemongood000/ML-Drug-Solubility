from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.oof import write_oof_artifact
from src.provenance import build_manifest, sha256_file
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def make_models(seed: int = 42) -> dict[str, object]:
    return {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "xgboost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_features=0.7,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
    }


def finite_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    quantile_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, quantile_level, method="higher"))


def main() -> None:
    frame = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    assignment = load_folds(frame, RESULTS_DIR / "folds.json")
    features, _ = build_feature_matrix(
        frame,
        include_solvent_identity=True,
        include_physchem_interactions=True,
        include_temperature_interactions=True,
    )
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
    groups = frame["drug_smiles_canonical"].to_numpy()
    inner_cv = GroupKFold(n_splits=5)
    fold_rows = []
    prediction_frames = []
    all_calibration_scores = []

    for outer_fold in sorted(np.unique(assignment)):
        outer_train = np.flatnonzero(assignment != outer_fold)
        outer_test = np.flatnonzero(assignment == outer_fold)
        calibration_scores = []
        test_predictions = []
        for inner_train_pos, calibration_pos in inner_cv.split(
            outer_train, groups=groups[outer_train]
        ):
            inner_train = outer_train[inner_train_pos]
            calibration = outer_train[calibration_pos]
            model_predictions = {"calibration": [], "test": []}
            for model in make_models().values():
                model.fit(features.iloc[inner_train], target[inner_train])
                model_predictions["calibration"].append(
                    model.predict(features.iloc[calibration])
                )
                model_predictions["test"].append(
                    model.predict(features.iloc[outer_test])
                )
            calibration_prediction = np.column_stack(
                model_predictions["calibration"]
            ).mean(axis=1)
            test_prediction = np.column_stack(
                model_predictions["test"]
            ).mean(axis=1)
            calibration_scores.extend(
                np.abs(target[calibration] - calibration_prediction).tolist()
            )
            test_predictions.append(test_prediction)
        scores = np.asarray(calibration_scores)
        all_calibration_scores.extend(scores.tolist())
        center = np.column_stack(test_predictions).mean(axis=1)
        radius_80 = finite_quantile(scores, 0.80)
        radius_90 = finite_quantile(scores, 0.90)
        y_test = target[outer_test]
        metrics = regression_metrics(y_test, center)
        fold_rows.append(
            {
                "fold": int(outer_fold),
                **metrics,
                "coverage_80": float(
                    np.mean(np.abs(y_test - center) <= radius_80)
                ),
                "coverage_90": float(
                    np.mean(np.abs(y_test - center) <= radius_90)
                ),
                "mean_width_80": 2 * radius_80,
                "mean_width_90": 2 * radius_90,
                "radius_80": radius_80,
                "radius_90": radius_90,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_id": frame.iloc[outer_test]["row_id"].to_numpy(),
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
    all_scores = np.asarray(all_calibration_scores)
    calibration = {
        "method": "5x5_grouped_cross_conformal_ensemble",
        "calibration_scores": len(all_scores),
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
            "experiment": "grouped_cross_conformal_ensemble",
            "clean_sha256": sha256_file(RESULTS_DIR / "clean.csv"),
            "folds_sha256": sha256_file(RESULTS_DIR / "folds.json"),
            "feature_config": "all_enhanced_features",
            "outer_folds": 5,
            "inner_folds": 5,
            "coverage_levels": [0.8, 0.9],
        }
    )
    write_oof_artifact(
        RESULTS_DIR / "oof",
        "conformal-ensemble",
        manifest,
        {
            "cvplus_ensemble": oof[
                ["row_id", "fold", "y_true", "y_pred"]
            ]
        },
        seed=42,
    )
    print("\nConformal calibration")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()

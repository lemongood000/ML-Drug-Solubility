from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.oof import write_oof_artifact
from src.provenance import build_manifest, sha256_file
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def make_xgb(huber: bool) -> XGBRegressor:
    params = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }
    if huber:
        params.update(
            {
                "objective": "reg:pseudohubererror",
                "huber_slope": 0.5,
            }
        )
    else:
        params["objective"] = "reg:squarederror"
    return XGBRegressor(**params)


def make_hgb(loss: str) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=42,
    )


CONFIGS = [
    {"name": "xgb_mse", "model": "xgb", "loss": "mse", "weighted": False},
    {"name": "xgb_huber", "model": "xgb", "loss": "huber", "weighted": False},
    {"name": "xgb_weighted_mse", "model": "xgb", "loss": "mse", "weighted": True},
    {
        "name": "xgb_weighted_huber",
        "model": "xgb",
        "loss": "huber",
        "weighted": True,
    },
    {"name": "hgb_mse", "model": "hgb", "loss": "squared_error", "weighted": False},
    {
        "name": "hgb_absolute",
        "model": "hgb",
        "loss": "absolute_error",
        "weighted": False,
    },
    {
        "name": "hgb_weighted_mse",
        "model": "hgb",
        "loss": "squared_error",
        "weighted": True,
    },
]


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
    sample_weight = 1.0 / (1.0 + frame["solubility_range"].to_numpy())
    sample_weight = np.clip(sample_weight, 0.1, 1.0)
    high_conflict = (frame["solubility_range"] > 0.05).to_numpy()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    config_predictions = {}

    for config in CONFIGS:
        fold_frames = []
        for fold in sorted(np.unique(assignment)):
            train_mask = assignment != fold
            test_mask = assignment == fold
            model = (
                make_xgb(config["loss"] == "huber")
                if config["model"] == "xgb"
                else make_hgb(config["loss"])
            )
            fit_kwargs = {}
            if config["weighted"]:
                fit_kwargs["sample_weight"] = sample_weight[train_mask]
            model.fit(features.loc[train_mask], target[train_mask], **fit_kwargs)
            prediction = model.predict(features.loc[test_mask])
            metrics = regression_metrics(target[test_mask], prediction)
            conflict_mask = high_conflict[test_mask]
            conflict_metrics = (
                regression_metrics(
                    target[test_mask][conflict_mask],
                    prediction[conflict_mask],
                )
                if conflict_mask.sum() >= 2
                else {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
            )
            metric_rows.append(
                {
                    "config": config["name"],
                    "fold": int(fold),
                    **metrics,
                    "conflict_rmse": conflict_metrics["rmse"],
                    "conflict_mae": conflict_metrics["mae"],
                    "conflict_rows": int(conflict_mask.sum()),
                }
            )
            fold_frames.append(
                pd.DataFrame(
                    {
                        "row_id": frame.loc[test_mask, "row_id"],
                        "fold": int(fold),
                        "y_true": target[test_mask],
                        "y_pred": prediction,
                    }
                )
            )
        oof = pd.concat(fold_frames, ignore_index=True)
        config_predictions[config["name"]] = oof
        oof.to_csv(
            PREDICTIONS_DIR / f"robust_{config['name']}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Completed {config['name']}", flush=True)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(RESULTS_DIR / "robustness_metrics.csv", index=False)
    summary = (
        metrics_frame.groupby("config")[
            ["rmse", "mae", "r2", "conflict_rmse", "conflict_mae"]
        ]
        .agg(["mean", "std"])
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary = summary.sort_values("rmse_mean").reset_index()
    summary.to_csv(RESULTS_DIR / "robustness_summary.csv", index=False)

    manifest = build_manifest(
        {
            "experiment": "noise_weighting_and_robust_loss",
            "clean_sha256": sha256_file(RESULTS_DIR / "clean.csv"),
            "folds_sha256": sha256_file(RESULTS_DIR / "folds.json"),
            "feature_config": "all_enhanced_features",
            "configs": CONFIGS,
            "weight_formula": "clip(1/(1+solubility_range),0.1,1.0)",
        }
    )
    oof_dir = write_oof_artifact(
        RESULTS_DIR / "oof",
        "robustness",
        manifest,
        config_predictions,
        seed=42,
    )
    print("\nRobustness summary")
    print(summary.to_string(index=False))
    print(f"OOF artifacts: {oof_dir}")


if __name__ == "__main__":
    main()

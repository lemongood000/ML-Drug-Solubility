from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.oof import write_oof_artifact
from src.provenance import build_manifest, sha256_file
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def make_models(seed: int) -> dict[str, object]:
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
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    model_predictions: dict[str, list[pd.DataFrame]] = {
        name: [] for name in make_models(42)
    }
    manifest = build_manifest(
        {
            "experiment": "enhanced_models",
            "clean_sha256": sha256_file(RESULTS_DIR / "clean.csv"),
            "folds_sha256": sha256_file(RESULTS_DIR / "folds.json"),
            "feature_config": {
                "drug_bits": 1024,
                "solvent_bits": 256,
                "solvent_identity": True,
                "physchem_interactions": True,
                "temperature_interactions": True,
            },
            "model_seed": 42,
            "xgboost_config": {
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 6,
                "min_child_weight": 5,
            },
        }
    )
    print(f"Enhanced feature matrix: {features.shape[1]} features", flush=True)

    for model_name, model in make_models(42).items():
        for fold in sorted(np.unique(assignment)):
            train_mask = assignment != fold
            test_mask = assignment == fold
            started = time.perf_counter()
            model.fit(features.loc[train_mask], target[train_mask])
            prediction = model.predict(features.loc[test_mask])
            elapsed = time.perf_counter() - started
            metrics = regression_metrics(target[test_mask], prediction)
            rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    **metrics,
                    "seconds": elapsed,
                }
            )
            prediction_frame = pd.DataFrame(
                {
                    "row_id": frame.loc[test_mask, "row_id"],
                    "fold": int(fold),
                    "drug_smiles_canonical": frame.loc[
                        test_mask, "drug_smiles_canonical"
                    ],
                    "y_true": target[test_mask],
                    "y_pred": prediction,
                    "residual": target[test_mask] - prediction,
                }
            )
            prediction_frame.to_csv(
                PREDICTIONS_DIR
                / f"enhanced_{model_name}_fold{fold}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            model_predictions[model_name].append(prediction_frame)
            print(
                f"enhanced_{model_name:24s} fold {fold}: "
                f"RMSE={metrics['rmse']:.4f} "
                f"MAE={metrics['mae']:.4f} "
                f"R2={metrics['r2']:.4f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(
        RESULTS_DIR / "enhanced_model_metrics.csv", index=False
    )
    summary = (
        metrics_frame.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .sort_values(("rmse", "mean"))
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(
        RESULTS_DIR / "enhanced_model_summary.csv", index=False
    )
    oof_dir = write_oof_artifact(
        RESULTS_DIR / "oof",
        "enhanced-main",
        manifest,
        {
            model: pd.concat(frames, ignore_index=True)
            for model, frames in model_predictions.items()
        },
        seed=42,
    )
    print("\nEnhanced model summary")
    print(summary.to_string())
    print(f"Canonical OOF artifacts: {oof_dir}")


if __name__ == "__main__":
    main()

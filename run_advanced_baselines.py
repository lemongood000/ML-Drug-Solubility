from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.provenance import build_manifest, sha256_file, write_json
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def make_model(name: str, seed: int):
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.7,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "svr_rbf":
        return make_pipeline(
            StandardScaler(),
            SVR(C=10.0, epsilon=0.1, gamma="scale", cache_size=2048),
        )
    if name == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=15,
                weights="distance",
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    seed = 42
    model_names = [
        "extra_trees",
        "hist_gradient_boosting",
        "svr_rbf",
        "knn",
    ]
    frame = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    folds_path = RESULTS_DIR / "folds.json"
    assignment = load_folds(frame, folds_path)
    features, _ = build_feature_matrix(frame)
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
    manifest = build_manifest(
        {
            "models": model_names,
            "seed": seed,
            "clean_sha256": sha256_file(RESULTS_DIR / "clean.csv"),
            "folds_sha256": sha256_file(folds_path),
            "drug_bits": 1024,
            "solvent_bits": 256,
        }
    )
    run_id = manifest["config_id"]
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    for model_name in model_names:
        for fold in sorted(np.unique(assignment)):
            train_mask = assignment != fold
            test_mask = assignment == fold
            started = time.perf_counter()
            model = make_model(model_name, seed)
            model.fit(features.loc[train_mask], target[train_mask])
            prediction = model.predict(features.loc[test_mask])
            elapsed = time.perf_counter() - started
            metrics = regression_metrics(target[test_mask], prediction)
            rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "run_id": run_id,
                    **metrics,
                    "seconds": elapsed,
                }
            )
            pd.DataFrame(
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
            ).to_csv(
                PREDICTIONS_DIR / f"{model_name}_fold{fold}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"{model_name:24s} fold {fold}: "
                f"RMSE={metrics['rmse']:.4f} "
                f"MAE={metrics['mae']:.4f} "
                f"R2={metrics['r2']:.4f} "
                f"({elapsed:.1f}s)"
            )

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(
        RESULTS_DIR / "advanced_baseline_metrics.csv", index=False
    )
    run_dir = RESULTS_DIR / "runs" / f"advanced-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "manifest.json", manifest)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    summary = (
        metrics_frame.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .sort_values(("rmse", "mean"))
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(
        RESULTS_DIR / "advanced_baseline_summary.csv", index=False
    )
    print("\nAdvanced baseline summary")
    print(summary.to_string())


if __name__ == "__main__":
    main()

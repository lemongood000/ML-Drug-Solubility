from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.provenance import build_manifest, sha256_file, write_json
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="mean,ridge,rf,xgb",
        help="Comma-separated subset of mean,ridge,rf,xgb",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drug-bits", type=int, default=1024, help="Morgan fingerprint size"
    )
    parser.add_argument(
        "--solvent-bits", type=int, default=256, help="Solvent fingerprint size"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=300, help="Trees for RF/XGBoost"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Parallel jobs for RF/XGBoost"
    )
    return parser.parse_args()


def make_model(name: str, args: argparse.Namespace):
    if name == "mean":
        return "mean"
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_features=0.7,
            min_samples_leaf=2,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if name == "xgb":
        return XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    args = parse_args()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    clean_path = RESULTS_DIR / "clean.csv"
    folds_path = RESULTS_DIR / "folds.json"
    identity_config = {
        "seed": args.seed,
        "drug_bits": args.drug_bits,
        "solvent_bits": args.solvent_bits,
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "clean_sha256": sha256_file(clean_path),
        "folds_sha256": sha256_file(folds_path),
    }
    manifest = build_manifest({**identity_config, "models": model_names})
    run_id = manifest["config_id"]

    frame = pd.read_csv(clean_path, encoding="utf-8-sig")
    assignment = load_folds(frame, folds_path)
    features, column_groups = build_feature_matrix(
        frame,
        drug_bits=args.drug_bits,
        solvent_bits=args.solvent_bits,
    )
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)

    print(
        f"Feature matrix: {features.shape[0]} rows x {features.shape[1]} columns "
        f"({len(column_groups['drug'])} drug, "
        f"{len(column_groups['solvent'])} solvent, "
        f"{len(column_groups['temperature'])} temperature)"
    )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float | int | str]] = []
    for model_name in model_names:
        for fold in sorted(np.unique(assignment)):
            started = time.perf_counter()
            train_mask = assignment != fold
            test_mask = assignment == fold
            x_train = features.loc[train_mask]
            x_test = features.loc[test_mask]
            y_train = target[train_mask]
            y_test = target[test_mask]

            model = make_model(model_name, args)
            if model == "mean":
                prediction = np.full(y_test.shape, y_train.mean(), dtype=float)
            else:
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)

            metrics = regression_metrics(y_test, prediction)
            elapsed = time.perf_counter() - started
            metric_rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
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
                    "y_true": y_test,
                    "y_pred": prediction,
                    "residual": y_test - prediction,
                }
            ).to_csv(
                PREDICTIONS_DIR / f"{model_name}_fold{fold}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"{model_name:5s} fold {fold}: "
                f"RMSE={metrics['rmse']:.4f} "
                f"MAE={metrics['mae']:.4f} "
                f"R2={metrics['r2']:.4f} "
                f"({elapsed:.1f}s)"
            )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame["run_id"] = run_id
    metrics_path = RESULTS_DIR / "baseline_metrics.csv"
    manifest_path = RESULTS_DIR / "baseline_manifest.json"
    if metrics_path.exists() and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = pd.read_csv(metrics_path)
        if previous_manifest["config_id"] == run_id:
            previous = previous.loc[~previous["model"].isin(model_names)]
            metrics_frame = pd.concat([previous, metrics_frame], ignore_index=True)
        else:
            print("Previous baseline results use a different configuration; discarded.")
    metrics_frame.to_csv(metrics_path, index=False)
    summary = (
        metrics_frame.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .sort_values(("rmse", "mean"))
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(
        RESULTS_DIR / "baseline_summary.csv", index=False
    )
    run_dir = RESULTS_DIR / "runs" / f"baseline-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    write_json(run_dir / "manifest.json", manifest)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    summary.reset_index().to_csv(run_dir / "summary.csv", index=False)
    print(f"Run manifest: {manifest_path}")
    print("\nCross-validation summary")
    print(summary.to_string())


if __name__ == "__main__":
    main()

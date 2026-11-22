from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.data import prepare_dataset, read_raw_data
from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.provenance import build_manifest, write_json
from src.splits import make_balanced_group_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def main() -> None:
    raw = read_raw_data(ROOT / "data_2026.csv")
    summary_rows = []
    fold_rows = []
    for method in ("median", "mean", "none"):
        frame, audit = prepare_dataset(raw, aggregate_method=method)
        assignment = make_balanced_group_folds(
            frame,
            group_column="drug_smiles_canonical",
            n_splits=5,
            seed=42,
        )
        features, _ = build_feature_matrix(
            frame,
            include_solvent_identity=True,
            include_physchem_interactions=True,
            include_temperature_interactions=True,
            include_maccs=True,
        )
        target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
        method_fold_rows = []
        for fold in sorted(np.unique(assignment)):
            train_mask = assignment != fold
            test_mask = assignment == fold
            model = XGBRegressor(
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
                random_state=42,
            )
            model.fit(
                features.loc[train_mask],
                target[train_mask],
            )
            prediction = model.predict(features.loc[test_mask])
            metrics = regression_metrics(target[test_mask], prediction)
            method_fold_rows.append(
                {
                    "aggregation": method,
                    "fold": int(fold),
                    "rows": len(frame),
                    **metrics,
                }
            )
        current = pd.DataFrame(method_fold_rows)
        fold_rows.append(current)
        row = {
            "aggregation": method,
            "rows": len(frame),
            "audit_rows_after_aggregation": audit.rows_after_aggregation,
            "rmse_mean": current["rmse"].mean(),
            "rmse_std": current["rmse"].std(),
            "mae_mean": current["mae"].mean(),
            "r2_mean": current["r2"].mean(),
        }
        summary_rows.append(row)
        print(row, flush=True)

    summary = pd.DataFrame(summary_rows).sort_values("rmse_mean")
    summary.to_csv(
        RESULTS_DIR / "repeated_measurement_sensitivity.csv",
        index=False,
    )
    pd.concat(fold_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "repeated_measurement_sensitivity_by_fold.csv",
        index=False,
    )
    manifest = build_manifest(
        {
            "experiment": "repeated_measurement_aggregation_sensitivity",
            "aggregation_methods": ["median", "mean", "none"],
            "feature_config": "all_enhanced_features_with_maccs",
            "model": "xgboost",
            "model_seed": 42,
            "split_group": "drug_smiles_canonical",
            "split_seed": 42,
        }
    )
    write_json(
        RESULTS_DIR / "repeated_measurement_sensitivity_manifest.json",
        manifest,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

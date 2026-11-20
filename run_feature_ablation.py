from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

CONFIGS = [
    {
        "name": "base",
        "include_solvent_identity": False,
        "include_physchem_interactions": False,
        "include_temperature_interactions": False,
    },
    {
        "name": "solvent_identity",
        "include_solvent_identity": True,
        "include_physchem_interactions": False,
        "include_temperature_interactions": False,
    },
    {
        "name": "physchem_interactions",
        "include_solvent_identity": False,
        "include_physchem_interactions": True,
        "include_temperature_interactions": False,
    },
    {
        "name": "temperature_interactions",
        "include_solvent_identity": False,
        "include_physchem_interactions": False,
        "include_temperature_interactions": True,
    },
    {
        "name": "all_interactions",
        "include_solvent_identity": False,
        "include_physchem_interactions": True,
        "include_temperature_interactions": True,
    },
    {
        "name": "all_enhanced_features",
        "include_solvent_identity": True,
        "include_physchem_interactions": True,
        "include_temperature_interactions": True,
    },
]


def main() -> None:
    frame = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    assignment = load_folds(frame, RESULTS_DIR / "folds.json")
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
    rows = []
    summaries = []

    for config in CONFIGS:
        features, _ = build_feature_matrix(
            frame,
            include_solvent_identity=config["include_solvent_identity"],
            include_physchem_interactions=config[
                "include_physchem_interactions"
            ],
            include_temperature_interactions=config[
                "include_temperature_interactions"
            ],
        )
        print(f"{config['name']}: {features.shape[1]} features", flush=True)
        current_rows = []
        for fold in sorted(np.unique(assignment)):
            train_mask = assignment != fold
            test_mask = assignment == fold
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.04,
                max_depth=6,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(features.loc[train_mask], target[train_mask])
            prediction = model.predict(features.loc[test_mask])
            row = {
                "feature_set": config["name"],
                "fold": int(fold),
                "feature_count": features.shape[1],
                **regression_metrics(target[test_mask], prediction),
            }
            rows.append(row)
            current_rows.append(row)
            print(
                f"  fold {fold}: RMSE={row['rmse']:.4f} R2={row['r2']:.4f}",
                flush=True,
            )
        current = pd.DataFrame(current_rows)
        summaries.append(
            {
                "feature_set": config["name"],
                "feature_count": features.shape[1],
                "rmse_mean": current["rmse"].mean(),
                "rmse_std": current["rmse"].std(),
                "mae_mean": current["mae"].mean(),
                "r2_mean": current["r2"].mean(),
            }
        )

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "feature_ablation_metrics.csv", index=False
    )
    summary = pd.DataFrame(summaries).sort_values("rmse_mean")
    summary.to_csv(RESULTS_DIR / "feature_ablation_summary.csv", index=False)
    print("\nFeature ablation summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

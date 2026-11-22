from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.provenance import build_manifest, write_json
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SEEDS = (42, 142, 242, 342, 442)


def main() -> None:
    frame = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    assignment = load_folds(frame, RESULTS_DIR / "folds.json")
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
    features, _ = build_feature_matrix(
        frame,
        include_solvent_identity=True,
        include_physchem_interactions=True,
        include_temperature_interactions=True,
        include_maccs=True,
    )

    fold_rows = []
    for fold in sorted(np.unique(assignment)):
        train_mask = assignment != fold
        test_mask = assignment == fold
        predictions = []
        for seed in SEEDS:
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
                random_state=seed,
            )
            model.fit(features.loc[train_mask], target[train_mask])
            predictions.append(model.predict(features.loc[test_mask]))
        for count in (1, 3, 5):
            prediction = np.column_stack(predictions[:count]).mean(axis=1)
            fold_rows.append(
                {
                    "fold": int(fold),
                    "seeds_used": count,
                    **regression_metrics(target[test_mask], prediction),
                }
            )
        print(f"Fold {fold} seed ensemble complete", flush=True)

    folded = pd.DataFrame(fold_rows)
    folded.to_csv(
        RESULTS_DIR / "seed_ensemble_by_fold.csv",
        index=False,
    )
    summary = folded.groupby("seeds_used")[["rmse", "mae", "r2"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index().sort_values("seeds_used")
    summary.to_csv(
        RESULTS_DIR / "seed_ensemble_summary.csv",
        index=False,
    )
    manifest = build_manifest(
        {
            "experiment": "multi_seed_xgboost_ensemble",
            "feature_config": "all_enhanced_features_with_maccs",
            "seeds": list(SEEDS),
            "reported_seed_counts": [1, 3, 5],
            "split_group": "drug_smiles_canonical",
        }
    )
    write_json(
        RESULTS_DIR / "seed_ensemble_manifest.json",
        manifest,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

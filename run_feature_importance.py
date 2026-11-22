from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.provenance import build_manifest, write_json
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


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

    rows = []
    for fold in sorted(np.unique(assignment)):
        train_mask = assignment != fold
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
        rows.append(
            pd.DataFrame(
                {
                    "feature": features.columns,
                    "fold": int(fold),
                    "gain": model.feature_importances_,
                }
            )
        )
        print(f"Fold {fold} feature importance complete", flush=True)

    folded = pd.concat(rows, ignore_index=True)
    folded.to_csv(
        RESULTS_DIR / "feature_importance_by_fold.csv",
        index=False,
    )
    summary = (
        folded.groupby("feature")["gain"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index=False,
    )
    manifest = build_manifest(
        {
            "experiment": "fold_xgboost_gain_importance",
            "feature_config": "all_enhanced_features_with_maccs",
            "importance_type": "gain",
            "model": "xgboost",
            "model_seed": 42,
            "split_group": "drug_smiles_canonical",
        }
    )
    write_json(
        RESULTS_DIR / "feature_importance_manifest.json",
        manifest,
    )
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

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
FEATURE_COUNTS = (200, 500, 1000, 1500, 1698)


def make_model(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
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
    summary_rows = []
    for fold in sorted(np.unique(assignment)):
        train_mask = assignment != fold
        test_mask = assignment == fold
        x_train = features.loc[train_mask]
        x_test = features.loc[test_mask]

        selector = make_model()
        selector.fit(x_train, target[train_mask])
        ranked_features = (
            pd.Series(selector.feature_importances_, index=features.columns)
            .sort_values(ascending=False)
            .index
        )
        for feature_count in FEATURE_COUNTS:
            if feature_count == features.shape[1]:
                prediction = selector.predict(x_test)
            else:
                selected = ranked_features[:feature_count]
                model = make_model()
                model.fit(x_train[selected], target[train_mask])
                prediction = model.predict(x_test[selected])
            row = {
                "feature_count": feature_count,
                "fold": int(fold),
                **regression_metrics(target[test_mask], prediction),
            }
            fold_rows.append(row)
        print(f"Fold {fold} feature selection complete", flush=True)

    folded = pd.DataFrame(fold_rows)
    folded.to_csv(
        RESULTS_DIR / "fold_feature_selection_by_fold.csv",
        index=False,
    )
    summary = folded.groupby("feature_count")[["rmse", "mae", "r2"]].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary = summary.reset_index().sort_values("rmse_mean")
    summary.to_csv(
        RESULTS_DIR / "fold_feature_selection_summary.csv",
        index=False,
    )
    manifest = build_manifest(
        {
            "experiment": "fold_internal_gain_feature_selection",
            "feature_config": "all_enhanced_features_with_maccs",
            "feature_counts": list(FEATURE_COUNTS),
            "selection_scope": "training_fold_only",
            "model": "xgboost",
            "model_seed": 42,
            "split_group": "drug_smiles_canonical",
        }
    )
    write_json(
        RESULTS_DIR / "fold_feature_selection_manifest.json",
        manifest,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

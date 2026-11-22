from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.oof import write_oof_artifact
from src.provenance import build_manifest, sha256_file
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SPLITS_DIR = RESULTS_DIR / "splits"

SPLITS = [
    ("drug", 42, RESULTS_DIR / "folds.json"),
    ("drug", 1729, SPLITS_DIR / "folds_drug_1729.json"),
    ("drug", 2026, SPLITS_DIR / "folds_drug_2026.json"),
    ("scaffold", 42, SPLITS_DIR / "folds_scaffold_42.json"),
    ("solvent", 42, SPLITS_DIR / "folds_solvent_42.json"),
    ("pair", 42, SPLITS_DIR / "folds_pair_42.json"),
    ("random", 42, SPLITS_DIR / "folds_random_42.json"),
]


def main() -> None:
    frame = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    features, _ = build_feature_matrix(
        frame,
        include_solvent_identity=True,
        include_physchem_interactions=True,
        include_temperature_interactions=True,
        include_maccs=True,
    )
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
    summary_rows = []

    for split_kind, split_seed, fold_path in SPLITS:
        assignment = load_folds(frame, fold_path)
        fold_frames = []
        fold_rows = []
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
            model.fit(features.loc[train_mask], target[train_mask])
            prediction = model.predict(features.loc[test_mask])
            metrics = regression_metrics(target[test_mask], prediction)
            fold_rows.append({"fold": int(fold), **metrics})
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
        fold_metrics = pd.DataFrame(fold_rows)
        summary_rows.append(
            {
                "split_kind": split_kind,
                "split_seed": split_seed,
                "rmse_mean": fold_metrics["rmse"].mean(),
                "rmse_std": fold_metrics["rmse"].std(),
                "mae_mean": fold_metrics["mae"].mean(),
                "r2_mean": fold_metrics["r2"].mean(),
            }
        )
        oof = pd.concat(fold_frames, ignore_index=True)
        manifest = build_manifest(
            {
                "experiment": "split_stability",
                "split_kind": split_kind,
                "split_seed": split_seed,
                "clean_sha256": sha256_file(RESULTS_DIR / "clean.csv"),
                "folds_sha256": sha256_file(fold_path),
                "feature_config": "all_enhanced_features",
                "include_maccs": True,
                "model": "xgboost",
                "model_seed": 42,
            }
        )
        oof_dir = write_oof_artifact(
            RESULTS_DIR / "oof",
            f"split-{split_kind}-{split_seed}",
            manifest,
            {"xgboost": oof},
            seed=42,
        )
        print(
            f"{split_kind:6s} seed={split_seed}: "
            f"RMSE={fold_metrics['rmse'].mean():.4f} "
            f"R2={fold_metrics['r2'].mean():.4f} "
            f"-> {oof_dir.name}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "split_stability_summary.csv", index=False)
    print("\nSplit stability")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

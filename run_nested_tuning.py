from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.oof import write_oof_artifact
from src.provenance import build_manifest, sha256_file
from src.splits import load_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

CONFIGS = [
    {
        "name": "d4_lr004_l300",
        "max_depth": 4,
        "min_child_weight": 2,
        "learning_rate": 0.04,
        "n_estimators": 300,
    },
    {
        "name": "d6_lr004_l300",
        "max_depth": 6,
        "min_child_weight": 2,
        "learning_rate": 0.04,
        "n_estimators": 300,
    },
    {
        "name": "d8_lr004_l300",
        "max_depth": 8,
        "min_child_weight": 4,
        "learning_rate": 0.04,
        "n_estimators": 300,
    },
    {
        "name": "d4_lr003_l500",
        "max_depth": 4,
        "min_child_weight": 1,
        "learning_rate": 0.03,
        "n_estimators": 500,
    },
    {
        "name": "d6_lr003_l500",
        "max_depth": 6,
        "min_child_weight": 5,
        "learning_rate": 0.03,
        "n_estimators": 500,
    },
    {
        "name": "d8_lr003_l500",
        "max_depth": 8,
        "min_child_weight": 5,
        "learning_rate": 0.03,
        "n_estimators": 500,
    },
]


def make_model(config: dict, seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
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
    features, _ = build_feature_matrix(
        frame,
        include_solvent_identity=True,
        include_physchem_interactions=True,
        include_temperature_interactions=True,
        include_maccs=True,
    )
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
    groups = frame["drug_smiles_canonical"].to_numpy()
    inner_cv = GroupKFold(n_splits=3)
    choices = []
    outer_metrics = []
    outer_predictions = []

    for outer_fold in sorted(np.unique(assignment)):
        outer_train = np.flatnonzero(assignment != outer_fold)
        outer_test = np.flatnonzero(assignment == outer_fold)
        tuning_rows = []
        for config in CONFIGS:
            inner_scores = []
            for inner_train_pos, inner_valid_pos in inner_cv.split(
                outer_train,
                groups=groups[outer_train],
            ):
                inner_train = outer_train[inner_train_pos]
                inner_valid = outer_train[inner_valid_pos]
                model = make_model(config)
                model.fit(features.iloc[inner_train], target[inner_train])
                prediction = model.predict(features.iloc[inner_valid])
                inner_scores.append(
                    regression_metrics(target[inner_valid], prediction)["rmse"]
                )
            tuning_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "config": config["name"],
                    "inner_rmse_mean": float(np.mean(inner_scores)),
                    "inner_rmse_std": float(np.std(inner_scores)),
                }
            )
        tuning_frame = pd.DataFrame(tuning_rows).sort_values("inner_rmse_mean")
        selected_name = str(tuning_frame.iloc[0]["config"])
        selected_config = next(
            config for config in CONFIGS if config["name"] == selected_name
        )
        model = make_model(selected_config)
        model.fit(features.iloc[outer_train], target[outer_train])
        prediction = model.predict(features.iloc[outer_test])
        metrics = regression_metrics(target[outer_test], prediction)
        outer_metrics.append({"fold": int(outer_fold), **metrics})
        choices.extend(tuning_rows)
        outer_predictions.append(
            pd.DataFrame(
                {
                    "row_id": frame.iloc[outer_test]["row_id"].to_numpy(),
                    "fold": int(outer_fold),
                    "y_true": target[outer_test],
                    "y_pred": prediction,
                }
            )
        )
        print(
            f"outer {outer_fold}: selected={selected_name} "
            f"RMSE={metrics['rmse']:.4f}",
            flush=True,
        )

    metrics_frame = pd.DataFrame(outer_metrics)
    choices_frame = pd.DataFrame(choices)
    metrics_frame.to_csv(RESULTS_DIR / "nested_tuning_metrics.csv", index=False)
    choices_frame.to_csv(RESULTS_DIR / "nested_tuning_choices.csv", index=False)
    summary = {
        "rmse_mean": float(metrics_frame["rmse"].mean()),
        "rmse_std": float(metrics_frame["rmse"].std()),
        "mae_mean": float(metrics_frame["mae"].mean()),
        "r2_mean": float(metrics_frame["r2"].mean()),
    }
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "nested_tuning_summary.csv", index=False
    )
    oof = pd.concat(outer_predictions, ignore_index=True)
    manifest = build_manifest(
        {
            "experiment": "nested_xgboost_tuning",
            "clean_sha256": sha256_file(RESULTS_DIR / "clean.csv"),
            "folds_sha256": sha256_file(RESULTS_DIR / "folds.json"),
            "feature_config": "all_enhanced_features_with_maccs",
            "include_maccs": True,
            "inner_folds": 3,
            "candidate_configs": CONFIGS,
            "model_seed": 42,
        }
    )
    oof_dir = write_oof_artifact(
        RESULTS_DIR / "oof",
        "nested-xgboost",
        manifest,
        {"nested_xgboost": oof},
        seed=42,
    )
    print("\nNested tuning summary")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"OOF artifacts: {oof_dir}")


if __name__ == "__main__":
    main()

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


def feature_sets(column_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    chemistry = column_groups["chem_interaction"]
    base = [column for column in column_groups["all"] if column not in chemistry]
    return {
        "maccs": base,
        "maccs_hbond": base
        + [column for column in chemistry if "hbd" in column or "hba" in column],
        "maccs_molecular_weight_ratio": base
        + [column for column in chemistry if "weight" in column],
        "maccs_morgan_tanimoto": base
        + [column for column in chemistry if "tanimoto" in column],
        "maccs_all_chem_interactions": base + chemistry,
    }


def main() -> None:
    frame = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    assignment = load_folds(frame, RESULTS_DIR / "folds.json")
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)
    features, column_groups = build_feature_matrix(
        frame,
        include_solvent_identity=True,
        include_physchem_interactions=True,
        include_temperature_interactions=True,
        include_maccs=True,
        include_chem_interactions=True,
    )
    column_groups["all"] = features.columns.tolist()
    sets = feature_sets(column_groups)

    fold_rows = []
    summary_rows = []
    for feature_name, columns in sets.items():
        current_rows = []
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
                features.loc[train_mask, columns],
                target[train_mask],
            )
            prediction = model.predict(features.loc[test_mask, columns])
            row = {
                "feature_set": feature_name,
                "feature_count": len(columns),
                "fold": int(fold),
                **regression_metrics(target[test_mask], prediction),
            }
            current_rows.append(row)
            fold_rows.append(row)
        current = pd.DataFrame(current_rows)
        summary_rows.append(
            {
                "feature_set": feature_name,
                "feature_count": len(columns),
                "rmse_mean": current["rmse"].mean(),
                "rmse_std": current["rmse"].std(),
                "mae_mean": current["mae"].mean(),
                "r2_mean": current["r2"].mean(),
            }
        )
        print(summary_rows[-1], flush=True)

    summary = pd.DataFrame(summary_rows).sort_values("rmse_mean")
    summary.to_csv(
        RESULTS_DIR / "interaction_feature_ablation.csv",
        index=False,
    )
    pd.DataFrame(fold_rows).to_csv(
        RESULTS_DIR / "interaction_feature_ablation_by_fold.csv",
        index=False,
    )
    manifest = build_manifest(
        {
            "experiment": "incremental_chemistry_interaction_features",
            "base_feature_set": "all_enhanced_features_with_maccs",
            "candidate_sets": list(sets),
            "model": "xgboost",
            "model_seed": 42,
            "split_group": "drug_smiles_canonical",
        }
    )
    write_json(
        RESULTS_DIR / "interaction_feature_ablation_manifest.json",
        manifest,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

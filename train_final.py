from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor

from src.features import build_feature_matrix
from src.provenance import build_manifest, sha256_file, write_json


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = RESULTS_DIR / "clean.csv"
    folds_path = RESULTS_DIR / "folds.json"
    frame = pd.read_csv(clean_path, encoding="utf-8-sig")
    include_solvent_identity = True
    include_physchem_interactions = True
    include_temperature_interactions = True
    include_maccs = True
    features, _ = build_feature_matrix(
        frame,
        include_solvent_identity=include_solvent_identity,
        include_physchem_interactions=include_physchem_interactions,
        include_temperature_interactions=include_temperature_interactions,
        include_maccs=include_maccs,
    )
    target = frame["Solubility_log10"].to_numpy(dtype=np.float32)

    models = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=42,
        ),
        "xgboost": XGBRegressor(
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
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_features=0.7,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }
    for name, model in models.items():
        print(f"Fitting {name} on all cleaned rows...")
        model.fit(features, target)

    drug_smiles = frame["drug_smiles_canonical"].drop_duplicates().tolist()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    training_drug_fps = np.stack(
        [
            generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(
                np.float32
            )
            for smiles in drug_smiles
        ]
    )
    fingerprint_product = training_drug_fps @ training_drug_fps.T
    fingerprint_sizes = training_drug_fps.sum(axis=1)
    union = fingerprint_sizes[:, None] + fingerprint_sizes[None, :]
    union -= fingerprint_product
    similarity_matrix = fingerprint_product / np.maximum(union, 1)
    np.fill_diagonal(similarity_matrix, 0.0)
    nearest_similarities = similarity_matrix.max(axis=1)
    ad_threshold = float(np.quantile(nearest_similarities, 0.05))
    ad_metadata = {
        "method": "fifth_percentile_of_training_drug_leave_one_out_nearest_tanimoto",
        "threshold": ad_threshold,
        "n_training_drugs": len(drug_smiles),
        "minimum": float(nearest_similarities.min()),
        "median": float(np.median(nearest_similarities)),
        "maximum": float(nearest_similarities.max()),
    }
    solvents = (
        frame[["Solvent", "solvent_smiles_canonical"]]
        .drop_duplicates()
        .sort_values("Solvent")
        .reset_index(drop=True)
    )
    model_names = list(models)
    weights = np.full(len(model_names), 1.0 / len(model_names))
    training_temperature_min = float(frame["Temperature"].min())
    training_temperature_max = float(frame["Temperature"].max())
    manifest = build_manifest(
        {
            "clean_sha256": sha256_file(clean_path),
            "folds_sha256": sha256_file(folds_path),
            "training_rows": len(frame),
            "feature_count": features.shape[1],
            "training_temperature_min_c": training_temperature_min,
            "training_temperature_max_c": training_temperature_max,
            "drug_bits": 1024,
            "solvent_bits": 256,
            "include_solvent_identity": include_solvent_identity,
            "include_physchem_interactions": include_physchem_interactions,
            "include_temperature_interactions": (include_temperature_interactions),
            "include_maccs": include_maccs,
            "models": model_names,
            "model_weights": weights.tolist(),
            "ad_calibration": ad_metadata,
        }
    )
    ensemble_reference = pd.read_csv(RESULTS_DIR / "ensemble_metrics.csv")
    conformal_path = RESULTS_DIR / "conformal_calibration.json"
    conformal = (
        json.loads(conformal_path.read_text(encoding="utf-8"))
        if conformal_path.exists()
        else None
    )
    if conformal:
        manifest["config"]["conformal_calibration"] = conformal
    bundle = {
        "models": models,
        "model_names": model_names,
        "model_weights": weights.tolist(),
        "feature_columns": features.columns.tolist(),
        "drug_bits": 1024,
        "solvent_bits": 256,
        "include_solvent_identity": include_solvent_identity,
        "include_physchem_interactions": include_physchem_interactions,
        "include_temperature_interactions": (include_temperature_interactions),
        "include_maccs": include_maccs,
        "training_drug_smiles": drug_smiles,
        "training_drug_fps": training_drug_fps,
        "solvents": solvents,
        "ad_threshold": ad_threshold,
        "ad_calibration": ad_metadata,
        "training_rows": len(frame),
        "feature_count": features.shape[1],
        "temperature_range_c": {
            "min": training_temperature_min,
            "max": training_temperature_max,
        },
        "cv_reference": {
            "rmse_mean": float(ensemble_reference["rmse"].mean()),
            "mae_mean": float(ensemble_reference["mae"].mean()),
            "r2_mean": float(ensemble_reference["r2"].mean()),
        },
        "conformal": conformal,
        "manifest": manifest,
    }
    artifact_path = ARTIFACTS_DIR / "solubility_models.joblib"
    joblib.dump(bundle, artifact_path)
    metadata = {
        "artifact": str(artifact_path.relative_to(ROOT)),
        "training_rows": len(frame),
        "feature_count": features.shape[1],
        "models": model_names,
        "model_weights": bundle["model_weights"],
        "cv_reference": bundle["cv_reference"],
        "temperature_range_c": bundle["temperature_range_c"],
        "applicability_domain": ad_metadata,
        "manifest": manifest,
    }
    write_json(ARTIFACTS_DIR / "model_metadata.json", metadata)
    print(f"Saved model bundle to {artifact_path}")


if __name__ == "__main__":
    main()

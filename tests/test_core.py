from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.batch import predict_batch
from src.data import (
    assert_no_group_leakage,
    canonicalize_smiles,
    prepare_dataset,
    read_raw_data,
)
from src.features import build_feature_matrix
from src.prediction import load_model_bundle, predict_rows, predict_solvents
from src.splits import fold_metadata, load_folds


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@pytest.fixture(scope="session")
def clean_frame() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")


@pytest.fixture(scope="session")
def bundle() -> dict:
    return load_model_bundle(ROOT / "artifacts" / "solubility_models.joblib")


def test_fold_metadata_and_leakage(clean_frame: pd.DataFrame) -> None:
    assignment = load_folds(clean_frame, RESULTS_DIR / "folds.json")
    metadata = fold_metadata(RESULTS_DIR / "folds.json")
    assert metadata["n_splits"] == 5
    assert metadata["seed"] == 42
    assert metadata["group_column"] == "drug_smiles_canonical"
    assert_no_group_leakage(clean_frame, assignment, metadata["group_column"])


def test_feature_matrix_is_finite(clean_frame: pd.DataFrame) -> None:
    features, groups = build_feature_matrix(
        clean_frame.head(12),
        drug_bits=128,
        solvent_bits=64,
        include_maccs=True,
        include_chem_interactions=True,
    )
    assert features.shape[0] == 12
    assert {"drug", "solvent", "temperature"}.issubset(groups)
    assert groups["maccs"]
    assert groups["chem_interaction"]
    assert np.isfinite(features.to_numpy()).all()


def test_repeated_measurement_aggregation_variants() -> None:
    raw = read_raw_data(ROOT / "data_2026.csv")
    median, _ = prepare_dataset(raw, aggregate_method="median")
    mean, _ = prepare_dataset(raw, aggregate_method="mean")
    unaggregated, _ = prepare_dataset(raw, aggregate_method="none")
    assert len(median) == 1785
    assert len(mean) == 1785
    assert len(unaggregated) == 1900
    assert np.isfinite(unaggregated["Solubility_log10"].to_numpy()).all()


def test_prediction_contract(bundle: dict) -> None:
    ranking = predict_solvents(bundle, "CC(=O)NC1=CC=C(C=C1)O", temperature_c=25.0)
    assert len(ranking) == len(bundle["solvents"])
    assert ranking["log10_solubility"].is_monotonic_decreasing
    assert (ranking["model_disagreement"] >= 0).all()
    assert np.isfinite(ranking[["prediction_low", "prediction_high"]].to_numpy()).all()
    assert (ranking["lower_90"] <= ranking["log10_solubility"]).all()
    assert (ranking["log10_solubility"] <= ranking["upper_90"]).all()
    assert (ranking["lower_80"] >= ranking["lower_90"]).all()
    assert (ranking["upper_80"] <= ranking["upper_90"]).all()


def test_invalid_smiles_is_rejected(bundle: dict) -> None:
    assert canonicalize_smiles("not-a-smiles") is None
    with pytest.raises(ValueError, match="could not be parsed"):
        predict_solvents(bundle, "not-a-smiles", temperature_c=25.0)


def test_temperature_applicability_domain(bundle: dict) -> None:
    ranking = predict_solvents(
        bundle,
        "CC(=O)NC1=CC=C(C=C1)O",
        temperature_c=200.0,
    )
    assert (ranking["temperature_applicability_status"] == "Extrapolation").all()
    assert (ranking["applicability_status"] == "Extrapolation warning").all()


def test_training_inference_feature_consistency(
    bundle: dict, clean_frame: pd.DataFrame
) -> None:
    row = clean_frame.iloc[0]
    input_frame = pd.DataFrame(
        {
            "drug_smiles_canonical": [row["drug_smiles_canonical"]],
            "solvent_smiles_canonical": [row["solvent_smiles_canonical"]],
            "Temperature": [float(row["Temperature"])],
            "Solubility_log10": [0.0],
        }
    )
    direct = predict_rows(bundle, input_frame)
    features, _ = build_feature_matrix(
        input_frame,
        drug_bits=bundle["drug_bits"],
        solvent_bits=bundle["solvent_bits"],
        include_solvent_identity=bundle["include_solvent_identity"],
        include_physchem_interactions=bundle["include_physchem_interactions"],
        include_temperature_interactions=bundle["include_temperature_interactions"],
        include_maccs=bundle.get("include_maccs", False),
    )
    features = features.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    independent = np.column_stack(
        [model.predict(features) for model in bundle["models"].values()]
    ) @ np.asarray(bundle["model_weights"])
    assert direct.loc[0, "log10_solubility"] == pytest.approx(independent[0], abs=1e-5)


def test_batch_prediction_reports_row_errors(bundle: dict) -> None:
    solvent = bundle["solvents"].iloc[0]
    batch = pd.DataFrame(
        {
            "Drug SMILES": [
                "CC(=O)NC1=CC=C(C=C1)O",
                "not-a-smiles",
                "CC(=O)NC1=CC=C(C=C1)O",
            ],
            "Solvent": [solvent["Solvent"], solvent["Solvent"], "Unknown solvent"],
            "Temperature": [25.0, 25.0, 25.0],
        }
    )
    result = predict_batch(bundle, batch)
    assert result["status"].tolist() == ["ok", "error", "error"]
    assert result.loc[1, "error"] == "Invalid or empty drug SMILES"
    assert "Unknown solvent" in result.loc[2, "error"]

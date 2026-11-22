from __future__ import annotations

from pathlib import Path
from typing import Any

import io
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from src.data import canonicalize_smiles
from src.features import build_feature_matrix


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def _tanimoto(query: np.ndarray, training: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(query[None, :], training).sum(axis=1)
    union = np.logical_or(query[None, :], training).sum(axis=1)
    return intersection / np.maximum(union, 1)


def predict_solvents(
    bundle: dict[str, Any],
    drug_smiles: str,
    temperature_c: float,
) -> pd.DataFrame:
    canonical_drug = canonicalize_smiles(drug_smiles)
    if canonical_drug is None:
        raise ValueError("The drug SMILES could not be parsed by RDKit.")

    solvents = bundle["solvents"].copy()
    frame = pd.DataFrame(
        {
            "Drug": "User input",
            "Solvent": solvents["Solvent"],
            "drug_smiles_canonical": canonical_drug,
            "solvent_smiles_canonical": solvents["solvent_smiles_canonical"],
            "Temperature": float(temperature_c),
            "Solubility_log10": 0.0,
        }
    )
    frame = _predict_frame(bundle, frame)
    return frame.sort_values("log10_solubility", ascending=False).reset_index(drop=True)


def predict_temperature_curve(
    bundle: dict[str, Any],
    drug_smiles: str,
    solvent_smiles: str,
    temperatures_c: np.ndarray,
) -> pd.DataFrame:
    canonical_drug = canonicalize_smiles(drug_smiles)
    canonical_solvent = canonicalize_smiles(solvent_smiles)
    if canonical_drug is None or canonical_solvent is None:
        raise ValueError("The supplied SMILES could not be parsed.")
    temperatures = np.asarray(temperatures_c, dtype=float)
    frame = pd.DataFrame(
        {
            "Drug": "User input",
            "Solvent": "Selected solvent",
            "drug_smiles_canonical": canonical_drug,
            "solvent_smiles_canonical": canonical_solvent,
            "Temperature": temperatures,
            "Solubility_log10": 0.0,
        }
    )
    result = _predict_frame(bundle, frame)
    result["Temperature"] = temperatures
    return result


def _predict_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    features, _ = build_feature_matrix(
        frame,
        drug_bits=bundle["drug_bits"],
        solvent_bits=bundle["solvent_bits"],
        include_solvent_identity=bundle.get("include_solvent_identity", False),
        include_physchem_interactions=bundle.get(
            "include_physchem_interactions",
            bundle.get("include_interactions", False),
        ),
        include_temperature_interactions=bundle.get(
            "include_temperature_interactions",
            bundle.get("include_interactions", False),
        ),
        include_maccs=bundle.get("include_maccs", False),
        include_chem_interactions=bundle.get("include_chem_interactions", False),
    )
    features = features.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    predictions = np.column_stack(
        [model.predict(features) for model in bundle["models"].values()]
    )
    weights = np.asarray(
        bundle.get("model_weights", np.full(predictions.shape[1], 1.0)),
        dtype=np.float32,
    )
    if len(weights) != predictions.shape[1] or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Model weights are invalid or do not sum to one.")
    result = frame.copy()
    result["log10_solubility"] = predictions @ weights
    result["model_disagreement"] = predictions.std(axis=1)
    result["model_std"] = result["model_disagreement"]
    result["prediction_low"] = result["log10_solubility"] - result["model_disagreement"]
    result["prediction_high"] = (
        result["log10_solubility"] + result["model_disagreement"]
    )
    conformal = bundle.get("conformal")
    if conformal:
        result["lower_80"] = result["log10_solubility"] - conformal["radius_80"]
        result["upper_80"] = result["log10_solubility"] + conformal["radius_80"]
        result["lower_90"] = result["log10_solubility"] - conformal["radius_90"]
        result["upper_90"] = result["log10_solubility"] + conformal["radius_90"]
    result["solubility_mol_l"] = np.power(10.0, result["log10_solubility"])

    nearest_by_drug = {}
    for drug_smiles in frame["drug_smiles_canonical"].unique():
        query_fingerprint = _drug_fingerprint(drug_smiles, bundle["drug_bits"])
        similarity = _tanimoto(query_fingerprint, bundle["training_drug_fps"])
        nearest_index = int(np.argmax(similarity))
        nearest_by_drug[drug_smiles] = (
            float(similarity[nearest_index]),
            bundle["training_drug_smiles"][nearest_index],
        )
    result["nearest_drug_similarity"] = frame["drug_smiles_canonical"].map(
        lambda smiles: nearest_by_drug[smiles][0]
    )
    result["nearest_training_drug"] = frame["drug_smiles_canonical"].map(
        lambda smiles: nearest_by_drug[smiles][1]
    )
    result["drug_applicability_status"] = np.where(
        result["nearest_drug_similarity"].to_numpy() >= bundle["ad_threshold"],
        "Within",
        "Extrapolation",
    )
    temperature_range = bundle.get("temperature_range_c")
    if temperature_range:
        temperature_inside = result["Temperature"].between(
            float(temperature_range["min"]),
            float(temperature_range["max"]),
            inclusive="both",
        )
        result["temperature_applicability_status"] = np.where(
            temperature_inside,
            "Within",
            "Extrapolation",
        )
        result["applicability_status"] = np.where(
            (result["drug_applicability_status"] == "Within")
            & (result["temperature_applicability_status"] == "Within"),
            "Within applicability domain",
            "Extrapolation warning",
        )
    else:
        result["temperature_applicability_status"] = "Unknown"
        result["applicability_status"] = np.where(
            result["drug_applicability_status"] == "Within",
            "Within applicability domain",
            "Extrapolation warning",
        )
    return result


def predict_rows(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    required = {"drug_smiles_canonical", "solvent_smiles_canonical", "Temperature"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    return _predict_frame(bundle, frame.copy())


def _drug_fingerprint(smiles: str, n_bits: int) -> np.ndarray:
    from rdkit.Chem import rdFingerprintGenerator

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return generator.GetFingerprintAsNumPy(molecule).astype(np.float32)


def molecule_png(smiles: str, width: int = 500, height: int = 320) -> bytes:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES")
    image = Draw.MolToImage(molecule, size=(width, height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def molecule_svg(smiles: str, width: int = 500, height: int = 320) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES")
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

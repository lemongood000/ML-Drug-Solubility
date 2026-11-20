from __future__ import annotations

from typing import Any

import pandas as pd

from src.data import canonicalize_smiles
from src.prediction import predict_rows


INPUT_ALIASES = {
    "drug_smiles": "Drug SMILES",
    "drug smiles": "Drug SMILES",
    "solvent": "Solvent",
    "temperature": "Temperature",
}


def normalise_batch_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: INPUT_ALIASES.get(str(column).strip().lower(), str(column).strip())
        for column in frame.columns
    }
    frame = frame.rename(columns=renamed)
    required = {"Drug SMILES", "Solvent", "Temperature"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Batch CSV must contain columns: Drug SMILES, Solvent, Temperature. "
            f"Missing: {sorted(missing)}"
        )
    return frame


def predict_batch(
    bundle: dict[str, Any], input_frame: pd.DataFrame
) -> pd.DataFrame:
    frame = normalise_batch_columns(input_frame)
    solvent_lookup = {
        str(name).strip().casefold(): smiles
        for name, smiles in zip(
            bundle["solvents"]["Solvent"],
            bundle["solvents"]["solvent_smiles_canonical"],
            strict=True,
        )
    }
    valid_rows = []
    errors = []
    for position, row in frame.iterrows():
        row_number = int(position) + 2
        drug_raw = str(row["Drug SMILES"]).strip()
        solvent_raw = str(row["Solvent"]).strip()
        try:
            temperature = float(row["Temperature"])
        except (TypeError, ValueError):
            temperature = None
        canonical_smiles = canonicalize_smiles(drug_raw) if drug_raw else None
        solvent_smiles = solvent_lookup.get(solvent_raw.casefold())

        error_messages = []
        if canonical_smiles is None:
            error_messages.append("Invalid or empty drug SMILES")
        if solvent_smiles is None:
            error_messages.append("Unknown solvent; only the 49 training solvents are supported")
        if temperature is None:
            error_messages.append("Temperature is not numeric")

        if error_messages:
            errors.append(
                {
                    "row_number": row_number,
                    "Drug SMILES": drug_raw,
                    "Solvent": solvent_raw,
                    "Temperature": row["Temperature"],
                    "status": "error",
                    "error": "; ".join(error_messages),
                }
            )
            continue
        valid_rows.append(
            {
                "row_number": row_number,
                "Drug SMILES": drug_raw,
                "Solvent": solvent_raw,
                "Temperature": temperature,
                "drug_smiles_canonical": canonical_smiles,
                "solvent_smiles_canonical": solvent_smiles,
                "Solubility_log10": 0.0,
            }
        )

    valid_output = pd.DataFrame()
    if valid_rows:
        valid_input = pd.DataFrame(valid_rows)
        predictions = predict_rows(bundle, valid_input)
        predictions["status"] = "ok"
        predictions["error"] = ""
        columns = [
            "row_number",
            "Drug SMILES",
            "Solvent",
            "Temperature",
            "status",
            "error",
            "log10_solubility",
            "solubility_mol_l",
            "model_disagreement",
            *[
                column
                for column in ("lower_80", "upper_80", "lower_90", "upper_90")
                if column in predictions.columns
            ],
            "nearest_drug_similarity",
            "applicability_status",
        ]
        valid_output = predictions[columns]

    error_output = pd.DataFrame(errors)
    combined = pd.concat([valid_output, error_output], ignore_index=True)
    return combined.sort_values("row_number").reset_index(drop=True)

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


REQUIRED_COLUMNS = {
    "Drug",
    "Solvent",
    "Drug SMILES",
    "Solvent SMILES",
    "Temperature",
    "Solubility_log10",
    "Solubility Unit",
}

GROUP_COLUMNS = [
    "drug_smiles_canonical",
    "solvent_smiles_canonical",
    "Temperature",
]


@dataclass
class DataAudit:
    source_rows: int
    exact_duplicate_rows_removed: int
    invalid_drug_smiles: int
    invalid_solvent_smiles: int
    rows_after_canonicalization: int
    condition_groups_before_aggregation: int
    repeated_condition_groups: int
    numeric_conflict_groups: int
    numeric_conflict_groups_gt_0_05_log: int
    max_condition_target_range: float
    rows_after_aggregation: int
    unique_drug_names: int
    unique_drug_smiles: int
    unique_solvent_names: int
    unique_solvent_smiles: int
    unique_smiles_across_roles: int
    drug_solvent_pairs: int
    singleton_drug_solvent_pairs: int
    missing_values: dict[str, int]
    solubility_unit_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    errors: list[str] = []
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    else:
        raise UnicodeError(f"Could not decode {path}: {'; '.join(errors)}")

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].str.strip()

    for column in ("Temperature", "Solubility_log10"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    return frame


def canonicalize_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def prepare_dataset(
    raw: pd.DataFrame,
    aggregate_method: str = "median",
) -> tuple[pd.DataFrame, DataAudit]:
    if aggregate_method not in {"median", "mean", "none"}:
        raise ValueError("aggregate_method must be one of: median, mean, none")
    frame = raw.copy()
    source_rows = len(frame)

    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    exact_duplicates = before - len(frame)

    frame["drug_smiles_canonical"] = frame["Drug SMILES"].map(canonicalize_smiles)
    frame["solvent_smiles_canonical"] = frame["Solvent SMILES"].map(canonicalize_smiles)
    invalid_drug = int(frame["drug_smiles_canonical"].isna().sum())
    invalid_solvent = int(frame["solvent_smiles_canonical"].isna().sum())
    if invalid_drug or invalid_solvent:
        raise ValueError(
            "Invalid SMILES found. "
            f"Drug rows: {invalid_drug}, solvent rows: {invalid_solvent}. "
            "Inspect the source data before continuing."
        )

    condition_stats = (
        frame.groupby(GROUP_COLUMNS, dropna=False)["Solubility_log10"]
        .agg(["size", "min", "max", "median"])
        .reset_index()
    )
    condition_stats["target_range"] = condition_stats["max"] - condition_stats["min"]
    repeated = condition_stats["size"] > 1
    conflicts = repeated & (condition_stats["target_range"] > 1e-12)
    large_conflicts = repeated & (condition_stats["target_range"] > 0.05)

    if aggregate_method == "none":
        clean = frame.rename(
            columns={
                "Drug SMILES": "drug_smiles",
                "Solvent SMILES": "solvent_smiles",
                "Solubility Unit": "Solubility_Unit",
            }
        ).copy()
        clean["solubility_std"] = 0.0
        clean["solubility_min"] = clean["Solubility_log10"]
        clean["solubility_max"] = clean["Solubility_log10"]
        clean["n_source_records"] = 1
        clean["solubility_range"] = 0.0
    else:
        grouped = frame.groupby(GROUP_COLUMNS, dropna=False, sort=False).agg(
            Drug=("Drug", "first"),
            Solvent=("Solvent", "first"),
            drug_smiles=("Drug SMILES", "first"),
            solvent_smiles=("Solvent SMILES", "first"),
            Solubility_log10=("Solubility_log10", aggregate_method),
            solubility_std=("Solubility_log10", "std"),
            solubility_min=("Solubility_log10", "min"),
            solubility_max=("Solubility_log10", "max"),
            n_source_records=("Solubility_log10", "size"),
            Solubility_Unit=("Solubility Unit", "first"),
        )
        clean = grouped.reset_index()
        clean["solubility_std"] = clean["solubility_std"].fillna(0.0)
        clean["solubility_range"] = clean["solubility_max"] - clean["solubility_min"]
    clean = clean.sort_values(
        ["drug_smiles_canonical", "solvent_smiles_canonical", "Temperature"]
    ).reset_index(drop=True)
    clean.insert(0, "row_id", [f"r{i:05d}" for i in range(1, len(clean) + 1)])

    pair_counts = clean.groupby(
        ["drug_smiles_canonical", "solvent_smiles_canonical"], dropna=False
    ).size()
    drug_smiles = set(clean["drug_smiles_canonical"])
    solvent_smiles = set(clean["solvent_smiles_canonical"])

    audit = DataAudit(
        source_rows=source_rows,
        exact_duplicate_rows_removed=exact_duplicates,
        invalid_drug_smiles=invalid_drug,
        invalid_solvent_smiles=invalid_solvent,
        rows_after_canonicalization=len(frame),
        condition_groups_before_aggregation=len(condition_stats),
        repeated_condition_groups=int(repeated.sum()),
        numeric_conflict_groups=int(conflicts.sum()),
        numeric_conflict_groups_gt_0_05_log=int(large_conflicts.sum()),
        max_condition_target_range=float(condition_stats["target_range"].max()),
        rows_after_aggregation=len(clean),
        unique_drug_names=int(clean["Drug"].nunique()),
        unique_drug_smiles=int(clean["drug_smiles_canonical"].nunique()),
        unique_solvent_names=int(clean["Solvent"].nunique()),
        unique_solvent_smiles=int(clean["solvent_smiles_canonical"].nunique()),
        unique_smiles_across_roles=len(drug_smiles | solvent_smiles),
        drug_solvent_pairs=len(pair_counts),
        singleton_drug_solvent_pairs=int((pair_counts == 1).sum()),
        missing_values={
            column: int(value) for column, value in raw.isna().sum().items()
        },
        solubility_unit_counts={
            str(key): int(value)
            for key, value in frame["Solubility Unit"].value_counts().items()
        },
    )
    return clean, audit


def assert_no_group_leakage(
    frame: pd.DataFrame, fold_assignment: np.ndarray, group_column: str
) -> None:
    groups_by_fold = [
        set(frame.loc[fold_assignment == fold, group_column])
        for fold in sorted(np.unique(fold_assignment))
    ]
    for left in range(len(groups_by_fold)):
        for right in range(left + 1, len(groups_by_fold)):
            overlap = groups_by_fold[left] & groups_by_fold[right]
            if overlap:
                raise AssertionError(
                    f"Group leakage between folds {left} and {right}: "
                    f"{sorted(overlap)[:3]}"
                )

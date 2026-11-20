from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FoldSpec:
    n_splits: int
    seed: int
    group_column: str


def make_balanced_group_folds(
    frame: pd.DataFrame,
    group_column: str = "drug_smiles_canonical",
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    group_sizes = (
        frame.groupby(group_column, dropna=False)
        .size()
        .rename("group_size")
        .reset_index()
    )
    if len(group_sizes) < n_splits:
        raise ValueError(
            f"Only {len(group_sizes)} groups are available for {n_splits} folds"
        )

    rng = np.random.default_rng(seed)
    tie_breaker = rng.permutation(len(group_sizes))
    group_sizes["tie_breaker"] = tie_breaker
    group_sizes = group_sizes.sort_values(
        ["group_size", "tie_breaker"], ascending=[False, True]
    ).reset_index(drop=True)

    fold_sizes = np.zeros(n_splits, dtype=int)
    group_to_fold: dict[object, int] = {}
    for row in group_sizes.itertuples(index=False):
        fold = int(np.argmin(fold_sizes))
        group_to_fold[getattr(row, group_column)] = fold
        fold_sizes[fold] += int(row.group_size)

    assignment = frame[group_column].map(group_to_fold).to_numpy(dtype=int)
    return assignment


def save_folds(
    frame: pd.DataFrame,
    assignment: np.ndarray,
    path: str | Path,
    *,
    seed: int,
    group_column: str = "drug_smiles_canonical",
) -> None:
    path = Path(path)
    payload = {
        "spec": asdict(
            FoldSpec(
                n_splits=int(assignment.max()) + 1,
                seed=seed,
                group_column=group_column,
            )
        ),
        "folds": {
            str(fold): frame.loc[assignment == fold, "row_id"].tolist()
            for fold in range(int(assignment.max()) + 1)
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def load_folds(
    frame: pd.DataFrame, path: str | Path
) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    row_to_fold = {
        row_id: int(fold)
        for fold, row_ids in payload["folds"].items()
        for row_id in row_ids
    }
    missing = set(frame["row_id"]).difference(row_to_fold)
    if missing:
        raise ValueError(f"{len(missing)} rows are missing from folds.json")
    return frame["row_id"].map(row_to_fold).to_numpy(dtype=int)


def fold_metadata(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["spec"]


def fold_summary(frame: pd.DataFrame, assignment: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame({"fold": assignment, "drug_group": frame["drug_smiles_canonical"]})
        .groupby("fold")
        .agg(
            rows=("drug_group", "size"),
            drug_groups=("drug_group", "nunique"),
        )
        .reset_index()
    )

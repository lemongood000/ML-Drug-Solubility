from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.splits import fold_summary, make_balanced_group_folds, save_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SPLITS_DIR = RESULTS_DIR / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-seeds", default="1729,2026")
    return parser.parse_args()


def random_folds(frame: pd.DataFrame, seed: int, n_splits: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    assignment = np.empty(len(frame), dtype=int)
    assignment[order] = np.arange(len(frame)) % n_splits
    return assignment


def scaffold_group(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    if scaffold:
        return scaffold
    return f"acyclic:{smiles}"


def main() -> None:
    args = parse_args()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")

    split_specs = []
    for seed in [int(value) for value in args.drug_seeds.split(",")]:
        assignment = make_balanced_group_folds(
            frame,
            group_column="drug_smiles_canonical",
            n_splits=5,
            seed=seed,
        )
        split_specs.append(("drug", seed, "drug_smiles_canonical", assignment, frame))

    pair_frame = frame.assign(
        pair_group=(
            frame["drug_smiles_canonical"] + "|" + frame["solvent_smiles_canonical"]
        )
    )
    pair_assignment = make_balanced_group_folds(
        pair_frame,
        group_column="pair_group",
        n_splits=5,
        seed=42,
    )
    split_specs.append(("pair", 42, "pair_group", pair_assignment, pair_frame))

    scaffold_frame = frame.assign(
        scaffold_group=frame["drug_smiles_canonical"].map(scaffold_group)
    )
    scaffold_assignment = make_balanced_group_folds(
        scaffold_frame,
        group_column="scaffold_group",
        n_splits=5,
        seed=42,
    )
    split_specs.append(
        ("scaffold", 42, "scaffold_group", scaffold_assignment, scaffold_frame)
    )

    solvent_assignment = make_balanced_group_folds(
        frame,
        group_column="solvent_smiles_canonical",
        n_splits=5,
        seed=42,
    )
    split_specs.append(
        (
            "solvent",
            42,
            "solvent_smiles_canonical",
            solvent_assignment,
            frame,
        )
    )

    random_assignment = random_folds(frame, seed=42)
    split_specs.append(("random", 42, "row_id", random_assignment, frame))

    for kind, seed, group_column, assignment, split_frame in split_specs:
        output_path = SPLITS_DIR / f"folds_{kind}_{seed}.json"
        save_folds(
            split_frame,
            assignment,
            output_path,
            seed=seed,
            group_column=group_column,
        )
        summary = fold_summary(
            split_frame.assign(drug_smiles_canonical=split_frame[group_column]),
            assignment,
        )
        summary.to_csv(SPLITS_DIR / f"fold_summary_{kind}_{seed}.csv", index=False)
        print(f"Wrote {kind} split seed={seed} to {output_path}")

    print(f"Split artifacts written to {SPLITS_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from src.data import assert_no_group_leakage, prepare_dataset, read_raw_data
from src.provenance import build_manifest, sha256_file, write_json
from src.splits import fold_summary, make_balanced_group_folds, save_folds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def resolve_data_path() -> Path:
    candidates = [
        ROOT / "data_2026.csv",
        ROOT / "data" / "data_2026.csv",
        ROOT.parent / "data_2026.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "data_2026.csv was not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


DATA_PATH = resolve_data_path()


def main() -> None:
    seed = 42
    group_column = "drug_smiles_canonical"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_raw_data(DATA_PATH)
    clean, audit = prepare_dataset(raw)

    assignment = make_balanced_group_folds(
        clean,
        group_column=group_column,
        n_splits=5,
        seed=seed,
    )
    assert_no_group_leakage(clean, assignment, group_column)

    clean.assign(fold=assignment).to_csv(
        RESULTS_DIR / "clean.csv", index=False, encoding="utf-8-sig"
    )
    (RESULTS_DIR / "audit.json").write_text(
        json.dumps(audit.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_folds(
        clean,
        assignment,
        RESULTS_DIR / "folds.json",
        seed=seed,
        group_column=group_column,
    )
    summary = fold_summary(clean, assignment)
    summary.to_csv(RESULTS_DIR / "fold_summary.csv", index=False)
    write_json(
        RESULTS_DIR / "data_manifest.json",
        build_manifest(
            {
                "source_data": str(DATA_PATH.relative_to(ROOT.parent)),
                "source_sha256": sha256_file(DATA_PATH),
                "clean_rows": len(clean),
                "n_splits": 5,
                "seed": seed,
                "group_column": group_column,
            }
        ),
    )

    print(f"Raw rows: {audit.source_rows}")
    print(f"Rows after aggregation: {audit.rows_after_aggregation}")
    print(
        f"Canonical groups: {audit.unique_drug_smiles} drugs, "
        f"{audit.unique_solvent_smiles} solvents"
    )
    print(summary.to_string(index=False))
    print(f"Artifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

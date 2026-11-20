from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.provenance import write_json


COLUMN_ORDER = ["row_id", "fold", "y_true", "y_pred"]
REQUIRED_COLUMNS = set(COLUMN_ORDER)


def write_oof_artifact(
    root: str | Path,
    name: str,
    manifest: dict[str, Any],
    model_predictions: dict[str, pd.DataFrame],
    seed: int,
) -> Path:
    run_dir = Path(root) / f"{name}-{manifest['config_id']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for model_name, frame in model_predictions.items():
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{model_name} is missing OOF columns: {sorted(missing)}"
            )
        frames.append(
            frame[COLUMN_ORDER]
            .assign(model=model_name, seed=seed)
        )
    combined = pd.concat(frames, ignore_index=True)
    predictions_path = run_dir / "predictions.csv"
    combined.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    payload = {
        **manifest,
        "models": sorted(model_predictions),
        "seed": seed,
        "prediction_rows": len(combined),
        "prediction_file": predictions_path.name,
    }
    write_json(run_dir / "manifest.json", payload)
    return run_dir


def latest_oof_run(root: str | Path, prefix: str) -> Path:
    candidates = sorted(
        (
            manifest
            for manifest in Path(root).glob(f"{prefix}-*/manifest.json")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No OOF run with prefix {prefix!r} found.")
    return candidates[0].parent

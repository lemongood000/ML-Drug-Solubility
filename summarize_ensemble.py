from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import regression_metrics


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
SPECS = {
    "base": {
        "models": ["hist_gradient_boosting", "xgb", "rf"],
        "output": RESULTS_DIR / "base_ensemble_metrics.csv",
    },
    "enhanced": {
        "models": [
            "enhanced_hist_gradient_boosting",
            "enhanced_xgboost",
            "enhanced_random_forest",
        ],
        "output": RESULTS_DIR / "ensemble_metrics.csv",
    },
}


def load_prediction(model_name: str) -> pd.DataFrame:
    files = sorted(
        glob.glob(str(PREDICTIONS_DIR / f"{model_name}_fold*.csv"))
    )
    if len(files) != 5:
        raise ValueError(
            f"Expected five prediction files for {model_name}, found {len(files)}"
        )
    return pd.concat(
        [pd.read_csv(path) for path in files], ignore_index=True
    ).sort_values("row_id")


def summarize(models: list[str], output: Path) -> None:
    predictions = {name: load_prediction(name) for name in models}
    reference = predictions[models[0]]
    for name in models[1:]:
        if not np.array_equal(
            predictions[name]["y_true"].to_numpy(),
            reference["y_true"].to_numpy(),
        ):
            raise AssertionError("Prediction files do not share the same targets")

    y_true = reference["y_true"].to_numpy()
    ensemble_prediction = np.column_stack(
        [predictions[name]["y_pred"].to_numpy() for name in models]
    ).mean(axis=1)
    fold = reference["fold"].to_numpy()
    rows = []
    for current_fold in sorted(np.unique(fold)):
        mask = fold == current_fold
        rows.append(
            {
                "fold": int(current_fold),
                **regression_metrics(y_true[mask], ensemble_prediction[mask]),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    overall = regression_metrics(y_true, ensemble_prediction)
    print(
        f"{output.stem} "
        f"mean_fold_rmse={result['rmse'].mean():.6f} "
        f"overall_rmse={overall['rmse']:.6f} "
        f"overall_mae={overall['mae']:.6f} "
        f"overall_r2={overall['r2']:.6f}"
    )


def main() -> None:
    for spec in SPECS.values():
        summarize(spec["models"], spec["output"])


if __name__ == "__main__":
    main()

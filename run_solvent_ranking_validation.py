from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.oof import latest_oof_run


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "ranking_validation"
FIGURES_DIR = ROOT / "figures"
TEMPERATURE = 25.0
TEMPERATURE_TOLERANCE = 0.5
MIN_SOLVENTS = 4


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    run_dir = latest_oof_run(RESULTS_DIR / "oof", "enhanced-main")
    predictions = pd.read_csv(
        run_dir / "predictions.csv", encoding="utf-8-sig"
    )
    pivot = predictions.pivot(
        index=["row_id", "fold", "y_true"],
        columns="model",
        values="y_pred",
    ).reset_index()
    pivot["y_pred"] = pivot[
        ["hist_gradient_boosting", "xgboost", "random_forest"]
    ].mean(axis=1)
    frame = pivot.merge(
        clean[
            [
                "row_id",
                "drug_smiles_canonical",
                "solvent_smiles_canonical",
                "Solvent",
                "Temperature",
            ]
        ],
        on="row_id",
        validate="one_to_one",
    )
    frame = frame.loc[
        np.abs(frame["Temperature"] - TEMPERATURE)
        <= TEMPERATURE_TOLERANCE
    ].copy()

    per_drug_rows = []
    for (fold, drug), subset in frame.groupby(
        ["fold", "drug_smiles_canonical"]
    ):
        solvent_means = (
            subset.groupby(["solvent_smiles_canonical", "Solvent"])
            .agg(
                observed=("y_true", "mean"),
                predicted=("y_pred", "mean"),
                n=("row_id", "size"),
            )
            .reset_index()
        )
        if len(solvent_means) < MIN_SOLVENTS:
            continue

        training = clean.loc[
            (clean["fold"] != fold)
            & (
                np.abs(clean["Temperature"] - TEMPERATURE)
                <= TEMPERATURE_TOLERANCE
            )
        ]
        baseline_solvent = training.groupby(
            "solvent_smiles_canonical"
        )["Solubility_log10"].mean()
        training_global = float(training["Solubility_log10"].mean())
        solvent_means["baseline"] = (
            solvent_means["solvent_smiles_canonical"]
            .map(baseline_solvent)
            .fillna(training_global)
        )
        predicted_rho = spearmanr(
            solvent_means["observed"], solvent_means["predicted"]
        ).statistic
        baseline_rho = spearmanr(
            solvent_means["observed"], solvent_means["baseline"]
        ).statistic
        observed_top = solvent_means.loc[
            solvent_means["observed"].idxmax(), "Solvent"
        ]
        predicted_top = solvent_means.loc[
            solvent_means["predicted"].idxmax(), "Solvent"
        ]
        baseline_top = solvent_means.loc[
            solvent_means["baseline"].idxmax(), "Solvent"
        ]
        per_drug_rows.append(
            {
                "fold": int(fold),
                "drug_smiles_canonical": drug,
                "n_solvents": len(solvent_means),
                "predicted_spearman": predicted_rho,
                "baseline_spearman": baseline_rho,
                "observed_top_solvent": observed_top,
                "predicted_top_solvent": predicted_top,
                "baseline_top_solvent": baseline_top,
                "predicted_top1_correct": predicted_top == observed_top,
                "baseline_top1_correct": baseline_top == observed_top,
            }
        )

    result = pd.DataFrame(per_drug_rows)
    if result.empty:
        raise RuntimeError("No drug had enough solvents at the reference temperature.")
    result.to_csv(OUTPUT_DIR / "solvent_ranking_by_drug.csv", index=False)

    summary = {
        "temperature_c": TEMPERATURE,
        "temperature_tolerance_c": TEMPERATURE_TOLERANCE,
        "minimum_solvents_per_drug": MIN_SOLVENTS,
        "evaluated_drugs": len(result),
        "predicted_spearman_median": float(
            result["predicted_spearman"].median()
        ),
        "predicted_spearman_q25": float(
            result["predicted_spearman"].quantile(0.25)
        ),
        "predicted_spearman_q75": float(
            result["predicted_spearman"].quantile(0.75)
        ),
        "baseline_spearman_median": float(
            result["baseline_spearman"].median()
        ),
        "baseline_spearman_q25": float(
            result["baseline_spearman"].quantile(0.25)
        ),
        "baseline_spearman_q75": float(
            result["baseline_spearman"].quantile(0.75)
        ),
        "predicted_top1_accuracy": float(
            result["predicted_top1_correct"].mean()
        ),
        "baseline_top1_accuracy": float(
            result["baseline_top1_correct"].mean()
        ),
        "predicted_beats_baseline_fraction": float(
            (
                result["predicted_spearman"]
                > result["baseline_spearman"]
            ).mean()
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        result["baseline_spearman"],
        result["predicted_spearman"],
        s=32,
        alpha=0.7,
        color="#2E6F73",
        edgecolors="none",
    )
    low = min(
        result["baseline_spearman"].min(),
        result["predicted_spearman"].min(),
    )
    high = max(
        result["baseline_spearman"].max(),
        result["predicted_spearman"].max(),
    )
    ax.plot([low, high], [low, high], color="#30363D", linewidth=1)
    ax.set_xlabel("Training-only solvent-mean baseline Spearman")
    ax.set_ylabel("OOF model solvent-ranking Spearman")
    ax.set_title("Solvent ranking at 25 C")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "solvent_ranking_validation.png", dpi=220)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

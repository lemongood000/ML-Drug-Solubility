from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr

from src.features import build_feature_matrix
from src.metrics import regression_metrics
from src.oof import latest_oof_run


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "error_analysis"
FIGURES_DIR = ROOT / "figures"


def nearest_training_similarities(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    drugs = frame["drug_smiles_canonical"].drop_duplicates().tolist()
    fingerprints = {
        smiles: generator.GetFingerprintAsNumPy(
            Chem.MolFromSmiles(smiles)
        ).astype(bool)
        for smiles in drugs
    }
    rows = []
    for fold in sorted(frame["fold"].unique()):
        train_drugs = frame.loc[
            frame["fold"] != fold, "drug_smiles_canonical"
        ].unique()
        test_drugs = frame.loc[
            frame["fold"] == fold, "drug_smiles_canonical"
        ].unique()
        for drug in test_drugs:
            query = fingerprints[drug]
            similarities = []
            for train_drug in train_drugs:
                candidate = fingerprints[train_drug]
                intersection = np.logical_and(query, candidate).sum()
                union = np.logical_or(query, candidate).sum()
                similarities.append(intersection / max(union, 1))
            rows.append(
                {
                    "fold": int(fold),
                    "drug_smiles_canonical": drug,
                    "nearest_similarity": float(max(similarities)),
                }
            )
    return pd.DataFrame(rows)


def load_ensemble_oof() -> pd.DataFrame:
    run_dir = latest_oof_run(RESULTS_DIR / "oof", "enhanced-main")
    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
    pivot = predictions.pivot(
        index=["row_id", "fold", "y_true"],
        columns="model",
        values="y_pred",
    ).reset_index()
    model_columns = [
        "hist_gradient_boosting",
        "xgboost",
        "random_forest",
    ]
    pivot["y_pred"] = pivot[model_columns].mean(axis=1)
    pivot["model_disagreement"] = pivot[model_columns].std(axis=1)
    pivot["abs_error"] = np.abs(pivot["y_true"] - pivot["y_pred"])
    return pivot


def grouped_metrics(
    frame: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(group_column, dropna=False):
        metrics = regression_metrics(
            subset["y_true"].to_numpy(),
            subset["y_pred"].to_numpy(),
        )
        if len(subset) < 2:
            metrics["r2"] = np.nan
        rows.append(
            {
                group_column: group,
                "n": len(subset),
                **metrics,
                "bias": float(
                    (subset["y_true"] - subset["y_pred"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in (0.2, 0.3, 0.4):
        warning = frame["nearest_similarity"].to_numpy() < threshold
        rows.append(
            {
                "threshold": threshold,
                "warning_fraction": float(warning.mean()),
                "inside_rows": int((~warning).sum()),
                "outside_rows": int(warning.sum()),
                "inside_rmse": regression_metrics(
                    frame.loc[~warning, "y_true"],
                    frame.loc[~warning, "y_pred"],
                )["rmse"],
                "outside_rmse": (
                    regression_metrics(
                        frame.loc[warning, "y_true"],
                        frame.loc[warning, "y_pred"],
                    )["rmse"]
                    if warning.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_by_drug(
    frame: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    drugs = frame["drug_smiles_canonical"].unique()
    groups = {
        drug: frame.loc[
            frame["drug_smiles_canonical"] == drug,
            ["y_true", "y_pred"],
        ].to_numpy()
        for drug in drugs
    }
    rmse_values = []
    mae_values = []
    for _ in range(n_bootstrap):
        sampled_drugs = rng.choice(drugs, size=len(drugs), replace=True)
        sample = np.concatenate([groups[drug] for drug in sampled_drugs])
        metrics = regression_metrics(sample[:, 0], sample[:, 1])
        rmse_values.append(metrics["rmse"])
        mae_values.append(metrics["mae"])
    return {
        "n_bootstrap": n_bootstrap,
        "rmse_ci_2_5": float(np.quantile(rmse_values, 0.025)),
        "rmse_ci_97_5": float(np.quantile(rmse_values, 0.975)),
        "mae_ci_2_5": float(np.quantile(mae_values, 0.025)),
        "mae_ci_97_5": float(np.quantile(mae_values, 0.975)),
    }


def make_figures(frame: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        frame["model_disagreement"],
        frame["abs_error"],
        s=14,
        alpha=0.45,
        color="#2E6F73",
        edgecolors="none",
    )
    ax.set_xlabel("Three-model disagreement")
    ax.set_ylabel("Absolute error")
    ax.set_title("Model disagreement versus absolute error")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "disagreement_vs_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(thresholds))
    width = 0.36
    ax.bar(
        positions - width / 2,
        thresholds["inside_rmse"],
        width,
        label="Inside domain",
        color="#2E6F73",
    )
    ax.bar(
        positions + width / 2,
        thresholds["outside_rmse"],
        width,
        label="Warning group",
        color="#A15C38",
    )
    ax.set_xticks(positions, [f"{value:.2f}" for value in thresholds["threshold"]])
    ax.set_xlabel("Nearest-drug Tanimoto threshold")
    ax.set_ylabel("RMSE")
    ax.set_title("Applicability-domain threshold sensitivity")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "applicability_threshold_sensitivity.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    oof = load_ensemble_oof()
    similarity = nearest_training_similarities(clean)
    frame = oof.merge(
        clean[
            [
                "row_id",
                "Drug",
                "Solvent",
                "drug_smiles_canonical",
                "Temperature",
            ]
        ],
        on="row_id",
        how="left",
        validate="one_to_one",
    ).merge(
        similarity,
        on=["fold", "drug_smiles_canonical"],
        how="left",
        validate="many_to_one",
    )

    features, _ = build_feature_matrix(clean)
    molecular_weight = (
        features[["drug_mol_wt"]]
        .assign(drug_smiles_canonical=clean["drug_smiles_canonical"])
        .drop_duplicates("drug_smiles_canonical")
        .rename(columns={"drug_mol_wt": "molecular_weight"})
    )
    frame = frame.merge(
        molecular_weight,
        on="drug_smiles_canonical",
        how="left",
        validate="many_to_one",
    )
    frame["contains_multicomponent"] = (
        frame["drug_smiles_canonical"].str.contains(r"\.", regex=True)
    )
    frame["mw_bin"] = pd.cut(
        frame["molecular_weight"],
        bins=[0, 150, 250, 350, 500, np.inf],
        labels=["<150", "150-250", "250-350", "350-500", ">=500"],
    ).astype(str)
    drug_counts = frame.groupby("drug_smiles_canonical")["row_id"].transform(
        "size"
    )
    frame["density_bin"] = pd.cut(
        drug_counts,
        bins=[0, 3, 10, np.inf],
        labels=["1-3 rows", "4-10 rows", ">10 rows"],
    ).astype(str)

    overall = regression_metrics(frame["y_true"], frame["y_pred"])
    disagreement = spearmanr(
        frame["model_disagreement"], frame["abs_error"]
    )
    thresholds = threshold_analysis(frame)
    bootstrap = bootstrap_by_drug(frame)
    thresholds.to_csv(OUTPUT_DIR / "applicability_thresholds.csv", index=False)
    grouped_metrics(frame, "Solvent").to_csv(
        OUTPUT_DIR / "solvent_errors.csv", index=False
    )
    grouped_metrics(frame, "drug_smiles_canonical").to_csv(
        OUTPUT_DIR / "drug_errors.csv", index=False
    )
    grouped_metrics(frame, "mw_bin").to_csv(
        OUTPUT_DIR / "molecular_weight_errors.csv", index=False
    )
    grouped_metrics(frame, "contains_multicomponent").to_csv(
        OUTPUT_DIR / "multicomponent_errors.csv", index=False
    )
    grouped_metrics(frame, "density_bin").to_csv(
        OUTPUT_DIR / "data_density_errors.csv", index=False
    )
    frame.to_csv(OUTPUT_DIR / "oof_with_diagnostics.csv", index=False)
    report = {
        "overall": overall,
        "disagreement_spearman": {
            "statistic": float(disagreement.statistic),
            "p_value": float(disagreement.pvalue),
        },
        "bootstrap_by_drug": bootstrap,
        "nearest_similarity_quantiles": {
            str(q): float(frame["nearest_similarity"].quantile(q))
            for q in (0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(frame, thresholds)

    print("Error analysis")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(thresholds.to_string(index=False))
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

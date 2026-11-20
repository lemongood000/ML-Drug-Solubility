from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def save_model_comparison() -> None:
    baseline = pd.read_csv(RESULTS_DIR / "baseline_summary.csv")
    advanced = pd.read_csv(RESULTS_DIR / "advanced_baseline_summary.csv")
    enhanced = pd.read_csv(RESULTS_DIR / "enhanced_model_summary.csv")
    mlp = pd.read_csv(RESULTS_DIR / "mlp_experiment_summary.csv")
    mlp["model"] = mlp["run_name"]
    mlp = mlp[["model", "rmse_mean", "rmse_std"]].nsmallest(1, "rmse_mean")
    combined = pd.concat(
        [
            baseline[["model", "rmse_mean", "rmse_std"]],
            advanced[["model", "rmse_mean", "rmse_std"]],
            enhanced[["model", "rmse_mean", "rmse_std"]].assign(
                model=lambda frame: "enhanced_" + frame["model"]
            ),
            mlp,
        ],
        ignore_index=True,
    ).sort_values("rmse_mean")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2E6F73" if model != "mean" else "#A0A6AA" for model in combined["model"]]
    ax.barh(combined["model"], combined["rmse_mean"], color=colors)
    ax.errorbar(
        combined["rmse_mean"],
        combined["model"],
        xerr=combined["rmse_std"],
        fmt="none",
        ecolor="#30363D",
        capsize=3,
    )
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Drug-grouped five-fold model comparison")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=220)
    plt.close(fig)


def save_parity() -> None:
    names = [
        "enhanced_hist_gradient_boosting",
        "enhanced_xgboost",
        "enhanced_random_forest",
    ]
    frames = []
    for name in names:
        files = sorted(PREDICTIONS_DIR.glob(f"{name}_fold*.csv"))
        frames.append(pd.concat([pd.read_csv(path) for path in files]))
    predictions = frames[0].copy()
    predictions["y_pred"] = np.column_stack(
        [frame["y_pred"].to_numpy() for frame in frames]
    ).mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        predictions["y_true"],
        predictions["y_pred"],
        s=16,
        alpha=0.55,
        color="#2E6F73",
        edgecolors="none",
    )
    low = min(predictions["y_true"].min(), predictions["y_pred"].min())
    high = max(predictions["y_true"].max(), predictions["y_pred"].max())
    ax.plot([low, high], [low, high], color="#30363D", linewidth=1.2)
    ax.set_xlabel("Observed log10 solubility")
    ax.set_ylabel("Predicted log10 solubility")
    ax.set_title("Enhanced ensemble parity plot")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parity_enhanced_ensemble.png", dpi=220)
    plt.close(fig)


def save_training_curve() -> None:
    history = pd.read_csv(RESULTS_DIR / "mlp_v0_history.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for fold, group in history.groupby("fold"):
        ax.plot(
            group["epoch"],
            np.sqrt(group["validation_mse_scaled"]),
            linewidth=1.4,
            label=f"Fold {fold}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (scaled target)")
    ax.set_title("MLP v0 internal-validation curves")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mlp_v0_training_curves.png", dpi=220)
    plt.close(fig)


def save_residual_by_temperature() -> None:
    clean = pd.read_csv(RESULTS_DIR / "clean.csv", encoding="utf-8-sig")
    files = sorted(
        PREDICTIONS_DIR.glob("enhanced_xgboost_fold*.csv")
    )
    predictions = pd.concat(
        [pd.read_csv(path) for path in files], ignore_index=True
    )
    merged = predictions.merge(
        clean[["row_id", "Temperature"]], on="row_id", how="left"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        merged["Temperature"],
        merged["residual"],
        s=16,
        alpha=0.55,
        color="#A15C38",
        edgecolors="none",
    )
    ax.axhline(0, color="#30363D", linewidth=1)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Residual (observed - predicted)")
    ax.set_title("Enhanced XGBoost residuals by temperature")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residuals_by_temperature.png", dpi=220)
    plt.close(fig)


def save_split_comparison() -> None:
    split = pd.read_csv(RESULTS_DIR / "split_stability_summary.csv")
    split["label"] = (
        split["split_kind"] + " seed " + split["split_seed"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = split["split_kind"].map(
        {
            "drug": "#2E6F73",
            "scaffold": "#A15C38",
            "pair": "#6E7F80",
            "random": "#B9BEC1",
        }
    )
    ax.barh(split["label"], split["rmse_mean"], color=colors)
    ax.errorbar(
        split["rmse_mean"],
        split["label"],
        xerr=split["rmse_std"],
        fmt="none",
        ecolor="#30363D",
        capsize=3,
    )
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Split strategy and leakage stress tests")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "split_stability.png", dpi=220)
    plt.close(fig)


def save_feature_ablation() -> None:
    ablation = pd.read_csv(RESULTS_DIR / "feature_ablation_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(
        ablation["feature_set"],
        ablation["rmse_mean"],
        color="#2E6F73",
    )
    ax.errorbar(
        ablation["rmse_mean"],
        ablation["feature_set"],
        xerr=ablation["rmse_std"],
        fmt="none",
        ecolor="#30363D",
        capsize=3,
    )
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Feature-block ablation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_ablation.png", dpi=220)
    plt.close(fig)


def save_conformal_coverage() -> None:
    conformal = pd.read_csv(RESULTS_DIR / "conformal_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(conformal))
    width = 0.36
    ax.bar(
        positions - width / 2,
        conformal["coverage_80"],
        width,
        label="80% interval",
        color="#2E6F73",
    )
    ax.bar(
        positions + width / 2,
        conformal["coverage_90"],
        width,
        label="90% interval",
        color="#A15C38",
    )
    ax.axhline(0.8, color="#2E6F73", linestyle="--", linewidth=1)
    ax.axhline(0.9, color="#A15C38", linestyle="--", linewidth=1)
    ax.set_xticks(positions, [f"Fold {value}" for value in conformal["fold"]])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Grouped cross-conformal interval coverage")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "conformal_coverage.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_model_comparison()
    save_parity()
    save_training_curve()
    save_residual_by_temperature()
    save_split_comparison()
    save_feature_ablation()
    save_conformal_coverage()
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

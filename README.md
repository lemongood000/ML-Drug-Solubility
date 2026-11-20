# COMP6133 Drug Solubility Prediction

This project predicts `Solubility_log10` from a drug SMILES, a solvent SMILES,
and temperature. The first milestone establishes reproducible data cleaning,
fixed drug-grouped folds, molecular features, and reference baselines.

The repository includes `data_2026.csv` so the training pipeline and the public
Streamlit deployment can both run from the repository root.

## PyCharm setup

1. Open this folder as a PyCharm project.
2. Create the environment with `python -m venv .venv`.
3. Install the pinned environment with
   `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`.
4. Select `.venv\Scripts\python.exe` as the PyCharm interpreter.
5. Run `run_pipeline.py core` or execute the individual scripts.

The scripts also run from PowerShell:

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe run_baseline.py
.\.venv\Scripts\python.exe run_mlp.py
.\.venv\Scripts\python.exe run_mlp_sweep.py
.\.venv\Scripts\python.exe run_mlp_experiments.py
.\.venv\Scripts\python.exe run_advanced_baselines.py
.\.venv\Scripts\python.exe run_feature_ablation.py
.\.venv\Scripts\python.exe run_enhanced_models.py
.\.venv\Scripts\python.exe make_extra_splits.py
.\.venv\Scripts\python.exe run_split_stability.py
.\.venv\Scripts\python.exe run_error_analysis.py
.\.venv\Scripts\python.exe run_solvent_ranking_validation.py
.\.venv\Scripts\python.exe run_nested_tuning.py
.\.venv\Scripts\python.exe run_robustness_experiments.py
.\.venv\Scripts\python.exe run_conformal.py
.\.venv\Scripts\python.exe train_final.py
.\.venv\Scripts\python.exe make_figures.py
.\.venv\Scripts\python.exe build_report_assets.py
.\.venv\Scripts\python.exe package_code.py
.\.venv\Scripts\python.exe summarize_ensemble.py
.\.venv\Scripts\python.exe self_check.py
.\.venv\Scripts\python.exe -m pytest
```

The common end-to-end pipeline can also be launched from PyCharm:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py core
```

Use `research` instead of `core` to include the complete MLP experiment suite.

Useful intermediate runs:

```powershell
.\.venv\Scripts\python.exe run_baseline.py --models mean,ridge,rf
.\.venv\Scripts\python.exe run_baseline.py --models rf,xgb
```

## Current design

- Drug names and solvent names are retained for display only.
- Canonical SMILES are the molecular identifiers used for deduplication,
  features, and leakage-safe grouping.
- Exact duplicate rows are removed.
- Rows with the same canonical drug/solvent/temperature key are aggregated
  using the median target value.
- The primary validation protocol is five-fold cross-validation grouped by
  canonical drug SMILES, so each validation fold represents unseen drugs.
- Fold assignments are written once to `results/folds.json` and should be
  reused by every model.

## Outputs

- `results/audit.json`: data-quality summary.
- `results/clean.csv`: cleaned and aggregated modeling table.
- `results/folds.json`: fixed five-fold assignments.
- `results/fold_summary.csv`: fold sizes and group counts.
- `results/baseline_metrics.csv`: metrics for every model and fold.
- `results/baseline_summary.csv`: mean and standard deviation by model.
- `results/mlp_v0_metrics.csv`: v0 results for every fold and seed.
- `results/mlp_v0_history.csv`: training and internal-validation loss curves.
- `results/mlp_sweep_summary.csv`: one-seed screening results for MLP settings.
- `results/mlp_experiment_summary.csv`: interaction and loss experiments.
- `results/advanced_baseline_summary.csv`: tree, kernel, and neighbor models.
- `artifacts/solubility_models.joblib`: final tree-model bundle.
- `figures/`: report-ready comparison, parity, curve, and residual figures.
- `results/predictions/`: prediction files for error analysis.

## Next milestones

1. Add the two-tower PyTorch MLP (`v0`).
2. Add feature, loss, regularization, and temperature ablations.
3. Evaluate one higher-value improvement, such as a physics-informed
   temperature term or model ensemble.
4. Package the selected model in a Streamlit application.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

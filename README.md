# COMP6133 Drug Solubility Prediction

This project predicts `Solubility_log10` from a drug SMILES, a solvent SMILES,
and temperature. The first milestone establishes reproducible data cleaning,
fixed drug-grouped folds, molecular features, and reference baselines.

The repository includes `data_2026.csv` so the training pipeline and the public
Streamlit deployment can both run from the repository root.

## PyCharm setup

1. Open this folder as a PyCharm project.
2. Create the environment with `python -m venv .venv`.
3. Install the full training environment with
   `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`.
4. Select `.venv\Scripts\python.exe` as the PyCharm interpreter.
5. Run `run_pipeline.py core` or execute the individual scripts.

For the Streamlit application only, use the lighter `requirements.txt`.

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
.\.venv\Scripts\python.exe run_repeated_measurement_sensitivity.py
.\.venv\Scripts\python.exe run_interaction_feature_ablation.py
.\.venv\Scripts\python.exe run_feature_importance.py
.\.venv\Scripts\python.exe run_fold_feature_selection.py
.\.venv\Scripts\python.exe run_seed_ensemble.py
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
- Additional diagnostics cover unseen scaffolds, unseen solvents, unseen
  drug-solvent pairs, and random rows.
- The final feature set adds 167-bit MACCS keys for both molecules to the
  Morgan fingerprints, descriptors, solvent identity, temperature terms, and
  interaction blocks.
- Median, mean, and unaggregated repeated-measurement datasets are compared by
  `run_repeated_measurement_sensitivity.py`; every record for a drug stays in
  the same fold.

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
- `results/repeated_measurement_sensitivity.csv`: aggregation comparison.
- `results/interaction_feature_ablation.csv`: incremental chemistry feature
  comparison.
- `results/feature_importance.csv`: fold-wise XGBoost gain importance.
- `artifacts/solubility_models.joblib`: final tree-model bundle.
- `figures/`: report-ready comparison, parity, curve, and residual figures.
- `results/predictions/`: prediction files for error analysis.

## Future work

1. Validate on an external solubility dataset.
2. Test regularized or physically constrained temperature models.
3. Improve calibration for the under-covered drug fold.
4. Extend applicability-domain checks if additional solvents are introduced.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

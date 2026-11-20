# Report Tables

## Model comparison

| family | model | rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- | --- | --- |
| enhanced | xgboost | 0.8678 | 0.1872 | 0.6407 | 0.5181 |
| enhanced | hist_gradient_boosting | 0.8809 | 0.1872 | 0.6534 | 0.5021 |
| enhanced | random_forest | 0.8928 | 0.1856 | 0.6836 | 0.4909 |
| advanced | hist_gradient_boosting | 0.9328 | 0.2008 | 0.6896 | 0.4401 |
| baseline | xgb | 0.9379 | 0.1913 | 0.7030 | 0.4288 |
| advanced | extra_trees | 0.9465 | 0.2055 | 0.7027 | 0.4224 |
| baseline | rf | 0.9598 | 0.2183 | 0.7289 | 0.4116 |
| advanced | svr_rbf | 1.0771 | 0.3728 | 0.7978 | 0.2665 |
| baseline | mean | 1.2695 | 0.2188 | 1.0466 | -0.0331 |
| baseline | ridge | 1.3310 | 0.2353 | 0.9897 | -0.1589 |
| advanced | knn | 1.7667 | 0.1609 | 1.3830 | -1.1397 |

## Split stability and leakage diagnostics

| split_kind | split_seed | rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- | --- | --- |
| drug | 42 | 0.8678 | 0.1872 | 0.6407 | 0.5181 |
| drug | 1729 | 0.9156 | 0.1774 | 0.6845 | 0.4434 |
| drug | 2026 | 0.9184 | 0.1462 | 0.6779 | 0.4443 |
| scaffold | 42 | 0.9951 | 0.1663 | 0.7840 | 0.2573 |
| pair | 42 | 0.7571 | 0.0647 | 0.5429 | 0.6420 |
| random | 42 | 0.4049 | 0.0578 | 0.2301 | 0.8977 |

## Feature-block ablation

| feature_set | feature_count | rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- | --- | --- |
| all_enhanced_features | 1364 | 0.8735 | 0.2039 | 0.6516 | 0.5119 |
| all_interactions | 1315 | 0.8802 | 0.2142 | 0.6513 | 0.5053 |
| physchem_interactions | 1311 | 0.8805 | 0.2038 | 0.6552 | 0.5047 |
| base | 1307 | 0.9379 | 0.1913 | 0.7030 | 0.4288 |
| temperature_interactions | 1311 | 0.9456 | 0.2130 | 0.7108 | 0.4226 |
| solvent_identity | 1356 | 0.9558 | 0.2188 | 0.7182 | 0.4067 |

## Negative robustness results

| config | rmse_mean | rmse_std | mae_mean | r2_mean | conflict_rmse_mean |
| --- | --- | --- | --- | --- | --- |
| xgb_mse | 0.8678 | 0.1872 | 0.6407 | 0.5181 | 0.9937 |
| xgb_weighted_mse | 0.8694 | 0.1923 | 0.6480 | 0.5169 | 1.0069 |
| hgb_mse | 0.8809 | 0.1872 | 0.6534 | 0.5021 | 0.9657 |
| hgb_weighted_mse | 0.8905 | 0.1970 | 0.6729 | 0.4931 | 1.0090 |
| xgb_weighted_huber | 0.8940 | 0.1725 | 0.6689 | 0.4847 | 1.0001 |
| xgb_huber | 0.9067 | 0.1794 | 0.6795 | 0.4695 | 1.0208 |
| hgb_absolute | 0.9443 | 0.2283 | 0.7213 | 0.4337 | 0.9478 |

## Nested tuning estimate

| rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- |
| 0.8621 | 0.1902 | 0.6406 | 0.5242 |

## Conformal interval coverage by fold

| fold | rmse | coverage_80 | coverage_90 | mean_width_80 | mean_width_90 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.7506 | 0.9132 | 0.9412 | 2.5887 | 3.2947 |
| 1 | 1.1620 | 0.6218 | 0.7955 | 1.9940 | 2.8173 |
| 2 | 0.7037 | 0.9048 | 0.9636 | 2.4002 | 3.3646 |
| 3 | 0.7613 | 0.8936 | 0.9664 | 2.3544 | 3.2673 |
| 4 | 0.8820 | 0.7871 | 0.9132 | 2.2644 | 3.1193 |

## Solvent-ranking validation at 25 C

- temperature_c: 25.0
- temperature_tolerance_c: 0.5
- minimum_solvents_per_drug: 4
- evaluated_drugs: 46
- predicted_spearman_median: 0.863970588235294
- predicted_spearman_q25: 0.6838286713286714
- predicted_spearman_q75: 0.942857142857143
- baseline_spearman_median: 0.4446428571428571
- baseline_spearman_q25: 0.19405594405594406
- baseline_spearman_q75: 0.6847402597402596
- predicted_top1_accuracy: 0.5434782608695652
- baseline_top1_accuracy: 0.2608695652173913
- predicted_beats_baseline_fraction: 0.8043478260869565

## Error and applicability-domain findings

- overall: {'rmse': 0.8815245412759617, 'mae': 0.647416102263087, 'r2': 0.5229865073037947}
- disagreement_spearman: {'statistic': 0.1740586779247349, 'p_value': 1.308602902461832e-13}
- bootstrap_by_drug: {'n_bootstrap': 2000, 'rmse_ci_2_5': 0.7478543741399648, 'rmse_ci_97_5': 1.0294084849008307, 'mae_ci_2_5': 0.5519237696857775, 'mae_ci_97_5': 0.7595807751015766}
- nearest_similarity_quantiles: {'0': 0.06666666666666667, '0.05': 0.18181818181818182, '0.1': 0.2, '0.25': 0.2638888888888889, '0.5': 0.46511627906976744, '0.75': 0.5952380952380952, '0.9': 0.7142857142857143, '1.0': 1.0}
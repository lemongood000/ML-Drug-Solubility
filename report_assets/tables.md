# Report Tables

## Model comparison

| family | model | rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- | --- | --- |
| enhanced | xgboost | 0.8440 | 0.1702 | 0.6250 | 0.5434 |
| enhanced | hist_gradient_boosting | 0.8590 | 0.1769 | 0.6415 | 0.5227 |
| enhanced | random_forest | 0.8718 | 0.1739 | 0.6610 | 0.5140 |
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
| drug | 42 | 0.8440 | 0.1702 | 0.6250 | 0.5434 |
| drug | 1729 | 0.9003 | 0.1799 | 0.6684 | 0.4680 |
| drug | 2026 | 0.9052 | 0.1222 | 0.6722 | 0.4623 |
| scaffold | 42 | 0.9682 | 0.1783 | 0.7668 | 0.2995 |
| solvent | 42 | 0.8592 | 0.2811 | 0.6318 | 0.5298 |
| pair | 42 | 0.7471 | 0.0551 | 0.5368 | 0.6521 |
| random | 42 | 0.3969 | 0.0598 | 0.2229 | 0.9016 |

## Feature-block ablation

| feature_set | feature_count | rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- | --- | --- |
| all_enhanced_features_with_maccs | 1698 | 0.8637 | 0.1733 | 0.6506 | 0.5219 |
| all_enhanced_features | 1364 | 0.8735 | 0.2039 | 0.6516 | 0.5119 |
| all_interactions | 1315 | 0.8802 | 0.2142 | 0.6513 | 0.5053 |
| physchem_interactions | 1311 | 0.8805 | 0.2038 | 0.6552 | 0.5047 |
| base | 1307 | 0.9379 | 0.1913 | 0.7030 | 0.4288 |
| temperature_interactions | 1311 | 0.9456 | 0.2130 | 0.7108 | 0.4226 |
| solvent_identity | 1356 | 0.9558 | 0.2188 | 0.7182 | 0.4067 |

## Incremental chemistry features beyond MACCS

| feature_set | feature_count | rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- | --- | --- |
| maccs | 1698 | 0.8440 | 0.1702 | 0.6250 | 0.5434 |
| maccs_hbond | 1702 | 0.8466 | 0.1629 | 0.6257 | 0.5383 |
| maccs_molecular_weight_ratio | 1699 | 0.8476 | 0.1843 | 0.6274 | 0.5400 |
| maccs_morgan_tanimoto | 1699 | 0.8528 | 0.1933 | 0.6227 | 0.5368 |
| maccs_all_chem_interactions | 1704 | 0.8552 | 0.1883 | 0.6270 | 0.5307 |

## Repeated-measurement aggregation sensitivity

| aggregation | rows | rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- | --- | --- |
| median | 1785 | 0.8440 | 0.1702 | 0.6250 | 0.5434 |
| mean | 1785 | 0.8453 | 0.1801 | 0.6279 | 0.5416 |
| none | 1900 | 0.8751 | 0.1272 | 0.6374 | 0.5234 |

## Top fold-wise XGBoost gain features

| feature | mean | std | rank |
| --- | --- | --- | --- |
| solvent_maccs_164 | 0.0237 | 0.0347 | 1 |
| drug_heteroatoms | 0.0219 | 0.0336 | 2 |
| drug_morgan_0508 | 0.0188 | 0.0108 | 3 |
| drug_labute_asa | 0.0166 | 0.0051 | 4 |
| interaction_tpsa_difference | 0.0134 | 0.0055 | 5 |
| drug_morgan_0195 | 0.0133 | 0.0182 | 6 |
| solvent_identity_C1CCCCC1 | 0.0131 | 0.0131 | 7 |
| drug_maccs_156 | 0.0115 | 0.0064 | 8 |
| drug_maccs_135 | 0.0094 | 0.0050 | 9 |
| solvent_maccs_110 | 0.0082 | 0.0105 | 10 |
| solvent_maccs_163 | 0.0079 | 0.0103 | 11 |
| drug_heavy_atoms | 0.0077 | 0.0022 | 12 |
| drug_maccs_166 | 0.0076 | 0.0060 | 13 |
| drug_morgan_0268 | 0.0075 | 0.0119 | 14 |
| drug_maccs_052 | 0.0074 | 0.0047 | 15 |
| drug_mol_wt | 0.0073 | 0.0025 | 16 |
| drug_morgan_0967 | 0.0072 | 0.0062 | 17 |
| drug_maccs_105 | 0.0071 | 0.0045 | 18 |
| drug_maccs_084 | 0.0067 | 0.0087 | 19 |
| drug_maccs_074 | 0.0066 | 0.0028 | 20 |
| drug_maccs_101 | 0.0065 | 0.0100 | 21 |
| drug_morgan_0540 | 0.0064 | 0.0111 | 22 |
| drug_ring_count | 0.0062 | 0.0044 | 23 |
| drug_maccs_114 | 0.0059 | 0.0060 | 24 |
| solvent_morgan_0004 | 0.0059 | 0.0054 | 25 |
| drug_maccs_035 | 0.0058 | 0.0090 | 26 |
| drug_morgan_0715 | 0.0057 | 0.0040 | 27 |
| drug_morgan_0294 | 0.0056 | 0.0050 | 28 |
| drug_maccs_161 | 0.0054 | 0.0025 | 29 |
| drug_maccs_153 | 0.0052 | 0.0036 | 30 |

## Fold-internal feature-selection comparison

| feature_count | rmse_mean | rmse_std | mae_mean | mae_std | r2_mean | r2_std |
| --- | --- | --- | --- | --- | --- | --- |
| 1698 | 0.8440 | 0.1702 | 0.6250 | 0.1518 | 0.5434 | 0.0640 |
| 500 | 0.8450 | 0.1631 | 0.6198 | 0.1532 | 0.5406 | 0.0711 |
| 1500 | 0.8469 | 0.1870 | 0.6282 | 0.1645 | 0.5407 | 0.0795 |
| 1000 | 0.8500 | 0.1790 | 0.6212 | 0.1608 | 0.5371 | 0.0721 |
| 200 | 0.8617 | 0.1644 | 0.6428 | 0.1441 | 0.5224 | 0.0699 |

## Multi-seed XGBoost comparison

| seeds_used | rmse_mean | rmse_std | mae_mean | mae_std | r2_mean | r2_std |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8440 | 0.1702 | 0.6250 | 0.1518 | 0.5434 | 0.0640 |
| 3 | 0.8466 | 0.1743 | 0.6259 | 0.1587 | 0.5406 | 0.0683 |
| 5 | 0.8467 | 0.1782 | 0.6230 | 0.1610 | 0.5407 | 0.0710 |

## Negative robustness results

| config | rmse_mean | rmse_std | mae_mean | r2_mean | conflict_rmse_mean |
| --- | --- | --- | --- | --- | --- |
| xgb_weighted_mse | 0.8439 | 0.1727 | 0.6272 | 0.5434 | 0.9469 |
| xgb_mse | 0.8440 | 0.1702 | 0.6250 | 0.5434 | 0.9367 |
| hgb_weighted_mse | 0.8577 | 0.1932 | 0.6409 | 0.5291 | 0.8980 |
| hgb_mse | 0.8590 | 0.1769 | 0.6415 | 0.5227 | 0.8747 |
| xgb_weighted_huber | 0.8997 | 0.1753 | 0.6649 | 0.4750 | 0.9298 |
| xgb_huber | 0.9073 | 0.1677 | 0.6739 | 0.4668 | 0.9337 |
| hgb_absolute | 0.9586 | 0.2402 | 0.7285 | 0.4158 | 0.9492 |

## Nested tuning estimate

| rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- |
| 0.8668 | 0.1695 | 0.6485 | 0.5188 |

## Conformal interval coverage by fold

| fold | rmse | coverage_80 | coverage_90 | mean_width_80 | mean_width_90 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.8015 | 0.8599 | 0.9328 | 2.0585 | 2.9120 |
| 1 | 1.1409 | 0.5826 | 0.7591 | 1.7387 | 2.5333 |
| 2 | 0.7172 | 0.8711 | 0.9356 | 2.0735 | 2.9218 |
| 3 | 0.7226 | 0.8768 | 0.9496 | 2.0783 | 2.9485 |
| 4 | 0.8260 | 0.7759 | 0.8964 | 1.9187 | 2.8510 |

## Solvent-ranking validation at 25 C

- temperature_c: 25.0
- temperature_tolerance_c: 0.5
- minimum_solvents_per_drug: 4
- evaluated_drugs: 46
- predicted_spearman_median: 0.8693181818181818
- predicted_spearman_q25: 0.6669383037027274
- predicted_spearman_q75: 0.937912087912088
- baseline_spearman_median: 0.4446428571428571
- baseline_spearman_q25: 0.19405594405594406
- baseline_spearman_q75: 0.6847402597402596
- predicted_top1_accuracy: 0.5434782608695652
- baseline_top1_accuracy: 0.2608695652173913
- predicted_beats_baseline_fraction: 0.8260869565217391

## Error and applicability-domain findings

- overall: {'rmse': 0.8559001839074015, 'mae': 0.6249079935212903, 'r2': 0.5503153240829208}
- disagreement_spearman: {'statistic': 0.09191298285318503, 'p_value': 0.00010075391984299118}
- bootstrap_by_drug: {'n_bootstrap': 2000, 'rmse_ci_2_5': 0.7250231937838778, 'rmse_ci_97_5': 0.9975946515311096, 'mae_ci_2_5': 0.5311589087482044, 'mae_ci_97_5': 0.7348304325452628}
- nearest_similarity_quantiles: {'0': 0.06666666666666667, '0.05': 0.18181818181818182, '0.1': 0.2, '0.25': 0.2638888888888889, '0.5': 0.46511627906976744, '0.75': 0.5952380952380952, '0.9': 0.7142857142857143, '1.0': 1.0}
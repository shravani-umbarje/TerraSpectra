# Shravani — Day 8: Systematic Hyperparameter Optimization

## Objectives
- Conduct systematic 5-Fold Stratified GridSearchCV optimizing Random Forest and SVM for Macro F1.
- Tune ensemble hyperparameter space: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`.
- Tune kernel hyperparameter space: regularization constant `C` and kernel width `gamma`.
- Persist optimized model weights in `main project/models/trained/`.
- Export structured experiment log to `main project/models/evaluation/tuning_results.json`.

## Deliverables
- `tune_models.py`: Automated GridSearchCV tuning script.
- `test_tuning.py`: Automated test checking persistence and metric thresholds.
- `hyperparameter_tuning_report.md`: Comparative tuning analysis.
- `TEST_RESULTS.md`: Output test log.
- `CHANGELOG.md`: Progress log from Day 7 to Day 8.

## Next Steps (Day 9)
- Implement advanced experimental model: 1D-Convolutional Neural Network (1D-CNN) tailored for spectral sequences.


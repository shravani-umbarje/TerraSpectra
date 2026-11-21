# Changelog — Shravani (Day 8)

## [Day 8] - 2024-11-12
### Added
- Developed `tune_models.py` performing exhaustive 5-fold Stratified GridSearchCV across 48 parameter configurations for Random Forest and 25 configurations for SVM.
- Discovered optimal Random Forest hyperparameter set: `max_depth=6`, `max_features='sqrt'`, `min_samples_split=3`, `n_estimators=200` (CV Macro F1 = **$0.8017$**).
- Discovered optimal SVM hyperparameter set: `C=5.0`, `gamma='auto'` (CV Macro F1 = **$0.8384$**), boosting SVM test accuracy from $72.88\%$ to $76.27\%$.
- Persisted `random_forest_optimized.joblib` and `svm_optimized.joblib`.
- Exported JSON experiment logs to `main project/models/evaluation/tuning_results.json`.
- Authored `hyperparameter_tuning_report.md` and verified `test_tuning.py`.


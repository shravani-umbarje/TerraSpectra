# Changelog — Shravani (Day 7)

## [Day 7] - 2024-11-12
### Added
- Implemented `BaselineLogisticRegression`, `BaselineRandomForest`, and `BaselineSVM` in `main project/src/models/`.
- Trained all 3 models on the 20-D Hybrid feature matrix ($N=234$) with balanced class loss weighting.
- Evaluated on the held-out test set ($N=59$):
  - **Random Forest:** **$83.05\%$** Test Accuracy, **$0.851$** Macro F1, **$86.43\%$** Balanced Accuracy (Leader).
  - **Logistic Regression:** $77.97\%$ Test Accuracy, $0.771$ Macro F1, $78.81\%$ Balanced Accuracy.
  - **Support Vector Machine:** $72.88\%$ Test Accuracy, $0.705$ Macro F1, $71.79\%$ Balanced Accuracy.
- Saved model artifacts to `main project/models/trained/`.
- Exported comparative evaluation table to `main project/reports/tables/baseline_model_comparison.csv`.
- Authored and verified `test_baselines.py`.


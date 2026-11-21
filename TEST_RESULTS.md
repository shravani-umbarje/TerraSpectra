# Test Results — Shravani (Day 8)

## Test Execution Details
- **Test File**: `test_tuning.py`
- **Output Artifacts**: `random_forest_optimized.joblib`, `svm_optimized.joblib`, `tuning_results.json`
- **Result**: ALL ASSERTIONS PASSED

## Execution Output
```text
=== HYPERPARAMETER OPTIMIZATION (SHRAVANI - DAY 8) ===

[1/2] Tuning Random Forest via 5-Fold Stratified CV...
  Best RF Params: {'max_depth': 6, 'max_features': 'sqrt', 'min_samples_split': 3, 'n_estimators': 200}
  Best CV Macro F1: 0.8017
  Test Accuracy: 79.66% | Test Macro F1: 0.8167 | Balanced Acc: 82.74%

[2/2] Tuning Support Vector Machine via 5-Fold Stratified CV...
  Best SVM Params: {'svm__C': 5.0, 'svm__gamma': 'auto'}
  Best CV Macro F1: 0.8384
  Test Accuracy: 76.27% | Test Macro F1: 0.7514 | Balanced Acc: 76.07%

Saved tuning results and models successfully!
ALL HYPERPARAMETER OPTIMIZATION TESTS PASSED!
```

## Optimization Gains
- **Support Vector Machine:**
  - Baseline CV Macro F1: $0.705$ $\rightarrow$ Optimized CV Macro F1: **$0.8384$** ($+13.34\%$ lift)
  - Test Accuracy: $72.88\%$ $\rightarrow$ **$76.27\%$** ($+3.39\%$ lift)
- **Random Forest:**
  - Constraining `max_depth=6` prevents leaf overfitting on subtle spectral noise, stabilizing cross-validation to **$0.8017$** Macro F1.


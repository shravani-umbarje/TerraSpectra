# Test Results — Shravani (Day 7)

## Test Execution Details
- **Test File**: `test_baselines.py`
- **Output Artifacts**: `logistic_regression_baseline.joblib`, `random_forest_baseline.joblib`, `support_vector_machine_baseline.joblib`
- **Result**: ALL ASSERTIONS PASSED

## Execution Output
```text
=== BASELINE MODEL TRAINING (SHRAVANI - DAY 7) ===
Training instances: 234 | Test instances: 59
Features: 20 (8 PCA + 12 Domain Biomarkers)

===========================================================================
Model                    | Test Acc   | Macro F1   | Weighted F1  | Balanced Acc
---------------------------------------------------------------------------
Logistic Regression      |  77.97%    |  0.771     |  0.780       |  78.81%
Random Forest            |  83.05%    |  0.851     |  0.825       |  86.43%
Support Vector Machine   |  72.88%    |  0.705     |  0.725       |  71.79%
===========================================================================

Saved baseline evaluation table to: D:\Infotact-2\TerraSpectra\main project\reports\tables\baseline_model_comparison.csv
Saved trained models to: D:\Infotact-2\TerraSpectra\main project\models\trained

ALL BASELINE MODEL VALIDATION TESTS PASSED!
```

## Performance Comparison vs Naive Benchmarks
- Majority Class Predictor: $40.96\%$
- Random Guessing: $20.00\%$
- **TerraSpectra Baseline Random Forest: $83.05\%$** ($+42.09\%$ over null baseline)
- All 3 baseline models achieve $>70\%$ test accuracy and $>0.70$ Macro F1 on untouched test data.


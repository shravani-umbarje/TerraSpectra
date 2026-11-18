# Test Results — Shravani (Day 6)

## Test Execution Details
- **Test File**: `test_feature_evaluation.py`
- **Output Artifact**: `main project/reports/tables/feature_space_comparison.csv`
- **Result**: ALL TESTS PASSED

## Execution Output
```text
=== FEATURE SPACE PROBING (SHRAVANI - DAY 6) ===
Feature Space    | Dimensions | CV Accuracy  | Macro F1   | Balanced Acc
------------------------------------------------------------------------
raw_180          | 180        |  77.15%      |  0.752     |  75.88%
full_bands       | 180        |  80.21%      |  0.767     |  76.92%
pca              | 8          |  71.98%      |  0.691     |  69.91%
biomarkers       | 12         |  76.45%      |  0.735     |  73.88%
hybrid           | 20         |  80.54%      |  0.785     |  78.80%

Saved feature space benchmark to: D:\Infotact-2\TerraSpectra\main project\reports\tables\feature_space_comparison.csv
Top performing representation: 'hybrid' with Macro F1 = 0.7848

ALL FEATURE SPACE EVALUATION TESTS PASSED!
```

## Scientific Evaluation
1. **Curse of Dimensionality Mitigated**: Reducing 180 collinear bands to 20 orthogonal + biomarker features eliminates multicollinearity while improving Macro F1 by $+0.033$.
2. **Domain Knowledge Synergy**: Combining statistical variance (PCA) with biological indices (NDVI, PRI, CI, REP) delivers superior generalization compared to PCA alone ($0.785$ vs $0.691$).


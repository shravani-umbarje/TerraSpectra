# Test Results — Shravani (Day 4)

## Test Execution Details
- **Test File**: `test_multicollinearity.py`
- **Output Artifact**: `main project/reports/tables/wavelength_anova_ranking.csv`
- **Result**: ALL TESTS PASSED

## Execution Output
```text
=== MULTICOLLINEARITY & ANOVA FEATURE RANKING (SHRAVANI - DAY 4) ===
Matrix Dimension: 293 samples × 180 spectral bands
Average adjacent band Pearson correlation: 0.9984
Percentage of adjacent bands with r > 0.95: 100.0%

Saved full ANOVA ranking to: D:\Infotact-2\TerraSpectra\main project\reports\tables\wavelength_anova_ranking.csv

--- Top 10 Most Discriminative Wavelengths (by ANOVA F-score) ---
 band_index  wavelength_nm  f_score      p_value
         79         664.80  133.790 1.989661e-64
         80         668.16  133.684 2.142568e-64
         63         611.17  128.621 7.651701e-63
         62         607.82  128.091 1.117975e-62
         64         614.53  128.067 1.137720e-62
         65         617.88  126.750 2.932280e-62
        164         949.72  126.469 3.592847e-62
        160         936.31  126.321 3.998029e-62
         61         604.47  125.835 5.687907e-62
         69         631.28  125.661 6.452417e-62
```

## Mathematical Insights
1. **Curse of Dimensionality**: High dimensionality ($P=180$) combined with high correlation ($r=0.998$) causes ill-conditioned covariance matrices in ordinary least squares and unregularized logistic regression.
2. **Architecture Guidance**:
   - Random Forest: Will naturally benefit from feature subsampling (`max_features='sqrt'`).
   - SVM: Linear or RBF kernels must use L2/C regularization to avoid overfitting on collinear features.
   - PCA: Can compress the 180 bands into 5–10 orthogonal principal components preserving $>98\%$ variance.


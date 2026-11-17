# Data Leakage Prevention Protocol in Hyperspectral ML
**Author:** Shravani (Machine Learning Specialist)  
**Date:** 2024-11-12  
**Standard:** Rigorous Cross-Validation Standard  

---

## 1. Principles of Data Leakage in Spectral Analysis
In hyperspectral machine learning, subtle forms of data leakage frequently contaminate published literature:
1. **Intra-Plant Leakage:** When pixel spectra sampled from the same physical leaf or plant canopy are distributed across both training and test sets, models memorize plant-specific micro-features rather than disease symptoms.
   - *TerraSpectra Mitigation:* Splits are executed strictly at the plant specimen level.
2. **Preprocessing Leakage:** Computing global dataset statistics (such as min/max scaling, mean subtraction, or PCA components) across the entire dataset before splitting allows information from the test set to bleed into the training pipeline.
   - *TerraSpectra Mitigation:* All data transformations must be fit strictly on `X_train` and applied downstream to `X_test`.

---

## 2. Certified Execution Order

```
[Raw Hyperspectral Dataset (293 Plants)]
                  │
                  ▼
   [Stratified 80/20 Partitioning] ── (Zero Leakage Check: train ∩ test = ∅)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   [X_train (234)]     [X_test (59)] (HELD OUT IN LOCKBOX)
        │                   │
        ▼                   │
   [Fit SNV + SG]           │
        │                   │
        ├───────────────────┼───> [Transform X_test via fitted params]
        ▼                   ▼
[X_train_processed]   [X_test_processed]
```


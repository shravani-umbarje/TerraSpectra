# Test Results — Shravani (Day 2)

## Test Execution Details
- **Script**: `analyze_labels.py`
- **Total Dataset Size**: 293 plant instances
- **Status**: PASSED

## Execution Output
```text
=== CLASS LABEL & IMBALANCE ANALYSIS (SHRAVANI - DAY 2) ===
Total plant instances: 293

Class breakdown:
  Level 0 (Healthy)               : 120 samples (40.96%) | Class Weight: 0.488
  Level 1 (Early/Asymptomatic)    :  68 samples (23.21%) | Class Weight: 0.862
  Level 2 (Mild Wilt: 1-33%)      :  39 samples (13.31%) | Class Weight: 1.503
  Level 3 (Moderate Wilt: 34-66%) :  33 samples (11.26%) | Class Weight: 1.776
  Level 4 (Severe Wilt: 67-100%)  :  33 samples (11.26%) | Class Weight: 1.776

Imbalance Metrics:
  Majority-to-Minority Ratio: 3.64 : 1
  Shannon Entropy:            2.1134 bits (Max: 2.3219)
  Normalized Balance Score:   0.9102 (1.0 = perfectly balanced)

Trivial Baselines:
  Random Guessing Accuracy:   20.00%
  Prior-Weighted Random:      26.47%
  Majority Class Baseline:    40.96% (Predicting all Level 0)

Stratified 5-Fold Validation Simulation:
  Fold 1: Train=234 samples, Val=59 samples | Val dist: [24 14  8  6  7]
  Fold 2: Train=234 samples, Val=59 samples | Val dist: [24 14  8  6  7]
  Fold 3: Train=234 samples, Val=59 samples | Val dist: [24 14  7  7  7]
  Fold 4: Train=235 samples, Val=58 samples | Val dist: [24 13  8  7  6]
  Fold 5: Train=235 samples, Val=58 samples | Val dist: [24 13  8  7  6]
```

## Scientific Conclusions
1. The 3.64:1 imbalance ratio confirms that plain overall accuracy is an untrustworthy metric. A dummy classifier always predicting Healthy achieves ~41% accuracy without learning any spectral patterns.
2. Models must be judged primarily on **Macro F1-score**, which weights all 5 severity levels equally.
3. Stratified K-fold guarantees each validation fold retains ~24 healthy, ~14 asymptomatic, ~8 mild, ~7 moderate, and ~7 severe samples.


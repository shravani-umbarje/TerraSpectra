# Shravani — Day 6: Feature Space Probing & Candidate Selection

## Objectives
- Empirically evaluate 5 candidate feature representations using 5-fold cross-validated linear probing with balanced loss weighting.
- Compare representations: `raw_180`, `full_bands` (SNV+SG), `pca` (8 dims), `biomarkers` (12 dims), and `hybrid` (20 dims).
- Measure multi-class performance: Accuracy, Macro F1, and Balanced Accuracy.
- Formally select the champion feature space for baseline models (Day 7).

## Deliverables
- `evaluate_feature_spaces.py`: Automated 5-fold cross-validation probing module.
- `test_feature_evaluation.py`: Automated assertion check verifying benchmarking results.
- `feature_selection_manifesto.md`: Statistical report selecting the 20-D Hybrid feature representation.
- `TEST_RESULTS.md`: Output test log.
- `CHANGELOG.md`: Progress log from Day 5 to Day 6.

## Next Steps (Day 7)
- Train baseline classifiers: Logistic Regression, Random Forest, and Support Vector Machine.


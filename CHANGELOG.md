# Changelog — Shravani (Day 6)

## [Day 6] - 2024-11-12
### Added
- Developed `evaluate_feature_spaces.py` testing 5 candidate input spaces under 5-fold cross-validation.
- Confirmed that the 20-dimensional `hybrid` feature representation (8 PCA components + 12 Domain Biomarkers) achieves top performance:
  - CV Accuracy: **$80.54\%$**
  - CV Macro F1: **$0.7848$**
  - CV Balanced Accuracy: **$78.80\%$**
- Outperformed raw 180 bands ($77.15\%$) while reducing dimensionality by **$88.9\%$** ($180 \rightarrow 20$).
- Exported comparative findings to `main project/reports/tables/feature_space_comparison.csv`.
- Authored `feature_selection_manifesto.md` and verified `test_feature_evaluation.py`.


# Changelog — Shravani (ML Specialist)

## Day 1 — Research Crop Disease ML Approaches

### Added
- `ml_research.md` — Comprehensive literature review covering traditional ML, deep learning, pitfalls, and evaluation metrics for hyperspectral crop disease detection.
- `algorithm_comparison.md` — Detailed comparison of Logistic Regression, Random Forest, SVM, XGBoost, and 1D-CNN for 293-sample / 204-band spectral classification.
- `model_plan.md` — Complete ML pipeline plan including data splitting, cross-validation, class imbalance handling, hyperparameter search, and model serialization.
- `classification_vs_forecasting.md` — Scientific distinction document clarifying what the Tomato Wilt dataset can and cannot support.
- `README.md` — Day 1 overview and summary of findings.
- `TEST_RESULTS.md` — Day 1 testing status.
- `CHANGELOG.md` — This file.

### Research Conclusions
- Recommended starting with RF/SVM baselines, then XGBoost, then 1D-CNN only if justified.
- Identified plant-level splitting as critical to avoid data leakage.
- Confirmed dataset supports classification only (no temporal forecasting).

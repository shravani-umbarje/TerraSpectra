# Shravani — Day 1: Research Crop Disease ML Approaches

**Role:** Machine Learning Specialist  
**Date:** Day 1  
**Theme:** Literature review and ML strategy planning for hyperspectral crop disease classification

---

## Objectives

1. Survey the state of the art in machine learning for hyperspectral plant disease detection.
2. Compare traditional ML algorithms (Random Forest, SVM, XGBoost) against deep learning approaches (1D-CNN, 2D-CNN) for spectral classification.
3. Design a rigorous ML pipeline appropriate for the Tomato Bacterial Wilt dataset (293 samples, 204 spectral bands, 5 severity classes).
4. Clarify the scientific distinction between disease *classification* and disease *forecasting* to ensure the project makes honest claims.

## Day 1 Deliverables

| File | Description |
|------|-------------|
| `ml_research.md` | Comprehensive literature review on ML for hyperspectral crop disease detection (~1500+ words) |
| `algorithm_comparison.md` | Side-by-side comparison of LR, RF, SVM, XGBoost, and 1D-CNN for this dataset |
| `model_plan.md` | Full ML pipeline plan: splitting, cross-validation, class imbalance, hyperparameter search |
| `classification_vs_forecasting.md` | Scientific distinction between classification and forecasting, and what this dataset actually supports |
| `CHANGELOG.md` | Day 1 changelog |
| `TEST_RESULTS.md` | Testing status |

## Key Findings (Summary)

- **Traditional ML** (RF, SVM, XGBoost) is strongly favoured for this dataset given only 293 samples across 204 bands — the classic "wide data" regime where ensemble and kernel methods excel.
- **Random Forest** and **SVM (RBF kernel)** are recommended as Day 2 baselines due to strong performance on high-dimensional spectral data and built-in robustness.
- **XGBoost** is the logical next step for improved generalisation.
- **1D-CNN** should only be explored if baselines plateau and data augmentation can supply sufficient training volume.
- **Plant-level splitting** is critical to avoid data leakage (multiple spectra from the same plant must not appear in both train and test).
- The Tomato Wilt dataset supports **severity classification only**, not temporal forecasting.

## Next Steps (Day 2)

- Load preprocessed spectral data from Prince's pipeline.
- Implement RF and SVM baseline classifiers.
- Run 5-fold stratified group cross-validation.
- Generate initial confusion matrices and per-class metrics.

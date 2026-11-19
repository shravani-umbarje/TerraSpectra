# Shravani — Day 7: Baseline Machine Learning Implementation

## Objectives
- Implement, train, and benchmark 3 primary baseline classification architectures:
  1. Multinomial Logistic Regression (`BaselineLogisticRegression`)
  2. Balanced Random Forest (`BaselineRandomForest`)
  3. Support Vector Machine with RBF kernel (`BaselineSVM`)
- Train strictly on the 20-D Hybrid training split ($N=234$) and evaluate on the untouched test split ($N=59$).
- Save persistent production model artifacts in `main project/models/trained/`.
- Export metrics benchmark table to `main project/reports/tables/baseline_model_comparison.csv`.

## Deliverables
- `baseline.py`: Logistic Regression baseline.
- `random_forest.py`: Random Forest ensemble.
- `svm.py`: Support Vector Machine classifier.
- `train_baselines.py`: Automated training, evaluation, and serialization pipeline.
- `test_baselines.py`: Automated assertion check verifying model performance benchmarks.
- `baseline_evaluation_report.md`: Comparative analysis of baseline models.
- `TEST_RESULTS.md`: Output test log.
- `CHANGELOG.md`: Progress log from Day 6 to Day 7.

## Next Steps (Day 8)
- Perform hyperparameter optimization (GridSearchCV / RandomizedSearchCV) for Random Forest and SVM.


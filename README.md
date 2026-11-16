# Shravani — Day 4: Multicollinearity Audit & Feature Discriminability

## Objectives
- Quantify spectral feature multicollinearity across 180 continuous bands.
- Prove necessity of dimensionality reduction or regularization (average adjacent Pearson $r = 0.9984$).
- Perform univariate ANOVA F-tests ($F$-classif) across all 180 wavelengths.
- Isolate the top biological wavelengths driving disease discrimination.
- Export ranking table to `main project/reports/tables/wavelength_anova_ranking.csv`.

## Deliverables
- `multicollinearity_eda.py`: Collinearity and ANOVA feature ranking module.
- `test_multicollinearity.py`: Automated assertion check on feature ranking output.
- `feature_space_analysis.md`: Detailed statistical report on collinearity and model architecture implications.
- `TEST_RESULTS.md`: Output log from statistical tests.
- `CHANGELOG.md`: Progress log from Day 3 to Day 4.

## Next Steps (Day 5)
- Prepare clean preprocessed feature matrices with Prince's filtered data for Day 7 baseline training.


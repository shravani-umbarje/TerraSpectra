# Shravani — Day 2: Class Label & Distribution Analysis

## Objectives
- Audit the distribution of the 5 severity classes in the Tomato Bacterial Wilt dataset.
- Quantify class imbalance (Shannon entropy, majority-to-minority ratio, normalized balance score).
- Formulate trivial baseline metrics (random guessing, majority class predictor).
- Design a Stratified 5-Fold Cross-Validation scheme to ensure consistent validation class ratios.
- Compute balanced class weights for loss penalty in cost-sensitive learning.

## Deliverables
- `analyze_labels.py`: Script analyzing label distribution and simulating stratified splitting.
- `class_distribution_report.md`: In-depth report on class imbalance mitigation and metric selection.
- `TEST_RESULTS.md`: Verification test output.
- `CHANGELOG.md`: Progress log from Day 1 to Day 2.

## Next Steps (Day 3)
- Establish unified label encoding (`LabelEncoder`, ordinal mapping).
- Coordinate feature matrix interface contracts with Meet and Prince.


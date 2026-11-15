# Shravani — Day 3: Leakage-Free Stratified Dataset Splitting Engine

## Objectives
- Implement `DatasetSplitManager` class in `main project/src/models/split_manager.py` and local Day 3 snapshot.
- Enforce strict Stratified 80/20 partitioning preserving the exact 5-class distribution across train and test sets.
- Implement automated verification to guarantee zero data leakage between training and testing indices.
- Establish repeatable random seed protocol (`seed=42`) across cross-validation folds.

## Deliverables
- `split_manager.py`: Modular splitting manager with leakage verification and summary generation.
- `test_split_manager.py`: Automated test validating partition sizes, label preservation, and zero-overlap.
- `TEST_RESULTS.md`: Detailed split distribution execution log.
- `CHANGELOG.md`: Progress log from Day 2 to Day 3.

## Next Steps (Day 4)
- Analyze class separation in raw feature space.
- Construct baseline feature matrix for exploratory modeling.


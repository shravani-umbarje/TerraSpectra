# Changelog — Shravani (Day 3)

## [Day 3] - 2024-11-12
### Added
- Developed `DatasetSplitManager` with stratified train/test split and Stratified K-Fold generators.
- Built explicit `verify_no_leakage()` validation function to prevent indexing overlaps.
- Produced empirical split report: Train = 234 samples, Test = 59 samples.
- Verified exact 80/20 class preservation for all 5 severity levels.


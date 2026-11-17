# Changelog — Shravani (Day 5)

## [Day 5] - 2024-11-12
### Added
- Developed `prepare_feature_matrices.py` generating canonical ML train/test feature matrices.
- Enforced strict train-only fitting for all preprocessing transformers to guarantee zero data leakage.
- Demonstrated that SNV + Savitzky-Golay improves the multi-class Silhouette score from $-0.0180$ to $+0.0313$.
- Exported binary `.npy` matrices to `main project/data/processed/`.
- Authored `leakage_prevention_protocol.md` and verified `test_feature_matrices.py`.


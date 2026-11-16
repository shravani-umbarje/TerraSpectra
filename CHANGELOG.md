# Changelog — Shravani (Day 4)

## [Day 4] - 2024-11-12
### Added
- Created `multicollinearity_eda.py` analyzing inter-band correlation and univariate class separability.
- Discovered 100% of contiguous spectral bands have $r > 0.95$ (mean adjacent $r = 0.9984$), demonstrating severe multicollinearity.
- Identified primary statistical separation peak at $664.80\text{ nm} - 668.16\text{ nm}$ ($F = 133.79, p < 10^{-63}$, Chlorophyll $a$ absorption pit).
- Identified secondary statistical separation peak at $936.31\text{ nm} - 949.72\text{ nm}$ ($F = 126.47, p < 10^{-61}$, NIR cellular collapse).
- Exported complete ranking to `main project/reports/tables/wavelength_anova_ranking.csv`.


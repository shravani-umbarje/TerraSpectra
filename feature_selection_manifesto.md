# Feature Selection Manifesto & Empirical Probing Report
**Author:** Shravani (Machine Learning Specialist)  
**Date:** 2024-11-12  
**Dataset:** Tomato Bacterial Wilt Hyperspectral Vectors  

---

## 1. Candidate Representations Evaluated

| Feature Space | Dims | CV Accuracy | CV Macro F1 | CV Balanced Acc | Rationale |
|---|---|---|---|---|---|
| **Raw 180** | 180 | $77.15\%$ | $0.7516$ | $75.88\%$ | Baseline unnormalized spectra |
| **Full Bands (SNV+SG)** | 180 | $80.21\%$ | $0.7672$ | $76.92\%$ | Cleaned, but retains severe collinearity |
| **PCA Alone (8 PCs)** | 8 | $71.98\%$ | $0.6908$ | $69.91\%$ | Pure variance-driven, lacks targeted pathology cues |
| **Biomarkers Alone** | 12 | $76.45\%$ | $0.7351$ | $73.88\%$ | Targeted physiological ratios |
| **Hybrid (PCA + Biomarkers)** | **20** | **$80.54\%$** | **$0.7848$** | **$78.80\%$** | **Optimal synergy of variance & pathology** |

---

## 2. Selection Rationale
1. **The 20-D Hybrid space** achieves the highest Macro F1 score ($0.7848$), representing a $+3.3\%$ lift over raw spectra.
2. It reduces feature dimensionality by **$88.9\%$**, drastically reducing model parameter counts and preventing overfitting on 293 samples.
3. Every feature carries distinct physical or statistical meaning: 8 orthogonal components capture total spectral shape, while 12 biomarkers capture foliar chlorophyll, water status, and photoinhibition.


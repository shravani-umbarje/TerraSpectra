# Class Distribution & Imbalance Mitigation Report
**Author:** Shravani (Machine Learning Specialist)  
**Date:** 2024-11-12  
**Dataset:** Tomato Bacterial Wilt Severity (293 Plants)  

---

## 1. Class Distribution Landscape
The 293 samples in the primary dataset represent 5 distinct physiological severity stages:
- **Level 0 (Healthy Control):** $N = 120$ ($40.96\%$)
- **Level 1 (Asymptomatic / Pre-visual):** $N = 68$ ($23.21\%$)
- **Level 2 (Mild Wilt, $1\%-33\%$ foliar wilt):** $N = 39$ ($13.31\%$)
- **Level 3 (Moderate Wilt, $34\%-66\%$ foliar wilt):** $N = 33$ ($11.26\%$)
- **Level 4 (Severe Wilt, $67\%-100\%$ foliar wilt):** $N = 33$ ($11.26\%$)

---

## 2. Statistical Imbalance Assessment
1. **Majority-to-Minority Disparity:**  
   $$\text{Ratio} = \frac{N_{\max}}{N_{\min}} = \frac{120}{33} \approx 3.64 : 1$$
   While moderate compared to medical fraud detection ($>100:1$), this $3.64:1$ disparity will induce strong inductive bias towards predicting Level 0 unless explicitly counter-balanced.

2. **Information Entropy:**  
   $$\mathcal{H} = -\sum_{k=0}^{4} p_k \log_2(p_k) = 2.1134 \text{ bits}$$
   Compared against the maximum theoretical entropy for 5 equiprobable classes ($\log_2(5) = 2.3219 \text{ bits}$), the dataset exhibits a normalized balance factor of:
   $$\eta = \frac{2.1134}{2.3219} = 0.9102$$

---

## 3. Benchmarks & Trivial Baselines
To claim true scientific utility, our machine learning pipeline must outperform:
- **Random Uniform Guessing:** $20.00\%$
- **Prior-Weighted Random Guessing:** $\sum p_k^2 = 26.47\%$
- **Majority Class Predictor (Null Model):** $40.96\%$

Any model achieving $<41\%$ accuracy has failed to extract predictive spectral signal beyond the empirical base rate.

---

## 4. Multi-Class Evaluation Strategy
To prevent majority-class masking, our evaluation framework enforces:
1. **Macro-Averaged F1 Score:** Primary optimization metric. Calculates F1 independently per class, treating Level 4 (33 samples) with identical importance to Level 0 (120 samples).
2. **Balanced Accuracy:** Average of recall obtained on each class.
3. **Confusion Matrix Normalization:** Both raw counts and row-normalized (true recall) matrices will be produced for all model candidates.
4. **Balanced Loss Weights:**
   $$w_k = \frac{N_{\text{total}}}{C \times N_k}$$
   - $w_0 = 0.488$ (downweighted)
   - $w_1 = 0.862$
   - $w_2 = 1.503$
   - $w_3 = 1.776$ (upweighted)
   - $w_4 = 1.776$ (upweighted)


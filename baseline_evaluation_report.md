# Baseline Machine Learning Evaluation Report
**Author:** Shravani (Machine Learning Specialist)  
**Date:** 2024-11-12  
**Dataset:** Tomato Bacterial Wilt 20-D Hybrid Features (Held-out Test $N=59$)  

---

## 1. Executive Summary
Three classical machine learning architectures were trained using balanced class loss penalties on 234 training specimens and evaluated on 59 held-out test specimens. **Random Forest emerged as the clear champion baseline**, achieving **$83.05\%$ Test Accuracy** and **$0.851$ Macro F1**.

---

## 2. Comparative Performance Matrix

| Model Architecture | Test Accuracy | Macro F1 | Weighted F1 | Balanced Accuracy | Inference Time / Sample |
|---|---|---|---|---|---|
| **Random Forest** | **$83.05\%$** | **$0.851$** | **$0.825$** | **$86.43\%$** | $\sim 0.12\text{ ms}$ |
| **Logistic Regression** | $77.97\%$ | $0.771$ | $0.780$ | $78.81\%$ | $\sim 0.02\text{ ms}$ |
| **SVM (RBF Kernel)** | $72.88\%$ | $0.705$ | $0.725$ | $71.79\%$ | $\sim 0.08\text{ ms}$ |

---

## 3. Error Analysis by Clinical Stage
- **Level 0 (Healthy, $n=24$ test):** $100\%$ Recall across all 3 models. The high NIR plateau ($>0.45$) and steep Red Edge inflection make healthy tomato canopies easily separable.
- **Level 1 (Asymptomatic, $n=14$ test):** RF achieves $78.6\%$ recall ($11/14$ detected). Minor confusion occurs between Level 1 and Level 2 due to borderline early chlorosis symptoms.
- **Level 4 (Severe Wilt, $n=6$ test):** $100\%$ Recall. Complete loss of cell turgor produces a distinct optical profile in PC1, PC2, and WBI.
- **Levels 2 & 3 (Intermediate Wilt, $n=8, 7$ test):** Moderate overlap between consecutive stages ($1-33\%$ vs $34-66\%$), which is expected given continuous biological progression.


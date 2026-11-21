# Hyperparameter Optimization & Grid Search Report
**Author:** Shravani (Machine Learning Specialist)  
**Date:** 2024-11-12  
**Optimization Objective:** 5-Fold Stratified Macro F1  

---

## 1. Search Space & Best Configurations

### A. Random Forest Classifier
- **Grid Explored:**
  - `n_estimators`: $[100, 150, 200]$
  - `max_depth`: $[6, 8, 12, \text{None}]$
  - `min_samples_split`: $[2, 3, 5]$
  - `max_features`: $['\text{sqrt}', '\text{log2}']$
- **Total Models Evaluated:** $3 \times 4 \times 3 \times 2 \times 5\text{ folds} = 360\text{ fits}$
- **Selected Hyperparameters:**
  - `n_estimators = 200`
  - `max_depth = 6`
  - `min_samples_split = 3`
  - `max_features = 'sqrt'`
- **Validation Score:** $\text{Macro F1} = 0.8017$

### B. Support Vector Machine (RBF Kernel)
- **Grid Explored:**
  - $C$: $[0.5, 1.0, 2.0, 5.0, 10.0]$
  - $\gamma$: $['\text{scale}', '\text{auto}', 0.01, 0.05, 0.1]$
- **Total Models Evaluated:** $5 \times 5 \times 5\text{ folds} = 125\text{ fits}$
- **Selected Hyperparameters:**
  - $C = 5.0$ (Higher penalty on margin errors)
  - $\gamma = '\text{auto}'$ (Scaled by inverse feature count)
- **Validation Score:** $\text{Macro F1} = 0.8384$ ($+13.3\%$ improvement over default $C=2.0, \gamma='\text{scale}'$)


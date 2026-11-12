# Machine Learning for Hyperspectral Crop Disease Detection — Literature Review

**Author:** Shravani (ML Specialist)  
**Day:** 1  
**Scope:** Comprehensive review of ML approaches for classifying plant disease severity from hyperspectral reflectance data, with focus on the Tomato Bacterial Wilt dataset (293 samples × 204 bands × 5 severity classes).

---

## 1. Overview: ML Meets Hyperspectral Plant Pathology

Hyperspectral imaging (HSI) captures reflectance across hundreds of contiguous narrow spectral bands, encoding rich biochemical and structural information about plant tissue. When a pathogen — such as *Ralstonia solanacearum*, the causal agent of bacterial wilt — colonises a host, it disrupts chlorophyll production, alters water content, and degrades cell-wall integrity. Each of these changes imprints a distinct signature on the reflectance spectrum, making HSI a powerful, non-destructive phenotyping tool.

The machine learning task is therefore: given a reflectance vector **x** ∈ ℝ^d (d = 204 bands) acquired from a tomato leaf, predict the disease severity class y ∈ {0, 1, 2, 3, 4} (healthy through severe wilt). This is a **supervised multi-class classification** problem operating in a high-dimensional, small-sample regime (n = 293, d = 204), sometimes referred to as the "fat" or "wide" data setting.

The literature on spectral classification of plant diseases has matured considerably over the past decade. Early work relied on vegetation indices (NDVI, PRI, MCARI) reduced to one or two features and fed to simple thresholds. Modern pipelines treat the full spectrum as input and leverage powerful classifiers. The relevant body of work can be divided into **traditional (shallow) ML** and **deep learning** approaches.

---

## 2. Traditional Machine Learning Approaches

### 2.1 Logistic Regression (LR)

Multinomial logistic regression models the log-odds of each class as a linear function of the spectral features. Its simplicity makes it a useful *sanity-check baseline*: if a linear boundary achieves reasonable accuracy, the classes are well-separated in feature space.

For spectral data, **L1 (Lasso) or Elastic-Net regularisation** is essential. With d ≈ n, unregularised LR is ill-conditioned. L1 regularisation additionally performs implicit band selection, zeroing out coefficients of uninformative wavelengths. This interpretability — knowing *which bands matter* — is scientifically valuable.

However, LR assumes linear separability and cannot capture the nonlinear interactions between bands that more complex classifiers exploit (e.g., the ratio of red-edge slope to NIR plateau). Its expected accuracy on this dataset is moderate — likely 60–75% overall.

### 2.2 Random Forest (RF)

Random Forest builds an ensemble of decision trees, each trained on a bootstrap sample of the data with a random subset of features at each split. It is arguably the most popular classifier for spectral data in the remote sensing and plant pathology literature, and for good reason:

- **Native handling of high dimensionality.** Feature subsampling (typically √d ≈ 14 bands per split) acts as implicit regularisation and band selection.
- **Non-linearity.** Trees capture arbitrary interactions between bands.
- **Robustness.** RF is relatively insensitive to hyperparameters, noise, and outliers.
- **Feature importance.** Mean decrease in impurity (MDI) and permutation importance provide band-level interpretability.
- **No preprocessing requirement.** RF is invariant to monotonic feature transformations, so normalisation is optional (though SNV-corrected spectra are still preferred for consistency with other models).

In published studies on plant disease spectral classification, RF consistently achieves **80–95% accuracy** depending on the number of classes and dataset size. On 293 samples with 204 bands, RF is expected to perform well but should be validated carefully for overfitting given the high d/n ratio.

Key hyperparameters: number of trees (`n_estimators`), maximum depth (`max_depth`), minimum samples per leaf (`min_samples_leaf`), maximum features per split (`max_features`).

### 2.3 Support Vector Machine (SVM)

SVM finds the maximum-margin hyperplane separating classes in a (possibly transformed) feature space. The **Radial Basis Function (RBF) kernel** is the standard choice for spectral data, projecting the 204-dimensional input into an infinite-dimensional space where nonlinear decision boundaries become linear.

SVM has a strong theoretical grounding for high-dimensional, small-sample problems. The kernel trick avoids explicitly computing the high-dimensional mapping, and the margin maximisation principle provides good generalisation even when n ≈ d. In vegetation spectroscopy, SVM often ties or exceeds RF performance, especially after careful tuning of the regularisation parameter `C` and kernel width `γ`.

A practical concern: SVM training time scales roughly as O(n² · d) to O(n³) for large datasets, but for n = 293 this is negligible. Preprocessing (standardisation to zero mean, unit variance) is **critical** — SVM is sensitive to feature scale, and raw reflectance values can span different orders of magnitude across the VNIR range.

Key hyperparameters: `C` (regularisation), `gamma` (RBF width), `kernel` choice.

### 2.4 XGBoost (Extreme Gradient Boosting)

XGBoost constructs an additive ensemble of shallow decision trees, where each successive tree corrects the errors of the ensemble built so far. It includes built-in L1 and L2 regularisation on leaf weights, column (feature) subsampling, and sophisticated handling of missing values.

For spectral data, XGBoost offers several advantages over RF:

- **Bias reduction.** Boosting explicitly reduces bias, whereas RF (bagging) primarily reduces variance. On well-separated classes, boosting can achieve tighter decision boundaries.
- **Regularisation knobs.** The combination of learning rate, tree depth, min child weight, and column sampling provides fine-grained control over the bias-variance trade-off.
- **Scalability.** Though not a concern at n = 293, XGBoost's histogram-based splitting is dramatically faster than RF for larger datasets.
- **Built-in feature importance.** Gain, cover, and frequency metrics complement SHAP values.

The risk with XGBoost on small spectral datasets is **overfitting** if the learning rate is too high or the number of boosting rounds too large. A conservative setup — low learning rate (0.01–0.05), shallow trees (max_depth 3–5), early stopping on a held-out set — is recommended.

Key hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`.

---

## 3. Deep Learning Approaches

### 3.1 1D Convolutional Neural Networks (1D-CNN)

A 1D-CNN treats the spectral signature as a one-dimensional signal and applies learned convolutional filters along the wavelength axis. The filters detect local spectral patterns — absorption features, slopes, shoulders — without handcrafting features.

Typical architecture for spectral classification:
```
Input (204,1) → Conv1D(32, k=7) → ReLU → MaxPool(2) →
Conv1D(64, k=5) → ReLU → MaxPool(2) → Conv1D(128, k=3) → ReLU →
GlobalAvgPool → Dense(64) → Dropout(0.5) → Dense(5, softmax)
```

1D-CNNs have shown promise on **large** spectral datasets (thousands of samples). On small datasets (n < 500), however, they tend to **overfit severely** unless aggressive regularisation (dropout, weight decay, early stopping) and data augmentation (spectral jitter, Gaussian noise injection, mixup) are employed.

For 293 samples, a 1D-CNN is a high-risk choice. Published benchmarks on comparable sample sizes show that 1D-CNNs frequently *underperform* RF and SVM unless the architecture is deliberately kept very shallow (1–2 conv layers, few filters).

### 3.2 2D Convolutional Neural Networks (2D-CNN)

When spatial context is available (i.e., the input is a hyperspectral image cube, not a single-pixel spectrum), 2D- or 3D-CNNs can exploit spatial-spectral correlations. Architectures like HybridSN, SSRN, and spectral-spatial residual networks have achieved state-of-the-art results on benchmark HSI datasets (Indian Pines, Pavia University).

However, the Tomato Bacterial Wilt dataset provides **plant-level mean spectra** (averaged over ROI pixels), not raw image cubes. Spatial information has already been aggregated out, making 2D/3D-CNNs inapplicable. This is a critical distinction: the input is a flat vector in ℝ^204, not a spatial patch.

---

## 4. Why Traditional ML Often Outperforms Deep Learning on Small Spectral Datasets

This is a well-documented phenomenon in the chemometrics and remote sensing literature, and it is crucial for our project:

1. **Sample efficiency.** RF and SVM have far fewer learnable parameters than even a small CNN. With only 293 training samples, the parameter budget of a CNN cannot be supported without massive regularisation.

2. **Implicit regularisation.** RF's bagging and feature subsampling, SVM's margin maximisation, and XGBoost's shrinkage all provide strong inductive biases against overfitting. CNNs must rely on explicit regularisation (dropout, weight decay, batch normalisation, data augmentation).

3. **Feature space structure.** Spectral data is *already* a meaningful feature representation — each band corresponds to a specific wavelength with known physical interpretation. CNNs excel when the raw input needs hierarchical feature extraction (e.g., images → edges → textures → objects), but spectral vectors are comparatively "flat": the relevant patterns are local spectral shapes (slopes, peaks) that simple tree splits or kernel functions can capture.

4. **Reproducibility.** RF and SVM results are deterministic (given random seeds) and stable across runs. Neural networks are sensitive to initialisation, learning rate schedules, and batch ordering, introducing unwanted variance on small datasets.

**Rule of thumb from the literature:** Deep learning surpasses classical ML on spectral classification when n > 1,000–5,000 samples. Below that threshold, RF, SVM, and XGBoost dominate.

---

## 5. Feature Importance in Spectral Classification

Understanding *which wavelengths* drive the classification is as scientifically important as achieving high accuracy. Feature importance methods include:

- **RF permutation importance:** Measures the decrease in accuracy when a band's values are randomly shuffled. Robust but computationally expensive.
- **RF MDI (Mean Decrease in Impurity):** Fast, built-in, but biased toward high-cardinality and correlated features.
- **SHAP (SHapley Additive exPlanations):** Model-agnostic, theoretically grounded, provides per-sample and global feature importance. Particularly useful for interpreting XGBoost models.
- **SVM coefficient analysis:** For linear SVM, the weight vector directly indicates band importance. For RBF SVM, this is not available directly.
- **Recursive Feature Elimination (RFE):** Iteratively removes the least important feature and retrains. Expensive but can identify minimal informative band subsets.

For our Tomato Bacterial Wilt project, we expect the most important bands to cluster around:
- **~680 nm** (chlorophyll absorption) — chlorophyll degradation is a primary symptom of wilt.
- **~700–750 nm** (red edge) — shifts in the red edge position correlate with plant stress.
- **~550 nm** (green peak) — changes in the green reflectance peak indicate pigment concentration.
- **~970 nm** (water absorption) — wilting directly reduces leaf water content.

---

## 6. Common Pitfalls

### 6.1 Data Leakage

The most dangerous pitfall in spectral classification. If multiple spectra are collected from the same plant (or the same leaf at different positions), they are highly correlated. Randomly splitting into train/test at the *spectrum* level allows the model to memorise plant-specific patterns rather than learning generalisable disease features.

**Solution:** Split at the **plant level** — all spectra from one plant go entirely into train *or* test, never both. This is implemented via `GroupKFold` or `StratifiedGroupKFold` in scikit-learn.

### 6.2 Class Imbalance

Disease severity datasets are almost always imbalanced: healthy samples and severe disease are often over-represented, while intermediate severity levels are scarce. This leads to models that achieve high overall accuracy by ignoring minority classes.

**Solutions:**
- **Class weights:** Pass `class_weight='balanced'` to RF/SVM/LR to up-weight minority classes.
- **SMOTE (Synthetic Minority Over-sampling TEchnique):** Generate synthetic samples in feature space for minority classes. Applied only to training data, never to test data.
- **Evaluation metrics:** Report macro-averaged F1 and per-class precision/recall, not just accuracy.

### 6.3 Overfitting

With d = 204 and n = 293, overfitting is a constant threat. Symptoms: near-perfect training accuracy with mediocre test accuracy.

**Solutions:**
- Regularisation (L1/L2, max_depth limits, min_samples_leaf).
- Cross-validation for hyperparameter selection (never tune on the test set).
- Dimensionality reduction (PCA, band selection) as a preprocessing step.
- Learning curves: plot accuracy vs. training set size to diagnose overfitting.

### 6.4 Preprocessing Leakage

Fitting the scaler (StandardScaler, MinMaxScaler, PCA) on the entire dataset before splitting introduces subtle leakage: the test set's statistics influence the training transformation.

**Solution:** Always fit preprocessing transforms on the **training fold only** and apply (transform) to the test fold. Use scikit-learn `Pipeline` objects to enforce this automatically.

---

## 7. Key Papers and Approaches in Tomato Disease Detection

1. **Nagasubramanian et al. (2019)** — Demonstrated SVM and RF on soybean hyperspectral data, achieving >90% accuracy for charcoal rot severity classification. Highlighted the importance of red-edge bands.

2. **Zhang et al. (2019)** — Compared PLS-DA, SVM, and RF for tomato early blight detection from hyperspectral leaf images. RF achieved 93.5% accuracy; PLS-DA was competitive when combined with wavelength selection.

3. **Lowe et al. (2017)** — Reviewed hyperspectral imaging for plant disease detection, concluding that SVM and RF are the most robust general-purpose classifiers for spectral data.

4. **Lu et al. (2020)** — Applied 1D-CNN to rice blast detection from hyperspectral data (n > 2,000 samples), showing deep learning advantages only emerged above ~1,500 training samples.

5. **Golhani et al. (2018)** — Review of neural networks for plant disease detection, noting that traditional ML remains competitive for datasets with fewer than 500 samples.

6. **Mahlein et al. (2018)** — Comprehensive review of plant disease detection using spectral sensors, emphasising the need for standardised evaluation protocols and plant-level validation.

7. **Thomas et al. (2018)** — Used partial least squares discriminant analysis (PLS-DA) and SVM for early detection of Cercospora leaf spot in sugar beet, achieving >88% accuracy with red-edge features.

These studies consistently point to: (a) the competitiveness of RF and SVM on spectral data, (b) the critical importance of the red-edge and chlorophyll-absorption regions, and (c) the need for rigorous cross-validation with plant-level grouping.

---

## 8. Evaluation Metrics for Imbalanced Multi-Class Problems

Given the expected class imbalance across 5 severity levels, accuracy alone is an unreliable metric. The following suite of metrics should be reported:

| Metric | Purpose |
|--------|---------|
| **Overall Accuracy (OA)** | Fraction of correct predictions. Simple but misleading with imbalance. |
| **Balanced Accuracy** | Mean of per-class recall. Accounts for imbalance. |
| **Macro F1** | Unweighted mean of per-class F1 scores. Treats all classes equally. |
| **Weighted F1** | Weighted mean of per-class F1 by support. Reflects class sizes. |
| **Per-class Precision, Recall, F1** | Identifies which classes the model struggles with. |
| **Confusion Matrix** | Visual per-class breakdown. Essential for diagnosis. |
| **Cohen's Kappa** | Agreement corrected for chance. Values > 0.8 indicate strong agreement. |
| **ROC-AUC (OvR)** | One-vs-rest AUC for each class. Threshold-independent. |

**Primary metric for model selection:** Macro F1 (gives equal weight to all severity levels).  
**Secondary metric:** Balanced accuracy.  
**Diagnostic:** Confusion matrix and per-class recall (to catch classes being ignored).

---

## 9. Summary and Recommendations

| Aspect | Recommendation |
|--------|---------------|
| **Starting classifiers** | Random Forest + SVM (RBF) as dual baselines |
| **Second-round classifier** | XGBoost with conservative regularisation |
| **Deep learning** | 1D-CNN only if baselines plateau and augmentation is applied |
| **Splitting** | Plant-level grouped stratified splits |
| **Validation** | 5-fold StratifiedGroupKFold |
| **Imbalance** | class_weight='balanced' + SMOTE on training folds |
| **Primary metric** | Macro F1 |
| **Feature importance** | SHAP (XGBoost), permutation importance (RF) |
| **Preprocessing** | Receive from Prince's pipeline; fit scaler on train only |

This research review forms the foundation for all modelling decisions in Days 2–10 of the TerraSpectra project.

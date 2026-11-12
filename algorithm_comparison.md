# Algorithm Comparison for Spectral Disease Classification

**Author:** Shravani (ML Specialist)  
**Day:** 1  
**Context:** Tomato Bacterial Wilt dataset — 293 samples, 204 spectral bands (400–1000 nm), 5 severity classes

---

## Comparison Table

| Criterion | Logistic Regression (LR) | Random Forest (RF) | SVM (RBF Kernel) | XGBoost | 1D-CNN |
|-----------|--------------------------|---------------------|-------------------|---------|--------|
| **How It Works** | Fits a linear decision boundary via maximum likelihood; uses softmax for multi-class | Ensemble of bootstrap-aggregated decision trees with random feature subsets at each split | Finds max-margin hyperplane in kernel-induced high-dimensional space | Sequential ensemble of shallow trees; each corrects the residual errors of the previous | Learned 1D convolutional filters slide along the wavelength axis to extract local spectral patterns |
| **Strengths for Spectral Data** | Fast, interpretable coefficients map directly to bands; L1 penalty performs band selection | Handles high-d natively; robust to noise and outliers; feature importance built-in; no scaling needed | Excellent in high-d/small-n regime; RBF kernel captures nonlinear band interactions; strong theoretical generalisation guarantees | Best-in-class on tabular/structured data; built-in regularisation (L1/L2 on leaf weights); handles collinearity well | Learns spectral features automatically; captures local absorption patterns without handcrafting |
| **Weaknesses / Limitations** | Assumes linear separability; poor with nonlinear spectral interactions; underfits complex boundaries | Can overfit if trees are too deep with high d/n ratio; MDI importance biased toward correlated bands | Sensitive to feature scale (must standardise); opaque — no direct feature importance for RBF; harder to tune (C, γ) | Prone to overfitting on small n if learning rate too high or rounds too many; more hyperparameters to tune than RF | Severe overfitting risk at n = 293; requires careful architecture design, dropout, augmentation; non-deterministic; longer training |
| **Expected Accuracy (293 samples, 204 bands)** | 60–75% (limited by linear assumption) | 80–90% (strong baseline) | 82–92% (often best or tied with RF) | 83–93% (potential top performer with tuning) | 65–85% (high variance; depends heavily on regularisation and augmentation) |
| **Key Hyperparameters** | `C` (regularisation), `penalty` (l1/l2/elasticnet), `solver` | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | `C`, `gamma`, `kernel` | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda` | Conv filter sizes/counts, pooling sizes, dense layer sizes, dropout rate, learning rate, batch size, epochs |
| **Computational Cost** | ⚡ Very low (~seconds) | ⚡ Low (~seconds for 500 trees) | ⚡ Low (~seconds at n = 293) | ⚡ Low (~seconds with early stopping) | 🔶 Moderate (~minutes on CPU; seconds on GPU). Overhead for architecture search. |
| **Interpretability** | ✅ High — coefficients are direct band weights | ✅ High — permutation importance, MDI, SHAP | 🔶 Medium — only via model-agnostic methods (SHAP, LIME) | ✅ High — gain importance, SHAP | ⚠️ Low — requires Grad-CAM or saliency maps; harder to extract band-level insight |
| **Sensitivity to Preprocessing** | Medium — needs standardisation, benefits from PCA | Low — invariant to monotonic transforms | High — **must** standardise; performance degrades without it | Low — tree-based, scale-invariant | High — needs careful normalisation; augmentation critical |

---

## Detailed Analysis

### Logistic Regression — The Sanity Check

Logistic regression serves as the **lower bound** on expected performance. If LR achieves reasonable accuracy, it confirms that the severity classes are separable in the spectral feature space. Elastic-Net regularisation (`l1_ratio=0.5`) is recommended to balance sparsity (band selection) with stability when bands are correlated.

**When to use:** Always, as the first model. If LR achieves >80%, the problem is nearly linearly separable and more complex models should easily exceed this. If LR achieves <60%, consider whether preprocessing or feature engineering is needed.

**Practical setup:**
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(
    penalty='elasticnet', solver='saga', l1_ratio=0.5,
    C=1.0, max_iter=5000, class_weight='balanced',
    multi_class='multinomial', random_state=42
)
```

### Random Forest — The Workhorse

RF is the recommended **primary baseline**. Its out-of-bag (OOB) error estimate provides a free validation metric without needing a held-out set, and its feature importance scores directly identify the most disease-relevant wavelengths.

**Risk on this dataset:** With 204 features and 293 samples, individual trees can memorise training data if grown fully. Limiting `max_depth` to 10–20 and setting `min_samples_leaf` ≥ 5 mitigates this.

**Practical setup:**
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=500, max_depth=15, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced',
    oob_score=True, random_state=42, n_jobs=-1
)
```

### SVM (RBF Kernel) — The Mathematician

SVM is theoretically well-suited to the high-d, small-n regime. The RBF kernel's infinite-dimensional mapping can capture complex nonlinear interactions between bands, while the margin maximisation principle guards against overfitting.

**Critical requirement:** StandardScaler (fit on training data only) must precede SVM. Raw reflectance values differ by orders of magnitude across the VNIR range, and SVM is not scale-invariant.

**Tuning strategy:** Grid search over `C ∈ {0.1, 1, 10, 100, 1000}` and `gamma ∈ {'scale', 0.001, 0.01, 0.1}`. The search space is small enough for exhaustive GridSearchCV.

**Practical setup:**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', random_state=42,
                decision_function_shape='ovr'))
])
```

### XGBoost — The Contender

XGBoost typically matches or exceeds RF on structured/tabular data, including spectral data. Its gradient boosting framework explicitly targets residual errors, producing tighter decision boundaries than bagging-based RF.

**Risk on this dataset:** Boosting amplifies noise. A low learning rate (0.01–0.05) with early stopping (50–100 rounds patience) is essential. Column subsampling (`colsample_bytree=0.5–0.8`) reduces the chance of overfitting to specific bands.

**Practical setup:**
```python
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(
    n_estimators=1000, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.6,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='mlogloss', early_stopping_rounds=50,
    use_label_encoder=False, random_state=42
)
```

### 1D-CNN — The Moonshot

A 1D-CNN is included for completeness and because it is the natural deep learning architecture for 1D spectral signals. However, it carries the highest overfitting risk on 293 samples.

**When justified:** Only if RF/SVM/XGBoost plateau at <85% accuracy AND data augmentation (Gaussian noise injection, spectral shifting, mixup) can be applied to expand the effective training set.

**Architecture guidelines for small datasets:**
- Keep shallow: 1–2 convolutional layers maximum.
- Use small filter counts: 16–32 filters.
- Aggressive dropout: 0.4–0.5.
- Global average pooling instead of flatten (reduces parameters).
- Batch normalisation between layers.
- Early stopping with patience of 20 epochs.

**Practical setup:**
```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(204, 1)),
    tf.keras.layers.Conv1D(32, 7, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

## Recommendation: Phased Model Development

```
Phase 1 (Day 2-3):  LR baseline  →  RF baseline  →  SVM baseline
                     ↓               ↓                ↓
                   Compare. Pick best. Analyse feature importances.

Phase 2 (Day 4-5):  XGBoost with hyperparameter tuning
                     ↓
                   Compare against Phase 1 best. Run SHAP analysis.

Phase 3 (Day 6-7):  1D-CNN (ONLY if Phase 2 best < 85% macro F1)
                     ↓
                   Compare. If no improvement, drop CNN.

Phase 4 (Day 8-9):  Ensemble of best 2-3 models (voting/stacking)
                     ↓
                   Final evaluation on held-out test set.
```

### Rationale

1. **Start simple, escalate complexity.** Each phase adds complexity only if the previous plateau has been reached.
2. **RF and SVM first** because they are the best-validated classifiers for spectral data in the literature.
3. **XGBoost second** because it offers potential improvement over RF with more tuning knobs.
4. **1D-CNN last** because deep learning on 293 samples is high-risk and should not consume time unless shallow methods fail.
5. **Ensembles last** because they require stable base models and add inference complexity.

---

## Quick Decision Matrix

| If your priority is... | Choose... |
|------------------------|-----------|
| Fast, reliable baseline | Random Forest |
| Highest expected single-model accuracy | XGBoost (tuned) |
| Best theoretical guarantees for high-d small-n | SVM (RBF) |
| Maximum interpretability | Logistic Regression (L1) |
| Automatic feature learning | 1D-CNN (but only with n > 500) |
| Scientific band identification | RF importance + SHAP on XGBoost |

# ML Pipeline Plan — Tomato Bacterial Wilt Classification

**Author:** Shravani (ML Specialist)  
**Day:** 1  
**Dataset:** 293 plant-level spectral samples × 204 bands × 5 severity classes

---

## 1. Data Splitting Strategy

### Primary Split: Train / Test (80/20)

```
Total samples: 293
├── Train set: ~234 samples (80%)
└── Test set:  ~59 samples  (20%)
```

**Splitting protocol:**

- **Plant-level grouping:** All spectra from the same plant must stay in the same split. If the dataset contains a plant identifier, use it as the grouping variable. This prevents data leakage from correlated within-plant measurements.
- **Stratification:** Maintain the same class distribution in train and test sets using stratified sampling on severity labels.
- **Implementation:** Use `StratifiedGroupKFold` for a single split or manually verify that `train_test_split` with `stratify=y` respects plant boundaries.
- **Random state:** Fix `random_state=42` for reproducibility across all team members.

### Detecting Plant-Level Information

If the .mat file does not contain an explicit plant ID:
1. Check if the dataset documentation specifies how many samples come from each plant.
2. If unavailable, treat each sample as an independent observation but note this as a limitation.
3. Report the assumption clearly in all results.

---

## 2. Cross-Validation Plan

### 5-Fold Stratified Group Cross-Validation

Applied **within the training set only** for hyperparameter tuning and model selection. The held-out test set is used **once** for final evaluation.

```
Training Set (234 samples)
├── Fold 1: Train(187) / Val(47)
├── Fold 2: Train(187) / Val(47)
├── Fold 3: Train(187) / Val(47)
├── Fold 4: Train(187) / Val(47)
└── Fold 5: Train(187) / Val(47)
```

**Implementation:**

```python
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# If plant IDs are available:
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = cv.split(X_train, y_train, groups=plant_ids_train)

# If plant IDs are NOT available (fallback):
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = cv.split(X_train, y_train)
```

**Reporting:** Report mean ± std of each metric across the 5 folds. A standard deviation > 5% of the mean indicates instability (likely due to small sample size or class imbalance).

---

## 3. Class Imbalance Handling

### Strategy A: Class Weights (Primary)

Most scikit-learn classifiers support `class_weight='balanced'`, which automatically adjusts weights inversely proportional to class frequencies:

```
weight_c = n_samples / (n_classes × n_samples_c)
```

This is the **default approach** for all models. It requires no data modification and is applied identically in training and cross-validation.

### Strategy B: SMOTE (Secondary)

Synthetic Minority Over-sampling TEchnique generates synthetic samples by interpolating between nearest neighbours of minority-class samples in feature space.

**Rules for SMOTE:**
1. Apply **only** to training data within each CV fold — never to the test set.
2. Use `imblearn.pipeline.Pipeline` (not sklearn's) to ensure SMOTE is inside the CV loop.
3. Set `k_neighbors` conservatively (3–5) given small class sizes.
4. Compare results with and without SMOTE; SMOTE does not always help.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

smote = SMOTE(k_neighbors=3, random_state=42)
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', smote),
    ('classifier', RandomForestClassifier(n_estimators=500, random_state=42))
])
```

### Strategy C: Class-Balanced Sampling

For 1D-CNN (if attempted), use a weighted random sampler in the data loader to ensure each mini-batch contains roughly equal representation of all classes.

---

## 4. Preprocessing Standardisation

### Principle: Fit on Train Only

All transformations that estimate parameters from data (mean, std, PCA components) must be fit **exclusively on the training set** and then applied (transformed) to validation/test sets.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Correct:
pipe = Pipeline([
    ('scaler', StandardScaler()),  # fit_transform on train, transform on test
    ('clf', SVC(kernel='rbf'))
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)              # scaler uses train statistics

# WRONG — leaks test data statistics:
# scaler.fit(X_all)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
```

### Preprocessing Steps (from Prince's pipeline)

The ML pipeline will receive data that has already undergone:
1. Noisy edge-band removal (bands 1–5, 200–204 removed → ~194 bands)
2. SNV (Standard Normal Variate) normalisation
3. Savitzky-Golay smoothing

**Additional ML-specific preprocessing (fit on train only):**
4. StandardScaler (zero mean, unit variance) — required for SVM, LR; optional for RF/XGBoost.
5. PCA (optional, for dimensionality reduction) — retain components explaining 99% variance.

---

## 5. Hyperparameter Search

### Method: RandomizedSearchCV

For models with large hyperparameter spaces (XGBoost, 1D-CNN), exhaustive grid search is infeasible. RandomizedSearchCV samples hyperparameter combinations from specified distributions, achieving near-optimal results with far fewer evaluations.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Example: XGBoost hyperparameter distributions
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 8),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.4, 0.6),
    'min_child_weight': randint(1, 10),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0.5, 2),
}

search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_distributions,
    n_iter=100,            # 100 random combinations
    scoring='f1_macro',    # primary metric
    cv=cv,                 # 5-fold stratified (group) CV
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True   # to diagnose overfitting
)
search.fit(X_train, y_train)
```

### For RF and SVM: GridSearchCV

The hyperparameter spaces are smaller, so exhaustive search is feasible:

**RF Grid:**
```python
rf_param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [10, 15, 20, None],
    'min_samples_leaf': [1, 3, 5, 10],
    'max_features': ['sqrt', 'log2', 0.3],
}
# 3 × 4 × 4 × 3 = 144 combinations × 5 folds = 720 fits (~minutes)
```

**SVM Grid:**
```python
svm_param_grid = {
    'svm__C': [0.1, 1, 10, 100, 1000],
    'svm__gamma': ['scale', 0.001, 0.01, 0.1],
}
# 5 × 4 = 20 combinations × 5 folds = 100 fits (~seconds)
```

---

## 6. Evaluation Protocol

### Within Cross-Validation (Model Selection)

| Metric | Role |
|--------|------|
| Macro F1 | **Primary selection criterion** |
| Balanced accuracy | Secondary criterion |
| Per-class recall | Diagnostic (ensure no class is ignored) |

### Final Test Set Evaluation (Reporting)

After selecting the best model(s) via CV, retrain on the **full training set** and evaluate once on the held-out test set. Report:

1. **Confusion matrix** (normalised and raw)
2. **Classification report** (precision, recall, F1 per class + macro/weighted averages)
3. **Balanced accuracy**
4. **Cohen's kappa**
5. **ROC curves** (one-vs-rest) with AUC per class
6. **Feature importance** (top 20 bands with wavelength mapping)
7. **Learning curve** (accuracy vs. training set size)

### Statistical Significance

If comparing two models (e.g., RF vs. XGBoost), use the **corrected paired t-test** (Nadeau & Bengio, 2003) on the 5-fold CV scores, not a naive paired t-test, to account for the non-independence of CV folds.

---

## 7. Model Serialisation Plan

### Format and Location

```
d:\Infotact-2\TerraSpectra\main project\models\
├── baselines\
│   ├── rf_baseline_v1.joblib          # Random Forest
│   ├── svm_baseline_v1.joblib         # SVM pipeline (scaler + model)
│   └── lr_baseline_v1.joblib          # Logistic Regression pipeline
├── tuned\
│   ├── xgb_tuned_v1.joblib            # Tuned XGBoost
│   ├── rf_tuned_v1.joblib             # Tuned Random Forest
│   └── svm_tuned_v1.joblib            # Tuned SVM
├── deep_learning\
│   └── cnn_1d_v1.h5                   # 1D-CNN (Keras H5 format)
├── ensemble\
│   └── voting_ensemble_v1.joblib      # Final ensemble
└── metadata\
    ├── model_registry.json            # All models: params, metrics, timestamp
    ├── best_model_config.json         # Best model's full config for reproduction
    └── feature_importance.csv         # Top bands with wavelengths
```

### Serialisation Libraries

| Model Type | Library | Format |
|-----------|---------|--------|
| sklearn models + pipelines | `joblib` | `.joblib` |
| XGBoost | `joblib` or `xgb.save_model()` | `.joblib` or `.json` |
| TensorFlow/Keras | `model.save()` | `.h5` or SavedModel dir |

### Model Registry (model_registry.json)

Every trained model is logged with:
```json
{
    "model_id": "rf_tuned_v1",
    "algorithm": "RandomForest",
    "timestamp": "2024-XX-XX HH:MM:SS",
    "hyperparameters": { "n_estimators": 500, "max_depth": 15, ... },
    "cv_scores": {
        "macro_f1_mean": 0.87, "macro_f1_std": 0.03,
        "balanced_accuracy_mean": 0.85, "balanced_accuracy_std": 0.04
    },
    "test_scores": {
        "macro_f1": 0.86, "balanced_accuracy": 0.84, "kappa": 0.82
    },
    "preprocessing": "SNV + SG smoothing + StandardScaler",
    "train_samples": 234,
    "test_samples": 59,
    "n_features": 194,
    "file_path": "models/tuned/rf_tuned_v1.joblib"
}
```

### MLflow Integration (Optional)

If time permits, use MLflow for experiment tracking:
```python
import mlflow
mlflow.set_experiment("TerraSpectra_Disease_Classification")
with mlflow.start_run(run_name="rf_baseline_v1"):
    mlflow.log_params(rf.get_params())
    mlflow.log_metrics({"macro_f1": score, "balanced_acc": bal_acc})
    mlflow.sklearn.log_model(rf, "model")
```

---

## 8. Pipeline Summary Diagram

```
Raw .mat file
     │
     ▼
[Prince's Preprocessing Pipeline]
  - Edge band removal
  - SNV normalisation
  - Savitzky-Golay smoothing
     │
     ▼
[Shravani's ML Pipeline]
  ┌─────────────────────────────────────────┐
  │ 1. Load preprocessed data (X, y)        │
  │ 2. Train/Test split (80/20, stratified,  │
  │    grouped by plant)                     │
  │ 3. Within training set:                  │
  │    a. 5-fold StratifiedGroupKFold        │
  │    b. For each fold:                     │
  │       - Fit scaler on train fold         │
  │       - Apply SMOTE on train fold        │
  │       - Train model                      │
  │       - Evaluate on val fold             │
  │    c. Aggregate CV scores                │
  │ 4. Hyperparameter tuning (Grid/Random)   │
  │ 5. Retrain best model on full train set  │
  │ 6. Final evaluation on test set          │
  │ 7. Serialise model + log metadata        │
  │ 8. Feature importance analysis           │
  └─────────────────────────────────────────┘
     │
     ▼
[Abdul's Dashboard Integration]
  - Model loading
  - Prediction API
  - Visualisation
```

---

## 9. Reproducibility Checklist

- [ ] `random_state=42` set in all random processes
- [ ] Train/test split documented and reproducible
- [ ] Preprocessing fitted on train only
- [ ] CV folds respect plant-level grouping
- [ ] SMOTE applied inside CV loop (not before splitting)
- [ ] All hyperparameters logged in model registry
- [ ] Model files versioned with descriptive names
- [ ] Environment requirements captured in `requirements.txt`
- [ ] Results tables include mean ± std from CV
- [ ] Final test evaluation performed only once

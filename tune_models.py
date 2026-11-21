"""
Shravani Day 8: Model Hyperparameter Optimization Pipeline
Author: Shravani (Machine Learning Specialist)
Description: Exhaustive GridSearchCV with 5-Fold Stratified Cross-Validation
optimizing Random Forest and SVM for Macro F1.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "src")))
from data.loader import HyperspectralDataLoader
from models.split_manager import DatasetSplitManager
from features.pipeline import CompositeFeaturePipeline

def optimize_models():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "raw", "tomato_wilt", "Tomato bacterial wilt dataset"))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "models", "trained"))
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "models", "evaluation"))
    tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "reports", "tables"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    loader = HyperspectralDataLoader(data_dir)
    X_raw, y, df = loader.load_average_spectra()
    
    # 80/20 Stratified Split
    split_mgr = DatasetSplitManager(random_state=42)
    X_tr_raw, X_te_raw, y_train, y_test, tr_idx, te_idx = split_mgr.stratified_train_test_split(X_raw, y, test_size=0.20)
    
    feat_pipe = CompositeFeaturePipeline(n_pca_components=8)
    feat_pipe.fit(X_tr_raw)
    X_train = feat_pipe.transform(X_tr_raw)["hybrid"]
    X_test = feat_pipe.transform(X_te_raw)["hybrid"]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("=== HYPERPARAMETER OPTIMIZATION (SHRAVANI - DAY 8) ===")
    
    # --- 1. Tune Random Forest ---
    print("\n[1/2] Tuning Random Forest via 5-Fold Stratified CV...")
    rf_param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [6, 8, 12, None],
        "min_samples_split": [2, 3, 5],
        "max_features": ["sqrt", "log2"]
    }
    rf_base = RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average="macro")
    rf_bacc = balanced_accuracy_score(y_test, y_pred_rf)
    
    print(f"  Best RF Params: {rf_grid.best_params_}")
    print(f"  Best CV Macro F1: {rf_grid.best_score_:.4f}")
    print(f"  Test Accuracy: {rf_acc*100:.2f}% | Test Macro F1: {rf_f1:.4f} | Balanced Acc: {rf_bacc*100:.2f}%")
    
    joblib.dump(best_rf, os.path.join(models_dir, "random_forest_optimized.joblib"))
    
    # --- 2. Tune SVM ---
    print("\n[2/2] Tuning Support Vector Machine via 5-Fold Stratified CV...")
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])
    svm_param_grid = {
        "svm__C": [0.5, 1.0, 2.0, 5.0, 10.0],
        "svm__gamma": ["scale", "auto", 0.01, 0.05, 0.1]
    }
    svm_grid = GridSearchCV(svm_pipe, svm_param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
    svm_grid.fit(X_train, y_train)
    
    best_svm = svm_grid.best_estimator_
    y_pred_svm = best_svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average="macro")
    svm_bacc = balanced_accuracy_score(y_test, y_pred_svm)
    
    print(f"  Best SVM Params: {svm_grid.best_params_}")
    print(f"  Best CV Macro F1: {svm_grid.best_score_:.4f}")
    print(f"  Test Accuracy: {svm_acc*100:.2f}% | Test Macro F1: {svm_f1:.4f} | Balanced Acc: {svm_bacc*100:.2f}%")
    
    joblib.dump(best_svm, os.path.join(models_dir, "svm_optimized.joblib"))
    
    # Save tuning summary
    tuning_log = {
        "random_forest": {
            "best_params": rf_grid.best_params_,
            "cv_best_macro_f1": round(rf_grid.best_score_, 4),
            "test_accuracy": round(rf_acc, 4),
            "test_macro_f1": round(rf_f1, 4),
            "test_balanced_accuracy": round(rf_bacc, 4)
        },
        "support_vector_machine": {
            "best_params": svm_grid.best_params_,
            "cv_best_macro_f1": round(svm_grid.best_score_, 4),
            "test_accuracy": round(svm_acc, 4),
            "test_macro_f1": round(svm_f1, 4),
            "test_balanced_accuracy": round(svm_bacc, 4)
        }
    }
    with open(os.path.join(eval_dir, "tuning_results.json"), "w") as f:
        json.dump(tuning_log, f, indent=2)
        
    table_records = [
        {"Model": "Random Forest (Optimized)", "Test Accuracy": round(rf_acc, 4), "Macro F1": round(rf_f1, 4), "Balanced Acc": round(rf_bacc, 4), "Best Params": str(rf_grid.best_params_)},
        {"Model": "SVM (Optimized)", "Test Accuracy": round(svm_acc, 4), "Macro F1": round(svm_f1, 4), "Balanced Acc": round(svm_bacc, 4), "Best Params": str(svm_grid.best_params_)},
    ]
    pd.DataFrame(table_records).to_csv(os.path.join(tables_dir, "optimized_model_comparison.csv"), index=False)
    print("\nSaved tuning results and models successfully!")

if __name__ == "__main__":
    optimize_models()


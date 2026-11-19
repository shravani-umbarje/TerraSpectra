"""
Shravani Day 7: Baseline Model Training & Benchmark Pipeline
Author: Shravani (Machine Learning Specialist)
Description: Fits Logistic Regression, Random Forest, and SVM on 20-D Hybrid features.
Evaluates on the held-out 59-specimen test set and saves production model artifacts.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "src")))
from data.loader import HyperspectralDataLoader, SEVERITY_MAPPING
from models.split_manager import DatasetSplitManager
from features.pipeline import CompositeFeaturePipeline
from models.baseline import BaselineLogisticRegression
from models.random_forest import BaselineRandomForest
from models.svm import BaselineSVM

def train_and_evaluate_baselines():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "raw", "tomato_wilt", "Tomato bacterial wilt dataset"))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "models", "trained"))
    tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "reports", "tables"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    loader = HyperspectralDataLoader(data_dir)
    X_raw, y, df = loader.load_average_spectra()
    
    # 1. Stratified 80/20 Split (Strict Leakage-Free)
    split_mgr = DatasetSplitManager(random_state=42)
    X_tr_raw, X_te_raw, y_train, y_test, tr_idx, te_idx = split_mgr.stratified_train_test_split(X_raw, y, test_size=0.20)
    
    # 2. Fit Feature Pipeline strictly on Train
    feat_pipe = CompositeFeaturePipeline(n_pca_components=8)
    feat_pipe.fit(X_tr_raw)
    
    # Extract 20-D Hybrid features
    X_train = feat_pipe.transform(X_tr_raw)["hybrid"]
    X_test = feat_pipe.transform(X_te_raw)["hybrid"]
    
    print(f"=== BASELINE MODEL TRAINING (SHRAVANI - DAY 7) ===")
    print(f"Training instances: {X_train.shape[0]} | Test instances: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]} (8 PCA + 12 Domain Biomarkers)")
    
    models = {
        "Logistic Regression": BaselineLogisticRegression(C=1.0, random_state=42),
        "Random Forest": BaselineRandomForest(n_estimators=150, max_depth=8, random_state=42),
        "Support Vector Machine": BaselineSVM(C=2.0, gamma="scale", random_state=42)
    }
    
    results = []
    print("\n" + "=" * 75)
    print(f"{'Model':<24} | {'Test Acc':<10} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Balanced Acc':<12}")
    print("-" * 75)
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        f1_wt = f1_score(y_test, y_pred, average="weighted")
        bacc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"{name:<24} | {acc*100:6.2f}%    | {f1_mac:6.3f}     | {f1_wt:6.3f}       | {bacc*100:6.2f}%")
        
        # Save model
        clean_name = name.lower().replace(" ", "_")
        save_file = os.path.join(models_dir, f"{clean_name}_baseline.joblib")
        clf.save(save_file)
        
        results.append({
            "model_name": name,
            "test_accuracy": round(acc, 4),
            "macro_f1": round(f1_mac, 4),
            "weighted_f1": round(f1_wt, 4),
            "balanced_accuracy": round(bacc, 4),
            "model_path": save_file
        })
        
    print("=" * 75)
    res_df = pd.DataFrame(results).sort_values(by="macro_f1", ascending=False)
    csv_path = os.path.join(tables_dir, "baseline_model_comparison.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved baseline evaluation table to: {csv_path}")
    print(f"Saved trained models to: {models_dir}")
    
    return res_df

if __name__ == "__main__":
    train_and_evaluate_baselines()


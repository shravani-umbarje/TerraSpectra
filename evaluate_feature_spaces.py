"""
Shravani Day 6: Feature Space Evaluation & Candidate Selection
Author: Shravani (Machine Learning Specialist)
Description: Runs 5-fold cross-validated linear probing across candidate feature spaces
to mathematically determine the optimal input representation for ML models.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "src")))
from data.loader import HyperspectralDataLoader
from features.pipeline import CompositeFeaturePipeline

def evaluate_all_feature_spaces():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "raw", "tomato_wilt", "Tomato bacterial wilt dataset"))
    output_table_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "reports", "tables"))
    os.makedirs(output_table_dir, exist_ok=True)
    
    loader = HyperspectralDataLoader(data_dir)
    X_raw, y, df = loader.load_average_spectra()
    
    comp_pipe = CompositeFeaturePipeline(n_pca_components=8)
    feature_dict = comp_pipe.fit_transform(X_raw)
    feature_dict["raw_180"] = X_raw
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "balanced_accuracy"]
    
    results = []
    print("=== FEATURE SPACE PROBING (SHRAVANI - DAY 6) ===")
    print(f"{'Feature Space':<16} | {'Dimensions':<10} | {'CV Accuracy':<12} | {'Macro F1':<10} | {'Balanced Acc':<12}")
    print("-" * 72)
    
    for name in ["raw_180", "full_bands", "pca", "biomarkers", "hybrid"]:
        mat = feature_dict[name]
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ])
        
        scores = cross_validate(clf, mat, y, cv=cv, scoring=scoring, n_jobs=-1)
        acc_mean = float(np.mean(scores["test_accuracy"]))
        f1_mean = float(np.mean(scores["test_f1_macro"]))
        bacc_mean = float(np.mean(scores["test_balanced_accuracy"]))
        
        print(f"{name:<16} | {mat.shape[1]:<10} | {acc_mean*100:6.2f}%      | {f1_mean:6.3f}     | {bacc_mean*100:6.2f}%")
        results.append({
            "feature_space": name,
            "n_features": mat.shape[1],
            "cv_accuracy": round(acc_mean, 4),
            "cv_macro_f1": round(f1_mean, 4),
            "cv_balanced_acc": round(bacc_mean, 4)
        })
        
    res_df = pd.DataFrame(results).sort_values(by="cv_macro_f1", ascending=False)
    csv_path = os.path.join(output_table_dir, "feature_space_comparison.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved feature space benchmark to: {csv_path}")
    print(f"Top performing representation: '{res_df.iloc[0]['feature_space']}' with Macro F1 = {res_df.iloc[0]['cv_macro_f1']:.4f}")
    
    return res_df

if __name__ == "__main__":
    evaluate_all_feature_spaces()


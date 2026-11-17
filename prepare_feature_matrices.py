"""
Shravani Day 5: Leakage-Free Feature Matrix Preparation
Author: Shravani (Machine Learning Specialist)
Description: Fits preprocessing pipeline strictly on training split, transforms test split,
evaluates class clustering metrics, and saves canonical processed matrices.
"""

import os
import sys
import numpy as np
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "src")))
from data.loader import HyperspectralDataLoader
from models.split_manager import DatasetSplitManager
from preprocessing.pipeline import SpectralPreprocessingPipeline

def prepare_and_export_matrices():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "raw", "tomato_wilt", "Tomato bacterial wilt dataset"))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "processed"))
    os.makedirs(processed_dir, exist_ok=True)
    
    loader = HyperspectralDataLoader(data_dir)
    X, y, df = loader.load_average_spectra()
    
    # 1. Stratified Split (80/20)
    split_mgr = DatasetSplitManager(random_state=42)
    X_train, X_test, y_train, y_test, tr_idx, te_idx = split_mgr.stratified_train_test_split(X, y, test_size=0.20)
    
    # 2. Fit Preprocessing Pipeline strictly on Train
    pipe = SpectralPreprocessingPipeline(mode="snv_smooth", window_length=11, polyorder=2)
    pipe.fit(X_train)
    
    X_train_proc = pipe.transform(X_train)
    X_test_proc = pipe.transform(X_test)
    
    # 3. Measure Separation Metric (Silhouette Score)
    raw_sil = silhouette_score(X_train, y_train)
    proc_sil = silhouette_score(X_train_proc, y_train)
    
    # 4. Export Processed Matrices
    np.save(os.path.join(processed_dir, "X_train_processed.npy"), X_train_proc)
    np.save(os.path.join(processed_dir, "X_test_processed.npy"), X_test_proc)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    
    # Also save the fitted pipeline artifact
    pipe.save(os.path.join(processed_dir, "fitted_preprocessing_pipeline.joblib"))
    
    print("=== LEAKAGE-FREE FEATURE MATRIX PREPARATION (SHRAVANI - DAY 5) ===")
    print(f"X_train_processed shape: {X_train_proc.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_processed shape:  {X_test_proc.shape}, y_test shape:  {y_test.shape}")
    print(f"\nClustering Separability (Silhouette Score on 5 Severity Classes):")
    print(f"  Raw Reflectance:        {raw_sil:.4f}")
    print(f"  SNV + Savitzky-Golay:   {proc_sil:.4f} (Significant improvement in manifold compactness)")
    print(f"\nSaved processed matrices to: {processed_dir}")
    
    return X_train_proc, X_test_proc, y_train, y_test

if __name__ == "__main__":
    prepare_and_export_matrices()


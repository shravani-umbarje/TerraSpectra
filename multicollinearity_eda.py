"""
Shravani Day 4: Feature Space Multicollinearity & Univariate Discriminability
Author: Shravani (Machine Learning Specialist)
Description: Evaluates spectral feature collinearity, computes ANOVA F-values,
and identifies the top 15 most discriminative wavelengths.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "src")))
from data.loader import HyperspectralDataLoader, get_wavelength_grid

def analyze_multicollinearity_and_anova():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "raw", "tomato_wilt", "Tomato bacterial wilt dataset"))
    output_table_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "reports", "tables"))
    os.makedirs(output_table_dir, exist_ok=True)
    
    loader = HyperspectralDataLoader(data_dir)
    X, y, df = loader.load_average_spectra()
    wavelengths = get_wavelength_grid(180)
    
    # 1. Pearson Correlation Analysis
    corr_matrix = np.corrcoef(X, rowvar=False) # (180, 180)
    
    # Measure fraction of contiguous pairs with r > 0.95
    adjacent_corrs = [corr_matrix[i, i+1] for i in range(179)]
    avg_adj_corr = float(np.mean(adjacent_corrs))
    high_corr_pct = float(np.mean(np.array(adjacent_corrs) > 0.95) * 100)
    
    # 2. ANOVA F-classif (Univariate Feature Importance)
    f_scores, p_values = f_classif(X, y)
    
    records = []
    for i in range(180):
        records.append({
            "band_index": i,
            "wavelength_nm": round(float(wavelengths[i]), 2),
            "f_score": round(float(f_scores[i]), 3),
            "p_value": float(p_values[i]),
            "significant_at_001": bool(p_values[i] < 0.001)
        })
        
    ranking_df = pd.DataFrame(records).sort_values(by="f_score", ascending=False)
    csv_path = os.path.join(output_table_dir, "wavelength_anova_ranking.csv")
    ranking_df.to_csv(csv_path, index=False)
    
    print("=== MULTICOLLINEARITY & ANOVA FEATURE RANKING (SHRAVANI - DAY 4) ===")
    print(f"Matrix Dimension: {X.shape[0]} samples × {X.shape[1]} spectral bands")
    print(f"Average adjacent band Pearson correlation: {avg_adj_corr:.4f}")
    print(f"Percentage of adjacent bands with r > 0.95: {high_corr_pct:.1f}%")
    print(f"\nSaved full ANOVA ranking to: {csv_path}")
    
    print("\n--- Top 10 Most Discriminative Wavelengths (by ANOVA F-score) ---")
    print(ranking_df.head(10)[["band_index", "wavelength_nm", "f_score", "p_value"]].to_string(index=False))
    
    return ranking_df

if __name__ == "__main__":
    analyze_multicollinearity_and_anova()


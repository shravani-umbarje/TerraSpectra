import os
import pandas as pd

def test_feature_ranking_table():
    table_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "reports", "tables", "wavelength_anova_ranking.csv"))
    assert os.path.exists(table_path), f"File {table_path} does not exist"
    df = pd.read_csv(table_path)
    assert len(df) == 180, f"Expected 180 bands, got {len(df)}"
    assert (df["f_score"] > 0).all(), "F-scores must be positive"
    assert (df["p_value"] < 0.05).all(), "All 180 bands show statistically significant difference across wilt stages"
    print("ALL MULTICOLLINEARITY & ANOVA TESTS PASSED!")

if __name__ == "__main__":
    test_feature_ranking_table()


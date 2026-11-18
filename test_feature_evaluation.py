import os
import pandas as pd

def test_feature_space_comparison():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "reports", "tables", "feature_space_comparison.csv"))
    assert os.path.exists(csv_path), f"File {csv_path} does not exist"
    df = pd.read_csv(csv_path)
    assert len(df) == 5, f"Expected 5 feature spaces evaluated, got {len(df)}"
    # Verify hybrid is top rank
    top_row = df.iloc[0]
    assert top_row["feature_space"] == "hybrid"
    assert top_row["cv_macro_f1"] > 0.75
    print("ALL FEATURE SPACE EVALUATION TESTS PASSED!")

if __name__ == "__main__":
    test_feature_space_comparison()


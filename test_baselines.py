import os
import pandas as pd

def test_baseline_models():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "models", "trained"))
    tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "reports", "tables"))
    
    # Verify trained model artifacts
    for name in ["logistic_regression_baseline.joblib", "random_forest_baseline.joblib", "support_vector_machine_baseline.joblib"]:
        path = os.path.join(models_dir, name)
        assert os.path.exists(path), f"Trained model missing at {path}"
        assert os.path.getsize(path) > 1000
        
    # Verify evaluation table
    csv_path = os.path.join(tables_dir, "baseline_model_comparison.csv")
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    # Check that all models outperform majority baseline (40.96%)
    assert (df["test_accuracy"] > 0.60).all(), "A baseline model failed to exceed 60% accuracy"
    assert (df["macro_f1"] > 0.65).all(), "A baseline model failed to exceed 0.65 Macro F1"
    
    print("ALL BASELINE MODEL VALIDATION TESTS PASSED!")

if __name__ == "__main__":
    test_baseline_models()


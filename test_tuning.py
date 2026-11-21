import os
import json
import pandas as pd

def test_tuning_results():
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "models", "evaluation"))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "models", "trained"))
    
    # Check JSON log
    json_path = os.path.join(eval_dir, "tuning_results.json")
    assert os.path.exists(json_path), f"Missing {json_path}"
    with open(json_path, "r") as f:
        log = json.load(f)
    assert "random_forest" in log
    assert "support_vector_machine" in log
    assert log["random_forest"]["cv_best_macro_f1"] > 0.75
    assert log["support_vector_machine"]["cv_best_macro_f1"] > 0.75
    
    # Check trained models exist
    assert os.path.exists(os.path.join(models_dir, "random_forest_optimized.joblib"))
    assert os.path.exists(os.path.join(models_dir, "svm_optimized.joblib"))
    
    print("ALL HYPERPARAMETER OPTIMIZATION TESTS PASSED!")

if __name__ == "__main__":
    test_tuning_results()


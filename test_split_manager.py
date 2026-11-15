import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "src")))
from data.loader import HyperspectralDataLoader
from models.split_manager import DatasetSplitManager, LABEL_ENCODING

def test_split_manager():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "raw", "tomato_wilt", "Tomato bacterial wilt dataset"))
    loader = HyperspectralDataLoader(data_dir)
    X, y, df = loader.load_average_spectra()
    
    manager = DatasetSplitManager(random_state=42)
    X_train, X_test, y_train, y_test, tr_idx, te_idx = manager.stratified_train_test_split(X, y, test_size=0.20)
    
    print(f"Dataset total: {len(y)} samples")
    print(f"Training split: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing split:  X_test  {X_test.shape}, y_test  {y_test.shape}")
    
    # Assert dimensions
    assert len(X_train) == 234, f"Expected 234 train samples, got {len(X_train)}"
    assert len(X_test) == 59, f"Expected 59 test samples, got {len(X_test)}"
    
    # Assert zero data leakage
    no_leakage = manager.verify_no_leakage(tr_idx, te_idx)
    assert no_leakage is True, "CRITICAL ERROR: Data leakage detected between train and test sets!"
    print("Zero data leakage verified: intersection(train_idx, test_idx) is EMPTY.")
    
    # Check stratification summary
    summary_df = manager.get_split_summary(y_train, y_test)
    print("\n--- Split Distribution Summary ---")
    print(summary_df.to_string(index=False))
    
    # Assert each class has representation in both train and test
    for lvl in range(5):
        assert (y_train == lvl).sum() > 0, f"Class {lvl} missing from train set"
        assert (y_test == lvl).sum() > 0, f"Class {lvl} missing from test set"
        
    print("\nALL SPLIT MANAGER UNIT TESTS PASSED!")

if __name__ == "__main__":
    test_split_manager()


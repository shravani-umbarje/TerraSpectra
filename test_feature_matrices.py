import os
import numpy as np

def test_feature_matrix_integrity():
    proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "processed"))
    
    x_tr = np.load(os.path.join(proc_dir, "X_train_processed.npy"))
    x_te = np.load(os.path.join(proc_dir, "X_test_processed.npy"))
    y_tr = np.load(os.path.join(proc_dir, "y_train.npy"))
    y_te = np.load(os.path.join(proc_dir, "y_test.npy"))
    
    assert x_tr.shape == (234, 180), f"Unexpected train shape: {x_tr.shape}"
    assert x_te.shape == (59, 180), f"Unexpected test shape: {x_te.shape}"
    assert len(y_tr) == 234
    assert len(y_te) == 59
    assert np.isnan(x_tr).sum() == 0, "NaN in training matrix"
    assert np.isnan(x_te).sum() == 0, "NaN in testing matrix"
    assert np.isinf(x_tr).sum() == 0, "Inf in training matrix"
    assert np.isinf(x_te).sum() == 0, "Inf in testing matrix"
    
    print(f"Verified processed matrices: Train {x_tr.shape}, Test {x_te.shape}, 0 NaNs, 0 Infs.")
    print("ALL FEATURE MATRIX PREPARATION TESTS PASSED!")

if __name__ == "__main__":
    test_feature_matrix_integrity()


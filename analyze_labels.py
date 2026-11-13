import os
import glob
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold

def analyze_class_distributions(data_dir):
    """
    Analyzes class labels, frequencies, and imbalance metrics for
    Tomato Bacterial Wilt severity identification.
    Evaluates baseline classifiers and 5-fold cross-validation distribution.
    """
    levels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    sample_names = []
    sample_labels = []
    
    all_dirs = sorted(os.listdir(data_dir))
    for d in all_dirs:
        dir_path = os.path.join(data_dir, d)
        if not os.path.isdir(dir_path):
            continue
        for lvl in range(5):
            if d.startswith(f"Level{lvl}_"):
                levels[lvl] += 1
                sample_names.append(d)
                sample_labels.append(lvl)
                break
                
    total = len(sample_labels)
    counts = np.array([levels[i] for i in range(5)])
    proportions = counts / total
    
    # Imbalance metrics
    max_count = np.max(counts)
    min_count = np.min(counts)
    imbalance_ratio = max_count / min_count # 120 / 33
    shannon_entropy = stats.entropy(proportions, base=2)
    max_entropy = np.log2(5) # Equal distribution
    balance_ratio = shannon_entropy / max_entropy # 1.0 = perfectly balanced
    
    # Baselines
    majority_baseline = max_count / total # Always predict Level 0
    random_baseline = 1.0 / 5.0 # Random guessing (20%)
    weighted_random_baseline = np.sum(proportions ** 2) # Guessing with prior probability
    
    # Class weights for loss functions (balanced)
    # weight = total / (n_classes * count)
    class_weights = total / (5.0 * counts)
    
    print("=== CLASS LABEL & IMBALANCE ANALYSIS (SHRAVANI - DAY 2) ===")
    print(f"Total plant instances: {total}")
    print("\nClass breakdown:")
    labels_desc = [
        "Level 0 (Healthy)",
        "Level 1 (Early/Asymptomatic)",
        "Level 2 (Mild Wilt: 1-33%)",
        "Level 3 (Moderate Wilt: 34-66%)",
        "Level 4 (Severe Wilt: 67-100%)"
    ]
    for i in range(5):
        print(f"  {labels_desc[i]:<32}: {counts[i]:3d} samples ({proportions[i]*100:5.2f}%) | Class Weight: {class_weights[i]:.3f}")
        
    print(f"\nImbalance Metrics:")
    print(f"  Majority-to-Minority Ratio: {imbalance_ratio:.2f} : 1")
    print(f"  Shannon Entropy:            {shannon_entropy:.4f} bits (Max: {max_entropy:.4f})")
    print(f"  Normalized Balance Score:   {balance_ratio:.4f} (1.0 = perfectly balanced)")
    
    print(f"\nTrivial Baselines:")
    print(f"  Random Guessing Accuracy:   {random_baseline*100:.2f}%")
    print(f"  Prior-Weighted Random:      {weighted_random_baseline*100:.2f}%")
    print(f"  Majority Class Baseline:    {majority_baseline*100:.2f}% (Predicting all Level 0)")
    
    print(f"\nStratified 5-Fold Validation Simulation:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = np.array(sample_labels)
    X_dummy = np.zeros((total, 1))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dummy, y), 1):
        val_counts = np.bincount(y[val_idx], minlength=5)
        print(f"  Fold {fold}: Train={len(train_idx)} samples, Val={len(val_idx)} samples | Val dist: {val_counts}")
        
    return {
        "counts": counts.tolist(),
        "imbalance_ratio": float(imbalance_ratio),
        "majority_baseline": float(majority_baseline),
        "class_weights": class_weights.tolist()
    }

if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "main project", "data", "raw", "tomato_wilt", "Tomato bacterial wilt dataset"))
    analyze_class_distributions(dataset_path)


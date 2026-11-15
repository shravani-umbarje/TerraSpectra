# Test Results — Shravani (Day 3)

## Test Execution Details
- **Test File**: `test_split_manager.py`
- **Total Input**: 293 samples across 5 severity classes
- **Result**: ALL ASSERTIONS PASSED

## Execution Output
```text
Dataset total: 293 samples
Training split: X_train (234, 180), y_train (234,)
Testing split:  X_test  (59, 180), y_test  (59,)
Zero data leakage verified: intersection(train_idx, test_idx) is EMPTY.

--- Split Distribution Summary ---
 Level             Class Label  Train Samples  Test Samples  Total  Train %  Test %
     0                 Healthy             96            24    120     80.0    20.0
     1 Infected (Asymptomatic)             54            14     68     79.4    20.6
     2       Mild Wilt (1-33%)             31             8     39     79.5    20.5
     3  Moderate Wilt (34-66%)             26             7     33     78.8    21.2
     4   Severe Wilt (67-100%)             27             6     33     81.8    18.2

ALL SPLIT MANAGER UNIT TESTS PASSED!
```

## Validation Findings
- 234 training specimens, 59 test specimens.
- Exact $80/20$ ratio maintained for each individual severity tier ($\pm 1.8\%$).
- Leakage index check passed: no test sample index appears in training set.


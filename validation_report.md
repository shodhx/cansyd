# CNSD Pipeline Validation Report (Issue #8)

This document serves as a professional record of the experiments, fixes, and validation runs performed to harden the CWRU baseline for the CNSD causal-neurosymbolic system.

## 1. Loader Bug Fix (Nested Subdirectories)
The `load_cwru()` function was missing the `@3`, `@6`, and `@12` outer-race classes due to a rigid directory parsing logic. This was fixed by utilizing `os.walk` to comprehensively scan all nested subdirectories for `.mat` files. 

**Result**: All 10 CWRU taxonomy classes are now correctly ingested.
```text
[train_data] Dataset 'CWRU_Train': 5806 samples, window=1024, fs=12000Hz, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], conditions=[0, 1, 2], physics=yes
[test_data] Dataset 'CWRU_Test': 2019 samples, window=1024, fs=12000Hz, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], conditions=[3], physics=yes
```

## 2. Root-Cause Printing Fix
The root-cause print loop was previously dumping raw physics vectors and showing "No defect detected". The logic was updated to specifically filter for `CONFIRMED` faults and uniquely print the human-readable physical descriptions of each fault class exactly once.

**Result**: Clean, auditable, and human-readable root-cause statements.
```text
[examples] auditable root-cause diagnoses (CONFIRMED faults only):
    [Class 2] [HIGH_CONFIDENCE] Defect on the rolling element (Ball), evidenced by a peak at the BSF (135.9 Hz, strength 3.5). a defect on a rolling element striking both races as it spins.
    [Class 4] [HIGH_CONFIDENCE] Defect on the inner race (Inner Race), evidenced by a peak at the BPFI (156.1 Hz, strength 4.4). a defect on the rotating inner race, modulated by shaft rotation.
    [Class 5] [HIGH_CONFIDENCE] Defect on the inner race (Inner Race), evidenced by a peak at the BPFI (156.1 Hz, strength 3.0). a defect on the rotating inner race, modulated by shaft rotation.
    [Class 6] [HIGH_CONFIDENCE] Defect on the inner race (Inner Race), evidenced by a peak at the BPFI (156.1 Hz, strength 3.3). a defect on the rotating inner race, modulated by shaft rotation.
    [Class 7] [HIGH_CONFIDENCE] Defect on the outer race (Outer Race), evidenced by a peak at the BPFO (103.4 Hz, strength 4.6). a defect on the stationary outer race struck by each rolling element.
```

## 3. Terminology Clarification
The script `cross_domain_validation.py` was renamed to `cross_condition_robustness.py` to correctly reflect the protocol (AWGN noise injection). "Cross-domain" is reserved for multi-rig experiments (e.g., SEU/Paderborn).

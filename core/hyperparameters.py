"""
hyperparameters.py - Explicit, honest accounting of every hyperparameter.

The Bible's objection #10: several hyperparameters are hand-set but were
presented under a 'calibrated' label. This module makes the distinction explicit
so reviewers (and the paper) can see exactly which values are DATA-DERIVED (from
the held-out calibration split) and which are DESIGN CHOICES (hand-set).
"""

# Derived from the held-out calibration split (data-driven).
CALIBRATED = {
    'cnn_confidence_threshold': {
        'value': 0.90,
        'derivation': '10th percentile of CNN confidence on correct calibration predictions',
    },
}

# Hand-set design choices (not derived from data). Stated as such.
DESIGN = {
    'prominence_threshold': {
        'value': 3.0,
        'rationale': 'envelope-spectrum peak prominence above local baseline to '
                     'count a characteristic frequency as present',
    },
    'lora_rank': {'value': 4, 'rationale': 'low-rank adapter dimension'},
    'envelope_band_rel_tol': {'value': 0.04, 'rationale': 'relative width of the '
                              'frequency band searched around each characteristic freq'},
    'n_harmonics_checked': {'value': 2, 'rationale': 'harmonics of each '
                            'characteristic frequency included in the evidence'},
    'window': {'value': 1024, 'rationale': 'samples per classifier input'},
    'train_stride': {'value': 256, 'rationale': 'overlap within training split only'},
    'test_stride': {'value': 1024, 'rationale': 'non-overlapping test windows (leakage-free)'},
}


def report():
    """Print the calibrated-vs-design table (for logs / paper appendix)."""
    print('=' * 64)
    print('HYPERPARAMETERS: calibrated (data-derived) vs design (hand-set)')
    print('=' * 64)
    print('\n[Calibrated from held-out split]')
    for k, v in CALIBRATED.items():
        print(f'  {k} = {v["value"]}')
        print(f'      <- {v["derivation"]}')
    print('\n[Design choices, hand-set]')
    for k, v in DESIGN.items():
        print(f'  {k} = {v["value"]}  ({v["rationale"]})')
    print('=' * 64)

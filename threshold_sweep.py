"""
threshold_sweep.py - sweep the envelope-prominence threshold on the CWRU dataset.
"""
import numpy as np
from cnsd import Dataset
from cnsd.diagnosis.system import CNSD

# Reuse CWRU loader and configs from validate_run
from validate_run import load_cwru, CWRU, TAXONOMY, headline_accuracy_by_verdict

def main():
    print('=' * 50)
    print('CWRU THRESHOLD SWEEP')
    print('=' * 50)

    X, y, cond = load_cwru()
    X = np.asarray(X, np.float32); y = np.asarray(y); cond = np.asarray(cond)
    
    test_mask = cond == 3
    train_mask = cond < 3
    
    train_data = Dataset.from_arrays(X[train_mask], y[train_mask], cond[train_mask], fs=12000, physics=CWRU, taxonomy=TAXONOMY, name='CWRU_Train')
    test_data = Dataset.from_arrays(X[test_mask], y[test_mask], cond[test_mask], fs=12000, physics=CWRU, taxonomy=TAXONOMY, name='CWRU_Test')
    
    print('Training baseline CNSD model (this happens once)...')
    model = CNSD()
    model.fit(train_data, epochs=30)
    
    print('\nStarting Threshold Sweep on Calibration/Train Data:')
    thresholds = [3.0, 2.5, 2.0, 1.5]
    
    for t in thresholds:
        print('-' * 60)
        print(f'Testing Threshold: {t}')
        model.symbolic.tau = t  # Hot-swap the threshold!
        report = model.diagnose(train_data)
        vr = report.verification_rate()
        print(f"  CONFIRMED   : {vr.get('CONFIRMED', 0.0):.1%}")
        print(f"  CONFLICT    : {vr.get('CONFLICT', 0.0):.1%}")
        print(f"  INCONCLUSIVE: {vr.get('INCONCLUSIVE', 0.0):.1%}")
        
        hb = headline_accuracy_by_verdict(report, train_data.y)
        if 'CONFIRMED' in hb and 'CONFLICT' in hb:
            gap = hb['CONFIRMED']['cnn_accuracy'] - hb['CONFLICT']['cnn_accuracy']
            print(f"  ACCURACY GAP: +{gap:.3f} (CONFIRMED vs CONFLICT)")
        else:
            print("  ACCURACY GAP: N/A (Missing CONFIRMED or CONFLICT samples)")

if __name__ == '__main__':
    main()

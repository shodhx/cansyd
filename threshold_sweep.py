"""
threshold_sweep.py - sweep the envelope-prominence threshold on the CWRU dataset.
"""
import numpy as np
from cnsd import Dataset
from cnsd.diagnosis.system import CNSD

# Reuse CWRU loader and configs from validate_run
from validate_run import load_cwru, CWRU, TAXONOMY

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
    
    print('\nStarting Threshold Sweep on Test Data:')
    thresholds = [3.0, 2.5, 2.0, 1.5]
    
    for t in thresholds:
        print('-' * 40)
        print(f'Testing Threshold: {t}')
        model.symbolic.tau = t  # Hot-swap the threshold!
        report = model.diagnose(test_data)
        vr = report.verification_rate()
        print(f"  CONFIRMED   : {vr.get('CONFIRMED', 0.0):.1%}")
        print(f"  CONFLICT    : {vr.get('CONFLICT', 0.0):.1%}")
        print(f"  INCONCLUSIVE: {vr.get('INCONCLUSIVE', 0.0):.1%}")

if __name__ == '__main__':
    main()

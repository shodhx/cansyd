import numpy as np
import tensorflow as tf

from cnsd import Dataset
from cnsd.diagnosis.system import CNSD
from validate_run import CWRU, TAXONOMY, headline_accuracy_by_verdict, load_cwru


def main():
    # Fix the random seeds to prevent INCONCLUSIVE rate drift between runs
    np.random.seed(42)
    tf.random.set_seed(42)

    print('=' * 68)
    print('CNSD THRESHOLD SWEEP (Held-out Calibration Protocol)')
    print('=' * 68)

    # 1. Load data
    X, y, cond = load_cwru()
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    cond = np.asarray(cond)

    # 2. Strict 3-way split to prevent test-set leakage
    # Train: Motor loads 0 and 1
    # Calibration (Sweep): Motor load 2 (CNN never sees this)
    # Test: Motor load 3
    train_mask = cond < 2
    calib_mask = cond == 2
    test_mask = cond == 3

    train_data = Dataset.from_arrays(
        X[train_mask],
        y[train_mask],
        cond[train_mask],
        fs=12000,
        physics=CWRU,
        taxonomy=TAXONOMY,
        name='CWRU_Train_Loads_0_1',
    )
    calib_data = Dataset.from_arrays(
        X[calib_mask],
        y[calib_mask],
        cond[calib_mask],
        fs=12000,
        physics=CWRU,
        taxonomy=TAXONOMY,
        name='CWRU_Calib_Load_2',
    )
    test_data = Dataset.from_arrays(
        X[test_mask],
        y[test_mask],
        cond[test_mask],
        fs=12000,
        physics=CWRU,
        taxonomy=TAXONOMY,
        name='CWRU_Test_Load_3',
    )

    print(f'\n[train_data] {train_data.summary()}')
    print(f'[calib_data] {calib_data.summary()}')
    print(f'[test_data]  {test_data.summary()}')

    # 3. Train the model exclusively on Loads 0 and 1
    print('\nTraining CNN on Motor Loads 0 and 1...')
    model = CNSD()
    model.fit(train_data, epochs=30)

    # 4. Perform the tau sweep on the unseen Calibration Set (Load 2)
    print('\n' + '-' * 68)
    print('SWEEPING TAU ON CALIBRATION SET (LOAD 2)')
    print('-' * 68)

    taus = np.arange(1.0, 4.1, 0.5)
    best_tau = None
    best_gap = -np.inf

    for tau in taus:
        # Hot-swap the threshold without retraining
        model.symbolic.tau = float(tau)
        report = model.diagnose(calib_data)

        hb = headline_accuracy_by_verdict(report, calib_data.y)

        conf = hb.get('CONFIRMED', {'cnn_accuracy': 0.0, 'n': 0})
        cnfl = hb.get('CONFLICT', {'cnn_accuracy': 0.0, 'n': 0})
        inc = hb.get('INCONCLUSIVE', {'cnn_accuracy': 0.0, 'n': 0})

        gap = conf['cnn_accuracy'] - cnfl['cnn_accuracy']

        # We need a meaningful yield rate so we aren't picking a threshold
        # that marks 99% of samples as INCONCLUSIVE.
        total_samples = len(calib_data.X)
        yield_rate = (conf['n'] + cnfl['n']) / total_samples

        print(
            f'tau={tau:4.1f} | gap={gap:+.3f} | yield={yield_rate:5.1%} | '
            f'CONF={conf["cnn_accuracy"]:.3f}(n={conf["n"]:4d})  '
            f'CNFL={cnfl["cnn_accuracy"]:.3f}(n={cnfl["n"]:4d})  '
            f'INC={inc["cnn_accuracy"]:.3f}(n={inc["n"]:4d})'
        )

        if gap > best_gap and yield_rate > 0.05:
            best_gap = gap
            best_tau = float(tau)

    print(f'\n=> Optimal threshold found on calibration data: tau = {best_tau}')
    print('=> CNN is frozen. Threshold is frozen.')

    # 5. Final, rigorous test on the Test Set (Load 3)
    print('\n' + '=' * 68)
    print(f'FINAL RIGOROUS TEST ON LOAD 3 (Frozen tau={best_tau})')
    print('=' * 68)

    model.symbolic.tau = best_tau
    test_report = model.diagnose(test_data)
    test_hb = headline_accuracy_by_verdict(test_report, test_data.y)

    for v, d in test_hb.items():
        print(f'    {v:13}: acc={d["cnn_accuracy"]:.3f}  (n={d["n"]})')

    if 'CONFIRMED' in test_hb and 'CONFLICT' in test_hb:
        final_gap = test_hb['CONFIRMED']['cnn_accuracy'] - test_hb['CONFLICT']['cnn_accuracy']
        print(f'    -> FINAL GAP: {final_gap:+.3f}')

    print('\n[Layer 2] Physics verification rate on Test Set:')
    vr = test_report.verification_rate()
    for k, v in vr.items():
        print(f'    {k:13}: {v:.1%}')


if __name__ == '__main__':
    main()

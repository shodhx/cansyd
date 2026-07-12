import numpy as np
import tensorflow as tf

from cansyd import Dataset
from cansyd.datasets.xjtusy import load_xjtusy_domain_split
from cansyd.diagnosis.system import CANSYD


def headline_accuracy_by_verdict(report, y_true):
    pred = np.array([r['predicted_class'] for r in report.records])
    correct = pred == np.asarray(y_true)
    verdicts = np.array([r['physics_verdict'] for r in report.records])
    out = {}
    for v in ('CONFIRMED', 'CONFLICT', 'INCONCLUSIVE'):
        m = verdicts == v
        if m.any():
            out[v] = {'n': int(m.sum()), 'cnn_accuracy': float(correct[m].mean())}
    return out


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)

    print('Loading XJTU-SY dataset (Cross-Domain RPM/Load Split)...')
    train_data, target_data = load_xjtusy_domain_split(window_size=4096)

    # Split target domain into Calib (50%) and Test (50%)
    indices = np.arange(len(target_data.y))
    np.random.shuffle(indices)
    calib_size = len(indices) // 2

    calib_idx = indices[:calib_size]
    test_idx = indices[calib_size:]

    X_calib, y_calib, cond_calib = (
        target_data.X[calib_idx],
        target_data.y[calib_idx],
        target_data.cond[calib_idx],
    )
    X_test, y_test, cond_test = (
        target_data.X[test_idx],
        target_data.y[test_idx],
        target_data.cond[test_idx],
    )

    calib_data = Dataset.from_arrays(
        X_calib,
        y_calib,
        cond_calib,
        fs=target_data.fs,
        physics=target_data.physics,
        taxonomy=target_data.taxonomy,
        name='XJTUSY_Calib',
    )
    test_data = Dataset.from_arrays(
        X_test,
        y_test,
        cond_test,
        fs=target_data.fs,
        physics=target_data.physics,
        taxonomy=target_data.taxonomy,
        name='XJTUSY_Test',
    )

    print(
        f'Train (2100 RPM)={len(train_data.y)} | Calib (2250 RPM)={len(y_calib)} | Test (2250 RPM)={len(y_test)}'
    )

    model = CANSYD()

    print('\n[1] Training Neural Network on 2100 RPM Source Data...')
    model.fit(train_data, epochs=20)

    print('\n[2] Calibrating Tau threshold on 2250 RPM Target Data...')
    taus = np.arange(1.0, 5.1, 0.5)
    best_gap = -np.inf
    best_tau = 1.0

    for tau in taus:
        model.symbolic.tau = float(tau)
        report = model.diagnose(calib_data)

        hb = headline_accuracy_by_verdict(report, y_calib)

        conf_acc = hb.get('CONFIRMED', {}).get('cnn_accuracy', 0.0)
        cnfl_acc = hb.get('CONFLICT', {}).get('cnn_accuracy', 0.0)
        gap = conf_acc - cnfl_acc if 'CONFIRMED' in hb and 'CONFLICT' in hb else 0.0

        print(f'Calib tau={tau:.1f} | Conf={conf_acc:.3f} | Cnfl={cnfl_acc:.3f} | Gap={gap:+.3f}')
        if gap > best_gap:
            best_gap = gap
            best_tau = float(tau)

    print(f'\n=> Selected optimal tau: {best_tau}')

    print('\n[3] Evaluating on Test Set (2250 RPM)...')
    model.symbolic.tau = best_tau
    report = model.diagnose(test_data)
    pred = np.array([r['predicted_class'] for r in report.records])
    baseline_acc = float((pred == np.asarray(y_test)).mean())

    print('\n--- FINAL TEST RESULTS (CROSS-DOMAIN XJTU-SY) ---')
    print(f'Baseline CNN Acc:       {baseline_acc:.3f}')
    print('--------------------------------------------')

    hb = headline_accuracy_by_verdict(report, y_test)
    if 'CONFIRMED' in hb:
        print(
            f'  Physics-Confirmed Acc:   {hb["CONFIRMED"]["cnn_accuracy"]:.3f} (n={hb["CONFIRMED"]["n"]})'
        )
    if 'CONFLICT' in hb:
        print(
            f'  Physics-Conflict Acc:    {hb["CONFLICT"]["cnn_accuracy"]:.3f} (n={hb["CONFLICT"]["n"]})'
        )
    if 'INCONCLUSIVE' in hb:
        inc_n = hb['INCONCLUSIVE']['n']
        inc_pct = (inc_n / len(y_test)) * 100
        print(
            f'  Physics-Inconclusive Acc:{hb["INCONCLUSIVE"]["cnn_accuracy"]:.3f} (n={inc_n}, {inc_pct:.1f}%)'
        )

    if 'CONFIRMED' in hb and 'CONFLICT' in hb:
        gap = hb['CONFIRMED']['cnn_accuracy'] - hb['CONFLICT']['cnn_accuracy']
        print(f'  GAP (CONF - CNFL):       {gap:+.3f}')
    print('--------------------------------------------')

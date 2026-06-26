import os
import glob
import numpy as np
import scipy.io as sio

from cnsd import Dataset
from cnsd.diagnosis.system import CNSD
from cnsd.physics import PhysicsConfig


def load_pu(data_dir=r'E:\301\PU-dataset', window_size=2048):
    """
    Loads authentic Paderborn University dataset .mat files.
    K00x = Healthy (0)
    KAxx = Outer Race Fault (1)
    KIxx = Inner Race Fault (2)
    """
    X = []
    y = []
    cond = []

    categories = [('K0*', 0), ('KA*', 1), ('KI*', 2)]

    for prefix, label in categories:
        pattern = os.path.join(data_dir, prefix, '*.mat')
        files = glob.glob(pattern)
        for fpath in files:
            fname = os.path.basename(fpath)
            key = fname.replace('.mat', '')

            # N15 -> 1500 RPM, N09 -> 900 RPM
            rpm_code = fname.split('_')[0]
            rpm = float(rpm_code[1:]) * 100.0 if rpm_code.startswith('N') else 1500.0

            try:
                mat = sio.loadmat(fpath)
                if key not in mat:
                    continue

                y_struct = mat[key]['Y'][0, 0]

                # Find the vibration_1 channel
                vib_idx = -1
                for i in range(y_struct['Name'].shape[1]):
                    if 'vibration' in str(y_struct['Name'][0, i][0]).lower():
                        vib_idx = i
                        break

                if vib_idx == -1:
                    continue

                sig = y_struct['Data'][0, vib_idx].flatten()

                # Segment signal
                for i in range(0, len(sig) - window_size, window_size):
                    segment = sig[i : i + window_size]
                    segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                    X.append(segment)
                    y.append(label)
                    cond.append(rpm)

            except Exception as e:
                print(f'Error loading {fname}: {e}')

    return np.array(X, dtype=np.float32), np.array(y), np.array(cond)


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
    print('Loading Authentic PU dataset (Real Fatigue Damages)...')
    X, y, cond = load_pu()

    print(f'Total samples extracted: {len(X)}')

    # Stratified split: 60% Train, 20% Calib, 20% Test
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    train_size = int(len(y) * 0.6)
    calib_size = int(len(y) * 0.2)

    train_idx = indices[:train_size]
    calib_idx = indices[train_size : train_size + calib_size]
    test_idx = indices[train_size + calib_size :]

    X_train, y_train, cond_train = X[train_idx], y[train_idx], cond[train_idx]
    X_calib, y_calib, cond_calib = X[calib_idx], y[calib_idx], cond[calib_idx]
    X_test, y_test, cond_test = X[test_idx], y[test_idx], cond[test_idx]

    print(f'Data split: Train={len(y_train)} | Calib={len(y_calib)} | Test={len(y_test)}')

    # Map all seen conditions directly to RPM
    unique_rpm = np.unique(cond)
    rpm_map = {float(r): float(r) for r in unique_rpm}

    # Paderborn 6203 Bearing Physics Config
    pu_physics = PhysicsConfig(
        bearing={'n_balls': 8, 'd_ball': 7.92, 'd_pitch': 28.55, 'contact_angle': 0.0},
        cond_to_rpm=rpm_map,
        fs=64000,
        name='PU-6203',
    )

    train_data = Dataset.from_arrays(
        X_train, y_train, cond_train, fs=64000, physics=pu_physics, name='PU_Train'
    )
    calib_data = Dataset.from_arrays(
        X_calib, y_calib, cond_calib, fs=64000, physics=pu_physics, name='PU_Calib'
    )
    test_data = Dataset.from_arrays(
        X_test, y_test, cond_test, fs=64000, physics=pu_physics, name='PU_Test'
    )

    model = CNSD()

    print('\n[1] Training Neural Network on Authentic PU Data...')
    model.fit(train_data, epochs=20)  # slightly fewer epochs to speed up for large dataset

    print('\n[2] Calibrating Tau threshold...')
    taus = np.arange(1.0, 4.1, 0.5)
    best_gap = -np.inf
    best_tau = 1.0

    for tau in taus:
        model.symbolic.tau = float(tau)
        report = model.diagnose(calib_data)

        hb = headline_accuracy_by_verdict(report, y_calib)
        if 'CONFIRMED' in hb and 'CONFLICT' in hb:
            gap = hb['CONFIRMED']['cnn_accuracy'] - hb['CONFLICT']['cnn_accuracy']
        else:
            gap = 0.0

        conf_acc = hb.get('CONFIRMED', {}).get('cnn_accuracy', 0.0)
        print(f'Calib tau={tau:.1f} | Conf={conf_acc:.3f} | Gap={gap:+.3f}')
        if gap > best_gap:
            best_gap = gap
            best_tau = float(tau)

    print(f'\n=> Selected optimal tau: {best_tau}')
    model.symbolic.tau = best_tau

    print('\n[3] Evaluating on Test Set...')
    report = model.diagnose(test_data)

    hb = headline_accuracy_by_verdict(report, y_test)
    print('\n--- FINAL TEST RESULTS (PU DATASET) ---')
    if 'CONFIRMED' in hb:
        print(
            f'Physics-Confirmed Acc:  {hb["CONFIRMED"]["cnn_accuracy"]:.3f} (n={hb["CONFIRMED"]["n"]})'
        )
    if 'CONFLICT' in hb:
        print(
            f'Physics-Conflict Acc:   {hb["CONFLICT"]["cnn_accuracy"]:.3f} (n={hb["CONFLICT"]["n"]})'
        )
    if 'INCONCLUSIVE' in hb:
        print(
            f'Physics-Inconclusive Acc:{hb["INCONCLUSIVE"]["cnn_accuracy"]:.3f} (n={hb["INCONCLUSIVE"]["n"]})'
        )

    if 'CONFIRMED' in hb and 'CONFLICT' in hb:
        gap = hb['CONFIRMED']['cnn_accuracy'] - hb['CONFLICT']['cnn_accuracy']
        print(f'GAP:                    {gap:+.3f}')
    print('---------------------------------------')

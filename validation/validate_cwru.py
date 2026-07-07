"""
validate_cwru.py - one clean end-to-end validation of the CNSD pipeline on CWRU.

Purpose: prove the rebuilt system actually RUNS on real data and produces the
numbers the paper claims - before writing the paper. This is a validation run,
not the full benchmark (no multi-seed, no baselines, no cross-rig). It confirms:

  1. the five-layer pipeline executes on real CWRU without error
  2. Layer 2 physics verification produces a real CONFIRMED/CONFLICT/INCONCLUSIVE rate
  3. the HEADLINE result computes: CNN accuracy split by physics verdict
  4. Layer 3 causal do(Z) runs and returns a Rung-2 effect
  5. the real Rung-3 counterfactual (DoWhy gcm) actually executes (not the fallback)
  6. example auditable root-cause diagnoses print

Run on Kaggle (GPU) after: pip install dowhy
Place CWRU data so your loader can read it, then adapt the load_cwru() stub below
to return X (n,1024), y (n,), cond (n,) - the only dataset-specific code here.
"""

import numpy as np
import tensorflow as tf

from cnsd import Dataset
from cnsd.causal import compute_vibration_rms, signal_kurtosis
from cnsd.counterfactual import build_scm, counterfactual_for_unit, dowhy_gcm_available
from cnsd.diagnosis.system import CNSD
from cnsd.physics import PhysicsConfig


# ── the ONLY dataset-specific code: return raw arrays from your CWRU files ────
def load_cwru():
    import os

    from scipy.io import loadmat

    base_dir = os.environ.get('CNSD_DATA_CWRU', r'E:\301\CWRU-dataset')
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f'CWRU dataset not found at {base_dir}')

    X, y, cond = [], [], []

    def read_mat(path, label, load):
        if not os.path.exists(path):
            return
        try:
            mat = loadmat(path)
            key = [k for k in mat.keys() if 'DE_time' in k]
            if not key:
                return
            time_series = mat[key[0]][:, 0]

            length = 1024
            idx_last = -(time_series.shape[0] % length)
            if idx_last == 0:
                clips = time_series.reshape(-1, length)
            else:
                clips = time_series[:idx_last].reshape(-1, length)

            for clip in clips:
                X.append(clip)
                y.append(label)
                cond.append(load)
        except Exception as e:
            print(f'Error reading {path}: {e}')

    normal_dir = os.path.join(base_dir, 'Normal')
    for f in os.listdir(normal_dir):
        if f.endswith('.mat'):
            load = int(f.split('_')[-1].split('.')[0])
            read_mat(os.path.join(normal_dir, f), 0, load)

    fault_dir = os.path.join(base_dir, '12k_Drive_End_Bearing_Fault_Data')
    fault_map = {
        'B': {'007': 1, '014': 2, '021': 3},
        'IR': {'007': 4, '014': 5, '021': 6},
        'OR': {'007': 7, '014': 8, '021': 9},
    }
    for ftype, size_map in fault_map.items():
        for size, label in size_map.items():
            dir_path = os.path.join(fault_dir, ftype, size)
            if not os.path.exists(dir_path):
                continue
            for root, _dirs, files in os.walk(dir_path):
                for f in files:
                    if f.endswith('.mat'):
                        load = int(f.split('_')[-1].split('.')[0])
                        read_mat(os.path.join(root, f), label, load)
    return np.array(X, dtype=np.float32), np.array(y), np.array(cond)


# CWRU 6205 physics + taxonomy (this is config, not hardcoded engine logic)
CWRU = PhysicsConfig(
    bearing={'n_balls': 9, 'd_ball': 0.3126, 'd_pitch': 1.537, 'contact_angle': 0.0},
    cond_to_rpm={0: 1797, 1: 1772, 2: 1750, 3: 1730},
    fs=12000,
    name='CWRU-6205',
)

TAXONOMY = {
    0: ('Normal', 'None'),
    1: ('Ball', 'Low'),
    2: ('Ball', 'Medium'),
    3: ('Ball', 'High'),
    4: ('Inner Race', 'Low'),
    5: ('Inner Race', 'Medium'),
    6: ('Inner Race', 'High'),
    7: ('Outer Race', 'Low'),
    8: ('Outer Race', 'Medium'),
    9: ('Outer Race', 'High'),
}


def headline_accuracy_by_verdict(report, y_true):
    """The core claim: is the CNN more accurate when physics CONFIRMS than CONFLICTS?

    Computed here (not in the report) because it needs the true labels.
    """
    pred = np.array([r['predicted_class'] for r in report.records])
    correct = pred == np.asarray(y_true)
    verdicts = np.array([r['physics_verdict'] for r in report.records])
    out = {}
    for v in ('CONFIRMED', 'CONFLICT', 'INCONCLUSIVE'):
        m = verdicts == v
        if m.any():
            out[v] = {'n': int(m.sum()), 'cnn_accuracy': float(correct[m].mean())}
    return out


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    print('=' * 68)
    print('CNSD VALIDATION RUN (CWRU, Protocol B)')
    print('=' * 68)

    # 1. data
    X, y, cond = load_cwru()
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    cond = np.asarray(cond)

    train_mask = cond < 3
    test_mask = cond == 3

    train_data = Dataset.from_arrays(
        X[train_mask],
        y[train_mask],
        cond[train_mask],
        fs=12000,
        physics=CWRU,
        taxonomy=TAXONOMY,
        name='CWRU_Train',
    )
    test_data = Dataset.from_arrays(
        X[test_mask],
        y[test_mask],
        cond[test_mask],
        fs=12000,
        physics=CWRU,
        taxonomy=TAXONOMY,
        name='CWRU_Test',
    )
    full_data = Dataset.from_arrays(
        X, y, cond, fs=12000, physics=CWRU, taxonomy=TAXONOMY, name='CWRU_Full'
    )

    print(f'[train_data] {train_data.summary()}')
    print(f'[test_data] {test_data.summary()}')

    # 2. fit + 3. diagnose (Layers 1,2,4 live)
    model = CNSD()
    model.fit(train_data, epochs=30)
    report = model.diagnose(test_data)
    print(f'\n[pipeline] {report.summary()}')

    # 3. Layer-2 verification rate
    vr = report.verification_rate()
    print('\n[Layer 2] physics verification rate:')
    for k, v in vr.items():
        print(f'    {k:13}: {v:.1%}')

    # 4. HEADLINE: CNN accuracy split by verdict
    hb = headline_accuracy_by_verdict(report, test_data.y)
    print('\n[HEADLINE] CNN accuracy by physics verdict:')
    for v, d in hb.items():
        print(f'    {v:13}: acc={d["cnn_accuracy"]:.3f}  (n={d["n"]})')
    if 'CONFIRMED' in hb and 'CONFLICT' in hb:
        gap = hb['CONFIRMED']['cnn_accuracy'] - hb['CONFLICT']['cnn_accuracy']
        print(
            f'    -> CONFIRMED minus CONFLICT accuracy gap: {gap:+.3f} '
            f'({"physics is a real reliability signal" if gap > 0 else "investigate"})'
        )

    # 5. Layer-3 causal do(Z)
    eff = model.condition_effect(full_data)
    print(
        f'\n[Layer 3] do(Z) operating-condition effect: rung={eff["rung"]} '
        f'max_contrast={eff["max_contrast"]:.4f} p={eff["p_value"]:.4f}'
    )

    # 6. Layer-3B: REAL Rung-3 counterfactual (must actually execute, not fallback)
    print(f'\n[Layer 3B] DoWhy available: {dowhy_gcm_available()}')
    if dowhy_gcm_available():
        feat = signal_kurtosis(full_data.X)
        rms = compute_vibration_rms(full_data.X)
        scm = build_scm(full_data.cond, feat, rms)
        if scm is not None:
            row = {
                'Z': float(full_data.cond[0]),
                'X': float(feat[0]),
                'Y': float(rms[0]),
            }
            cf_cond = int(min(np.unique(full_data.cond)))
            cf = counterfactual_for_unit(scm, row, cf_cond)
            print(f'    counterfactual executed: {cf["method"]}')
            print(
                f'    factual Y={cf["factual"]["Y"]:.2f} -> '
                f'counterfactual Y={cf["counterfactual"]["Y"]:.2f} at Z={cf_cond}'
            )
        else:
            print('    SCM build returned None (check DoWhy install)')
    else:
        print(
            '    DoWhy not installed - the real Rung-3 path was NOT validated. '
            'Run: pip install dowhy'
        )

    # 7. example auditable diagnoses
    print('\n[examples] auditable root-cause diagnoses (CONFIRMED faults only):')
    seen_classes = set()
    examples = []
    for idx, r in enumerate(report.records):
        y_true = test_data.y[idx]
        if r['physics_verdict'] == 'CONFIRMED' and y_true > 0:
            if y_true not in seen_classes:
                examples.append(
                    f'    [Class {y_true}] [{r["status"]}] {r["root_cause"]["statement"]}'
                )
                seen_classes.add(y_true)
            if len(examples) >= 5:
                break
    for ex in examples:
        print(ex)

    print('\n' + '=' * 68)
    print('VALIDATION COMPLETE - if all sections printed, the pipeline runs end-to-end.')
    print('=' * 68)


if __name__ == '__main__':
    main()

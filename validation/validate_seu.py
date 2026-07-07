"""
validate_seu.py - cross-domain validation: run the full CNSD pipeline on the SEU
gearbox dataset (a NON-bearing domain) using the GearProvider.

This is the universality test: the same five-layer pipeline, the same engine, a
different machine class - only the physics provider changes. If CNSD diagnoses
gear faults via gear-mesh physics, that demonstrates the provider interface is
genuinely domain-agnostic, not bearing-bound.

Run after the bearing baseline. Wire your SEU loader into load_seu() below - it
is the only dataset-specific code. SEU gearset data: tab-separated, 8 channels,
header ends at the line 'Data' (~line 16), 5 classes, 2 operating conditions.
"""

import glob
import os

import numpy as np

from cnsd import Dataset
from cnsd.diagnosis.system import CNSD
from cnsd.physics.configs import PhysicsConfig
from cnsd.physics.providers.gear import GearProvider

# ── SEU gearbox parameters ───────────────────────────────────────────────────
# Replace N_TEETH_INPUT with the actual driving-gear tooth count of the SEU rig.
# The SEU bench uses a planetary + parallel gearbox; use the parallel-stage
# driving gear tooth count for the gearset experiments. Confirm from the dataset
# documentation and set it here.
N_TEETH_INPUT = 20  # <-- CONFIRM against SEU rig spec
SEU_FS = 20000  # sampling rate (Hz) - confirm against your files
SEU_COND_TO_RPM = {0: 1800, 1: 1800}  # SEU '20_0' and '30_2' conditions -> rpm

# SEU 5-class taxonomy (gearset). Map your integer labels to these families.
SEU_TAXONOMY = {
    0: ('Health', 'None'),
    1: ('Chipped Tooth', 'Medium'),
    2: ('Missing Tooth', 'High'),
    3: ('Root Crack', 'Medium'),
    4: ('Surface Wear', 'Medium'),
}


def load_seu():
    """Return (X, y, cond) for the SEU gearset. Replace the body with your loader.

    X    : (n, 1024) float   per-window normalized vibration windows
    y    : (n,) int          0=Health, 1=Chipped, 2=Miss, 3=Root, 4=Surface
    cond : (n,) int          operating condition (0='20_0', 1='30_2')

    SEU gearset files are tab-separated with 8 channels; the header ends at a line
    containing 'Data' (~line 16), data starts the next line. Pick one channel
    (pre-commit to it before seeing results - no channel cherry-picking).
    """
    base_dir = os.environ.get('CNSD_DATA_SEU', r'E:\301\SEU-dataset\gearbox\gearset')

    label_map = {'Health': 0, 'Chipped': 1, 'Miss': 2, 'Root': 3, 'Surface': 4}
    X_list, y_list, cond_list = [], [], []

    for filepath in glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True):
        filename = os.path.basename(filepath)

        cond_idx = 0 if '20_0' in filename else (1 if '30_2' in filename else None)
        if cond_idx is None:
            continue

        label_idx = next((v for k, v in label_map.items() if filename.startswith(k)), None)
        if label_idx is None:
            continue

        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        data_start = next((i + 1 for i, line in enumerate(lines) if 'Data' in line), 0)

        channel_data = []
        for line in lines[data_start:]:
            parts = line.strip().split('\t')
            if len(parts) >= 8:
                try:
                    # Picking channel 2 (index 1: planetary gearbox vibration x)
                    channel_data.append(float(parts[1]))
                except ValueError:
                    pass

        channel_data = np.array(channel_data, dtype=np.float32)
        n_windows = len(channel_data) // 1024
        if n_windows == 0:
            continue

        windows = channel_data[: n_windows * 1024].reshape((n_windows, 1024))

        means = windows.mean(axis=1, keepdims=True)
        stds = windows.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        windows = (windows - means) / stds

        X_list.append(windows)
        y_list.extend([label_idx] * n_windows)
        cond_list.extend([cond_idx] * n_windows)

    if not X_list:
        raise ValueError(f'No SEU dataset files found or parsed successfully in {base_dir}')

    return np.vstack(X_list), np.array(y_list), np.array(cond_list)


def headline_by_verdict(report, y_true):
    pred = np.array([r['predicted_class'] for r in report.records])
    correct = pred == np.asarray(y_true)
    verd = np.array([r['physics_verdict'] for r in report.records])
    out = {}
    for v in ('CONFIRMED', 'CONFLICT', 'INCONCLUSIVE'):
        m = verd == v
        if m.any():
            out[v] = {'n': int(m.sum()), 'acc': float(correct[m].mean())}
    return out


def main():
    print('=' * 68)
    print('CNSD CROSS-DOMAIN VALIDATION - SEU GEARBOX (GearProvider)')
    print('=' * 68)

    X, y, cond = load_seu()
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    cond = np.asarray(cond)

    # gear physics config: the provider is built directly (no bearing geometry)
    physics = PhysicsConfig(
        bearing=None, cond_to_rpm=SEU_COND_TO_RPM, fs=SEU_FS, name='SEU-gearset'
    )
    # train/test split by condition (cross-condition within the gear domain)
    te = cond == 1
    tr = ~te
    train = Dataset.from_arrays(
        X[tr], y[tr], cond[tr], fs=SEU_FS, physics=physics, taxonomy=SEU_TAXONOMY, name='SEU_Train'
    )
    test = Dataset.from_arrays(
        X[te], y[te], cond[te], fs=SEU_FS, physics=physics, taxonomy=SEU_TAXONOMY, name='SEU_Test'
    )
    print(
        f'[data] train {len(train.X)}  test {len(test.X)}  classes={sorted(np.unique(y).tolist())}'
    )

    # CNSD with the gear provider explicitly (config path also works via
    # domain.type: gear once you add a YAML)
    model = CNSD()
    model.fit(train, epochs=30)
    # override the symbolic layer with the gear provider for this domain
    model.symbolic.provider = GearProvider(
        n_teeth_input=N_TEETH_INPUT, cond_to_rpm=SEU_COND_TO_RPM, fs=SEU_FS
    )
    model.symbolic.taxonomy = SEU_TAXONOMY

    report = model.diagnose(test)
    print(f'\n[pipeline] {report.summary()}')

    vr = report.verification_rate()
    print('\n[Layer 2] gear-physics verification rate:')
    for k, v in vr.items():
        print(f'    {k:13}: {v:.1%}')

    hb = headline_by_verdict(report, test.y)
    print('\n[HEADLINE] CNN accuracy by gear-physics verdict:')
    for v, d in hb.items():
        print(f'    {v:13}: acc={d["acc"]:.3f}  (n={d["n"]})')
    if 'CONFIRMED' in hb and 'CONFLICT' in hb:
        gap = hb['CONFIRMED']['acc'] - hb['CONFLICT']['acc']
        print(f'    -> CONFIRMED minus CONFLICT gap: {gap:+.3f}')

    print('\n[examples] gear root-cause diagnoses (CONFIRMED faults):')
    seen = set()
    for i, r in enumerate(report.records):
        if r['physics_verdict'] == 'CONFIRMED' and test.y[i] > 0 and test.y[i] not in seen:
            print(f'    [Class {test.y[i]}] {r["root_cause"]["statement"]}')
            seen.add(test.y[i])
        if len(seen) >= 4:
            break

    print('\n' + '=' * 68)
    print('CROSS-DOMAIN VALIDATION COMPLETE')
    print('=' * 68)


if __name__ == '__main__':
    main()

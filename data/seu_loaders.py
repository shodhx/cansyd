"""SEU (Southeast University) gearbox dataset loader - gear-fault subset.

Tab-separated multi-channel vibration files with a multi-line header (data
begins after a line starting with 'Data'). Five gear classes (health, chipped,
miss, root, surface) under two operating conditions (20_0, 30_2). The operating
condition is the variable across which causal invariance is tested.

Channel choice is pre-committed: column index 1 (the 2nd of 8 channels, the
planetary-gearbox vibration commonly used in the SEU literature). Place files in
data/raw/seu_gear/.

Returns the same 6-tuple as load_cwru_all.
"""
import os, glob
import numpy as np

SEU_DIR = './data/raw/seu_gear'
WINDOW, STEP_TRAIN, STEP_TEST = 1024, 256, 1024
CHANNEL = 1  # pre-committed channel (2nd column)
MAX_ROWS = 200000  # cap rows read per file (files are ~1M samples)
CLASSES = {'health': 0, 'chipped': 1, 'miss': 2, 'root': 3, 'surface': 4}
CONDS = {'20_0': 0, '30_2': 1}


def _read_signal(path, channel=CHANNEL, max_rows=MAX_ROWS):
    with open(path) as fh:
        lines = fh.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('Data')) + 1
    vals = []
    for l in lines[start:start + max_rows]:
        parts = [p for p in l.split('\t') if p.strip() != '']
        if len(parts) >= 8:
            vals.append(float(parts[channel]))
    return np.array(vals)


def _seg(sig, w, s):
    return np.array([sig[i:i + w] for i in range(0, len(sig) - w, s)])

def _norm(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)


def load_seu_gear_all(train_conds=('20_0',), test_conds=('30_2',)):
    files = sorted(glob.glob(f'{SEU_DIR}/*.csv'))
    if not files:
        raise FileNotFoundError(
            f"No SEU gear .csv files in {SEU_DIR}. Obtain the SEU 'Mechanical-datasets' "
            "gearbox/gearset files and place them there.")
    Xtr, ytr, ctr, Xte, yte, cte = [], [], [], [], [], []
    for f in files:
        base = os.path.basename(f).lower()
        label = next((v for k, v in CLASSES.items() if base.startswith(k)), None)
        cond = next((c for c in CONDS if c in base), None)
        if label is None or cond is None:
            continue
        sig = _read_signal(f)
        if cond in train_conds:
            s = _seg(sig, WINDOW, STEP_TRAIN)
            Xtr.append(s); ytr += [label] * len(s); ctr += [CONDS[cond]] * len(s)
        if cond in test_conds:
            s = _seg(sig, WINDOW, STEP_TEST)
            Xte.append(s); yte += [label] * len(s); cte += [CONDS[cond]] * len(s)
    if not Xtr or not Xte:
        raise RuntimeError("SEU parsed no usable segments; check filenames/conditions.")
    return (_norm(np.concatenate(Xtr))[..., None], np.array(ytr), np.array(ctr),
            _norm(np.concatenate(Xte))[..., None], np.array(yte), np.array(cte))

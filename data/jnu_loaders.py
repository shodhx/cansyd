"""JNU (Jiangnan University) bearing dataset loader.

Vibration signals at 50 kHz under three rotational speeds (600/800/1000 rpm),
four classes (normal, inner, outer, ball). Speed is the operating-condition
variable across which causal invariance is tested. CSV files, one per
(fault, speed). Place files in data/raw/jnu/.

Returns the same 6-tuple as load_cwru_all:
  (X_train, y_train, cond_train, X_test, y_test, cond_test)
"""
import os, glob
import numpy as np

JNU_DIR = './data/raw/jnu'
WINDOW, STEP_TRAIN, STEP_TEST = 1024, 256, 1024
SPEEDS = (600, 800, 1000)
FAULT_TOKENS = {'normal': 0, 'health': 0, 'inner': 1, 'outer': 2, 'ball': 3,
                'ib': 1, 'ir': 1, 'ob': 2, 'or': 2, 'tb': 3, 'roll': 3, 're': 3, 'n': 0}


def _parse(fname):
    base = os.path.basename(fname).lower().replace('.csv', '')
    speed = next((s for s in SPEEDS if str(s) in base), None)
    label = None
    for tok in sorted(FAULT_TOKENS, key=len, reverse=True):  # longest first
        if tok in base:
            label = FAULT_TOKENS[tok]; break
    return label, speed


def _seg(sig, w, s):
    sig = np.asarray(sig, float).flatten()
    return np.array([sig[i:i + w] for i in range(0, len(sig) - w, s)])

def _norm(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)


def load_jnu_all(train_speeds=(600, 800), test_speeds=(1000,)):
    files = sorted(glob.glob(f'{JNU_DIR}/*.csv'))
    if not files:
        raise FileNotFoundError(
            f"No JNU .csv files in {JNU_DIR}. Download from "
            "github.com/ClarkGableWang/JNU-Bearing-Dataset and place csv files there.")
    Xtr, ytr, ctr, Xte, yte, cte = [], [], [], [], [], []
    for f in files:
        lab, spd = _parse(f)
        if lab is None or spd is None:
            continue
        sig = np.loadtxt(f, delimiter=',').flatten()
        if spd in train_speeds:
            s = _seg(sig, WINDOW, STEP_TRAIN)
            Xtr.append(s); ytr += [lab] * len(s); ctr += [spd] * len(s)
        if spd in test_speeds:
            s = _seg(sig, WINDOW, STEP_TEST)
            Xte.append(s); yte += [lab] * len(s); cte += [spd] * len(s)
    if not Xtr or not Xte:
        raise RuntimeError("JNU parsed no usable segments; check filenames vs _parse().")
    return (_norm(np.concatenate(Xtr))[..., None], np.array(ytr), np.array(ctr),
            _norm(np.concatenate(Xte))[..., None], np.array(yte), np.array(cte))

"""
paderborn_loader.py - Paderborn (PU) bearing dataset loader for CROSS-RIG testing.

THE EXPERIMENT (the Bible's highest-leverage test): train on CWRU (EDM-machined
notches, 6205 bearing, 12/48 kHz) and test on Paderborn NATURALLY-damaged
bearings (real fatigue spalls, 6203 bearing, 64 kHz). If the model survives this
genuine cross-mechanism, cross-rig distribution shift, that is a real result. If
it collapses, that is also a real and publishable finding about what the CNN
learned. Either outcome is honest and worth reporting.

FILE FORMAT. MATLAB files named e.g. N15_M07_F10_KA01_1.mat. The prefix encodes
the operating condition; the 4-char code encodes the bearing:
  K0xx -> healthy ;  KIxx -> inner race ;  KAxx -> outer race.
Each .mat holds a nested struct; the vibration channel is 'vibration_1' under a
field named like the file stem. Sampling rate 64 kHz. Bearing type 6203.

CLASS MAPPING to the CWRU 3-superclass space used for cross-rig transfer
(CWRU 10-class collapses to {Normal, Inner, Outer}; Ball is CWRU-only and is
excluded from the cross-rig comparison since PU's 3-class scheme has no ball
class):
  Normal=0, Inner Race=1, Outer Race=2

DAMAGE-TYPE FILTER. `damage='natural'` selects only the real-spall bearings
(the meaningful test); `damage='artificial'` selects EDM/drill; `damage='all'`
takes both. Naturally-damaged codes are listed below per the PU fact sheets.
"""
import os
import glob
import numpy as np
from scipy.io import loadmat

PADERBORN_DIR = './data/raw/paderborn'
WINDOW = 1024
STEP = 1024            # non-overlapping (this is a held-out test set)
PU_FS = 64000          # Hz

# 6203 deep-groove ball bearing geometry (for characteristic frequencies)
BEARING_6203 = {'n_balls': 8, 'd_ball': 0.2520, 'd_pitch': 1.122, 'contact_angle': 0.0}

# Naturally-damaged bearing codes (real fatigue spalls, accelerated lifetime).
# Inner-race natural damage and outer-race natural damage, per PU fact sheets.
NATURAL_INNER = {'KI04', 'KI14', 'KI16', 'KI18', 'KI21'}
NATURAL_OUTER = {'KA04', 'KA15', 'KA16', 'KA22', 'KA30'}
HEALTHY = {'K001', 'K002', 'K003', 'K004', 'K005', 'K006'}
# Artificial (EDM/drill/engraving) - kept for completeness
ARTIFICIAL_INNER = {'KI01', 'KI03', 'KI05', 'KI07', 'KI08'}
ARTIFICIAL_OUTER = {'KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09'}


def _bearing_code(fname):
    """Extract the 4-char bearing code (e.g. KA04) from the filename."""
    stem = os.path.basename(fname).replace('.mat', '')
    for part in stem.split('_'):
        if len(part) == 4 and (part[0] == 'K'):
            return part
    return None


def _label_for(code, damage):
    """Map a bearing code to (label, is_selected) given the damage filter."""
    if code in HEALTHY:
        return 0, True
    inner = NATURAL_INNER | (ARTIFICIAL_INNER if damage in ('artificial', 'all') else set())
    outer = NATURAL_OUTER | (ARTIFICIAL_OUTER if damage in ('artificial', 'all') else set())
    if damage == 'natural':
        inner = NATURAL_INNER
        outer = NATURAL_OUTER
    if code in inner:
        return 1, True
    if code in outer:
        return 2, True
    return None, False


def _extract_vibration(mat, stem):
    """Pull the vibration_1 channel out of PU's nested MATLAB struct."""
    # the top-level field is usually the file stem
    key = stem if stem in mat else next((k for k in mat if not k.startswith('__')), None)
    if key is None:
        return None
    s = mat[key]
    # navigate the struct to Y -> entries -> find Name 'vibration_1' -> Data
    try:
        Y = s['Y'][0, 0]
        for i in range(Y.shape[1]):
            name = str(Y[0, i]['Name'][0])
            if 'vibration' in name.lower():
                return np.asarray(Y[0, i]['Data'][0]).flatten()
    except Exception:
        return None
    return None


def _seg(sig, w, s):
    sig = np.asarray(sig, float).flatten()
    return np.array([sig[i:i + w] for i in range(0, len(sig) - w, s)])

def _norm(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)


def load_paderborn_test(damage='natural', max_files_per_class=None):
    """Load Paderborn as a HELD-OUT TEST set (no train split - we train on CWRU).

    Returns (X_test, y_test) with y in {0:Normal, 1:Inner, 2:Outer}. Use
    damage='natural' for the meaningful cross-mechanism test.
    """
    files = sorted(glob.glob(f'{PADERBORN_DIR}/**/*.mat', recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No Paderborn .mat files under {PADERBORN_DIR}. Download from the KAt "
            "DataCenter (mb.uni-paderborn.de/kat) and place the per-bearing folders "
            "there (e.g. data/raw/paderborn/KA04/N15_M07_F10_KA04_1.mat).")
    X, y, counts = [], [], {0: 0, 1: 0, 2: 0}
    skipped = 0
    for f in files:
        code = _bearing_code(f)
        if code is None:
            continue
        label, sel = _label_for(code, damage)
        if not sel:
            continue
        if max_files_per_class and counts[label] >= max_files_per_class:
            continue
        stem = os.path.basename(f).replace('.mat', '')
        try:
            sig = _extract_vibration(loadmat(f), stem)
        except Exception:
            skipped += 1
            continue
        if sig is None or len(sig) < WINDOW:
            skipped += 1
            continue
        segs = _seg(sig, WINDOW, STEP)
        X.append(segs); y += [label] * len(segs); counts[label] += 1
    if not X:
        raise RuntimeError(
            "Paderborn parsed no usable signals. Check the folder layout and that "
            "the .mat files contain a 'vibration_1' channel.")
    print(f"Paderborn ({damage} damage): bearings used per class "
          f"{counts}; {skipped} files skipped")
    return _norm(np.concatenate(X))[..., None], np.array(y)

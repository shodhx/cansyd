import time
import os
import requests
import scipy.io
import numpy as np
import pandas as pd

# Local cache directories for the raw datasets
BASE_DIR   = './data/raw'
CWRU_DIR   = f'{BASE_DIR}/cwru'

for d in [CWRU_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 1. CASE WESTERN RESERVE UNIVERSITY LOADER ──
CWRU_FILES = {
    'Normal':   {0:'97.mat',  1:'98.mat',  2:'99.mat',  3:'100.mat'},
    'Ball_007': {0:'118.mat', 1:'119.mat', 2:'120.mat', 3:'121.mat'},
    'Ball_014': {0:'185.mat', 1:'186.mat', 2:'187.mat', 3:'188.mat'},
    'Ball_021': {0:'222.mat', 1:'223.mat', 2:'224.mat', 3:'225.mat'},
    'IR_007':   {0:'105.mat', 1:'106.mat', 2:'107.mat', 3:'108.mat'},
    'IR_014':   {0:'169.mat', 1:'170.mat', 2:'171.mat', 3:'172.mat'},
    'IR_021':   {0:'209.mat', 1:'210.mat', 2:'211.mat', 3:'212.mat'},
    'OR_007':   {0:'130.mat', 1:'131.mat', 2:'132.mat', 3:'133.mat'},
    'OR_014':   {0:'197.mat', 1:'198.mat', 2:'199.mat', 3:'200.mat'},
    'OR_021':   {0:'234.mat', 1:'235.mat', 2:'236.mat', 3:'237.mat'},
}
LABEL_TO_INT = {name: i for i, name in enumerate(CWRU_FILES.keys())}

BASE_CWRU = 'https://engineering.case.edu/sites/default/files/'

def _download_and_verify(url, path, fname):
    """Download one CWRU .mat with up to 3 retries; verify it is a valid signal file."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60); r.raise_for_status()
            with open(path, 'wb') as f:
                f.write(r.content)
            mat = scipy.io.loadmat(path)
            keys = [k for k in mat if 'DE_time' in k or 'BA_time' in k]
            if not keys:
                keys = [k for k in mat if not k.startswith('_')]
            assert len(mat[keys[0]].flatten()) > 1000
            return True
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            if attempt < 2:
                time.sleep(2)
            else:
                print(f'  FAILED: {fname} - {e}')
    return False

def download_cwru():
    """Fetch all 40 CWRU files from Case Western, saved as {fault}_load{ld}.mat.
    Skips files already present. Set up a Kaggle dataset instead if the CWRU
    servers are unreachable (place the same {fault}_load{ld}.mat files in CWRU_DIR).
    """
    ok = 0
    for fault, loads in CWRU_FILES.items():
        for ld, fname in loads.items():
            path = f'{CWRU_DIR}/{fault}_load{ld}.mat'
            if os.path.exists(path):
                try:
                    scipy.io.loadmat(path); ok += 1; continue
                except Exception:
                    os.remove(path)
            if _download_and_verify(BASE_CWRU + fname, path, fname):
                ok += 1
    print(f'CWRU: {ok}/40 files ready')
    return ok

def segment_signal(signal, window=1024, step=256):
    return np.array([signal[i:i+window] for i in range(0, len(signal)-window, step)])

def normalize_segments(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

def load_cwru_all(train_loads=(0,1,2), test_loads=(3,)):
    """Encapsulated extraction mapping CWRU windows to balanced training/test sets."""
    download_cwru()  # fetch the 40 .mat files if not already present
    X_tr, y_tr, ld_tr = [], [], []
    X_te, y_te, ld_te = [], [], []
    
    for fault_name, loads in CWRU_FILES.items():
        label = LABEL_TO_INT[fault_name]
        for ld in train_loads:
            path = f'{CWRU_DIR}/{fault_name}_load{ld}.mat'
            if not os.path.exists(path): continue
            mat = scipy.io.loadmat(path)
            key = [k for k in mat if 'DE_time' in k or 'BA_time' in k][0]
            sig = mat[key].flatten()
            segs = segment_signal(sig, 1024, 256)
            X_tr.append(segs); y_tr.extend([label]*len(segs)); ld_tr.extend([ld]*len(segs))
            
        for ld in test_loads:
            path = f'{CWRU_DIR}/{fault_name}_load{ld}.mat'
            if not os.path.exists(path): continue
            mat = scipy.io.loadmat(path)
            key = [k for k in mat if 'DE_time' in k or 'BA_time' in k][0]
            sig = mat[key].flatten()
            segs = segment_signal(sig, 1024, 1024)  # Non-overlapping evaluation split
            X_te.append(segs); y_te.extend([label]*len(segs)); ld_te.extend([ld]*len(segs))
            
    if not X_tr or not X_te:
        raise FileNotFoundError(
            f"No CWRU .mat files found in {CWRU_DIR}. The Case Western download may "
            "have failed (servers are sometimes unreachable). Either retry, or attach "
            "a CWRU dataset on Kaggle and place files named {fault}_load{ld}.mat "
            "(e.g. Normal_load0.mat) in that directory.")
    X_train = normalize_segments(np.concatenate(X_tr))[..., np.newaxis]
    X_test  = normalize_segments(np.concatenate(X_te))[..., np.newaxis]
    return X_train, np.array(y_tr), np.array(ld_tr), X_test, np.array(y_te), np.array(ld_te)


# ── rotating-machinery replication datasets ──
from data.jnu_loader import load_jnu_all
from data.seu_loader import load_seu_gear_all

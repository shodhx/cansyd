import time
import os
import requests
import scipy.io
import numpy as np
import pandas as pd

# Local cache directories for the raw datasets
BASE_DIR   = './data/raw'
CWRU_DIR   = f'{BASE_DIR}/cwru'
CMAPSS_DIR = f'{BASE_DIR}/cmapss'
MFPT_DIR   = f'{BASE_DIR}/mfpt'

for d in [CWRU_DIR, CMAPSS_DIR, MFPT_DIR]:
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

# ── 2. NASA CMAPSS LOADER ──
def load_cmapss():
    try:
        # Read from the local cache dir (data/raw/cmapss).
        df_train = pd.read_csv(f'{CMAPSS_DIR}/train_FD001.txt', sep=' ', header=None)
        df_train.drop(df_train.columns[[26, 27]], axis=1, inplace=True)
        df_train.columns = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
        
        max_cycle = df_train.groupby('unit')['cycle'].max().reset_index()
        max_cycle.columns = ['unit', 'max_cycle']
        df_train = df_train.merge(max_cycle, on='unit')
        df_train['rul'] = df_train['max_cycle'] - df_train['cycle']
        df_train['fault'] = (df_train['rul'] < 125).astype(int)
        
        X = df_train[[f's{i}' for i in range(2, 22)]].values
        y = df_train['fault'].values
        op = df_train[['op1', 'op2', 'op3']].values
        
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, op, test_size=0.3, random_state=42)
        
    except Exception as e:
        print("  [SYNTHETIC] CMAPSS raw data not found at "
              f"{CMAPSS_DIR}/ - using a reproducible synthetic generator. "
              "Results from this fallback must NOT be reported as real-data results.")
        # Self-contained simulation to perfectly match expected matrices [30, 14] or [X, 20]
        np.random.seed(42)
        total_samples = 800
        
        # Match data configuration shape constraints: 20 telemetry variables
        X_sim = np.random.randn(total_samples, 20)
        # Create an operational conditions control array (3 confounder channels)
        op_sim = np.random.randn(total_samples, 3)
        # Generate target failure vector binary flags
        y_sim = np.random.choice([0, 1], size=total_samples, p=[0.7, 0.3])
        
        from sklearn.model_selection import train_test_split
        return train_test_split(X_sim, y_sim, op_sim, test_size=0.3, random_state=42)

# ── 3. MIT-BIH ARRHYTHMIA LOADER ──
def load_mitbih_split(record_ids, seg_len=256):
    """Processes biomedical multi-channel waveforms into localized R-peak centered matrices."""
    import wfdb
    AAMI_MAP = {'N':0,'L':0,'R':0,'e':0,'j':0, 'A':1,'a':1,'J':1,'S':1, 'V':2,'E':2}
    X, y, rr = [], [], []
    
    for rid in record_ids:
        try:
            rec = wfdb.rdrecord(str(rid), pn_dir='mitdb')
            ann = wfdb.rdann(str(rid), 'atr', pn_dir='mitdb')
        except Exception:
            try:
                rec = wfdb.rdrecord(str(rid))
                ann = wfdb.rdann(str(rid), 'atr')
            except Exception:
                continue
                
        signal = rec.p_signal[:, 0]
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        peaks, symbols = ann.sample, ann.symbol
        rr_intervals = np.diff(peaks, prepend=peaks[0])
        
        for peak, sym, r_val in zip(peaks, symbols, rr_intervals):
            if sym not in AAMI_MAP: continue
            cls = AAMI_MAP[sym]
            start, end = peak - seg_len//2, peak + seg_len//2
            if start < 0 or end > len(signal): continue
            X.append(signal[start:end])
            y.append(cls)
            rr.append(r_val / rec.fs)
            
    return np.array(X).reshape(-1, seg_len, 1), np.array(y), np.array(rr)

# ── 4. MFPT BEARING LOADER ──
def load_mfpt():
    """Extracts variable RPM industrial rolling-element diagnostics.

    NOTE: this loader currently has no real-MFPT download path - it returns a
    reproducible synthetic signal that mirrors the Section 9C fallback. Numbers
    from it characterise the pipeline on synthetic data only and must not be
    reported as real MFPT results until a real-data path is wired in.
    """
    print("  [SYNTHETIC] MFPT real data not bundled - using reproducible "
          "synthetic signals (seed=42). Do not report as real-data results.")
    np.random.seed(42)
    # Uniform simulation fallback matching Section 9C logic models
    t = np.linspace(0, 2, 48828 * 2)
    normal = 1.0 * np.sin(2*np.pi*60*t) + 0.1*np.random.randn(len(t))
    inner  = 2.5 * np.sin(2*np.pi*162*t) + 0.1*np.random.randn(len(t))
    outer  = 2.0 * np.sin(2*np.pi*105*t) + 0.1*np.random.randn(len(t))
    
    X, y, rpm = [], [], []
    for label, sig in [(0, normal), (1, inner), (2, outer)]:
        for i in range(0, len(sig)-1024, 512):
            seg = sig[i:i+1024]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            X.append(seg)
            y.append(label)
            rpm.append(np.random.uniform(250, 300))
            
    from sklearn.model_selection import train_test_split
    X = np.array(X).reshape(-1, 1024, 1)
    y = np.array(y)
    rpm = np.array(rpm)
    # Split into train/test so callers get 6 values (X_tr, X_te, y_tr, y_te, rpm_tr, rpm_te)
    return train_test_split(X, y, rpm, test_size=0.3, random_state=42, stratify=y)
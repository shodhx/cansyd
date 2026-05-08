import urllib.request
import os
from scipy.io import loadmat
import numpy as np
import wfdb
import pandas as pd

CWRU_DIR = 'cwru'
os.makedirs(CWRU_DIR, exist_ok=True)

files = {
    'Normal_0': 'https://engineering.case.edu/sites/default/files/97.mat',
    'Normal_1': 'https://engineering.case.edu/sites/default/files/98.mat',
    'Normal_2': 'https://engineering.case.edu/sites/default/files/99.mat',
    'Normal_3': 'https://engineering.case.edu/sites/default/files/100.mat',
    'Ball_007_0': 'https://engineering.case.edu/sites/default/files/118.mat',
    'Ball_007_1': 'https://engineering.case.edu/sites/default/files/119.mat',
    'Ball_007_2': 'https://engineering.case.edu/sites/default/files/120.mat',
    'Ball_007_3': 'https://engineering.case.edu/sites/default/files/121.mat',
    'Ball_014_0': 'https://engineering.case.edu/sites/default/files/185.mat',
    'Ball_014_1': 'https://engineering.case.edu/sites/default/files/186.mat',
    'Ball_014_2': 'https://engineering.case.edu/sites/default/files/187.mat',
    'Ball_014_3': 'https://engineering.case.edu/sites/default/files/188.mat',
    'Ball_021_0': 'https://engineering.case.edu/sites/default/files/222.mat',
    'Ball_021_1': 'https://engineering.case.edu/sites/default/files/223.mat',
    'Ball_021_2': 'https://engineering.case.edu/sites/default/files/224.mat',
    'Ball_021_3': 'https://engineering.case.edu/sites/default/files/225.mat',
    'IR_007_0': 'https://engineering.case.edu/sites/default/files/105.mat',
    'IR_007_1': 'https://engineering.case.edu/sites/default/files/106.mat',
    'IR_007_2': 'https://engineering.case.edu/sites/default/files/107.mat',
    'IR_007_3': 'https://engineering.case.edu/sites/default/files/108.mat',
    'IR_014_0': 'https://engineering.case.edu/sites/default/files/169.mat',
    'IR_014_1': 'https://engineering.case.edu/sites/default/files/170.mat',
    'IR_014_2': 'https://engineering.case.edu/sites/default/files/171.mat',
    'IR_014_3': 'https://engineering.case.edu/sites/default/files/172.mat',
    'IR_021_0': 'https://engineering.case.edu/sites/default/files/209.mat',
    'IR_021_1': 'https://engineering.case.edu/sites/default/files/210.mat',
    'IR_021_2': 'https://engineering.case.edu/sites/default/files/211.mat',
    'IR_021_3': 'https://engineering.case.edu/sites/default/files/212.mat',
    'OR_007_3': 'https://engineering.case.edu/sites/default/files/130.mat',
    'OR_007_6': 'https://engineering.case.edu/sites/default/files/131.mat',
    'OR_007_12': 'https://engineering.case.edu/sites/default/files/132.mat',
    'OR_014_6': 'https://engineering.case.edu/sites/default/files/197.mat',
    'OR_021_6': 'https://engineering.case.edu/sites/default/files/234.mat',
    'OR_007_0': 'https://engineering.case.edu/sites/default/files/144.mat',
    'OR_014_0': 'https://engineering.case.edu/sites/default/files/189.mat',
    'OR_021_0': 'https://engineering.case.edu/sites/default/files/226.mat',
    'OR_007_1': 'https://engineering.case.edu/sites/default/files/145.mat',
    'OR_014_1': 'https://engineering.case.edu/sites/default/files/190.mat',
    'OR_021_1': 'https://engineering.case.edu/sites/default/files/227.mat',
    'OR_007_2': 'https://engineering.case.edu/sites/default/files/146.mat',
    'OR_014_2': 'https://engineering.case.edu/sites/default/files/191.mat',
    'OR_021_2': 'https://engineering.case.edu/sites/default/files/228.mat',
}

for name, url in files.items():
    path = f'{CWRU_DIR}/{name}.mat'
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except:
            pass

def load_cwru_class(pattern, label, loads=[0,1,2,3], seg_len=1024, step=256):
    X = []
    y = []
    for load in loads:
        fpath = f'{CWRU_DIR}/{pattern}_{load}.mat'
        if not os.path.exists(fpath):
            continue
        try:
            mat = loadmat(fpath)
            key = [k for k in mat if 'DE_time' in k or 'BA_time' in k or 'FE_time' in k]
            if not key:
                key = [k for k in mat if not k.startswith('_')]
            signal = mat[key[0]].flatten()
            for i in range(0, len(signal)-seg_len+1, step):
                seg = signal[i:i+seg_len]
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)
                X.append(seg)
                y.append(label)
        except:
            continue
    return np.array(X), np.array(y)

X_norm, y_norm = load_cwru_class('Normal', 0)
X_b07, y_b07 = load_cwru_class('Ball_007', 1)
X_b14, y_b14 = load_cwru_class('Ball_014', 2)
X_b21, y_b21 = load_cwru_class('Ball_021', 3)
X_i07, y_i07 = load_cwru_class('IR_007', 4)
X_i14, y_i14 = load_cwru_class('IR_014', 5)
X_i21, y_i21 = load_cwru_class('IR_021', 6)
X_o07, y_o07 = load_cwru_class('OR_007', 7)
X_o14, y_o14 = load_cwru_class('OR_014', 8)
X_o21, y_o21 = load_cwru_class('OR_021', 9)

X_train_all = np.concatenate([X_norm, X_b07, X_b14, X_b21, X_i07, X_i14, X_i21, X_o07, X_o14, X_o21])
y_train_all = np.concatenate([y_norm, y_b07, y_b14, y_b21, y_i07, y_i14, y_i21, y_o07, y_o14, y_o21])

X_norm_t, y_norm_t = load_cwru_class('Normal', 0, [3])
X_b07_t, y_b07_t = load_cwru_class('Ball_007', 1, [3])
X_b14_t, y_b14_t = load_cwru_class('Ball_014', 2, [3])
X_b21_t, y_b21_t = load_cwru_class('Ball_021', 3, [3])
X_i07_t, y_i07_t = load_cwru_class('IR_007', 4, [3])
X_i14_t, y_i14_t = load_cwru_class('IR_014', 5, [3])
X_i21_t, y_i21_t = load_cwru_class('IR_021', 6, [3])
X_o07_t, y_o07_t = load_cwru_class('OR_007', 7, [3])
X_o14_t, y_o14_t = load_cwru_class('OR_014', 8, [3])
X_o21_t, y_o21_t = load_cwru_class('OR_021', 9, [3])

X_test_all = np.concatenate([X_norm_t, X_b07_t, X_b14_t, X_b21_t, X_i07_t, X_i14_t, X_i21_t, X_o07_t, X_o14_t, X_o21_t])
y_test_all = np.concatenate([y_norm_t, y_b07_t, y_b14_t, y_b21_t, y_i07_t, y_i14_t, y_i21_t, y_o07_t, y_o14_t, y_o21_t])

X_train_all = X_train_all.reshape(-1, 1024, 1)
X_test_all = X_test_all.reshape(-1, 1024, 1)

train_recs = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
test_recs = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

aami = {'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'A': 1, 'a': 1, 'J': 1, 'S': 1, 'V': 2, 'E': 2}

def load_mitbih_split(recs):
    X = []
    y = []
    rr = []
    for rec in recs:
        try:
            record = wfdb.rdrecord(f'mitdb/{rec}', pn_dir='mitdb')
            ann = wfdb.rdann(f'mitdb/{rec}', 'atr', pn_dir='mitdb')
            signal = record.p_signal[:, 0]
            for i, (idx, sym) in enumerate(zip(ann.sample, ann.symbol)):
                if sym not in aami:
                    continue
                if idx < 128 or idx > len(signal) - 128:
                    continue
                seg = signal[idx - 128:idx + 128]
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)
                if i > 0:
                    rr_int = (idx - ann.sample[i-1]) / record.fs
                else:
                    rr_int = 1.0
                X.append(seg)
                y.append(aami[sym])
                rr.append(rr_int)
        except:
            continue
    return np.array(X).reshape(-1, 256, 1), np.array(y), np.array(rr)

X_train_ecg, y_train_ecg, rr_train_ecg = load_mitbih_split(train_recs)
X_test_ecg, y_test_ecg, rr_test_ecg = load_mitbih_split(test_recs)

SYNTHETIC = False

def load_mfpt():
    global SYNTHETIC
    try:
        pass
    except:
        SYNTHETIC = True
        np.random.seed(42)
        t = np.linspace(0, 3, 97656*3)
        normal = 1.0 * np.sin(2*np.pi*60*t) + 0.1*np.random.randn(len(t))
        inner = 2.5 * np.sin(2*np.pi*170*t) + 0.1*np.random.randn(len(t))
        outer = 2.0 * np.sin(2*np.pi*105*t) + 0.1*np.random.randn(len(t))
        
        X = []
        y = []
        rpm = []
        for label, sig, r in [(0, normal, 1500), (1, inner, 1500), (2, outer, 1500)]:
            for i in range(0, len(sig)-1024, 256):
                seg = sig[i:i+1024]
                seg = (seg-seg.mean())/(seg.std()+1e-8)
                X.append(seg)
                y.append(label)
                rpm.append(r)
        
        X = np.array(X).reshape(-1,1024,1)
        y = np.array(y)
        rpm = np.array(rpm)
        
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, rpm, test_size=0.3, stratify=y, random_state=42)

X_train_mfpt, X_test_mfpt, y_train_mfpt, y_test_mfpt, rpm_train, rpm_test = load_mfpt()

def load_cmapss():
    url = 'https://ti.arc.nasa.gov/c/6/'
    df_train = pd.read_csv('train_FD001.txt', sep=' ', header=None)
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

X_train_cm, X_test_cm, y_train_cm, y_test_cm, op_train, op_test = load_cmapss()
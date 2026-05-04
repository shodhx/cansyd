import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

class SignalLoader:
    def __init__(self, config):
        self.cfg = config
        self.win = config['params']['window']
        self.stride = config['params']['stride']

    def fetch_cwru(self, file_path):
        """Extracts DE vibration; handles varying .mat keys."""
        res = sio.loadmat(file_path)
        # Regex-style lookup for the DE_time signal
        key = [k for k in res.keys() if 'DE_time' in k][0]
        sig = res[key].flatten()
        return self._slice(sig)

    def fetch_cmapss(self, file_path):
        """Loads NASA telemetry and drops zero-variance sensors."""
        cols = ['id', 'cycle', 'os1', 'os2', 'os3'] + [f's_{i}' for i in range(1, 22)]
        df = pd.read_csv(file_path, sep='\s+', header=None, names=cols)
        
        # Standard cleaning for diagnostic research
        drop_list = ['os3', 's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
        df.drop(columns=drop_list, inplace=True)
        return df

    def _slice(self, sig):
        """Fast sliding window using list comprehension."""
        count = (len(sig) - self.win) // self.stride
        windows = [sig[i*self.stride : i*self.stride + self.win] for i in range(count)]
        return np.array(windows).reshape(-1, self.win, 1)

    def get_cwru_batch(self):
        """Aggregates all .mat files in the CWRU directory."""
        path = self.cfg['paths']['cwru']
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mat')]
        data = np.vstack([self.fetch_cwru(f) for f in files])
        return data
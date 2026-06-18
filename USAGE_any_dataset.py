"""
USAGE_any_dataset.py - how to run CNSD on ANY vibration dataset.

CNSD is a system: it consumes a universal Dataset object, not a hardcoded
loader. To use a NEW dataset, wrap your arrays and (optionally) give the bearing
physics. No change to any pipeline code is required.
"""
import numpy as np
from core.dataset import Dataset, PhysicsConfig

# --- Option A: a NEW dataset, geometry unknown (geometry-free verification) ---
# X: (n, window) or (n, window, 1) signals; y: int labels (0=normal);
# cond: operating condition per sample; fs: sampling rate in Hz.
def example_byo():
    X = np.random.randn(200, 1024)
    y = np.random.randint(0, 4, 200)
    cond = np.random.choice([0, 1, 2], 200)
    data = Dataset.from_arrays(X, y, cond, fs=25600, name='my_rig')
    print(data.summary())
    return data

# --- Option B: a NEW bearing dataset WITH geometry (named-frequency check) ---
def example_with_physics():
    X = np.random.randn(200, 1024)
    y = np.random.randint(0, 3, 200)
    cond = np.full(200, 0)
    phys = PhysicsConfig(
        bearing={'n_balls': 8, 'd_ball': 0.276, 'd_pitch': 1.245, 'contact_angle': 0.0},
        cond_to_rpm={0: 1500},      # this condition runs at 1500 rpm
        fs=64000, name='my_bearing')
    data = Dataset.from_arrays(X, y, cond, fs=64000, physics=phys, name='my_rig_phys')
    print(data.summary())
    return data

# --- Then run the SAME pipeline on either, with no dataset-specific code ---
def run(cnn, jepa_probe, encoder, patchify_fn, data):
    from core.pipeline import CNSDPipeline
    pipe = CNSDPipeline.for_dataset(cnn, jepa_probe, encoder, patchify_fn, data)
    records = pipe.predict_dataset(data)     # auditable diagnoses for every sample
    return records

if __name__ == '__main__':
    example_byo()
    example_with_physics()

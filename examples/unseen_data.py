"""Run CANSYD on any vibration dataset - no bespoke loader, no prior knowledge."""

import numpy as np

from cansyd import CANSYD, Dataset, PhysicsConfig

# your arrays: signals, labels, operating condition per sample, sampling rate
X = np.random.randn(300, 1024)
y = np.random.randint(0, 4, 300)
cond = np.random.choice([0, 1, 2], 300)

# optional: give bearing geometry for named-frequency verification
physics = PhysicsConfig(
    bearing={'n_balls': 8, 'd_ball': 0.276, 'd_pitch': 1.245, 'contact_angle': 0.0},
    cond_to_rpm={0: 1500, 1: 1772, 2: 2000},
    fs=64000,
    name='my_bearing',
)

data = Dataset.from_arrays(X, y, cond, fs=64000, physics=physics, name='my_rig')
report = CANSYD().fit(data).diagnose(data)
print(report.summary())

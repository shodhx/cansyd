"""The universal data contract. Any vibration dataset plugs in through
Dataset.from_arrays(X, y, cond, fs); the system needs no prior knowledge of it."""

from cansyd.datasets.contract import Dataset
from cansyd.physics.configs import PhysicsConfig

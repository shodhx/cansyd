"""The universal data contract. Any vibration dataset plugs in through
Dataset.from_arrays(X, y, cond, fs); the system needs no prior knowledge of it."""
from cnsd.datasets.contract import Dataset
from cnsd.physics.configs import PhysicsConfig

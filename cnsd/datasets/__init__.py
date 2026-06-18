"""Datasets: the universal Dataset contract + built-in loaders.

Any vibration dataset plugs in via Dataset.from_arrays(X, y, cond, fs). 
"""
from cnsd.datasets.contract import Dataset, load_dataset
from cnsd.physics.configs import PhysicsConfig, CWRU_PHYSICS, JNU_PHYSICS, SEU_PHYSICS

"""
The universal dataset contract for CNSD.

A dataset provides:
  X     : (n, window, 1) float array of per-segment-normalised signals
  y      : (n,) int fault labels (0 = normal/healthy)
  cond  : (n,) operating condition per sample (load / speed / torque code)
  fs    : sampling rate (Hz) - needed to convert frequencies to Hz
  physics: optional PhysicsConfig (bearing geometry + rpm map). If absent, the
           symbolic layer falls back to geometry-free verification.

Built-in adapters wrap the existing CWRU/JNU/SEU/Paderborn loaders into this
contract, but ANY user array can be wrapped with Dataset.from_arrays(...).
"""
import numpy as np
from cnsd.physics.configs import PhysicsConfig, CWRU_PHYSICS, JNU_PHYSICS, SEU_PHYSICS
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable




@dataclass
class Dataset:
    """The universal CNSD dataset object. Hold the data + the physics needed to
    interpret it. The pipeline consumes ONLY this - never a dataset-specific API.
    """
    X: np.ndarray
    y: np.ndarray
    cond: np.ndarray
    fs: int
    physics: Optional[PhysicsConfig] = None
    name: str = 'dataset'

    def __post_init__(self):
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        self.cond = np.asarray(self.cond)
        if self.X.ndim == 2:
            self.X = self.X[..., np.newaxis]
        n = len(self.X)
        assert len(self.y) == n and len(self.cond) == n, \
            "X, y, cond must have the same length"

    @property
    def has_physics(self):
        return self.physics is not None

    @property
    def window(self):
        return self.X.shape[1]

    @classmethod
    def from_arrays(cls, X, y, cond, fs, physics=None, name='custom'):
        """Wrap arbitrary user arrays as a CNSD dataset. This is the entry point
        for ANY new vibration dataset - no bespoke loader required."""
        return cls(X=X, y=y, cond=cond, fs=fs, physics=physics, name=name)

    def summary(self):
        classes = sorted(np.unique(self.y).tolist())
        conds = sorted(np.unique(self.cond).tolist())
        return (f"Dataset '{self.name}': {len(self.X)} samples, window={self.window}, "
                f"fs={self.fs}Hz, classes={classes}, conditions={conds}, "
                f"physics={'yes' if self.has_physics else 'NO (geometry-free fallback)'}")


# ── Built-in physics configs for the bundled datasets ───────────────────────



# SEU gears: no rolling-bearing characteristic frequencies apply (gear-mesh
# physics differs); use geometry-free fallback for the symbolic layer.


# ── Adapters: wrap the existing loaders into the universal Dataset ──────────

def _split_to_dataset(loaded6, fs, physics, name):
    """Combine a loader's (Xtr,ytr,ctr,Xte,yte,cte) 6-tuple into one Dataset."""
    Xtr, ytr, ctr, Xte, yte, cte = loaded6
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    cond = np.concatenate([ctr, cte])
    return Dataset(X=X, y=y, cond=cond, fs=fs, physics=physics, name=name)


def load_dataset(name, **kwargs):
    """Factory: return a Dataset for any bundled dataset by name. New datasets
    are added via Dataset.from_arrays(...) without touching the pipeline.
    """
    name = name.lower()
    if name == 'cwru':
        from cnsd.datasets.cwru import load_cwru_all
        return _split_to_dataset(load_cwru_all(**kwargs), 12000, CWRU_PHYSICS, 'CWRU')
    if name == 'jnu':
        from cnsd.datasets.cwru import load_jnu_all
        return _split_to_dataset(load_jnu_all(**kwargs), 50000, JNU_PHYSICS, 'JNU')
    if name == 'seu':
        from cnsd.datasets.cwru import load_seu_gear_all
        return _split_to_dataset(load_seu_gear_all(**kwargs), 5120, SEU_PHYSICS, 'SEU')
    raise ValueError(f"Unknown dataset '{name}'. For a new dataset use "
                     "Dataset.from_arrays(X, y, cond, fs, physics=...).")

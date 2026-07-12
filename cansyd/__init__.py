"""
CANSYD - Causal Neuro-Symbolic Diagnosis.

A deployable five-layer fault-diagnosis system for rotating machinery:

    Layer 1  perception      1D CNN + S-JEPA
    Layer 2  symbolic        physics verification + root cause (can override CNN)
    Layer 3  causal          Pearl Rung-2 do(Z) intervention + refutation
    Layer 3B counterfactual  Pearl Rung-3 via DoWhy gcm (sensitivity fallback)
    Layer 4  consensus        confidence + physics verdict -> actionable status

Quick start:

    from cansyd import CANSYD, Dataset
    data   = Dataset.from_arrays(signals, labels, condition, fs=12000)
    report = CANSYD().fit(data).diagnose(data)
    print(report.summary())

Subpackages are independently importable - e.g. `from cansyd.causal import
intervention_effect_of_condition` does NOT require TensorFlow. Only the full
CANSYD system (which trains the CNN) needs the perception backend.
"""

__version__ = '1.0.0'

# Lazy top-level names (PEP 562): keeps `from cansyd import CANSYD` working without
# forcing TensorFlow on users who only want the causal / physics / dataset tools.
_LAZY = {
    'CANSYD': ('cansyd.api', 'CANSYD'),
    'DiagnosisReport': ('cansyd.diagnosis', 'DiagnosisReport'),
    'Dataset': ('cansyd.datasets', 'Dataset'),
    'PhysicsConfig': ('cansyd.physics', 'PhysicsConfig'),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'cansyd' has no attribute '{name}'")

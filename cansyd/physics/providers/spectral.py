"""Generic spectral provider: the zero-knowledge universal fallback.

When no domain physics is configured, this provider verifies only that a
dominant impulsive frequency is present in the envelope spectrum. It names no
component (it has no machine model) but still gives the engine a physics-based
CONFIRMED / INCONCLUSIVE signal, so CNSD runs on any signal from any machine.
"""

import numpy as np

from cnsd.physics.bearing import envelope_spectrum
from cnsd.physics.providers.base import PhysicsProvider


class SpectralProvider(PhysicsProvider):
    families = ['Fault']

    def __init__(self, fs: int = 12000):
        self.fs = fs

    def evidence(self, signal, condition=None):
        freqs, mag = envelope_spectrum(signal, fs=self.fs)
        if len(mag) < 2:
            return {'family_strength': {'Fault': 0.0}, 'frequencies_hz': {}}
        peak_i = int(np.argmax(mag))
        baseline = np.median(mag) + 1e-12
        prominence = float(mag[peak_i] / baseline)
        return {
            'family_strength': {'Fault': prominence},
            'frequencies_hz': {'dominant': float(freqs[peak_i])},
        }

    def root_cause(self, family, evidence):
        f = evidence.get('frequencies_hz', {}).get('dominant')
        strength = evidence.get('family_strength', {}).get('Fault', 0.0)
        if f is not None and strength > 0:
            statement = (
                f'A dominant impulsive component at {f:.1f} Hz '
                f'(strength {strength:.1f}) indicates a fault, but the '
                f'machine type is unspecified so the component cannot be named.'
            )
        else:
            statement = 'No dominant impulsive frequency; no fault localized.'
        return {
            'component': None,
            'fault_type': family,
            'mechanism': None,
            'evidence_frequency': 'dominant',
            'evidence_frequency_hz': f,
            'evidence_strength': float(strength),
            'statement': statement,
        }

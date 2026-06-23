"""Bearing physics provider: rolling-element fault verification.

Computes the ball-pass frequencies (BPFO/BPFI/BSF) from bearing geometry and
shaft speed and measures their prominence in the envelope spectrum. This is the
reference implementation of PhysicsProvider.
"""

from cnsd.physics.bearing import fault_frequency_evidence
from cnsd.physics.providers.base import PhysicsProvider

_ROOT_CAUSE = {
    'Outer Race': (
        'outer race',
        'BPFO',
        'a defect on the stationary outer race struck by each rolling element',
    ),
    'Inner Race': (
        'inner race',
        'BPFI',
        'a defect on the rotating inner race, modulated by shaft rotation',
    ),
    'Ball': (
        'rolling element',
        'BSF',
        'a defect on a rolling element striking both races as it spins',
    ),
    'Normal': (None, None, 'no localized defect; broadband vibration with no dominant frequency'),
}


class BearingProvider(PhysicsProvider):
    families = ['Outer Race', 'Inner Race', 'Ball']

    def __init__(self, bearing: dict, cond_to_rpm: dict, fs: int):
        self.bearing = bearing
        self.cond_to_rpm = cond_to_rpm
        self.fs = fs

    def evidence(self, signal, condition):
        rpm = self.cond_to_rpm.get(int(condition), next(iter(self.cond_to_rpm.values()), 1797))
        ev = fault_frequency_evidence(signal, rpm, fs=self.fs, bearing=self.bearing)
        return {
            'family_strength': {
                'Outer Race': ev['BPFO'],
                'Inner Race': ev['BPFI'],
                'Ball': ev['BSF'],
            },
            'frequencies_hz': ev['freqs_hz'],
        }

    def root_cause(self, family, evidence):
        comp, freq_name, mechanism = _ROOT_CAUSE.get(family, (None, None, ''))
        freq_hz = evidence.get('frequencies_hz', {}).get(freq_name) if freq_name else None
        strength = evidence.get('family_strength', {}).get(family, 0.0)
        if family == 'Normal':
            statement = (
                'No localized bearing defect; vibration is consistent with a healthy bearing.'
            )
        elif freq_hz is not None:
            statement = (
                f'Defect on the {comp} ({family}), evidenced by a peak at the '
                f'{freq_name} ({freq_hz:.1f} Hz, strength {strength:.1f}). {mechanism}.'
            )
        else:
            statement = f'{family} defect on the {comp}.'
        return {
            'component': comp,
            'fault_type': family,
            'mechanism': mechanism,
            'evidence_frequency': freq_name,
            'evidence_frequency_hz': freq_hz,
            'evidence_strength': float(strength),
            'statement': statement,
        }

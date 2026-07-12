"""Gear physics provider: gearbox fault verification via gear-mesh signatures.

Implements the PhysicsProvider interface for gearboxes. Verifies a predicted gear
fault against the gear-mesh frequency (GMF) and its shaft-rate sidebands:
localized tooth faults (chip, root crack, missing tooth) raise the sidebands;
distributed surface wear raises the mesh harmonics. The SEU gearbox taxonomy
(Health, Chipped, Miss, Root, Surface) maps onto these two evidence channels.
"""

from cansyd.physics.gear import gear_fault_evidence
from cansyd.physics.providers.base import PhysicsProvider

# SEU gearbox fault families -> which evidence channel signifies them.
# 'localized' faults show shaft-rate sidebands; 'distributed' wear shows mesh.
_FAMILY_CHANNEL = {
    'Chipped Tooth': 'sideband',
    'Missing Tooth': 'sideband',
    'Root Crack': 'sideband',
    'Surface Wear': 'mesh',
}

_ROOT_CAUSE = {
    'Chipped Tooth': (
        'gear tooth',
        'GMF sidebands',
        'a chipped tooth striking once per revolution, producing '
        'shaft-rate sidebands around the gear-mesh frequency',
    ),
    'Missing Tooth': (
        'gear tooth',
        'GMF sidebands',
        'a missing tooth producing a strong once-per-revolution impact and shaft-rate sidebands',
    ),
    'Root Crack': (
        'gear tooth root',
        'GMF sidebands',
        'a root crack reducing tooth stiffness, modulating the mesh at the shaft rate',
    ),
    'Surface Wear': (
        'gear flank',
        'GMF harmonics',
        'distributed surface wear raising the gear-mesh harmonics',
    ),
    'Health': (None, None, 'no gear fault; mesh spectrum is clean'),
}


class GearProvider(PhysicsProvider):
    families = ['Chipped Tooth', 'Missing Tooth', 'Root Crack', 'Surface Wear']

    def __init__(self, n_teeth_input, n_teeth_output=None, cond_to_rpm=None, fs=5120):
        self.n_teeth_input = n_teeth_input
        self.n_teeth_output = n_teeth_output
        self.cond_to_rpm = cond_to_rpm or {0: 1800}
        self.fs = fs

    def evidence(self, signal, condition):
        rpm = self.cond_to_rpm.get(int(condition), next(iter(self.cond_to_rpm.values()), 1800))
        ev = gear_fault_evidence(signal, rpm, self.n_teeth_input, self.n_teeth_output, fs=self.fs)
        # localized fault families share the sideband channel; surface wear the mesh
        side, mesh = ev['sideband_strength'], ev['mesh_strength']
        return {
            'family_strength': {
                'Chipped Tooth': side,
                'Missing Tooth': side,
                'Root Crack': side,
                'Surface Wear': mesh,
            },
            'frequencies_hz': ev['freqs_hz'],
            '_channels': {'sideband': side, 'mesh': mesh},
        }

    def root_cause(self, family, evidence):
        comp, freq_name, mechanism = _ROOT_CAUSE.get(family, (None, None, ''))
        channel = _FAMILY_CHANNEL.get(family)
        strength = evidence.get('_channels', {}).get(channel, 0.0) if channel else 0.0
        gmf = evidence.get('frequencies_hz', {}).get('GMF')
        if family in ('Health', None):
            statement = 'No gear fault detected; the mesh spectrum is clean.'
        elif gmf is not None:
            statement = (
                f'Defect on the {comp} ({family}), evidenced by '
                f'{freq_name} around GMF={gmf:.1f} Hz '
                f'(strength {strength:.1f}). {mechanism}.'
            )
        else:
            statement = f'{family} on the {comp}.'
        return {
            'component': comp,
            'fault_type': family,
            'mechanism': mechanism,
            'evidence_frequency': freq_name,
            'evidence_frequency_hz': gmf,
            'evidence_strength': float(strength),
            'statement': statement,
        }

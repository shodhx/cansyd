"""Smoke tests for the CANSYD layers (no TensorFlow / DoWhy required)."""

import numpy as np

from cansyd.causal import intervention_effect_of_condition
from cansyd.consensus import fuse
from cansyd.physics import characteristic_frequencies
from cansyd.physics.providers import BearingProvider, SpectralProvider, get_provider
from cansyd.symbolic import PhysicsRuleEngine

_BEARING = {'n_balls': 9, 'd_ball': 0.3126, 'd_pitch': 1.537, 'contact_angle': 0.0}
_TAXONOMY = {0: ('Normal', 'None'), 5: ('Inner Race', 'Medium'), 8: ('Outer Race', 'Medium')}


def _impulse(freq, fs=12000, n=1024, a=6.0):
    p = int(fs / freq)
    s = np.zeros(n)
    for i in range(0, n, p):
        d = np.exp(-(np.arange(min(p, n - i))) / 25.0)
        s[i : i + len(d)] += a * d
    return s + 0.3 * np.random.randn(n)


def _bearing_engine():
    prov = BearingProvider(bearing=_BEARING, cond_to_rpm={0: 1797}, fs=12000)
    return PhysicsRuleEngine(provider=prov, taxonomy=_TAXONOMY)


def test_physics_frequencies_match_published_cwru():
    cf = characteristic_frequencies(1797, bearing=_BEARING)
    assert abs(cf['BPFO'] - 107.4) < 1.0
    assert abs(cf['BPFI'] - 162.2) < 1.0


def test_symbolic_confirms_matching_fault():
    r = _bearing_engine().diagnose(
        _impulse(characteristic_frequencies(1797, bearing=_BEARING)['BPFI']), 5, 0
    )
    assert r['verdict'] == 'CONFIRMED'
    assert r['predicted_family'] == 'Inner Race'


def test_symbolic_conflicts_when_physics_disagrees():
    # inner-race signal but the classifier said outer race (class 8)
    r = _bearing_engine().diagnose(
        _impulse(characteristic_frequencies(1797, bearing=_BEARING)['BPFI']), 8, 0
    )
    assert r['verdict'] == 'CONFLICT'
    # root cause follows the physics, overriding the classifier
    assert r['root_cause']['fault_type'] == 'Inner Race'


def test_spectral_provider_runs_without_geometry():
    # the universal fallback: no machine model, still flags a dominant fault freq
    eng = PhysicsRuleEngine(provider=SpectralProvider(fs=12000))
    r = eng.diagnose(_impulse(160.0), 1, 0)
    assert r['verdict'] in ('CONFIRMED', 'INCONCLUSIVE')
    assert 'root_cause' in r


def test_get_provider_falls_back_to_spectral():
    assert get_provider('unknown_machine') is SpectralProvider
    assert get_provider('bearing') is BearingProvider


def test_causal_doZ_is_rung2():
    y = np.random.binomial(1, 0.6, 500)
    cond = np.random.choice([0, 1, 2, 3], 500)
    assert intervention_effect_of_condition(y, cond, n_perm=50)['rung'] == 2


def test_consensus_conflict_forces_review():
    assert fuse('CONFLICT', 0.99) == 'MANUAL_REVIEW'
    assert fuse('CONFIRMED', 0.99) == 'HIGH_CONFIDENCE'


# ── gear domain (cross-domain universality) ──────────────────────────────────


def test_gear_mesh_frequency_math():
    from cansyd.physics.gear import gear_mesh_frequencies

    gf = gear_mesh_frequencies(1800, 20)  # 20 teeth at 1800 rpm (30 Hz)
    assert abs(gf['GMF'] - 600.0) < 0.1
    assert abs(gf['shaft_input'] - 30.0) < 0.1


def test_gear_provider_implements_interface():
    from cansyd.physics.providers.base import PhysicsProvider
    from cansyd.physics.providers.gear import GearProvider

    p = GearProvider(n_teeth_input=20, cond_to_rpm={0: 1800}, fs=5120)
    assert isinstance(p, PhysicsProvider)
    sig = np.sin(2 * np.pi * 600 * np.arange(2048) / 5120) + 0.1 * np.random.randn(2048)
    ev = p.evidence(sig, 0)
    assert 'family_strength' in ev and 'frequencies_hz' in ev
    assert p.dominant_family(ev) in p.families


def test_gear_registered():
    from cansyd.physics.providers import available_domains, get_provider
    from cansyd.physics.providers.gear import GearProvider

    assert get_provider('gear') is GearProvider
    assert 'gear' in available_domains()

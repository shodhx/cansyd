"""Smoke tests for the CNSD layers (no TF / no DoWhy required)."""
import numpy as np
from cnsd.physics import characteristic_frequencies, CWRU_PHYSICS
from cnsd.symbolic import PhysicsRuleEngine
from cnsd.causal import intervention_effect_of_condition
from cnsd.consensus import fuse


def _impulse(freq, fs=12000, n=1024, a=6.0):
    p = int(fs / freq); s = np.zeros(n)
    for i in range(0, n, p):
        d = np.exp(-(np.arange(min(p, n - i))) / 25.0); s[i:i+len(d)] += a * d
    return s + 0.3 * np.random.randn(n)


def test_physics_frequencies_match_published_cwru():
    cf = characteristic_frequencies(1797)
    assert abs(cf['BPFO'] - 107.4) < 1.0
    assert abs(cf['BPFI'] - 162.2) < 1.0


def test_symbolic_confirms_matching_fault():
    eng = PhysicsRuleEngine(physics=CWRU_PHYSICS)
    r = eng.diagnose(_impulse(characteristic_frequencies(1797)['BPFI']), 5, 0)
    assert r['verdict'] in ('CONFIRMED', 'INCONCLUSIVE')
    assert 'root_cause' in r


def test_symbolic_conflicts_when_physics_disagrees():
    eng = PhysicsRuleEngine(physics=CWRU_PHYSICS)
    # inner-race signal but tell it the CNN said outer race (class 8)
    r = eng.diagnose(_impulse(characteristic_frequencies(1797)['BPFI']), 8, 0)
    assert r['verdict'] in ('CONFLICT', 'INCONCLUSIVE')


def test_causal_doZ_is_rung2():
    y = np.random.binomial(1, 0.6, 500); cond = np.random.choice([0, 1, 2, 3], 500)
    assert intervention_effect_of_condition(y, cond, n_perm=50)['rung'] == 2


def test_consensus_conflict_forces_review():
    assert fuse('CONFLICT', 0.99) == 'MANUAL_REVIEW'
    assert fuse('CONFIRMED', 0.99) == 'HIGH_CONFIDENCE'

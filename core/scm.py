"""
scm.py - The structural causal model behind CNSD's causal layer, stated honestly.

THE DAG (corrected). Earlier framings drew vibration -> fault, treating
vibration energy as a cause of the fault. That is physically backwards: a
bearing defect is the cause; vibration is a *measurement* of it. The honest DAG
has a latent bearing-health state H:

        H (latent bearing health)
       /          \
      v            v
   X (vibration   Y (fault label /
      measurement)   failure event)
      ^
      |
   Z (operating load / speed)   --modulates how H manifests in X

  - H is the latent common cause of both the measured vibration X and the
    operational fault label Y.
  - Z (operating condition) modulates how the latent fault manifests in the
    measured signal. Z is something we can physically intervene on (set the
    motor load), so it is a valid do-target.
  - There is NO direct causal arrow X -> Y. Vibration does not cause faults.

WHAT CNSD ESTIMATES (Rung 2, honestly). Because Z (operating condition) is
manipulable, the well-posed interventional question is:

    What is the causal effect of operating condition on the manifestation /
    detectability of the fault?

This is a real do-operation (do(Z = load)), needs no instrument, and avoids the
backwards-arrow problem entirely. CNSD estimates it by backdoor adjustment and
- importantly - tests whether the effect is INVARIANT across operating
conditions, which is the property a deployable diagnostic needs.

Rung 3 (counterfactual) is NOT claimed here. The counterfactual module is a
local sensitivity analysis under an explicit linear assumption (see
sensitivity.py), not a Pearl-type structural counterfactual.
"""

CNSD_DAG = {
    'nodes': {
        'H': 'latent bearing health state (unobserved)',
        'X': 'measured vibration signal / descriptor',
        'Y': 'fault label / failure event',
        'Z': 'operating condition (load / speed) - manipulable',
    },
    'edges': [
        ('H', 'X'),   # health manifests in vibration
        ('H', 'Y'),   # health determines the fault label
        ('Z', 'X'),   # operating condition modulates the measurement
    ],
    'no_edge': [
        ('X', 'Y'),   # vibration does NOT cause faults (the corrected arrow)
    ],
    'intervention_target': 'Z',
    'estimand': 'E[Y | do(Z)] contrasts across operating conditions, '
                'backdoor-adjusted; tested for invariance.',
    'rung': 2,
    'rung3_claimed': False,
}


def describe():
    """Human-readable statement of the SCM (for reports / paper)."""
    d = CNSD_DAG
    lines = ['CNSD structural causal model:']
    for n, desc in d['nodes'].items():
        lines.append(f'  {n}: {desc}')
    lines.append('  edges: ' + ', '.join(f'{a}->{b}' for a, b in d['edges']))
    lines.append('  explicitly NO edge: ' + ', '.join(f'{a}->{b}' for a, b in d['no_edge']))
    lines.append(f"  intervention target: do({d['intervention_target']})")
    lines.append(f"  estimand: {d['estimand']}")
    lines.append(f"  Pearl rung: {d['rung']} (Rung 3 claimed: {d['rung3_claimed']})")
    return '\n'.join(lines)

"""

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

"""

CANSYD_DAG = {
    'nodes': {
        'H': 'latent bearing health state (unobserved)',
        'X': 'measured vibration signal / descriptor',
        'Y': 'fault label / failure event',
        'Z': 'operating condition (load / speed) - manipulable',
    },
    'edges': [
        ('H', 'X'),  # health manifests in vibration
        ('H', 'Y'),  # health determines the fault label
        ('Z', 'X'),  # operating condition modulates the measurement
    ],
    'no_edge': [
        ('X', 'Y'),  # vibration does NOT cause faults (the corrected arrow)
    ],
    'intervention_target': 'Z',
    'estimand': 'E[Y | do(Z)] contrasts across operating conditions, '
    'backdoor-adjusted; tested for invariance.',
    'rung': 2,
    'rung3_claimed': False,
}


def describe():
    """Human-readable statement of the SCM (for reports / paper)."""
    d = CANSYD_DAG
    lines = ['CANSYD structural causal model:']
    for n, desc in d['nodes'].items():
        lines.append(f'  {n}: {desc}')
    lines.append('  edges: ' + ', '.join(f'{a}->{b}' for a, b in d['edges']))
    lines.append('  explicitly NO edge: ' + ', '.join(f'{a}->{b}' for a, b in d['no_edge']))
    lines.append(f'  intervention target: do({d["intervention_target"]})')
    lines.append(f'  estimand: {d["estimand"]}')
    lines.append(f'  Pearl rung: {d["rung"]} (Rung 3 claimed: {d["rung3_claimed"]})')
    return '\n'.join(lines)

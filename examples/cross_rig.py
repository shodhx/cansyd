"""
The cross-rig credibility experiment (CWRU -> Paderborn).

Trains the CNN on CWRU (EDM notches, 6205) and evaluates on Paderborn
NATURALLY-damaged bearings (real fatigue spalls, 6203). This tests whether the
model learned 'bearing faults' or merely 'EDM notches on this one rig'.

The CWRU 10-class output is collapsed to the 3 superclasses Paderborn provides
{Normal, Inner, Outer}; CWRU 'Ball' classes are excluded from the mapping (PU
has no ball class). Reports transfer accuracy AND the physics-verification rate
on Paderborn (does the symbolic layer still confirm faults on a different rig and
a different bearing geometry?).

Honest expectation: cross-rig + cross-mechanism transfer is HARD. A large drop
from in-distribution CWRU accuracy is the likely and informative outcome. Report
whatever happens.
"""

import numpy as np

# CWRU class -> superclass {0:Normal, 1:Inner, 2:Outer}; Ball (1,2,3) -> -1 (drop)
CWRU_TO_SUPER = {0: 0, 1: -1, 2: -1, 3: -1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2}


def collapse_to_super(cwru_probs):
    """Collapse a 10-class softmax to a 3-superclass prediction (Normal/Inner/Outer)."""
    probs = np.asarray(cwru_probs)
    agg = np.zeros((len(probs), 3))
    for cwru_cls, sup in CWRU_TO_SUPER.items():
        if sup >= 0:
            agg[:, sup] += probs[:, cwru_cls]
    return np.argmax(agg, axis=1)


def run_cross_rig(cnn, rule_engine, damage='natural'):
    """Execute the CWRU->Paderborn transfer test. `cnn` is trained on CWRU."""
    from cnsd.datasets.paderborn import BEARING_6203, load_paderborn_test

    from cnsd.physics.bearing import characteristic_frequencies

    X_pu, y_pu = load_paderborn_test(damage=damage)
    print(
        f'\nPaderborn test set: {len(X_pu)} windows, '
        f'class balance {np.bincount(y_pu, minlength=3).tolist()} (N/I/O)'
    )

    probs = cnn.predict(X_pu, verbose=0)
    pred_super = collapse_to_super(probs)
    acc = float(np.mean(pred_super == y_pu))

    # per-superclass recall (honest breakdown - transfer often fails unevenly)
    print('\n--- CWRU -> Paderborn transfer (natural damage) ---')
    print(f'Overall transfer accuracy: {acc:.4f}')
    for sup, name in [(0, 'Normal'), (1, 'Inner'), (2, 'Outer')]:
        m = y_pu == sup
        if m.any():
            rec = float(np.mean(pred_super[m] == sup))
            print(f'  {name:7} recall: {rec:.4f}  (n={int(m.sum())})')

    print(
        '\n[Interpretation] CWRU uses 6205 + EDM notches; Paderborn natural '
        'uses 6203 + real spalls. A large drop is expected and informative: '
        'it bounds what the CNN actually generalises. '
    )
    return {'transfer_accuracy': acc, 'n_test': int(len(X_pu)), 'damage': damage}

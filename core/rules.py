"""
rules.py - Physics-grounded symbolic verification layer.

This layer does NOT trust the neural network's class label on its own. For each
prediction it INDEPENDENTLY computes the bearing characteristic frequencies from
geometry + shaft speed, measures their prominence in the signal's envelope
spectrum, and checks whether the physical evidence supports the predicted fault
type. The output is an auditable verdict:

  CONFIRMED   - CNN prediction agrees with the dominant fault frequency
  CONFLICT    - CNN prediction disagrees with the physical evidence
  INCONCLUSIVE- no fault frequency is sufficiently prominent (e.g. weak/normal)

This turns a black-box class label into a diagnosis a maintenance engineer can
audit against physics, and provides an independent conflict signal that the
consensus layer can act on. It replaces the previous lookup-table 'rule engine',
which only checked the CNN's feature-norm against itself.
"""
import numpy as np
from core.envelope import (
    fault_frequency_evidence, characteristic_frequencies,
    CWRU_LOAD_RPM, CWRU_FS,
)

# CWRU 10-class taxonomy -> (fault family, defect size). Family drives the
# physical frequency check; size drives severity/action.
CWRU_CLASSES = {
    0: ('Normal',     None),
    1: ('Ball',       0.007), 2: ('Ball',       0.014), 3: ('Ball',       0.021),
    4: ('Inner Race', 0.007), 5: ('Inner Race', 0.014), 6: ('Inner Race', 0.021),
    7: ('Outer Race', 0.007), 8: ('Outer Race', 0.014), 9: ('Outer Race', 0.021),
}

# which envelope-evidence key corresponds to each fault family
FAMILY_TO_FREQ = {'Outer Race': 'BPFO', 'Inner Race': 'BPFI', 'Ball': 'BSF'}

# prominence above this counts as a genuine characteristic-frequency peak
PROMINENCE_THRESHOLD = 3.0

SEVERITY = {0.007: 'Low', 0.014: 'Medium', 0.021: 'High'}
ACTIONS = {
    ('Normal', None):      'Continue monitoring; inspect in 30 days.',
    ('Ball', 'Low'):       'Schedule inspection within 14 days.',
    ('Ball', 'Medium'):    'Reduce load; inspect within 7 days.',
    ('Ball', 'High'):      'Replace within 48h.',
    ('Inner Race', 'Low'): 'Lubricate; re-inspect in 10 days.',
    ('Inner Race', 'Medium'): 'Check shaft alignment; inspect within 5 days.',
    ('Inner Race', 'High'):'Immediate shutdown; full replacement.',
    ('Outer Race', 'Low'): 'Increase monitoring; inspect in 10 days.',
    ('Outer Race', 'Medium'):'Reduce speed; inspect within 5 days.',
    ('Outer Race', 'High'):'Immediate shutdown; replace outer-race assembly.',
}


class PhysicsRuleEngine:
    """Independent physics verification of a neural fault prediction."""

    def __init__(self, prominence_threshold=PROMINENCE_THRESHOLD,
                 load_rpm=None, fs=CWRU_FS):
        self.tau = prominence_threshold
        self.load_rpm = load_rpm or CWRU_LOAD_RPM
        self.fs = fs

    def diagnose(self, signal, cnn_class, load):
        """Produce an auditable, physics-checked diagnosis for one window.

        signal    : raw 1D vibration window
        cnn_class : the CNN's predicted class (0..9)
        load      : operating load (maps to shaft rpm)
        """
        rpm = self.load_rpm.get(int(load), 1797)
        evidence = fault_frequency_evidence(signal, rpm, fs=self.fs)
        family, size = CWRU_CLASSES.get(int(cnn_class), ('Unknown', None))

        # which fault family does the PHYSICS most support?
        fam_evidence = {'Outer Race': evidence['BPFO'],
                        'Inner Race': evidence['BPFI'],
                        'Ball': evidence['BSF']}
        phys_family = max(fam_evidence, key=fam_evidence.get)
        phys_strength = fam_evidence[phys_family]

        # verdict logic
        if family == 'Normal':
            # normal is supported when NO fault frequency is prominent
            if phys_strength < self.tau:
                verdict = 'CONFIRMED'
            else:
                verdict = 'CONFLICT'   # CNN says normal but a fault freq is present
        elif phys_strength < self.tau:
            verdict = 'INCONCLUSIVE'   # CNN names a fault but physics is weak
        elif phys_family == family:
            verdict = 'CONFIRMED'      # physics agrees with the CNN's fault family
        else:
            verdict = 'CONFLICT'       # physics points to a different fault family

        severity = SEVERITY.get(size, 'None') if size else 'None'
        action = ACTIONS.get((family, severity if size else None),
                             'Review manually.')
        return {
            'cnn_class': int(cnn_class),
            'cnn_family': family,
            'defect_size_in': size,
            'severity': severity,
            'physics_family': phys_family,
            'physics_strength': float(phys_strength),
            'characteristic_freqs_hz': evidence['freqs_hz'],
            'band_evidence': {k: float(evidence[k]) for k in ('BPFO', 'BPFI', 'BSF')},
            'verdict': verdict,
            'action': action,
            'explanation': self._explain(family, phys_family, phys_strength,
                                         verdict, evidence['freqs_hz']),
        }

    def _explain(self, cnn_family, phys_family, strength, verdict, freqs):
        if verdict == 'CONFIRMED' and cnn_family == 'Normal':
            return ("No characteristic fault frequency is prominent in the "
                    "envelope spectrum; a Normal classification is physically "
                    "consistent.")
        if verdict == 'CONFIRMED':
            f = freqs.get(FAMILY_TO_FREQ.get(cnn_family, ''), 0.0)
            return (f"Envelope spectrum shows a prominent peak near "
                    f"{f:.1f} Hz (the {FAMILY_TO_FREQ.get(cnn_family)} for this "
                    f"speed), confirming a {cnn_family} fault.")
        if verdict == 'CONFLICT' and cnn_family == 'Normal':
            return (f"CNN predicts Normal, but a {phys_family} characteristic "
                    f"frequency is prominent (strength {strength:.1f}). "
                    f"Flag for review.")
        if verdict == 'CONFLICT':
            return (f"CNN predicts {cnn_family}, but the envelope spectrum is "
                    f"dominated by the {phys_family} characteristic frequency "
                    f"(strength {strength:.1f}). Physics and network disagree; "
                    f"flag for review.")
        return (f"CNN predicts {cnn_family}, but no characteristic fault "
                f"frequency is sufficiently prominent (max strength "
                f"{strength:.1f} < {self.tau}); evidence inconclusive.")


# module-level instance for the pipeline
rule_engine = PhysicsRuleEngine()

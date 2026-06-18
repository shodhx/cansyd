"""
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
from cnsd.physics.bearing import (
    fault_frequency_evidence, characteristic_frequencies,
    envelope_spectrum, band_energy, CWRU_LOAD_RPM, CWRU_FS, BEARING_6205,
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



# Physical root cause per fault family: the mechanism and the characteristic
# frequency whose presence in the envelope spectrum is the EVIDENCE for it.
ROOT_CAUSE = {
    'Outer Race': {
        'mechanism': 'a defect (spall/pit) on the stationary outer race; each '
                     'rolling element striking it produces an impulse',
        'evidence_freq': 'BPFO',
        'component': 'outer race',
    },
    'Inner Race': {
        'mechanism': 'a defect on the rotating inner race, amplitude-modulated '
                     'by shaft rotation as the defect enters/exits the load zone',
        'evidence_freq': 'BPFI',
        'component': 'inner race',
    },
    'Ball': {
        'mechanism': 'a defect on a rolling element, striking both races as it '
                     'spins',
        'evidence_freq': 'BSF',
        'component': 'rolling element',
    },
    'Normal': {
        'mechanism': 'no localized defect; vibration is broadband bearing/'
                     'structural noise with no dominant characteristic frequency',
        'evidence_freq': None,
        'component': None,
    },
}

class PhysicsRuleEngine:
    """Independent physics verification of a neural fault prediction."""

    def __init__(self, prominence_threshold=PROMINENCE_THRESHOLD, physics=None,
                 load_rpm=None, fs=CWRU_FS):
        """physics: an optional PhysicsConfig (bearing geometry + cond->rpm map +
        fs). When provided, the engine verifies against that dataset's
        characteristic frequencies. When None, it falls back to geometry-free
        verification (impulsiveness/energy at the dominant envelope peak), so the
        engine still runs on ANY vibration dataset - it just cannot name the
        specific fault frequency without geometry."""
        self.tau = prominence_threshold
        if physics is not None:
            self.bearing = physics.bearing
            self.load_rpm = physics.cond_to_rpm
            self.fs = physics.fs
            self.geometry_free = False
        else:
            self.bearing = BEARING_6205
            self.load_rpm = load_rpm or CWRU_LOAD_RPM
            self.fs = fs
            self.geometry_free = physics is None and load_rpm is None

    def diagnose(self, signal, cnn_class, load):
        """Produce an auditable, physics-checked diagnosis for one window.

        signal    : raw 1D vibration window
        cnn_class : the CNN's predicted class (0..9)
        load      : operating load (maps to shaft rpm)
        """
        rpm = self.load_rpm.get(int(load), next(iter(self.load_rpm.values())))
        evidence = fault_frequency_evidence(signal, rpm, fs=self.fs, bearing=self.bearing)
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

        # ── ROOT CAUSE (the headline CNSD output) ───────────────────────────
        # The cause is named from the PHYSICAL EVIDENCE, not the CNN label: the
        # dominant characteristic frequency identifies the defective component.
        # When physics is inconclusive we report the cause as unconfirmed.
        if verdict == 'CONFIRMED' and family == 'Normal':
            cause_family = 'Normal'
            cause_confidence = 'physically confirmed'
        elif phys_strength >= self.tau:
            cause_family = phys_family          # physics names the cause
            cause_confidence = ('physically confirmed' if verdict == 'CONFIRMED'
                                else 'physics-indicated (conflicts with network)')
        else:
            cause_family = family if family != 'Unknown' else None
            cause_confidence = 'unconfirmed (weak physical evidence)'

        rc = ROOT_CAUSE.get(cause_family, {})
        ev_freq_name = rc.get('evidence_freq')
        ev_freq_hz = evidence['freqs_hz'].get(ev_freq_name) if ev_freq_name else None
        root_cause = {
            'component': rc.get('component'),
            'fault_type': cause_family,
            'mechanism': rc.get('mechanism'),
            'evidence_frequency': ev_freq_name,
            'evidence_frequency_hz': ev_freq_hz,
            'evidence_strength': float(phys_strength),
            'confidence': cause_confidence,
            'statement': self._root_cause_statement(cause_family, ev_freq_name,
                                                    ev_freq_hz, phys_strength,
                                                    cause_confidence, rc),
        }
        return {
            'root_cause': root_cause,
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

    def _root_cause_statement(self, family, freq_name, freq_hz, strength,
                              confidence, rc):
        """One-line, human-readable root cause - the diagnosis an engineer wants."""
        if family == 'Normal':
            return ("ROOT CAUSE: none - no localized bearing defect detected; "
                    "vibration is consistent with a healthy bearing.")
        if family is None:
            return ("ROOT CAUSE: undetermined - the network named a fault but no "
                    "characteristic frequency is prominent enough to localize it.")
        comp = rc.get('component', family)
        if freq_hz is not None:
            return (f"ROOT CAUSE: defect on the {comp} ({family}), evidenced by a "
                    f"prominent peak at the {freq_name} ({freq_hz:.1f} Hz, "
                    f"strength {strength:.1f}). Mechanism: {rc.get('mechanism','')}. "
                    f"[{confidence}]")
        return (f"ROOT CAUSE: {family} defect on the {comp} [{confidence}].")

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

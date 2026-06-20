"""Physics verification of a neural fault prediction.

The engine is domain-agnostic: it delegates all physical reasoning to a
PhysicsProvider (bearing, spectral, or any registered domain) and a taxonomy
that maps classifier labels to fault families. It returns a verdict
(CONFIRMED / CONFLICT / INCONCLUSIVE) and a root cause from the provider.
"""
PROMINENCE_THRESHOLD = 3.0
_ACTION = {'Low': 'Schedule inspection.', 'Medium': 'Reduce load and inspect.',
           'High': 'Immediate shutdown and replacement.', 'None': 'Continue monitoring.'}


class PhysicsRuleEngine:
    """Independent physics verification via a pluggable provider.

    provider : a PhysicsProvider (computes evidence + root cause for its domain)
    taxonomy : {int_label: (family, severity)} mapping classifier outputs to
               fault families the provider understands. Optional; without it the
               classifier label is used only for agreement/disagreement.
    """

    def __init__(self, provider, taxonomy=None, prominence_threshold=PROMINENCE_THRESHOLD):
        if provider is None:
            raise ValueError("PhysicsRuleEngine requires a PhysicsProvider.")
        self.provider = provider
        self.taxonomy = taxonomy or {}
        self.tau = prominence_threshold

    def diagnose(self, signal, predicted_class, condition):
        evidence = self.provider.evidence(signal, condition)
        phys_family = self.provider.dominant_family(evidence)
        phys_strength = evidence.get('family_strength', {}).get(phys_family, 0.0)

        family, severity = self.taxonomy.get(int(predicted_class), (None, 'None'))
        verdict = self._verdict(family, phys_family, phys_strength)
        cause_family = self._cause_family(family, phys_family, phys_strength, verdict)
        root_cause = self.provider.root_cause(cause_family, evidence)

        return {
            'predicted_class': int(predicted_class),
            'predicted_family': family,
            'severity': severity,
            'physics_family': phys_family,
            'physics_strength': float(phys_strength),
            'characteristic_freqs_hz': evidence.get('frequencies_hz', {}),
            'verdict': verdict,
            'action': _ACTION.get(severity, 'Review manually.'),
            'root_cause': root_cause,
            'explanation': root_cause.get('statement', ''),
        }

    def _verdict(self, family, phys_family, strength):
        normal_family = family in (None, 'Normal') and family == 'Normal'
        if family == 'Normal':
            return 'CONFIRMED' if strength < self.tau else 'CONFLICT'
        if strength < self.tau:
            return 'INCONCLUSIVE'
        if family is None:
            return 'CONFIRMED'
        return 'CONFIRMED' if phys_family == family else 'CONFLICT'

    def _cause_family(self, family, phys_family, strength, verdict):
        if verdict == 'CONFIRMED' and family == 'Normal':
            return 'Normal'
        if strength >= self.tau:
            return phys_family
        return family

class BearingRule:
    def __init__(self, label, diagnosis, root_cause, severity, action):
        self.label = label
        self.diagnosis = diagnosis
        self.root_cause = root_cause
        self.severity = severity
        self.action = action

class BearingDiagnosisEngine:
    def __init__(self):
        self.rules = {
            0: BearingRule(0, 'NORMAL', 'No fault. Baseline vibration.', 'NONE', 'Monitor. Inspect in 30 days.'),
            1: BearingRule(1, 'BALL_FAULT', 'Ball defect 0.007in. Cyclic impact at BPF.', 'LOW', 'Inspect within 14 days.'),
            2: BearingRule(2, 'BALL_FAULT', 'Ball defect 0.014in. Rolling element stress.', 'MEDIUM', 'Reduce load 20%. Inspect 7 days.'),
            3: BearingRule(3, 'BALL_FAULT', 'Ball defect 0.021in. Spalling imminent.', 'HIGH', 'IMMEDIATE shutdown. Replace 48h.'),
            4: BearingRule(4, 'INNER_RACE', 'IR defect 0.007in. BPFI harmonics.', 'LOW', 'Lubricate. Re-inspect 10 days.'),
            5: BearingRule(5, 'INNER_RACE', 'IR defect 0.014in. Misalignment risk.', 'MEDIUM', 'Check alignment. Inspect 5 days.'),
            6: BearingRule(6, 'INNER_RACE', 'IR defect 0.021in. Structural compromise.', 'HIGH', 'IMMEDIATE shutdown. Replace.'),
            7: BearingRule(7, 'OUTER_RACE', 'OR defect 0.007in. BPFO sidebands.', 'LOW', 'Increase monitoring. Inspect 10 days.'),
            8: BearingRule(8, 'OUTER_RACE', 'OR defect 0.014in. Load asymmetry.', 'MEDIUM', 'Reduce speed 15%. Inspect 5 days.'),
            9: BearingRule(9, 'OUTER_RACE', 'OR defect 0.021in. Catastrophic risk.', 'HIGH', 'IMMEDIATE shutdown. Replace assembly.')
        }

    def evaluate(self, label):
        r = self.rules.get(label)
        if not r:
            return {'severity': 'UNKNOWN', 'action': 'Manual inspect'}
        return {'diagnosis': r.diagnosis, 'severity': r.severity, 'action': r.action}
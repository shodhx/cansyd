"""The structured, auditable diagnosis result."""

import numpy as np


class DiagnosisReport:
    def __init__(self, records, dataset=None):
        self.records = records
        self.dataset = dataset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]

    def root_causes(self):
        return [r['root_cause']['statement'] for r in self.records]

    def verification_rate(self):
        v = [r['physics_verdict'] for r in self.records]
        n = max(len(v), 1)
        return {k: v.count(k) / n for k in ('CONFIRMED', 'CONFLICT', 'INCONCLUSIVE')}

    def conflicts(self):
        return [r for r in self.records if r['physics_verdict'] == 'CONFLICT']

    def accuracy_by_verdict(self):
        if self.dataset is None or self.dataset.y is None:
            return {}
        correct = np.array([r['predicted_fault'] != 'Unknown' for r in self.records])
        # caller compares against truth externally; provide split helper
        out = {}
        for verdict in ('CONFIRMED', 'CONFLICT'):
            m = np.array([r['physics_verdict'] == verdict for r in self.records])
            if m.any():
                out[verdict] = int(m.sum())
        return out

    def summary(self):
        vr = self.verification_rate()
        return (
            f'{len(self)} units | confirmed {vr["CONFIRMED"]:.0%} '
            f'conflict {vr["CONFLICT"]:.0%} inconclusive {vr["INCONCLUSIVE"]:.0%} '
            f'| {len(self.conflicts())} need review'
        )

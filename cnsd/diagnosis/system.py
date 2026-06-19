"""
CNSD's front door.

CNSD's whole five-layer pipeline is behind
one object:

    from cnsd import CNSD, load_dataset
    data   = load_dataset('cwru')
    model  = CNSD().fit(data)
    report = model.diagnose(data)

Layers: perception (1) -> symbolic verification + root cause (2) -> causal
Rung-2 (3) -> counterfactual Rung-3 (3B) -> consensus (4).
"""
import numpy as np

from cnsd.datasets import Dataset
from cnsd.perception import build_cnn, train_jepa_backbone
from cnsd.symbolic import PhysicsRuleEngine
from cnsd.causal import intervention_effect_of_condition, analyze_causal, signal_kurtosis
from cnsd.counterfactual import build_scm, counterfactual_for_unit, what_if
from cnsd.consensus import fuse
from cnsd.diagnosis.report import DiagnosisReport


class CNSD:
    """Five-layer Causal Neuro-Symbolic Diagnosis system."""

    def __init__(self, conf_thresh=0.90, prominence_threshold=3.0, config=None):
        self.conf_thresh = conf_thresh
        self.prominence_threshold = prominence_threshold
        self.config = config
        self.cnn = None
        self.symbolic = None
        self.scm = None
        self._fitted = False

    def fit(self, data: Dataset, epochs=30):
        import tensorflow as tf
        from cnsd.perception.cnn import _train_cnn
        nc = int(data.y.max()) + 1
        self.cnn = _train_cnn(data.X, data.y, num_classes=nc, epochs=epochs)
        self.symbolic = PhysicsRuleEngine(physics=data.physics,
                                          prominence_threshold=self.prominence_threshold)
        # fit the Rung-3 SCM (graceful None if DoWhy absent)
        feat = signal_kurtosis(data.X)
        self.scm = build_scm(data.cond, feat, data.y)
        self._fitted = True
        return self

    def diagnose(self, data: Dataset) -> DiagnosisReport:
        if not self._fitted:
            raise RuntimeError('call .fit(data) before .diagnose(data)')
        probs = self.cnn.predict(data.X, batch_size=64, verbose=0)
        cnn_class, cnn_conf = probs.argmax(1), probs.max(1)
        records = []
        for i in range(len(data.X)):
            diag = self.symbolic.diagnose(data.X[i].flatten(), cnn_class[i], data.cond[i])
            status = fuse(diag['verdict'], cnn_conf[i], self.conf_thresh)
            records.append({
                'root_cause': diag['root_cause'],
                'predicted_fault': diag['cnn_family'],
                'severity': diag['severity'],
                'cnn_confidence': float(cnn_conf[i]),
                'physics_verdict': diag['verdict'],
                'status': status,
                'action': diag['action'],
                'explanation': diag['explanation'],
            })
        return DiagnosisReport(records, data)

    def condition_effect(self, data: Dataset):
        """Rung-2 do(Z) interventional effect of operating condition."""
        return intervention_effect_of_condition(data.y, data.cond)

    def what_if(self, data: Dataset, unit_index, condition_cf):
        """Rung-3 counterfactual for one unit (sensitivity fallback w/o DoWhy)."""
        # Future refactor: process the full condition_cf dictionary for multiple interventions.
        # Currently, the core SCM supports a single Z variable, so we extract the first value.
        if isinstance(condition_cf, dict):
            cf_val = list(condition_cf.values())[0] if condition_cf else 0.0
        else:
            cf_val = condition_cf
            
        feat = signal_kurtosis(data.X[unit_index:unit_index+1])[0]
        return what_if(feat, data.cond[unit_index], cf_val,
                       scm=self.scm, X_sample=data.X[unit_index].flatten(),
                       factual_y=(data.y[unit_index] > 0))

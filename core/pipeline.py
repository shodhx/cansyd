"""
pipeline.py - CNSD diagnostic pipeline.

Per sample it produces an auditable diagnosis that combines the neural
prediction with an INDEPENDENT physics check (the symbolic layer). The headline
per-sample output is no longer a relabelled CNN guess: it is a verdict
(CONFIRMED / CONFLICT / INCONCLUSIVE) grounded in the bearing's characteristic
fault frequencies, plus a severity and a maintenance action.
"""
import numpy as np


class CNSDPipeline:
    def __init__(self, cnn_model, jepa_probe, encoder_model, patchify_fn,
                 rule_engine, conf_thresh=0.90):
        self.cnn = cnn_model
        self.jepa = jepa_probe
        self.encoder = encoder_model
        self.patchify = patchify_fn
        self.rules = rule_engine
        self.thresh = conf_thresh

    def predict(self, X, loads):
        """Diagnose a batch. `loads` gives each sample's operating load (for the
        physics frequency computation). Returns one auditable record per sample.
        """
        import tensorflow as tf
        cnn_pred = self.cnn.predict(X, verbose=0)
        cnn_conf = np.max(cnn_pred, axis=1)
        cnn_class = np.argmax(cnn_pred, axis=1)
        loads = np.asarray(loads)

        results = []
        n_conflict = 0
        for i in range(len(X)):
            sig = X[i].flatten()
            diag = self.rules.diagnose(sig, cnn_class[i], loads[i])

            # consensus status combines CNN confidence with the physics verdict
            if diag['verdict'] == 'CONFLICT':
                status = 'MANUAL_REVIEW'      # physics disagrees -> never auto-trust
                n_conflict += 1
            elif diag['verdict'] == 'CONFIRMED' and cnn_conf[i] >= self.thresh:
                status = 'HIGH_CONFIDENCE'    # network confident AND physics agrees
            elif diag['verdict'] == 'CONFIRMED':
                status = 'RELIABLE'           # physics agrees, network less sure
            else:
                status = 'UNCERTAIN'          # inconclusive physical evidence

            results.append({
                'cnn_class': diag['cnn_class'],
                'cnn_confidence': float(cnn_conf[i]),
                'diagnosis': diag['cnn_family'],
                'severity': diag['severity'],
                'physics_verdict': diag['verdict'],
                'physics_family': diag['physics_family'],
                'physics_strength': diag['physics_strength'],
                'characteristic_freqs_hz': diag['characteristic_freqs_hz'],
                'action': diag['action'],
                'status': status,
                'explanation': diag['explanation'],
            })

        self.last_conflict_rate = n_conflict / max(len(X), 1)
        return results

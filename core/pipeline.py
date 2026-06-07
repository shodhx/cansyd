# Save as core/pipeline.py
import tensorflow as tf
import numpy as np

class CNSDPipeline:
    def __init__(self, cnn_model, jepa_probe, encoder_model, patchify_fn, rule_engine, conf_thresh=0.95, agreement_weight=0.3):
        self.cnn = cnn_model
        self.jepa = jepa_probe
        self.encoder = encoder_model
        self.patchify = patchify_fn
        self.rules = rule_engine
        self.thresh = conf_thresh
        self.weight = agreement_weight
        self.suspicion = False # Active state feedback variable from your notebook
        
        # Prevent graph tracing compilation bottlenecks by building the extractor once
        self.feat_model = tf.keras.Model(self.cnn.layers[0].input, self.cnn.layers[-3].output)
        
    def predict(self, X):
        cnn_pred = self.cnn.predict(X, verbose=0)
        cnn_conf = np.max(cnn_pred, axis=1)
        cnn_class = np.argmax(cnn_pred, axis=1)
        
        p = self.patchify(X)
        num_patches = p.shape[1]
        
        jepa_emb = []
        for j in range(num_patches):
            jepa_emb.append(self.encoder(p[:, j], training=False).numpy())
        jepa_emb = np.mean(jepa_emb, axis=0)
        
        jepa_pred = self.jepa.predict_proba(jepa_emb)
        jepa_class = np.argmax(jepa_pred, axis=1)
        
        agreement = (cnn_class == jepa_class).astype(float)
        consensus = cnn_conf * (1 + self.weight * agreement)
        
        feats = self.feat_model.predict(X, verbose=0)
        feat_norms = np.linalg.norm(feats, axis=1)
        
        results = []
        for i in range(len(X)):
            # Adaptive thresholding logic pulled directly from your notebook's backward path
            current_thresh = self.thresh + 0.049 if self.suspicion else self.thresh
            
            # Evaluate hard physical constraints
            is_symbolically_valid = self.rules.evaluate(cnn_class[i], cnn_conf[i], feat_norms[i])
            
            # If a major contradiction is encountered, alter the global network suspicion state
            if not is_symbolically_valid:
                self.suspicion = True
                final_class = jepa_class[i] # Force fallback to self-supervised foundational probe
            else:
                if cnn_conf[i] < current_thresh and agreement[i] == 0:
                    final_class = jepa_class[i]
                else:
                    final_class = cnn_class[i]
                    
            results.append({
                'class': int(final_class),
                'confidence': float(consensus[i]),
                'symbolic_validity': is_symbolically_valid,
                'causal_risk': float(feat_norms[i]),
                'suspicion_triggered': self.suspicion
            })
            
        return results
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
        
        # Instantiate feature extraction sub-graph ONCE during initialization
        self.feat_model = tf.keras.Model(self.cnn.layers[0].input, self.cnn.layers[-3].output)
    
    def predict(self, X):
        cnn_pred = self.cnn.predict(X, verbose=0)
        cnn_conf = np.max(cnn_pred, axis=1)
        cnn_class = np.argmax(cnn_pred, axis=1)
        
        # Context-aware patch execution
        p = self.patchify(X)
        num_patches = p.shape[1] # Dynamically handle slice dimensions
        
        jepa_emb = []
        for j in range(num_patches):
            jepa_emb.append(self.encoder(p[:, j], training=False).numpy())
        jepa_emb = np.mean(jepa_emb, axis=0)
        
        jepa_pred = self.jepa.predict_proba(jepa_emb)
        jepa_class = np.argmax(jepa_pred, axis=1)
        
        agreement = (cnn_class == jepa_class).astype(float)
        consensus = cnn_conf * (1 + self.weight * agreement)
        
        # Zero tracing overhead for feature norms
        feats = self.feat_model.predict(X, verbose=0)
        feat_norms = np.linalg.norm(feats, axis=1)
        
        results = []
        for i in range(len(X)):
            # Evaluate symbolic rule validation (returns True/False or a weight modifier)
            is_symbolically_valid = self.rules.evaluate(cnn_class[i], cnn_conf[i], feat_norms[i])
            
            # True Neuro-Symbolic fallback routing logic
            if (cnn_conf[i] < self.thresh and agreement[i] == 0) or not is_symbolically_valid:
                final_class = jepa_class[i]
            else:
                final_class = cnn_class[i]
            
            results.append({
                'class': int(final_class),
                'confidence': float(consensus[i]),
                'symbolic_validity': is_symbolically_valid,
                'causal_risk': float(feat_norms[i])
            })
        
        return results

# Example invocation adjustment:
# pipeline = CNSDPipeline(model, probe, encoder, patchify, rule_engine)
# outputs = pipeline.predict(X_test_all[:10])
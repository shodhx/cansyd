import tensorflow as tf
import numpy as np

class CNSDPipeline:
    def __init__(self, cnn_model, jepa_probe, encoder_model, patchify_fn, rule_engine, conf_thresh=0.95, agreement_weight=0.3):
        """
        Neuro-Symbolic Pipeline coordinating joint connectionist networks,
        self-supervised patch encoders, and symbolic physical veto guardrails.
        """
        self.cnn = cnn_model
        self.jepa = jepa_probe
        self.encoder = encoder_model
        self.patchify = patchify_fn
        self.rules = rule_engine
        self.thresh = conf_thresh
        self.weight = agreement_weight
        self.suspicion = False  # Active suspicion flag for adaptive thresholding
        
        # Sub-graph tracking instantiation to prevent compilation graph leaks
        self.feat_model = tf.keras.Model(inputs=self.cnn.input, outputs=self.cnn.layers[-3].output)
        
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
            # Raise the confidence threshold once suspicion has been triggered
            current_thresh = self.thresh + 0.05 if self.suspicion else self.thresh
            
            # Call our newly unified boolean validator gate
            is_symbolically_valid = self.rules.evaluate(cnn_class[i], cnn_conf[i], feat_norms[i])
            
            # True neuro-symbolic routing and veto enforcement
            if not is_symbolically_valid:
                self.suspicion = True
                final_class = jepa_class[i]  # Hard fallback to self-supervised probe
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
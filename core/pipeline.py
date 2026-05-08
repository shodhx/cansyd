import tensorflow as tf
import numpy as np

class CNSDPipeline:
    def __init__(self, cnn_model, jepa_probe, rule_engine, conf_thresh=0.95, agreement_weight=0.3):
        self.cnn = cnn_model
        self.jepa = jepa_probe
        self.rules = rule_engine
        self.thresh = conf_thresh
        self.weight = agreement_weight
    
    def predict(self, X):
        cnn_pred = self.cnn.predict(X, verbose=0)
        cnn_conf = np.max(cnn_pred, axis=1)
        cnn_class = np.argmax(cnn_pred, axis=1)
        
        p = patchify(X)
        jepa_emb = []
        for j in range(8):
            jepa_emb.append(encoder(p[:, j], training=False).numpy())
        jepa_emb = np.mean(jepa_emb, axis=0)
        jepa_pred = self.jepa.predict_proba(jepa_emb)
        jepa_class = np.argmax(jepa_pred, axis=1)
        
        agreement = (cnn_class == jepa_class).astype(float)
        consensus = cnn_conf * (1 + self.weight * agreement)
        
        feat_model = tf.keras.Model(self.cnn.layers[0].input, self.cnn.layers[-3].output)
        feats = feat_model.predict(X, verbose=0)
        feat_norms = np.linalg.norm(feats, axis=1)
        
        results = []
        for i in range(len(X)):
            symbolic = self.rules.evaluate(cnn_class[i], cnn_conf[i], feat_norms[i])
            
            if cnn_conf[i] < self.thresh and agreement[i] == 0:
                final_class = jepa_class[i]
            else:
                final_class = cnn_class[i]
            
            results.append({
                'class': final_class,
                'confidence': consensus[i],
                'symbolic': symbolic,
                'causal_risk': feat_norms[i]
            })
        
        return results

pipeline = CNSDPipeline(model, probe, rule_engine)
outputs = pipeline.predict(X_test_all[:10])
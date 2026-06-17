import tensorflow as tf
import numpy as np

class LoRAAdapter(tf.keras.layers.Layer):
    def __init__(self, dim, rank=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.rank = rank
    
    def build(self, input_shape):
        self.A = self.add_weight(name='A', shape=(self.dim, self.rank), initializer='glorot_uniform', trainable=True)
        self.B = self.add_weight(name='B', shape=(self.rank, self.dim), initializer='zeros', trainable=True)
        super().build(input_shape)
    
    def call(self, x):
        return x + tf.matmul(x, tf.matmul(self.A, self.B))

def train_ccr_lora(base_model, X_new, y_new, ate_old, lam=1.0):
    """
    Adapts late-stage layer weights via Low-Rank Adaptation matrices
    constrained by a soft Causal Consistency Regularization (CCR) penalty variable.
    """
    # Freeze the base CNN layers to lock feature representations
    for layer in base_model.layers[:-1]:
        layer.trainable = False

    inp = tf.keras.Input(shape=base_model.input_shape[1:])
    x = base_model.layers[0](inp)
    for layer in base_model.layers[1:-3]:
        x = layer(x)
        
    feat = base_model.layers[-3](x)
    lora_feat = LoRAAdapter(dim=feat.shape[-1], rank=4)(feat)
    drop = base_model.layers[-2](lora_feat)
    out = tf.keras.layers.Dense(len(np.unique(y_new)) + 1, activation='softmax')(drop)
    
    ccr_model = tf.keras.Model(inp, out)
    optimizer = tf.keras.optimizers.Adam(0.001)
    
    # Isolate feature tracking subgraph
    feat_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
    
    # Parameterized optimization loop
    for epoch in range(10):
        for idx in range(0, len(X_new), 64):
            bx = X_new[idx:idx+64]
            by = y_new[idx:idx+64]
            
            with tf.GradientTape() as tape:
                pred = ccr_model(bx, training=True)
                ce_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(by, pred))
                
                # Feature-norm consistency penalty against the frozen base ATE
                feats_new = feat_extractor(bx, training=False)
                norms_new = tf.reduce_mean(tf.norm(feats_new, axis=1))
                causal_penalty = tf.square(norms_new - ate_old)
                
                total_loss = ce_loss + lam * causal_penalty
                
            grads = tape.gradient(total_loss, ccr_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, ccr_model.trainable_weights))
            
    return ccr_model
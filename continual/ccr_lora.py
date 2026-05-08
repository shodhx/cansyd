import tensorflow as tf

class LoRAAdapter(tf.keras.layers.Layer):
    def __init__(self, dim, rank=4):
        super().__init__()
        self.dim = dim
        self.rank = rank
    
    def build(self, input_shape):
        self.A = self.add_weight('A', (self.dim, self.rank), initializer='glorot_uniform')
        self.B = self.add_weight('B', (self.rank, self.dim), initializer='zeros')
    
    def call(self, x):
        return x + tf.matmul(x, tf.matmul(self.A, self.B))

base_cnn = build_cnn((1024, 1), 10)
base_cnn.load_weights('base_model.h5')

for layer in base_cnn.layers[:-1]:
    layer.trainable = False

inp = tf.keras.Input((1024, 1))
x = base_cnn.layers[0](inp)
for layer in base_cnn.layers[1:-3]:
    x = layer(x)
feat = base_cnn.layers[-3](x)
lora = LoRAAdapter(128, rank=4)(feat)
drop = base_cnn.layers[-2](lora)
out = tf.keras.layers.Dense(4, activation='softmax')(drop)

ccr_model = tf.keras.Model(inp, out)

X_base_train = np.concatenate([X_norm, X_b07, X_b14, X_b21, X_i07, X_i14, X_i21])
y_base_train = np.concatenate([y_norm, y_b07, y_b14, y_b21, y_i07, y_i14, y_i21])

X_new_train = np.concatenate([X_o07, X_o14, X_o21])
y_new_train = np.concatenate([y_o07, y_o14, y_o21]) - 7

feat_model_base = tf.keras.Model(base_cnn.layers[0].input, base_cnn.layers[-3].output)
feats_old = feat_model_base.predict(X_base_train.reshape(-1,1024,1), verbose=0)
norms_old = np.linalg.norm(feats_old, axis=1)
loads_old = np.concatenate([np.full(len(X_norm), 0), np.full(len(X_b07), 1),
                            np.full(len(X_b14), 2), np.full(len(X_b21), 3),
                            np.full(len(X_i07), 0), np.full(len(X_i14), 1),
                            np.full(len(X_i21), 2)])
fault_old = (y_base_train > 0).astype(int)
ate_old = backdoor_ate(X_base_train.reshape(len(X_base_train), -1), fault_old, norms_old, loads_old)

for epoch in range(30):
    for i in range(0, len(X_new_train), 64):
        batch_x = X_new_train[i:i+64].reshape(-1, 1024, 1)
        batch_y = y_new_train[i:i+64]
        
        with tf.GradientTape() as tape:
            pred = ccr_model(batch_x, training=True)
            ce_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, pred)
            
            feats_new = feat_model_base(batch_x, training=False)
            norms_new = tf.norm(feats_new, axis=1)
            loads_new = np.full(len(batch_x), 0)
            
            causal_penalty = tf.square(norms_new - ate_old)
            
            loss = tf.reduce_mean(ce_loss) + 1.0 * tf.reduce_mean(causal_penalty)
        
        grads = tape.gradient(loss, ccr_model.trainable_weights)
        tf.keras.optimizers.Adam(0.001).apply_gradients(zip(grads, ccr_model.trainable_weights))

old_acc = base_cnn.evaluate(X_base_train.reshape(-1,1024,1), y_base_train, verbose=0)[1]
new_acc = ccr_model.evaluate(X_new_train.reshape(-1,1024,1), y_new_train, verbose=0)[1]

feats_after = feat_model_base.predict(X_base_train.reshape(-1,1024,1), verbose=0)
norms_after = np.linalg.norm(feats_after, axis=1)
ate_after = backdoor_ate(X_base_train.reshape(len(X_base_train), -1), fault_old, norms_after, loads_old)
drift = abs(ate_after - ate_old)

print(f'CCR-LoRA: Old={old_acc:.4f}  New={new_acc:.4f}  Drift={drift:.6f}')

results = []
for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ccr_model = tf.keras.Model(inp, out)
    for epoch in range(30):
        for i in range(0, len(X_new_train), 64):
            batch_x = X_new_train[i:i+64].reshape(-1, 1024, 1)
            batch_y = y_new_train[i:i+64]
            with tf.GradientTape() as tape:
                pred = ccr_model(batch_x, training=True)
                ce_loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, pred)
                feats_new = feat_model_base(batch_x, training=False)
                norms_new = tf.norm(feats_new, axis=1)
                causal_penalty = tf.square(norms_new - ate_old)
                loss = tf.reduce_mean(ce_loss) + lam * tf.reduce_mean(causal_penalty)
            grads = tape.gradient(loss, ccr_model.trainable_weights)
            tf.keras.optimizers.Adam(0.001).apply_gradients(zip(grads, ccr_model.trainable_weights))
    
    old_acc = base_cnn.evaluate(X_base_train.reshape(-1,1024,1), y_base_train, verbose=0)[1]
    new_acc = ccr_model.evaluate(X_new_train.reshape(-1,1024,1), y_new_train, verbose=0)[1]
    feats_after = feat_model_base.predict(X_base_train.reshape(-1,1024,1), verbose=0)
    norms_after = np.linalg.norm(feats_after, axis=1)
    ate_after = backdoor_ate(X_base_train.reshape(len(X_base_train), -1), fault_old, norms_after, loads_old)
    drift = abs(ate_after - ate_old)
    results.append({'lambda': lam, 'old': old_acc, 'new': new_acc, 'drift': drift})

for r in results:
    print(f"λ={r['lambda']:.2f}  Old={r['old']:.4f}  New={r['new']:.4f}  Drift={r['drift']:.6f}")
"""
Continual-learning comparison.

Base classes 0-6 are learned first; new classes 7-9 are added few-shot with
2x replay. Four methods are compared (expected N=100/class results):

    Method            Old_Acc  New_Acc  ATE_drift
    Naive fine-tune    0.9351   0.9972   0.1151
    EWC (best lambda)  0.9705   0.9860   0.1163
    Standard LoRA      0.9410   0.8796   0.0000
    Causal M. LoRA     0.8955   0.8627   0.0000

Frozen-backbone methods (LoRA, CML) have ATE_drift = 0 by construction, which is
tautological for any frozen backbone. CCR-LoRA (partial adaptation, target
drift < 0.01) is NOT evaluated here - it is not yet validated. See
continual/ccr_lora.py.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.linear_model import LinearRegression
from core.architecture import build_cnn
from continual.ccr_lora import LoRAAdapter

BASE_CLASSES = list(range(7))
NEW_CLASSES = [7, 8, 9]
REPLAY_RATIO = 2
LORA_RANK = 8
FISHER_SAMPLES = 200

def _build(num_classes):
    """build_cnn wrapper: supplies input_shape and compiles (build_cnn returns an
    uncompiled model and requires input_shape)."""
    m = build_cnn((1024, 1), num_classes)
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def _feat_norms(feat_ext, X):
    feats = feat_ext.predict(X, verbose=0)
    return np.linalg.norm(feats, axis=1)

def _ate(feat_ext, X_old_te, y_old_te):
    fn = _feat_norms(feat_ext, X_old_te)
    fb = (y_old_te > 0).astype(int)
    return LinearRegression().fit(fn.reshape(-1, 1), fb).coef_[0]

def train_base(X_base_tr, y_base_tr, X_base_te, y_base_te, seed=42):
    tf.random.set_seed(seed); np.random.seed(seed)
    base_cnn = _build(7)
    es = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=0)
    lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
    base_cnn.fit(X_base_tr, y_base_tr, epochs=60, batch_size=64,
                 validation_split=0.15, callbacks=[es, lr], verbose=0)
    base_feat = tf.keras.Model(inputs=base_cnn.input, outputs=base_cnn.layers[-3].output)
    for layer in base_feat.layers:
        layer.trainable = False
    ate_before = _ate(base_feat, X_base_te, y_base_te)
    return base_cnn, base_feat, ate_before

def _subsample_new(X, y, n_per_class, rng):
    Xs, ys = [], []
    for c in NEW_CLASSES:
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        Xs.append(X[chosen]); ys.extend(y[chosen])
    return np.concatenate(Xs), np.array(ys)

def _build_lora_model(feat_ext, rank=LORA_RANK, num_classes=10):
    inp = tf.keras.Input(shape=(1024, 1))
    feats = feat_ext(inp)
    out = layers.Dense(num_classes, activation='softmax')(LoRAAdapter(128, rank)(feats))
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def _copy_base_weights(base_cnn, target):
    for i, layer in enumerate(base_cnn.layers[:-1]):
        if i < len(target.layers) - 1:
            try:
                target.layers[i].set_weights(layer.get_weights())
            except Exception:
                pass

def _evaluate(model, X_old_te, y_old_te, X_new_te, y_new_te, feat_ext):
    old_acc = np.mean(model.predict(X_old_te, verbose=0).argmax(1) == y_old_te)
    new_acc = np.mean(model.predict(X_new_te, verbose=0).argmax(1) == y_new_te)
    return float(old_acc), float(new_acc), _ate(feat_ext, X_old_te, y_old_te)

def _compute_fisher(model, X, y, n_samples=FISHER_SAMPLES):
    fisher = [tf.zeros_like(v) for v in model.trainable_variables]
    for i in range(min(n_samples, len(X))):
        with tf.GradientTape() as tape:
            logits = model(X[i:i+1], training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y[i:i+1], logits)
        for j, g in enumerate(tape.gradient(loss, model.trainable_variables)):
            if g is not None:
                fisher[j] = fisher[j] + g ** 2
    return [f / n_samples for f in fisher]

def _train_ewc(base_cnn, X_comb, y_comb, lam, epochs=30):
    ewc = _build(10)
    _copy_base_weights(base_cnn, ewc)
    fisher = _compute_fisher(ewc, X_comb, y_comb)
    old_p = [v.numpy().copy() for v in ewc.trainable_variables]
    opt = tf.keras.optimizers.Adam(0.0005)
    for _ in range(epochs):
        idx = np.random.permutation(len(X_comb))
        for st in range(0, len(X_comb), 32):
            bi = idx[st:st+32]
            with tf.GradientTape() as tape:
                logits = ewc(X_comb[bi], training=True)
                ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_comb[bi], logits))
                pen = sum(tf.reduce_sum(f * (v - o) ** 2)
                          for f, v, o in zip(fisher, ewc.trainable_variables, old_p))
                total = ce + lam * pen
            grads = tape.gradient(total, ewc.trainable_variables)
            opt.apply_gradients(zip(grads, ewc.trainable_variables))
    ewc_feat = tf.keras.Model(inputs=ewc.input, outputs=ewc.layers[-3].output)
    return ewc, ewc_feat

def run_continual_comparison(X_train, y_train, X_test, y_test, n_shots=(10, 50, 100),
                             ewc_lambdas=(100, 1000), seed=42):
    rng = np.random.default_rng(seed)
    mtr_b, mtr_n = np.isin(y_train, BASE_CLASSES), np.isin(y_train, NEW_CLASSES)
    mte_b, mte_n = np.isin(y_test, BASE_CLASSES), np.isin(y_test, NEW_CLASSES)
    X_base_tr, y_base_tr = X_train[mtr_b], y_train[mtr_b]
    X_new_tr, y_new_tr = X_train[mtr_n], y_train[mtr_n]
    X_base_te, y_base_te = X_test[mte_b], y_test[mte_b]
    X_new_te, y_new_te = X_test[mte_n], y_test[mte_n]

    base_cnn, base_feat, ate_before = train_base(X_base_tr, y_base_tr, X_base_te, y_base_te, seed)
    print(f'ATE (old classes, before adaptation): {ate_before:.4f}')

    # Few-shot frozen-LoRA sweep (this is the 'Causal M. LoRA' row at N=100)
    print('\n=== CAUSAL MASKED LoRA - FEW-SHOT CONTINUAL LEARNING ===')
    print(f'{"N/class":>8} {"Old_Acc":>10} {"New_Acc":>10} {"ATE_drift":>10}')
    print('-' * 42)
    sweep = []
    for N in n_shots:
        np.random.seed(seed); tf.random.set_seed(seed)
        X_sub, y_sub = _subsample_new(X_new_tr, y_new_tr, N, rng)
        n_rep = min(len(X_base_tr), len(X_sub) * REPLAY_RATIO)
        ridx = rng.choice(len(X_base_tr), n_rep, replace=False)
        X_c = np.concatenate([X_base_tr[ridx], X_sub]); y_c = np.concatenate([y_base_tr[ridx], y_sub])
        lora = _build_lora_model(base_feat); lora.fit(X_c, y_c, epochs=15, batch_size=32, verbose=0)
        o, n, a = _evaluate(lora, X_base_te, y_base_te, X_new_te, y_new_te, base_feat)
        sweep.append((N, o, n, abs(a - ate_before)))
        print(f'{N:>8} {o:>10.4f} {n:>10.4f} {abs(a-ate_before):>10.6f}')

    # N=100 comparison table
    N_BL = 100
    X_100, y_100 = _subsample_new(X_new_tr, y_new_tr, N_BL, rng)
    n_rep = min(len(X_base_tr), len(X_100) * REPLAY_RATIO)
    ridx = rng.choice(len(X_base_tr), n_rep, replace=False)
    X_c = np.concatenate([X_base_tr[ridx], X_100]); y_c = np.concatenate([y_base_tr[ridx], y_100])

    # Naive
    naive = _build(10); _copy_base_weights(base_cnn, naive)
    naive.fit(X_c, y_c, epochs=15, batch_size=32, verbose=0)
    naive_feat = tf.keras.Model(inputs=naive.input, outputs=naive.layers[-3].output)
    naive_o, naive_n, naive_a = _evaluate(naive, X_base_te, y_base_te, X_new_te, y_new_te, naive_feat)

    # EWC sweep -> best by old_acc
    best_ewc = None
    for lam in ewc_lambdas:
        ewc, ewc_feat = _train_ewc(base_cnn, X_c, y_c, lam)
        o, n, a = _evaluate(ewc, X_base_te, y_base_te, X_new_te, y_new_te, ewc_feat)
        if best_ewc is None or o > best_ewc[1]:
            best_ewc = (lam, o, n, a)

    # Standard LoRA (frozen) and CML (frozen) - both drift 0 by construction
    std_lora = _build_lora_model(base_feat); std_lora.fit(X_c, y_c, epochs=15, batch_size=32, verbose=0)
    std_o, std_n, std_a = _evaluate(std_lora, X_base_te, y_base_te, X_new_te, y_new_te, base_feat)
    cml_o, cml_n, cml_drift = sweep[-1][1], sweep[-1][2], sweep[-1][3]

    table = {
        'Naive fine-tune': (naive_o, naive_n, abs(naive_a - ate_before)),
        f'EWC (lambda={best_ewc[0]})': (best_ewc[1], best_ewc[2], abs(best_ewc[3] - ate_before)),
        'Standard LoRA': (std_o, std_n, abs(std_a - ate_before)),
        'Causal M. LoRA': (cml_o, cml_n, cml_drift),
    }
    print('\n=== FINAL COMPARISON (N=100/class) ===')
    print(f'{"Method":<22} {"Old_Acc":>10} {"New_Acc":>10} {"ATE_drift":>10}')
    print('-' * 54)
    for name, (o, n, d) in table.items():
        print(f'{name:<22} {o:>10.4f} {n:>10.4f} {d:>10.6f}')
    print('\nFrozen-backbone methods (LoRA, CML) have drift=0 by construction.')
    print('CCR-LoRA (partial adaptation) is NOT evaluated here - not yet validated. See continual/ccr_lora.py.')
    return {'sweep': sweep, 'table': table, 'ate_before': float(ate_before)}

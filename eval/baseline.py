import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import f1_score
from core.architecture import build_cnn

# ── Published baseline architectures (WDCNN, TICNN) ──────────────────────

def build_wdcnn():
    model = models.Sequential([
        tf.keras.Input(shape=(1024, 1)),
        layers.Conv1D(16, 64, strides=8, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.GlobalAveragePooling1D(),
        layers.Dense(100, activation='relu'), layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_ticnn():
    inp = tf.keras.Input(shape=(1024, 1))
    x = layers.GaussianNoise(0.1)(inp)
    x = layers.Conv1D(16, 64, strides=8, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(2)(x); x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(100, activation='relu')(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def _train_baseline(builder, name, X_train, y_train, X_test, y_test, seeds=(42, 43, 44)):
    f1s = []
    for s in seeds:
        tf.random.set_seed(s); np.random.seed(s)
        m = builder()
        es = callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                     restore_best_weights=True, verbose=0)
        lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=5, min_lr=1e-5, verbose=0)
        m.fit(X_train, y_train, epochs=30, batch_size=64,
              validation_split=0.15, callbacks=[es, lr], verbose=0)
        yp = m.predict(X_test, verbose=0).argmax(axis=1)
        f1s.append(f1_score(y_test, yp, average='weighted'))
    print(f'{name:<12} F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}')
    return f1s

def run_published_baselines(X_train, y_train, X_test, y_test, seeds=(42, 43, 44)):
    """
    Protocol B comparison. Expected:
      WDCNN      0.8815 +/- 0.0046
      TICNN      0.8720 +/- 0.0269
      CNSD-WDCNN 0.8784 +/- 0.0063   (build_cnn backbone, no contrastive training)
    """
    print('=== PUBLISHED BASELINES (Protocol B - Cross-Load) ===')
    f1_wdcnn = _train_baseline(build_wdcnn, 'WDCNN', X_train, y_train, X_test, y_test, seeds)
    f1_ticnn = _train_baseline(build_ticnn, 'TICNN', X_train, y_train, X_test, y_test, seeds)
    f1_cnsd = _train_baseline(build_cnn, 'CNSD-WDCNN', X_train, y_train, X_test, y_test, seeds)
    return {
        'WDCNN': (float(np.mean(f1_wdcnn)), float(np.std(f1_wdcnn))),
        'TICNN': (float(np.mean(f1_ticnn)), float(np.std(f1_ticnn))),
        'CNSD-WDCNN': (float(np.mean(f1_cnsd)), float(np.std(f1_cnsd))),
    }

# ── Invariant Risk Minimization (IRMv1) ─────────────────────────────────────

def _build_irm_net():
    inp = tf.keras.Input(shape=(1024, 1))
    x = layers.Conv1D(32, 64, strides=4, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(64, 16, strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(4)(x)
    x = layers.Conv1D(128, 8, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.GlobalAveragePooling1D()(x)
    feat = layers.Dense(128, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(feat)
    out = layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inp, out)

def train_irm(X_tr, y_tr, loads_tr, X_te, y_te, lam_irm=1.0, epochs=60, seed=42):
    """IRMv1: ERM + penalty on the gradient of each environment's loss w.r.t a dummy scale."""
    tf.random.set_seed(seed); np.random.seed(seed)
    model = _build_irm_net()
    opt = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    envs = sorted(np.unique(loads_tr))
    env_indices = {e: np.where(loads_tr == e)[0] for e in envs}

    for _ in range(epochs):
        for e in envs:
            idx = np.random.choice(env_indices[e], min(64, len(env_indices[e])), replace=False)
            with tf.GradientTape() as tape:
                scale = tf.Variable(1.0, dtype=tf.float32)
                preds = model(X_tr[idx], training=True) * scale
                e_loss = loss_fn(y_tr[idx], preds)
            grad_scale = tape.gradient(e_loss, scale)
            irm_penalty = grad_scale ** 2 if grad_scale is not None else 0.0
            with tf.GradientTape() as tape2:
                preds2 = model(X_tr[idx], training=True)
                total = loss_fn(y_tr[idx], preds2) + lam_irm * irm_penalty
            grads = tape2.gradient(total, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

    yp = model.predict(X_te, verbose=0).argmax(axis=1)
    return f1_score(y_te, yp, average='weighted')

def run_irm(X_train, y_train, loads_train, X_test, y_test, lambdas=(0.1, 1.0), seeds=(42, 43, 44)):
    """Expected: best lambda=1.0 -> F1 = 0.8542 +/- 0.1029."""
    print('=== INVARIANT RISK MINIMIZATION (IRMv1) ===')
    print(f'{"Lambda":>10} {"F1_mean":>10} {"F1_std":>10} {"Seeds":>6}')
    print('-' * 40)
    results = {}
    best = (0.0, None, 0.0)
    for lam in lambdas:
        f1s = [train_irm(X_train, y_train, loads_train, X_test, y_test, lam_irm=lam, seed=s)
               for s in seeds]
        mean, std = float(np.mean(f1s)), float(np.std(f1s))
        results[lam] = (mean, std)
        print(f'{lam:>10.1f} {mean:>10.4f} {std:>10.4f} {len(f1s):>6}')
        if mean > best[0]:
            best = (mean, lam, std)
    print(f'Best IRM: lambda={best[1]}, F1={best[0]:.4f} +/- {best[2]:.4f}')
    return results

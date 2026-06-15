import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from core.architecture import build_cnn

def run_wdcnn(X_train, y_train, X_test, y_test, seeds=(42, 123, 456)):
    """Plain WDCNN baseline, multi-seed, Protocol B."""
    results = []
    for seed in seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        wdcnn = build_cnn((1024, 1), 10)
        wdcnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        wdcnn.fit(X_train, y_train, epochs=30, batch_size=64, verbose=0)
        pred = wdcnn.predict(X_test, verbose=0).argmax(axis=1)
        f1 = f1_score(y_test, pred, average='weighted')
        results.append({'seed': seed, 'f1': f1})
        print(f"WDCNN seed={seed}: F1={f1:.4f}")
    return results

def run_irm(X_train, y_train, X_test, y_test, lambdas=(0.1, 1.0), seeds=(42, 123, 456)):
    """IRMv1 baseline. Penalty = squared grad of the loss w.r.t a dummy scale=1."""
    results = []
    for lam in lambdas:
        for seed in seeds:
            np.random.seed(seed)
            tf.random.set_seed(seed)
            model = build_cnn((1024, 1), 10)
            opt = tf.keras.optimizers.Adam(0.001)
            for _ in range(30):
                for i in range(0, len(X_train), 64):
                    bx, by = X_train[i:i+64], y_train[i:i+64]
                    with tf.GradientTape() as tape:
                        scale = tf.constant(1.0)
                        with tf.GradientTape() as inner:
                            inner.watch(scale)
                            logits = model(bx, training=True) * scale
                            ce = tf.reduce_mean(
                                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=by, logits=logits))
                        penalty = tf.square(inner.gradient(ce, scale))
                        loss = ce + lam * penalty
                    grads = tape.gradient(loss, model.trainable_weights)
                    opt.apply_gradients(zip(grads, model.trainable_weights))
            pred = model.predict(X_test, verbose=0).argmax(axis=1)
            f1 = f1_score(y_test, pred, average='weighted')
            results.append({'lambda': lam, 'seed': seed, 'f1': f1})
            print(f"IRM lambda={lam} seed={seed}: F1={f1:.4f}")
    return results

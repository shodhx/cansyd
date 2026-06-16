import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, classification_report
from core.architecture import build_cnn

def train_cnn(X_train, y_train, seed=42, epochs=30, batch_size=64):
    """Train one CNSD-CNN backbone with a fixed seed."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = build_cnn((1024, 1), 10)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def evaluate_protocol_b(X_train, y_train, X_test, y_test, seeds=(42, 123, 456), epochs=30, report=False):
    """
    CNSD-CNN backbone on Protocol B (cross-load: train loads 0/1/2, test load 3).
    Reports mean/std weighted F1 across seeds (Protocol B, cross-load).
    Returns the mean/std F1 and the last trained model (handy for downstream layers).
    """
    f1s = []
    last_model = None
    for seed in seeds:
        model = train_cnn(X_train, y_train, seed=seed, epochs=epochs)
        pred = model.predict(X_test, verbose=0).argmax(axis=1)
        f1 = f1_score(y_test, pred, average='weighted')
        f1s.append(f1)
        last_model = model
        print(f"  CNN seed={seed}: F1={f1:.4f}")
        if report:
            print(classification_report(y_test, pred, zero_division=0))
    f1s = np.array(f1s)
    print(f"CNSD-CNN (Protocol B): F1 = {f1s.mean():.4f} +/- {f1s.std():.4f}")
    return {
        'f1_mean': float(f1s.mean()),
        'f1_std': float(f1s.std()),
        'f1_per_seed': f1s.tolist(),
        'model': last_model,
    }

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def build_cnn(input_shape, num_classes):
    inp = layers.Input(input_shape)
    x = layers.Conv1D(16, 64, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(
        inp
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(16)(x)
    x = layers.Conv1D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out)


def patchify(X, num_patches=8):
    batch_size = tf.shape(X)[0]
    seq_len = tf.shape(X)[1]
    patch_size = seq_len // num_patches
    patches = tf.reshape(X, [batch_size, num_patches, patch_size, 1])
    return patches


def build_jepa_encoder(patch_dim, encoder_dim):
    inp = layers.Input((patch_dim, 1))
    x = layers.Conv1D(64, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(encoder_dim)(x)
    return tf.keras.Model(inp, x)


def vicreg_loss(z1, z2, lam=25.0, mu=25.0, nu=1.0):
    inv = tf.reduce_mean(tf.square(z1 - z2))
    z1_norm = (z1 - tf.reduce_mean(z1, axis=0)) / (tf.math.reduce_std(z1, axis=0) + 1e-8)
    z2_norm = (z2 - tf.reduce_mean(z2, axis=0)) / (tf.math.reduce_std(z2, axis=0) + 1e-8)
    std1 = tf.sqrt(tf.nn.relu(1.0 - tf.math.reduce_std(z1_norm, axis=0)) + 1e-8)
    std2 = tf.sqrt(tf.nn.relu(1.0 - tf.math.reduce_std(z2_norm, axis=0)) + 1e-8)
    var = tf.reduce_mean(std1) + tf.reduce_mean(std2)
    cov1 = tf.matmul(tf.transpose(z1_norm), z1_norm) / tf.cast(tf.shape(z1)[0], tf.float32)
    cov2 = tf.matmul(tf.transpose(z2_norm), z2_norm) / tf.cast(tf.shape(z2)[0], tf.float32)
    off_diag1 = cov1 - tf.linalg.diag(tf.linalg.diag_part(cov1))
    off_diag2 = cov2 - tf.linalg.diag(tf.linalg.diag_part(cov2))
    cov = tf.reduce_sum(tf.square(off_diag1)) + tf.reduce_sum(tf.square(off_diag2))
    return lam * inv + mu * var + nu * cov


def train_jepa_backbone(X_train_all, y_train_all, epochs=15):
    """Encapsulates the S-JEPA training execution sequence to protect global scope imports"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    encoder = build_jepa_encoder(128, 256)
    target_encoder = build_jepa_encoder(128, 256)

    for w1, w2 in zip(encoder.weights, target_encoder.weights, strict=True):
        w2.assign(w1)

    optimizer = tf.keras.optimizers.Adam(0.001)

    for _ in range(epochs):
        for i in range(0, len(X_train_all), 64):
            batch = X_train_all[i : i + 64]
            p = patchify(batch)
            with tf.GradientTape() as tape:
                z1 = encoder(p[:, 0], training=True)
                z2 = target_encoder(p[:, 1], training=False)
                loss = vicreg_loss(z1, z2)
            grads = tape.gradient(loss, encoder.trainable_weights)
            optimizer.apply_gradients(
                 zip(grads, encoder.trainable_weights, strict=True)
            )

            for w1, w2 in zip(encoder.weights, target_encoder.weights, strict=True):
                w2.assign(0.99 * w2 + 0.01 * w1)

    jepa_embeddings = []
    for i in range(0, len(X_train_all), 64):
        batch = X_train_all[i : i + 64]
        p = patchify(batch)
        e = [encoder(p[:, j], training=False) for j in range(8)]
        jepa_embeddings.append(tf.reduce_mean(tf.stack(e, axis=1), axis=1).numpy())
    jepa_embeddings = np.concatenate(jepa_embeddings)

    scaler = StandardScaler()
    jepa_tr = scaler.fit_transform(jepa_embeddings)
    probe = LogisticRegression(max_iter=1000)
    probe.fit(jepa_tr, y_train_all)

    return encoder, probe, scaler


def _train_cnn(X, y, num_classes, epochs=30, seed=42):
    """Train the 1D CNN classifier and return the fitted model."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    import numpy as np

    np.random.seed(seed)
    model = build_cnn(input_shape=(X.shape[1], 1), num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(X, y, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1)
    return model

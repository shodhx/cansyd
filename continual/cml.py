import tensorflow as tf
import numpy as np

X_base_train = np.concatenate([X_norm, X_b07, X_b14, X_b21, X_i07, X_i14, X_i21])
y_base_train = np.concatenate([y_norm, y_b07, y_b14, y_b21, y_i07, y_i14, y_i21])

X_new_train = np.concatenate([X_o07, X_o14, X_o21])
y_new_train = np.concatenate([y_o07, y_o14, y_o21]) - 7

base_model = build_cnn((1024, 1), 7)
base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
base_model.fit(X_base_train.reshape(-1,1024,1), y_base_train, epochs=30, batch_size=64, verbose=0)

for layer in base_model.layers:
    layer.trainable = False

inp = tf.keras.Input((1024, 1))
x = base_model(inp)
out = tf.keras.layers.Dense(4, activation='softmax')(x)
cml_model = tf.keras.Model(inp, out)
cml_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

for n_shot in [10, 50, 100]:
    idx = []
    for c in range(4):
        c_idx = np.where(y_new_train == c)[0]
        idx.extend(np.random.choice(c_idx, n_shot, replace=False))
    
    X_shot = X_new_train[idx].reshape(-1, 1024, 1)
    y_shot = y_new_train[idx]
    
    cml_model.fit(X_shot, y_shot, epochs=15, batch_size=32, verbose=0)
    
    old_acc = base_model.evaluate(X_base_train.reshape(-1,1024,1), y_base_train, verbose=0)[1]
    new_acc = cml_model.evaluate(X_new_train.reshape(-1,1024,1), y_new_train, verbose=0)[1]
    
    print(f'CML {n_shot}-shot: Old={old_acc:.4f} New={new_acc:.4f}')
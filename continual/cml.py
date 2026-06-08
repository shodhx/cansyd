import tensorflow as tf
import numpy as np

def train_cml(base_model, X_shot, y_shot, epochs=15, batch_size=32):
    """
    Executes a comparative standard few-shot parameter-tuning routine
    without causal consistency constraints.
    """
    for layer in base_model.layers:
        layer.trainable = False
        
    inp = tf.keras.Input(shape=base_model.input_shape[1:])
    x = base_model(inp, training=False)
    out = tf.keras.layers.Dense(len(np.unique(y_shot)), activation='softmax')(x)
    
    cml_model = tf.keras.Model(inp, out)
    cml_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    cml_model.fit(X_shot, y_shot, epochs=epochs, batch_size=batch_size, verbose=0)
    return cml_model
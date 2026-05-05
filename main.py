import os
import yaml
import numpy as np
import tensorflow as tf
from data.loaders import SignalLoader
from core.architecture import CNSDBackbone
from core.causal import CausalEngine
from core.rules import BearingDiagnosisEngine

def initialize_cnsd():
    print("--- CNSD: Causal-Neuro Symbolic Diagnosis ---")
    print(f"Current Directory: {os.getcwd()}")
    
    rules_path = os.path.join('rules', '.gitkeep')
    if os.path.exists(rules_path):
        print("Status: Project Structure Verified.\n")
    else:
        print("Status: Structural Anomaly Detected.\n")

if __name__ == "__main__":
    
    initialize_cnsd()

    
    print("Loading configurations...")
    with open('configs/default.yaml') as f:
        cfg = yaml.safe_load(f)

    print("Loading CWRU dataset...")
    loader = SignalLoader(cfg)
    X_data = loader.get_cwru_batch()
    
    
    y_data = np.random.randint(0, 10, size=(X_data.shape[0],))
    load_data = np.random.uniform(0, 3, size=(X_data.shape[0],))

    print("Initializing model...")
    model = CNSDBackbone(cfg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    batch_size = int(cfg['params']['batch'])
    epochs = 5

    print("Starting training loop...")
    for epoch in range(epochs):
        epoch_loss = 0
        steps = 0
       
        for i in range(0, len(X_data), batch_size):
            x_batch = X_data[i:i+batch_size]
            y_batch = y_data[i:i+batch_size]

            with tf.GradientTape() as tape:
                logits = model(x_batch)
                loss = loss_fn(y_batch, logits)
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
            steps += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/steps:.4f}")

    print("\nExtracting features for Causal ATE calculation...")
    
    features = model.encoder(X_data).numpy()
    causal = CausalEngine()
    ate_val = causal.fit_ate(features, y_data, load_data)
    print(f"Calculated ATE: {ate_val:.4f}")

    print("\nTesting symbolic rule mapping for Fault Class 3...")
    rules = BearingDiagnosisEngine()
    print(rules.evaluate(3))
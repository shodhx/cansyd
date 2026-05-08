import tensorflow as tf
import numpy as np

def irm_penalty(logits, y, scale=1.0):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    grad = tf.gradients(loss, [scale])[0]
    return tf.reduce_sum(grad ** 2)

irm_results = []
for lam in [0.1, 1.0]:
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        irm_model = build_cnn((1024, 1), 10)
        opt = tf.keras.optimizers.Adam(0.001)
        
        for epoch in range(30):
            for i in range(0, len(X_train_all), 64):
                batch_x = X_train_all[i:i+64]
                batch_y = y_train_all[i:i+64]
                
                with tf.GradientTape() as tape:
                    scale = tf.Variable(1.0)
                    logits = irm_model(batch_x, training=True)
                    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=logits))
                    penalty = irm_penalty(logits * scale, batch_y, scale)
                    loss = ce + lam * penalty
                
                grads = tape.gradient(loss, irm_model.trainable_weights)
                opt.apply_gradients(zip(grads, irm_model.trainable_weights))
        
        loss, acc = irm_model.evaluate(X_test_all, y_test_all, verbose=0)
        pred = irm_model.predict(X_test_all, verbose=0).argmax(axis=1)
        
        from sklearn.metrics import classification_report
        report = classification_report(y_test_all, pred, output_dict=True, zero_division=0)
        
        irm_results.append({'lambda': lam, 'seed': seed, 'acc': acc, 'f1': report['weighted avg']['f1-score']})

for r in irm_results:
    print(f"IRM λ={r['lambda']} seed={r['seed']}: F1={r['f1']:.4f}")

wdcnn_results = []
for seed in [42, 123, 456]:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    wdcnn = build_cnn((1024, 1), 10)
    wdcnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    wdcnn.fit(X_train_all, y_train_all, epochs=30, batch_size=64, verbose=0)
    loss, acc = wdcnn.evaluate(X_test_all, y_test_all, verbose=0)
    pred = wdcnn.predict(X_test_all, verbose=0).argmax(axis=1)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test_all, pred, average='weighted')
    wdcnn_results.append({'seed': seed, 'f1': f1})
    print(f"WDCNN seed={seed}: F1={f1:.4f}")
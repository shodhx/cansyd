import sys
import traceback

try:
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import numpy as np
    import scipy.stats as stats
    import tensorflow as tf

    from cnsd import Dataset
    from cnsd.diagnosis.system import CNSD
    from cnsd.perception.cnn import _train_cnn  # <--- WE IMPORT JUST THE CNN TRAINER
    from cnsd.physics import PhysicsConfig
    from validate_pu import load_pu_domain_split

    print('Loading Authentic PU dataset (Cross-Domain RPM Split)...')
    (X_train, y_train, cond_train), (X_target, y_target, cond_target) = load_pu_domain_split()

    unique_rpm = set(cond_train).union(set(cond_target))
    rpm_map = {float(r): float(r) for r in unique_rpm}
    pu_physics = PhysicsConfig(
        bearing={'n_balls': 8, 'd_ball': 6.75, 'd_pitch': 28.5, 'contact_angle': 0.0},
        cond_to_rpm=rpm_map,
        fs=64000,
        name='PU-6203',
    )
    pu_taxonomy = {
        0: ('Normal', 'None'),
        1: ('Outer Race', 'Medium'),
        2: ('Inner Race', 'High'),
    }

    # Shuffle train data exactly like validate_pu.py to ensure parity
    train_indices = np.arange(len(y_train))
    np.random.seed(42)
    np.random.shuffle(train_indices)
    X_train, y_train, cond_train = (
        X_train[train_indices],
        y_train[train_indices],
        cond_train[train_indices],
    )

    train_ds = Dataset.from_arrays(
        X_train,
        y_train,
        cond_train,
        fs=64000,
        physics=pu_physics,
        taxonomy=pu_taxonomy,
        name='PU_Train',
    )
    test_ds = Dataset.from_arrays(
        X_target,
        y_target,
        cond_target,
        fs=64000,
        physics=pu_physics,
        taxonomy=pu_taxonomy,
        name='PU_Test',
    )

    # Train Primary Model
    print('\n[A] Training Primary CNSD Model (Seed=42)...')
    np.random.seed(42)
    tf.random.set_seed(42)
    model = CNSD()
    model.fit(train_ds, epochs=20)

    # Train Ensemble Models
    # FIX: We only need the CNN part for the ensemble, not the full Physics+DoWhy SCM pipeline!
    # Training the full CNSD() 4 times was causing the joblib multiprocessing memory leak.
    ens = []
    nc = int(train_ds.y.max()) + 1
    for s in [100, 101, 102]:
        print(f'\n[B] Training Ensemble Member CNN only (Seed={s})...')
        np.random.seed(s)
        tf.random.set_seed(s)
        # Train just the Keras CNN directly, skipping the heavy memory SCM
        m_cnn = _train_cnn(train_ds.X, train_ds.y, num_classes=nc, epochs=20, seed=s)
        ens.append(m_cnn)

    # ---- shared: test signals, labels, condition ----
    sig = np.stack([test_ds.X[i].reshape(-1) for i in range(len(test_ds.X))]).astype(np.float32)
    yte = test_ds.y
    cond_te = test_ds.cond
    Xin = sig[..., None]  # (n, L, 1) for the conv nets

    def gap_by_score(score, correct, thr):
        hi = score >= thr
        lo = ~hi
        ah = correct[hi].mean() if hi.any() else float('nan')
        al = correct[lo].mean() if lo.any() else float('nan')
        return ah - al, ah, al, int(hi.sum()), int(lo.sum())

    print('=' * 60, '\n1. PHYSICS gap across tau\n', '=' * 60)
    probs = model.cnn.predict(Xin, batch_size=128, verbose=0)
    pred = probs.argmax(1)
    correct = pred == yte
    print(f'baseline CNN acc on test: {correct.mean():.3f}')
    for tau in [1.0, 2.0, 3.0]:
        model.symbolic.tau = float(tau)
        verds = np.array(
            [
                model.symbolic.diagnose(sig[i], pred[i], cond_te[i])['verdict']
                for i in range(len(sig))
            ]
        )
        conf = verds == 'CONFIRMED'
        cnfl = verds == 'CONFLICT'
        ca = correct[conf].mean() if conf.any() else float('nan')
        fa = correct[cnfl].mean() if cnfl.any() else float('nan')
        print(
            f'  tau={tau:.1f}  GAP={ca - fa:+.3f}  CONF={ca:.3f}(n={conf.sum()})  CNFL={fa:.3f}(n={cnfl.sum()})'
        )

    print('\n', '=' * 60, '\n2. SOFTMAX confidence\n', '=' * 60)
    softmax_conf = probs.max(1)
    for thr in [0.5, 0.7, 0.9, 0.95, 0.99]:
        g, ah, al, nh, nl = gap_by_score(softmax_conf, correct, thr)
        print(f'  conf>={thr:.2f}  GAP={g:+.3f}  HI={ah:.3f}(n={nh})  LO={al:.3f}(n={nl})')

    print('\n', '=' * 60, '\n3. MC-DROPOUT (30 passes)\n', '=' * 60)

    class AlwaysDropout(tf.keras.layers.Dropout):
        def call(self, inputs, training=None):
            return super().call(inputs, training=True)

    def clone_for_mc(layer):
        if isinstance(layer, tf.keras.layers.Dropout):
            return AlwaysDropout(layer.rate)
        return layer.__class__.from_config(layer.get_config())

    mc_model = tf.keras.models.clone_model(model.cnn, clone_function=clone_for_mc)
    mc_model.set_weights(model.cnn.get_weights())

    T = 30
    mc = []
    for _ in range(T):
        mc.append(mc_model.predict(Xin, batch_size=128, verbose=0))
    mc = np.stack(mc)  # (T,n,classes)
    mc_mean = mc.mean(0)
    mc_pred = mc_mean.argmax(1)
    mc_correct = mc_pred == yte
    eps = 1e-12
    certainty = (mc_mean * np.log(mc_mean + eps)).sum(1)  # = -entropy; higher = more certain
    print(f'MC mean acc: {mc_correct.mean():.3f}')
    for q in [50, 30, 20, 10]:  # keep top (100-q)% most certain as 'reliable'
        thr = np.percentile(certainty, q)
        g, ah, al, nh, nl = gap_by_score(certainty, mc_correct, thr)
        print(f'  top {100 - q}% certain  GAP={g:+.3f}  HI={ah:.3f}(n={nh})  LO={al:.3f}(n={nl})')

    print('\n', '=' * 60, '\n4. ENSEMBLE disagreement\n', '=' * 60)
    ens_preds = np.stack([m.predict(Xin, batch_size=128, verbose=0).argmax(1) for m in ens])
    vote = stats.mode(ens_preds, axis=0, keepdims=False).mode
    unanimous = (ens_preds == vote).all(0)
    ens_correct = vote == yte
    print(f'ensemble vote acc: {ens_correct.mean():.3f} | unanimous: {unanimous.mean():.1%}')
    g, ah, al, nh, nl = gap_by_score(
        unanimous.astype(float), ens_correct, 1.0
    )  # unanimous=reliable
    print(f'  unanimous=reliable  GAP={g:+.3f}  HI={ah:.3f}(n={nh})  LO={al:.3f}(n={nl})')

    print('\n', '=' * 60, '\n5. NOISE TEST: physics CONFLICT on unanimous-but-wrong\n', '=' * 60)
    rng = np.random.RandomState(42)
    sig_power = (sig**2).mean()
    model.symbolic.tau = 3.0
    print(
        f'{"noise":>6} | {"ens_acc":>7} | {"unanim-wrong":>12} | {"phys catches":>12} | {"base rate":>9}'
    )
    print('-' * 60)
    for snr_db in [np.inf, 20, 10, 5, 0]:
        if np.isinf(snr_db):
            sig_n = sig
        else:
            npow = sig_power / (10 ** (snr_db / 10))
            sig_n = sig + rng.randn(*sig.shape).astype(np.float32) * np.sqrt(npow)
        Xin_n = sig_n[..., None]
        ep = np.stack([m.predict(Xin_n, batch_size=128, verbose=0).argmax(1) for m in ens])
        v = stats.mode(ep, axis=0, keepdims=False).mode
        unan = (ep == v).all(0)
        ok = v == yte
        pv = np.array(
            [
                model.symbolic.diagnose(sig_n[i], v[i], cond_te[i])['verdict']
                for i in range(len(sig_n))
            ]
        )
        pc = pv == 'CONFLICT'
        uw = unan & (~ok)
        catch = pc[uw].mean() if uw.sum() > 0 else float('nan')
        s = 'clean' if np.isinf(snr_db) else f'{snr_db}dB'
        print(f'{s:>6} | {ok.mean():>7.3f} | {uw.sum():>12} | {catch:>12.3f} | {pc.mean():>9.3f}')

    print('================ DONE ================')

except Exception:
    with open('crash_traceback.txt', 'w') as f:
        traceback.print_exc(file=f)
    print('CRASHED. Check crash_traceback.txt')
    sys.exit(1)

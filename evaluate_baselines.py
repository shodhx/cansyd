import os
import sys
import traceback

import numpy as np
import scipy.stats as stats

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf

    from cnsd import Dataset
    from cnsd.diagnosis.system import CNSD
    from cnsd.perception.cnn import _train_cnn
    from cnsd.physics import PhysicsConfig
    from validate_pu import load_pu_domain_split

    print('Loading Authentic PU dataset (Cross-Domain RPM Split)...')
    (X_train_full, y_train_full, cond_train_full), (X_target, y_target, cond_target) = (
        load_pu_domain_split()
    )

    unique_rpm = set(cond_train_full).union(set(cond_target))
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

    def get_matched_coverage_gap(score, correct, target_n):
        if target_n == 0:
            return float('nan')
        # Score is higher for MORE confident
        # Sort descending by score
        sorted_indices = np.argsort(score)[::-1]
        hi_indices = sorted_indices[:target_n]

        hi_mask = np.zeros(len(score), dtype=bool)
        hi_mask[hi_indices] = True
        lo_mask = ~hi_mask

        ah = correct[hi_mask].mean() if hi_mask.any() else float('nan')
        al = correct[lo_mask].mean() if lo_mask.any() else float('nan')
        return ah - al

    class AlwaysDropout(tf.keras.layers.Dropout):
        def call(self, inputs, training=None):
            return super().call(inputs, training=True)

    def clone_for_mc(layer):
        if isinstance(layer, tf.keras.layers.Dropout):
            return AlwaysDropout(layer.rate)
        return layer.__class__.from_config(layer.get_config())

    seeds = [42, 43, 44, 45, 46]

    results = {
        'phys_gap': [],
        'soft_gap': [],
        'mc_gap': [],
        'ens_gap': [],
        'noise_catch': {db: [] for db in [np.inf, 20, 10, 5, 0]},
    }

    # Test data processing
    test_ds = Dataset.from_arrays(
        X_target,
        y_target,
        cond_target,
        fs=64000,
        physics=pu_physics,
        taxonomy=pu_taxonomy,
        name='PU_Test',
    )
    sig_te = np.stack([test_ds.X[i].reshape(-1) for i in range(len(test_ds.X))]).astype(np.float32)
    yte = test_ds.y
    cond_te = test_ds.cond
    Xin_te = sig_te[..., None]

    for seed in seeds:
        print(f'\n{"=" * 80}\n=== RUNNING SEED {seed} ===\n{"=" * 80}')
        tf.keras.backend.clear_session()
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # 1. 80/20 Split for Calibration
        indices = np.arange(len(y_train_full))
        np.random.shuffle(indices)

        split_idx = int(0.8 * len(indices))
        train_idx = indices[:split_idx]
        calib_idx = indices[split_idx:]

        X_tr, y_tr, cond_tr = (
            X_train_full[train_idx],
            y_train_full[train_idx],
            cond_train_full[train_idx],
        )
        X_ca, y_ca, cond_ca = (
            X_train_full[calib_idx],
            y_train_full[calib_idx],
            cond_train_full[calib_idx],
        )

        train_ds = Dataset.from_arrays(
            X_tr, y_tr, cond_tr, fs=64000, physics=pu_physics, taxonomy=pu_taxonomy, name='PU_Train'
        )
        calib_ds = Dataset.from_arrays(
            X_ca, y_ca, cond_ca, fs=64000, physics=pu_physics, taxonomy=pu_taxonomy, name='PU_Calib'
        )

        # 2. Train Primary Model (Bypass SCM to prevent multiprocess deadlocks)
        model = CNSD()
        nc = int(train_ds.y.max()) + 1
        model.cnn = _train_cnn(train_ds.X, train_ds.y, num_classes=nc, epochs=20, seed=seed)
        model.symbolic = model._build_symbolic(train_ds)
        model._fitted = True

        # 3. Train Ensemble Models
        ens = []
        nc = int(train_ds.y.max()) + 1
        for s in [seed * 10, seed * 10 + 1, seed * 10 + 2]:
            np.random.seed(s)
            tf.random.set_seed(s)
            m_cnn = _train_cnn(train_ds.X, train_ds.y, num_classes=nc, epochs=20, seed=s)
            ens.append(m_cnn)

        # 4. Calibrate Tau
        sig_ca = np.stack([calib_ds.X[i].reshape(-1) for i in range(len(calib_ds.X))]).astype(
            np.float32
        )
        Xin_ca = sig_ca[..., None]
        probs_ca = model.cnn.predict(Xin_ca, batch_size=128, verbose=0)
        pred_ca = probs_ca.argmax(1)
        correct_ca = pred_ca == calib_ds.y

        best_tau = 1.0
        best_gap = -100.0
        for tau in [1.0, 1.5, 2.0, 2.5, 3.0]:
            model.symbolic.tau = float(tau)
            verds = np.array(
                [
                    model.symbolic.diagnose(sig_ca[i], pred_ca[i], calib_ds.cond[i])['verdict']
                    for i in range(len(sig_ca))
                ]
            )
            conf = verds == 'CONFIRMED'
            cnfl = verds == 'CONFLICT'
            ca = correct_ca[conf].mean() if conf.any() else 0.0
            fa = correct_ca[cnfl].mean() if cnfl.any() else 1.0
            gap = ca - fa
            if gap > best_gap:
                best_gap = gap
                best_tau = tau

        print(f'Calibrated best tau = {best_tau} (Calib GAP = {best_gap:+.3f})')
        model.symbolic.tau = float(best_tau)

        # 5. Evaluate on Test Set
        probs_te = model.cnn.predict(Xin_te, batch_size=128, verbose=0)
        pred_te = probs_te.argmax(1)
        correct_te = pred_te == yte

        # Physics evaluation
        verds = np.array(
            [
                model.symbolic.diagnose(sig_te[i], pred_te[i], cond_te[i])['verdict']
                for i in range(len(sig_te))
            ]
        )
        conf = verds == 'CONFIRMED'
        cnfl = verds == 'CONFLICT'
        ca = correct_te[conf].mean() if conf.any() else float('nan')
        fa = correct_te[cnfl].mean() if cnfl.any() else float('nan')
        phys_gap = ca - fa
        target_n = int(conf.sum())
        results['phys_gap'].append(phys_gap)
        print(f'Physics GAP={phys_gap:+.3f} (Coverage N={target_n})')

        # Softmax evaluation at matched coverage
        softmax_score = probs_te.max(1)
        soft_gap = get_matched_coverage_gap(softmax_score, correct_te, target_n)
        results['soft_gap'].append(soft_gap)

        # MC-Dropout at matched coverage
        mc_model = tf.keras.models.clone_model(model.cnn, clone_function=clone_for_mc)
        mc_model.set_weights(model.cnn.get_weights())
        T = 30
        mc_preds = []
        for _ in range(T):
            mc_preds.append(mc_model.predict(Xin_te, batch_size=128, verbose=0))
        mc_preds = np.stack(mc_preds)
        mc_mean = mc_preds.mean(0)
        mc_pred_class = mc_mean.argmax(1)
        mc_correct = mc_pred_class == yte
        eps = 1e-12
        mc_score = (mc_mean * np.log(mc_mean + eps)).sum(1)  # Certainty (negative entropy)
        mc_gap = get_matched_coverage_gap(mc_score, mc_correct, target_n)
        results['mc_gap'].append(mc_gap)

        # Ensemble at matched coverage
        ens_preds_probs = np.stack(
            [m.predict(Xin_te, batch_size=128, verbose=0) for m in ens]
        )  # (3, n, c)
        ens_mean = ens_preds_probs.mean(0)  # (n, c)
        ens_pred_class = ens_mean.argmax(1)
        ens_correct = ens_pred_class == yte
        # Score = negative entropy of ensemble mean
        ens_score = (ens_mean * np.log(ens_mean + eps)).sum(1)
        ens_gap = get_matched_coverage_gap(ens_score, ens_correct, target_n)
        results['ens_gap'].append(ens_gap)

        print(
            f'Matched Coverage GAPs -> Softmax:{soft_gap:+.3f} | MC-Drop:{mc_gap:+.3f} | Ens:{ens_gap:+.3f}'
        )

        # 6. Noise Test
        rng = np.random.RandomState(seed)
        sig_power = (sig_te**2).mean()
        for snr_db in [np.inf, 20, 10, 5, 0]:
            if np.isinf(snr_db):
                sig_n = sig_te
            else:
                npow = sig_power / (10 ** (snr_db / 10))
                sig_n = sig_te + rng.randn(*sig_te.shape).astype(np.float32) * np.sqrt(npow)

            Xin_n = sig_n[..., None]
            # Use ensemble mode vote to define 'unanimous' exactly like Abhi's template
            ep_class = np.stack(
                [m.predict(Xin_n, batch_size=128, verbose=0).argmax(1) for m in ens]
            )
            v = stats.mode(ep_class, axis=0, keepdims=False).mode
            unan = (ep_class == v).all(0)
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
            results['noise_catch'][snr_db].append(catch)
            s = 'clean' if np.isinf(snr_db) else f'{snr_db}dB'
            print(f'  Noise={s:>5} | catch_rate={catch:.3f}')

    print('\n' + '=' * 60 + '\nFINAL AGGREGATED RESULTS (5 Seeds)\n' + '=' * 60)
    print(
        f'Physics GAP: {np.nanmean(results["phys_gap"]):+.3f} ± {np.nanstd(results["phys_gap"]):.3f}'
    )
    print(
        f'Softmax GAP: {np.nanmean(results["soft_gap"]):+.3f} ± {np.nanstd(results["soft_gap"]):.3f}'
    )
    print(f'MC-Drop GAP: {np.nanmean(results["mc_gap"]):+.3f} ± {np.nanstd(results["mc_gap"]):.3f}')
    print(
        f'Ensemble GAP: {np.nanmean(results["ens_gap"]):+.3f} ± {np.nanstd(results["ens_gap"]):.3f}'
    )

    print("\nNoise Test Catch Rate (Physics catches Ensemble's confident errors):")
    for snr_db in [np.inf, 20, 10, 5, 0]:
        s = 'clean' if np.isinf(snr_db) else f'{snr_db}dB'
        vals = [v for v in results['noise_catch'][snr_db] if not np.isnan(v)]
        m = np.nanmean(vals) if len(vals) > 0 else float('nan')
        std = np.nanstd(vals) if len(vals) > 0 else float('nan')
        print(f'  {s:>6}: {m:.3f} ± {std:.3f}')

    print('================ DONE ================')

except Exception:
    with open('crash_traceback.txt', 'w') as f:
        traceback.print_exc(file=f)
    print('CRASHED. Check crash_traceback.txt')
    sys.exit(1)

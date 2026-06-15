"""
reproduce_results.py - one script that regenerates the paper's tables.

Phases:
  1. Load CWRU (Protocol B, cross-load)
  2. Train the S-JEPA backbone + one canonical CNN
  3. Derive shared test-set signals (preds, confs, feature norms, severities,
     JEPA agreement, U_f, ATE) from that single CNN
  4. Call each evaluation module: classification, baselines, causal, ablation,
     calibration, continual learning

This is a full GPU reproduction (Kaggle/Colab T4 is enough). Numbers should match
the notebook within seed variance. See README for expected values.
"""
import numpy as np
import tensorflow as tf

from data.loaders import load_cwru_all
from core.architecture import build_cnn, train_jepa_backbone, patchify
from core.rules import rule_engine
from core.causal import (analyze_causal, cate_by_group, causal_invariance_across_loads,
                         extract_feature_norms)
from eval.classification import train_cnn, evaluate_protocol_b
from eval.baseline import run_published_baselines, run_irm
from eval.ablation import run_ablation, consensus_scores
from eval.calibration import run_ece, run_proposition1
from continual.experiment import run_continual_comparison

SEV_MAP = {'Low': 1, 'Medium': 2, 'High': 3}

def jepa_agreement(encoder, probe, scaler, preds, X, num_patches=8, batch=64):
    """Mean-pool patch embeddings, run the probe, return 1.0 where it agrees with the CNN."""
    embs = []
    for i in range(0, len(X), batch):
        p = patchify(X[i:i+batch], num_patches)
        e = [encoder(p[:, j], training=False).numpy() for j in range(num_patches)]
        embs.append(np.mean(e, axis=0))
    embs = np.concatenate(embs, axis=0)
    jepa_class = probe.predict(scaler.transform(embs))
    return (preds == jepa_class).astype(float)

def abduct_u_f(feat_norms_tr, load_tr, fault_tr, feat_norms_te, load_te, fault_te):
    """Structural abduction of exogenous noise U_f (cell 24)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(load_tr.reshape(-1, 1))
    Xtr = np.column_stack([feat_norms_tr, sc.transform(load_tr.reshape(-1, 1)).flatten()])
    se2 = LinearRegression().fit(Xtr, fault_tr)
    Xte = np.column_stack([feat_norms_te, sc.transform(load_te.reshape(-1, 1)).flatten()])
    return fault_te - se2.predict(Xte)

def main():
    print('=' * 70)
    print('CNSD - REPRODUCE RESULTS (Protocol B, cross-load)')
    print('=' * 70)

    # ── Phase 1: data ───────────────────────────────────────────────────────
    X_train, y_train, load_train, X_test, y_test, load_test = load_cwru_all()
    print(f'Train {X_train.shape} | Test {X_test.shape}')

    # ── Phase 2: backbone + canonical CNN ───────────────────────────────────
    encoder, probe, scaler = train_jepa_backbone(X_train, y_train, epochs=15)
    cnn = train_cnn(X_train, y_train, seed=42, epochs=30)

    # ── Phase 3: shared per-sample signals (one model, used everywhere) ─────
    probs = cnn.predict(X_test, batch_size=64, verbose=0)
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    feat_norms_te = extract_feature_norms(cnn, X_test)
    feat_norms_tr = extract_feature_norms(cnn, X_train)
    severities = np.array([SEV_MAP.get(rule_engine.get_metadata(int(p), float(n))['severity'], 0)
                           for p, n in zip(preds, feat_norms_te)], dtype=float)
    jepa_agrees = jepa_agreement(encoder, probe, scaler, preds, X_test)

    ate = analyze_causal(feat_norms_tr, y_train, load_train, 'CWRU')['ate']
    cal_median_norm = float(np.median(feat_norms_te))
    risk_midpoint = cal_median_norm * abs(ate)
    u_f = abduct_u_f(feat_norms_tr, load_train, (y_train > 0).astype(int),
                     feat_norms_te, load_test, (y_test > 0).astype(int))

    # ── Phase 4: tables ─────────────────────────────────────────────────────
    print('\n[1/6] CLASSIFICATION (Protocol B)')
    evaluate_protocol_b(X_train, y_train, X_test, y_test)

    print('\n[2/6] PUBLISHED BASELINES')
    run_published_baselines(X_train, y_train, X_test, y_test)
    run_irm(X_train, y_train, load_train, X_test, y_test)

    print('\n[3/6] CAUSAL ANALYSIS')
    res = analyze_causal(feat_norms_tr, y_train, load_train, 'CWRU')
    print(f"ATE={res['ate']:+.4f}  CI=[{res['ci'][0]:+.4f},{res['ci'][1]:+.4f}]  "
          f"placebo={res['placebo_ratio']:.1f}x  p={res['p_value']:.4f}")
    # CATE on logit(P(fault)) over the combined set
    fault_prob = 1.0 - probs[:, 0]
    eps = 1e-6
    fault_logit = np.log((fault_prob + eps) / (1.0 - fault_prob + eps))
    cate_by_group(feat_norms_te, fault_logit, y_test, load_test)
    causal_invariance_across_loads(feat_norms_te, y_test, load_test)

    print('\n[4/6] ABLATION')
    run_ablation(y_test, preds, confs, feat_norms_te, severities, jepa_agrees,
                 u_f, ate, risk_midpoint)

    print('\n[5/6] CALIBRATION + PROPOSITION 1')
    correct = (preds == y_test).astype(int)
    cnsd_scores = consensus_scores(preds, confs, feat_norms_te, severities,
                                   jepa_agrees, u_f, ate, risk_midpoint)
    run_ece(confs, cnsd_scores, correct)
    run_proposition1(feat_norms_te, ate, correct)

    print('\n[6/6] CONTINUAL LEARNING')
    run_continual_comparison(X_train, y_train, X_test, y_test)

    print('\n' + '=' * 70)
    print('Done. Compare printed tables against the README expected values.')
    print('=' * 70)

if __name__ == '__main__':
    main()

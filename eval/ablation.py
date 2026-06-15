import numpy as np

# Consensus weights derived from the calibration set in the notebook (cell 13/26).
DEFAULT_WEIGHTS = {'cnn': 0.05, 'sym': 0.2, 'causal': 0.3, 'cf': 0.15, 'jepa': 0.3}

def consensus_scores(preds, confs, feat_norms, severities, jepa_agrees, u_f, ate,
                     risk_midpoint, weights=DEFAULT_WEIGHTS,
                     dis_caus=False, dis_jepa=False, forward_only=False, alpha=0.3):
    """
    Per-sample CNSD consensus score (faithful port of run_ablation_fast).
      severities : per-sample severity level in {0,1,2,3}
      jepa_agrees: per-sample 1.0 if JEPA agrees with the CNN, else 0.0
      u_f        : per-sample exogenous counterfactual noise (abducted)
      ate        : scalar causal effect (CNN feature-norm estimand)
      risk_midpoint = CAL_MEDIAN_NORM * abs(ate)
    Toggles reproduce each ablation row.
    """
    w = weights
    scores = np.empty(len(preds))
    risk_ema = risk_midpoint
    for i in range(len(preds)):
        conf = float(confs[i])
        norm = float(feat_norms[i])
        sev = float(severities[i])
        risk = 0.0 if dis_caus else norm * ate
        cf_conf = max(0.0, 1.0 - min(abs(float(u_f[i])), 1.0))
        jepa_signal = 1.0 if dis_jepa else float(jepa_agrees[i])

        # Bidirectional EMA risk scaling (forward-only freezes it at the midpoint)
        if not forward_only:
            risk_ema = alpha * abs(risk) + (1 - alpha) * risk_ema
        else:
            risk_ema = risk_midpoint
        ema_ratio = risk_ema / risk_midpoint if risk_midpoint > 0 else 1.0
        bidir_scale = np.clip(1.0 - 0.1 * (ema_ratio - 1.0), 0.7, 1.1)

        c_fwd = (conf * w['cnn'] +
                 (sev / 3.0) * w['sym'] +
                 min(abs(risk), 1.0) * w['causal'] +
                 cf_conf * w['cf'] +
                 jepa_signal * w['jepa'])
        scores[i] = c_fwd * bidir_scale
    return scores

def run_ablation(y_test, preds, confs, feat_norms, severities, jepa_agrees, u_f, ate,
                 risk_midpoint, weights=DEFAULT_WEIGHTS):
    """
    Five-config ablation (Protocol B). Notebook targets:
      Full CNSD (bidir)  acc 0.9909  reliable 0.9236  score 0.7667
      Forward only       acc 0.9909  reliable 0.9100  score 0.7675
      -Causal            acc 0.9909  reliable 0.6587  score 0.5623
      -JEPA              acc 0.9909  reliable 0.9909  score 0.8128
      CNN only           acc 0.9909  reliable 0.8051  score 0.6108
    Accuracy is identical across rows by construction: ablation changes only the
    consensus score, never the CNN's predictions.
    """
    configs = [
        ('Full CNSD (bidir)', {}),
        ('Forward only',      {'forward_only': True}),
        ('-Causal',           {'dis_caus': True}),
        ('-JEPA',             {'dis_jepa': True}),
        ('CNN only',          {'dis_caus': True, 'dis_jepa': True}),
    ]
    correct = (preds == y_test)
    acc = float(correct.mean())
    print('=== ABLATION STUDY (Protocol B) ===')
    print(f'{"Config":<22} {"Acc":>8} {"Reliable":>10} {"Score_mu":>9} {"Score_sd":>9}')
    print('-' * 62)
    results = {}
    for name, kw in configs:
        s = consensus_scores(preds, confs, feat_norms, severities, jepa_agrees,
                             u_f, ate, risk_midpoint, weights, **kw)
        reliable = float((correct & (s > 0.50)).mean())
        results[name] = {'acc': acc, 'reliable': reliable,
                         'score_mean': float(s.mean()), 'score_std': float(s.std())}
        print(f'{name:<22} {acc:>8.4f} {reliable:>10.4f} {s.mean():>9.4f} {s.std():>9.4f}')

    bidir, fwd = results['Full CNSD (bidir)'], results['Forward only']
    print('\n--- Bidirectional vs Forward-Only ---')
    print(f"Score:    {bidir['score_mean']:.4f} vs {fwd['score_mean']:.4f} "
          f"(delta={bidir['score_mean']-fwd['score_mean']:+.4f})")
    print(f"Reliable: {bidir['reliable']:.4f} vs {fwd['reliable']:.4f} "
          f"(delta={bidir['reliable']-fwd['reliable']:+.4f})")
    return results

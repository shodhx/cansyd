"""
causal.py - CNSD causal layer (Pearl Rung 2, honest framing).

The structural model is stated in core/scm.py: there is NO causal arrow from
vibration to fault. The manipulable variable is the operating condition Z, and
the well-posed interventional quantity is the effect of Z on fault manifestation,
estimated by backdoor adjustment and tested for INVARIANCE across conditions.

The functions below estimate backdoor-adjusted associations between a physically
grounded signal descriptor and the (binarised) fault label, controlling for
operating condition, with bootstrap CIs and permutation placebo tests. These are
reported as measurement-purification / partial-association quantities, NOT as
"vibration causes fault" effects. The interventional do(Z) contrast is provided
by intervention_effect_of_condition().
"""
import numpy as np

# ── Treatments ──────────────────────────────────────────────────────────────

def compute_vibration_rms(X):
    """Raw-signal RMS energy."""
    return np.sqrt(np.mean(X.reshape(len(X), -1) ** 2, axis=1))

def feature_norm(features):
    """CNN feature-norm treatment ('vibration_energy' proxy). Learned, run-dependent."""
    return np.linalg.norm(features, axis=1)

def signal_kurtosis(X):
    """
    Signal kurtosis - the classic bearing-fault impulsiveness indicator.
    Scale-invariant (survives the loader's per-segment normalization, unlike RMS)
    and fully deterministic given the signal, so the resulting ATE is reproducible
    across training runs - unlike the CNN feature-norm, which is model-dependent.
    """
    x = X.reshape(len(X), -1)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-8
    z = (x - mu) / sd
    return np.mean(z ** 4, axis=1)

def extract_feature_norms(cnn, X, batch_size=128):
    """Build the penultimate-layer extractor (matching the pipeline) and return norms."""
    import tensorflow as tf
    feat_model = tf.keras.Model(inputs=cnn.input, outputs=cnn.layers[-3].output)
    feats = feat_model.predict(X, batch_size=batch_size, verbose=0)
    return feature_norm(feats)

# ── Backdoor adjustment (OLS) ───────────────────────────────────────────────

def _ols_coef(treatment, confounder, y):
    """ATE = OLS coefficient on treatment, controlling for confounder (backdoor)."""
    n = len(y)
    cols = [np.ones(n), np.asarray(treatment, float)]
    if confounder is not None:
        cols.append(np.asarray(confounder, float).reshape(n, -1))
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, np.asarray(y, float), rcond=None)
    return coef[1]

def bootstrap_ci(treatment, fault, confounder, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(fault)
    ates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c = None if confounder is None else np.asarray(confounder)[idx]
        ates.append(_ols_coef(treatment[idx], c, np.asarray(fault)[idx]))
    return float(np.percentile(ates, 2.5)), float(np.percentile(ates, 97.5))

def placebo_test(treatment, fault, confounder, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    real = _ols_coef(treatment, confounder, fault)
    placebo = np.abs([_ols_coef(treatment, confounder, rng.permutation(fault))
                      for _ in range(n_perm)])
    p_val = float(np.mean(placebo >= abs(real)))
    ratio = abs(real) / (placebo.mean() + 1e-12)
    return p_val, float(ratio)

def analyze_causal(treatment, fault, confounder, domain='CWRU'):
    """
    Backdoor-adjusted ATE with bootstrap CI + permutation placebo test.
      treatment : a physically grounded signal descriptor (e.g. kurtosis, RMS).
      fault     : labels; binarised as (label > 0).
      confounder: operating load / condition.
    Returns the ATE with a bootstrap CI, permutation placebo ratio, and p-value.
    """
    fault = (np.asarray(fault) > 0).astype(int)
    treatment = np.asarray(treatment, float)
    ate = _ols_coef(treatment, confounder, fault)
    ci = bootstrap_ci(treatment, fault, confounder)
    p_val, placebo_ratio = placebo_test(treatment, fault, confounder)
    return {
        'domain': domain, 'ate': float(ate), 'ci': ci, 'p_value': p_val,
        'placebo_ratio': placebo_ratio,
        'treatment_mean': float(treatment.mean()), 'treatment_std': float(treatment.std()),
    }

# ── CATE: per-fault-type heterogeneity ──────────────────────────────────────

def cate_by_group(treatment, outcome, group_labels, confounder, n_boot=500, seed=42, min_n=30):
    """
    Conditional ATE per group. Outcome is logit(P(fault)) to avoid softmax saturation.
    Reports the per-group CATE with a bootstrap CI and a heterogeneity summary.
    """
    rng = np.random.default_rng(seed)
    treatment = np.asarray(treatment, float)
    outcome = np.asarray(outcome, float)
    results = {}
    print('=== CONDITIONAL AVERAGE TREATMENT EFFECT (CATE) ===')
    print(f'{"Fault":>6} {"N":>6} {"CATE":>10} {"95% CI":>22} {"Sig":>5}')
    print('-' * 56)
    for g in sorted(np.unique(group_labels)):
        m = group_labels == g
        if m.sum() < min_n:
            continue
        t, o = treatment[m], outcome[m]
        c = None if confounder is None else np.asarray(confounder)[m]
        if np.std(t) < 1e-9 or np.std(o) < 1e-9:
            continue
        cate = _ols_coef(t, c, o)
        nn = m.sum()
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, nn, nn)
            cc = None if c is None else c[idx]
            boots.append(_ols_coef(t[idx], cc, o[idx]))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        sig = (lo > 0 or hi < 0)
        results[int(g)] = {'cate': float(cate), 'ci': (lo, hi), 'n': int(nn), 'significant': bool(sig)}
        print(f'{int(g):>6} {int(nn):>6} {cate:>10.4f} [{lo:>8.4f},{hi:>8.4f}] {("YES" if sig else "NO"):>5}')
    if results:
        var = float(np.var([r['cate'] for r in results.values()]))
        print(f'\nCATE variance: {var:.6f}  ->  Heterogeneity: {"HIGH" if var > 0.01 else "LOW"}')
    return results

# ── Causal invariance across operating loads ────────────────────────────────

def causal_invariance_across_loads(treatment, fault, loads):
    """
    Per-load ATE via simple OLS (a single load has no confounder variation).
    Reports the per-load ATE and summarises whether the effect direction is
    invariant across operating conditions (magnitude may vary).
    """
    fault = (np.asarray(fault) > 0).astype(int)
    treatment = np.asarray(treatment, float)
    print('=== CAUSAL INVARIANCE ACROSS LOADS ===')
    print(f'{"Load":>6} {"N":>6} {"ATE":>12} {"Fault_rate":>12}')
    print('-' * 40)
    rows = []
    for ld in sorted(np.unique(loads)):
        m = loads == ld
        fb = fault[m]
        if len(np.unique(fb)) < 2:
            print(f'{int(ld):>6} {int(m.sum()):>6} {"DEGEN":>12} {fb.mean():>12.4f}')
            rows.append({'load': int(ld), 'n': int(m.sum()), 'ate': None, 'fault_rate': float(fb.mean())})
            continue
        ate = _ols_coef(treatment[m], None, fb)
        rows.append({'load': int(ld), 'n': int(m.sum()), 'ate': float(ate), 'fault_rate': float(fb.mean())})
        print(f'{int(ld):>6} {int(m.sum()):>6} {ate:>12.4f} {fb.mean():>12.4f}')

    valid = [r['ate'] for r in rows if r['ate'] is not None]
    summary = {}
    if len(valid) >= 2:
        mean, std = float(np.mean(valid)), float(np.std(valid))
        cv = abs(std / mean) if mean else None
        same_dir = all(np.sign(v) == np.sign(valid[0]) for v in valid)
        summary = {'ate_mean': mean, 'ate_std': std, 'ate_cv': cv, 'direction_consistent': bool(same_dir)}
        print(f'\nATE mean={mean:.4f} std={std:.4f} CV={cv:.4f}  '
              f'direction {"INVARIANT" if same_dir else "VARIES"} across loads')
    return rows, summary


# ── Rung 2: interventional effect of operating condition (the honest do-op) ──

def intervention_effect_of_condition(fault, condition, n_perm=1000, seed=42):
    """Estimate the effect of operating condition Z on fault manifestation.

    Z (load/speed) is physically manipulable, so contrasting fault rates across
    conditions is a genuine Rung-2 interventional quantity, E[Y | do(Z=a)] vs
    E[Y | do(Z=b)], identified here under the SCM in core/scm.py (no instrument,
    no backwards arrow). Returns the per-condition fault rate, the max pairwise
    contrast, and a permutation p-value for whether condition affects the rate.
    """
    fault = (np.asarray(fault) > 0).astype(int)
    condition = np.asarray(condition)
    conds = sorted(np.unique(condition))
    rates = {int(c): float(fault[condition == c].mean()) for c in conds}
    vals = list(rates.values())
    contrast = max(vals) - min(vals)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(condition)
        r = [fault[perm == c].mean() for c in conds]
        null.append(max(r) - min(r))
    p_val = float(np.mean(np.asarray(null) >= contrast))
    return {'per_condition_fault_rate': rates, 'max_contrast': float(contrast),
            'p_value': p_val, 'rung': 2}

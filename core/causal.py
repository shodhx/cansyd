import numpy as np

def compute_vibration_rms(X):
    return np.sqrt(np.mean(X.reshape(len(X), -1) ** 2, axis=1))

def _ols_ate(treatment, confounder, y):
    """ATE = OLS coefficient on the treatment, controlling for the confounder.
    This is the backdoor adjustment: regress fault on [treatment, confounder]."""
    n = len(y)
    design = np.column_stack([np.ones(n), treatment, confounder.reshape(n, -1)])
    coef, *_ = np.linalg.lstsq(design, y.astype(float), rcond=None)
    return coef[1]

def backdoor_ate(X_signal, y_binary, confounder):
    treatment = compute_vibration_rms(X_signal)
    return _ols_ate(treatment, confounder, y_binary), treatment

def bootstrap_ci(X_signal, y_binary, confounder, n_boot=200):
    treatment = compute_vibration_rms(X_signal)
    n = len(y_binary)
    ates = [_ols_ate(treatment[idx], confounder[idx], y_binary[idx])
            for idx in (np.random.choice(n, n, replace=True) for _ in range(n_boot))]
    return np.percentile(ates, 2.5), np.percentile(ates, 97.5)

def placebo_test(X_signal, y_binary, confounder, n_perm=200):
    treatment = compute_vibration_rms(X_signal)
    real_ate = _ols_ate(treatment, confounder, y_binary)
    placebo = np.abs([_ols_ate(treatment, confounder, np.random.permutation(y_binary))
                      for _ in range(n_perm)])
    p_val = float(np.mean(placebo >= abs(real_ate)))
    ratio = abs(real_ate) / (placebo.mean() + 1e-8)
    return p_val, ratio

def analyze_causal(X_train, y_train, load_train, X_test, y_test, load_test, domain='CWRU'):
    fault = (y_train > 0).astype(int)
    ate, treatment = backdoor_ate(X_train, fault, load_train)
    ci_low, ci_high = bootstrap_ci(X_train, fault, load_train)
    p_val, placebo_ratio = placebo_test(X_train, fault, load_train)
    return {
        'domain': domain,
        'ate': ate,
        'ci': (ci_low, ci_high),
        'p_value': p_val,
        'placebo_ratio': placebo_ratio,
        'treatment': 'vibration_rms',
        'treatment_mean': float(treatment.mean()),
        'treatment_std': float(treatment.std()),
    }

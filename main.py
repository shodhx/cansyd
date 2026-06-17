# main.py - CNSD rotating-machinery treatment-reproducibility experiment.
# Runs the pre-committed treatment comparison across three vibration datasets:
# CWRU bearings, JNU bearings, SEU gears. The condition variable (load/speed)
# is the axis across which causal invariance is tested.
import numpy as np
from scipy.stats import skew


# ---- physically grounded treatments (deterministic, per-segment) ----
def t_kurtosis(X):
    x = X.reshape(len(X), -1)
    z = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    return np.mean(z ** 4, axis=1)

def t_rms(X):
    x = X.reshape(len(X), -1); return np.sqrt(np.mean(x ** 2, axis=1))

def t_skewness(X):
    return skew(X.reshape(len(X), -1), axis=1)

def t_crest(X):
    x = X.reshape(len(X), -1)
    return np.max(np.abs(x), axis=1) / (np.sqrt(np.mean(x ** 2, axis=1)) + 1e-8)

def t_entropy(X):
    x = X.reshape(len(X), -1)
    ps = np.abs(np.fft.rfft(x, axis=1)) ** 2
    ps = ps / (ps.sum(axis=1, keepdims=True) + 1e-12)
    return -np.sum(ps * np.log(ps + 1e-12), axis=1)

TREATMENTS = {'Kurtosis': t_kurtosis, 'RMS': t_rms, 'Skewness': t_skewness,
              'Crest factor': t_crest, 'Spectral entropy': t_entropy}


def _standardize(v):
    v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-12)


def run_comparison(X, y, cond, name):
    """Pre-committed treatment comparison on one dataset. Reports every treatment."""
    from core.causal import analyze_causal, causal_invariance_across_loads
    print('\n' + '=' * 72)
    print(f'TREATMENT COMPARISON - {name} (condition = operating load/speed)')
    print('=' * 72)
    print(f'Samples: {len(X)} across conditions {sorted(set(cond.tolist()))}')
    print(f"{'Treatment':18}{'ATE/SD':>10}{'p':>8}{'placebo':>9}{'CV(cond)':>10} direction")
    print('-' * 72)
    for tname, fn in TREATMENTS.items():
        t = _standardize(fn(X))
        r = analyze_causal(t, y, cond, name)
        _, summary = causal_invariance_across_loads(t, y, cond)
        cv = summary.get('ate_cv')
        direction = 'invariant' if summary.get('direction_consistent') else 'FLIPS'
        cv_str = f"{cv:10.3f}" if cv is not None else f"{'n/a':>10}"
        print(f"{tname:18}{r['ate']:+10.4f}{r['p_value']:8.4f}"
              f"{r['placebo_ratio']:8.1f}x{cv_str} {direction}")


def main():
    np.random.seed(42)
    from data.loaders import load_cwru_all, load_jnu_all, load_seu_gear_all

    print('=' * 72)
    print('   CNSD: PHYSICALLY-GROUNDED CAUSAL TREATMENT REPRODUCIBILITY')
    print('   across rotating-machinery vibration datasets')
    print('=' * 72)

    # each loader returns (X_tr, y_tr, cond_tr, X_te, y_te, cond_te); the
    # treatment comparison uses the combined data across all conditions.
    datasets = [
        ('CWRU (bearings)', load_cwru_all),
        ('JNU (bearings)', load_jnu_all),
        ('SEU (gears)', load_seu_gear_all),
    ]
    for name, loader in datasets:
        try:
            X0, y0, c0, X1, y1, c1 = loader()
            X = np.concatenate([X0, X1]); y = np.concatenate([y0, y1])
            cond = np.concatenate([c0, c1])
            run_comparison(X, y, cond, name)
        except (FileNotFoundError, RuntimeError) as e:
            print(f'\n[{name}] skipped: {e}')


if __name__ == '__main__':
    main()

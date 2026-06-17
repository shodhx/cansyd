import numpy as np
from scipy.stats import skew
from data.loaders import load_cwru_all
from core.causal import analyze_causal, causal_invariance_across_loads


# ----------------------------------------------------------------------
# Physically-motivated treatments (all per-segment, deterministic).
# Each maps X (n, 1024, 1) -> (n,) treatment value.
# These are FIXED in advance; we report all of them.
# ----------------------------------------------------------------------
def t_kurtosis(X):
    x = X.reshape(len(X), -1)
    z = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    return np.mean(z ** 4, axis=1)

def t_rms(X):
    x = X.reshape(len(X), -1)
    return np.sqrt(np.mean(x ** 2, axis=1))

def t_skewness(X):
    x = X.reshape(len(X), -1)
    return skew(x, axis=1)

def t_crest_factor(X):
    x = X.reshape(len(X), -1)
    peak = np.max(np.abs(x), axis=1)
    rms = np.sqrt(np.mean(x ** 2, axis=1)) + 1e-8
    return peak / rms

def t_spectral_entropy(X):
    x = X.reshape(len(X), -1)
    # power spectral density via FFT, normalised to a probability distribution
    ps = np.abs(np.fft.rfft(x, axis=1)) ** 2
    ps = ps / (ps.sum(axis=1, keepdims=True) + 1e-12)
    return -np.sum(ps * np.log(ps + 1e-12), axis=1)


TREATMENTS = {
    'Kurtosis':         t_kurtosis,
    'RMS':              t_rms,
    'Skewness':         t_skewness,
    'Crest factor':     t_crest_factor,
    'Spectral entropy': t_spectral_entropy,
}


def main():
    print('=' * 70)
    print('TREATMENT COMPARISON (pre-committed; all treatments reported)')
    print('=' * 70)

    # combined all-loads data (same source as the invariance analysis)
    X0, y0, l0, X1, y1, l1 = load_cwru_all()
    X = np.concatenate([X0, X1]); y = np.concatenate([y0, y1])
    loads = np.concatenate([l0, l1])
    print(f'Samples: {len(X)} across loads {sorted(set(loads.tolist()))}\n')

    # ---- Table 1: ATE + cross-load stability (CV) for every treatment ----
    print(f"{'Treatment':18} {'ATE':>10} {'p':>8} {'placebo':>9} "
          f"{'CV(load)':>9} {'direction':>10}")
    print('-' * 70)
    for name, fn in TREATMENTS.items():
        t = fn(X)
        res = analyze_causal(t, y, loads, 'CWRU')
        rows, summary = causal_invariance_across_loads(t, y, loads)
        cv = summary.get('ate_cv')
        direction = 'invariant' if summary.get('direction_consistent') else 'FLIPS'
        cv_str = f"{cv:9.3f}" if cv is not None else f"{'n/a':>9}"
        print(f"{name:18} {res['ate']:+10.4f} {res['p_value']:8.4f} "
              f"{res['placebo_ratio']:8.1f}x {cv_str} {direction:>10}")
    print('\n(Report this table verbatim. The most load-stable treatment is the')
    print(' one with smallest CV AND invariant direction - whatever it turns out')
    print(' to be. Do not reorder to favour a chosen treatment.)\n')

    # ---- Table 2: sensitivity of the kurtosis result ----
    # (kurtosis is the pre-registered primary treatment; we stress-test IT)
    print('=' * 70)
    print('SENSITIVITY ANALYSIS (primary treatment = kurtosis)')
    print('=' * 70)
    t = t_kurtosis(X)

    # (a) with confounder adjustment (backdoor) vs (b) without (naive)
    full = analyze_causal(t, y, loads, 'CWRU')
    naive = analyze_causal(t, y, np.zeros_like(loads), 'CWRU')  # confounder removed
    print(f"  Backdoor-adjusted ATE : {full['ate']:+.4f}  "
          f"CI {full['ci']}  p={full['p_value']:.4f}")
    print(f"  Confounder-removed ATE: {naive['ate']:+.4f}  "
          f"CI {naive['ci']}  p={naive['p_value']:.4f}")
    print("  (If these differ, the confounder matters - report both honestly.)\n")

    # (c) alternate load splits: leave-one-load-out ATE
    print("  Leave-one-load-out ATE (robustness to which load is held out):")
    for held in sorted(set(loads.tolist())):
        m = loads != held
        r = analyze_causal(t[m], y[m], loads[m], 'CWRU')
        print(f"    exclude load {held}: ATE={r['ate']:+.4f}  p={r['p_value']:.4f}")
    print("\n  (Stable ATE across these exclusions = robust. Report all rows.)")


if __name__ == '__main__':
    main()

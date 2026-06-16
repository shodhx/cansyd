import numpy as np
from .metrics import proposition_a_test

def expected_calibration_error(confidences, accuracies, n_bins=10):
    """General ECE: works for softmax confidences or arbitrary consensus scores.
    `accuracies` is the per-sample correctness vector (0/1)."""
    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() / len(confidences) * abs(confidences[m].mean() - accuracies[m].mean())
    return float(ece)

def run_ece(cnn_confs, cnsd_scores, correct):
    """
    Expected: CNN softmax ECE=0.0015, CNSD bidirectional ECE=0.2242 (acc 0.9909).
    The CNSD consensus score is worse-calibrated than the raw softmax.
    """
    correct = np.asarray(correct, dtype=float)
    ece_cnn = expected_calibration_error(cnn_confs, correct)
    ece_cnsd = expected_calibration_error(cnsd_scores, correct)
    acc = float(correct.mean())
    print('=== EXPECTED CALIBRATION ERROR (ECE) ===')
    print(f'{"Method":<28} {"ECE":>8} {"Accuracy":>10}')
    print('-' * 50)
    print(f'{"CNN softmax only":<28} {ece_cnn:>8.4f} {acc:>10.4f}')
    print(f'{"CNSD bidirectional":<28} {ece_cnsd:>8.4f} {acc:>10.4f}')
    return {'cnn': ece_cnn, 'cnsd': ece_cnsd, 'accuracy': acc}

def run_proposition1(feat_norms, ate, correct):
    """
    Proposition 1 check with a direction-invariant risk measure:
        risk = |abs(norm*ate) - median(abs(norm*ate))|
    Expected: VIOLATED, rho=0.0356, p=0.1625. A stated limitation: causal
    contrastive features don't make extreme risk predict error here.
    """
    feat_norms = np.asarray(feat_norms, dtype=float)
    correct = np.asarray(correct, dtype=int)
    risk_raw = np.abs(feat_norms * ate)
    risks = np.abs(risk_raw - np.median(risk_raw))
    errors = (1 - correct).astype(float)

    rho, p, monotonic, quartile_errs = proposition_a_test(risks, errors)

    print('=== PROPOSITION 1: CONDITION VERIFICATION ===')
    print('Risk measure: |risk - median(risk)| (direction-invariant)\n')
    print(f'{"Quartile":<14} {"Error Rate":>12}')
    print('-' * 28)
    for lbl, e in zip(['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'], quartile_errs):
        print(f'{lbl:<14} {e:>12.4f}')
    print(f'\nStrict monotonicity: {"SATISFIED" if monotonic else "VIOLATED"}')
    print(f'Spearman (risk deviation, error): rho={rho:.4f}, p={p:.4f}')
    if monotonic and rho > 0 and p < 0.05:
        print('Condition 1 SATISFIED: Proposition A empirically supported.')
    elif rho > 0 and p < 0.05:
        print('Condition 1 partially satisfied (weaker sufficient condition).')
    else:
        print('Condition 1 NOT supported on this dataset - stated as a limitation:')
        print('extreme risk deviations do not predict classification error here.')
        print('The bidirectional mechanism affects ECE via EMA smoothing, not monotonicity.')
    return {'rho': float(rho), 'p': float(p), 'monotonic': bool(monotonic),
            'quartile_errors': [float(x) for x in quartile_errs]}

import numpy as np
from scipy import stats

def compute_ece(y_true, y_pred_proba, n_bins=10):
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            bin_weight = np.sum(mask) / len(y_true)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return ece

def proposition_a_test(causal_risk, errors):
    valid_idx = ~np.isnan(causal_risk) & ~np.isnan(errors)
    rho, p = stats.spearmanr(causal_risk[valid_idx], errors[valid_idx])

    quartiles = np.percentile(causal_risk[valid_idx], [25, 50, 75])
    q1_err = np.mean(errors[valid_idx][causal_risk[valid_idx] <= quartiles[0]])
    q2_err = np.mean(errors[valid_idx][(causal_risk[valid_idx] > quartiles[0]) & (causal_risk[valid_idx] <= quartiles[1])])
    q3_err = np.mean(errors[valid_idx][(causal_risk[valid_idx] > quartiles[1]) & (causal_risk[valid_idx] <= quartiles[2])])
    q4_err = np.mean(errors[valid_idx][causal_risk[valid_idx] > quartiles[2]])

    monotonic = (q1_err <= q2_err <= q3_err <= q4_err)

    return rho, p, monotonic, [q1_err, q2_err, q3_err, q4_err]

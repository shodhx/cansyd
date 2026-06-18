"""
counterfactual.py - DEPRECATED Rung-3 framing; delegates to honest sensitivity.

This module previously claimed to perform Pearl's Rung-3 abduction-action-
prediction. That was an overclaim (residual extraction under a linear model is
not structural abduction). It now delegates to core.sensitivity.local_sensitivity
and is kept only so existing callers do not break. New code should call
local_sensitivity directly and describe it as a local sensitivity analysis.
"""
from core.sensitivity import local_sensitivity


def generate_counterfactual(X_sample, y_actual, load_actual, load_counterfactual,
                            structural_coefficients=None):
    """Backward-compatible shim. Returns a local sensitivity result, NOT a
    Pearl-type counterfactual. The keys mirror the old structure where sensible
    but the framing is corrected.
    """
    slope = (structural_coefficients or {}).get('beta', 0.8)
    coupling = (structural_coefficients or {}).get('alpha', 0.05)
    s = local_sensitivity(X_sample, load_actual, load_counterfactual,
                          response_slope=slope, condition_coupling=coupling)
    return {
        'actual': {'load': float(load_actual), 'predicted_prob': s['predicted_prob']},
        'perturbed': {'load': float(load_counterfactual),
                      'predicted_prob': s['perturbed_prob'],
                      'prob_change': s['prob_change']},
        'sensitivity': s['sensitivity'],
        'framing': 'local sensitivity analysis (NOT Pearl Rung 3)',
        'explanation': (
            f"Under a linear response assumption, perturbing operating condition "
            f"from {load_actual} to {load_counterfactual} changes the predicted "
            f"fault probability by {s['prob_change']:+.4f} (sensitivity "
            f"{s['sensitivity']:.3f}). This is a robustness probe, not a "
            f"structural counterfactual."
        ),
    }

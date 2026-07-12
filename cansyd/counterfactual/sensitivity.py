"""
Local sensitivity analysis (NOT a Pearl Rung-3 counterfactual).

This module computes a LOCAL
SENSITIVITY ANALYSIS. For a given sample it asks how much the model's predicted
fault probability changes under a perturbation of the operating condition, under
an explicit linear response assumption. Samples whose prediction is highly
sensitive to small perturbations are the fragile ones - useful to flag - but the
output is a robustness diagnostic, not a counterfactual claim.
"""

import numpy as np

from cnsd.causal.estimators import compute_vibration_rms


def local_sensitivity(
    X_sample, condition_actual, condition_perturbed, response_slope=0.8, condition_coupling=0.05
):
    """How sensitive is the predicted fault response to a change in operating
    condition, for THIS sample, under a linear response assumption.

    Returns the predicted-probability change and a sensitivity magnitude. The
    assumptions (linear coupling, fixed slope) are stated, not hidden; this is a
    Rung-1/2 robustness probe, explicitly NOT a structural counterfactual.
    """
    vibration = float(compute_vibration_rms(np.asarray(X_sample).reshape(1, -1))[0])

    # linear response model (stated assumption)
    def prob(v):
        return 1.0 / (1.0 + np.exp(-response_slope * v))

    # perturb the measurement by the modelled condition coupling
    dv = condition_coupling * (condition_perturbed - condition_actual)
    p0, p1 = prob(vibration), prob(vibration + dv)
    return {
        'predicted_prob': p0,
        'perturbed_prob': p1,
        'prob_change': float(p1 - p0),
        'sensitivity': float(abs(p1 - p0) / (abs(dv) + 1e-9)),
        'assumption': 'linear response; local sensitivity estimate counterfactual',
        'note': 'flags samples whose prediction is fragile to operating-condition perturbation',
    }

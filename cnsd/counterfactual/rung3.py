"""
Pearl Rung-3 counterfactuals via DoWhy gcm.

  1. ABDUCTION : recover the EXACT exogenous noise for the observed unit by
                 inverting the fitted structural mechanisms (only possible with
                 an invertible SCM - this is what makes it real Rung 3, not a
                 residual approximation).
  2. ACTION    : apply the atomic intervention do(Z := z') to the SCM.
  3. PREDICTION: propagate downstream with the recovered noise to get the
                 unit-level counterfactual outcome.

"""

import numpy as np


def dowhy_gcm_available():
    try:
        from dowhy import gcm  # noqa: F401

        return True
    except Exception:
        return False


def build_scm(condition, signal_feature, degradation_outcome):
        """ Z (condition) -> X (signal feature) -> Y (degradation outcome)
        Z -> Y
    Y must be a CONTINUOUS degradation quantity (e.g. vibration RMS), not a
    binary fault label - a binary Y collapses unit-level counterfactuals to
    flips and cannot express gradual, direction-sensible deltas.
    """
    if not dowhy_gcm_available():
        return None
    import networkx as nx
    import pandas as pd
    from dowhy import gcm

    df = pd.DataFrame(
        {
            'Z': np.asarray(condition, float),
            'X': np.asarray(signal_feature, float),
            'Y': np.asarray(degradation_outcome, float),
        }
    )
    # operational graph over the measured descriptor X (condition Z -> descriptor
    # X -> outcome Y); used to fit mechanisms for counterfactual queries on Z.
    graph = nx.DiGraph([('Z', 'X'), ('X', 'Y'), ('Z', 'Y')])
    scm = gcm.InvertibleStructuralCausalModel(graph)
    gcm.auto.assign_causal_mechanisms(scm, df)
    gcm.fit(scm, df)
    return scm


def counterfactual_for_unit(scm, observed_row, condition_cf):
    """Real Rung-3 counterfactual for ONE observed unit.

    observed_row : dict with the unit's factual {'Z','X','Y'}
    condition_cf : the counterfactual operating condition (do(Z := condition_cf))
    Returns the factual vs counterfactual degradation severity (vibration RMS)
    for this unit - a gradual, direction-sensible delta, not a fault flip.
    """
    import pandas as pd
    from dowhy import gcm

    observed = pd.DataFrame([observed_row])
    cf = gcm.counterfactual_samples(scm, {'Z': lambda z, v=condition_cf: v}, observed_data=observed)
    return {
        'factual': {'Z': float(observed_row['Z']), 'Y': float(observed_row['Y'])},
        'counterfactual': {'Z': float(condition_cf), 'Y': float(cf['Y'].iloc[0])},
        'delta_Y': float(cf['Y'].iloc[0] - observed_row['Y']),
        'method': 'structural_counterfactual',
    }


def what_if(
    signal_feature_value, condition_actual, condition_cf, scm=None, X_sample=None, factual_y=0.0
):
    """Counterfactual query for one unit.

    Returns a structural counterfactual (DoWhy gcm) when a fitted SCM is provided
    and DoWhy is installed; otherwise returns a local sensitivity estimate, with
    the 'method' field indicating which was used.
    """
    if scm is not None and dowhy_gcm_available():
        row = {
            'Z': float(condition_actual),
            'X': float(signal_feature_value),
            'Y': float(factual_y),
        }  # Y filled by abduction; requires factual observation
        # abduction recovers noise from Z,X,Y; Y is recomputed counterfactually
        return counterfactual_for_unit(scm, row, condition_cf)

    from cnsd.counterfactual.sensitivity import local_sensitivity

    if X_sample is None:
        X_sample = np.array([signal_feature_value])
    s = local_sensitivity(X_sample, condition_actual, condition_cf)
    return {
        'factual': {'Z': float(condition_actual)},
        'counterfactual': {'Z': float(condition_cf), 'prob_change': s['prob_change']},
        'method': 'local_sensitivity',
        'note': 'install dowhy for structural counterfactuals',
    }

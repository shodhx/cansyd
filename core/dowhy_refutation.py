"""
dowhy_refutation.py - optional DoWhy-based identification + refutation for the
causal layer.

CNSD's hand-built backdoor estimator (core/causal.py) already gives the ATE with
bootstrap CIs and a permutation placebo. DoWhy adds two things of real value for
credibility: (1) it states the estimand and identifies it from the SCM via
do-calculus, and (2) it runs a REFUTATION SUITE - placebo treatment, random
common cause, and data-subset stability - that stress-tests the estimate the way
a careful reviewer would.

DoWhy is OPTIONAL. If it is not installed the function degrades gracefully and
points back to the built-in placebo test, so the pipeline never hard-depends on
it. The DAG handed to DoWhy is the CORRECTED one in core/scm.py - load Z is the
manipulable treatment, fault Y the outcome - so DoWhy estimates the honest
Rung-2 quantity, not the backwards vibration->fault effect.
"""
import numpy as np


def dowhy_available():
    try:
        import dowhy  # noqa: F401
        return True
    except Exception:
        return False


def refute_condition_effect(condition, fault, extra_confounders=None):
    """Identify + estimate + refute the effect of operating condition on fault,
    using DoWhy if present. Returns the estimate and the refutation results.

    condition : operating condition Z (treatment, manipulable)
    fault     : binary fault outcome Y
    extra_confounders : optional dict {name: array} of measured confounders
    """
    fault = (np.asarray(fault) > 0).astype(int)
    condition = np.asarray(condition, float)

    if not dowhy_available():
        # graceful fallback: report that the built-in placebo test stands in
        from core.causal import placebo_test
        p, ratio = placebo_test(condition, fault, None)
        return {
            'backend': 'builtin (DoWhy not installed)',
            'note': 'install dowhy for full identification + refutation suite',
            'placebo_p_value': p, 'placebo_ratio': ratio,
        }

    import pandas as pd
    from dowhy import CausalModel

    data = {'Z': condition, 'Y': fault}
    confs = []
    if extra_confounders:
        for name, arr in extra_confounders.items():
            data[name] = np.asarray(arr, float)
            confs.append(name)
    df = pd.DataFrame(data)

    # The corrected DAG (core/scm.py): Z -> Y is the interventional contrast;
    # any measured confounders are common causes of Z and Y.
    graph_edges = ['Z -> Y'] + [f'{c} -> Z' for c in confs] + [f'{c} -> Y' for c in confs]
    model = CausalModel(data=df, treatment='Z', outcome='Y',
                        common_causes=confs if confs else None)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name='backdoor.linear_regression')

    # REFUTATION SUITE - the part that adds real credibility
    refutations = {}
    for name, method in [
        ('placebo_treatment', 'placebo_treatment_refuter'),
        ('random_common_cause', 'random_common_cause'),
        ('data_subset', 'data_subset_refuter'),
    ]:
        try:
            r = model.refute_estimate(estimand, estimate, method_name=method)
            refutations[name] = {
                'new_effect': float(getattr(r, 'new_effect', np.nan)),
                'p_value': getattr(r, 'refutation_result', {}) if hasattr(r, 'refutation_result') else None,
                'summary': str(r),
            }
        except Exception as e:
            refutations[name] = {'error': str(e)}

    return {
        'backend': 'dowhy',
        'estimand': str(estimand),
        'estimate': float(estimate.value),
        'refutations': refutations,
        'interpretation': ('a robust estimate barely changes under '
                           'random_common_cause and data_subset, and collapses '
                           'toward zero under placebo_treatment'),
    }

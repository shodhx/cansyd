"""Causal layer (Pearl Rung 2): backdoor-adjusted do(Z) intervention on operating
condition, invariance testing, and an optional DoWhy refutation suite. The
corrected DAG lives in cnsd.scm."""

from cnsd.causal.estimators import (
    analyze_causal,
    cate_by_group,
    causal_invariance_across_loads,
    compute_vibration_rms,
    extract_feature_norms,
    intervention_effect_of_condition,
    signal_kurtosis,
)
from cnsd.causal.refutation import dowhy_available, refute_condition_effect

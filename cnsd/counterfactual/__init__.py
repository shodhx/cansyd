"""Counterfactual layer (Pearl Rung 3): unit-level counterfactuals via an
invertible SCM (DoWhy gcm abduction-action-prediction). Falls back to honest
local sensitivity (clearly labelled NOT Rung 3) when DoWhy is unavailable."""
from cnsd.counterfactual.rung3 import (
    build_scm, counterfactual_for_unit, what_if, dowhy_gcm_available,
)
from cnsd.counterfactual.sensitivity import local_sensitivity

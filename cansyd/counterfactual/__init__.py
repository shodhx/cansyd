"""Counterfactual layer (Pearl Rung 3): unit-level counterfactuals via an
invertible SCM (DoWhy gcm). Falls back to a local sensitivity estimate when
DoWhy is unavailable; the result's 'method' field indicates which was used."""

from cnsd.counterfactual.rung3 import (
    build_scm,
    counterfactual_for_unit,
    dowhy_gcm_available,
    what_if,
)
from cnsd.counterfactual.sensitivity import local_sensitivity

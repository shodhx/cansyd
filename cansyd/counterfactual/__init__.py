"""Counterfactual layer (Pearl Rung 3): unit-level counterfactuals via an
invertible SCM (DoWhy gcm). Falls back to a local sensitivity estimate when
DoWhy is unavailable; the result's 'method' field indicates which was used."""

from cansyd.counterfactual.rung3 import (
    build_scm,
    counterfactual_for_unit,
    dowhy_gcm_available,
    what_if,
)
from cansyd.counterfactual.sensitivity import local_sensitivity

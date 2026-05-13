import numpy as np
import pandas as pd
from dowhy import CausalModel

def generate_counterfactual(X_sample, y_actual, load_actual, load_counterfactual, ate):
    from core.causal import compute_vibration_rms
    
    vibration = compute_vibration_rms(X_sample.reshape(1, -1))[0]
    
    df = pd.DataFrame({
        'vibration_rms': [vibration],
        'load': [load_actual],
        'fault': [int(y_actual > 0)]
    })
    
    model = CausalModel(
        data=df,
        treatment='vibration_rms',
        outcome='fault',
        common_causes=['load']
    )
    
    identified = model.identify_effect()
    
    load_effect = ate * (load_counterfactual - load_actual) * 0.1
    
    cf_result = {
        'actual': {
            'load': load_actual,
            'vibration_rms': vibration,
            'fault': int(y_actual > 0)
        },
        'counterfactual': {
            'load': load_counterfactual,
            'vibration_rms': vibration,
            'estimated_fault_prob_change': load_effect
        },
        'explanation': f"If load were {load_counterfactual} instead of {load_actual}, "
                      f"fault probability would change by approximately {load_effect:.3f}"
    }
    
    return cf_result
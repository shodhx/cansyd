import numpy as np
from core.causal import compute_vibration_rms

def generate_counterfactual(X_sample, y_actual, load_actual, load_counterfactual, structural_coefficients=None):
    """
    Executes structural Abduction-Action-Prediction (Pearl's Ladder Rung 3)
    to compute individual fault probability changes under counterfactual load interventions.
    """
    if structural_coefficients is None:
        # Synced with your notebook's validated structural coefficients
        structural_coefficients = {'alpha': 0.05, 'beta': 0.8}
        
    vibration = compute_vibration_rms(X_sample.reshape(1, -1))[0]
    actual_fault = int(y_actual > 0)
    
    # --- STEP 1: ABDUCTION ---
    # Isolate unique background structural noise (U) for this specific machine unit
    u_vibration = vibration - (structural_coefficients['alpha'] * load_actual)
    
    # --- STEP 2: ACTION ---
    # Apply do-calculus intervention: force load to counterfactual target
    cf_vibration = (structural_coefficients['alpha'] * load_counterfactual) + u_vibration
    
    # --- STEP 3: PREDICTION ---
    # Calculate causal risk scaling through your notebook's sigmoid mapper
    actual_latent_score = structural_coefficients['beta'] * vibration
    actual_fault_prob = 1.0 / (1.0 + np.exp(-actual_latent_score))
    
    cf_latent_score = structural_coefficients['beta'] * cf_vibration
    cf_fault_prob = 1.0 / (1.0 + np.exp(-cf_latent_score))
    
    prob_change = cf_fault_prob - actual_fault_prob
    
    return {
        'actual': {
            'load': float(load_actual),
            'vibration_rms': float(vibration),
            'fault': actual_fault,
            'fault_prob': float(actual_fault_prob)
        },
        'counterfactual': {
            'load': float(load_counterfactual),
            'vibration_rms': float(cf_vibration),
            'estimated_fault_prob_change': float(prob_change),
            'cf_fault_prob': float(cf_fault_prob)
        },
        'explanation': (
            f"If operational load were altered to {load_counterfactual} instead of {load_actual}, "
            f"structural vibration would adjust to {cf_vibration:.3f}, shifting "
            f"individual fault probability by {prob_change:+.4f}."
        )
    }
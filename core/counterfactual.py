import numpy as np

def generate_counterfactual(X_sample, y_actual, load_actual, load_counterfactual, structural_coefficients=None):
    from core.causal import compute_vibration_rms
    
    # Default structural weights if not provided via config
    if structural_coefficients is None:
        structural_coefficients = {'alpha': 0.05, 'beta': 0.8, 'threshold': 1.5}
        
    vibration = compute_vibration_rms(X_sample.reshape(1, -1))[0]
    actual_fault = int(y_actual > 0)
    
    # --- STEP 1: ABDUCTION ---
    # Back-calculate the exogenous structural noise (U) for this specific unit sample
    # Assuming structural equation: vibration_rms = alpha * load + U_vibration
    u_vibration = vibration - (structural_coefficients['alpha'] * load_actual)
    
    # --- STEP 2: ACTION (Intervention via Do-Calculus) ---
    # Counterfactual intervention: set load = load_counterfactual
    cf_vibration = (structural_coefficients['alpha'] * load_counterfactual) + u_vibration
    
    # --- STEP 3: PREDICTION ---
    # Evaluate fault probability using the structural equation under counterfactual conditions
    cf_latent_score = structural_coefficients['beta'] * cf_vibration
    cf_fault_prob = 1.0 / (1.0 + np.exp(-cf_latent_score)) # Logistic response function
    
    # Compute actual latent score baseline for comparison
    actual_latent_score = structural_coefficients['beta'] * vibration
    actual_fault_prob = 1.0 / (1.0 + np.exp(-actual_latent_score))
    
    prob_change = cf_fault_prob - actual_fault_prob
    
    return {
        'actual': {
            'load': load_actual,
            'vibration_rms': vibration,
            'fault': actual_fault
        },
        'counterfactual': {
            'load': load_counterfactual,
            'vibration_rms': cf_vibration,
            'estimated_fault_prob_change': prob_change
        },
        'explanation': (
            f"If load were {load_counterfactual} instead of {load_actual}, "
            f"counterfactual vibration would adjust to {cf_vibration:.3f}, changing "
            f"the fault risk probability by {prob_change:+.3f}."
        )
    }
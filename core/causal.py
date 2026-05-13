import numpy as np
import pandas as pd
from dowhy import CausalModel
from sklearn.linear_model import LinearRegression
from scipy import stats

def compute_vibration_rms(X):
    return np.sqrt(np.mean(X.reshape(len(X), -1)**2, axis=1))

def backdoor_ate_dowhy(X_signal, y_binary, confounder, domain='CWRU'):
    treatment = compute_vibration_rms(X_signal)
    
    df = pd.DataFrame({
        'vibration_rms': treatment,
        'load': confounder.flatten(),
        'fault': y_binary.astype(int)
    })
    
    model = CausalModel(
        data=df,
        treatment='vibration_rms',
        outcome='fault',
        common_causes=['load']
    )
    
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name='backdoor.linear_regression'
    )
    
    ate = estimate.value
    return ate, treatment, df

def bootstrap_ci(X_signal, y_binary, confounder, n_boot=1000):
    ates = []
    n = len(X_signal)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        ate, _, _ = backdoor_ate_dowhy(X_signal[idx], y_binary[idx], confounder[idx])
        ates.append(ate)
    return np.percentile(ates, 2.5), np.percentile(ates, 97.5)

def placebo_test(X_signal, y_binary, confounder, n_perm=1000):
    real_ate, _, _ = backdoor_ate_dowhy(X_signal, y_binary, confounder)
    placebo_ates = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y_binary)
        ate, _, _ = backdoor_ate_dowhy(X_signal, y_perm, confounder)
        placebo_ates.append(ate)
    p_val = np.mean(np.abs(placebo_ates) >= np.abs(real_ate))
    ratio = np.abs(real_ate) / (np.mean(np.abs(placebo_ates)) + 1e-8)
    return p_val, ratio

def counterfactual_scenario(X_sample, load_current, load_cf, ate_estimate):
    vibration = compute_vibration_rms(X_sample.reshape(1, -1))[0]
    fault_prob_current = vibration * ate_estimate
    
    df_cf = pd.DataFrame({
        'vibration_rms': [vibration],
        'load': [load_cf],
        'fault': [0]
    })
    
    model = CausalModel(
        data=df_cf,
        treatment='vibration_rms',
        outcome='fault',
        common_causes=['load']
    )
    
    identified = model.identify_effect()
    
    cf_explanation = {
        'actual_load': load_current,
        'cf_load': load_cf,
        'vibration_rms': vibration,
        'estimated_effect': ate_estimate,
        'interpretation': f"Under load={load_cf}, estimated fault probability would differ by {ate_estimate*(load_cf-load_current):.3f}"
    }
    
    return cf_explanation

def analyze_causal(X_train, y_train, load_train, X_test, y_test, load_test, domain='CWRU'):
    fault_train = (y_train > 0).astype(int)
    
    ate, treatment_train, df_train = backdoor_ate_dowhy(X_train, fault_train, load_train, domain)
    ci_low, ci_high = bootstrap_ci(X_train, fault_train, load_train)
    p_val, placebo_ratio = placebo_test(X_train, fault_train, load_train)
    
    results = {
        'domain': domain,
        'ate': ate,
        'ci': (ci_low, ci_high),
        'p_value': p_val,
        'placebo_ratio': placebo_ratio,
        'treatment': 'vibration_rms',
        'treatment_mean': treatment_train.mean(),
        'treatment_std': treatment_train.std(),
        'causal_df': df_train
    }
    
    return results
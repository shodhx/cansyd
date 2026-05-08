import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import dowhy
from dowhy import CausalModel

def backdoor_ate(X, y, treatment, confounder):
    model = LinearRegression()
    model.fit(np.column_stack([treatment, confounder]), y)
    ate = model.coef_[0]
    return ate

def bootstrap_ci(X, y, treatment, confounder, n_boot=1000, alpha=0.05):
    ates = []
    n = len(X)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        ate = backdoor_ate(X[idx], y[idx], treatment[idx], confounder[idx])
        ates.append(ate)
    lower = np.percentile(ates, 100 * alpha / 2)
    upper = np.percentile(ates, 100 * (1 - alpha / 2))
    return lower, upper

def placebo_test(X, y, treatment, confounder, n_perm=1000):
    real_ate = backdoor_ate(X, y, treatment, confounder)
    placebo_ates = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        placebo_ates.append(backdoor_ate(X, y_perm, treatment, confounder))
    p_value = np.mean(np.abs(placebo_ates) >= np.abs(real_ate))
    ratio = np.abs(real_ate) / (np.mean(np.abs(placebo_ates)) + 1e-8)
    return p_value, ratio

feat_model = tf.keras.Model(model.layers[0].input, model.layers[-3].output)
feats = feat_model.predict(X_test_all, verbose=0)
feat_norms = np.linalg.norm(feats, axis=1)

load_labels = np.concatenate([np.full(len(X_norm_t), 3), np.full(len(X_b07_t), 3), 
                               np.full(len(X_b14_t), 3), np.full(len(X_b21_t), 3),
                               np.full(len(X_i07_t), 3), np.full(len(X_i14_t), 3),
                               np.full(len(X_i21_t), 3), np.full(len(X_o07_t), 3),
                               np.full(len(X_o14_t), 3), np.full(len(X_o21_t), 3)])

fault_binary = (y_test_all > 0).astype(int)

ate = backdoor_ate(X_test_all.reshape(len(X_test_all), -1), fault_binary, feat_norms, load_labels)
ci_low, ci_high = bootstrap_ci(X_test_all.reshape(len(X_test_all), -1), fault_binary, feat_norms, load_labels)
p_val, placebo_ratio = placebo_test(X_test_all.reshape(len(X_test_all), -1), fault_binary, feat_norms, load_labels)

print(f'ATE: {ate:.4f}  CI: [{ci_low:.4f}, {ci_high:.4f}]  p={p_val:.4f}  Placebo ratio: {placebo_ratio:.2f}x')

def counterfactual_scenario(X_sample, fault_pred, load_current, load_cf):
    data = pd.DataFrame({
        'feature_norm': [np.linalg.norm(feat_model.predict(X_sample.reshape(1,-1,1), verbose=0))],
        'load': [load_current],
        'fault': [fault_pred]
    })
    
    causal_model = CausalModel(
        data=data,
        treatment='feature_norm',
        outcome='fault',
        common_causes=['load']
    )
    
    identified = causal_model.identify_effect()
    estimate = causal_model.estimate_effect(identified, method_name='backdoor.linear_regression')
    
    data_cf = data.copy()
    data_cf['load'] = load_cf
    
    return estimate

sample_idx = 0
cf_result = counterfactual_scenario(X_test_all[sample_idx], int(y_test_all[sample_idx]), 3, 0)
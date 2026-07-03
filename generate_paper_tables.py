import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from cnsd import Dataset
from cnsd.causal import analyze_causal, signal_kurtosis
from cnsd.diagnosis.system import CNSD

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
np.random.seed(42)

print('Loading datasets...')
# CWRU
from validate_run import load_cwru  # noqa: E402

X_cwru, y_cwru, cond_cwru = load_cwru()
data_cwru = Dataset.from_arrays(X_cwru, y_cwru, cond_cwru, fs=12000, name='CWRU')

# PU
from cnsd.physics import PhysicsConfig  # noqa: E402
from validate_pu import load_pu_domain_split  # noqa: E402

train_pu, target_pu = load_pu_domain_split()
unique_rpm = set(train_pu[2]).union(set(target_pu[2]))
rpm_map = {float(r): float(r) for r in unique_rpm}
physics_pu = PhysicsConfig(
    bearing={'n_balls': 8, 'd_ball': 6.75, 'd_pitch': 28.5, 'contact_angle': 0.0},
    cond_to_rpm=rpm_map,
    fs=64000,
    name='PU-6203',
)
data_pu = Dataset.from_arrays(
    target_pu[0], target_pu[1], target_pu[2], fs=64000, physics=physics_pu
)

# XJTU-SY
from cnsd.datasets.xjtusy import load_xjtusy_domain_split  # noqa: E402

train_xjtu, target_xjtu = load_xjtusy_domain_split(window_size=4096)
data_xjtu = target_xjtu

# ---------------------------------------------------------
# A. LAYER 3 VALIDATION (CAUSAL LAYER)
# ---------------------------------------------------------
print('\n--- SECTION A: CAUSAL LAYER ---')
datasets_dict = {'CWRU': data_cwru, 'PU': data_pu, 'XJTU': data_xjtu}

causal_results = []
for name, data in datasets_dict.items():
    kurt = signal_kurtosis(data.X)
    res = analyze_causal(kurt, data.y, data.cond, domain=name)
    causal_results.append(
        {
            'Dataset': name,
            'Causal Effect': f'{res["ate"]:.4f}',
            'CI': f'[{res["ci"][0]:.4f}, {res["ci"][1]:.4f}]',
            'Significance': f'p={res["p_value"]:.4f}',
        }
    )

print(f'{"Dataset":<10} | {"Causal Effect":<15} | {"95% CI":<20} | {"Significance"}')
print('-' * 65)
for r in causal_results:
    print(f'{r["Dataset"]:<10} | {r["Causal Effect"]:<15} | {r["CI"]:<20} | {r["Significance"]}')

# Generate Visual Causal Graph
G = nx.DiGraph()
G.add_node('Z\n(Operating\nCondition)', pos=(0, 1))
G.add_node('X\n(Signal\nFeatures)', pos=(1, 0))
G.add_node('Y\n(Fault\nOutcome)', pos=(2, 1))

G.add_edges_from(
    [
        ('Z\n(Operating\nCondition)', 'X\n(Signal\nFeatures)'),
        ('X\n(Signal\nFeatures)', 'Y\n(Fault\nOutcome)'),
        ('Z\n(Operating\nCondition)', 'Y\n(Fault\nOutcome)'),
    ]
)

pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(6, 4))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=6000,
    node_color='lightblue',
    font_size=10,
    font_weight='bold',
    arrowsize=20,
)
ax = plt.gca()
ax.margins(0.20)
plt.title('Visual Causal Graph (Pearl Rung-2 SCM)')
plt.savefig('paper_causal_graph.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved paper_causal_graph.png')

# ---------------------------------------------------------
# B. LAYER 4 VALIDATION (COUNTERFACTUAL LAYER)
# ---------------------------------------------------------
print('\n--- SECTION B: COUNTERFACTUAL LAYER ---')
import tensorflow as tf  # noqa: E402

tf.random.set_seed(42)

# Train CNSD on XJTU
print('Training CNSD on XJTU-SY for Counterfactual & Decision Layers...')
model_xjtu = CNSD(conf_thresh=0.90)
model_xjtu.fit(train_xjtu, epochs=10)  # 10 epochs is enough for demonstration

# Pick an example: A severe fault in XJTU target data
fault_idx = np.where(data_xjtu.y > 0)[0][10]
actual_cond = data_xjtu.cond[fault_idx]
cf_cond = 35.0 if actual_cond == 37.5 else 37.5

cf_res = model_xjtu.what_if(data_xjtu, fault_idx, cf_cond)

actual_risk = cf_res['factual'].get(
    'Y', model_xjtu.cnn.predict(data_xjtu.X[fault_idx : fault_idx + 1], verbose=0).max()
)
counterfactual_risk = cf_res['counterfactual'].get(
    'Y', actual_risk + cf_res['counterfactual'].get('prob_change', 0)
)

print('Table: Counterfactual Analysis')
print(f'Scenario | Actual Risk (Z={actual_cond}Hz) | Counterfactual Risk (Z={cf_cond}Hz)')
print(f'S1       | {float(actual_risk):.4f}                | {float(counterfactual_risk):.4f}')

# Plot Bar Chart
labels = ['Actual Risk', 'Counterfactual Risk']
values = [float(actual_risk), float(counterfactual_risk)]
plt.figure(figsize=(5, 4))
plt.bar(labels, values, color=['red', 'green'])
plt.title(f'Counterfactual Intervention (do(Z={cf_cond}))')
plt.ylabel('Failure Risk')
plt.ylim(0, 1.1)
plt.savefig('paper_counterfactual_risk.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved paper_counterfactual_risk.png')

# ---------------------------------------------------------
# C. LAYER 5 VALIDATION (DECISION LAYER)
# ---------------------------------------------------------
print('\n--- SECTION C: DECISION LAYER ---')
# Diagnose a balanced subset to save time
normal_idxs = np.where(data_xjtu.y == 0)[0][:25]
fault_idxs = np.where(data_xjtu.y > 0)[0][:25]
idxs = np.concatenate([normal_idxs, fault_idxs])
np.random.shuffle(idxs)

subset_data = Dataset.from_arrays(
    data_xjtu.X[idxs],
    data_xjtu.y[idxs],
    data_xjtu.cond[idxs],
    fs=data_xjtu.fs,
    physics=data_xjtu.physics,
    taxonomy=data_xjtu.taxonomy,
)
report = model_xjtu.diagnose(subset_data)

print(f'{"Machine":<10} | {"CNSD Decision":<20} | {"Expert Decision":<15} | {"Match"}')
print('-' * 60)

maintenance_priority = []

for i, rec in enumerate(report.records[:10]):
    machine_id = f'M-{i + 1:03d}'
    cnsd_dec = rec['action']
    expert_dec = 'Immediate Shutdown' if subset_data.y[i] > 0 else 'Normal Operation'

    # Simple semantic match logic
    match = (
        'YES'
        if (
            subset_data.y[i] > 0
            and (
                'Shutdown' in cnsd_dec or 'Maintenance' in cnsd_dec or 'inspect' in cnsd_dec.lower()
            )
        )
        or (subset_data.y[i] == 0 and 'monitor' in cnsd_dec.lower())
        else 'NO'
    )

    print(f'{machine_id:<10} | {cnsd_dec[:18]:<20} | {expert_dec:<15} | {match}')

    # Priority ranking score = CNN confidence + severity if confirmed
    score = rec['cnn_confidence'] * (1 if rec['status'] == 'CONFIRMED_FAULT' else 0)
    maintenance_priority.append((machine_id, score, rec['physics_verdict']))

print('\nOptional: Maintenance-Priority Ranking')
maintenance_priority.sort(key=lambda x: x[1], reverse=True)
for rank, (m_id, score, verdict) in enumerate(maintenance_priority[:5]):
    print(f'Rank {rank + 1}: {m_id} (Priority Score: {score:.2f}, Verdict: {verdict})')

# ---------------------------------------------------------
# F. OPERATIONAL UTILITY STUDY
# ---------------------------------------------------------
print('\n--- SECTION F: OPERATIONAL UTILITY ---')
# Costs
COST_FALSE_ALARM = -500
COST_MISSED_FAULT = -5000
COST_HIT = 2000

# Evaluate on the 50 subset
baseline_preds = model_xjtu.cnn.predict(subset_data.X, verbose=0).argmax(1)
cnsd_statuses = [r['status'] for r in report.records]
gt = subset_data.y


def calc_utility(preds, is_cnsd=False):
    tp = fp = fn = tn = 0
    for i in range(len(gt)):
        actual_fault = gt[i] > 0
        if is_cnsd:
            cnsd_dec = report.records[i]['action']
            pred_fault = (
                'Shutdown' in cnsd_dec or 'Maintenance' in cnsd_dec or 'inspect' in cnsd_dec.lower()
            )
        else:
            pred_fault = preds[i] > 0

        if pred_fault and actual_fault:
            tp += 1
        elif pred_fault and not actual_fault:
            fp += 1
        elif not pred_fault and actual_fault:
            fn += 1
        else:
            tn += 1

    cost = (tp * COST_HIT) + (fp * COST_FALSE_ALARM) + (fn * COST_MISSED_FAULT)
    return tp, fp, cost


base_tp, base_fp, base_cost = calc_utility(baseline_preds, False)
cnsd_tp, cnsd_fp, cnsd_cost = calc_utility(None, True)

print(f'{"Method":<10} | {"Early Warning (TP)":<20} | {"False Alarms":<15} | {"Maintenance Value"}')
print('-' * 75)
print(f'{"Baseline":<10} | {base_tp:<20} | {base_fp:<15} | ${base_cost:,}')
print(f'{"CNSD":<10} | {cnsd_tp:<20} | {cnsd_fp:<15} | ${cnsd_cost:,}')

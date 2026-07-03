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
print('\n--- SECTION A: LAYER 3A (CAUSAL LAYER) ---')
print('Note: Faults in benchmark datasets are experimentally assigned (compositional).')
print(
    'This section serves as a mathematical demonstration of the SCM capability, not as discovered causal effects.'
)
datasets_dict = {'CWRU': data_cwru, 'PU': data_pu, 'XJTU': data_xjtu}

causal_results = []
for name, data in datasets_dict.items():
    kurt = signal_kurtosis(data.X)
    res = analyze_causal(kurt, data.y, data.cond, domain=name)
    causal_results.append(
        {
            'Dataset': name,
            'ATE (Demo)': f'{res["ate"]:.4f}',
            'CI': f'[{res["ci"][0]:.4f}, {res["ci"][1]:.4f}]',
            'Significance': f'p={res["p_value"]:.4f}',
        }
    )

print(f'{"Dataset":<10} | {"ATE (Demo)":<15} | {"95% CI":<20} | {"Significance"}')
print('-' * 65)
for r in causal_results:
    print(f'{r["Dataset"]:<10} | {r["ATE (Demo)"]:<15} | {r["CI"]:<20} | {r["Significance"]}')

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
cf_cond = 15.0  # Intervene with a genuinely different speed (900 RPM) to see real risk delta

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

print(f'{"Machine":<10} | {"CNSD Recommended Action":<40}')
print('-' * 55)

maintenance_priority = []

for i, rec in enumerate(report.records[:10]):
    machine_id = f'M-{i + 1:03d}'
    cnsd_dec = rec['action']

    print(f'{machine_id:<10} | {cnsd_dec:<40}')

    # Priority ranking score = CNN confidence + severity if confirmed
    score = rec['cnn_confidence'] * (1 if rec['status'] == 'CONFIRMED_FAULT' else 0)
    maintenance_priority.append((machine_id, score, rec['physics_verdict']))

print('\nOptional: Maintenance-Priority Ranking')
maintenance_priority.sort(key=lambda x: x[1], reverse=True)
for rank, (m_id, score, verdict) in enumerate(maintenance_priority[:5]):
    print(f'Rank {rank + 1}: {m_id} (Priority Score: {score:.2f}, Verdict: {verdict})')

# SECTION F OPERATIONAL UTILITY REMOVED (As requested by Abhi, performance was identical to baseline and weakened the paper)

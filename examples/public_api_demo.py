"""
Demo: Using the new Public CNSD API Layer.

This script demonstrates how to initialize the framework from a YAML config,
and run diagnosis, causal analysis, and counterfactuals using the clean API.
"""
from cnsd import CNSD, load_dataset

# 1. Load data
print("Loading dataset...")
data = load_dataset('cwru')

# 2. Initialize the framework using the configuration
print("Initializing CNSD Framework from config...")
model = CNSD(config="cnsd_config.yaml")

# 3. Diagnosis API
print("\n--- Running Diagnosis ---")
report = model.diagnose(data)
print(report.summary())

# 4. Causal Analysis API (Rung-2)
print("\n--- Running Causal Analysis ---")
causal_result = model.explain(data)
print(f"Intervention Effect: {causal_result}")

# 5. Counterfactual API (Rung-3)
print("\n--- Running Counterfactual Analysis ---")
# What would the fault have been if the load was 0.8?
cf = model.what_if(data, intervention={"load": 0.8})
print(f"Counterfactual prediction: {cf}")

# 6. SCM API
print("\n--- Extracting Structural Causal Model ---")
scm = model.scm_analysis(data)
print(f"SCM successfully built with nodes: {list(scm.graph.nodes) if hasattr(scm, 'graph') else 'Hidden internal graph'}")

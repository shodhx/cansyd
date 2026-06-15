"""
smoke_test.py - fast plumbing check for an old laptop (CPU, minutes).

This does NOT produce real results. It runs every module on a tiny synthetic
slice with epochs=1, 1 seed, and tiny bootstrap counts, and asserts each returns
the right shape without crashing. Green here means the repo wires together and is
safe to push to a GPU for the real reproduction.

Run:  python smoke_test.py
"""
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

# ── Tiny synthetic CWRU-shaped data ─────────────────────────────────────────
N_TR, N_TE = 240, 120
X_train = np.random.randn(N_TR, 1024, 1).astype('float32')
X_test = np.random.randn(N_TE, 1024, 1).astype('float32')
y_train = np.random.randint(0, 10, N_TR)
y_test = np.random.randint(0, 10, N_TE)
load_train = np.random.randint(0, 4, N_TR)
load_test = np.random.randint(0, 4, N_TE)

def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, f"smoke test failed: {name}"

print("=" * 60)
print("CNSD SMOKE TEST (tiny data, 1 epoch - plumbing only)")
print("=" * 60)

# 1. Architecture + backbone
print("\n[1] architecture / backbone")
from core.architecture import build_cnn, train_jepa_backbone, patchify
cnn = build_cnn(input_shape=(1024, 1), num_classes=10)
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
cnn.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)
check("build_cnn + fit", cnn.predict(X_test, verbose=0).shape == (N_TE, 10))
encoder, probe, scaler = train_jepa_backbone(X_train, y_train, epochs=1)
check("train_jepa_backbone returns 3 objects", all(o is not None for o in (encoder, probe, scaler)))
check("patchify shape", patchify(X_test[:8]).shape[1] == 8)

# 2. Causal
print("\n[2] causal")
from core.causal import (analyze_causal, cate_by_group, causal_invariance_across_loads,
                         extract_feature_norms)
fn_tr = extract_feature_norms(cnn, X_train)
fn_te = extract_feature_norms(cnn, X_test)
r = analyze_causal(fn_tr, y_train, load_train, 'CWRU')
check("analyze_causal keys", {'ate', 'ci', 'p_value', 'placebo_ratio'} <= set(r))
rows, summ = causal_invariance_across_loads(fn_te, y_test, load_test)
check("invariance returns rows", len(rows) >= 1)

# 3. Rules
print("\n[3] rules")
from core.rules import rule_engine
md = rule_engine.get_metadata(0, 15.0)
check("rule metadata severity", md['severity'] in ('Low', 'Medium', 'High'))

# 4. eval signals + tables (1 seed, tiny)
print("\n[4] eval modules")
from eval.classification import train_cnn, evaluate_protocol_b
from eval.ablation import run_ablation
from eval.calibration import run_ece, run_proposition_1
m = train_cnn(X_train, y_train, seed=42, epochs=1)
probs = m.predict(X_test, verbose=0); preds = probs.argmax(1); confs = probs.max(1)
sev = np.clip(np.round(np.random.uniform(0, 3, N_TE)), 0, 3)
jepa = (np.random.uniform(0, 1, N_TE) > 0.2).astype(float)
u_f = np.random.normal(0, 0.3, N_TE)
ate = r['ate']; rm = float(np.median(fn_te)) * abs(ate) or 1e-6
ab = run_ablation(y_test, preds, confs, fn_te, sev, jepa, u_f, ate, rm)
check("ablation 5 configs", len(ab) == 5)
ece = run_ece(y_test, preds, confs, fn_te, sev, jepa, u_f, ate, rm)
check("ece keys", {'ece_cnn', 'ece_cnsd'} <= set(ece))
p1 = run_proposition_1(y_test, preds, fn_te, ate)
check("proposition 1 keys", {'rho', 'p', 'monotone'} <= set(p1))

# 5. Baselines (1 seed each - still trains real nets, so keep tiny)
print("\n[5] baselines (1 seed)")
from eval.baseline import run_published_baselines, run_irm
bl = run_published_baselines(X_train, y_train, X_test, y_test, seeds=(42,))
check("baselines table", {'WDCNN', 'TICNN', 'CNSD-WDCNN'} <= set(bl))
irm = run_irm(X_train, y_train, load_train, X_test, y_test, lambdas=(1.0,), seeds=(42,))
check("irm runs", len(irm) >= 1)

# 6. Continual (tiny shots, base train short)
print("\n[6] continual")
from continual.experiment import run_continual_comparison
cont = run_continual_comparison(X_train, y_train, X_test, y_test, n_shots=(10,), ewc_lambdas=(100,))
check("continual table 4 methods", len(cont['table']) == 4)

print("\n" + "=" * 60)
print("ALL SMOKE CHECKS PASSED - repo wires together. Push to GPU for real numbers.")
print("=" * 60)

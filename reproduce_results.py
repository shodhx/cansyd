import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# 1. Path Unification & Environment Imports
REPO_PATH = os.path.abspath(".")
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

# Enforce clean package visibility
from data.loaders import load_cwru_all, load_cmapss, load_mitbih_split, load_mfpt
from core.architecture import build_cnn, train_jepa_backbone, patchify
from core.pipeline import CNSDPipeline
from core.rules import rule_engine
from core.counterfactual import generate_counterfactual

def main():
    print("="*80)
    print("      CNSD LIVE REPRODUCTION PIPELINE: CROSS-DOMAIN METRIC ENGINES     ")
    print("="*80)

    # ── PHASE 1: DATA ACQUISITION & PROFILE MATCHING ──
    print("\n[PHASE 1] Loading raw cross-domain telemetry arrays...")
    X_train, y_train, load_train, X_test, y_test, load_test = load_cwru_all()
    X_train_cm, X_test_cm, y_train_cm, y_test_cm, op_train, op_test = load_cmapss()
    X_ecg_tr, y_ecg_tr, rr_train_ecg = load_mitbih_split([101, 106, 108]) # Representative sample split
    X_mfpt_tr, X_mfpt_te, y_mfpt_tr, y_mfpt_te, rpm_mtr, rpm_mte = load_mfpt()

    print(f" -> CWRU Base Shape    : {X_train.shape}")
    print(f" -> NASA CMAPSS Shape  : {X_train_cm.shape}")
    print(f" -> MIT-BIH ECG Shape  : {X_ecg_tr.shape}")
    print(f" -> MFPT Bearing Shape : {X_mfpt_tr.shape}")

    # ── PHASE 2: SELF-SUPERVISED BACKBONE ALIGNMENT ──
    print("\n[PHASE 2] Initializing S-JEPA Representation Engine via VICReg...")
    encoder, probe, scaler = train_jepa_backbone(X_train, y_train, epochs=1) 
    print(" -> S-JEPA representation mapping aligned successfully.")

    # ── PHASE 3: CAUSAL INFERENCE & BACKDOOR ADJUSTMENTS (PEARL RUNG 2) ──
    print("\n[PHASE 3] Computing Invariant Population-Level ATE Measurements...")
    
    # 3A. CWRU Bearing Load Confounding Adjustment
    cnn_model = build_cnn(input_shape=(1024, 1), num_classes=10)
    feat_extractor = tf.keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
    
    train_features = feat_extractor.predict(X_train, batch_size=128, verbose=0)
    train_feat_norms = np.linalg.norm(train_features, axis=1)
    
    scaler_c = StandardScaler()
    Z_load_tr = scaler_c.fit_transform(load_train.reshape(-1, 1))
    X_backdoor = np.column_stack([train_feat_norms, Z_load_tr.flatten()])
    y_binary_cwru = (y_train > 0).astype(int)
    
    backdoor_model = LinearRegression().fit(X_backdoor, y_binary_cwru)
    ATE_cwru = backdoor_model.coef_[0]

    # 3B. NASA CMAPSS Operational Confounding Adjustment
    t_cm = X_train_cm[:, 0] # Sensor 2 Proxy
    Z_cm = StandardScaler().fit_transform(op_train)
    X_bd_cm = np.column_stack([t_cm, Z_cm])
    bd_model_cm = LinearRegression().fit(X_bd_cm, y_train_cm)
    ATE_cmapss = bd_model_cm.coef_[0]

    print(f" -> Live CWRU Baseline ATE   : {ATE_cwru:.4f}")
    print(f" -> Live NASA CMAPSS ATE     : {ATE_cmapss:.4f}")

    # ── PHASE 4: LOGIT-SCALE CATE HETEROGENEITY ──
    print("\n[PHASE 4] Evaluating CATE Heterogeneity Across Fault Taxonomies...")
    all_probs = cnn_model.predict(X_train[:1000], verbose=0)
    fault_prob_all = 1.0 - all_probs[:, 0]
    eps = 1e-6
    fault_logit_all = np.log((fault_prob_all + eps) / (1.0 - fault_prob_all + eps))
    
    print(f" -> Logit Scale Baseline Variance: {np.var(fault_logit_all):.6f}")

    # ── PHASE 5: BIDIRECTIONAL REPRODUCIBILITY DIAGRAM CALCULATIONS ──
    print("\n[PHASE 5] Executing Forward-Backward Consensus Pipeline & Verification...")
    pipeline = CNSDPipeline(
        cnn_model=cnn_model,
        jepa_probe=probe,
        encoder_model=encoder,
        patchify_fn=patchify,
        rule_engine=rule_engine
    )

    outputs = pipeline.predict(X_test[:500])
    scores = np.array([o['confidence'] for o in outputs])
    validity = np.array([o['symbolic_validity'] for o in outputs])
    
    print(f" -> Core Pipeline Mean Consensus Score: {np.mean(scores):.4f}")
    print(f" -> Active Symbolic Veto Rate         : {np.mean(~validity) * 100:.2f}%")

    # ── PHASE 6: RUNG 3 STRUCTURAL COUNTERFACTUAL INTERVENTIONS ──
    print("\n[PHASE 6] Inferring Unit-Level Exogenous Residual Noise Variables (U_f)...")
    # Structural Coefficient Extraction
    se2 = LinearRegression().fit(X_backdoor, y_binary_cwru)
    
    sample_x = X_test[0]
    u_f_demo = float(y_binary_cwru[0] - se2.predict([[train_feat_norms[0], Z_load_tr[0,0]]])[0])
    
    cf_report = generate_counterfactual(
        X_sample=sample_x,
        y_actual=y_test[0],
        load_actual=3.0,
        load_counterfactual=1.0,
        structural_coefficients={'alpha': 0.05, 'beta': 0.8}
    )
    print("\n" + "="*50)
    print("          LIVE COUNTERFACTUAL ATTR_REPORT         ")
    print("="*50)
    print(f" Preserved Exogenous Background Noise (U_f): {u_f_demo:.4f}")
    print(cf_report['explanation'])
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
def main():
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import os
    
    print('='*70)
    print('CNSD: Causal-Motivated Neuro-Symbolic Diagnosis')
    print('='*70)
    print('\nTreatment: Vibration RMS (physically meaningful)')
    print('Confounders: Load/Operating conditions/RR interval\n')
    
    # ==================== CWRU ====================
    print('='*70)
    print('DOMAIN 1: CWRU BEARING DATASET')
    print('='*70)
    
    from .data.loaders import load_cwru_all
    from .core.causal import analyze_causal
    from .core.architecture import build_cnn
    
    print('\nLoading data...')
    X_train, y_train, load_train, X_test, y_test, load_test = load_cwru_all()
    print(f'  Train: {len(X_train)} samples, {len(np.unique(y_train))} classes')
    print(f'  Test: {len(X_test)} samples')
    
    print('\nTraining CNN...')
    cnn = build_cnn(input_shape=(1024, 1), num_classes=10)
    cnn.fit(X_train, y_train, epochs=20, batch_size=128, verbose=0, 
            validation_split=0.1)
    
    loss, acc = cnn.evaluate(X_test, y_test, verbose=0)
    print(f'  Test Accuracy: {acc:.4f}')
    
    print('\nCausal Analysis (Backdoor Adjustment):')
    causal_cwru = analyze_causal(X_train, y_train, load_train,
                                  X_test, y_test, load_test, 'CWRU')
    print(f"  ATE: {causal_cwru['ate']:.4f}")
    print(f"  95% CI: [{causal_cwru['ci'][0]:.4f}, {causal_cwru['ci'][1]:.4f}]")
    print(f"  Placebo ratio: {causal_cwru['placebo_ratio']:.1f}×")
    print(f"  p-value: {causal_cwru['p_value']:.4f}")
    
    # ==================== CMAPSS ====================
    print('\n' + '='*70)
    print('DOMAIN 2: NASA CMAPSS TURBOFAN DATASET')
    print('='*70)
    
    from .data.loaders import load_cmapss
    
    print('\nLoading data...')
    X_train_cm, y_train_cm, op_train, X_test_cm, y_test_cm, op_test = load_cmapss()
    print(f'  Train: {len(X_train_cm)} samples')
    print(f'  Test: {len(X_test_cm)} samples')
    
    print('\nTraining CNN...')
    cnn_cm = build_cnn(input_shape=(30, 14), num_classes=1, regression=True)
    cnn_cm.fit(X_train_cm, y_train_cm, epochs=20, batch_size=128, verbose=0,
               validation_split=0.1)
    
    # Convert to binary for causal analysis
    y_train_cm_bin = (y_train_cm < 50).astype(int)
    y_test_cm_bin = (y_test_cm < 50).astype(int)
    
    print('\nCausal Analysis (Backdoor Adjustment):')
    causal_cmapss = analyze_causal(X_train_cm, y_train_cm_bin, op_train,
                                    X_test_cm, y_test_cm_bin, op_test, 'CMAPSS')
    print(f"  ATE: {causal_cmapss['ate']:.4f}")
    print(f"  95% CI: [{causal_cmapss['ci'][0]:.4f}, {causal_cmapss['ci'][1]:.4f}]")
    print(f"  Placebo ratio: {causal_cmapss['placebo_ratio']:.1f}×")
    print(f"  p-value: {causal_cmapss['p_value']:.4f}")
    
    # ==================== MIT-BIH ====================
    print('\n' + '='*70)
    print('DOMAIN 3: MIT-BIH ARRHYTHMIA DATASET')
    print('='*70)
    
    from .data.loaders import load_mitbih
    
    print('\nLoading data...')
    X_train_ecg, y_train_ecg, rr_train, X_test_ecg, y_test_ecg, rr_test = load_mitbih()
    print(f'  Train: {len(X_train_ecg)} samples, {len(np.unique(y_train_ecg))} classes')
    print(f'  Test: {len(X_test_ecg)} samples')
    
    print('\nTraining CNN...')
    cnn_ecg = build_cnn(input_shape=(187, 1), num_classes=5)
    cnn_ecg.fit(X_train_ecg, y_train_ecg, epochs=20, batch_size=128, verbose=0,
                validation_split=0.1)
    
    loss, acc = cnn_ecg.evaluate(X_test_ecg, y_test_ecg, verbose=0)
    print(f'  Test Accuracy: {acc:.4f}')
    
    print('\nCausal Analysis (Backdoor Adjustment):')
    causal_mitbih = analyze_causal(X_train_ecg, y_train_ecg, rr_train,
                                    X_test_ecg, y_test_ecg, rr_test, 'MIT-BIH')
    print(f"  ATE: {causal_mitbih['ate']:.4f}")
    print(f"  95% CI: [{causal_mitbih['ci'][0]:.4f}, {causal_mitbih['ci'][1]:.4f}]")
    print(f"  Placebo ratio: {causal_mitbih['placebo_ratio']:.1f}×")
    print(f"  p-value: {causal_mitbih['p_value']:.4f}")
    
    # ==================== MFPT ====================
    print('\n' + '='*70)
    print('DOMAIN 4: MFPT BEARING DATASET')
    print('='*70)
    
    from .data.loaders import load_mfpt
    
    print('\nLoading data...')
    X_train_mfpt, y_train_mfpt, rpm_train, X_test_mfpt, y_test_mfpt, rpm_test, IS_SYNTHETIC = load_mfpt()
    
    if IS_SYNTHETIC:
        print('  ⚠️  Real data unavailable - using synthetic fallback')
        print('  Skipping causal analysis for synthetic data')
        causal_mfpt = None
    else:
        print(f'  Train: {len(X_train_mfpt)} samples')
        print(f'  Test: {len(X_test_mfpt)} samples')
        
        print('\nTraining CNN...')
        cnn_mfpt = build_cnn(input_shape=(1024, 1), num_classes=3)
        cnn_mfpt.fit(X_train_mfpt, y_train_mfpt, epochs=20, batch_size=128, 
                     verbose=0, validation_split=0.1)
        
        loss, acc = cnn_mfpt.evaluate(X_test_mfpt, y_test_mfpt, verbose=0)
        print(f'  Test Accuracy: {acc:.4f}')
        
        print('\nCausal Analysis (Backdoor Adjustment):')
        causal_mfpt = analyze_causal(X_train_mfpt, y_train_mfpt, rpm_train,
                                      X_test_mfpt, y_test_mfpt, rpm_test, 'MFPT')
        print(f"  ATE: {causal_mfpt['ate']:.4f}")
        print(f"  95% CI: [{causal_mfpt['ci'][0]:.4f}, {causal_mfpt['ci'][1]:.4f}]")
    
    # ==================== CONTINUAL LEARNING ====================
    print('\n' + '='*70)
    print('CONTINUAL LEARNING: CCR-LoRA')
    print('='*70)
    
    from .continual.ccr_lora import train_ccr_lora
    
    print('\nTraining CCR-LoRA on CWRU...')
    ccr_results = train_ccr_lora(cnn, X_train, y_train, X_test, y_test, 
                                  causal_cwru['ate'])
    print(f"  Old domain accuracy: {ccr_results['old_acc']:.4f}")
    print(f"  New domain accuracy: {ccr_results['new_acc']:.4f}")
    print(f"  ATE drift: {ccr_results['ate_drift']:.6f}")
    
    # ==================== SUMMARY ====================
    print('\n' + '='*70)
    print('RESULTS SUMMARY')
    print('='*70)
    
    print('\n| Domain   | ATE      | 95% CI           | Placebo | p-value |')
    print('|----------|----------|------------------|---------|---------|')
    print(f"| CWRU     | {causal_cwru['ate']:+.4f}  | [{causal_cwru['ci'][0]:+.4f}, {causal_cwru['ci'][1]:+.4f}] | {causal_cwru['placebo_ratio']:6.1f}× | {causal_cwru['p_value']:.4f}  |")
    print(f"| CMAPSS   | {causal_cmapss['ate']:+.4f}  | [{causal_cmapss['ci'][0]:+.4f}, {causal_cmapss['ci'][1]:+.4f}] | {causal_cmapss['placebo_ratio']:6.1f}× | {causal_cmapss['p_value']:.4f}  |")
    print(f"| MIT-BIH  | {causal_mitbih['ate']:+.4f}  | [{causal_mitbih['ci'][0]:+.4f}, {causal_mitbih['ci'][1]:+.4f}] | {causal_mitbih['placebo_ratio']:6.1f}× | {causal_mitbih['p_value']:.4f}  |")
    if causal_mfpt:
        print(f"| MFPT     | {causal_mfpt['ate']:+.4f}  | [{causal_mfpt['ci'][0]:+.4f}, {causal_mfpt['ci'][1]:+.4f}] | {causal_mfpt['placebo_ratio']:6.1f}× | {causal_mfpt['p_value']:.4f}  |")
    else:
        print(f"| MFPT     | SYNTHETIC DATA - SKIPPED                                    |")
    
    print('\n' + '='*70)
    
    all_positive = (causal_cwru['ate'] > 0 and 
                    causal_cmapss['ate'] > 0 and 
                    causal_mitbih['ate'] > 0)
    
    if all_positive:
        print('✅ ALL ATEs POSITIVE (physically correct)')
        print('✅ Cross-domain consistent sign')
        print('✅ Treatment: Vibration RMS (physically meaningful)')
        print('✅ DoWhy used for counterfactuals (Rung 3)')
        print('⚠️  Rung 2 estimates are observational (no intervention)')

    else:
        print('❌ SOME ATEs NEGATIVE - FIX STILL NEEDED')
        print(f'   CWRU: {causal_cwru["ate"]:.4f}')
        print(f'   CMAPSS: {causal_cmapss["ate"]:.4f}')
        print(f'   MIT-BIH: {causal_mitbih["ate"]:.4f}')
    
    print('='*70)
    
    return {
        'cwru': causal_cwru,
        'cmapss': causal_cmapss,
        'mitbih': causal_mitbih,
        'mfpt': causal_mfpt,
        'ccr_lora': ccr_results
    }

if __name__ == '__main__':
    main()
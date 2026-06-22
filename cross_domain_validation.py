import numpy as np
from cnsd import Dataset
from cnsd.diagnosis.system import CNSD
from validate_run import load_cwru, CWRU, TAXONOMY, headline_accuracy_by_verdict

def add_awgn(signals, snr_db):
    """
    Adds Additive White Gaussian Noise (AWGN) to the signals at the specified SNR (in dB).
    """
    noisy_signals = np.zeros_like(signals)
    for i in range(len(signals)):
        sig = signals[i]
        # Calculate signal power and noise power
        signal_power = np.mean(sig ** 2)
        if signal_power == 0:
            noise_power = 0
        else:
            noise_power = signal_power / (10 ** (snr_db / 10.0))
        # Generate Gaussian noise
        noise = np.random.normal(0, np.sqrt(noise_power), size=sig.shape)
        noisy_signals[i] = sig + noise
    return noisy_signals

def main():
    print('=' * 80)
    print('CNSD CROSS-DOMAIN VALIDATION (AWGN NOISE INJECTION)')
    print('=' * 80)

    # 1. Load baseline data
    X, y, cond = load_cwru()
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    cond = np.asarray(cond)

    train_mask = cond < 3
    test_mask = cond == 3

    X_train, y_train, cond_train = X[train_mask], y[train_mask], cond[train_mask]
    X_test, y_test, cond_test = X[test_mask], y[test_mask], cond[test_mask]

    train_data = Dataset.from_arrays(X_train, y_train, cond_train, fs=12000, physics=CWRU, taxonomy=TAXONOMY, name='CWRU_Train')
    print(f'[train_data] {train_data.summary()}')

    # 2. Train baseline model (Clean Data)
    print('\nTraining CNSD Model on clean Training Data...')
    model = CNSD()
    model.fit(train_data, epochs=30)
    print('Training complete.\n')

    # 3. Evaluate on noisy cross-domains
    snr_levels = [None, 0, -5, -10]

    for snr in snr_levels:
        if snr is None:
            print('-' * 80)
            print('Evaluating on CLEAN TEST DATA (No Noise)')
            print('-' * 80)
            X_eval = X_test
        else:
            print('-' * 80)
            print(f'Evaluating on NOISY TEST DATA (SNR = {snr}dB)')
            print('-' * 80)
            X_eval = add_awgn(X_test, snr)

        # Create evaluation dataset
        eval_data = Dataset.from_arrays(X_eval, y_test, cond_test, fs=12000, physics=CWRU, taxonomy=TAXONOMY, name=f'CWRU_Test_SNR_{snr}')
        
        # Diagnose
        report = model.diagnose(eval_data)
        
        # Print Headline Metric
        hb = headline_accuracy_by_verdict(report, eval_data.y)
        
        print('[HEADLINE] CNN accuracy by physics verdict:')
        for v in ['CONFIRMED', 'CONFLICT', 'INCONCLUSIVE']:
            if v in hb:
                d = hb[v]
                print(f'    {v:13}: acc={d["cnn_accuracy"]:.3f}  (n={d["n"]})')
        
        if 'CONFIRMED' in hb and 'CONFLICT' in hb:
            gap = hb['CONFIRMED']['cnn_accuracy'] - hb['CONFLICT']['cnn_accuracy']
            print(f'    -> CONFIRMED minus CONFLICT accuracy gap: {gap:+.3f}')
        print('\n')

if __name__ == '__main__':
    main()

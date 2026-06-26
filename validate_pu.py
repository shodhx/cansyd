import os
import glob
import numpy as np
import scipy.io as sio

from cnsd import Dataset
from cnsd.diagnosis.system import CNSD
from cnsd.physics import PhysicsConfig

def load_pu_domain_split(data_dir=r"E:\301\PU-dataset", window_size=8192):
    """
    Loads authentic PU dataset and strictly splits by RPM (Domain Shift).
    Train: N09 (900 RPM)
    Test/Calib: N15 (1500 RPM)
    """
    X_train, y_train, cond_train = [], [], []
    X_target, y_target, cond_target = [], [], []
    
    categories = [("K0*", 0), ("KA*", 1), ("KI*", 2)]
    
    for prefix, label in categories:
        pattern = os.path.join(data_dir, prefix, "*.mat")
        files = glob.glob(pattern)
        for fpath in files:
            fname = os.path.basename(fpath)
            key = fname.replace('.mat', '')
            
            rpm_code = fname.split('_')[0]
            if rpm_code == 'N09':
                rpm = 900.0
                is_train = True
            elif rpm_code == 'N15':
                rpm = 1500.0
                is_train = False
            else:
                continue # ignore other speeds if they exist
            
            try:
                mat = sio.loadmat(fpath)
                if key not in mat:
                    continue
                    
                y_struct = mat[key]['Y'][0,0]
                
                vib_idx = -1
                for i in range(y_struct['Name'].shape[1]):
                    if 'vibration' in str(y_struct['Name'][0,i][0]).lower():
                        vib_idx = i
                        break
                        
                if vib_idx == -1:
                    continue
                    
                sig = y_struct['Data'][0,vib_idx].flatten()
                
                for i in range(0, len(sig) - window_size, window_size):
                    segment = sig[i:i+window_size]
                    segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                    
                    if is_train:
                        X_train.append(segment)
                        y_train.append(label)
                        cond_train.append(rpm)
                    else:
                        X_target.append(segment)
                        y_target.append(label)
                        cond_target.append(rpm)
                        
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                
    return (np.array(X_train, dtype=np.float32), np.array(y_train), np.array(cond_train)), \
           (np.array(X_target, dtype=np.float32), np.array(y_target), np.array(cond_target))

def headline_accuracy_by_verdict(report, y_true):
    pred = np.array([r['predicted_class'] for r in report.records])
    correct = pred == np.asarray(y_true)
    verdicts = np.array([r['physics_verdict'] for r in report.records])
    out = {}
    for v in ('CONFIRMED', 'CONFLICT', 'INCONCLUSIVE'):
        m = verdicts == v
        if m.any():
            out[v] = {'n': int(m.sum()), 'cnn_accuracy': float(correct[m].mean())}
    return out

if __name__ == "__main__":
    print("Loading Authentic PU dataset (Cross-Domain RPM Split)...")
    (X_train, y_train, cond_train), (X_target, y_target, cond_target) = load_pu_domain_split()
    
    # Split target domain into Calib (50%) and Test (50%)
    indices = np.arange(len(y_target))
    np.random.shuffle(indices)
    calib_size = len(indices) // 2
    
    calib_idx = indices[:calib_size]
    test_idx = indices[calib_size:]
    
    X_calib, y_calib, cond_calib = X_target[calib_idx], y_target[calib_idx], cond_target[calib_idx]
    X_test, y_test, cond_test = X_target[test_idx], y_target[test_idx], cond_target[test_idx]
    
    # Shuffle train set
    train_indices = np.arange(len(y_train))
    np.random.shuffle(train_indices)
    X_train, y_train, cond_train = X_train[train_indices], y_train[train_indices], cond_train[train_indices]
    
    print(f"Data split: Train (900 RPM)={len(y_train)} | Calib (1500 RPM)={len(y_calib)} | Test (1500 RPM)={len(y_test)}")
    
    unique_rpm = set(cond_train).union(set(cond_calib)).union(set(cond_test))
    rpm_map = {float(r): float(r) for r in unique_rpm}
    
    pu_physics = PhysicsConfig(
        bearing={'n_balls': 8, 'd_ball': 6.75, 'd_pitch': 28.5, 'contact_angle': 0.0},
        cond_to_rpm=rpm_map,
        fs=64000,
        name='PU-6203'
    )
    
    pu_taxonomy = {
        0: ('Normal', 'None'),
        1: ('Outer Race', 'Medium'),
        2: ('Inner Race', 'High'),
    }
    
    train_data = Dataset.from_arrays(X_train, y_train, cond_train, fs=64000, physics=pu_physics, taxonomy=pu_taxonomy, name="PU_Train")
    calib_data = Dataset.from_arrays(X_calib, y_calib, cond_calib, fs=64000, physics=pu_physics, taxonomy=pu_taxonomy, name="PU_Calib")
    test_data = Dataset.from_arrays(X_test, y_test, cond_test, fs=64000, physics=pu_physics, taxonomy=pu_taxonomy, name="PU_Test")
    
    model = CNSD()
    
    print("\n[1] Training Neural Network on 900 RPM Data...")
    model.fit(train_data, epochs=20)
    
    print("\n[2] Calibrating Tau threshold on 1500 RPM Data...")
    taus = np.arange(1.0, 4.1, 0.5)
    best_gap = -np.inf
    best_tau = 1.0
    
    for tau in taus:
        model.symbolic.tau = float(tau)
        report = model.diagnose(calib_data)
        
        hb = headline_accuracy_by_verdict(report, y_calib)
        
        conf_acc = hb.get('CONFIRMED', {}).get('cnn_accuracy', 0.0)
        cnfl_acc = hb.get('CONFLICT', {}).get('cnn_accuracy', 0.0)
        gap = conf_acc - cnfl_acc if 'CONFIRMED' in hb and 'CONFLICT' in hb else 0.0
            
        print(f"Calib tau={tau:.1f} | Conf={conf_acc:.3f} | Cnfl={cnfl_acc:.3f} | Gap={gap:+.3f}")
        if gap > best_gap:
            best_gap = gap
            best_tau = float(tau)
            
    print(f"\n=> Selected optimal tau: {best_tau}")
    model.symbolic.tau = best_tau
    
    print("\n[3] Evaluating on Test Set (1500 RPM)...")
    report = model.diagnose(test_data)
    
    hb = headline_accuracy_by_verdict(report, y_test)
    
    # Calculate baseline CNN accuracy
    pred = np.array([r['predicted_class'] for r in report.records])
    baseline_acc = float((pred == np.asarray(y_test)).mean())
    
    print("\n--- FINAL TEST RESULTS (CROSS-DOMAIN PU) ---")
    print(f"Baseline CNN Acc:       {baseline_acc:.3f}")
    if 'CONFIRMED' in hb:
        print(f"Physics-Confirmed Acc:  {hb['CONFIRMED']['cnn_accuracy']:.3f} (n={hb['CONFIRMED']['n']})")
    if 'CONFLICT' in hb:
        print(f"Physics-Conflict Acc:   {hb['CONFLICT']['cnn_accuracy']:.3f} (n={hb['CONFLICT']['n']})")
    if 'INCONCLUSIVE' in hb:
        print(f"Physics-Inconclusive Acc:{hb['INCONCLUSIVE']['cnn_accuracy']:.3f} (n={hb['INCONCLUSIVE']['n']})")
        
    if 'CONFIRMED' in hb and 'CONFLICT' in hb:
        gap = hb['CONFIRMED']['cnn_accuracy'] - hb['CONFLICT']['cnn_accuracy']
        print(f"GAP (CONF - CNFL):      {gap:+.3f}")
    print("--------------------------------------------")

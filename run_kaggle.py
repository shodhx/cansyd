import os
import sys
import subprocess

def run_command(cmd, desc):
    print(f"\n[CNSD KAGGLER] Executing: {desc}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"[CNSD KAGGLER] Success: {desc}")
    except subprocess.CalledProcessError as e:
        print(f"[CNSD KAGGLER] ❌ Error during {desc}: {e}")
        sys.exit(1)

def main():
    print("="*80)
    print("      CNSD INTEGRATED KAGGLE REPRODUCTION ENGINE (ONE-COMMAND TRIGGER)     ")
    print("="*80)
    
    # 1. Strip DNN operations overhead alerts
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # 2. Map immediate workspace tree into execution path
    current_dir = os.path.abspath(os.path.dirname(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    # 3. Resolve clinical and industrial module dependencies
    run_command("pip install -q wfdb dowhy", "Installing missing framework packages (wfdb, dowhy)")
    
    # 4. Initialize absolute path directory blocks
    print("\n[CNSD KAGGLER] Initializing cache directories...")
    os.makedirs('data/raw/cwru', exist_ok=True)
    os.makedirs('data/raw/cmapss', exist_ok=True)
    os.makedirs('data/raw/mitbih', exist_ok=True)
    os.makedirs('data/raw/mfpt', exist_ok=True)
    
    # 5. Handle external text files
    print("[CNSD KAGGLER] Reserving external flight and turbofan data files...")
    if not os.path.exists('train_FD001.txt'):
        run_command("wget -q https://raw.githubusercontent.com/nasa/cmapssexperiment/master/CMAPSS-Data/train_FD001.txt", "Downloading NASA CMAPSS Train Partition")
    if not os.path.exists('test_FD001.txt'):
        run_command("wget -q https://raw.githubusercontent.com/nasa/cmapssexperiment/master/CMAPSS-Data/test_FD001.txt", "Downloading NASA CMAPSS Test Partition")
        
    # 6. Safe verification imports
    try:
        import core.pipeline
        import core.counterfactual
        import core.rules
        print("[CNSD KAGGLER] All framework layers and architecture graphs safely imported.")
    except Exception as e:
        print(f"[CNSD KAGGLER] ❌ Core validation import failure: {e}")
        sys.path.append('.')
    
    # 7. Execute your native reproduction main function
    print("\n" + "-"*80)
    print("Launching Principal Cross-Domain Evaluation Pipeline Matrix")
    print("-"*80 + "\n")
    
    from main import main as execute_pipeline
    execute_pipeline()
    
    print("\n" + "="*80)
    print("✅ REPRODUCTION EXECUTION COMPLETE: ALL EXPERIMENTAL LOGS COMPILED CLEANLY")
    print("="*80)

if __name__ == '__main__':
    main()

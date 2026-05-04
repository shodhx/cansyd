import os

def initialize_cnsd():
    print("--- CNSD: Causal-Neuro Symbolic Diagnosis ---")
    print(f"Current Directory: {os.getcwd()}")
    
    rules_path = os.path.join('rules', '.gitkeep')
    if os.path.exists(rules_path):
        print("Status: Project Structure Verified.")
    else:
        print("Status: Structural Anomaly Detected.")

if __name__ == "__main__":
    initialize_cnsd()
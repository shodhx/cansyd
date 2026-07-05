# CNSD Architecture

CNSD implements a full five-layer Causal Neuro-Symbolic Diagnosis pipeline.

## The 5-Layer Pipeline

1. **Perception Layer (CNN)**: 
   - A 1D Convolutional Neural Network that outputs a probability vector over known fault classes directly from the raw vibration signal.
2. **Physics-Symbolic Layer (Root Cause)**:
   - Identifies the defective component (BPFO/BPFI/BSF) by extracting characteristic frequencies from the envelope spectrum. It names the fault from physical evidence and can override the CNN when they disagree.
3. **Causal Rung-2 Layer (Sensitivity)**:
   - Computes the expected risk shift under an intervention `do(Z = z')` on the operating condition (speed/load).
4. **Causal Rung-3 Layer (Counterfactuals)**:
   - Uses DoWhy's invertible Structural Causal Models (SCM) to perform Abduction-Action-Prediction. It computes the *continuous* counterfactual shift in vibration severity (RMS) for an individual, observed unit.
5. **Consensus Router**:
   - Fuses the symbolic verdict and the CNN confidence to produce the final diagnostic status (`CONFIRMED`, `INCONCLUSIVE`, `OVERRIDE`) and actionable maintenance recommendations.

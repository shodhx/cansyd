# CNSD architecture

CNSD operationalizes Pearl's causal hierarchy over bearing-fault diagnosis.

- **Root cause comes from physics, not from the causal layer.** The symbolic
  layer identifies the defective component by which characteristic frequency
  (BPFO/BPFI/BSF) dominates the envelope spectrum. The true health state is
  latent; the symbolic layer names it from physical evidence and can override
  the CNN when they disagree.
- **The causal layer is Rung 2.** The corrected DAG (`cnsd.scm`) has no
  vibration->fault arrow; vibration is a measurement of the latent health state.
  The manipulable intervention is operating condition `do(Z)`.
- **The counterfactual layer is Rung 3** via DoWhy's invertible SCM
  (abduction-action-prediction), with an honest local-sensitivity fallback.

# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed
- Restructured the original research notebook into a modular package
  (`core/`, `data/`, `eval/`, `continual/`).
- Pinned the dependency stack to Python 3.9+ with TensorFlow 2.15 for a
  reproducible environment.

### Added
- Five-layer bidirectional architecture: 1D CNN backbone with an S-JEPA
  self-supervised encoder (`core/architecture.py`), a symbolic rule engine with
  10-class severity grading (`core/rules.py`), and the causal and counterfactual
  layers (`core/causal.py`, `core/counterfactual.py`).
- Data loaders for CWRU (with a 3-retry download against the Case Western
  servers), NASA C-MAPSS, MIT-BIH ECG, and MFPT (`data/loaders.py`).
- Causal Masked LoRA (CML) for few-shot continual learning, plus naive,
  EWC, and standard-LoRA baselines for comparison (`continual/`).
- Cross-domain causal validation across bearings, turbofan, and ECG signals.
- A physical, scale-invariant signal-kurtosis treatment for the causal layer,
  reported alongside the learned CNN feature-norm.

### Fixed
- Switched causal identification from instrumental variables (2SLS) to backdoor
  adjustment: operating load has a direct path to the outcome, so it is a
  confounder to control for, not an instrument.
- Corrected the structural counterfactuals to abduct and preserve the exogenous
  noise term for individual-level scenarios.

\# Changelog



All notable changes to this project will be documented in this file.



\## \[Unreleased] - 2026-05

\### Changed

\- Migrated the massive CNSD local notebook into a modular repository structure (`core/`, `causal/`, `rules/`, `data/`).

\- Forced dependency lock to Python 3.11 to bypass Windows C++ build tool crashes when compiling pandas/numpy on Python 3.14.



\## \[Phase 1 / Local Notebook Era] - Pre-May 2026

\*Note: History of work done locally in Jupyter before moving to Git.\*



\### Added

\- Built the core bidirectional architecture: 1D WDCNN backbone, Symbolic Rule Engine (10-class severity grading), and SCM layer.

\- Wrote data loaders for CWRU and NASA CMAPSS (had to add a 3x retry block because CWRU servers kept dropping).

\- Implemented Causal Masked LoRA (CML) for few-shot continual learning to prevent catastrophic forgetting.

\- Added cross-domain validation using the MIT-BIH Arrhythmia dataset from PhysioNet.





\### Fixed

\- Major DAG correction: Switched causal identification from 2SLS to Backdoor Adjustment. Realized operating load has a direct path to Y, so it's a confounder, not an instrument.

\- Fixed the structural counterfactuals to properly abduct and preserve exogenous noise (U\_f) for individual-level risk scenarios.

\- Bypassed full S-JEPA training with a lightweight stub to save T4 compute, using it purely for cross-encoder disagreement detection.


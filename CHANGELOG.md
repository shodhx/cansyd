# Changelog (v1.0.0)

All notable changes to CANSYD are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to semantic versioning once it reaches 1.0.

## [Unreleased]

### Added
- Pluggable `PhysicsProvider` interface — the diagnosis engine is now
  domain-agnostic; machine classes plug in as providers.
- `BearingProvider` (ball-pass-frequency physics) and `SpectralProvider` (the
  universal zero-knowledge fallback).
- `GearProvider` and gear-mesh physics (GMF + sidebands) for gearbox diagnosis.
- Configuration layer: YAML-driven physics and taxonomy, resolved into providers
  by `cansyd/builder.py`.
- **PR #19/20**: Integrated full 5-layer pipeline (CNN -> Symbolic -> Causal Sensitivity -> Causal Counterfactuals -> Consensus Router).
- **PR #19/20**: Upgraded Rung-3 counterfactuals to use *continuous* vibration RMS outcomes instead of binary labels, enabling direction-sensible physical deltas.
- Universal `Dataset` contract (`Dataset.from_arrays`) so any vibration dataset
  plugs in without a bespoke loader.
- Validation scripts for CWRU (`validate_cwru.py`) and SEU gears
  (`validate_seu.py`), and a cross-condition robustness script.
- Test suite covering the physics, symbolic, causal, consensus, and provider
  layers.

### Changed
- Symbolic layer rewritten to verify predictions against physics via a provider
  and a configurable taxonomy (replacing a hardcoded, CWRU-specific lookup).
- Causal layer scoped honestly to Pearl Rung 2 (effect of operating condition)
  on a corrected DAG; the counterfactual layer carries Rung 3.
- Changed rung-3 continuous degradation outcome (RMS) for unit-level counterfactuals (#20)

### Notes
- Interfaces may change. Not intended for
  safety-critical deployment without independent validation.

## History

CANSYD began as a five-layer causal-neuro-symbolic diagnosis prototype and was
rebuilt into a domain-agnostic framework: the physics-verification layer was made
real (independent characteristic-frequency checks that can override the neural
prediction), the causal claims were scoped honestly, and the system was
reorganized into a deployable package.

## Project history


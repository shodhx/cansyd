# Changelog

All notable changes to CNSD are documented here. The format is based on
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
  by `cnsd/builder.py`.
- Real Pearl Rung-3 counterfactuals via DoWhy `gcm`, with a local-sensitivity
  fallback when DoWhy is unavailable.
- Universal `Dataset` contract (`Dataset.from_arrays`) so any vibration dataset
  plugs in without a bespoke loader.
- Validation scripts for CWRU (`validate_run.py`) and SEU gears
  (`validate_seu.py`), and a cross-condition robustness script.
- Test suite covering the physics, symbolic, causal, consensus, and provider
  layers.

### Changed
- Symbolic layer rewritten to verify predictions against physics via a provider
  and a configurable taxonomy (replacing a hardcoded, CWRU-specific lookup).
- Causal layer scoped honestly to Pearl Rung 2 (effect of operating condition)
  on a corrected DAG; the counterfactual layer carries Rung 3.

### Notes
- Pre-1.0 research software. Interfaces may change. Not intended for
  safety-critical deployment without independent validation.

## History

CNSD began as a five-layer causal-neuro-symbolic diagnosis prototype and was
rebuilt into a domain-agnostic framework: the physics-verification layer was made
real (independent characteristic-frequency checks that can override the neural
prediction), the causal claims were scoped honestly, and the system was
reorganized into a deployable package.

## Project history


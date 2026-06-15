# CNSD — Causal Neuro-Symbolic Diagnosis

Code for the CNSD framework: a five-layer bidirectional pipeline for
safety-critical fault diagnosis that operationalises all three rungs of
Pearl's causal hierarchy (association, intervention, counterfactual). It is
evaluated on CWRU bearings, NASA CMAPSS turbofans, and MIT-BIH ECG.

This repo is the modular version of the original research notebook.

## Install

```bash
pip install -r requirements.txt
# or, as a package:
pip install -e .
```

Python 3.11 is recommended (matches the pinned TensorFlow 2.15 build).

## Data

The loaders cache raw data under `data/raw/`:

- **CWRU** — downloaded automatically from Case Western (40 `.mat` files).
- **CMAPSS** — place `train_FD001.txt` etc. in `data/raw/cmapss/`.
- **MIT-BIH** — pulled via `wfdb` from PhysioNet.
- **MFPT** — currently a synthetic fallback only (see note below).

If real data is missing, CMAPSS and MFPT fall back to a reproducible
synthetic generator. The loaders print a clear `[SYNTHETIC]` warning in
that case — numbers from synthetic fallbacks must not be reported as
real-data results.

## Run

```bash
python main.py               # full cross-domain pipeline (CWRU/CMAPSS/MIT-BIH/MFPT)
python reproduce_results.py  # lighter reproduction: train CNN, causal ATEs, pipeline, counterfactual
python run_kaggle.py         # one-command runner for Kaggle/Colab
```

## What to expect

The notebook's main reported numbers (Protocol B — cross-load, train on
loads 0/1/2, test on unseen load 3):

| Quantity                | Notebook value |
|-------------------------|----------------|
| CWRU CNN F1 (Protocol B)| ~0.88          |
| WDCNN baseline F1       | ~0.8815        |
| CWRU causal ATE         | significant, p<0.001, placebo >100x |
| Cross-domain ATEs       | significant in all of CWRU/CMAPSS/MIT-BIH |

Exact ATE magnitudes depend on the treatment definition. This repo uses
vibration RMS as the treatment (matching the pipeline narrative); re-run on
GPU to confirm the numbers land where your paper reports them.

## Layout

```
core/        architecture, causal inference, rules, counterfactual, pipeline
data/        dataset loaders
continual/   CML and CCR-LoRA continual learning
eval/        baselines (WDCNN, IRM) and metrics (ECE, Proposition A)
configs/     default hyperparameters
```

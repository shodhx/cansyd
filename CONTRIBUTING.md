# Contributing to CNSD

Thanks for your interest in CNSD (Causal Neuro-Symbolic Diagnosis). This guide
covers how to set up a development environment, the checks your change must pass,
and the review conventions we follow.

CNSD is maintained by two people (see [MAINTAINERS.md](MAINTAINERS.md)). Changes
land through pull requests reviewed by a maintainer.

## Development setup

CNSD requires **Python 3.11+**.

Clone the repo and install it in editable mode. The core install is
deliberately lightweight — no TensorFlow, no DoWhy:

```bash
git clone https://github.com/abhiprd2000/CNSD
cd CNSD
python -m pip install --upgrade pip
pip install -e .
```

Optional feature sets are installed as extras:

```bash
pip install -e ".[perception]"      # 1D CNN / S-JEPA backend (TensorFlow, Keras)
pip install -e ".[counterfactual]"  # Rung-3 counterfactual layer (DoWhy, pandas, networkx)
pip install -e ".[all]"             # everything
```

For reproducing the validation experiments against the exact pinned environment,
use `requirements.txt` instead.

## The core-vs-extras contract

A hard rule in this codebase: **the core package must import without
TensorFlow.** The causal, physics, symbolic, config, SCM, dataset, and diagnosis
layers all work on a core-only install. Only the perception backend (the CNN)
requires the `perception` extra, and only the counterfactual layer requires
`counterfactual`.

Concretely, this must always hold on a core-only install:

```python
import cnsd
from cnsd import Dataset, PhysicsConfig, DiagnosisReport, CNSD  # all resolve, no TensorFlow
import cnsd.causal, cnsd.physics, cnsd.config, cnsd.scm          # import cleanly
```

CI enforces this in a dedicated job that asserts TensorFlow is absent and then
imports the dependency-light subpackages. If you add a heavy dependency to a
core module, that job will fail. Keep TensorFlow/DoWhy imports lazy (inside the
functions or methods that need them), not at module top level.

## Before you open a PR

Run the same checks CI runs. All three must be green.

**1. Lint and format (Ruff):**

```bash
pip install ruff
ruff check .
ruff format --check .
```

To auto-fix formatting before pushing: `ruff format .`

**2. Tests:**

```bash
pip install pytest
pytest test/ -v
```

The test suite in `test/` is TensorFlow-free by design — it covers the physics
math, the symbolic verdict logic, the causal (Rung-2) layer, the gear/spectral
providers, and consensus fusion. Please keep new tests TF-free where possible;
add them under `test/`.

**3. Import independence:** confirm the core package still imports without
TensorFlow (see the contract above).

## Project layout

The library lives in `cnsd/`, organized one concern per package:

- `cnsd/perception` — Layer 1: 1D CNN + S-JEPA (requires `perception` extra)
- `cnsd/symbolic`, `cnsd/physics` — Layer 2: physics verification and providers
- `cnsd/causal`, `cnsd/scm` — Layer 3: Pearl Rung-2 intervention + refutation
- `cnsd/counterfactual` — Layer 3B: Rung-3 (requires `counterfactual` extra)
- `cnsd/consensus` — Layer 4: verdict + confidence → actionable status
- `cnsd/datasets`, `cnsd/diagnosis` — data contract and pipeline orchestration

`validation/` holds the per-dataset validation runners and the multi-seed
benchmark. These are **not** shipped in the installed package — they're
development/research scripts. Run them as modules from the repo root, e.g.
`python -m validation.multi_seed_benchmark --dataset cwru`. Dataset paths can be
overridden with the `CNSD_DATA_CWRU`, `CNSD_DATA_PU`, and `CNSD_DATA_SEU`
environment variables.

## Pull request conventions

- Open a PR and request review from a maintainer. Tag **both** maintainers for
  changes that cross package boundaries or touch the public API
  (`cnsd/__init__.py`, `cnsd/diagnosis/system.py`).
- Keep each PR to one scoped change. Smaller PRs are reviewed faster and are
  easier to reason about.
- Don't rewrite published history on a shared branch. Avoid force-pushing amends
  or replacing an open PR wholesale — it orphans review threads and breaks
  auditability. Add follow-up commits instead.
- Write a clear description of what changed and why. If your change was proposed
  or shaped by someone else, credit them in the description and commit message.

## Reporting empirical results (integrity)

CNSD is a research framework, and the results it produces go into a paper. We
hold contributions that touch experiments or reported numbers to a specific
standard:

- **Report null and negative results honestly.** A saturated-regime null or a
  transfer that degrades is a valid, informative outcome — report it as such, at
  equal prominence with positive results. Do not quietly drop, reframe, or
  re-run experiments until they look favorable.
- **No overclaiming.** State what an experiment shows, and no more. Don't imply a
  layer drives accuracy when it provides a causal or stability perspective; don't
  present a demonstration as a discovered effect.
- **Results come from artifacts, not by hand.** Record experimental results as
  logged artifacts (e.g. JSON) produced by the run, not hand-transcribed into
  tables. The git commit timestamp is the authoritative date. Hand-edited result
  tables will be sent back in review.
- **Reviewers check for this.** PRs that add or modify results are reviewed for
  overclaims, misframing, and data-integrity issues before merge. This is not
  personal — it's how the project stays trustworthy.

## Questions

Open an issue or start a discussion on GitHub. For anything about who reviews
what, see [MAINTAINERS.md](MAINTAINERS.md).
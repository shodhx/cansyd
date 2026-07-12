# 🗺️ CANSYD Roadmap

> Where **Causal Neuro-Symbolic Diagnosis** is headed, and where you can jump in.

This roadmap is a living document, not a contract. Priorities shift as the community grows and as evidence comes in — and in the spirit of the project, we'd rather revise a plan honestly than quietly let it rot. If something here is stale, that's a bug: [open an issue](https://github.com/shodhx/cansyd/issues).

---

## How to read this

Work is grouped by **horizon**, following a convention that keeps roadmaps useful to contributors:

- **🔜 Now** — actively in progress, mostly maintainer-led. Getting us to the public release.
- **🙌 Next** — *not currently being worked on, and ready for you to pick up.* This is the section for new contributors.
- **🔭 Later** — direction and vision. Bigger ideas that need design before code.

Tasks in **Next** carry a difficulty signal:

| Label | Meaning |
|-------|---------|
| 🟢 **good first issue** | Self-contained, well-scoped, no deep context needed. A great first PR. |
| 🟡 **help wanted** | Substantial and valuable; some familiarity with the relevant layer helps. |
| 🔴 **needs design** | Worth doing, but let's agree on the approach in an issue before coding. |

> 💡 Many **Next** items are things we'd happily accept *now* if someone wrote them. If an item excites you, comment on (or open) the matching issue and claim it — see [How to get involved](#-how-to-get-involved).

---

## ✅ Where we are now

The core is built, tested, and validated. In place today:

- The full **five-layer pipeline** — perception → symbolic → causal → counterfactual → consensus — with a physics veto.
- **Physics verification** with a three-valued verdict, harmonic prominence, and adaptive tolerance.
- **Causal layer**: Pearl Rung-2 `do(Z)` intervention, CATE, cross-load invariance, and a DoWhy refutation suite.
- **Counterfactual layer**: Pearl Rung-3 on an invertible SCM, with a sensitivity-analysis fallback.
- **Cross-domain validation** on three bearing datasets (PU, XJTU-SY, CWRU) — see [`EXPERIMENTS.md`](EXPERIMENTS.md).
- A **pluggable provider registry**, a **dependency-light core**, CI (ruff + pytest), and a full set of community files.

See the [README status board](README.md#-project-status--roadmap) for the checklist view.

---

## 🔜 Now 


## 🙌 Next — where you can help

**This is the heart of the roadmap.** Each item below is real, scoped, and *not* currently being worked on. Pick one, claim it in an issue, and go. "Where to start" points you at the code.

### 🔬 Physics & new machinery

| Task | Why it matters | Label | Where to start |
|------|----------------|-------|----------------|
| **Gear-mesh physics provider** | The provider registry is built and `gear` is a stub; a real gear provider (mesh frequency, sidebands, hunting tooth) is the single biggest step toward a mechanism-agnostic verifier. The SEU gearbox failure in [`EXPERIMENTS.md`](EXPERIMENTS.md) documents exactly why it's needed. | 🔴 needs design | `cansyd/physics/gear.py`, `cansyd/physics/providers/gear.py` |
| **New dataset loaders** (MFPT, IMS, JNU, …) | The universal dataset contract means a new dataset is a small, self-contained loader — perfect first contribution, and each one strengthens the evidence base. | 🟢 good first issue | `cansyd/datasets/contract.py`, `cansyd/datasets/xjtusy.py` (reference) |
| **Improve the spectral fallback** | The zero-knowledge `spectral` provider diagnoses when geometry is unknown. Better peak-picking / cepstral cues would raise its recall. | 🟡 help wanted | `cansyd/physics/providers/spectral.py` |
| **Shaft/imbalance/misalignment families** | Order-domain signatures (1×, 2×, 3× shaft) would extend verification beyond bearing faults on the same signal. | 🔴 needs design | `cansyd/physics/`, provider registry |

### 🎯 Causal & counterfactual

| Task | Why it matters | Label | Where to start |
|------|----------------|-------|----------------|
| **Additional refuters** | More DoWhy refuters (bootstrap, unobserved-common-cause sensitivity) make the Rung-2 warrant harder to fool. | 🟡 help wanted | `cansyd/causal/refutation.py` |
| **Counterfactual outcome choices** | The Rung-3 outcome is vibration RMS; adding other model-independent severity measures (kurtosis, crest factor) as configurable outcomes would broaden the stability check. | 🟢 good first issue | `cansyd/counterfactual/rung3.py` |

### 🛠️ Developer experience & software

| Task | Why it matters | Label | Where to start |
|------|----------------|-------|----------------|
| **`cansyd` CLI** | A `cansyd diagnose <data>` command would let non-Python users run the pipeline from a terminal. | 🟡 help wanted | new `cansyd/cli.py`, wraps `cansyd/api.py` |
| **Visualization helpers** | A `cansyd.viz` module to plot the envelope spectrum with the characteristic-frequency comb and verdict overlay — great for reports and notebooks. | 🟢 good first issue | new module; `cansyd/physics/bearing.py` for the spectrum |
| **Config schema validation** | Validate `cansyd_config.yaml` against a schema with friendly error messages. | 🟢 good first issue | `cansyd/config.py` |
| **Expand the test suite** | More coverage on the physics and causal logic (which run without heavy deps) — always welcome. | 🟢 good first issue | `test/test_layers.py` |

### 📚 Documentation

| Task | Why it matters | Label | Where to start |
|------|----------------|-------|----------------|
| **Worked example notebooks** | End-to-end notebooks on a public dataset are often a contributor's first experience of the project. | 🟢 good first issue | `examples/` |
| **API reference site** | A small docs site (mkdocs / Sphinx) generated from docstrings. | 🟡 help wanted | `docs/` |
| **Tutorials for each layer** | Short guides: "reading a verdict", "interpreting a counterfactual". | 🟢 good first issue | `docs/` |

---

## 🔭 Later — vision & research

Bigger directions that shape where CANSYD goes. These need discussion and design before implementation; issues tagged `discussion` are the place for that.

- **A mechanism-agnostic verifier.** Bearings, gears, and shafts under one verification interface, so CANSYD checks any rotating-machinery diagnosis against the right physics automatically.
- **Physics-informed self-supervised perception.** A JEPA-style backbone that bakes characteristic-frequency structure into the learned representation, tightening the loop between perception and verification.
- **Online / streaming diagnosis.** Windowed, real-time verification for condition-monitoring deployments rather than batch evaluation.
- **Calibration-aware consensus.** Fuse the physics verdict with *calibrated* uncertainty, not raw softmax, and study how the veto threshold should adapt.
- **Multi-component machines.** Diagnose and verify combined bearing + gear signatures in a single drivetrain signal.

If you're doing research adjacent to any of these, we'd love to collaborate — open a `discussion` issue.

---

## 🧭 Non-goals (for now)

Being clear about scope keeps the project honest and focused:

- **CANSYD is not a new classifier.** The perception layer is deliberately a standard CNN; the contribution is verification, not peak accuracy. PRs that swap in a fancier backbone for its own sake are out of scope.
- **CANSYD does not claim domain adaptation.** It does not adapt the network or repair the raw signal under shift — it flags which predictions to trust. Framing that blurs this line will be sent back.
- **No fabricated or cherry-picked results, ever.** See the integrity norms below.
- **Non-rotating machinery** (e.g. reciprocating, structural) is out of current scope.

---

## 🤝 How to get involved

New here? Welcome — we're glad you found the project.

1. **Find something.** Browse **🙌 Next** above, or the [issue tracker](https://github.com/shodhx/cansyd/issues) filtered by `good first issue` / `help wanted`.
2. **Claim it.** Comment on the issue (or open one if it doesn't exist yet) describing what you plan to do. This avoids two people building the same thing.
3. **Build it.** Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow. In short: fork → branch → `ruff check . && ruff format .` and `pytest test/` → PR with a clear description → green CI.
4. **Ask.** Stuck or unsure if an idea fits? Open a `discussion` issue. Half-formed questions are welcome; that's how scope gets refined in the open.

### The one rule that never bends: integrity

CANSYD is a scientific tool, and its credibility is the whole point. Contributions are expected to uphold the project's core discipline:

- **Honest results.** Null, weak, and unflattering results are reported alongside the good ones — the CWRU null in [`EXPERIMENTS.md`](EXPERIMENTS.md) is a feature, not something to hide.
- **Real artifacts.** Benchmark numbers come from the run scripts, not hand-transcribed tables.
- **No overclaiming.** Say what the evidence supports, and no more.

If a change trades honesty for a better-looking number, it will not be merged — no matter how good the number looks.

---

<div align="center">
<sub>Have an idea that isn't here? <a href="https://github.com/shodhx/cansyd/issues">Open an issue</a> — the roadmap grows with its community.</sub>
</div>
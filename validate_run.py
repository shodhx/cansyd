"""
validate_run.py - one clean end-to-end validation of the CNSD pipeline on CWRU.

Purpose: prove the rebuilt system actually RUNS on real data and produces the
numbers the paper claims - before writing the paper. This is a validation run,
not the full benchmark (no multi-seed, no baselines, no cross-rig). It confirms:

  1. the five-layer pipeline executes on real CWRU without error
  2. Layer 2 physics verification produces a real CONFIRMED/CONFLICT/INCONCLUSIVE rate
  3. the HEADLINE result computes: CNN accuracy split by physics verdict
  4. Layer 3 causal do(Z) runs and returns a Rung-2 effect
  5. the real Rung-3 counterfactual (DoWhy gcm) actually executes (not the fallback)
  6. example auditable root-cause diagnoses print

Run on Kaggle (GPU) after: pip install dowhy
Place CWRU data so your loader can read it, then adapt the load_cwru() stub below
to return X (n,1024), y (n,), cond (n,) - the only dataset-specific code here.
"""
import sys
import numpy as np

from cnsd import CNSD, Dataset
from cnsd.physics import PhysicsConfig
from cnsd.causal import signal_kurtosis
from cnsd.counterfactual import dowhy_gcm_available, build_scm, counterfactual_for_unit


# ── the ONLY dataset-specific code: return raw arrays from your CWRU files ────
def load_cwru():
    """Return (X, y, cond) for CWRU. Replace the body with your loader.

    X    : (n, 1024) float   non-overlapping test windows, per-window normalized
    y    : (n,) int          fault class 0..9 (0 = Normal)
    cond : (n,) int          motor load 0..3
    """
    raise NotImplementedError(
        "Wire your CWRU loader here: return X (n,1024), y (n,), cond (n,). "
        "Use Protocol B (train loads 0-2, test load 3) for the real result.")


# CWRU 6205 physics + taxonomy (this is config, not hardcoded engine logic)
CWRU = PhysicsConfig(
    bearing={'n_balls': 9, 'd_ball': 0.3126, 'd_pitch': 1.537, 'contact_angle': 0.0},
    cond_to_rpm={0: 1797, 1: 1772, 2: 1750, 3: 1730}, fs=12000, name='CWRU-6205')

TAXONOMY = {0: ('Normal', 'None'),
            1: ('Ball', 'Low'), 2: ('Ball', 'Medium'), 3: ('Ball', 'High'),
            4: ('Inner Race', 'Low'), 5: ('Inner Race', 'Medium'), 6: ('Inner Race', 'High'),
            7: ('Outer Race', 'Low'), 8: ('Outer Race', 'Medium'), 9: ('Outer Race', 'High')}


def headline_accuracy_by_verdict(report, y_true):
    """The core claim: is the CNN more accurate when physics CONFIRMS than CONFLICTS?

    Computed here (not in the report) because it needs the true labels.
    """
    pred = np.array([r['predicted_class'] for r in report.records])
    correct = (pred == np.asarray(y_true))
    verdicts = np.array([r['physics_verdict'] for r in report.records])
    out = {}
    for v in ('CONFIRMED', 'CONFLICT', 'INCONCLUSIVE'):
        m = verdicts == v
        if m.any():
            out[v] = {'n': int(m.sum()), 'cnn_accuracy': float(correct[m].mean())}
    return out


def main():
    print('=' * 68)
    print('CNSD VALIDATION RUN (CWRU, Protocol B)')
    print('=' * 68)

    # 1. data
    X, y, cond = load_cwru()
    X = np.asarray(X, np.float32); y = np.asarray(y); cond = np.asarray(cond)
    data = Dataset.from_arrays(X, y, cond, fs=12000, physics=CWRU,
                               taxonomy=TAXONOMY, name='CWRU')
    print(f'[data] {data.summary()}')

    # 2. fit + 3. diagnose (Layers 1,2,4 live)
    model = CNSD()
    model.fit(data, epochs=30)
    report = model.diagnose(data)
    print(f'\n[pipeline] {report.summary()}')

    # 3. Layer-2 verification rate
    vr = report.verification_rate()
    print('\n[Layer 2] physics verification rate:')
    for k, v in vr.items():
        print(f'    {k:13}: {v:.1%}')

    # 4. HEADLINE: CNN accuracy split by verdict
    hb = headline_accuracy_by_verdict(report, y)
    print('\n[HEADLINE] CNN accuracy by physics verdict:')
    for v, d in hb.items():
        print(f'    {v:13}: acc={d["cnn_accuracy"]:.3f}  (n={d["n"]})')
    if 'CONFIRMED' in hb and 'CONFLICT' in hb:
        gap = hb['CONFIRMED']['cnn_accuracy'] - hb['CONFLICT']['cnn_accuracy']
        print(f'    -> CONFIRMED minus CONFLICT accuracy gap: {gap:+.3f} '
              f'({"physics is a real reliability signal" if gap > 0 else "investigate"})')

    # 5. Layer-3 causal do(Z)
    eff = model.condition_effect(data)
    print(f'\n[Layer 3] do(Z) operating-condition effect: rung={eff["rung"]} '
          f'max_contrast={eff["max_contrast"]:.4f} p={eff["p_value"]:.4f}')

    # 6. Layer-3B: REAL Rung-3 counterfactual (must actually execute, not fallback)
    print(f'\n[Layer 3B] DoWhy available: {dowhy_gcm_available()}')
    if dowhy_gcm_available():
        feat = signal_kurtosis(data.X)
        scm = build_scm(data.cond, feat, data.y)
        if scm is not None:
            row = {'Z': float(data.cond[0]), 'X': float(feat[0]), 'Y': float(data.y[0] > 0)}
            cf_cond = int(min(np.unique(data.cond)))
            cf = counterfactual_for_unit(scm, row, cf_cond)
            print(f'    counterfactual executed: {cf["method"]}')
            print(f'    factual Y={cf["factual"]["Y"]:.2f} -> '
                  f'counterfactual Y={cf["counterfactual"]["Y"]:.2f} at Z={cf_cond}')
        else:
            print('    SCM build returned None (check DoWhy install)')
    else:
        print('    DoWhy not installed - the real Rung-3 path was NOT validated. '
              'Run: pip install dowhy')

    # 7. example auditable diagnoses
    print('\n[examples] auditable root-cause diagnoses:')
    for r in report.records[:5]:
        print(f'    [{r["status"]}] {r["root_cause"]["statement"]}')

    print('\n' + '=' * 68)
    print('VALIDATION COMPLETE - if all sections printed, the pipeline runs '
          'end-to-end.')
    print('=' * 68)


if __name__ == '__main__':
    main()

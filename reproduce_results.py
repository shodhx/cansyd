"""
reproduce_results.py - runs the full CNSD five-layer pipeline and regenerates
the paper's results, with the physics-grounded layers in the LIVE path.

The five layers, all actually executed here:
  Layer 1  (Perception)  : 1D CNN + S-JEPA backbone
  Layer 2  (Symbolic)    : physics verification of every prediction against the
                           bearing characteristic frequencies -> CONFIRMED /
                           CONFLICT / INCONCLUSIVE  (core/rules.py via pipeline)
  Layer 3  (Causal R2)   : effect of operating condition do(Z) + backdoor ATE of
                           physical vs learned treatments + invariance
  Layer 3B (Sensitivity) : local sensitivity analysis (NOT Pearl Rung 3)
  Layer 4  (Consensus)   : status from CNN confidence + physics verdict

The headline result is the PHYSICS-VERIFICATION RATE: how often the network's
prediction is independently confirmed by the bearing physics, and how often the
two conflict. That is the causally/physically grounded number the pipeline now
actually produces - not a relabelled feature-norm.
"""
import numpy as np

from data.loaders import load_cwru_all
from core.architecture import train_jepa_backbone, patchify
from core.rules import rule_engine
from core.pipeline import CNSDPipeline
from core.causal import (analyze_causal, cate_by_group, causal_invariance_across_loads,
                         extract_feature_norms, signal_kurtosis,
                         intervention_effect_of_condition)
from core.sensitivity import local_sensitivity
from core.scm import describe as describe_scm
from core.hyperparameters import report as report_hyperparams
from eval.classification import train_cnn, evaluate_protocol_b
from eval.baseline import run_published_baselines, run_irm


def summarize_pipeline(records, y_test):
    """Layer 2 + Layer 4 result: physics-verification and status breakdown."""
    verdicts = [r['physics_verdict'] for r in records]
    statuses = [r['status'] for r in records]
    n = len(records)
    conf = verdicts.count('CONFIRMED')
    confl = verdicts.count('CONFLICT')
    inconc = verdicts.count('INCONCLUSIVE')

    print('\n--- LAYER 2: PHYSICS VERIFICATION (independent of the CNN) ---')
    print(f'  CONFIRMED   : {conf:5d}  ({conf/n:.1%})  physics agrees with the network')
    print(f'  CONFLICT    : {confl:5d}  ({confl/n:.1%})  physics disagrees -> manual review')
    print(f'  INCONCLUSIVE: {inconc:5d}  ({inconc/n:.1%})  no fault frequency prominent')

    # Is the physics verdict actually informative about correctness?
    correct = np.array([r['cnn_class'] == int(y) for r, y in zip(records, y_test)])
    conf_mask = np.array([v == 'CONFIRMED' for v in verdicts])
    confl_mask = np.array([v == 'CONFLICT' for v in verdicts])
    if conf_mask.any() and confl_mask.any():
        print(f'\n  CNN accuracy WHEN physics CONFIRMS : {correct[conf_mask].mean():.3f}')
        print(f'  CNN accuracy WHEN physics CONFLICTS: {correct[confl_mask].mean():.3f}')
        print('  (if confirm-accuracy > conflict-accuracy, the physics layer is '
              'a genuine, independent reliability signal)')

    print('\n--- LAYER 4: CONSENSUS STATUS ---')
    for st in ['HIGH_CONFIDENCE', 'RELIABLE', 'UNCERTAIN', 'MANUAL_REVIEW']:
        c = statuses.count(st)
        print(f'  {st:16}: {c:5d}  ({c/n:.1%})')


def main():
    print('=' * 70)
    print('CNSD - FULL FIVE-LAYER PIPELINE (Protocol B, cross-load)')
    print('=' * 70)
    print(describe_scm())

    # ── Phase 1: data ───────────────────────────────────────────────────────
    X_train, y_train, load_train, X_test, y_test, load_test = load_cwru_all()
    print(f'\nTrain {X_train.shape} | Test {X_test.shape}')

    # ── Phase 2: Layer 1 - backbone + CNN ───────────────────────────────────
    encoder, probe, scaler = train_jepa_backbone(X_train, y_train, epochs=15)
    cnn = train_cnn(X_train, y_train, seed=42, epochs=30)

    # ── Phase 3: RUN THE FULL PIPELINE on the test set (Layers 1-4 live) ─────
    print('\n[1/6] FIVE-LAYER PIPELINE ON TEST SET')
    pipeline = CNSDPipeline(cnn, probe, encoder, patchify, rule_engine)
    records = pipeline.predict(X_test, load_test)
    summarize_pipeline(records, y_test)

    # show a few fully-worked auditable diagnoses
    print('\n--- EXAMPLE AUDITABLE DIAGNOSES ---')
    for r in records[:3]:
        print(f"  class={r['cnn_class']} ({r['diagnosis']}, {r['severity']}) "
              f"conf={r['cnn_confidence']:.2f} | verdict={r['physics_verdict']} "
              f"| status={r['status']}")
        print(f"    {r['root_cause']['statement']}")

    # ── Phase 4: Layer 3 - honest causal analysis ───────────────────────────
    print('\n[2/6] CLASSIFICATION (Protocol B)')
    evaluate_protocol_b(X_train, y_train, X_test, y_test)

    print('\n[3/6] PUBLISHED BASELINES')
    run_published_baselines(X_train, y_train, X_test, y_test)
    run_irm(X_train, y_train, load_train, X_test, y_test)

    print('\n[4/6] LAYER 3 - CAUSAL (Rung 2, honest)')
    # (a) the genuine do(Z) intervention: effect of operating condition
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    load_all = np.concatenate([load_train, load_test])
    doZ = intervention_effect_of_condition(y_all, load_all)
    print(f"  do(Z) effect of operating condition on fault rate: "
          f"max contrast={doZ['max_contrast']:.4f}  p={doZ['p_value']:.4f}")
    print(f"  per-condition fault rate: {doZ['per_condition_fault_rate']}")

    # optional DoWhy identification + refutation suite (graceful if not installed)
    from core.dowhy_refutation import refute_condition_effect, dowhy_available
    print(f"\n  DoWhy refutation suite ({'available' if dowhy_available() else 'not installed - using builtin'}):")
    ref = refute_condition_effect(load_all, y_all)
    if ref['backend'] == 'dowhy':
        print(f"    estimate: {ref['estimate']:+.4f}")
        for name, r in ref['refutations'].items():
            ne = r.get('new_effect')
            print(f"    refute[{name}]: new_effect={ne:+.4f}" if ne == ne else f"    refute[{name}]: {r.get('error','')}")
    else:
        print(f"    builtin placebo: p={ref['placebo_p_value']:.4f} ratio={ref['placebo_ratio']:.1f}x")
        print(f"    ({ref['note']})")

    # (b) physical vs learned treatment (why physical is reproducible)
    feat_norms_tr = extract_feature_norms(cnn, X_train)
    res_fn = analyze_causal(feat_norms_tr, y_train, load_train, 'CWRU')
    kurt_tr = signal_kurtosis(X_train)
    res_ph = analyze_causal(kurt_tr, y_train, load_train, 'CWRU')
    print('\n  backdoor-adjusted ATE (treatment -> fault | load):')
    print(f"    feature-norm (learned) : {res_fn['ate']:+.4f}  "
          f"placebo {res_fn['placebo_ratio']:.1f}x  p={res_fn['p_value']:.4f}")
    print(f"    kurtosis (physical)    : {res_ph['ate']:+.4f}  "
          f"placebo {res_ph['placebo_ratio']:.1f}x  p={res_ph['p_value']:.4f}")

    kurt_all = signal_kurtosis(X_all)
    probs_all = cnn.predict(X_all, batch_size=64, verbose=0)
    fault_prob_all = 1.0 - probs_all[:, 0]
    eps = 1e-6
    fault_logit_all = np.log((fault_prob_all + eps) / (1.0 - fault_prob_all + eps))
    cate_by_group(kurt_all, fault_logit_all, y_all, load_all)
    causal_invariance_across_loads(kurt_all, y_all, load_all)

    # ── Phase 5: Layer 3B - sensitivity (honest, not Rung 3) ────────────────
    print('\n[5/6] LAYER 3B - LOCAL SENSITIVITY (not Pearl Rung 3)')
    sens = [local_sensitivity(X_test[i], load_test[i], load_test[i] - 2)['sensitivity']
            for i in range(min(len(X_test), 500))]
    print(f"  mean prediction sensitivity to a 2-step load perturbation: "
          f"{np.mean(sens):.4f}")
    print(f"  most-fragile decile sensitivity: {np.percentile(sens, 90):.4f}")
    print('  (high-sensitivity samples are the fragile predictions to flag)')

    # ── Phase 6: hyperparameter honesty table ───────────────────────────────
    print('\n[6/6] HYPERPARAMETERS (calibrated vs design)')
    report_hyperparams()

    print('\n' + '=' * 70)
    print('Reproduction complete - all five layers executed on real data.')
    print('=' * 70)


if __name__ == '__main__':
    main()

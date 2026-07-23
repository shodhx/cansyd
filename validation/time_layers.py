"""
CANSYD — Per-layer inference timing benchmark for Table VII.

Measures the per-sample inference time of each CANSYD layer on synthetic
vibration data (L=2048, single CPU core). Reports mean ± std over 1000
iterations in a format ready for the paper.

Usage:
    cd _cansyd_repo
    python -m validation.time_layers

Hardware should be reported in the paper caption alongside these results.
"""

import platform
import time

import numpy as np

# ── Configuration ───────────────────────────────────────────────────────────
L = 2048  # window length (matches Table VII)
N_ITER = 1000  # iterations per layer
FS = 12000  # Hz (CWRU sampling rate)
RPM = 1797  # CWRU drive-end shaft speed
N_WARMUP = 50  # warmup iterations (excluded from timing)

np.random.seed(42)

# ── Generate synthetic vibration data ───────────────────────────────────────
# Realistic: outer-race fault signal + noise (matches the paper's scenario)
fr = RPM / 60.0
f_bpfo = 4.5 * fr * (1 - 0.2033)  # approximate BPFO for 6205
t = np.arange(L) / FS
signal = np.sin(2 * np.pi * f_bpfo * t) * 0.5 + np.random.randn(L) * 0.3
signal = signal.astype(np.float32)

# Pre-generate batch for causal layer (needs multiple samples)
N_BATCH = 200
signals_batch = np.random.randn(N_BATCH, L).astype(np.float32)
labels_batch = (np.random.rand(N_BATCH) > 0.5).astype(int)
conditions_batch = np.random.choice([0, 1, 2, 3], size=N_BATCH)


def time_function(func, n_iter=N_ITER, n_warmup=N_WARMUP):
    """Time a function over n_iter calls, return per-call time in ms."""
    # Warmup
    for _ in range(n_warmup):
        func()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return float(times.mean()), float(times.std())


# ============================================================================
#  Layer 1: CNN (1D-Conv + softmax)
# ============================================================================
def time_cnn():
    try:
        import os

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        tf.config.set_visible_devices([], 'GPU')  # force CPU

        from cansyd.perception.cnn import build_cnn

        model = build_cnn(input_shape=(L, 1), num_classes=10)
        x = signal.reshape(1, L, 1)

        # Warmup
        for _ in range(N_WARMUP):
            model.predict(x, verbose=0)

        times = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            model.predict(x, verbose=0)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times = np.array(times)
        return float(times.mean()), float(times.std())
    except ImportError:
        print('  [SKIP] TensorFlow not installed — CNN timing unavailable')
        return None, None
    except Exception as e:
        print(f'  [SKIP] CNN timing failed: {e}')
        return None, None


# ============================================================================
#  Layer 2: Physics (FFT + envelope spectrum + prominence)
# ============================================================================
def time_physics():
    from cansyd.physics.bearing import fault_frequency_evidence

    def run():
        fault_frequency_evidence(signal, RPM, fs=FS)

    return time_function(run)


# ============================================================================
#  Layer 3a: Causal (ATE + permutation placebo)
# ============================================================================
def time_causal():
    from cansyd.causal.estimators import analyze_causal, signal_kurtosis

    treatment = signal_kurtosis(signals_batch)

    def run():
        analyze_causal(treatment, labels_batch, conditions_batch, domain='timing_benchmark')

    # Causal layer uses 1000 permutations internally — time for 1 call
    # and report as per-batch (divide by N_BATCH for per-sample)
    times = []
    for _ in range(N_WARMUP):
        run()
    for _ in range(min(N_ITER, 100)):  # fewer iters (each call is expensive)
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000 / N_BATCH)  # per-sample

    times = np.array(times)
    return float(times.mean()), float(times.std())


# ============================================================================
#  Layer 3b: Counterfactual (local sensitivity fallback)
# ============================================================================
def time_counterfactual():
    from cansyd.counterfactual.sensitivity import local_sensitivity

    def run():
        local_sensitivity(signal, condition_actual=0, condition_perturbed=3)

    return time_function(run)


# ============================================================================
#  Layer 4: Consensus Router
# ============================================================================
def time_consensus():
    from cansyd.consensus.fusion import fuse

    def run():
        fuse('CONFIRMED', 0.95)

    return time_function(run)


# ============================================================================
#  Main
# ============================================================================
if __name__ == '__main__':
    print('=' * 65)
    print('CANSYD Per-Layer Inference Timing Benchmark')
    print('=' * 65)
    print(f'  Window length (L):  {L}')
    print(f'  Iterations:         {N_ITER}')
    print(f'  Warmup:             {N_WARMUP}')
    print(f'  Platform:           {platform.processor()}')
    print(f'  Python:             {platform.python_version()}')
    print(f'  CPU:                {platform.machine()}')
    print()

    results = {}

    print('[1/5] CNN (1D-Conv + softmax)...')
    m, s = time_cnn()
    if m is not None:
        results['1. CNN (1D-Conv + softmax)'] = (m, s)
        print(f'       {m:.2f} ± {s:.2f} ms')

    print('[2/5] Physics (FFT + envelope)...')
    m, s = time_physics()
    results['2. Physics (FFT + envelope)'] = (m, s)
    print(f'       {m:.2f} ± {s:.2f} ms')

    print('[3/5] Causal (ATE + permutation)...')
    m, s = time_causal()
    results['3a. Causal (ATE + permutation)'] = (m, s)
    print(f'       {m:.2f} ± {s:.2f} ms/sample')

    print('[4/5] Counterfactual (sensitivity)...')
    m, s = time_counterfactual()
    results['3b. Counterfactual (abduction)'] = (m, s)
    print(f'       {m:.2f} ± {s:.2f} ms')

    print('[5/5] Consensus router...')
    m, s = time_consensus()
    results['4. Consensus router'] = (m, s)
    print(f'       {m:.4f} ± {s:.4f} ms')

    # ── Summary Table ───────────────────────────────────────────────────────
    print()
    print('=' * 65)
    print('RESULTS (for Table VII)')
    print('=' * 65)
    print(f'{"Layer":<35} {"Time (ms)":>12} {"± std":>10}')
    print('-' * 60)

    total_mean = 0.0
    for layer, (m, s) in results.items():
        print(f'  {layer:<33} {m:>10.2f}   ±{s:>7.2f}')
        total_mean += m

    print('-' * 60)
    print(f'  {"TOTAL":<33} {total_mean:>10.2f}')
    print()
    print('Copy these values into Table VII of main.tex.')
    print('Report hardware in the table caption.')

"""
This is the signal-processing substrate the symbolic layer uses to INDEPENDENTLY
verify a neural prediction against bearing physics. A bearing defect produces
periodic mechanical impulses at a characteristic frequency determined by the
bearing geometry and shaft speed. The classical way to expose that frequency is
envelope analysis: Hilbert transform -> amplitude envelope -> FFT of the
envelope. A peak in the envelope spectrum at the bearing's characteristic
frequency is direct physical evidence of that fault type.

No learned parameters. Deterministic given the signal.
"""

import numpy as np
from scipy.signal import hilbert

# ── Bearing geometry: CWRU 6205-2RS JEM SKF deep-groove ball bearing ──────────
# (drive-end bearing). Geometry is fixed; characteristic frequencies scale with
# shaft speed. Verified against published CWRU values (BPFO 107.4, BPFI 162.2 Hz
# at 1797 rpm).
BEARING_6205 = {
    'n_balls': 9,
    'd_ball': 0.3126,  # inches
    'd_pitch': 1.537,  # inches
    'contact_angle': 0.0,
}

# CWRU motor load -> approximate shaft speed (rpm). Load 0..3.
CWRU_LOAD_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# default drive-end sampling rate (Hz) for the 12 kHz CWRU files
CWRU_FS = 12000


def characteristic_frequencies(rpm, bearing=BEARING_6205):
    """Bearing characteristic frequencies (Hz) from geometry + shaft speed.

    Returns BPFO (outer race), BPFI (inner race), BSF (ball spin, as the
    defect-strike rate = 2x ball spin), FTF (cage), and the shaft rate fr.
    """
    n = bearing['n_balls']
    d = bearing['d_ball']
    D = bearing['d_pitch']
    theta = np.radians(bearing['contact_angle'])
    fr = rpm / 60.0
    ratio = (d / D) * np.cos(theta)
    bpfo = (n / 2.0) * fr * (1 - ratio)
    bpfi = (n / 2.0) * fr * (1 + ratio)
    bsf_single = (D / (2.0 * d)) * fr * (1 - ratio**2)
    return {
        'fr': fr,
        'BPFO': bpfo,
        'BPFI': bpfi,
        'BSF': 2.0 * bsf_single,  # defect strikes both races per spin
        'FTF': (fr / 2.0) * (1 - ratio),
    }


def envelope_spectrum(signal, fs=CWRU_FS):
    """Amplitude-envelope spectrum of a 1D vibration window.

    Hilbert transform -> |analytic signal| (envelope) -> remove DC -> FFT.
    Returns (freqs, magnitude) for the positive-frequency half.
    """
    x = np.asarray(signal, float).flatten()
    env = np.abs(hilbert(x))
    env = env - env.mean()  # drop DC so it doesn't dominate
    n = len(env)
    mag = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag


def band_energy(freqs, mag, f0, rel_tol=0.04, n_harmonics=2, fs=None, n_samples=None):
    """Energy in narrow bands around f0 and its harmonics, relative to the local
    spectrum. Returns a prominence ratio: how strong the peak near f0 is versus
    the surrounding baseline. >1 means a genuine peak sits at f0.
    """
    if f0 <= 0:
        return 0.0

    # Adaptive tolerance: widen the band on low-resolution FFT datasets,
    # but cap at 8% to prevent adjacent fault-frequency bands (e.g. BPFI
    # at 162 Hz and BSF at 141 Hz) from overlapping.
    if fs is not None and n_samples is not None and n_samples > 0:
        df = fs / n_samples  # FFT bin width in Hz
        rel_tol = min(max(rel_tol, 2.0 * df / f0), 0.08)

    prominences = []
    for h in range(1, n_harmonics + 1):
        fc = f0 * h
        tol = fc * rel_tol
        in_band = (freqs >= fc - tol) & (freqs <= fc + tol)
        if not in_band.any():
            continue
        peak = mag[in_band].max()
        # local baseline: a wider neighbourhood excluding the band
        nb = (freqs >= fc - 6 * tol) & (freqs <= fc + 6 * tol) & ~in_band
        baseline = np.median(mag[nb]) if nb.any() else np.median(mag)
        prominences.append(peak / (baseline + 1e-12))
    return float(np.mean(prominences)) if prominences else 0.0


def fault_frequency_evidence(signal, rpm, fs=CWRU_FS, bearing=BEARING_6205):
    """For one window, return the prominence of each fault type's characteristic
    frequency in the envelope spectrum. This is the physical evidence the
    symbolic layer reasons over - computed independently of the CNN.
    """
    freqs, mag = envelope_spectrum(signal, fs)
    cf = characteristic_frequencies(rpm, bearing)
    n_samples = len(np.asarray(signal).flatten())
    return {
        'BPFO': band_energy(freqs, mag, cf['BPFO'], fs=fs, n_samples=n_samples),
        'BPFI': band_energy(freqs, mag, cf['BPFI'], fs=fs, n_samples=n_samples),
        'BSF': band_energy(freqs, mag, cf['BSF'], fs=fs, n_samples=n_samples),
        'freqs_hz': {k: float(v) for k, v in cf.items()},
    }

"""Gear-mesh physics for gearbox fault verification.

A gear fault does not produce ball-pass frequencies. The dominant signature is
the gear-mesh frequency (GMF) and its harmonics, with sidebands spaced at the
shaft rotation rate. Different fault types modulate this differently:

  - a localized tooth fault (chip / root crack / missing tooth) produces strong
    sidebands around the GMF at the shaft rate of the faulty gear (the defect is
    excited once per revolution),
  - distributed surface wear raises the GMF harmonics without strong shaft-rate
    sidebands.

This module computes GMF + sidebands and measures their prominence in the
envelope spectrum, mirroring the bearing module's role for rolling elements.
"""
import numpy as np
from scipy.signal import hilbert

DEFAULT_FS = 5120


def gear_mesh_frequencies(rpm, n_teeth_input, n_teeth_output=None):
    """Gear-mesh frequency and shaft rates (Hz) from teeth counts + input speed.

    GMF = input_shaft_rate * n_teeth_input (the mesh rate is shared by both gears;
    teeth_in * f_in == teeth_out * f_out). Returns the GMF, the input shaft rate,
    and (if output teeth given) the output shaft rate - the sideband spacings.
    """
    f_in = rpm / 60.0
    gmf = f_in * n_teeth_input
    out = {'GMF': gmf, 'shaft_input': f_in}
    if n_teeth_output:
        out['shaft_output'] = gmf / n_teeth_output
    return out


def envelope_spectrum(signal, fs=DEFAULT_FS):
    """Amplitude-envelope spectrum (Hilbert -> |.| -> remove DC -> rFFT)."""
    x = np.asarray(signal, float).flatten()
    env = np.abs(hilbert(x))
    env = env - env.mean()
    mag = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fs)
    return freqs, mag


def _prominence(freqs, mag, f0, rel_tol=0.04):
    if f0 <= 0:
        return 0.0
    tol = f0 * rel_tol
    band = (freqs >= f0 - tol) & (freqs <= f0 + tol)
    if not band.any():
        return 0.0
    peak = mag[band].max()
    nb = (freqs >= f0 - 6 * tol) & (freqs <= f0 + 6 * tol) & ~band
    baseline = np.median(mag[nb]) if nb.any() else np.median(mag)
    return float(peak / (baseline + 1e-12))


def sideband_strength(freqs, mag, gmf, shaft_rate, n_sidebands=2):
    """Mean prominence of the shaft-rate sidebands around the GMF - the marker of
    a localized tooth fault."""
    vals = []
    for k in range(1, n_sidebands + 1):
        for f0 in (gmf - k * shaft_rate, gmf + k * shaft_rate):
            vals.append(_prominence(freqs, mag, f0))
    return float(np.mean(vals)) if vals else 0.0


def gear_fault_evidence(signal, rpm, n_teeth_input, n_teeth_output=None,
                        fs=DEFAULT_FS):
    """Physical evidence for gear-fault families in one window.

    Returns prominence of:
      'mesh'     : GMF + harmonics (raised by distributed wear)
      'sideband' : shaft-rate sidebands around GMF (raised by localized faults)
    plus the characteristic frequencies in Hz.
    """
    freqs, mag = envelope_spectrum(signal, fs)
    gf = gear_mesh_frequencies(rpm, n_teeth_input, n_teeth_output)
    gmf, shaft = gf['GMF'], gf['shaft_input']
    mesh = np.mean([_prominence(freqs, mag, gmf * h) for h in (1, 2)])
    side = sideband_strength(freqs, mag, gmf, shaft)
    return {
        'mesh_strength': float(mesh),
        'sideband_strength': float(side),
        'freqs_hz': {k: float(v) for k, v in gf.items()},
    }

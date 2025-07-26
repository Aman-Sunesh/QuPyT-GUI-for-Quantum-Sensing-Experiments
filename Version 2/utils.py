# utils.py

# ────────────────────────────────────────────────────────────────
# Utility functions for fitting Optically Detected Magnetic Resonance
# (ODMR) spectra. Provides standard lineshape models (Lorentzian and
# Gaussian) used throughout QuPyt for data fitting.
# ────────────────────────────────────────────────────────────────

import numpy as np

def lorentzian(x, x0, gamma, A, y0):
    """Lorentzian line shape for fitting ODMR dips.
    Args:
        x: Frequency values (GHz)
        x0: Center frequency
        gamma: Half-width at half-maximum (HWHM)
        A: Amplitude (negative for dip)
        y0: Baseline offset
    Returns:
        Lorentzian curve evaluated at x
    """
    return y0 + A * (gamma**2 / ((x - x0)**2 + gamma**2))

def gaussian(x, mu, sigma, A, y0):
    """Gaussian line shape for alternative dip fitting.
    Args:
        x: Frequency values
        mu: Mean (center)
        sigma: Standard deviation
        A: Amplitude
        y0: Baseline offset
    Returns:
        Gaussian curve evaluated at x
    """
    return y0 + A * np.exp(-((x - mu)**2) / (2*sigma**2))

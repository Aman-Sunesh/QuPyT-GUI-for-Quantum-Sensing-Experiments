"""Analyze Rabi data: fit one trace, then fit Rabi frequency against sqrt(power) for an antenna.

Procedure: docs/antenna-test-procedure.md. The quantity that characterizes an antenna is
the slope of Rabi frequency against sqrt(incident power), in MHz per sqrt(W).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from . import nv


def rabi_model(t, offset, amp, f, tau, phi):
    """offset + amp * exp(-t/tau) * cos(2 pi f t + phi); t in s, f in Hz."""
    return offset + amp * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + phi)


def fit_rabi(t_s, signal):
    """Fit one damped Rabi trace. Returns dict with f_hz, f_err_hz, tau_s, pi_s and the model curve."""
    t = np.asarray(t_s, dtype=float)
    y = np.asarray(signal, dtype=float)
    y0 = y - y.mean()
    dt = np.median(np.diff(t))
    spec = np.abs(np.fft.rfft(y0 * np.hanning(len(y0)), n=8 * len(y0)))
    freqs = np.fft.rfftfreq(8 * len(y0), dt)
    f0 = freqs[1 + np.argmax(spec[1:])]
    p0 = [y.mean(), y0[0] if abs(y0[0]) > 0 else np.ptp(y) / 2, f0, t[-1] / 2, 0.0]
    popt, pcov = curve_fit(rabi_model, t, y, p0=p0, maxfev=20000)
    f = abs(popt[2])
    return {
        "f_hz": f,
        "f_err_hz": float(np.sqrt(pcov[2, 2])),
        "tau_s": abs(popt[3]),
        "pi_s": 1.0 / (2.0 * f),
        "popt": popt,
        "model": rabi_model(t, *popt),
    }


def power_at_antenna(p_source_w, path_loss_db):
    """Power reaching the antenna after cables, circulator and connectors (loss in dB, positive)."""
    return np.asarray(p_source_w, dtype=float) * 10 ** (-path_loss_db / 10.0)


def fit_slope(power_w, rabi_hz):
    """Least-squares line through the origin: f_Rabi = k * sqrt(P).

    Returns k (Hz per sqrt(W)), its standard error, and the implied drive field per sqrt(W) in T.
    """
    x = np.sqrt(np.asarray(power_w, dtype=float))
    y = np.asarray(rabi_hz, dtype=float)
    k = float(np.sum(x * y) / np.sum(x * x))
    resid = y - k * x
    se = float(np.sqrt(np.sum(resid**2) / max(len(x) - 1, 1) / np.sum(x * x)))
    return {"k_hz_per_sqrt_w": k, "k_err": se, "b1_t_per_sqrt_w": nv.b1_from_rabi(k)}

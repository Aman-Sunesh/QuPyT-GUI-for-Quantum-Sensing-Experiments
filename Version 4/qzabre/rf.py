"""Quasi-static microstrip design equations (Hammerstad-Jensen) for the 2-4 GHz NV antenna feed.

Valid while the substrate is thin compared with the wavelength. Frequency dispersion
(Kirschning-Jansen) is not included; at 2.87 GHz on 1.6 mm FR4 it shifts eps_eff by a few percent.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

ETA0 = 376.730313
C0 = 299792458.0


def _z01(u):
    f = 6 + (2 * np.pi - 6) * np.exp(-((30.666 / u) ** 0.7528))
    return ETA0 / (2 * np.pi) * np.log(f / u + np.sqrt(1 + (2 / u) ** 2))


def _eeff_static(u, er):
    a = 1 + np.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49 + np.log(1 + (u / 18.1) ** 3) / 18.7
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    return (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)


def microstrip(w_mm: float, h_mm: float, er: float, t_mm: float = 0.0):
    """Return (z0_ohm, eps_eff) for a microstrip of width w on substrate height h."""
    u = w_mm / h_mm
    if t_mm > 0:
        tt = t_mm / h_mm
        du1 = tt / np.pi * np.log(1 + 4 * np.e / (tt / np.tanh(np.sqrt(6.517 * u)) ** 2))
        dur = 0.5 * (1 + 1 / np.cosh(np.sqrt(er - 1))) * du1
        ur = u + dur
        eeff = _eeff_static(ur, er)
        return _z01(ur) / np.sqrt(eeff), eeff
    eeff = _eeff_static(u, er)
    return _z01(u) / np.sqrt(eeff), eeff


def width_for_z0(z0_ohm: float, h_mm: float, er: float, t_mm: float = 0.0) -> float:
    """Trace width (mm) giving the target impedance."""
    return h_mm * brentq(lambda u: microstrip(u * h_mm, h_mm, er, t_mm)[0] - z0_ohm, 0.05, 20)


def electrical_length_deg(length_mm: float, f_hz: float, eeff: float) -> float:
    return 360.0 * length_mm * 1e-3 * f_hz * np.sqrt(eeff) / C0


def quarter_wave_mm(f_hz: float, eeff: float) -> float:
    return C0 / (4 * f_hz * np.sqrt(eeff)) * 1e3

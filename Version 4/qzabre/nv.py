"""NV-center relations used in the Quantum Edukit manual v1.0 and the Bucher 2019 protocol.

All frequencies in Hz, fields in tesla, power in watts unless a name says otherwise.
"""

from __future__ import annotations

import numpy as np

D_HZ = 2.87e9  # zero-field splitting
GAMMA_HZ_PER_T = 28.02495164e9  # gamma_e / 2pi
GAMMA_MANUAL_HZ_PER_T = 28.0e9  # rounded value the Edukit manual uses (Table 2)
COS_100_TO_111 = 1.0 / np.sqrt(3.0)  # cos(54.74 deg): angle between [100] and an NV axis
RABI_AT_1W_HZ = 14e6  # manual Eq. 6.2: Rabi frequency = 14 MHz * sqrt(P / 1 W)

# The four NV orientations, in the order of manual Table 2.
NV_AXES = np.array(
    [[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]], dtype=float
) / np.sqrt(3.0)


def field_vector(magnitude_t: float, theta_deg: float, phi_deg: float) -> np.ndarray:
    """Cartesian field vector from magnitude and spherical angles (crystal frame, z = [001])."""
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    return magnitude_t * np.array(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    )


def resonances_first_order(
    b_vec_t, gamma: float = GAMMA_HZ_PER_T, d_hz: float = D_HZ
) -> np.ndarray:
    """Resonance pairs (f_minus, f_plus) for each NV axis, first order in B.

    Shape (4, 2). Only the field component along the NV axis matters at this order.
    """
    shift = gamma * np.abs(NV_AXES @ np.asarray(b_vec_t, dtype=float))
    return np.stack([d_hz - shift, d_hz + shift], axis=1)


def resonances_exact(
    b_vec_t, gamma: float = GAMMA_HZ_PER_T, d_hz: float = D_HZ
) -> np.ndarray:
    """Resonance pairs per NV axis from diagonalizing H/h = D Sz^2 + gamma B.S (spin 1).

    Includes the transverse-field (second-order) shifts that the first-order formula drops.
    Shape (4, 2): transitions from the lower-energy of the two m = +-1-like states and the higher one.
    """
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    sz = np.diag([1.0, 0.0, -1.0])
    b = np.asarray(b_vec_t, dtype=float)
    out = np.empty((4, 2))
    for i, n in enumerate(NV_AXES):
        bz = float(b @ n)
        b_perp = float(np.linalg.norm(b - bz * n))
        h = d_hz * (sz @ sz) + gamma * (bz * sz + b_perp * sx)
        e = np.sort(np.linalg.eigvalsh(h))
        # e[0] is the state that evolves from m = 0; e[1], e[2] from m = -1, +1
        out[i] = [e[1] - e[0], e[2] - e[0]]
    return out


def b_nv_from_splitting(f_plus_hz: float, f_minus_hz: float, gamma: float = GAMMA_HZ_PER_T) -> float:
    """Field component along the NV axis from the two resonances (manual Sec. 6.1.4)."""
    return (f_plus_hz - f_minus_hz) / (2.0 * gamma)


def b_total_for_100_field(b_nv_t: float) -> float:
    """Total field when B points along [100] (manual Sec. 6.1.4): B = B_NV / cos(54.74 deg)."""
    return b_nv_t / COS_100_TO_111


def aligned_resonances(b_t: float, gamma: float = GAMMA_HZ_PER_T, d_hz: float = D_HZ):
    """(f_minus, f_plus) for a field along an NV axis. Eq. 6 of Bucher 2019 is the f_minus branch."""
    return d_hz - gamma * b_t, d_hz + gamma * b_t


def rabi_freq_from_power(p_w: float, rabi_at_1w_hz: float = RABI_AT_1W_HZ) -> float:
    """Rabi frequency (cycles/s) for coupled microwave power p_w (manual Eq. 6.2).

    Valid for the Edukit antenna only; the 14 MHz comes from its 1 W field simulation.
    """
    return rabi_at_1w_hz * np.sqrt(p_w)


def b1_from_rabi(f_rabi_hz: float, gamma: float = GAMMA_HZ_PER_T) -> float:
    """Amplitude of the linearly polarized drive field perpendicular to the NV axis.

    From f_rabi = gamma * B1 / sqrt(2): the m = 0 to m = -1 matrix element of Sx is 1/sqrt(2).
    """
    return np.sqrt(2.0) * f_rabi_hz / gamma


def rabi_from_b1(b1_t: float, gamma: float = GAMMA_HZ_PER_T) -> float:
    return gamma * b1_t / np.sqrt(2.0)

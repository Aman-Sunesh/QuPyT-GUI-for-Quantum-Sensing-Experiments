"""Simulated ODMR dip tracking under field drift and shot noise.

The loop parks the microwave at the dip and holds it there while the field drifts:

1. Probe the signal at f_c + delta and f_c - delta (two short dwells per step).
2. Error signal e = (n_plus - n_minus) / (n_plus + n_minus). It is zero when f_c sits on the dip.
3. Divide e by the model slope to get an offset estimate x_hat in Hz.
4. Velocity-form PI update: f_c -= ki * x_hat + kp * (x_hat - x_hat_previous).

Everything here is a MODEL. Default numbers are placeholders (marked ESTIMATE below). Replace them
with values fitted from your own spectra (see TrackerConfig.from_dip). No number in this file is
a measurement of the Edukit.

SI units throughout (Hz, T, s). Single NV axis, field along that axis, so
f_lower = D - gamma*B and f_upper = D + gamma*B (nv.aligned_resonances).
D is held constant. A temperature shift of D would look like a field change to a single-dip tracker.

Command line:
    python -m qzabre.tracking --plot out.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace

import numpy as np

from . import nv, odmr

# Lorentzian prefactor 4 / (3 sqrt 3) = 0.77. Derived below and checked in tests/test_tracking.py.
LORENTZIAN_PREFACTOR = 4.0 / (3.0 * np.sqrt(3.0))


@dataclass
class TrackerConfig:
    """Dip shape, photon budget, and loop gains. All defaults are ESTIMATES (placeholders)."""

    fwhm_hz: float = 5e6  # ESTIMATE: dip width. Fit your own spectrum.
    contrast: float = 0.03  # ESTIMATE: fractional dip depth. Fit your own spectrum.
    photon_rate_hz: float = 1e6  # ESTIMATE: detected photons per second off resonance.
    step_s: float = 0.01  # loop period; each probe gets half of it. Choice, not a measurement.
    delta_hz: float | None = None  # probe offset; None = FWHM / (2 sqrt 3), the steepest point
    ki: float = 0.2  # integrator gain. 1 = full correction per step, 0 < ki < 2 is stable
    kp: float = 0.0  # proportional (difference) gain
    side: str = "lower"  # which dip to follow: "lower" (D - gamma B) or "upper" (D + gamma B)
    gamma: float = nv.GAMMA_HZ_PER_T
    d_hz: float = nv.D_HZ

    def __post_init__(self):
        if self.side not in ("lower", "upper"):
            raise ValueError("side must be 'lower' or 'upper'")
        if self.delta_hz is None:
            self.delta_hz = self.fwhm_hz / (2.0 * np.sqrt(3.0))

    @classmethod
    def from_dip(cls, dip: odmr.Dip, **kw) -> "TrackerConfig":
        """Take width and contrast from a fitted odmr.Dip (the right way to set them)."""
        return cls(fwhm_hz=dip.fwhm_hz, contrast=dip.depth, **kw)

    def dip_freq(self, b_t) -> np.ndarray:
        s = -1.0 if self.side == "lower" else 1.0
        return self.d_hz + s * self.gamma * np.asarray(b_t, dtype=float)

    def field_from_freq(self, f_hz) -> np.ndarray:
        s = -1.0 if self.side == "lower" else 1.0
        return s * (np.asarray(f_hz, dtype=float) - self.d_hz) / self.gamma


@dataclass
class Drift:
    """Field drift model, in tesla. All zero by default.

    ramp_t_per_s: constant drift rate. walk_t_per_rthz: random-walk strength, so the field wanders
    by walk * sqrt(t). offset_t: starting offset between the true field and the tracker's first guess.
    """

    ramp_t_per_s: float = 0.0
    walk_t_per_rthz: float = 0.0
    offset_t: float = 0.0


@dataclass
class TrackResult:
    t_s: np.ndarray
    f_dip_hz: np.ndarray  # true dip center (model)
    f_c_hz: np.ndarray  # tracker's microwave center
    error_hz: np.ndarray  # f_c - f_dip, evaluated at the start of each step
    b_true_t: np.ndarray
    b_est_t: np.ndarray
    cfg: TrackerConfig = field(repr=False)

    def rms_error_hz(self, skip: int = 0) -> float:
        e = self.error_hz[skip:]
        return float(np.sqrt(np.mean(e**2)))

    def rms_error_t(self, skip: int = 0) -> float:
        return self.rms_error_hz(skip) / self.cfg.gamma

    def mean_error_hz(self, skip: int = 0) -> float:
        return float(np.mean(self.error_hz[skip:]))


def _lor(x, hw):
    return hw**2 / (x**2 + hw**2)


def _dlor(x, hw):
    return -2.0 * x * hw**2 / (x**2 + hw**2) ** 2


def signal(f_hz, f_dip_hz, cfg: TrackerConfig):
    """Normalized signal (baseline 1) at f_hz for a dip at f_dip_hz. Same shape as odmr.lorentzian_dips."""
    return odmr.lorentzian_dips(np.asarray(f_hz, dtype=float), 1.0, f_dip_hz, cfg.fwhm_hz, cfg.contrast)


def error_signal_mean(offset_hz, cfg: TrackerConfig):
    """Noiseless error signal e for f_c - f_dip = offset_hz."""
    hw = cfg.fwhm_hz / 2.0
    sp = 1.0 - cfg.contrast * _lor(offset_hz + cfg.delta_hz, hw)
    sm = 1.0 - cfg.contrast * _lor(offset_hz - cfg.delta_hz, hw)
    return (sp - sm) / (sp + sm)


def error_slope(cfg: TrackerConfig) -> float:
    """de/d(offset) at offset 0, per Hz. Positive: f_c above the dip makes e > 0."""
    hw = cfg.fwhm_hz / 2.0
    d = cfg.delta_hz
    return float(-cfg.contrast * _dlor(d, hw) / (1.0 - cfg.contrast * _lor(d, hw)))


def step_noise_hz(cfg: TrackerConfig) -> float:
    """1-sigma shot-noise error of one offset estimate x_hat, in Hz (model, small-noise limit).

    Each probe collects mu = R * step/2 * (1 - C L(delta)) photons. For e = (n+ - n-)/(n+ + n-),
    var(e) = 1 / (2 mu), so sigma_x = sqrt(1 / (2 mu)) / slope.
    """
    hw = cfg.fwhm_hz / 2.0
    mu = cfg.photon_rate_hz * cfg.step_s / 2.0 * (1.0 - cfg.contrast * _lor(cfg.delta_hz, hw))
    return float(np.sqrt(1.0 / (2.0 * mu)) / error_slope(cfg))


def sensitivity_t_per_rthz(cfg: TrackerConfig) -> float:
    """Field sensitivity of this two-point protocol in T/sqrt(Hz) (model).

    sigma_x * sqrt(step) / gamma. For small contrast and the default delta this equals the standard
    shot-noise-limited Lorentzian result eta = (4/(3 sqrt 3)) * FWHM / (gamma * C * sqrt(R))
    (see lorentzian_limit_t_per_rthz). Excludes laser noise, spin-state readout loss, and drift.
    """
    return step_noise_hz(cfg) * np.sqrt(cfg.step_s) / cfg.gamma


def lorentzian_limit_t_per_rthz(fwhm_hz: float, contrast: float, photon_rate_hz: float,
                                gamma: float = nv.GAMMA_HZ_PER_T) -> float:
    """Textbook shot-noise limit for a Lorentzian dip probed at its steepest point (model).

    eta = (4/(3 sqrt 3)) * FWHM / (gamma * C * sqrt(R)). Literature: Barry et al., Rev. Mod. Phys.
    92, 015004 (2020); the equation number was not checked here. The prefactor is re-derived
    numerically in the tests.
    """
    return LORENTZIAN_PREFACTOR * fwhm_hz / (gamma * contrast * np.sqrt(photon_rate_hz))


def expected_steady_state_rms_hz(cfg: TrackerConfig) -> float:
    """Predicted rms tracking error from shot noise alone, kp = 0, no drift (model).

    Integrator on white noise: eps' = (1 - ki) eps - ki n, so var = ki / (2 - ki) * sigma_x^2.
    """
    return float(step_noise_hz(cfg) * np.sqrt(cfg.ki / (2.0 - cfg.ki)))


def expected_ramp_lag_hz(cfg: TrackerConfig, ramp_t_per_s: float) -> float:
    """Steady lag of f_c behind a constant-rate dip, in Hz (model). One integrator: lag = rate*step/ki.

    Signed like error_hz = f_c - f_dip: for a lower dip and positive field ramp the dip moves down, so
    the lag is positive. kp does not change it.
    """
    s = -1.0 if cfg.side == "lower" else 1.0
    return float(-s * cfg.gamma * ramp_t_per_s * cfg.step_s / cfg.ki)


def simulate(cfg: TrackerConfig, drift: Drift | None = None, b0_t: float = 1.5e-3,
             duration_s: float = 10.0, seed: int | None = 0, shot_noise: bool = True) -> TrackResult:
    """Run the tracking loop. b0_t is the true starting field. shot_noise=False uses mean counts."""
    drift = drift or Drift()
    if not 0.0 < cfg.ki < 2.0:
        raise ValueError("ki must be in (0, 2) for a stable loop")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / cfg.step_s))
    t_probe = cfg.step_s / 2.0
    slope = error_slope(cfg)

    b = b0_t
    # offset_t is the tracker's initial field error: f_c starts at the dip for B = b0_t + offset_t
    f_c = float(cfg.dip_freq(b0_t + drift.offset_t))
    x_prev = 0.0

    t = np.arange(n) * cfg.step_s
    f_dip = np.empty(n)
    f_cs = np.empty(n)
    b_true = np.empty(n)
    b_est = np.empty(n)
    for k in range(n):
        f_d = float(cfg.dip_freq(b))
        f_dip[k], f_cs[k], b_true[k] = f_d, f_c, b
        mean_p = cfg.photon_rate_hz * t_probe * float(signal(f_c + cfg.delta_hz, f_d, cfg))
        mean_m = cfg.photon_rate_hz * t_probe * float(signal(f_c - cfg.delta_hz, f_d, cfg))
        if shot_noise:
            n_p, n_m = rng.poisson(mean_p), rng.poisson(mean_m)
        else:
            n_p, n_m = mean_p, mean_m
        e = (n_p - n_m) / (n_p + n_m) if (n_p + n_m) > 0 else 0.0
        x_hat = e / slope
        b_est[k] = float(cfg.field_from_freq(f_c - x_hat))
        f_c -= cfg.ki * x_hat + cfg.kp * (x_hat - x_prev)
        x_prev = x_hat
        b += drift.ramp_t_per_s * cfg.step_s
        if drift.walk_t_per_rthz:
            b += drift.walk_t_per_rthz * np.sqrt(cfg.step_s) * rng.standard_normal()
    return TrackResult(t, f_dip, f_cs, f_cs - f_dip, b_true, b_est, cfg)


def sweep_ki(cfg: TrackerConfig, ki_values, drift: Drift, duration_s: float = 20.0, seed: int = 0,
             skip_s: float = 2.0):
    """Rms tracking error (Hz) for each integrator gain. Use it to pick ki for your drift and noise."""
    out = []
    for ki in ki_values:
        r = simulate(replace(cfg, ki=float(ki)), drift, duration_s=duration_s, seed=seed)
        out.append(r.rms_error_hz(skip=int(skip_s / cfg.step_s)))
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fwhm-mhz", type=float, default=5.0, help="dip width (ESTIMATE default)")
    ap.add_argument("--contrast", type=float, default=0.03, help="fractional depth (ESTIMATE default)")
    ap.add_argument("--rate-mcps", type=float, default=1.0, help="photon rate, Mcounts/s (ESTIMATE default)")
    ap.add_argument("--step-ms", type=float, default=10.0)
    ap.add_argument("--ki", type=float, default=0.2)
    ap.add_argument("--kp", type=float, default=0.0)
    ap.add_argument("--ramp-ut-per-s", type=float, default=0.5, help="field ramp, uT/s (model input)")
    ap.add_argument("--walk-nt-per-rthz", type=float, default=50.0, help="random walk, nT/sqrt(s) (model input)")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default=None, help="save a PNG")
    a = ap.parse_args(argv)

    cfg = TrackerConfig(fwhm_hz=a.fwhm_mhz * 1e6, contrast=a.contrast, photon_rate_hz=a.rate_mcps * 1e6,
                        step_s=a.step_ms * 1e-3, ki=a.ki, kp=a.kp)
    drift = Drift(ramp_t_per_s=a.ramp_ut_per_s * 1e-6, walk_t_per_rthz=a.walk_nt_per_rthz * 1e-9)
    res = simulate(cfg, drift, duration_s=a.duration, seed=a.seed)
    skip = int(2.0 / cfg.step_s)
    print("All numbers below are MODEL outputs for the inputs shown, not measurements.")
    print(f"shot-noise sensitivity (model): {sensitivity_t_per_rthz(cfg)*1e9:.1f} nT/sqrt(Hz)")
    print(f"textbook Lorentzian limit (model, small contrast): "
          f"{lorentzian_limit_t_per_rthz(cfg.fwhm_hz, cfg.contrast, cfg.photon_rate_hz)*1e9:.1f} nT/sqrt(Hz)")
    print(f"per-step offset noise (model): {step_noise_hz(cfg)/1e3:.1f} kHz")
    print(f"rms tracking error (simulated): {res.rms_error_hz(skip)/1e3:.1f} kHz = {res.rms_error_t(skip)*1e6:.2f} uT")
    print(f"ramp lag (model): {expected_ramp_lag_hz(cfg, drift.ramp_t_per_s)/1e3:.1f} kHz")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        a1.plot(res.t_s, res.b_true_t * 1e3, label="true field")
        a1.plot(res.t_s, res.b_est_t * 1e3, ".", ms=2, label="tracker estimate")
        a1.set_ylabel("B along NV axis (mT)")
        a1.legend()
        a2.plot(res.t_s, res.error_hz / 1e3)
        a2.set_ylabel("f_c - f_dip (kHz)")
        a2.set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)


if __name__ == "__main__":
    main()

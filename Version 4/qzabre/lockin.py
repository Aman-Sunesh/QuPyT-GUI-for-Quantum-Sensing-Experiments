"""Lock-in resonance tracking simulator.

A fixed-frequency microwave drive sits on the side of an ODMR dip. To follow the dip as it drifts:

1. Dither the microwave frequency: f(t) = fc + A sin(2 pi f_m t).
2. Read the (noisy) fluorescence contrast y(t) at every sample.
3. Demodulate: X = (2/N) sum y sin(phase), over a whole number of dither periods.
   For small A, X = A dy/df at fc. That slope is zero at the dip center and changes sign across it.
4. Turn X into a frequency error in Hz with the nominal lineshape, then close the loop with a PI controller.

The lineshape comes from odmr.lorentzian_dips, the same model the ODMR fit uses, so a fitted Dip
(freq, FWHM, depth) can be passed straight in. Everything is SI (Hz, s). Dither, sample rate, noise
and drift defaults are simulation assumptions, not Edukit specifications.

Command line: python -m qzabre.lockin --plot out.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .odmr import Dip, FitResult, fit_odmr, lorentzian_dips


@dataclass
class LockinConfig:
    dither_amp_hz: float = 2e6  # A. Keep well below the FWHM or the slope flattens.
    dither_freq_hz: float = 1e3  # f_m. Assumption.
    cycles_per_step: int = 10  # dither periods averaged per controller update
    samples_per_cycle: int = 20
    noise: float = 0.0015  # white noise per sample, contrast units (same as the synthetic ODMR tests)
    kp: float = 0.3  # fraction of the estimated error corrected each step
    ki: float = 20.0  # 1/s, integral gain on the estimated error
    limit_hz: float = 50e6  # clamp on |fc - start|; stops windup if the lock is lost

    @property
    def step_s(self) -> float:
        return self.cycles_per_step / self.dither_freq_hz


@dataclass
class Lineshape:
    """Single Lorentzian dip, baseline 1. center_hz is where the dip sits at t = 0."""

    center_hz: float = 2.87e9
    fwhm_hz: float = 13e6  # mid-range of the 10-16 MHz fits quoted by the owner
    depth: float = 0.06  # mid-range of the 5-7 % fits quoted by the owner
    # Neighboring dips as (offset_hz, fwhm_hz, depth) relative to the locked dip. They move with it
    # (common-mode drift, such as temperature). The controller's gain ignores them, as a real lock would.
    others: tuple = ()

    @classmethod
    def from_dip(cls, dip: Dip) -> "Lineshape":
        return cls(dip.freq_hz, dip.fwhm_hz, dip.depth)

    @classmethod
    def from_fit(cls, res: FitResult, index: int) -> "Lineshape":
        """Lock on dips[index] of a fit and keep the other fitted dips as neighbors."""
        d = res.dips[index]
        others = tuple((o.freq_hz - d.freq_hz, o.fwhm_hz, o.depth) for i, o in enumerate(res.dips) if i != index)
        return cls(d.freq_hz, d.fwhm_hz, d.depth, others)

    def signal(self, f, center_hz):
        params = [center_hz, self.fwhm_hz, self.depth]
        for off, w, a in self.others:
            params += [center_hz + off, w, a]
        return lorentzian_dips(f, 1.0, *params)

    @property
    def slope_per_hz2(self) -> float:
        """d(dy/df)/df at the center: dy/df = k (f - f0) near the dip. k = 8 depth / FWHM^2."""
        return 8.0 * self.depth / self.fwhm_hz**2


@dataclass
class LockResult:
    t: np.ndarray  # s, end of each controller step
    dip_hz: np.ndarray  # true dip center
    fc_hz: np.ndarray  # controller output (carrier)
    error_hz: np.ndarray  # fc - dip, the quantity we want near zero
    cfg: LockinConfig = field(repr=False)

    def rms_error_hz(self, last_fraction: float = 0.5) -> float:
        n = max(1, int(self.error_hz.size * last_fraction))
        return float(np.sqrt(np.mean(self.error_hz[-n:] ** 2)))

    def mean_error_hz(self, last_fraction: float = 0.5) -> float:
        n = max(1, int(self.error_hz.size * last_fraction))
        return float(np.mean(self.error_hz[-n:]))


def ramp(center_hz: float, rate_hz_per_s: float):
    """Linear drift of the dip center. Returns a function of time (array-safe)."""
    return lambda t: center_hz + rate_hz_per_s * np.asarray(t, dtype=float)


def simulate(
    shape: Lineshape | None = None,
    cfg: LockinConfig | None = None,
    duration_s: float = 10.0,
    dip_at=None,
    start_offset_hz: float = 0.0,
    seed: int = 0,
) -> LockResult:
    """Run the lock. dip_at(t) gives the true center in Hz (array-safe). Default: fixed at shape.center_hz.

    start_offset_hz is how far the carrier starts from the dip. Pull-in works only within about
    half a FWHM, because the error signal fades far from the dip.
    """
    shape = shape or Lineshape()
    cfg = cfg or LockinConfig()
    dip_at = dip_at or (lambda t: np.full_like(np.asarray(t, dtype=float), shape.center_hz))
    rng = np.random.default_rng(seed)

    n_steps = int(round(duration_s / cfg.step_s))
    n_samp = cfg.cycles_per_step * cfg.samples_per_cycle
    local_t = np.arange(n_samp) / (cfg.dither_freq_hz * cfg.samples_per_cycle)
    phase = 2 * np.pi * cfg.dither_freq_hz * local_t
    sin_ph = np.sin(phase)
    # Nominal conversion from X to Hz: X = A k (fc - f0), so (f0 - fc) = -X / (A k).
    gain = cfg.dither_amp_hz * shape.slope_per_hz2

    start = float(dip_at(0.0)) + start_offset_hz
    fc, integ = start, 0.0
    t_out, dip_out, fc_out = np.empty(n_steps), np.empty(n_steps), np.empty(n_steps)
    for i in range(n_steps):
        t0 = i * cfg.step_s
        t = t0 + local_t
        f_drive = fc + cfg.dither_amp_hz * sin_ph
        y = shape.signal(f_drive, dip_at(t)) + rng.normal(0.0, cfg.noise, n_samp)
        x = 2.0 * np.mean(y * sin_ph)  # lock-in output
        err_hz = -x / gain  # estimated dip - carrier
        u = cfg.kp * err_hz + cfg.ki * (integ + err_hz * cfg.step_s)
        if abs(u) < cfg.limit_hz:  # freeze the integrator at the clamp (anti-windup)
            integ += err_hz * cfg.step_s
        fc = start + float(np.clip(u, -cfg.limit_hz, cfg.limit_hz))
        t_out[i], dip_out[i], fc_out[i] = t0 + cfg.step_s, float(dip_at(t0 + cfg.step_s)), fc
    return LockResult(t_out, dip_out, fc_out, fc_out - dip_out, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--odmr", default=None, help="EduControl run folder or odmr.h5; lock on a dip of its fit instead of the defaults")
    ap.add_argument("--dip", type=int, default=0, help="which fitted dip to lock on, counted from the lowest frequency")
    ap.add_argument("--fwhm-mhz", type=float, default=13.0)
    ap.add_argument("--depth", type=float, default=0.06)
    ap.add_argument("--drift-khz-per-s", type=float, default=100.0, help="linear drift of the dip (assumption)")
    ap.add_argument("--noise", type=float, default=0.0015)
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.add_argument("--offset-mhz", type=float, default=3.0, help="starting distance of the carrier from the dip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    if args.odmr:
        from .edukit_io import load_spectrum

        freq, contrast, _ = load_spectrum(args.odmr)
        fit = fit_odmr(freq, contrast)
        shape = Lineshape.from_fit(fit, args.dip)
        args.fwhm_mhz, args.depth = shape.fwhm_hz / 1e6, shape.depth
        print(f"fit: {len(fit.dips)} dips; locking on #{args.dip} at {shape.center_hz / 1e9:.5f} GHz, neighbors at "
              + ", ".join(f"{o / 1e6:+.1f} MHz" for o, _, _ in shape.others))
    else:
        shape = Lineshape(fwhm_hz=args.fwhm_mhz * 1e6, depth=args.depth)
    cfg = LockinConfig(noise=args.noise)
    res = simulate(shape, cfg, args.duration_s, ramp(shape.center_hz, args.drift_khz_per_s * 1e3), args.offset_mhz * 1e6, args.seed)
    print(f"dip FWHM {args.fwhm_mhz} MHz, depth {args.depth:.3f}, drift {args.drift_khz_per_s} kHz/s, noise {args.noise} per sample")
    print(f"step {cfg.step_s * 1e3:.1f} ms; dither {cfg.dither_amp_hz / 1e6} MHz at {cfg.dither_freq_hz / 1e3} kHz")
    print(f"last-half error: mean {res.mean_error_hz() / 1e3:+.1f} kHz, rms {res.rms_error_hz() / 1e3:.1f} kHz")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
        f = shape.center_hz + np.linspace(-30e6, 30e6, 400)
        axs[0].plot((f - shape.center_hz) / 1e6, shape.signal(f, shape.center_hz), color="tab:blue")
        axs[0].set_xlabel("Detuning from dip (MHz)")
        axs[0].set_ylabel("Contrast (baseline = 1)")
        axs[0].set_title("Lineshape (model)")
        axs[1].plot(res.t, (res.dip_hz - shape.center_hz) / 1e6, color="tab:gray", label="dip (true)")
        axs[1].plot(res.t, (res.fc_hz - shape.center_hz) / 1e6, color="tab:red", lw=1, label="carrier (lock)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Offset from start center (MHz)")
        axs[1].set_title("Lock follows the drift")
        axs[1].legend(fontsize=8)
        axs[2].plot(res.t, res.error_hz / 1e3, color="tab:red", lw=1)
        axs[2].axhline(0, color="k", lw=0.5)
        lim = 6 * res.rms_error_hz() / 1e3  # zoom on steady state; the start-up pull-in is off scale
        axs[2].set_ylim(-lim, lim)
        axs[2].text(0.97, 0.05, "start-up transient off scale", transform=axs[2].transAxes, ha="right", fontsize=8, color="gray")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Carrier minus dip (kHz)")
        axs[2].set_title("Tracking error")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()

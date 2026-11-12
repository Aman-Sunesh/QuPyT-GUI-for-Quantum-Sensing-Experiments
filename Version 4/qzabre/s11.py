"""Read one-port Touchstone (.s1p) files and extract resonance, loaded Q and -10 dB bandwidth.

Use it on the VNA files from Part A of docs/antenna-test-procedure.md.

What you get:
- f_res: frequency of the lowest |S11| (parabola through the three points around the minimum).
- s11_min_db: lowest |S11| in the data, in dB.
- bw10: the frequency range around f_res where |S11| <= -10 dB, edges interpolated between points.
- Q (loaded): from a fit of the complex S11 near the dip with the one-port resonator model

      S11(f) = a * (1 - kappa / (1 + 2j Q (f - f0) / f0))

  a and kappa are complex, so cable phase and impedance mismatch do not bias Q. For this model,
  |S11|^2 sits halfway between its minimum and its off-resonance level at f0 +- f0/(2 Q).

Q is not reported when the dip is shallower than min_depth_db. A loop antenna reflects almost
everything by design (docs/rf-notes.md), so Q has no meaning there.

Command line:
    python -m qzabre.s11 antenna.s1p --plot s11.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

_UNITS_HZ = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
_FORMATS = ("DB", "MA", "RI")
_MIN_FIT_POINTS = 10


@dataclass
class Touchstone:
    freq_hz: np.ndarray
    s11: np.ndarray  # complex reflection coefficient
    z0_ohm: float


@dataclass
class S11Result:
    f_res_hz: float  # frequency of the lowest |S11|
    s11_min_db: float
    bw10_hz: float  # 0.0 if |S11| never reaches the level
    bw10_lo_hz: float  # nan if never reached
    bw10_hi_hz: float
    bw10_clipped: bool  # the band touches the end of the sweep, so bw10 is a lower bound
    f0_hz: float = float("nan")  # fitted resonance frequency of the model
    f0_err_hz: float = float("nan")
    q_loaded: float = float("nan")
    q_err: float = float("nan")
    fit_rms: float = float("nan")  # rms residual of the complex fit, in |S11| units
    fit_window_hz: tuple = (float("nan"), float("nan"))
    model: np.ndarray | None = None  # fitted complex S11 on the input grid
    notes: list = field(default_factory=list)


def read_s1p(path) -> Touchstone:
    """Read a one-port Touchstone file (v1.x layout: option line, then freq and two numbers per row).

    Handles DB, MA and RI formats, Hz/kHz/MHz/GHz units, "!" comments and the default
    option line "# GHz S MA R 50". Rows are returned sorted by frequency.
    """
    unit, fmt, z0 = 1e9, "MA", 50.0
    seen_option = False
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.split("!", 1)[0].strip()
            if not line or line.startswith("["):  # blank, comment-only, or a v2 keyword line
                continue
            if line.startswith("#"):
                if seen_option:  # only the first option line counts
                    continue
                seen_option = True
                toks = line[1:].upper().split()
                i = 0
                while i < len(toks):
                    t = toks[i]
                    if t in _UNITS_HZ:
                        unit = _UNITS_HZ[t]
                    elif t in _FORMATS:
                        fmt = t
                    elif t == "R":
                        i += 1
                        z0 = float(toks[i])
                    elif t in ("S", "Y", "Z", "G", "H"):
                        if t != "S":
                            raise ValueError(f"{path}:{n}: only S-parameters are supported, got {t}")
                    else:
                        raise ValueError(f"{path}:{n}: cannot read option token {t!r}")
                    i += 1
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{n}: expected 3 columns (frequency and one S11 value), got {len(parts)}. "
                    "Only one-port (.s1p) files are supported."
                )
            rows.append([float(p) for p in parts])
    if len(rows) < 5:
        raise ValueError(f"{path}: found {len(rows)} data rows, need at least 5")
    data = np.array(rows)
    order = np.argsort(data[:, 0])
    data = data[order]
    freq = data[:, 0] * unit
    a, b = data[:, 1], data[:, 2]
    if fmt == "RI":
        s11 = a + 1j * b
    elif fmt == "MA":
        s11 = a * np.exp(1j * np.deg2rad(b))
    else:  # DB
        s11 = 10 ** (a / 20) * np.exp(1j * np.deg2rad(b))
    return Touchstone(freq, s11, z0)


def _db(s):
    return 20 * np.log10(np.maximum(np.abs(s), 1e-12))


def _contiguous_below(x, y, i0, level):
    """Interpolated edges of the run of samples with y <= level that contains index i0.

    Returns (lo, hi, lo_clipped, hi_clipped), or None if y[i0] > level. An edge is clipped when
    the run reaches the first or last sample.
    """
    if y[i0] > level:
        return None
    n = len(y)
    i = i0
    while i > 0 and y[i - 1] <= level:
        i -= 1
    j = i0
    while j < n - 1 and y[j + 1] <= level:
        j += 1
    lo_clipped, hi_clipped = i == 0, j == n - 1
    lo = x[0] if lo_clipped else x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
    hi = x[-1] if hi_clipped else x[j] + (level - y[j]) * (x[j + 1] - x[j]) / (y[j + 1] - y[j])
    return lo, hi, lo_clipped, hi_clipped


def _refine_min(freq, db, i):
    """Parabola through the three dB points around index i; returns the vertex frequency."""
    if i == 0 or i == len(freq) - 1:
        return float(freq[i])
    y0, y1, y2 = db[i - 1], db[i], db[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom <= 0:
        return float(freq[i])
    off = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(freq[i] + off * 0.5 * (freq[i + 1] - freq[i - 1]))


def resonator_model(f, f0, q, kappa, a):
    """One-port resonator reflection; kappa and a are complex."""
    x = 2 * q * (np.asarray(f, dtype=float) - f0) / f0
    return a * (1 - kappa / (1 + 1j * x))


def _fit_resonator(freq, s11, i_min, f_res, notes):
    """Complex fit in a window around the dip. Returns a dict, or None if the fit fails."""
    p = np.abs(s11) ** 2
    base = np.percentile(p, 95)
    half = 0.5 * (base + p[i_min])
    run = _contiguous_below(freq, p, i_min, half)
    span = freq[-1] - freq[0]
    if run is None or (run[2] and run[3]):
        w = span / 10
        notes.append("half-power width not found; starting Q guess from the sweep span")
    else:
        w = max(run[1] - run[0], np.median(np.diff(freq)))
    q0 = f_res / w

    mask = np.abs(freq - f_res) <= 4 * w
    lo_i, hi_i = (int(np.argmax(mask)), int(len(mask) - 1 - np.argmax(mask[::-1])))
    while hi_i - lo_i + 1 < _MIN_FIT_POINTS and (lo_i > 0 or hi_i < len(freq) - 1):
        lo_i, hi_i = max(lo_i - 1, 0), min(hi_i + 1, len(freq) - 1)
    sl = slice(lo_i, hi_i + 1)
    f, s = freq[sl], s11[sl]
    if f.size < _MIN_FIT_POINTS:
        notes.append(f"only {f.size} points near the dip; Q not fitted")
        return None
    if f.size < 3 * _MIN_FIT_POINTS:
        notes.append(f"only {f.size} points in the fit window; Q is a rough estimate")

    a0 = 0.5 * (s[:3].mean() + s[-3:].mean())
    k0 = 1 - s11[i_min] / a0
    p0 = [f_res, q0, k0.real, k0.imag, a0.real, a0.imag]
    lower = [f[0], 0.1, -50, -50, -5, -5]
    upper = [f[-1], 1e7, 50, 50, 5, 5]

    def resid(v):
        r = resonator_model(f, v[0], v[1], v[2] + 1j * v[3], v[4] + 1j * v[5]) - s
        return np.concatenate([r.real, r.imag])

    try:
        sol = least_squares(resid, p0, bounds=(lower, upper), x_scale=[w, q0, 1, 1, 1, 1], max_nfev=2000)
    except ValueError as exc:
        notes.append(f"fit failed: {exc}")
        return None
    if not sol.success:
        notes.append("fit did not converge; Q not reported")
        return None
    dof = sol.fun.size - len(p0)
    err = np.full(len(p0), np.nan)
    if dof > 0:
        cov = (2 * sol.cost / dof) * np.linalg.pinv(sol.jac.T @ sol.jac)
        err = np.sqrt(np.abs(np.diag(cov)))
    return {
        "p": sol.x,
        "err": err,
        "rms": float(np.sqrt(np.mean(sol.fun**2) * 2)),
        "window": (float(f[0]), float(f[-1])),
    }


def analyze(freq_hz, s11, level_db: float = -10.0, min_depth_db: float = 1.0) -> S11Result:
    """Resonance frequency, -10 dB bandwidth and loaded Q from a one-port sweep.

    freq_hz: frequency axis in Hz. s11: complex reflection coefficient.
    level_db: bandwidth threshold (default -10 dB). min_depth_db: no Q fit if the dip is shallower.
    """
    order = np.argsort(np.asarray(freq_hz, dtype=float))
    freq = np.asarray(freq_hz, dtype=float)[order]
    s = np.asarray(s11, dtype=complex)[order]
    if freq.size < 5:
        raise ValueError("need at least 5 points")
    db = _db(s)
    i_min = int(np.argmin(db))
    f_res = _refine_min(freq, db, i_min)

    res = S11Result(f_res, float(db[i_min]), 0.0, float("nan"), float("nan"), False)

    band = _contiguous_below(freq, db, i_min, level_db)
    if band is None:
        res.notes.append(f"|S11| never reaches {level_db:g} dB (minimum {db[i_min]:.2f} dB)")
    else:
        res.bw10_lo_hz, res.bw10_hi_hz = float(band[0]), float(band[1])
        res.bw10_hz = res.bw10_hi_hz - res.bw10_lo_hz
        res.bw10_clipped = bool(band[2] or band[3])
        if res.bw10_clipped:
            res.notes.append(f"the {level_db:g} dB band touches the end of the sweep; bandwidth is a lower bound")

    if db[i_min] > -min_depth_db:
        res.notes.append(f"dip is shallower than {min_depth_db:g} dB; Q not fitted")
        return res
    if i_min in (0, len(freq) - 1):
        res.notes.append("minimum sits at the end of the sweep; Q not fitted")
        return res

    fit = _fit_resonator(freq, s, i_min, f_res, res.notes)
    if fit is not None:
        p, err = fit["p"], fit["err"]
        res.f0_hz, res.q_loaded = float(p[0]), float(p[1])
        res.f0_err_hz, res.q_err = float(err[0]), float(err[1])
        res.fit_rms, res.fit_window_hz = fit["rms"], fit["window"]
        res.model = resonator_model(freq, p[0], p[1], p[2] + 1j * p[3], p[4] + 1j * p[5])
    return res


def analyze_file(path, **kwargs) -> S11Result:
    ts = read_s1p(path)
    return analyze(ts.freq_hz, ts.s11, **kwargs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("s1p")
    ap.add_argument("--level", type=float, default=-10.0, help="bandwidth threshold in dB (default -10)")
    ap.add_argument("--plot", default=None, help="save a PNG of |S11| and the fit")
    args = ap.parse_args(argv)

    ts = read_s1p(args.s1p)
    res = analyze(ts.freq_hz, ts.s11, level_db=args.level)
    print(f"f_res = {res.f_res_hz/1e9:.5f} GHz")
    print(f"S11 min = {res.s11_min_db:.2f} dB (reflected power fraction {10**(res.s11_min_db/10):.4f})")
    if res.bw10_hz > 0:
        print(f"{args.level:g} dB bandwidth = {res.bw10_hz/1e6:.2f} MHz "
              f"({res.bw10_lo_hz/1e9:.5f} to {res.bw10_hi_hz/1e9:.5f} GHz)")
    else:
        print(f"{args.level:g} dB bandwidth = 0 (not reached)")
    if np.isfinite(res.q_loaded):
        print(f"fit: f0 = {res.f0_hz/1e9:.5f} GHz +- {res.f0_err_hz/1e6:.2f} MHz | "
              f"Q_loaded = {res.q_loaded:.1f} +- {res.q_err:.1f} | rms = {res.fit_rms:.4f}")
    for note in res.notes:
        print(f"note: {note}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(ts.freq_hz / 1e9, _db(ts.s11), ".", ms=3, label="data")
        if res.model is not None:
            ax.plot(ts.freq_hz / 1e9, _db(res.model), "-", label="resonator fit")
        ax.axhline(args.level, color="gray", ls="--", lw=1, label=f"{args.level:g} dB")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("|S11| (dB)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()

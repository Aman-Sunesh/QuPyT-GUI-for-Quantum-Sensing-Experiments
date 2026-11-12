"""Fit ODMR spectra with a sum of Lorentzian dips and convert the splitting to field.

Input is a frequency axis and a normalized signal (baseline near 1, dips pointing down),
which matches the lower "contrast" trace that EduControl plots. A signal with peaks instead
of dips is detected automatically and flipped.

Command line:
    python -m qzabre.odmr spectrum.csv --dips 2 --plot fit.png
The CSV needs two columns: frequency (Hz or GHz) and signal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from . import nv


@dataclass
class Dip:
    freq_hz: float
    freq_err_hz: float
    fwhm_hz: float
    fwhm_err_hz: float
    depth: float  # fractional dip depth relative to baseline
    depth_err: float


@dataclass
class FitResult:
    baseline: float
    dips: list[Dip]
    popt: np.ndarray
    model: np.ndarray  # fitted curve on the input frequency axis (in the input's own orientation)
    flipped: bool


def lorentzian_dips(f, baseline, *params):
    """baseline - sum_i depth_i * (w_i/2)^2 / ((f - f_i)^2 + (w_i/2)^2); params = (f, fwhm, depth) * n."""
    y = np.full_like(np.asarray(f, dtype=float), baseline)
    for f0, w, a in zip(params[0::3], params[1::3], params[2::3]):
        hw2 = (w / 2.0) ** 2
        y = y - a * hw2 / ((f - f0) ** 2 + hw2)
    return y


def _guess(freq, y, n_dips):
    baseline = np.percentile(y, 95)
    depth = baseline - y
    step = np.median(np.diff(freq))
    prom = 0.15 * depth.max()
    idx, props = find_peaks(depth, prominence=prom, width=1)
    order = np.argsort(props["prominences"])[::-1]
    if n_dips is not None:
        order = order[:n_dips]
        if len(order) < n_dips:
            raise ValueError(f"found only {len(order)} dips, asked for {n_dips}; lower the noise or pass fewer")
    idx = idx[order]
    widths = props["widths"][order] * abs(step)
    return baseline, idx, widths, depth[idx]


def fit_odmr(freq_hz, signal, n_dips: int | None = None) -> FitResult:
    """Fit n_dips Lorentzian dips (auto-detected if n_dips is None)."""
    freq = np.asarray(freq_hz, dtype=float)
    y = np.asarray(signal, dtype=float)
    order = np.argsort(freq)
    freq, y = freq[order], y[order]

    # dips point down when the median sits in the upper half of the range
    flipped = bool(np.median(y) < 0.5 * (y.max() + y.min()))
    if flipped:
        y = y.max() + y.min() - y

    baseline, idx, widths, depths = _guess(freq, y, n_dips)
    p0, lo, hi = [baseline], [0.0], [np.inf]
    for i, w, a in zip(idx, widths, depths):
        w = max(w, 2 * abs(np.median(np.diff(freq))))
        p0 += [freq[i], w, a]
        lo += [freq[0], abs(np.median(np.diff(freq))), 0.0]
        hi += [freq[-1], freq[-1] - freq[0], np.inf]
    popt, pcov = curve_fit(lorentzian_dips, freq, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    err = np.sqrt(np.diag(pcov))

    dips = [
        Dip(popt[1 + 3 * k], err[1 + 3 * k], popt[2 + 3 * k], err[2 + 3 * k],
            popt[3 + 3 * k] / popt[0], err[3 + 3 * k] / popt[0])
        for k in range((len(popt) - 1) // 3)
    ]
    dips.sort(key=lambda d: d.freq_hz)
    model = lorentzian_dips(freq, *popt)
    if flipped:
        model = model.max() + model.min() - model  # best effort; plot uses flipped data anyway
    return FitResult(popt[0], dips, popt, model, flipped)


def field_from_two_dips(res: FitResult, gamma: float = nv.GAMMA_HZ_PER_T):
    """B along the NV axis (T), its 1-sigma error, and the total B if the field is along [100]."""
    if len(res.dips) != 2:
        raise ValueError("need exactly two dips")
    lo, hi = res.dips
    b_nv = nv.b_nv_from_splitting(hi.freq_hz, lo.freq_hz, gamma)
    err = np.hypot(hi.freq_err_hz, lo.freq_err_hz) / (2 * gamma)
    return b_nv, err, nv.b_total_for_100_field(b_nv)


def pair_dips(dips, d_hz: float = nv.D_HZ, tol_hz: float = 3e6, gamma: float = nv.GAMMA_HZ_PER_T):
    """Pair dips that sit symmetrically about D (one pair per NV orientation).

    Greedy match on |center - D|; centers within tol_hz count. Returns dicts with the two dips,
    the pair center, the splitting, and the field along that NV axis B_NV = splitting / (2 gamma).
    Orientations whose dips fall outside the sweep window, or overlap, will not appear.
    """
    cands = []
    ds = sorted(dips, key=lambda d: d.freq_hz)
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            center = 0.5 * (ds[i].freq_hz + ds[j].freq_hz)
            if abs(center - d_hz) <= tol_hz:
                cands.append((abs(center - d_hz), i, j))
    used, out = set(), []
    for _, i, j in sorted(cands):
        if i in used or j in used:
            continue
        used.update((i, j))
        split = ds[j].freq_hz - ds[i].freq_hz
        out.append({"low": ds[i], "high": ds[j], "center_hz": 0.5 * (ds[i].freq_hz + ds[j].freq_hz),
                    "split_hz": split, "b_nv_t": split / (2 * gamma)})
    return sorted(out, key=lambda p_: -p_["split_hz"])


def _load_csv(path):
    data = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1))
    f = data[:, 0]
    if f.max() < 1e3:  # GHz
        f = f * 1e9
    return f, data[:, 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--dips", type=int, default=None, help="number of dips (default: auto)")
    ap.add_argument("--plot", default=None, help="save a PNG of data and fit")
    args = ap.parse_args(argv)

    f, y = _load_csv(args.csv)
    res = fit_odmr(f, y, args.dips)
    print(f"baseline {res.baseline:.5f}")
    for k, d in enumerate(res.dips, 1):
        print(f"dip {k}: f = {d.freq_hz/1e9:.5f} GHz +- {d.freq_err_hz/1e6:.2f} MHz | "
              f"FWHM = {d.fwhm_hz/1e6:.2f} +- {d.fwhm_err_hz/1e6:.2f} MHz | depth = {d.depth:.4f}")
    if len(res.dips) == 2:
        b_nv, err, b100 = field_from_two_dips(res)
        print(f"B along NV axis = {b_nv*1e3:.3f} +- {err*1e3:.3f} mT")
        print(f"if B || [100]: |B| = {b100*1e3:.3f} mT")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(f / 1e9, y, ".", ms=3, label="data")
        ax.plot(f / 1e9, res.model, "-", label="Lorentzian fit")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Signal (normalized)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()

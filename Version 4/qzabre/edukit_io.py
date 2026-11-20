"""Read EduControl (QS3) ODMR run folders and fit them.

Layout (from tools/inspect_h5.py on an Oct 16 2026 run):
    <run>/scalarData.h5          /data (1,1) compound: EduControl's own fit (odmr:resonanceFrequency0, ...)
    <run>/seq0/eval/odmr.h5      /data (1,1,N) compound: frequency [Hz], sig [counts], contrast [normalized, dips down]
    <run>/seq0/integratedCounts.h5   /data (1,1,N): counts, timetag

Command line:
    python -m qzabre.edukit_io RUN_FOLDER [--dips N] [--plot out.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import nv, odmr


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("h5py is needed: pip install h5py") from exc
    return h5py


def find_odmr_file(path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    hits = sorted(p.rglob("odmr.h5"))
    if not hits:
        raise FileNotFoundError(f"no odmr.h5 under {p}")
    return hits[0]


def load_spectrum(path):
    """Return (frequency_hz, contrast, sig_counts) as 1-D arrays from an eval/odmr.h5 file or run folder."""
    h5py = _h5py()
    with h5py.File(find_odmr_file(path), "r") as f:
        d = f["data"][()]
    freq = np.asarray(d["frequency"], dtype=float).ravel()
    contrast = np.asarray(d["contrast"], dtype=float).ravel()
    sig = np.asarray(d["sig"], dtype=float).ravel()
    return freq, contrast, sig


def load_scalars(run_folder):
    """EduControl's own fit results as a dict (keys without the 'odmr:' prefix), or {} if absent."""
    h5py = _h5py()
    hits = sorted(Path(run_folder).rglob("scalarData.h5")) if Path(run_folder).is_dir() else []
    if not hits:
        return {}
    with h5py.File(hits[0], "r") as f:
        row = f["data"][()].ravel()[0]
        names = f["data"].dtype.names
    return {n.split(":", 1)[-1]: float(row[n]) for n in names}


def field_from_single_dip(f_hz, gamma=nv.GAMMA_HZ_PER_T):
    """Field component along the NV axis (T) implied by one dip at f_hz, assuming it is the lower branch."""
    return abs(nv.D_HZ - f_hz) / gamma


def lorentz_curve(f, f0, fwhm, depth, baseline=1.0):
    return odmr.lorentzian_dips(np.asarray(f, dtype=float), baseline, f0, fwhm, depth * baseline)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--dips", type=int, default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    freq, contrast, sig = load_spectrum(args.run)
    scal = load_scalars(args.run)
    print(f"{freq.size} points, {freq[0]/1e9:.4f} to {freq[-1]/1e9:.4f} GHz, step {np.median(np.diff(freq))/1e3:.0f} kHz")
    print(f"contrast: min {contrast.min():.4f}  max {contrast.max():.4f}  (a 1 % dip means min near 0.99)")
    res = odmr.fit_odmr(freq, contrast, n_dips=args.dips)
    print("\nOur fit (qzabre.odmr):")
    for k, d in enumerate(res.dips, 1):
        print(f"  dip {k}: f = {d.freq_hz/1e6:.3f} +- {d.freq_err_hz/1e6:.3f} MHz | FWHM = {d.fwhm_hz/1e6:.2f} +- {d.fwhm_err_hz/1e6:.2f} MHz | depth = {d.depth:.4f}")
        print(f"          offset from 2.87 GHz -> B along NV axis = {field_from_single_dip(d.freq_hz)*1e3:.3f} mT (single-dip, lower-branch assumption)")
    pairs = odmr.pair_dips(res.dips)
    if pairs:
        print("\nDips paired symmetrically about 2.87 GHz (one pair per NV orientation):")
        for k, pr in enumerate(pairs, 1):
            print(f"  pair {k}: {pr['low'].freq_hz/1e6:.2f} and {pr['high'].freq_hz/1e6:.2f} MHz | center {pr['center_hz']/1e6:.2f} MHz "
                  f"| splitting {pr['split_hz']/1e6:.2f} MHz | B along this NV axis = {pr['b_nv_t']*1e3:.3f} mT")
        print(f"  largest projection {pairs[0]['b_nv_t']*1e3:.3f} mT, so |B| is at least that. Orientations outside the sweep window are not seen.")
    if len(res.dips) == 2:
        b_nv, err, b100 = odmr.field_from_two_dips(res)
        print(f"  two dips: B_NV = {b_nv*1e3:.3f} +- {err*1e3:.3f} mT; |B| = {b100*1e3:.3f} mT if B || [100]")
    if scal:
        print("\nEduControl's own fit (scalarData.h5):")
        print(f"  f = {scal['resonanceFrequency0']/1e6:.3f} +- {scal['resonanceFrequency_err0']/1e6:.3f} MHz | FWHM = {scal['FWHM0']/1e6:.2f} +- {scal['FWHM_err0']/1e6:.2f} MHz | contrast = {scal['contrast0']:.4f}")
        print(f"  B = {scal['mTesla0']:.3f} mT | sensitivity = {scal['sensitivity0']:.3g} T/sqrt(Hz) | integration time = {scal['integrationTime']:.1f} s")
        if res.dips:
            d = res.dips[0]
            print(f"\nAgreement (first dip): frequency {abs(d.freq_hz-scal['resonanceFrequency0'])/1e6:.2f} MHz apart, "
                  f"FWHM ratio {d.fwhm_hz/scal['FWHM0']:.2f}, depth ratio {d.depth/scal['contrast0']:.2f}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(freq / 1e9, contrast, ".", ms=3, label="data (contrast)")
        ax.plot(freq / 1e9, res.model, "-", lw=2, label="qzabre fit")
        if scal:
            ax.plot(freq / 1e9, lorentz_curve(freq, scal["resonanceFrequency0"], scal["FWHM0"], scal["contrast0"]), "--", label="EduControl fit (baseline 1 assumed)")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Contrast")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    main()

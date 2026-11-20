"""Read EduControl (QS3) pulsed-experiment run folders: Rabi, Ramsey, echo, CPMG, relaxation.

Status: written from field names seen in an Oct 30 2026 Rabi run (eval/rabi.h5 with fields
pulseDuration, sig, ref, contrast; scalarData.h5 with fields like rabi:freq0, rabi:t_pi).
Tested on synthetic files with that layout only. Ramsey, echo and CPMG layouts are assumed to
follow the same pattern (first field = sweep variable, then sig, ref, contrast). Check with
tools/inspect_h5.py before trusting a result, and compare with EduControl's own fit.

Run folders are named YYYY-MM-DD-HH-MM-SS-<module>, for example 2026-10-30-21-38-28-rabi.

Command line:
    python -m qzabre.pulsed_io RUN_FOLDER [--module rabi] [--time-unit auto|s|ns|us] [--plot out.png]
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from . import edukit_io, rabi

_VALUE_FIELDS = ("sig", "ref", "contrast")
_UNIT_S = {"s": 1.0, "us": 1e-6, "ns": 1e-9}
_OSCILLATING = ("rabi", "ramsey")


@dataclass
class PulsedRun:
    module: str
    path: Path
    x_name: str
    x: np.ndarray  # sweep variable, in the file's own unit
    columns: dict  # every field of the eval table
    scalars: dict  # EduControl's own fit (keys without the module prefix), may hold NaN


def module_from_name(run) -> str:
    """Module name from a run folder such as 2026-10-30-21-38-28-rabi."""
    m = re.match(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-(.+)$", Path(run).name)
    if not m:
        raise ValueError(f"cannot read the module from {Path(run).name!r}; pass module=...")
    return m.group(1)


def find_eval_file(run, module: str) -> Path:
    p = Path(run)
    if p.is_file():
        return p
    hits = sorted(p.rglob(f"{module}.h5"))
    if not hits:
        raise FileNotFoundError(f"no {module}.h5 under {p}; run tools/inspect_h5.py on the folder to see what is there")
    return hits[0]


def load_run(run, module: str | None = None) -> PulsedRun:
    """Load the eval table and EduControl's scalar fit for one run folder."""
    module = module or module_from_name(run)
    h5py = edukit_io._h5py()
    path = find_eval_file(run, module)
    with h5py.File(path, "r") as f:
        d = f["data"][()]
    names = d.dtype.names
    if not names:
        raise ValueError(f"{path}: /data is not a compound table; inspect it with tools/inspect_h5.py")
    columns = {n: np.asarray(d[n], dtype=float).ravel() for n in names}
    x_candidates = [n for n in names if n not in _VALUE_FIELDS]
    if not x_candidates:
        raise ValueError(f"{path}: no sweep-variable field found among {names}")
    x_name = x_candidates[0]
    scalars = edukit_io.load_scalars(run) if Path(run).is_dir() else {}
    return PulsedRun(module, path, x_name, columns[x_name], columns, scalars)


def time_scale_to_s(x, unit: str = "auto") -> float:
    """Factor that converts the sweep variable to seconds. 'auto' assumes seconds if max(x) < 1e-3, else ns."""
    if unit == "auto":
        return 1.0 if np.nanmax(np.abs(x)) < 1e-3 else 1e-9
    return _UNIT_S[unit]


def _decay(t, offset, amp, tau):
    return offset + amp * np.exp(-t / tau)


def fit_decay(t_s, y):
    """Single exponential offset + amp * exp(-t/tau). Returns dict with tau_s, tau_err_s, model."""
    t = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)
    p0 = [y[-1], y[0] - y[-1], max(t[-1] - t[0], 1e-12) / 3]
    popt, pcov = curve_fit(_decay, t, y, p0=p0, maxfev=20000)
    return {"tau_s": abs(popt[2]), "tau_err_s": float(np.sqrt(pcov[2, 2])), "popt": popt, "model": _decay(t, *popt)}


def analyze(run: PulsedRun, time_unit: str = "auto") -> dict:
    """Fit the contrast trace. Oscillating modules (rabi, ramsey) get a damped cosine, others an exponential."""
    y = run.columns.get("contrast", run.columns.get("sig"))
    if y is None:
        raise ValueError("no contrast or sig column to fit")
    scale = time_scale_to_s(run.x, time_unit)
    t = np.asarray(run.x, dtype=float) * scale
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    out = {"t_s": t, "y": y, "time_scale_to_s": scale}
    if run.module in _OSCILLATING:
        fit = rabi.fit_rabi(t, y)
        out.update(kind="damped cosine", f_hz=fit["f_hz"], f_err_hz=fit["f_err_hz"], tau_s=fit["tau_s"],
                   pi_s=fit["pi_s"], model=fit["model"])
    else:
        fit = fit_decay(t, y)
        out.update(kind="exponential decay", tau_s=fit["tau_s"], tau_err_s=fit["tau_err_s"], model=fit["model"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--module", default=None, help="rabi, ramsey, echo, cpmg, ... (default: from the folder name)")
    ap.add_argument("--time-unit", default="auto", choices=["auto", "s", "us", "ns"])
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    run = load_run(args.run, args.module)
    x = run.x
    print(f"module: {run.module} | file: {run.path}")
    print(f"fields: {', '.join(run.columns)}")
    print(f"sweep variable '{run.x_name}': {x.size} points, {np.nanmin(x):.6g} to {np.nanmax(x):.6g} (file units)")
    res = analyze(run, args.time_unit)
    unit_note = "seconds" if res["time_scale_to_s"] == 1.0 else "nanoseconds" if res["time_scale_to_s"] == 1e-9 else "chosen unit"
    print(f"time unit used: {unit_note} (check this against the plot; pass --time-unit to override)")
    print(f"\nOur fit ({res['kind']}):")
    if "f_hz" in res:
        print(f"  frequency = {res['f_hz']/1e6:.4f} +- {res['f_err_hz']/1e6:.4f} MHz | pi time = {res['pi_s']*1e9:.2f} ns "
              f"| decay time = {res['tau_s']*1e6:.3f} us")
    else:
        print(f"  decay time = {res['tau_s']*1e6:.3f} +- {res['tau_err_s']*1e6:.3f} us")
    if run.scalars:
        print("\nEduControl's own fit (scalarData.h5), raw values, units not assumed:")
        for k, v in run.scalars.items():
            print(f"  {k} = {v:.6g}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(res["t_s"] * 1e9, res["y"], ".", ms=4, label="data (contrast)")
        ax.plot(res["t_s"] * 1e9, res["model"], "-", label="qzabre fit")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Contrast")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    main()

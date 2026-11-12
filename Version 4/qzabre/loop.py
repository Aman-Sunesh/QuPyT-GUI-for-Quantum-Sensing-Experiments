"""Single-turn loop antenna on a coax line: microwave power -> B1 -> Rabi frequency.

Model (Bucher 2019, Step 47: the inner conductor of a semi-rigid coax is bent into a loop and soldered
back to the outer conductor, so the loop is the termination of the line):

    load        Z_L = r + j*2*pi*f*L,   L = mu0 * R * (ln(8R/a) - 2)   (thin-wire circular loop)
    current     |I| = 2*sqrt(2*P*Z0) / |Z0 + Z_L|                    (peak, P = incident power)
    field       B_axis = mu0 * I * R^2 / (2 * (R^2 + z^2)^(3/2))        (peak, on the loop axis)
    drive       B1_perp = B_axis * sin(angle between loop axis and NV axis)
    Rabi        f_Rabi = gamma * B1_perp / sqrt(2)

Assumptions to remember: lossless line, lumped inductance (loop circumference below about lambda/10),
field evaluated on the axis, one turn, no conductor or substrate effects. Use it to size a loop and
to see trends. It is not a replacement for a full-wave simulation or a measurement.

Command line:  python -m qzabre.loop --plot examples/loop_antenna.png
"""

from __future__ import annotations

import argparse

import numpy as np

from . import nv

MU0 = 4e-7 * np.pi
C0 = 299792458.0
SIN_100_TO_111 = np.sqrt(2.0 / 3.0)  # sin(54.74 deg): drive component perpendicular to an NV axis for B1 along [100]


def loop_inductance(radius_m, wire_radius_m):
    """Self-inductance (H) of a circular loop of thin round wire."""
    return MU0 * radius_m * (np.log(8.0 * radius_m / wire_radius_m) - 2.0)


def loop_current(p_w, f_hz, radius_m, wire_radius_m, z0_ohm=50.0, r_ohm=0.0):
    """Peak current (A) in a loop terminating a z0_ohm line carrying incident power p_w."""
    zl = r_ohm + 1j * 2 * np.pi * f_hz * loop_inductance(radius_m, wire_radius_m)
    return 2.0 * np.sqrt(2.0 * p_w * z0_ohm) / np.abs(z0_ohm + zl)


def b_on_axis(current_a, radius_m, z_m=0.0):
    """Peak field (T) on the axis of a loop, a distance z from its plane."""
    return MU0 * current_a * radius_m**2 / (2.0 * (radius_m**2 + z_m**2) ** 1.5)


def is_electrically_small(radius_m, f_hz) -> bool:
    """True if the circumference is below lambda/10 (the lumped-inductor model is then reasonable)."""
    return 2 * np.pi * radius_m < (C0 / f_hz) / 10.0


def drive(p_w, f_hz, radius_m, wire_radius_m, z_m=0.0, nv_factor=SIN_100_TO_111, z0_ohm=50.0, r_ohm=0.0):
    """Return a dict with current, field, effective drive field, Rabi frequency and pi-pulse time."""
    i = loop_current(p_w, f_hz, radius_m, wire_radius_m, z0_ohm, r_ohm)
    b = b_on_axis(i, radius_m, z_m)
    b1 = b * nv_factor
    f_rabi = nv.rabi_from_b1(b1)
    return {
        "current_a": i,
        "b_axis_t": b,
        "b1_perp_t": b1,
        "rabi_hz": f_rabi,
        "pi_pulse_s": 1.0 / (2.0 * f_rabi),
        "small": is_electrically_small(radius_m, f_hz),
    }


def power_for_rabi(target_rabi_hz, f_hz, radius_m, wire_radius_m, z_m=0.0, nv_factor=SIN_100_TO_111, **kw):
    """Incident power (W) needed for a target Rabi frequency (field and Rabi scale as sqrt(P))."""
    r1 = drive(1.0, f_hz, radius_m, wire_radius_m, z_m, nv_factor, **kw)["rabi_hz"]
    return (target_rabi_hz / r1) ** 2


# ---------------------------------------------------------------------------------------------
# Reflection (S11) and matching network.  Everything below is a MODEL; resistances are ESTIMATES.
#
#   VNA --- lossless line (Zc, eps_eff, length) --- shunt Cp --- series Cs --- loop (r + j*w*L)
#
# A lossless inductor reflects everything (|S11| = 1), so matching needs a real part r. r is the
# copper skin-effect loss plus radiation. Both are tiny (order 0.1 ohm), which makes the match
# narrowband and very sensitive to Cs. Capacitors are ideal unless q_cap is given.
# ---------------------------------------------------------------------------------------------

SIGMA_CU = 5.8e7  # S/m, bulk copper (handbook value; real PCB copper is a bit worse)


def strip_equivalent_radius(width_m, thickness_m=0.0):
    """Round-wire radius (m) with the same self-inductance as a flat trace (GMD rule, a = 0.2235*(w+t)).

    Approximation. Ignores any ground plane under the loop, which lowers the real inductance.
    """
    return 0.2235 * (width_m + thickness_m)


def ohmic_resistance(f_hz, radius_m, perimeter_m, sigma=SIGMA_CU):
    """Skin-effect resistance (ohm) of a loop whose conducting perimeter is perimeter_m.

    Estimate: uniform current over the perimeter. Current crowding on a flat trace makes the
    real value higher.
    """
    rs = np.sqrt(np.pi * f_hz * MU0 / sigma)
    return rs * (2 * np.pi * radius_m) / perimeter_m


def radiation_resistance(f_hz, radius_m):
    """Small-loop radiation resistance (ohm), 320*pi^6*(R/lambda)^4, free space."""
    lam = C0 / f_hz
    return 320.0 * np.pi**6 * (radius_m / lam) ** 4


def loss_resistance(f_hz, radius_m, perimeter_m, sigma=SIGMA_CU):
    """Estimated total series resistance (ohm): ohmic plus radiation."""
    return ohmic_resistance(f_hz, radius_m, perimeter_m, sigma) + radiation_resistance(f_hz, radius_m)


def reflection(z_ohm, z0_ohm=50.0):
    """S11 (complex) of impedance z_ohm seen from a z0_ohm reference."""
    return (z_ohm - z0_ohm) / (z_ohm + z0_ohm)


def through_line(z_load, f_hz, zc_ohm, eeff, length_m):
    """Input impedance of a lossless line (characteristic impedance zc_ohm) ending in z_load."""
    th = 2 * np.pi * f_hz * np.sqrt(eeff) * length_m / C0
    c, s = np.cos(th), np.sin(th)
    return zc_ohm * (z_load * c + 1j * zc_ohm * s) / (zc_ohm * c + 1j * z_load * s)


class Match:
    """Series capacitor cs_f (next to the loop) and shunt capacitor cp_f (at the feed side), in farad."""

    def __init__(self, cs_f, cp_f):
        self.cs_f = float(cs_f)
        self.cp_f = float(cp_f)

    def __repr__(self):
        return f"Match(cs={self.cs_f*1e12:.4f} pF, cp={self.cp_f*1e12:.4f} pF)"


def branch_impedance(f_hz, l_h, r_ohm, cs_f=None, q_cap=None):
    """Loop branch: r + j*w*L, plus the series capacitor (with ESR = X_c/q_cap if q_cap is given)."""
    w = 2 * np.pi * np.asarray(f_hz, float)
    z = r_ohm + 1j * w * l_h
    if cs_f is not None:
        xc = 1.0 / (w * cs_f)
        z = z - 1j * xc + (xc / q_cap if q_cap else 0.0)
    return z


def network_impedance(f_hz, l_h, r_ohm, match=None, q_cap=None):
    """Impedance at the feed side of the matching network (shunt Cp across the series branch)."""
    if match is None:
        return branch_impedance(f_hz, l_h, r_ohm)
    zb = branch_impedance(f_hz, l_h, r_ohm, match.cs_f, q_cap)
    return 1.0 / (1.0 / zb + 1j * 2 * np.pi * np.asarray(f_hz, float) * match.cp_f)


def s11_loop(f_hz, radius_m, wire_radius_m, r_ohm, match=None, z0_ohm=50.0, q_cap=None):
    """S11 (complex) of the loop, bare or behind a matching network, on an ideal z0_ohm line."""
    l_h = loop_inductance(radius_m, wire_radius_m)
    return reflection(network_impedance(f_hz, l_h, r_ohm, match, q_cap), z0_ohm)


def design_match(f0_hz, l_h, r_ohm, z0_ohm=50.0, q_cap=None):
    """Series-C plus shunt-C match of a loop (L, r) to z0_ohm at f0_hz.

    Series branch is made r_b + jX with X = sqrt(r_b*(z0 - r_b)); the shunt susceptance X/(r_b*z0) then
    cancels it. With q_cap the capacitor ESR adds to r_b, solved by fixed-point iteration.
    """
    if r_ohm <= 0:
        raise ValueError("r_ohm must be > 0: a lossless inductor cannot be matched. See loss_resistance().")
    w = 2 * np.pi * f0_hz
    rb = r_ohm
    for _ in range(50):
        if rb >= z0_ohm:
            raise ValueError("series resistance (incl. capacitor ESR) must be below z0")
        x = np.sqrt(rb * (z0_ohm - rb))
        if w * l_h <= x:
            raise ValueError("loop reactance is too small for this match; the loop would need a series inductor")
        xc = w * l_h - x
        rb_new = r_ohm + (xc / q_cap if q_cap else 0.0)
        if abs(rb_new - rb) < 1e-15:
            break
        rb = rb_new
    x = np.sqrt(rb * (z0_ohm - rb))
    return Match(cs_f=1.0 / (w * (w * l_h - x)), cp_f=x / (rb * z0_ohm) / w)


def port_current(p_w, zin_ohm, zbranch_ohm, z0_ohm=50.0):
    """Peak current (A) in the loop branch for incident power p_w (shunt Cp does not carry loop current)."""
    v_inc = np.sqrt(2.0 * p_w * z0_ohm)
    return np.abs(v_inc * (1.0 + reflection(zin_ohm, z0_ohm)) / zbranch_ohm)


def drive_matched(p_w, f_hz, radius_m, wire_radius_m, r_ohm, match=None, z_m=0.0, nv_factor=SIN_100_TO_111,
                  z0_ohm=50.0, q_cap=None):
    """Like drive(), but with a matching network and a resistive loop. Returns the same keys plus s11_mag."""
    l_h = loop_inductance(radius_m, wire_radius_m)
    zb = branch_impedance(f_hz, l_h, r_ohm, None if match is None else match.cs_f, q_cap)
    zin = network_impedance(f_hz, l_h, r_ohm, match, q_cap)
    i = port_current(p_w, zin, zb, z0_ohm)
    b = b_on_axis(i, radius_m, z_m)
    f_rabi = nv.rabi_from_b1(b * nv_factor)
    return {"current_a": i, "b_axis_t": b, "rabi_hz": f_rabi, "s11_mag": np.abs(reflection(zin, z0_ohm))}


def s11_bandwidth(f_hz, gamma, level_db=-10.0):
    """Width (Hz) of the contiguous band around the S11 minimum that stays below level_db. 0 if never below."""
    f_hz = np.asarray(f_hz, float)
    db = 20 * np.log10(np.maximum(np.abs(gamma), 1e-12))
    k = int(np.argmin(db))
    if db[k] > level_db:
        return 0.0
    lo = k
    while lo > 0 and db[lo - 1] <= level_db:
        lo -= 1
    hi = k
    while hi < len(db) - 1 and db[hi + 1] <= level_db:
        hi += 1
    return float(f_hz[hi] - f_hz[lo])


def _print_and_plot_s11(args, f0, a, z):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_m = args.match_radius_mm * 1e-3
    l_h = loop_inductance(r_m, a)
    r_ohm = loss_resistance(f0, r_m, 2 * np.pi * a)  # round wire: conducting perimeter 2*pi*a
    m = design_match(f0, l_h, r_ohm, q_cap=args.q_cap)
    fg = np.linspace(0.5 * f0, 1.5 * f0, 2001)
    g_bare = s11_loop(fg, r_m, a, r_ohm)
    g_match = s11_loop(fg, r_m, a, r_ohm, m, q_cap=args.q_cap)
    bw = s11_bandwidth(fg, g_match)
    d_bare = drive_matched(args.power_w, f0, r_m, a, r_ohm, None, z)
    d_match = drive_matched(args.power_w, f0, r_m, a, r_ohm, m, z, q_cap=args.q_cap)
    print(f"\nS11 / match model (all MODEL values; resistance is an ESTIMATE): R = {args.match_radius_mm} mm")
    print(f"L = {l_h*1e9:.2f} nH, r = {r_ohm*1e3:.0f} mohm (ohmic + radiation, estimate), {m}")
    print(f"-10 dB bandwidth = {bw/1e6:.0f} MHz;  Rabi at {args.power_w} W: bare {d_bare['rabi_hz']/1e6:.2f} MHz, "
          f"matched {d_match['rabi_hz']/1e6:.2f} MHz")
    fz = np.linspace(f0 - 40e6, f0 + 40e6, 2001)
    gz = s11_loop(fz, r_m, a, r_ohm, m, q_cap=args.q_cap)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].plot(fg / 1e9, 20 * np.log10(np.abs(g_bare)), label="bare loop")
    axs[0].plot(fg / 1e9, 20 * np.log10(np.maximum(np.abs(g_match), 1e-6)), label="series C + shunt C match")
    axs[0].set(xlabel="Frequency (GHz)", ylabel="|S11| (dB)", ylim=(-40, 1), title="Wide view (model, not a measurement)")
    axs[0].legend(fontsize=8, loc="lower left")
    axs[1].plot((fz - f0) / 1e6, 20 * np.log10(np.maximum(np.abs(gz), 1e-6)), color="tab:orange")
    axs[1].set(xlabel="Offset from design frequency (MHz)", ylabel="|S11| (dB)", ylim=(-40, 1), title="Zoom on the matched notch")
    for a_ in axs:
        a_.axhline(-10, color="gray", lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(args.s11_plot, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq-ghz", type=float, default=2.87)
    ap.add_argument("--wire-radius-mm", type=float, default=0.15, help="default is about 0.047 in semi-rigid inner conductor")
    ap.add_argument("--z-mm", type=float, default=0.2, help="NV layer distance below the loop plane")
    ap.add_argument("--power-w", type=float, default=1.0)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--s11-plot", default=None, help="plot S11 versus frequency, bare loop and matched (model)")
    ap.add_argument("--match-radius-mm", type=float, default=1.5, help="loop radius for the S11 / match model")
    ap.add_argument("--q-cap", type=float, default=None, help="capacitor Q for the match model (default: ideal)")
    args = ap.parse_args(argv)

    f = args.freq_ghz * 1e9
    a = args.wire_radius_mm * 1e-3
    z = args.z_mm * 1e-3
    print(f"f = {args.freq_ghz} GHz, wire radius {args.wire_radius_mm} mm, z = {args.z_mm} mm, incident power {args.power_w} W")
    print("R (mm) | L (nH) | I (A) | B1perp (uT) | Rabi (MHz) | pi (ns) | P for 25 MHz Rabi (W) | small?")
    radii = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    for r_mm in radii:
        r = r_mm * 1e-3
        d = drive(args.power_w, f, r, a, z)
        p25 = power_for_rabi(25e6, f, r, a, z)
        print(f"{r_mm:6.2f} | {loop_inductance(r, a)*1e9:6.2f} | {d['current_a']:5.3f} | {d['b1_perp_t']*1e6:10.1f} | "
              f"{d['rabi_hz']/1e6:9.2f} | {d['pi_pulse_s']*1e9:7.1f} | {p25:21.1f} | {'yes' if d['small'] else 'NO'}")

    if args.s11_plot:
        _print_and_plot_s11(args, f, a, z)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = np.linspace(0.4e-3, 5e-3, 200)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for a_mm, ls in ((0.1, ":"), (0.15, "-"), (0.3, "--")):
            rabi = np.array([drive(1.0, f, x, a_mm * 1e-3, z)["rabi_hz"] for x in r])
            axs[0].plot(r * 1e3, rabi / 1e6, ls, color="tab:blue", label=f"wire radius {a_mm} mm")
        axs[0].axhline(nv.RABI_AT_1W_HZ / 1e6, color="tab:green", lw=1)
        axs[0].text(0.45, nv.RABI_AT_1W_HZ / 1e6 - 0.9, "Edukit resonator at 1 W (manual Eq. 6.2)", color="tab:green", fontsize=8)
        axs[0].set_xlabel("Loop radius (mm)")
        axs[0].set_ylabel("Rabi frequency at 1 W (MHz)")
        axs[0].set_title("Smaller loop, more field")
        axs[0].legend(fontsize=8, loc="center right")
        p = np.logspace(-1, 1.3, 100)
        for r_mm in (0.75, 1.5, 3.0):
            pi_ns = np.array([drive(x, f, r_mm * 1e-3, a, z)["pi_pulse_s"] for x in p]) * 1e9
            axs[1].loglog(p, pi_ns, label=f"R = {r_mm} mm")
        axs[1].axhspan(20, 60, color="tab:orange", alpha=0.15)
        axs[1].text(0.12, 62, "Bucher: pi pulse 20-60 ns", fontsize=8, color="tab:orange")
        axs[1].set_xlabel("Incident power (W)")
        axs[1].set_ylabel("pi-pulse duration (ns)")
        axs[1].set_title(f"Power needed ({args.freq_ghz} GHz, z = {args.z_mm} mm)")
        axs[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)


if __name__ == "__main__":
    main()

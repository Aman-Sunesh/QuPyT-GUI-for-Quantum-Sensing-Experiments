# rf_tab.py

# ────────────────────────────────────────────────────────────────
# RF tab (Version 4): microwave antenna and PCB helpers.
#
#   Microstrip    trace impedance and the width for a target impedance
#   Loop antenna  power -> current -> B1 -> Rabi frequency, plus a
#                 series-C / shunt-C matching model and its S11
#   S11 viewer    read a VNA .s1p file: resonance, bandwidth, loaded Q
#
# The first two are MODELS, not measurements. The S11 viewer reports
# what is in the file you load. Math lives in the qzabre package.
# ────────────────────────────────────────────────────────────────

import os

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QFormLayout, QGroupBox, QVBoxLayout

from tab_helpers import DASH, Readout, missing_backend, note, side_by_side, spin

try:
    from qzabre import loop, rf, s11
    _IMPORT_ERROR = None
except Exception as exc:  # the tab still opens and explains what is missing
    loop = rf = s11 = None
    _IMPORT_ERROR = exc

_DASHED = QtCore.Qt.PenStyle.DashLine


def _db(gamma):
    return 20 * np.log10(np.maximum(np.abs(gamma), 1e-6))


class MicrostripPanel(QtWidgets.QWidget):
    """Hammerstad-Jensen quasi-static microstrip: impedance, eps_eff, width for a target impedance."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Defaults are generic FR4 values, not a specific board.
        self.width = spin(3.0, 0.01, 100.0, 3, 0.05, "mm")
        self.height = spin(1.6, 0.01, 20.0, 3, 0.05, "mm")
        self.er = spin(4.4, 1.0, 100.0, 3, 0.1)
        self.thickness = spin(0.035, 0.0, 1.0, 3, 0.005, "mm")
        self.target = spin(50.0, 5.0, 300.0, 2, 1.0, "ohm")
        self.freq = spin(2.87, 0.01, 40.0, 4, 0.01, "GHz")

        inputs = QGroupBox("Inputs")
        form = QFormLayout(inputs)
        form.addRow("Trace width", self.width)
        form.addRow("Substrate height", self.height)
        form.addRow("Relative permittivity", self.er)
        form.addRow("Copper thickness", self.thickness)
        form.addRow("Target impedance", self.target)
        form.addRow("Frequency", self.freq)

        self.out = Readout("Results (model)", [
            ("z0", "Impedance of this trace"),
            ("eeff", "Effective permittivity"),
            ("w_target", "Width for target impedance"),
            ("quarter", "Quarter wavelength on this trace"),
        ])

        left = QtWidgets.QWidget()
        col = QVBoxLayout(left)
        col.addWidget(inputs)
        col.addWidget(self.out)
        col.addWidget(note(
            "Quasi-static Hammerstad-Jensen equations. Frequency dispersion is not included, "
            "so eps_eff is a few percent low at a few GHz on thick substrates."
        ))
        col.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Trace width (mm)")
        self.plot.setLabel("left", "Impedance (ohm)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen=pg.mkPen("c", width=2))
        self.width_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("y", style=_DASHED))
        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("g", style=_DASHED))
        self.plot.addItem(self.width_line)
        self.plot.addItem(self.target_line)

        QVBoxLayout(self).addWidget(side_by_side(left, self.plot))

        for box in (self.width, self.height, self.er, self.thickness, self.target, self.freq):
            box.valueChanged.connect(self.refresh)
        self.refresh()

    def refresh(self, *_):
        w, h, er, t = self.width.value(), self.height.value(), self.er.value(), self.thickness.value()
        z0, eeff = rf.microstrip(w, h, er, t)
        self.out.set("z0", f"{z0:.1f} ohm")
        self.out.set("eeff", f"{eeff:.3f}")
        try:
            w_target = rf.width_for_z0(self.target.value(), h, er, t)
            self.out.set("w_target", f"{w_target:.3f} mm")
        except ValueError:
            self.out.set("w_target", "outside 0.05 to 20 times the substrate height")
        self.out.set("quarter", f"{rf.quarter_wave_mm(self.freq.value() * 1e9, eeff):.2f} mm")

        widths = np.linspace(0.1 * h, max(5.0 * h, 1.2 * w), 300)
        self.curve.setData(widths, [rf.microstrip(x, h, er, t)[0] for x in widths])
        self.width_line.setValue(w)
        self.target_line.setValue(self.target.value())


class LoopPanel(QtWidgets.QWidget):
    """Single-turn loop on a 50 ohm line: drive strength, and a two-capacitor match."""

    SPAN_HZ = 40e6  # half-width of the S11 plot around the design frequency

    def __init__(self, parent=None):
        super().__init__(parent)

        self.freq = spin(2.87, 0.1, 20.0, 4, 0.01, "GHz")
        self.power = spin(1.0, 0.001, 100.0, 3, 0.1, "W")
        self.radius = spin(1.5, 0.1, 20.0, 3, 0.1, "mm")
        self.wire = spin(0.15, 0.01, 5.0, 3, 0.01, "mm")
        self.z = spin(0.2, 0.0, 20.0, 3, 0.05, "mm")
        self.q_cap = spin(0.0, 0.0, 100000.0, 0, 50.0)
        self.q_cap.setSpecialValueText("ideal")

        inputs = QGroupBox("Inputs")
        form = QFormLayout(inputs)
        form.addRow("Frequency", self.freq)
        form.addRow("Incident power", self.power)
        form.addRow("Loop radius", self.radius)
        form.addRow("Wire radius", self.wire)
        form.addRow("NV distance from loop plane", self.z)
        form.addRow("Capacitor Q", self.q_cap)

        self.out = Readout("Bare loop (model)", [
            ("l", "Inductance"),
            ("i", "Peak current"),
            ("b", "Field on axis"),
            ("b1", "Drive field, perpendicular to NV"),
            ("rabi", "Rabi frequency"),
            ("pi", "pi-pulse time"),
            ("small", "Electrically small"),
        ])
        self.match = Readout("Matched loop (model, resistance is an estimate)", [
            ("r", "Series resistance"),
            ("cs", "Series capacitor"),
            ("cp", "Shunt capacitor"),
            ("bw", "-10 dB bandwidth"),
            ("rabi", "Rabi frequency, matched"),
        ])

        left = QtWidgets.QWidget()
        col = QVBoxLayout(left)
        col.addWidget(inputs)
        col.addWidget(self.out)
        col.addWidget(self.match)
        col.addWidget(note(
            "Lossless line, lumped inductance, field on the loop axis, one turn. The drive field "
            "assumes a [100]-cut diamond with the loop axis along [100]. Use it for sizing and "
            "trends, not in place of a full-wave simulation or a measurement."
        ))
        col.addStretch(1)

        self.rabi_plot = pg.PlotWidget()
        self.rabi_plot.setLabel("bottom", "Loop radius (mm)")
        self.rabi_plot.setLabel("left", "Rabi frequency, bare loop (MHz)")
        self.rabi_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rabi_curve = self.rabi_plot.plot(pen=pg.mkPen("c", width=2))
        self.radius_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("y", style=_DASHED))
        self.rabi_plot.addItem(self.radius_line)

        self.s11_plot = pg.PlotWidget()
        self.s11_plot.setLabel("bottom", "Offset from design frequency (MHz)")
        self.s11_plot.setLabel("left", "|S11| (dB)")
        self.s11_plot.showGrid(x=True, y=True, alpha=0.3)
        self.s11_plot.addLegend(offset=(-10, -10))
        self.bare_curve = self.s11_plot.plot(pen=pg.mkPen("w", width=1), name="bare loop")
        self.match_curve = self.s11_plot.plot(pen=pg.mkPen("m", width=2), name="series C + shunt C")
        self.s11_plot.addItem(pg.InfiniteLine(pos=-10, angle=0, pen=pg.mkPen("g", style=_DASHED)))
        self.s11_plot.setYRange(-40, 1)

        plots = QtWidgets.QWidget()
        plot_col = QVBoxLayout(plots)
        plot_col.setContentsMargins(0, 0, 0, 0)
        plot_col.addWidget(self.rabi_plot)
        plot_col.addWidget(self.s11_plot)

        QVBoxLayout(self).addWidget(side_by_side(left, plots))

        for box in (self.freq, self.power, self.radius, self.wire, self.z, self.q_cap):
            box.valueChanged.connect(self.refresh)
        self.refresh()

    def refresh(self, *_):
        f = self.freq.value() * 1e9
        p = self.power.value()
        r = self.radius.value() * 1e-3
        a = self.wire.value() * 1e-3
        z = self.z.value() * 1e-3
        q_cap = self.q_cap.value() or None

        if r <= a:
            self.out.clear()
            self.match.clear()
            self.out.set("l", "loop radius must be larger than the wire radius")
            for curve in (self.rabi_curve, self.bare_curve, self.match_curve):
                curve.setData([], [])
            return

        d = loop.drive(p, f, r, a, z)
        l_h = loop.loop_inductance(r, a)
        self.out.set("l", f"{l_h * 1e9:.2f} nH")
        self.out.set("i", f"{d['current_a']:.3f} A")
        self.out.set("b", f"{d['b_axis_t'] * 1e6:.1f} uT")
        self.out.set("b1", f"{d['b1_perp_t'] * 1e6:.1f} uT")
        self.out.set("rabi", f"{d['rabi_hz'] / 1e6:.2f} MHz")
        self.out.set("pi", f"{d['pi_pulse_s'] * 1e9:.1f} ns")
        self.out.set("small", "yes" if d["small"] else "NO: circumference above lambda/10, model unreliable")

        radii = np.linspace(max(0.4e-3, 1.2 * a), max(5e-3, 1.2 * r), 200)
        self.rabi_curve.setData(radii * 1e3, [loop.drive(p, f, x, a, z)["rabi_hz"] / 1e6 for x in radii])
        self.radius_line.setValue(r * 1e3)

        r_ohm = loop.loss_resistance(f, r, 2 * np.pi * a)  # round wire: conducting perimeter 2*pi*a
        fz = np.linspace(f - self.SPAN_HZ, f + self.SPAN_HZ, 2001)
        offset_mhz = (fz - f) / 1e6
        self.bare_curve.setData(offset_mhz, _db(loop.s11_loop(fz, r, a, r_ohm)))
        self.match.clear()
        self.match.set("r", f"{r_ohm * 1e3:.0f} mohm (ohmic + radiation)")
        try:
            m = loop.design_match(f, l_h, r_ohm, q_cap=q_cap)
        except ValueError as exc:
            self.match.set("cs", str(exc))
            self.match_curve.setData([], [])
            return
        gamma = loop.s11_loop(fz, r, a, r_ohm, m, q_cap=q_cap)
        dm = loop.drive_matched(p, f, r, a, r_ohm, m, z, q_cap=q_cap)
        self.match.set("cs", f"{m.cs_f * 1e12:.4f} pF")
        self.match.set("cp", f"{m.cp_f * 1e12:.4f} pF")
        self.match.set("bw", f"{loop.s11_bandwidth(fz, gamma) / 1e6:.2f} MHz")
        self.match.set("rabi", f"{dm['rabi_hz'] / 1e6:.2f} MHz")
        self.match_curve.setData(offset_mhz, _db(gamma))


class S11Panel(QtWidgets.QWidget):
    """Load a one-port Touchstone file and report resonance, bandwidth and loaded Q."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ts = None

        self.open_btn = QtWidgets.QPushButton("Open .s1p file…")
        self.path_label = QtWidgets.QLabel("No file loaded")
        self.path_label.setWordWrap(True)
        self.level = spin(-10.0, -60.0, -0.1, 1, 1.0, "dB")

        inputs = QGroupBox("File")
        box = QVBoxLayout(inputs)
        box.addWidget(self.open_btn)
        box.addWidget(self.path_label)
        form = QFormLayout()
        form.addRow("Bandwidth level", self.level)
        box.addLayout(form)

        self.out = Readout("Results (from the file)", [
            ("f_res", "Lowest |S11| at"),
            ("s_min", "Lowest |S11|"),
            ("refl", "Reflected power there"),
            ("bw", "Bandwidth at level"),
            ("edges", "Band edges"),
            ("f0", "Fitted resonance"),
            ("q", "Loaded Q"),
            ("rms", "Fit rms residual"),
            ("notes", "Notes"),
        ])

        left = QtWidgets.QWidget()
        col = QVBoxLayout(left)
        col.addWidget(inputs)
        col.addWidget(self.out)
        col.addWidget(note(
            "Q comes from a complex fit of S11 near the dip, so cable phase and mismatch do not "
            "bias it. It is not fitted when the dip is shallower than 1 dB."
        ))
        col.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Frequency (GHz)")
        self.plot.setLabel("left", "|S11| (dB)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend(offset=(-10, -10))
        self.data_curve = self.plot.plot(pen=None, symbol="o", symbolSize=4, symbolBrush="c",
                                         symbolPen=None, name="data")
        self.fit_curve = self.plot.plot(pen=pg.mkPen("m", width=2), name="resonator fit")
        self.level_line = pg.InfiniteLine(pos=-10, angle=0, pen=pg.mkPen("g", style=_DASHED))
        self.plot.addItem(self.level_line)

        QVBoxLayout(self).addWidget(side_by_side(left, self.plot))

        self.open_btn.clicked.connect(self._choose_file)
        self.level.valueChanged.connect(self.refresh)

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Touchstone file", "", "Touchstone one-port (*.s1p);;All files (*)"
        )
        if path:
            self.load_file(path)

    def load_file(self, path):
        try:
            self._ts = s11.read_s1p(path)
        except (OSError, ValueError) as exc:
            self._ts = None
            self.out.clear()
            self.path_label.setText(f"Could not read {os.path.basename(path)}")
            self.out.set("notes", str(exc))
            self.data_curve.setData([], [])
            self.fit_curve.setData([], [])
            return
        self.path_label.setText(f"{os.path.basename(path)}\n{self._ts.freq_hz.size} points, "
                                f"reference {self._ts.z0_ohm:g} ohm")
        self.refresh()

    def refresh(self, *_):
        level = self.level.value()
        self.level_line.setValue(level)
        if self._ts is None:
            return
        ts = self._ts
        self.out.clear()
        try:
            res = s11.analyze(ts.freq_hz, ts.s11, level_db=level)
        except ValueError as exc:
            self.out.set("notes", str(exc))
            return

        self.out.set("f_res", f"{res.f_res_hz / 1e9:.5f} GHz")
        self.out.set("s_min", f"{res.s11_min_db:.2f} dB")
        self.out.set("refl", f"{100 * 10 ** (res.s11_min_db / 10):.2f} %")
        if res.bw10_hz > 0:
            bound = " (lower bound)" if res.bw10_clipped else ""
            self.out.set("bw", f"{res.bw10_hz / 1e6:.2f} MHz{bound}")
            self.out.set("edges", f"{res.bw10_lo_hz / 1e9:.5f} to {res.bw10_hi_hz / 1e9:.5f} GHz")
        else:
            self.out.set("bw", "0 (level not reached)")
        if np.isfinite(res.q_loaded):
            self.out.set("f0", f"{res.f0_hz / 1e9:.5f} GHz ± {res.f0_err_hz / 1e6:.2f} MHz")
            self.out.set("q", f"{res.q_loaded:.1f} ± {res.q_err:.1f}")
            self.out.set("rms", f"{res.fit_rms:.4f}")
        self.out.set("notes", "\n".join(res.notes) if res.notes else DASH)

        order = np.argsort(ts.freq_hz)
        freq_ghz = ts.freq_hz[order] / 1e9
        self.data_curve.setData(freq_ghz, _db(ts.s11[order]))
        if res.model is not None:
            self.fit_curve.setData(freq_ghz, _db(res.model))
        else:
            self.fit_curve.setData([], [])


class RFTab(QtWidgets.QWidget):
    """The 'RF' tab of the main window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        if _IMPORT_ERROR is not None:
            layout.addWidget(missing_backend(_IMPORT_ERROR))
            return
        self.microstrip = MicrostripPanel()
        self.loop = LoopPanel()
        self.s11 = S11Panel()
        self.pages = QtWidgets.QTabWidget()
        self.pages.addTab(self.microstrip, "Microstrip")
        self.pages.addTab(self.loop, "Loop antenna")
        self.pages.addTab(self.s11, "S11 viewer")
        layout.addWidget(self.pages)

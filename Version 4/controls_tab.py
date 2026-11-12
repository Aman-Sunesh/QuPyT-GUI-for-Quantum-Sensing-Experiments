# controls_tab.py

# ────────────────────────────────────────────────────────────────
# Controls tab (Version 4): SIMULATED resonance tracking.
#
#   Dip tracking  probe both sides of an ODMR dip, steer the microwave
#                 back to the center with an integrator (photon shot noise)
#   Lock-in       dither the microwave, demodulate, close a PI loop
#
# Nothing here talks to hardware. Use it to choose loop gains and to
# see what tracking error to expect before building the real loop.
# Every output is a model value for the inputs shown.
# Math lives in the qzabre package.
# ────────────────────────────────────────────────────────────────

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout

from tab_helpers import Readout, int_spin, missing_backend, note, side_by_side, spin

try:
    from qzabre import lockin, tracking
    _IMPORT_ERROR = None
except Exception as exc:  # the tab still opens and explains what is missing
    lockin = tracking = None
    _IMPORT_ERROR = exc

MAX_TRACK_STEPS = 200_000
MAX_LOCKIN_SAMPLES = 20_000_000


class _SimPanel(QtWidgets.QWidget):
    """Shared behavior: a Run button, a status line, and a first run when the panel is first shown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._has_run = False
        self.run_btn = QtWidgets.QPushButton("Run simulation")
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #d08000;")
        self.run_btn.clicked.connect(self.run)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._has_run:
            self.run()

    def run(self):
        self._has_run = True
        self.status.setText("")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._simulate()
        except ValueError as exc:
            self.status.setText(str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _simulate(self):
        raise NotImplementedError


class TrackingPanel(_SimPanel):
    """Two-point dip tracking under field drift and photon shot noise."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Dip and photon defaults are estimates (placeholders). Use values fitted from your own spectra.
        self.fwhm = spin(5.0, 0.1, 200.0, 2, 0.5, "MHz")
        self.contrast = spin(0.03, 0.0001, 0.9, 4, 0.005)
        self.rate = spin(1.0, 0.001, 10000.0, 3, 0.1, "Mcounts/s")
        self.side = QtWidgets.QComboBox()
        self.side.addItems(["lower", "upper"])
        self.b0 = spin(1.5, 0.0, 100.0, 3, 0.1, "mT")

        self.step = spin(10.0, 0.01, 10000.0, 2, 1.0, "ms")
        self.ki = spin(0.2, 0.01, 1.99, 2, 0.05)
        self.kp = spin(0.0, -2.0, 2.0, 2, 0.05)

        self.ramp = spin(0.5, -1000.0, 1000.0, 3, 0.1, "uT/s")
        self.walk = spin(50.0, 0.0, 100000.0, 1, 10.0, "nT/sqrt(s)")
        self.offset = spin(0.0, -1000.0, 1000.0, 2, 1.0, "uT")

        self.duration = spin(20.0, 0.1, 3600.0, 1, 1.0, "s")
        self.seed = int_spin(0, 0, 999999)
        self.shot_noise = QtWidgets.QCheckBox("Photon shot noise")
        self.shot_noise.setChecked(True)

        dip = QGroupBox("Dip and photons (estimates: fit your own spectrum)")
        form = QFormLayout(dip)
        form.addRow("Dip width (FWHM)", self.fwhm)
        form.addRow("Contrast (dip depth)", self.contrast)
        form.addRow("Photon rate off resonance", self.rate)
        form.addRow("Dip to follow", self.side)
        form.addRow("Starting field along NV", self.b0)

        ctrl = QGroupBox("Loop")
        form = QFormLayout(ctrl)
        form.addRow("Loop period", self.step)
        form.addRow("Integrator gain ki", self.ki)
        form.addRow("Proportional gain kp", self.kp)

        drift = QGroupBox("Field drift (model input)")
        form = QFormLayout(drift)
        form.addRow("Ramp", self.ramp)
        form.addRow("Random walk", self.walk)
        form.addRow("Starting error", self.offset)

        sim = QGroupBox("Run")
        form = QFormLayout(sim)
        form.addRow("Duration", self.duration)
        form.addRow("Random seed", self.seed)
        form.addRow(self.shot_noise)
        form.addRow(self.run_btn)
        form.addRow(self.status)

        self.out = Readout("Results (model, not a measurement)", [
            ("sens", "Shot-noise sensitivity"),
            ("limit", "Textbook Lorentzian limit"),
            ("step_noise", "Offset noise per step"),
            ("rms_expected", "Expected rms error (noise only, kp = 0)"),
            ("lag", "Expected ramp lag"),
            ("rms", "Simulated rms error"),
            ("mean", "Simulated mean error"),
        ])

        left = QtWidgets.QWidget()
        col = QVBoxLayout(left)
        for box in (dip, ctrl, drift, sim, self.out):
            col.addWidget(box)
        col.addWidget(note(
            "Single NV axis with the field along it. The zero-field splitting is held constant, "
            "so a temperature shift would look like a field change. Laser noise is not included."
        ))
        col.addStretch(1)

        self.field_plot = pg.PlotWidget()
        self.field_plot.setLabel("bottom", "Time (s)")
        self.field_plot.setLabel("left", "Field along NV axis (mT)")
        self.field_plot.showGrid(x=True, y=True, alpha=0.3)
        self.field_plot.addLegend(offset=(10, 10))
        self.est_curve = self.field_plot.plot(pen=None, symbol="o", symbolSize=2, symbolBrush="c",
                                              symbolPen=None, name="tracker estimate")
        self.true_curve = self.field_plot.plot(pen=pg.mkPen("y", width=2), name="true field")

        self.err_plot = pg.PlotWidget()
        self.err_plot.setLabel("bottom", "Time (s)")
        self.err_plot.setLabel("left", "Microwave minus dip center (kHz)")
        self.err_plot.showGrid(x=True, y=True, alpha=0.3)
        self.err_curve = self.err_plot.plot(pen=pg.mkPen("m", width=1))
        self.err_plot.setXLink(self.field_plot)

        plots = QtWidgets.QWidget()
        plot_col = QVBoxLayout(plots)
        plot_col.setContentsMargins(0, 0, 0, 0)
        plot_col.addWidget(self.field_plot)
        plot_col.addWidget(self.err_plot)

        QVBoxLayout(self).addWidget(side_by_side(left, plots, left_width=360))

    def _simulate(self):
        step_s = self.step.value() * 1e-3
        duration_s = self.duration.value()
        n = int(round(duration_s / step_s))
        if n < 10:
            raise ValueError("Duration must cover at least 10 loop periods.")
        if n > MAX_TRACK_STEPS:
            raise ValueError(f"{n} steps is too many (limit {MAX_TRACK_STEPS}). "
                             "Shorten the duration or lengthen the loop period.")

        cfg = tracking.TrackerConfig(
            fwhm_hz=self.fwhm.value() * 1e6,
            contrast=self.contrast.value(),
            photon_rate_hz=self.rate.value() * 1e6,
            step_s=step_s,
            ki=self.ki.value(),
            kp=self.kp.value(),
            side=self.side.currentText(),
        )
        drift = tracking.Drift(
            ramp_t_per_s=self.ramp.value() * 1e-6,
            walk_t_per_rthz=self.walk.value() * 1e-9,
            offset_t=self.offset.value() * 1e-6,
        )
        res = tracking.simulate(cfg, drift, b0_t=self.b0.value() * 1e-3, duration_s=duration_s,
                                seed=self.seed.value(), shot_noise=self.shot_noise.isChecked())

        # Skip the pull-in at the start: 2 s, or a fifth of the run if it is short.
        skip = min(int(2.0 / step_s), n // 5)
        limit = tracking.lorentzian_limit_t_per_rthz(cfg.fwhm_hz, cfg.contrast, cfg.photon_rate_hz)
        self.out.set("sens", f"{tracking.sensitivity_t_per_rthz(cfg) * 1e9:.1f} nT/sqrt(Hz)")
        self.out.set("limit", f"{limit * 1e9:.1f} nT/sqrt(Hz)")
        self.out.set("step_noise", f"{tracking.step_noise_hz(cfg) / 1e3:.1f} kHz")
        self.out.set("rms_expected", f"{tracking.expected_steady_state_rms_hz(cfg) / 1e3:.1f} kHz")
        self.out.set("lag", f"{tracking.expected_ramp_lag_hz(cfg, drift.ramp_t_per_s) / 1e3:+.1f} kHz")
        self.out.set("rms", f"{res.rms_error_hz(skip) / 1e3:.1f} kHz = {res.rms_error_t(skip) * 1e6:.2f} uT "
                            f"(after the first {skip * step_s:.1f} s)")
        self.out.set("mean", f"{res.mean_error_hz(skip) / 1e3:+.1f} kHz")

        self.true_curve.setData(res.t_s, res.b_true_t * 1e3)
        self.est_curve.setData(res.t_s, res.b_est_t * 1e3)
        self.err_curve.setData(res.t_s, res.error_hz / 1e3)


class LockinPanel(_SimPanel):
    """Lock-in resonance tracking: frequency dither, demodulation, PI controller."""

    def __init__(self, parent=None):
        super().__init__(parent)

        defaults = lockin.LockinConfig()
        shape = lockin.Lineshape()

        self.center = spin(shape.center_hz / 1e9, 0.1, 20.0, 5, 0.001, "GHz")
        self.fwhm = spin(shape.fwhm_hz / 1e6, 0.1, 200.0, 2, 0.5, "MHz")
        self.depth = spin(shape.depth, 0.0001, 0.9, 4, 0.005)

        self.dither_amp = spin(defaults.dither_amp_hz / 1e6, 0.001, 100.0, 3, 0.1, "MHz")
        self.dither_freq = spin(defaults.dither_freq_hz / 1e3, 0.001, 1000.0, 3, 0.1, "kHz")
        self.cycles = int_spin(defaults.cycles_per_step, 1, 10000)
        self.samples = int_spin(defaults.samples_per_cycle, 4, 1000)
        self.noise = spin(defaults.noise, 0.0, 1.0, 5, 0.0005)

        self.kp = spin(defaults.kp, 0.0, 5.0, 3, 0.05)
        self.ki = spin(defaults.ki, 0.0, 10000.0, 2, 1.0, "1/s")
        self.limit = spin(defaults.limit_hz / 1e6, 0.1, 1000.0, 1, 5.0, "MHz")

        self.drift = spin(100.0, -100000.0, 100000.0, 1, 10.0, "kHz/s")
        self.offset = spin(3.0, -200.0, 200.0, 2, 0.5, "MHz")
        self.duration = spin(10.0, 0.1, 3600.0, 1, 1.0, "s")
        self.seed = int_spin(0, 0, 999999)

        dip = QGroupBox("Dip (use a fitted width and depth)")
        form = QFormLayout(dip)
        form.addRow("Dip center at start", self.center)
        form.addRow("Dip width (FWHM)", self.fwhm)
        form.addRow("Dip depth", self.depth)

        mod = QGroupBox("Dither and readout (assumptions)")
        form = QFormLayout(mod)
        form.addRow("Dither amplitude", self.dither_amp)
        form.addRow("Dither frequency", self.dither_freq)
        form.addRow("Dither periods per update", self.cycles)
        form.addRow("Samples per period", self.samples)
        form.addRow("Noise per sample (contrast)", self.noise)

        ctrl = QGroupBox("Controller")
        form = QFormLayout(ctrl)
        form.addRow("Proportional gain kp", self.kp)
        form.addRow("Integral gain ki", self.ki)
        form.addRow("Output clamp", self.limit)

        sim = QGroupBox("Run")
        form = QFormLayout(sim)
        form.addRow("Dip drift rate", self.drift)
        form.addRow("Carrier start offset", self.offset)
        form.addRow("Duration", self.duration)
        form.addRow("Random seed", self.seed)
        form.addRow(self.run_btn)
        form.addRow(self.status)

        self.out = Readout("Results (model, not a measurement)", [
            ("step", "Controller update period"),
            ("mean", "Mean error, last half"),
            ("rms", "Rms error, last half"),
            ("locked", "Lock"),
        ])

        left = QtWidgets.QWidget()
        col = QVBoxLayout(left)
        for box in (dip, mod, ctrl, sim, self.out):
            col.addWidget(box)
        col.addWidget(note(
            "Pull-in works only within about half a dip width, because the error signal fades "
            "far from the dip. Keep the dither amplitude well below the dip width."
        ))
        col.addStretch(1)

        self.track_plot = pg.PlotWidget()
        self.track_plot.setLabel("bottom", "Time (s)")
        self.track_plot.setLabel("left", "Offset from starting dip center (MHz)")
        self.track_plot.showGrid(x=True, y=True, alpha=0.3)
        self.track_plot.addLegend(offset=(10, 10))
        self.dip_curve = self.track_plot.plot(pen=pg.mkPen("y", width=2), name="dip (true)")
        self.carrier_curve = self.track_plot.plot(pen=pg.mkPen("c", width=1), name="carrier (lock)")

        self.err_plot = pg.PlotWidget()
        self.err_plot.setLabel("bottom", "Time (s)")
        self.err_plot.setLabel("left", "Carrier minus dip (kHz)")
        self.err_plot.showGrid(x=True, y=True, alpha=0.3)
        self.err_curve = self.err_plot.plot(pen=pg.mkPen("m", width=1))
        self.err_plot.setXLink(self.track_plot)

        plots = QtWidgets.QWidget()
        plot_col = QVBoxLayout(plots)
        plot_col.setContentsMargins(0, 0, 0, 0)
        plot_col.addWidget(self.track_plot)
        plot_col.addWidget(self.err_plot)

        QVBoxLayout(self).addWidget(side_by_side(left, plots, left_width=360))

    def _simulate(self):
        cfg = lockin.LockinConfig(
            dither_amp_hz=self.dither_amp.value() * 1e6,
            dither_freq_hz=self.dither_freq.value() * 1e3,
            cycles_per_step=self.cycles.value(),
            samples_per_cycle=self.samples.value(),
            noise=self.noise.value(),
            kp=self.kp.value(),
            ki=self.ki.value(),
            limit_hz=self.limit.value() * 1e6,
        )
        duration_s = self.duration.value()
        n_steps = int(round(duration_s / cfg.step_s))
        n_samples = n_steps * cfg.cycles_per_step * cfg.samples_per_cycle
        if n_steps < 10:
            raise ValueError("Duration must cover at least 10 controller updates.")
        if n_samples > MAX_LOCKIN_SAMPLES:
            raise ValueError(f"{n_samples:.2e} samples is too many (limit {MAX_LOCKIN_SAMPLES:.0e}). "
                             "Shorten the duration or lower the dither frequency or samples per period.")

        shape = lockin.Lineshape(center_hz=self.center.value() * 1e9, fwhm_hz=self.fwhm.value() * 1e6,
                                 depth=self.depth.value())
        res = lockin.simulate(shape, cfg, duration_s, lockin.ramp(shape.center_hz, self.drift.value() * 1e3),
                              self.offset.value() * 1e6, self.seed.value())

        rms = res.rms_error_hz()
        self.out.set("step", f"{cfg.step_s * 1e3:.2f} ms")
        self.out.set("mean", f"{res.mean_error_hz() / 1e3:+.1f} kHz")
        self.out.set("rms", f"{rms / 1e3:.1f} kHz")
        # Outside half a dip width the error signal has faded: call that a lost lock.
        held = rms < shape.fwhm_hz / 2
        self.out.set("locked", "held" if held else "LOST: error is larger than half the dip width")

        self.dip_curve.setData(res.t, (res.dip_hz - shape.center_hz) / 1e6)
        self.carrier_curve.setData(res.t, (res.fc_hz - shape.center_hz) / 1e6)
        self.err_curve.setData(res.t, res.error_hz / 1e3)
        if held and rms > 0:
            # Zoom on steady state; the start-up pull-in is off scale.
            self.err_plot.setYRange(-6 * rms / 1e3, 6 * rms / 1e3)
        else:
            self.err_plot.enableAutoRange(axis="y")


class ControlsTab(QtWidgets.QWidget):
    """The 'Controls' tab of the main window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        if _IMPORT_ERROR is not None:
            layout.addWidget(missing_backend(_IMPORT_ERROR))
            return
        layout.addWidget(note(
            "Simulation only. These loops do not drive the microwave source. "
            "Every result is a model value for the inputs shown."
        ))
        self.tracking = TrackingPanel()
        self.lockin = LockinPanel()
        self.pages = QtWidgets.QTabWidget()
        self.pages.addTab(self.tracking, "Dip tracking")
        self.pages.addTab(self.lockin, "Lock-in")
        layout.addWidget(self.pages, 1)

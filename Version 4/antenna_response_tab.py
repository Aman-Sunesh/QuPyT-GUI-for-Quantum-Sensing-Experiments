# antenna_response_tab.py

# ----------------------------------------------------------------
# Antenna response tab (Version 4): read saved EduControl Rabi runs,
# fit them with qzabre.pulsed_io, and compare Rabi frequency against
# the microwave frequency that was actually driven.
#
# This tab only reads saved files. It never talks to lab hardware.
# ----------------------------------------------------------------

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout

from tab_helpers import missing_backend, note

try:
    from qzabre import pulsed_io
    _IMPORT_ERROR = None
except Exception as exc:  # the tab still opens and explains what is missing
    pulsed_io = None
    _IMPORT_ERROR = exc


EXCLUDE_RELATIVE_UNCERTAINTY = 0.10
WARN_RELATIVE_UNCERTAINTY = 0.05
EXCLUDE_BELOW_PERIODS = 1.0
WARN_BELOW_PERIODS = 2.0
EDUCONTROL_DIFFERENCE_WARNING = 0.10
MIN_PLAUSIBLE_DRIVE_HZ = 1e8
MAX_PLAUSIBLE_DRIVE_HZ = 100e9
MIN_GHZ_SCALE_FREQUENCY = 1.0
MAX_GHZ_SCALE_FREQUENCY = 10.0
UNKNOWN = "unknown"

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SETTING_RE = re.compile(rf"^\s*\$(freq|df|mwAmp)\s*=\s*({_NUMBER})\s*$")


@dataclass(frozen=True)
class MicrowaveSettings:
    source: Path | None = None
    frequency_hz: float | None = None
    detuning_hz: float | None = None
    amplitude: float | None = None
    frequency_warning: str | None = None
    frequency_conversion_warning: str | None = None

    @property
    def driven_frequency_hz(self):
        if self.frequency_warning or self.frequency_hz is None or self.detuning_hz is None:
            return None
        return self.frequency_hz + self.detuning_hz


@dataclass
class AntennaRunResult:
    path: Path
    settings: MicrowaveSettings
    rabi_hz: float | None = None
    rabi_err_hz: float | None = None
    pi_s: float | None = None
    r_squared: float | None = None
    periods_in_window: float | None = None
    educontrol_rabi_hz: float | None = None
    failure_reasons: list[str] = field(default_factory=list)
    exclusion_reasons: list[str] = field(default_factory=list)
    warning_reasons: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def fit_failed(self):
        return bool(self.error or self.failure_reasons)

    @property
    def excluded(self):
        return bool(self.fit_failed or self.exclusion_reasons)

    @property
    def classification(self):
        if self.fit_failed:
            return "failed"
        if self.exclusion_reasons:
            return "excluded"
        if self.warning_reasons:
            return "warning"
        return "accepted"

    @property
    def run_date(self):
        match = re.match(r"^(\d{4}-\d{2}-\d{2})", self.path.name)
        return match.group(1) if match else "unknown date"

    @property
    def can_plot(self):
        return (
            self.rabi_hz is not None
            and self.rabi_err_hz is not None
            and self.settings.driven_frequency_hz is not None
            and self.settings.amplitude is not None
        )


def find_rabi_runs(folder):
    """Return run folders containing seq0/eval/rabi.h5, including *folder* itself."""
    root = Path(folder)
    runs = set()
    direct = root / "seq0" / "eval" / "rabi.h5"
    if direct.is_file():
        runs.add(root)
    if root.is_dir():
        for hit in root.rglob("rabi.h5"):
            if hit.parent.name == "eval" and hit.parent.parent.name.startswith("seq"):
                runs.add(hit.parent.parent.parent)
    return sorted(runs, key=lambda path: str(path).lower())


def read_microwave_settings(run_folder):
    """Read the executed $freq, $df and $mwAmp values from the saved rabi.seq."""
    run = Path(run_folder)
    exact = run / "seq0" / "src" / "rabi.seq"
    candidates = [exact] if exact.is_file() else sorted(run.rglob("rabi.seq"))
    if not candidates:
        return MicrowaveSettings(frequency_warning="no saved rabi.seq file was found")

    source = candidates[0]
    values = {}
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = _SETTING_RE.match(line)
            if match and match.group(1) not in values:
                values[match.group(1)] = float(match.group(2))
    saved_frequency = values.get("freq")
    detuning = values.get("df")
    warning = None
    conversion_warning = None
    frequency = saved_frequency
    if saved_frequency is None:
        warning = "$freq is missing from rabi.seq"
    elif detuning is None:
        warning = "$df is missing from rabi.seq"
    elif MIN_GHZ_SCALE_FREQUENCY <= saved_frequency <= MAX_GHZ_SCALE_FREQUENCY:
        if detuning == 0.0:
            frequency = saved_frequency * 1e9
            conversion_warning = (
                f"converted saved $freq from GHz to Hz: {saved_frequency:.12g} GHz "
                f"= {frequency:.0f} Hz ($df = 0)"
            )
        else:
            warning = (
                f"saved $freq is {saved_frequency:g}, which is in the GHz-scale range, "
                f"but $df is nonzero ({detuning:g}); GHz-to-Hz conversion was not applied"
            )
    elif not MIN_PLAUSIBLE_DRIVE_HZ <= saved_frequency <= MAX_PLAUSIBLE_DRIVE_HZ:
        warning = (
            f"saved $freq is {saved_frequency:g}; expected either 1 to 10 GHz or "
            f"{MIN_PLAUSIBLE_DRIVE_HZ:g} to {MAX_PLAUSIBLE_DRIVE_HZ:g} Hz"
        )
    else:
        driven = saved_frequency + detuning
        if not MIN_PLAUSIBLE_DRIVE_HZ <= driven <= MAX_PLAUSIBLE_DRIVE_HZ:
            warning = (
                f"saved $freq + $df is {driven:g}; expected hertz between "
                f"{MIN_PLAUSIBLE_DRIVE_HZ:g} and {MAX_PLAUSIBLE_DRIVE_HZ:g}"
            )
    return MicrowaveSettings(
        source=source,
        frequency_hz=frequency,
        detuning_hz=detuning,
        amplitude=values.get("mwAmp"),
        frequency_warning=warning,
        frequency_conversion_warning=conversion_warning,
    )


def r_squared(y, model):
    y = np.asarray(y, dtype=float)
    model = np.asarray(model, dtype=float)
    residual = float(np.sum((y - model) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return 1.0 - residual / total


def fit_failure_reasons(f_hz, f_err_hz):
    reasons = []
    if not np.isfinite(f_hz) or f_hz <= 0:
        reasons.append("Rabi frequency is nonpositive or nonfinite")
    if not np.isfinite(f_err_hz) or f_err_hz <= 0:
        reasons.append("uncertainty is zero or nonfinite")
    return reasons


def fit_classification_reasons(f_hz, f_err_hz, periods_in_window):
    """Return (exclusions, warnings) for a numerically valid fit."""
    exclusions = []
    warnings = []
    relative = f_err_hz / f_hz
    if relative > EXCLUDE_RELATIVE_UNCERTAINTY:
        exclusions.append("relative uncertainty above 10%")
    elif relative > WARN_RELATIVE_UNCERTAINTY:
        warnings.append("relative uncertainty from 5% to 10%")
    if periods_in_window < EXCLUDE_BELOW_PERIODS:
        exclusions.append("fewer than 1 fitted period in the window")
    elif periods_in_window < WARN_BELOW_PERIODS:
        warnings.append("only 1 to 2 fitted periods in the window")
    return exclusions, warnings


def educontrol_difference_warning(f_hz, educontrol_hz):
    if educontrol_hz is None or not np.isfinite(educontrol_hz) or educontrol_hz == 0:
        return None
    difference = abs(f_hz - educontrol_hz) / abs(educontrol_hz)
    if difference > EDUCONTROL_DIFFERENCE_WARNING:
        return f"fit differs from EduControl by {100 * difference:.1f}%"
    return None


def analyze_run_folder(run_folder):
    """Load and fit one saved Rabi run without making any hardware calls."""
    path = Path(run_folder)
    settings = read_microwave_settings(path)
    result = AntennaRunResult(path=path, settings=settings)
    try:
        run = pulsed_io.load_run(path, module="rabi")
        educontrol = run.scalars.get("freq0")
        if educontrol is not None and np.isfinite(educontrol):
            result.educontrol_rabi_hz = float(educontrol)
        fit = pulsed_io.analyze(run)
        result.rabi_hz = float(fit["f_hz"])
        result.rabi_err_hz = float(fit["f_err_hz"])
        result.pi_s = float(fit["pi_s"])
        result.r_squared = r_squared(fit["y"], fit["model"])
        result.periods_in_window = float(np.ptp(fit["t_s"]) * result.rabi_hz)
        result.failure_reasons = fit_failure_reasons(result.rabi_hz, result.rabi_err_hz)
        if not result.failure_reasons:
            result.exclusion_reasons, result.warning_reasons = fit_classification_reasons(
                result.rabi_hz, result.rabi_err_hz, result.periods_in_window
            )
            comparison = educontrol_difference_warning(
                result.rabi_hz, result.educontrol_rabi_hz
            )
            if comparison:
                result.warning_reasons.append(comparison)
        if result.settings.frequency_warning:
            result.warning_reasons.append(
                f"driven frequency unavailable: {result.settings.frequency_warning}"
            )
        if result.settings.frequency_conversion_warning:
            result.warning_reasons.append(result.settings.frequency_conversion_warning)
    except Exception as exc:
        result.error = str(exc)
    return result


def _number(value, scale=1.0, decimals=4):
    if value is None or not math.isfinite(value):
        return UNKNOWN
    return f"{value * scale:.{decimals}f}"


def _quality(result):
    if result.error:
        return f"FAILED: {result.error}"
    if result.rabi_hz is None or result.rabi_err_hz is None:
        return "FAILED: fit result unavailable"
    rel = 100.0 * result.rabi_err_hz / result.rabi_hz if result.rabi_hz else float("inf")
    periods = _number(result.periods_in_window, decimals=2)
    summary = f"R² {result.r_squared:.3f}, σf/f {rel:.2f}%, {periods} periods"
    reasons = result.failure_reasons + result.exclusion_reasons + result.warning_reasons
    prefix = result.classification.upper()
    if reasons:
        return f"{prefix}: {'; '.join(reasons)} ({summary})"
    return f"{prefix}: {summary}"


class AntennaResponseTab(QtWidgets.QWidget):
    """Read-only analysis of saved EduControl Rabi run folders."""

    HEADERS = [
        "Plot",
        "Run name",
        "Driven MW frequency (GHz)",
        "MW amplitude",
        "Rabi frequency (MHz)",
        "EduControl Rabi (MHz)",
        "Pi time (ns)",
        "Fit quality",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = []
        self._row_results = []
        layout = QVBoxLayout(self)
        if _IMPORT_ERROR is not None:
            layout.addWidget(missing_backend(_IMPORT_ERROR))
            return

        layout.addWidget(note(
            "Reads saved EduControl files only; no hardware is contacted. Driven frequency is "
            "$freq + $df from seq0/src/rabi.seq. Fits use qzabre.pulsed_io."
        ))

        buttons = QHBoxLayout()
        self.add_run_btn = QtWidgets.QPushButton("Add run folder…")
        self.scan_parent_btn = QtWidgets.QPushButton("Scan parent folder…")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.export_btn = QtWidgets.QPushButton("Export table to CSV…")
        self.export_btn.setEnabled(False)
        buttons.addWidget(self.add_run_btn)
        buttons.addWidget(self.scan_parent_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.export_btn)
        layout.addLayout(buttons)

        self.status = QtWidgets.QLabel("Choose one Rabi run or scan a parent folder.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #d08000;")
        layout.addWidget(self.status)

        self.table = QtWidgets.QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.frequency_plot = pg.PlotWidget()
        self.frequency_plot.setLabel("bottom", "Driven microwave frequency (GHz)")
        self.frequency_plot.setLabel("left", "Rabi frequency (MHz)")
        self.frequency_plot.showGrid(x=True, y=True, alpha=0.3)
        self.frequency_plot.setLimits(yMin=0)
        self.frequency_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.frequency_legend = self.frequency_plot.addLegend(offset=(10, 10))

        self.amplitude_plot = pg.PlotWidget()
        self.amplitude_plot.setLabel("bottom", "MW amplitude")
        self.amplitude_plot.setLabel("left", "Rabi frequency (MHz)")
        self.amplitude_plot.showGrid(x=True, y=True, alpha=0.3)
        self.amplitude_plot.setLimits(yMin=0)
        self.amplitude_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.amplitude_legend = self.amplitude_plot.addLegend(offset=(10, 10))

        self.plot_pages = QtWidgets.QTabWidget()
        self.plot_pages.addTab(
            self._plot_page(self.amplitude_plot, self.amplitude_legend), "Amplitude response"
        )
        self.plot_pages.addTab(
            self._plot_page(self.frequency_plot, self.frequency_legend), "Frequency response"
        )
        # Keep the old attribute as an alias for code that used the original single plot.
        self.plot = self.frequency_plot

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.plot_pages)
        splitter.setSizes([850, 500])
        layout.addWidget(splitter, 1)

        self.add_run_btn.clicked.connect(self._choose_run)
        self.scan_parent_btn.clicked.connect(self._choose_parent)
        self.clear_btn.clicked.connect(self.clear)
        self.export_btn.clicked.connect(self._export_csv)
        self.table.itemChanged.connect(self._table_item_changed)

    def _choose_run(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a Rabi run folder")
        if folder:
            self.add_runs([Path(folder)])

    def _choose_parent(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder containing Rabi runs")
        if not folder:
            return
        runs = find_rabi_runs(folder)
        if not runs:
            self.status.setText(f"No seq0/eval/rabi.h5 files found under {folder}.")
            return
        self.add_runs(runs)

    def add_runs(self, paths):
        existing = {result.path.resolve() for result in self.results}
        new_paths = [Path(path) for path in paths if Path(path).resolve() not in existing]
        if not new_paths:
            self.status.setText("Those run folders are already in the table.")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for path in new_paths:
                self.results.append(analyze_run_folder(path))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._populate_table()

    def clear(self):
        self.results.clear()
        self._row_results.clear()
        self.table.setRowCount(0)
        self.frequency_plot.clear()
        self.amplitude_plot.clear()
        self.export_btn.setEnabled(False)
        self.status.setText("Choose one Rabi run or scan a parent folder.")

    def _populate_table(self):
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.results))
            self._row_results = list(self.results)
            for row, result in enumerate(self._row_results):
                include = QtWidgets.QTableWidgetItem()
                include.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                )
                include.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if result.can_plot
                    else QtCore.Qt.CheckState.Unchecked
                )
                values = [
                    include,
                    QtWidgets.QTableWidgetItem(result.path.name),
                    QtWidgets.QTableWidgetItem(_number(result.settings.driven_frequency_hz, 1e-9, 6)),
                    QtWidgets.QTableWidgetItem(_number(result.settings.amplitude, 1.0, 6)),
                    QtWidgets.QTableWidgetItem(
                        UNKNOWN if result.rabi_hz is None else
                        f"{result.rabi_hz / 1e6:.4f} ± {result.rabi_err_hz / 1e6:.4f}"
                    ),
                    QtWidgets.QTableWidgetItem(_number(result.educontrol_rabi_hz, 1e-6, 4)),
                    QtWidgets.QTableWidgetItem(_number(result.pi_s, 1e9, 2)),
                    QtWidgets.QTableWidgetItem(_quality(result)),
                ]
                tooltip = f"Run: {result.path.name}\nPath: {result.path}\nFit quality: {_quality(result)}"
                if result.settings.source:
                    tooltip += f"\nSettings: {result.settings.source}"
                if result.excluded:
                    tooltip += "\nShown in gray on plots."
                for column, item in enumerate(values):
                    item.setToolTip(tooltip)
                    self.table.setItem(row, column, item)
                    if result.fit_failed:
                        item.setBackground(QtGui.QColor(90, 35, 35))
                    elif result.excluded:
                        item.setBackground(QtGui.QColor(80, 80, 80))
                    elif result.warning_reasons:
                        item.setBackground(QtGui.QColor(90, 70, 25))
        finally:
            self.table.blockSignals(False)
        self.export_btn.setEnabled(bool(self.results))
        self._refresh_plot_and_status()

    def _table_item_changed(self, item):
        if item.column() == 0:
            self._refresh_plot_and_status()

    def _refresh_plot_and_status(self):
        self.frequency_plot.clear()
        self.amplitude_plot.clear()
        selected = []
        for row, result in enumerate(self._row_results):
            item = self.table.item(row, 0)
            if item is None or item.checkState() != QtCore.Qt.CheckState.Checked or not result.can_plot:
                continue
            selected.append(result)

        accepted = [result for result in selected if not result.excluded]
        excluded = [result for result in selected if result.excluded]

        # Amplitude-response lines only connect runs from the same date and driven frequency.
        amplitude_groups = {}
        for result in accepted:
            key = (result.run_date, result.settings.driven_frequency_hz)
            amplitude_groups.setdefault(key, []).append(result)
        for index, ((date, frequency), results) in enumerate(sorted(amplitude_groups.items())):
            ordered = sorted(results, key=lambda result: result.settings.amplitude)
            x = np.array([result.settings.amplitude for result in ordered])
            y = np.array([result.rabi_hz / 1e6 for result in ordered])
            err = np.array([result.rabi_err_hz / 1e6 for result in ordered])
            color = pg.intColor(index, hues=max(len(amplitude_groups), 1), values=1, maxValue=255)
            label = f"{date}, {frequency / 1e9:.6f} GHz"
            self._plot_group(self.amplitude_plot, x, y, err, color, label, beam=0.005)

        # Frequency response preserves the original view. Lines only connect equal-amplitude
        # runs acquired on the same date, so measurements from separate sweeps are not merged.
        frequency_groups = {}
        for result in accepted:
            key = (result.run_date, result.settings.amplitude)
            frequency_groups.setdefault(key, []).append(result)
        for index, ((date, amplitude), results) in enumerate(sorted(frequency_groups.items())):
            ordered = sorted(results, key=lambda result: result.settings.driven_frequency_hz)
            x = np.array([result.settings.driven_frequency_hz / 1e9 for result in ordered])
            y = np.array([result.rabi_hz / 1e6 for result in ordered])
            err = np.array([result.rabi_err_hz / 1e6 for result in ordered])
            color = pg.intColor(index, hues=max(len(frequency_groups), 1), values=1, maxValue=255)
            label = f"{date}, MW amplitude {amplitude:g}"
            self._plot_group(self.frequency_plot, x, y, err, color, label, beam=0.002)

        self._plot_excluded(self.amplitude_plot, excluded, x_value=lambda result: result.settings.amplitude,
                            beam=0.005)
        self._plot_excluded(self.frequency_plot, excluded,
                            x_value=lambda result: result.settings.driven_frequency_hz / 1e9,
                            beam=0.002)

        self._set_plot_range(
            self.amplitude_plot,
            accepted,
            x_value=lambda result: result.settings.amplitude,
            minimum_x_padding=0.01,
            zero_x=True,
        )
        self._set_plot_range(
            self.frequency_plot,
            accepted,
            x_value=lambda result: result.settings.driven_frequency_hz / 1e9,
            minimum_x_padding=0.005,
        )

        known_amplitudes = sorted({
            result.settings.amplitude for result in self.results
            if result.settings.amplitude is not None
        })
        failed = sum(result.fit_failed for result in self.results)
        excluded_count = sum(result.excluded and not result.fit_failed for result in self.results)
        warning_count = sum(result.classification == "warning" for result in self.results)
        comparison_warnings = sum(
            any("EduControl" in reason for reason in result.warning_reasons)
            for result in self.results
        )
        missing_frequency = sum(result.settings.driven_frequency_hz is None for result in self.results)
        missing_amplitude = sum(result.settings.amplitude is None for result in self.results)
        messages = [f"Loaded {len(self.results)} run(s)."]
        if len(known_amplitudes) > 1:
            values = ", ".join(f"{value:g}" for value in known_amplitudes)
            messages.append(f"Warning: MW amplitudes differ ({values}); each amplitude is a separate series.")
        if failed:
            messages.append(f"{failed} fit(s) failed.")
        if excluded_count:
            messages.append(f"{excluded_count} suspect fit(s) are shown in gray on the plots.")
        if warning_count:
            messages.append(f"{warning_count} fit(s) have limited-window or comparison warnings.")
        if comparison_warnings:
            messages.append(f"{comparison_warnings} fit(s) differ from EduControl by more than 10%.")
        if missing_frequency:
            messages.append(
                f"{missing_frequency} run(s) have unknown driven frequency and are not plotted."
            )
        if missing_amplitude:
            messages.append(f"{missing_amplitude} run(s) have unknown MW amplitude.")
        self.status.setText(" ".join(messages))

    @staticmethod
    def _plot_group(plot, x, y, err, color, label, beam):
        plot.plot(
            x, y, pen=pg.mkPen(color, width=2), symbol="o", symbolSize=8,
            symbolBrush=color, symbolPen=color, name=label
        )
        plot.addItem(pg.ErrorBarItem(
            x=x, y=y, top=err, bottom=err, beam=beam, pen=pg.mkPen(color)
        ))

    @staticmethod
    def _plot_excluded(plot, results, x_value, beam):
        if not results:
            return
        gray = QtGui.QColor(145, 145, 145, 180)
        x = np.array([x_value(result) for result in results])
        y = np.array([result.rabi_hz / 1e6 for result in results])
        err = np.array([result.rabi_err_hz / 1e6 for result in results])
        points = pg.PlotDataItem(
            x=x, y=y, pen=None, symbol="o", symbolSize=8, symbolBrush=gray,
            symbolPen=pg.mkPen(gray), name="excluded"
        )
        plot.addItem(points, ignoreBounds=True)
        error_bars = pg.ErrorBarItem(
            x=x, y=y, top=err, bottom=err, beam=beam, pen=pg.mkPen(gray)
        )
        plot.addItem(error_bars, ignoreBounds=True)

    @staticmethod
    def _set_plot_range(plot, results, x_value, minimum_x_padding, zero_x=False):
        """Range from accepted/warning runs only; excluded and failed items stay visible if in range."""
        if not results:
            plot.setXRange(0, 1, padding=0)
            plot.setYRange(0, 1, padding=0)
            return
        x = np.array([x_value(result) for result in results], dtype=float)
        y = np.array([result.rabi_hz / 1e6 for result in results], dtype=float)
        err = np.array([result.rabi_err_hz / 1e6 for result in results], dtype=float)
        x_span = float(np.ptp(x))
        x_padding = max(0.05 * x_span, minimum_x_padding)
        y_high = float(np.max(y + err))
        y_padding = max(0.05 * y_high, 0.05)
        x_low = 0.0 if zero_x else float(np.min(x)) - x_padding
        plot.setXRange(x_low, float(np.max(x)) + x_padding, padding=0)
        plot.setYRange(0.0, y_high + y_padding, padding=0)

    @staticmethod
    def _plot_page(plot, legend):
        page = QtWidgets.QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        toggle = QtWidgets.QCheckBox("Show legend")
        toggle.setChecked(False)
        legend.setVisible(False)
        toggle.toggled.connect(legend.setVisible)
        layout.addWidget(toggle, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(plot, 1)
        return page

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export antenna response table", "antenna_response.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.HEADERS)
            for row in range(self.table.rowCount()):
                writer.writerow([
                    "yes" if self.table.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked else "no",
                    *[self.table.item(row, column).text() for column in range(1, self.table.columnCount())],
                ])
        self.status.setText(f"Exported {self.table.rowCount()} row(s) to {path}.")

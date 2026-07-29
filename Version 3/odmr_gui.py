import sys
import os
import shutil
import glob
import yaml
import csv
import json
import re
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
import importlib
import logging
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from jinja2 import Template
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_filter
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (QFileDialog, QPlainTextEdit, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QTextEdit, QSplitter, QLabel, QGroupBox, QVBoxLayout, QFormLayout,
                             QSpinBox, QDoubleSpinBox, QHBoxLayout)

from presets import PRESETS
from utils import lorentzian, gaussian
from odmr_yaml import render_experiment_yaml
from channels import CHANNEL_MAPPING
from experiment_factory import load_experiments
from experiment_editor import ExperimentEditor
from generic_generator import generate_from_descriptor
from power_supply import PowerSupplyDialog
from stop_pb import stop_pulse_blaster

# Global exception hook
logging.basicConfig(level=logging.ERROR)

def excepthook(exc_type, exc_value, exc_tb):
    QMessageBox.critical(None, "Unhandled Error", str(exc_value))
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

sys.excepthook = excepthook

NUMBER_PATTERN = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)

from PyQt6.QtWidgets import QComboBox

GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
QUPYT_ROOT = PROJECT_ROOT / "QuPyt-master"
LAST_CFG_PATH = PROJECT_ROOT / ".qupyt" / "last_config.json"

TIMING_PARAMETER_NAMES = {
    "mw_duration",
    "read_time",
    "laserduration",
    "start_pulse_dur",
    "I_pulse",
    "Q_pulse",
    "tau",
    "pi",
    "pi_half",
    "laser_duration",
    "readout_offset",
    "laser_mw_offset",
    "mw_laser_offset",
}

NON_PULSE_SWEEP_PARAMETERS = {
    "freq_start",
    "freq_stop",
    "sweep_start",
    "sweep_stop",
    "power",
    "rf_power",
    "averages",
    "frames",
    "n_dynamic_steps",
    "dynamic_steps",
    "max_rate",
    "max_framerate",
    "mw_device_type",
    "mw_output",
}

class ODMRGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.experiments_dir = GUI_DIR / "experiments"
        self.output_dir = QUPYT_ROOT
        self.experiment_descs = load_experiments(self.experiments_dir)
        self._suppress_auto_switch = False
        self.setWindowTitle('QuPyt Experiment GUI')
        self.process = None
        self.param_widgets = {}
        self._build_ui()

        # Always begin with an empty QuPyt waiting room.
        self._clear_waiting_room()

        self._restore_last_config()

        # populate the Experiments tab list
        self._refresh_experiment_list()

        # for live-plot data
        self.live_freqs = []
        self.live_ch0 = []
        self.live_ch1 = []
        self.live_normalized = []
        self.live_combined = []

        # load all descriptors
        self.experiment_descs = load_experiments(self.experiments_dir)
        self.exp_combo.clear()
        self.exp_combo.addItems(self.experiment_descs.keys())
        self.exp_combo.currentTextChanged.emit(self.exp_combo.currentText())

        # force the “apply preset” step so the Setup tab reflects any edits:
        self.exp_combo.currentTextChanged.emit(self.exp_combo.currentText())

        # now override with your last‐used JSON, if it exists
        self._restore_last_config()  

        self.output_watcher = QtCore.QFileSystemWatcher(
            [str(self.output_dir)],
            self
        )
        self.output_watcher.directoryChanged.connect(
            self._populate_file_selector
        )
        self._populate_file_selector() 
        self.tabs.setCurrentIndex(0)

    
    def _populate_file_selector(self):
        exp_name = self.exp_combo.currentText()
        files = sorted(
            self.output_dir.glob(f"{exp_name}_*.npy"),
            key=lambda path: path.stat().st_mtime
        )
        self.file_selector.clear()
        self.file_selector.addItems([str(path) for path in files])

        if files:
            self.file_selector.setCurrentIndex(len(files) - 1)

    def _on_started(self):
        # called when QProcess starts
        self._suppress_auto_switch = False
        self.file_selector.blockSignals(False)
        self.status_led.setStyleSheet("background-color: green; border-radius: 8px;")
        self.status_label.setText("Running")

    def _on_results_control_changed(self, *_):
        self._update_results_view_visibility()
        self._show_results()

    def _update_results_view_visibility(self):
        if not hasattr(self, "result_view_combo"):
            return

        showing_heatmap = (
            self.result_view_combo.currentText()
            == "2D heatmap"
        )

        self.proc_box.setVisible(
            not showing_heatmap
        )
        self.fit_box.setVisible(
            not showing_heatmap
        )
        self.mean_proc_box.setVisible(
            not showing_heatmap
        )
        self.mean_fit_box.setVisible(
            not showing_heatmap
        )
        self.heatmap_box.setVisible(
            showing_heatmap
        )

        self.result_ps_step.setEnabled(
            not showing_heatmap
        )
        self.heatmap_signal_combo.setEnabled(
            showing_heatmap
        )

    def _load_result_configuration(self):
        candidates = []

        current_file = getattr(
            self,
            "current_file",
            None,
        )

        if current_file is not None:
            candidates.append(
                current_file.with_suffix(".yaml")
            )

        exp_name = self.exp_combo.currentText()

        candidates.extend(
            [
                Path.home()
                / "Desktop"
                / f"{exp_name}.yaml",

                Path.home()
                / ".qupyt"
                / "waiting_room"
                / f"{exp_name}.yaml",
            ]
        )

        for candidate in candidates:
            if not candidate.exists():
                continue

            try:
                with open(
                    candidate,
                    "r",
                    encoding="utf-8",
                ) as file:
                    config = yaml.safe_load(file)

                if isinstance(config, dict):
                    return config, candidate
            except Exception:
                logging.exception(
                    "Could not read result metadata %s",
                    candidate,
                )

        config = {
            "dynamic_steps": self.dynamic_input.value(),
            "pulse_sequence_steps": 1,
            "averages": self.avg_input.value(),
            "result_schema_version": 2,
            "data_layout": "channel_ps_dynamic_frame_roi",
            "sensor": {
                "config": {
                    "number_measurements": (
                        self.frames_input.value()
                    ),
                    "reference_channels": (
                        self.refch_input.value()
                    ),
                }
            },
            "data": {
                "averaging_mode": (
                    self.mode_input.currentText()
                ),
                "reference_channels": (
                    self.refch_input.value()
                ),
                "ps_steps": 1,
            },
            "dynamic_devices": {
                "mw_source": {
                    "config": {
                        "frequency": [
                            self.start_input.value()
                            * 1e9,
                            self.stop_input.value()
                            * 1e9,
                        ]
                    }
                }
            },
            "pulse_sequence": {
                "mw_duration": (
                    self.mw_dur.value()
                    * self.time_factor
                ),
                "readout_time": (
                    self.read_dur.value()
                    * self.time_factor
                ),
            },
        }

        return config, None

    def _canonical_result_data(
        self,
        data,
        config,
    ):
        """
        Return data as:

            [channel, pulse_sequence, dynamic, ...]

        Old files are converted from:

            [channel, dynamic, ...]
        """
        data = np.asarray(data)

        configured_ps_steps = int(
            config.get(
                "pulse_sequence_steps",
                config.get(
                    "data",
                    {},
                ).get("ps_steps", 1),
            )
        )

        dynamic_steps = int(
            config.get(
                "dynamic_steps",
                self.dynamic_input.value(),
            )
        )

        has_ps_metadata = (
            "pulse_sequence_steps" in config
            or "ps_steps" in config.get("data", {})
        )

        if (
            has_ps_metadata
            and data.ndim >= 3
            and data.shape[1] == configured_ps_steps
        ):
            return data

        if (
            data.ndim >= 2
            and data.shape[1] == dynamic_steps
        ):
            return np.expand_dims(
                data,
                axis=1,
            )

        if data.ndim >= 5:
            return data

        return np.expand_dims(
            data,
            axis=1,
        )

    def _scale_result_data_for_analysis(
        self,
        canonical_data,
        config,
    ):
        """
        Convert QuPyt's accumulated data into per-acquisition means.

        spread:
            Each stored frame position is accumulated across averages.

        sum:
            Every frame belonging to a reference channel is summed for
            every average.
        """
        scaled_data = np.asarray(
            canonical_data,
            dtype=float,
        ).copy()

        averages = max(
            1,
            int(config.get("averages", 1)),
        )

        data_config = config.get("data", {})
        averaging_mode = str(
            data_config.get(
                "averaging_mode",
                config.get("averaging_mode", "spread"),
            )
        ).lower()

        if averaging_mode == "spread":
            return scaled_data / averages

        if averaging_mode == "sum":
            sensor_config = (
                config.get("sensor", {})
                .get("config", {})
            )

            number_measurements = max(
                1,
                int(
                    sensor_config.get(
                        "number_measurements",
                        self.frames_input.value(),
                    )
                ),
            )

            reference_channels = max(
                1,
                int(
                    sensor_config.get(
                        "reference_channels",
                        data_config.get(
                            "reference_channels",
                            self.refch_input.value(),
                        ),
                    )
                ),
            )

            if number_measurements % reference_channels != 0:
                raise ValueError(
                    "number_measurements must be divisible by "
                    "reference_channels."
                )

            measurements_per_channel = (
                number_measurements
                // reference_channels
            )

            return scaled_data / (
                averages * measurements_per_channel
            )

        raise ValueError(
            f"Unsupported averaging mode {averaging_mode!r}."
        )
    
    def _frequency_axis_from_config(
        self,
        config,
        dynamic_steps,
    ):
        frequency_config = (
            config.get(
                "dynamic_devices",
                {},
            )
            .get(
                "mw_source",
                {},
            )
            .get(
                "config",
                {},
            )
            .get("frequency")
        )

        if frequency_config is None:
            return np.linspace(
                self.start_input.value() * 1e9,
                self.stop_input.value() * 1e9,
                dynamic_steps,
            )

        try:
            if (
                isinstance(
                    frequency_config,
                    (list, tuple),
                )
                and len(frequency_config) == 2
                and isinstance(
                    frequency_config[1],
                    (list, tuple),
                )
            ):
                start_hz, stop_hz = map(
                    float,
                    frequency_config[1],
                )
            else:
                start_hz, stop_hz = map(
                    float,
                    frequency_config,
                )
        except (
            TypeError,
            ValueError,
            KeyError,
        ):
            start_hz = (
                self.start_input.value() * 1e9
            )
            stop_hz = (
                self.stop_input.value() * 1e9
            )

        return np.linspace(
            start_hz,
            stop_hz,
            dynamic_steps,
        )

    def _pulse_sequence_axis_from_config(
        self,
        config,
        ps_steps,
    ):
        axis_config = config.get(
            "pulse_sequence_axis",
            {},
        )

        pulse_sequence_config = config.get(
            "pulse_sequence",
            {},
        )

        parameter = (
            axis_config.get("parameter")
            or pulse_sequence_config.get(
                "sweep_param"
            )
            or "step"
        )

        unit = str(
            axis_config.get("unit")
            or pulse_sequence_config.get(
                "sweep_unit"
            )
            or ""
        )

        values = axis_config.get("values")

        if values is None:
            values = pulse_sequence_config.get(
                "sweep_values"
            )

        if isinstance(values, np.ndarray):
            values = values.tolist()

        if not isinstance(values, (list, tuple)):
            values = [values]

        values = list(values)

        if not values or values == [None]:
            values = list(range(ps_steps))

        if len(values) == ps_steps:
            values = [
                float(value)
                for value in values
            ]
        elif len(values) == 2 and ps_steps > 1:
            values = np.linspace(
                float(values[0]),
                float(values[1]),
                ps_steps,
            ).tolist()
        else:
            values = list(
                range(ps_steps)
            )

        return (
            str(parameter),
            unit,
            np.asarray(values, dtype=float),
        )

    def _selected_heatmap_matrix(self):
        choice = (
            self.heatmap_signal_combo.currentText()
        )

        if choice == "Combined mean":
            return self.result_combined_all

        if choice == "CH0":
            return self.result_channel_means_all[0]

        if choice == "CH1":
            if self.result_channel_means_all.shape[0] >= 2:
                return self.result_channel_means_all[1]

            return self.result_channel_means_all[0]

        if choice == "Normalized difference":
            return self.result_processed_all

        return self.result_processed_all
    
    def _draw_result_heatmap(self):
        matrix = np.asarray(
            self._selected_heatmap_matrix(),
            dtype=float,
        )

        frequencies = self.result_freqs_ghz
        ps_values = self.result_ps_values

        self.heatmap_image.setImage(
            matrix,
            autoLevels=True,
        )

        if len(frequencies) > 1:
            dx = (
                frequencies[-1]
                - frequencies[0]
            ) / (len(frequencies) - 1)
        else:
            dx = 1e-6

        rectangle = QtCore.QRectF(
            frequencies[0] - dx / 2,
            -0.5,
            dx * len(frequencies),
            matrix.shape[0],
        )

        self.heatmap_image.setRect(rectangle)

        maximum_ticks = min(
            12,
            len(ps_values),
        )

        tick_indices = np.unique(
            np.linspace(
                0,
                len(ps_values) - 1,
                maximum_ticks,
                dtype=int,
            )
        )

        ticks = [
            (
                int(index),
                f"{ps_values[index]:.6g}",
            )
            for index in tick_indices
        ]

        self.heatmap_plot.getAxis(
            "left"
        ).setTicks([ticks])

        y_label = self.result_ps_parameter

        if self.result_ps_unit:
            y_label += (
                f" ({self.result_ps_unit})"
            )

        self.heatmap_plot.setLabel(
            "bottom",
            "Frequency (GHz)",
        )
        self.heatmap_plot.setLabel(
            "left",
            y_label,
        )
        self.heatmap_plot.setTitle(
            self.heatmap_signal_combo.currentText()
        )

    def _on_experiment_changed(self, name):
        if not name:
            return

        self._apply_descriptor_defaults(name)

        if hasattr(self, "ps_sweep_param"):
            self._refresh_ps_sweep_parameters(name)

        self._update_pulse_diagram()
        self._populate_file_selector()

    def _on_file_selected(self, filename: str):
        if self._suppress_auto_switch:
            return
        
        if not filename:
            return
        try:
            self.data = np.load(filename)
            self.current_file = Path(filename)
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Could not load {filename}:\n{e}")
            return

        # and redraw everything
        self._show_results()

    def _descriptor_parameter_definition(
        self,
        parameter_name,
    ):
        exp_name = self.exp_combo.currentText()

        descriptor_path = (
            self.experiments_dir
            / f"{exp_name}.yaml"
        )

        try:
            descriptor = yaml.safe_load(
                descriptor_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return None

        for parameter in descriptor.get(
            "parameters",
            [],
        ):
            if parameter.get("name") == parameter_name:
                return parameter

        return None

    def _refresh_ps_sweep_parameters(
        self,
        exp_name,
    ):
        if not hasattr(self, "ps_sweep_param"):
            return

        descriptor_path = (
            self.experiments_dir
            / f"{exp_name}.yaml"
        )

        try:
            descriptor = yaml.safe_load(
                descriptor_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return

        previous_name = (
            self.ps_sweep_param.currentData()
        )

        self.ps_sweep_param.blockSignals(True)
        self.ps_sweep_param.clear()

        for parameter in descriptor.get(
            "parameters",
            [],
        ):
            name = parameter.get("name")
            parameter_type = parameter.get("type")

            if (
                parameter_type not in ("int", "float")
                or name in NON_PULSE_SWEEP_PARAMETERS
            ):
                continue

            self.ps_sweep_param.addItem(
                parameter.get("label", name),
                name,
            )

        if previous_name is not None:
            previous_index = (
                self.ps_sweep_param.findData(
                    previous_name
                )
            )

            if previous_index >= 0:
                self.ps_sweep_param.setCurrentIndex(
                    previous_index
                )

        self.ps_sweep_param.blockSignals(False)
        self._on_ps_sweep_parameter_changed()

    def _on_ps_sweep_parameter_changed(self, *_):
        if not hasattr(self, "ps_sweep_param"):
            return

        parameter_name = (
            self.ps_sweep_param.currentData()
        )

        if not parameter_name:
            return

        definition = (
            self._descriptor_parameter_definition(
                parameter_name
            )
        )

        if definition is None:
            return

        parameter_type = definition.get(
            "type",
            "float",
        )

        minimum = float(
            definition.get("min", -1e12)
        )
        maximum = float(
            definition.get("max", 1e12)
        )

        if parameter_name in TIMING_PARAMETER_NAMES:
            minimum /= self.time_factor
            maximum /= self.time_factor
            suffix = (
                f" {self.unit_combo.currentText()}"
            )
        else:
            unit = str(
                definition.get("unit", "")
            ).strip()
            suffix = f" {unit}" if unit else ""

        decimals = (
            0
            if parameter_type == "int"
            else 6
        )

        for widget in (
            self.ps_start_input,
            self.ps_stop_input,
        ):
            widget.blockSignals(True)
            widget.setDecimals(decimals)
            widget.setRange(minimum, maximum)
            widget.setSuffix(suffix)
            widget.blockSignals(False)

        source_widget = self.param_widgets.get(
            parameter_name
        )

        if (
            source_widget is not None
            and hasattr(source_widget, "value")
        ):
            current_value = source_widget.value()
        else:
            current_value = float(
                definition.get("default", 0)
            )

            if parameter_name in TIMING_PARAMETER_NAMES:
                current_value /= self.time_factor

        self.ps_start_input.setValue(
            current_value
        )
        self.ps_stop_input.setValue(
            current_value
        )
        self._update_pulse_diagram()

    def _sync_ps_preview_range(self, *_):
        if not hasattr(self, "ps_preview_step"):
            return

        maximum = max(
            0,
            self.ps_steps_input.value() - 1,
        )

        self.ps_preview_step.setMaximum(maximum)

        if self.ps_preview_step.value() > maximum:
            self.ps_preview_step.setValue(maximum)

        self._update_pulse_diagram()

    def _update_ps_sweep_controls(self, *_):
        if not hasattr(self, "ps_sweep_enable"):
            return

        enabled = self.ps_sweep_enable.isChecked()
        explicit = (
            self.ps_sweep_mode.currentText()
            == "Explicit list"
        )

        self.ps_sweep_param.setEnabled(enabled)
        self.ps_sweep_mode.setEnabled(enabled)
        self.ps_steps_input.setEnabled(
            enabled and not explicit
        )
        self.ps_start_input.setEnabled(
            enabled and not explicit
        )
        self.ps_stop_input.setEnabled(
            enabled and not explicit
        )
        self.ps_values_input.setEnabled(
            enabled and explicit
        )
        self.ps_preview_step.setEnabled(enabled)

        if not enabled:
            self.ps_steps_input.setValue(1)
            self.ps_preview_step.setValue(0)

        self._sync_ps_preview_range()

    def _collect_ps_sweep_config(self):
        if not self.ps_sweep_enable.isChecked():
            return {
                "steps": 1,
                "parameter": None,
                "values": [],
                "display_values": [0.0],
                "unit": "",
            }

        parameter_name = (
            self.ps_sweep_param.currentData()
        )

        if not parameter_name:
            raise ValueError(
                "Select a pulse-sequence sweep parameter."
            )

        definition = (
            self._descriptor_parameter_definition(
                parameter_name
            )
        )

        if definition is None:
            raise ValueError(
                f"Could not find descriptor parameter "
                f"{parameter_name!r}."
            )

        explicit = (
            self.ps_sweep_mode.currentText()
            == "Explicit list"
        )

        if explicit:
            tokens = [
                token
                for token in re.split(
                    r"[,;\s]+",
                    self.ps_values_input.text().strip(),
                )
                if token
            ]

            if not tokens:
                raise ValueError(
                    "Enter at least one explicit sweep value."
                )

            try:
                display_values = [
                    float(token)
                    for token in tokens
                ]
            except ValueError as error:
                raise ValueError(
                    "Explicit sweep values must be numeric."
                ) from error

            steps = len(display_values)

            self.ps_steps_input.blockSignals(True)
            self.ps_steps_input.setValue(steps)
            self.ps_steps_input.blockSignals(False)
            self._sync_ps_preview_range()
        else:
            steps = self.ps_steps_input.value()
            display_values = np.linspace(
                self.ps_start_input.value(),
                self.ps_stop_input.value(),
                steps,
            ).tolist()

        if definition.get("type") == "int":
            display_values = [
                int(round(value))
                for value in display_values
            ]

            if (
                steps > 1
                and len(set(display_values)) != steps
            ):
                raise ValueError(
                    "The requested integer sweep produces "
                    "duplicate values. Reduce the number of "
                    "steps or use an explicit list."
                )

        if parameter_name in TIMING_PARAMETER_NAMES:
            internal_values = [
                float(value) * self.time_factor
                for value in display_values
            ]
            unit = "µs"
        else:
            internal_values = [
                int(value)
                if definition.get("type") == "int"
                else float(value)
                for value in display_values
            ]
            unit = str(
                definition.get("unit", "")
            )

        return {
            "steps": steps,
            "parameter": parameter_name,
            "values": internal_values,
            "display_values": display_values,
            "unit": unit,
        }
    
    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Setup Tab ---
        setup = QtWidgets.QWidget()
        self.tabs.addTab(setup, 'Setup')

        # Main vertical layout for the complete Setup tab.
        self.setup_root_layout = QVBoxLayout(setup)
        self.setup_root_layout.setContentsMargins(
            8,
            8,
            8,
            8,
        )
        self.setup_root_layout.setSpacing(8)

        # Two-column region:
        #   left  = existing experiment/acquisition controls
        #   right = pulse-sequence sweep controls
        self.setup_columns_widget = QtWidgets.QWidget()
        setup_columns_layout = QHBoxLayout(
            self.setup_columns_widget
        )
        setup_columns_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        setup_columns_layout.setSpacing(12)

        self.setup_left_widget = QtWidgets.QWidget()
        form = QFormLayout(
            self.setup_left_widget
        )
        form.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.setup_right_widget = QtWidgets.QWidget()
        self.setup_right_layout = QVBoxLayout(
            self.setup_right_widget
        )
        self.setup_right_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        self.setup_right_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )

        setup_columns_layout.addWidget(
            self.setup_left_widget,
            3,
        )
        setup_columns_layout.addWidget(
            self.setup_right_widget,
            2,
        )

        self.setup_form = form

        # Watcher button
        self.start_watcher_btn = QtWidgets.QPushButton("Start watcher")
        self.start_watcher_btn.clicked.connect(self._start_watcher)
        self.setup_root_layout.addWidget(
            self.start_watcher_btn
        )

        self.setup_root_layout.addWidget(
            self.setup_columns_widget
        )

        # Experiment type
        self.exp_combo = QtWidgets.QComboBox()
        self.exp_combo.addItems(self.experiment_descs.keys())

        self.exp_combo.currentTextChanged.connect(
            self._on_experiment_changed
        )

        form.addRow('Experiment:', self.exp_combo)

        # dynamically build parameter widgets from descriptor
        desc = self.experiment_descs[self.exp_combo.currentText()]
        skip = {
            'mw_duration',
            'read_time',
            'laserduration',
            'frames',
            'mw_device_type',
            'mw_output',
        }

        for p in desc.get("parameters", []):
            if p["name"] in skip:
                continue

            w = self.make_widget_for(p)
            form.addRow(f"{p['label']}:", w)
            self.param_widgets[p['name']] = w

        # Time-unit selector
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['ns', 'µs', 'ms'])
        self.unit_combo.setCurrentText('µs')
        self.time_factor = 1.0
        form.addRow('Time unit:', self.unit_combo)

        # Sweep & Power
        self.start_input = QtWidgets.QDoubleSpinBox(); self.start_input.setSuffix(' GHz')
        self.stop_input = QtWidgets.QDoubleSpinBox();  self.stop_input.setSuffix(' GHz')
        self.steps_input = QtWidgets.QSpinBox()
        self.power_input = QtWidgets.QDoubleSpinBox(); self.power_input.setSuffix(' dBm')

        # Physical microwave-generator output port.
        self.mw_out_combo = QtWidgets.QComboBox()
        self.mw_out_combo.addItems(['A', 'B'])
        self.mw_out_combo.setCurrentText('B')

        self.mw_dev_combo = QtWidgets.QComboBox()
        self.mw_dev_combo.addItems([
            'WindFreak',
            'WindFreakSHDMini',
            'WindFreakHDM',
            'WindFreakSNV',
        ])
        self.mw_dev_combo.setCurrentText('WindFreak')

        for frequency_input in (self.start_input, self.stop_input):
            frequency_input.setRange(0.0, 100.0)
            frequency_input.setDecimals(6)
            frequency_input.setSingleStep(0.001)

        self.start_input.setValue(2.65)
        self.power_input.setRange(0.01, 8.0)
        self.power_input.setValue(1.0)
        self.power_input.setDecimals(2)
        self.power_input.setSingleStep(0.5)        
        
        form.addRow('Sweep start:', self.start_input)
        form.addRow('Sweep stop:', self.stop_input)
        form.addRow('MW source:', self.mw_dev_combo)
        form.addRow('MW output:', self.mw_out_combo)
        form.addRow('RF power:', self.power_input)

        # Averaging & Acquisition
        self.avg_input = QtWidgets.QSpinBox()
        self.frames_input = QtWidgets.QSpinBox()
        self.dynamic_input = QtWidgets.QSpinBox()
        self.refch_input = QtWidgets.QSpinBox()
        self.avg_input.setRange(1, 9999)
        self.frames_input.setRange(1, 9999)
        self.dynamic_input.setRange(1, 9999)
        self.refch_input.setRange(1, 2)
                
        self.mode_input = QtWidgets.QComboBox(); self.mode_input.addItems(['spread', 'sum'])
        self.mode_input.setCurrentText('sum')

        form.addRow('Averages:', self.avg_input)
        form.addRow('Frames:', self.frames_input)
        form.addRow('Dynamic steps:', self.dynamic_input)
        form.addRow('Averaging mode:', self.mode_input)
        form.addRow('Ref channels:', self.refch_input)

        # Pulse timings
        self.mw_dur = QtWidgets.QDoubleSpinBox(); self.mw_dur.setSuffix(' μs')
        self.read_dur = QtWidgets.QDoubleSpinBox(); self.read_dur.setSuffix(' μs')
        self.las_dur = QtWidgets.QDoubleSpinBox(); self.las_dur.setSuffix(' μs')
        self.rate = QtWidgets.QSpinBox(); self.rate.setSuffix(' Hz')
        self.ref_rep = QtWidgets.QSpinBox()
        self.unit_combo.currentTextChanged.connect(self._update_time_units)

        defaults = { p["name"]: p["default"] for p in desc.get("parameters", []) }
        self.mw_dur  .setValue(defaults.get("mw_duration",  0.0))
        self.read_dur.setValue(defaults.get("read_time",     0.0))
        self.las_dur .setValue(defaults.get("laserduration", 0.0))

        form.addRow('MW duration:', self.mw_dur)
        form.addRow('Readout duration:', self.read_dur)
        form.addRow('Laser duration:', self.las_dur)
        form.addRow('Max rate:', self.rate)

        # make sure the three timing‐params available to Jinja
        self.param_widgets['mw_duration']   = self.mw_dur
        self.param_widgets['read_time']     = self.read_dur
        self.param_widgets['laserduration'] = self.las_dur

        self.rate.setRange(1, 1_000_000)
        self.rate.setValue(16_000)
        self.start_pulse_dur = QtWidgets.QDoubleSpinBox()
        self.start_pulse_dur.setSuffix(' μs')
        self.start_pulse_dur.setRange(0.0, 1e6)   # e.g. up to 1 s
        self.start_pulse_dur.setDecimals(3)
        self.start_pulse_dur.setValue(10.0)       # default 10 μs

        form.addRow('Start pulse duration:', self.start_pulse_dur)
        self.param_widgets['start_pulse_dur'] = self.start_pulse_dur


        # Pulse controls
        self.I_pulse_dur     = QtWidgets.QDoubleSpinBox()
        self.Q_pulse_dur     = QtWidgets.QDoubleSpinBox()
        self.tau_input       = QtWidgets.QDoubleSpinBox()
        self.blocks_input    = QtWidgets.QSpinBox()

        self.I_pulse_dur.setValue(0.25)    # µs
        self.Q_pulse_dur.setValue(0.25)    # µs
        self.tau_input  .setValue(2.0)     # µs

        form.addRow('I pulse duration:',  self.I_pulse_dur)
        form.addRow('Q pulse duration:',  self.Q_pulse_dur)
        form.addRow('τ (inter-pulse):',   self.tau_input)
        form.addRow('Number of blocks:',  self.blocks_input)

        # register I/Q/τ so Load Defaults and experiment-switch will set them
        self.param_widgets['I_pulse'] = self.I_pulse_dur
        self.param_widgets['Q_pulse'] = self.Q_pulse_dur
        self.param_widgets['tau']     = self.tau_input

        self.blocks_input.setRange(1, 100)
        self.blocks_input.setValue(1)

        # make tau and blocks available to the pulse‐diagram’s Jinja context
        self.param_widgets['tau']    = self.tau_input
        self.param_widgets['blocks'] = self.blocks_input

        # ------------------------------------------------------------
        # Pulse-sequence sweep
        # ------------------------------------------------------------
        self.ps_sweep_group = QGroupBox(
            "Pulse-Sequence Sweep"
        )
        ps_form = QFormLayout(
            self.ps_sweep_group
        )

        self.ps_sweep_enable = QtWidgets.QCheckBox(
            "Enable pulse-sequence sweep"
        )

        self.ps_sweep_param = QtWidgets.QComboBox()

        self.ps_steps_input = QtWidgets.QSpinBox()
        self.ps_steps_input.setRange(1, 9999)
        self.ps_steps_input.setValue(1)

        self.ps_sweep_mode = QtWidgets.QComboBox()
        self.ps_sweep_mode.addItems(
            [
                "Start / Stop",
                "Explicit list",
            ]
        )

        self.ps_start_input = QtWidgets.QDoubleSpinBox()
        self.ps_stop_input = QtWidgets.QDoubleSpinBox()

        for widget in (
            self.ps_start_input,
            self.ps_stop_input,
        ):
            widget.setRange(-1e12, 1e12)
            widget.setDecimals(6)

        self.ps_values_input = QtWidgets.QLineEdit()
        self.ps_values_input.setPlaceholderText(
            "Example: 0.01, 0.02, 0.05, 0.10"
        )

        self.ps_preview_step = QtWidgets.QSpinBox()
        self.ps_preview_step.setRange(0, 0)
        self.ps_preview_step.setValue(0)

        ps_form.addRow(
            self.ps_sweep_enable
        )
        ps_form.addRow(
            "Parameter:",
            self.ps_sweep_param,
        )
        ps_form.addRow(
            "Number of steps:",
            self.ps_steps_input,
        )
        ps_form.addRow(
            "Values mode:",
            self.ps_sweep_mode,
        )
        ps_form.addRow(
            "Start:",
            self.ps_start_input,
        )
        ps_form.addRow(
            "Stop:",
            self.ps_stop_input,
        )
        ps_form.addRow(
            "Explicit values:",
            self.ps_values_input,
        )
        ps_form.addRow(
            "Preview step:",
            self.ps_preview_step,
        )

        self.ps_sweep_group.setMinimumWidth(300)

        self.setup_right_layout.addWidget(
            self.ps_sweep_group
        )
        self.setup_right_layout.addStretch(1)

        self.ps_sweep_enable.toggled.connect(
            self._update_ps_sweep_controls
        )
        self.ps_sweep_mode.currentTextChanged.connect(
            self._update_ps_sweep_controls
        )
        self.ps_sweep_param.currentIndexChanged.connect(
            self._on_ps_sweep_parameter_changed
        )
        self.ps_steps_input.valueChanged.connect(
            self._sync_ps_preview_range
        )
        self.ps_preview_step.valueChanged.connect(
            self._update_pulse_diagram
        )
        self.ps_start_input.valueChanged.connect(
            self._update_pulse_diagram
        )
        self.ps_stop_input.valueChanged.connect(
            self._update_pulse_diagram
        )
        self.ps_values_input.textChanged.connect(
            self._update_pulse_diagram
        )

        self._refresh_ps_sweep_parameters(
            self.exp_combo.currentText()
        )
        self._update_ps_sweep_controls()

        self._init_pulse_diagram()
        self._update_time_units(self.unit_combo.currentText())

        # Controls below the full-width pulse diagram.
        self.setup_post_widget = QtWidgets.QWidget()
        post_form = QFormLayout(
            self.setup_post_widget
        )
        post_form.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.setup_post_form = post_form
        self.setup_root_layout.addWidget(
            self.setup_post_widget
        )

        # Processing & Display
        self.sub_input = QtWidgets.QCheckBox(
            'Top result plot: normalized difference (S-R)/(S+R)'
        )

        self.sub_input.setChecked(True)

        self.smooth_input = QtWidgets.QSpinBox()
        self.smooth_input.setRange(0, 9999)
        self.smooth_input.setValue(0)
        self.smooth_input.setSpecialValueText("Off")
        self.fit_input = QtWidgets.QComboBox(); self.fit_input.addItems(['Lorentzian', 'Gaussian'])
        self.errb_input = QtWidgets.QCheckBox(
            'Channel PNG export: include error bars'
        )
       
        self.sub_input.stateChanged.connect(
            lambda *_: self._show_results()
        )
        self.fit_input.currentTextChanged.connect(
            lambda *_: self._show_results()
        )
        self.smooth_input.valueChanged.connect(
            lambda *_: self._show_results()
        )

        post_form.addRow(self.sub_input)
        post_form.addRow(
            'Smoothing window:',
            self.smooth_input,
        )
        post_form.addRow(
            'Fit type:',
            self.fit_input,
        )
        post_form.addRow(self.errb_input)

        # Buttons
        h = QtWidgets.QHBoxLayout()
        self.defaults_btn    = QtWidgets.QPushButton('Load Defaults')
        self.defaults_btn.clicked.connect(self._load_defaults)
        self.start_setup_btn = QtWidgets.QPushButton('Start')
        self.stop_btn        = QtWidgets.QPushButton('Stop')

        # Save/Load configuration buttons
        self.save_cfg_btn    = QtWidgets.QPushButton('Save Config…')
        self.load_cfg_btn    = QtWidgets.QPushButton('Load Config…')

        self.powersupply_btn = QtWidgets.QPushButton(
            "Power Supply…"
        )
        self.powersupply_btn.clicked.connect(
            self._open_power_supply_dialog
        )
        post_form.addRow(self.powersupply_btn)

        h.addWidget(self.defaults_btn)
        h.addWidget(self.start_setup_btn)
        h.addWidget(self.stop_btn)
        h.addWidget(self.save_cfg_btn)
        h.addWidget(self.load_cfg_btn)
        post_form.addRow(h)

        self.start_setup_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._double_stop)
        self.save_cfg_btn.clicked.connect(self._save_config)
        self.load_cfg_btn.clicked.connect(self._load_config)

        # --- Live Tab ---
        live = QtWidgets.QWidget()
        self.tabs.addTab(live, 'Live')
        live_layout = QtWidgets.QVBoxLayout(live)
        self.run_live_btn = QtWidgets.QPushButton("Run experiment")
        self.run_live_btn.clicked.connect(self._deploy_yaml_and_run)
        live_layout.addWidget(self.run_live_btn)

        # Clear Live tab button
        self.clear_live_btn = QtWidgets.QPushButton("Clear Live")
        self.clear_live_btn.clicked.connect(self._clear_live)
        live_layout.addWidget(self.clear_live_btn)
        
        # # Clear waiting-room button
        # self.clear_waiting_btn = QtWidgets.QPushButton("Clear waiting room")
        # self.clear_waiting_btn.clicked.connect(self._clear_waiting_room)
        # live_layout.addWidget(self.clear_waiting_btn)

        # ------------------------------------------------------------
        # Live plots:
        #
        #   CH0 mean             CH1 mean
        #   Normalized diff.     Combined mean
        # ------------------------------------------------------------
        self.live_plots_widget = QtWidgets.QWidget()
        live_plots_layout = QtWidgets.QGridLayout(
            self.live_plots_widget
        )
        live_plots_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        live_plots_layout.setHorizontalSpacing(8)
        live_plots_layout.setVerticalSpacing(8)

        # Top-left: raw CH0 signal.
        self.live_ch0_plot = pg.PlotWidget(
            title="CH0 Mean — First READ"
        )
        self.live_ch0_curve = self.live_ch0_plot.plot(
            [],
            [],
            pen=None,
            symbol="o",
        )
        self.live_ch0_plot.setLabel(
            "bottom",
            "Frequency (GHz)",
        )
        self.live_ch0_plot.setLabel(
            "left",
            "CH0 voltage (V)",
        )

        # Top-right: raw CH1 signal.
        self.live_ch1_plot = pg.PlotWidget(
            title="CH1 Mean — Second READ"
        )
        self.live_ch1_curve = self.live_ch1_plot.plot(
            [],
            [],
            pen=None,
            symbol="o",
        )
        self.live_ch1_plot.setLabel(
            "bottom",
            "Frequency (GHz)",
        )
        self.live_ch1_plot.setLabel(
            "left",
            "CH1 voltage (V)",
        )

        # Bottom-left: normalized referenced signal.
        self.live_normalized_plot = pg.PlotWidget(
            title="Normalized Difference"
        )
        self.live_normalized_curve = (
            self.live_normalized_plot.plot(
                [],
                [],
                pen=None,
                symbol="o",
            )
        )
        self.live_normalized_plot.setLabel(
            "bottom",
            "Frequency (GHz)",
        )
        self.live_normalized_plot.setLabel(
            "left",
            "(CH0 - CH1) / (CH0 + CH1)",
        )

        # Bottom-right: average of the two reference channels.
        self.live_combined_plot = pg.PlotWidget(
            title="Combined Mean"
        )
        self.live_combined_curve = (
            self.live_combined_plot.plot(
                [],
                [],
                pen=None,
                symbol="o",
            )
        )
        self.live_combined_plot.setLabel(
            "bottom",
            "Frequency (GHz)",
        )
        self.live_combined_plot.setLabel(
            "left",
            "(CH0 + CH1) / 2 (V)",
        )

        live_plots_layout.addWidget(
            self.live_ch0_plot,
            0,
            0,
        )
        live_plots_layout.addWidget(
            self.live_ch1_plot,
            0,
            1,
        )
        live_plots_layout.addWidget(
            self.live_normalized_plot,
            1,
            0,
        )
        live_plots_layout.addWidget(
            self.live_combined_plot,
            1,
            1,
        )

        # Keep all four plot panels equally sized.
        live_plots_layout.setColumnStretch(0, 1)
        live_plots_layout.setColumnStretch(1, 1)
        live_plots_layout.setRowStretch(0, 1)
        live_plots_layout.setRowStretch(1, 1)

        # ——— Current values display ———
        hl = QtWidgets.QHBoxLayout()
        self.live_ps_label = QLabel(
            "Pulse-sequence step: 1/1"
        )
        self.freq_label = QLabel("Frequency: -- GHz")
        self.voltage_label = QLabel("APD voltage: -- V")
        self.ch0_label = QLabel("CH0: -- V")
        self.ch1_label = QLabel("CH1: -- V")
        self.normalized_label = QLabel(
            "Normalized: --"
        )
        self.combined_label = QLabel(
            "Combined: -- V"
        )

        for w in (
            self.live_ps_label,
            self.freq_label,
            self.voltage_label,
            self.ch0_label,
            self.ch1_label,
            self.normalized_label,
            self.combined_label,
        ):
            w.setStyleSheet("font-size: 11pt; font-weight: bold;")

        hl.addWidget(self.live_ps_label)
        hl.addStretch()
        hl.addWidget(self.freq_label)
        hl.addWidget(self.ch0_label)
        hl.addStretch()
        hl.addWidget(self.ch1_label)
        hl.addStretch()
        hl.addWidget(self.normalized_label)
        hl.addStretch()
        hl.addWidget(self.combined_label)
        hl.addStretch()
        hl.addStretch()
        hl.addWidget(self.voltage_label)
        live_layout.addLayout(hl)

        # Status Section
        status_box = QGroupBox("Status")
        sb = QtWidgets.QHBoxLayout()
        self.status_led = QLabel()
        self.status_led.setFixedSize(16,16)
        self.status_led.setStyleSheet("background-color: red; border-radius: 8px;")
        sb.addWidget(self.status_led)
        self.status_label = QLabel("Idle")
        sb.addWidget(self.status_label)
        sb.addStretch()
        status_box.setLayout(sb)

        # Progress Section
        prog_box = QGroupBox("Progress")
        pb = QtWidgets.QVBoxLayout()

        # 1) create the label, then style it
        self.step_label = QLabel("Step 0/0")
        self.step_label.setStyleSheet("font-size: 14pt; font-weight: bold;")

        # 2) now create the bars
        self.sweep_bar  = QtWidgets.QProgressBar()
        self.sweep_bar.setFormat("Sweep %p%")
        self.count_gauge= QtWidgets.QProgressBar()
        self.count_gauge.setFormat("Steps: %v/%m")

        # 3) add to layout
        pb.addWidget(self.step_label)
        pb.addWidget(self.sweep_bar)
        pb.addWidget(self.count_gauge)
        prog_box.setLayout(pb)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(2000)

        # Put plot and console into a splitter for adjustable space
        live_splitter = QSplitter(QtCore.Qt.Orientation.Vertical)
        live_splitter.addWidget(self.live_plots_widget)
        live_splitter.addWidget(self.log_output)
        
        # Give the four-plot grid more space than the console.
        live_splitter.setStretchFactor(0, 4)
        live_splitter.setStretchFactor(1, 3)
        live_layout.addWidget(live_splitter)

        # Status Section
        live_layout.addWidget(status_box)

        # Progress Section 
        live_layout.addWidget(prog_box)

        # Results Tab
        res = QtWidgets.QWidget()
        self.tabs.addTab(res, 'Results')

        # ─── File-selector dropdown ───
        self.file_selector = QtWidgets.QComboBox()
        self.file_selector.setToolTip("Select an ODMR .npy file to display")
        self.file_selector.currentTextChanged.connect(self._on_file_selected)

        self.result_controls_widget = QtWidgets.QWidget()
        result_controls = QHBoxLayout(
            self.result_controls_widget
        )
        result_controls.setContentsMargins(0, 0, 0, 0)

        self.result_view_combo = QtWidgets.QComboBox()
        self.result_view_combo.addItems(
            [
                "1D spectrum",
                "2D heatmap",
            ]
        )

        self.result_ps_step = QtWidgets.QSpinBox()
        self.result_ps_step.setRange(0, 0)

        self.result_ps_value_label = QLabel(
            "Pulse-sequence value: --"
        )

        self.heatmap_signal_combo = QtWidgets.QComboBox()
        self.heatmap_signal_combo.addItems(
            [
                "Normalized difference",
                "Combined mean",
                "CH0",
                "CH1",
            ]
        )

        result_controls.addWidget(QLabel("View:"))
        result_controls.addWidget(
            self.result_view_combo
        )
        result_controls.addSpacing(15)
        result_controls.addWidget(
            QLabel("Pulse-sequence step:")
        )
        result_controls.addWidget(
            self.result_ps_step
        )
        result_controls.addWidget(
            self.result_ps_value_label,
            1,
        )
        result_controls.addWidget(
            QLabel("Heatmap signal:")
        )
        result_controls.addWidget(
            self.heatmap_signal_combo
        )

        self.result_view_combo.currentTextChanged.connect(
            self._on_results_control_changed
        )
        self.result_ps_step.valueChanged.connect(
            self._show_results
        )
        self.heatmap_signal_combo.currentTextChanged.connect(
            self._show_results
        )
        
        splitter = QSplitter(QtCore.Qt.Orientation.Vertical, res)

        # 1. Summary & Metadata
        meta_box = QGroupBox('Summary & Metadata')
        meta_layout = QVBoxLayout()
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        meta_layout.addWidget(self.meta_text)
        meta_box.setLayout(meta_layout)

        # 2. Processed ODMR Spectrum
        self.proc_box = QGroupBox('Processed ODMR Spectrum')
        proc_layout = QVBoxLayout()
        self.proc_plot = pg.PlotWidget()
        proc_layout.addWidget(self.proc_plot)

        # Buttons for saving the processed spectrum and raw channel plots
        save_plot_layout = QHBoxLayout()
        save_plot_layout.addStretch()
        save_spec_btn = QtWidgets.QPushButton("Save Spectrum…")
        save_spec_btn.clicked.connect(lambda: self._save_plot(self.proc_plot))
        save_channels_btn = QtWidgets.QPushButton("Save Channel Plots…")
        save_channels_btn.clicked.connect(self._save_channel_plots)

        save_plot_layout.addWidget(save_spec_btn)
        save_plot_layout.addWidget(save_channels_btn)
        proc_layout.addLayout(save_plot_layout)

        self.proc_box.setLayout(proc_layout)

        # 3. Fit & Parameter Readout
        self.fit_box = QGroupBox('Fit & Parameters')
        fit_layout = QVBoxLayout()
        self.fit_table = QTableWidget(4, 2)
        self.fit_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        params = ['Center (GHz)', 'FWHM (MHz)', 'Contrast (%)', 'R²']
        for i, p in enumerate(params):
            self.fit_table.setItem(i, 0, QTableWidgetItem(p))
        fit_layout.addWidget(self.fit_table)
        self.fit_box.setLayout(fit_layout)

        # 3b. Combined-channel ODMR spectrum: (CH0 + CH1) / 2
        self.mean_proc_box = QGroupBox(
            'Combined-Channel ODMR Spectrum — (CH0 + CH1) / 2'
        )
        mean_proc_layout = QVBoxLayout()

        self.mean_proc_plot = pg.PlotWidget()
        mean_proc_layout.addWidget(self.mean_proc_plot)

        mean_save_layout = QHBoxLayout()
        mean_save_layout.addStretch()

        save_mean_btn = QtWidgets.QPushButton(
            "Save Combined Spectrum…"
        )
        save_mean_btn.clicked.connect(
            lambda: self._save_plot(self.mean_proc_plot)
        )

        mean_save_layout.addWidget(save_mean_btn)
        mean_proc_layout.addLayout(mean_save_layout)
        self.mean_proc_box.setLayout(mean_proc_layout)

        # Fit parameters for the combined-channel spectrum
        self.mean_fit_box = QGroupBox(
            'Fit & Parameters — (CH0 + CH1) / 2'
        )
        mean_fit_layout = QVBoxLayout()

        self.mean_fit_table = QTableWidget(4, 2)
        self.mean_fit_table.setHorizontalHeaderLabels(
            ['Parameter', 'Value']
        )

        mean_params = [
            'Center (GHz)',
            'FWHM (MHz)',
            'Contrast (%)',
            'R²',
        ]

        for row, parameter in enumerate(mean_params):
            self.mean_fit_table.setItem(
                row,
                0,
                QTableWidgetItem(parameter)
            )

        mean_fit_layout.addWidget(self.mean_fit_table)
        self.mean_fit_box.setLayout(mean_fit_layout)

        self.heatmap_box = QGroupBox(
            "Pulse-Sequence Sweep Heatmap"
        )
        heatmap_layout = QVBoxLayout()

        self.heatmap_plot = pg.PlotWidget()
        self.heatmap_image = pg.ImageItem(
            axisOrder="row-major"
        )
        self.heatmap_plot.addItem(
            self.heatmap_image
        )

        heatmap_layout.addWidget(
            self.heatmap_plot
        )
        self.heatmap_box.setLayout(
            heatmap_layout
        )
        self.heatmap_box.setVisible(False)

        # 4. Data Summary & “View” Button
        summary_box = QGroupBox("Data Summary")
        summary_layout = QHBoxLayout()
        summary_layout.setContentsMargins(8, 4, 8, 4)
        self.summary_label = QLabel("No data loaded")
        self.summary_label.setWordWrap(False)
        summary_layout.addWidget(self.summary_label, 1)

        self.view_data_btn = QtWidgets.QPushButton("View Data…")
        self.view_data_btn.clicked.connect(self._on_view_data)
        summary_layout.addWidget(
            self.view_data_btn,
            0,
            QtCore.Qt.AlignmentFlag.AlignTop
        )

        summary_box.setLayout(summary_layout)
        summary_box.setMaximumHeight(75)

        splitter.addWidget(summary_box)
        splitter.addWidget(meta_box)
        splitter.addWidget(self.proc_box)
        splitter.addWidget(self.fit_box)
        splitter.addWidget(self.mean_proc_box)
        splitter.addWidget(self.mean_fit_box)
        splitter.addWidget(self.heatmap_box)

        layout = QtWidgets.QVBoxLayout(res)
        layout.addWidget(self.file_selector)
        layout.addWidget(self.result_controls_widget)
        layout.addWidget(splitter)
        res.setLayout(layout)

        # Export button
        self.export_btn = QtWidgets.QPushButton('Export Results')
        layout.addWidget(self.export_btn)
        self.export_btn.clicked.connect(self._export)

        # ——— Experiments Tab ———
        exp_mgmt = QtWidgets.QWidget()
        self.tabs.addTab(exp_mgmt, "Experiments")
        mgmt_layout = QVBoxLayout(exp_mgmt)

        # List of existing experiments
        self.exp_list = QtWidgets.QListWidget()
        mgmt_layout.addWidget(self.exp_list)

        # Buttons: Add / Edit / Delete
        h = QtWidgets.QHBoxLayout()
        self.btn_add    = QtWidgets.QPushButton("Add…")
        self.btn_edit   = QtWidgets.QPushButton("Edit…")
        self.btn_remove = QtWidgets.QPushButton("Remove")
        h.addWidget(self.btn_add)
        h.addWidget(self.btn_edit)
        h.addWidget(self.btn_remove)
        mgmt_layout.addLayout(h)

        # wire up
        self.btn_add.clicked.connect(self._on_add_experiment)
        self.btn_edit.clicked.connect(self._on_edit_experiment)
        self.btn_remove.clicked.connect(self._on_remove_experiment)

        # finally: populate the list
        self._refresh_experiment_list()


    def _start_watcher(self):
        self._suppress_auto_switch = True

        self._live_reference_channels = max(
            1,
            self.refch_input.value(),
        )

        if self.process and self.process.state() == QtCore.QProcess.ProcessState.Running:
            self.process.kill()

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.started.connect(self._on_started)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.finished.connect(self._on_finished)

        cmd  = sys.executable
        args = ['-u', '-m', 'qupyt.main']

        self.process.setWorkingDirectory(
            str(QUPYT_ROOT)
        )

        max_steps = self.dynamic_input.value()
        ps_steps = max(
            1,
            int(
                getattr(
                    self,
                    "_active_pulse_sequence_steps",
                    1,
                )
            ),
        )

        self._live_ps_steps = ps_steps
        self.count_gauge.setMinimum(0)
        self.count_gauge.setMaximum(max_steps)
        self.count_gauge.setValue(0)

        self.max_live_points = max_steps

        self.sweep_bar.setValue(0)

        # One plotted point per dynamic frequency step. Each point is the
        # running average of the acquisitions collected at that frequency.
        number_steps = max(1, self.dynamic_input.value())

        frequency_axis = np.linspace(
            self.start_input.value(),
            self.stop_input.value(),
            number_steps
        )

        self.live_freqs = np.tile(
            frequency_axis,
            (ps_steps, 1),
        )

        self.live_ch0 = np.full(
            (ps_steps, number_steps),
            np.nan,
            dtype=float,
        )
        self.live_ch1 = np.full(
            (ps_steps, number_steps),
            np.nan,
            dtype=float,
        )

        self.live_normalized = np.full(
            (ps_steps, number_steps),
            np.nan,
            dtype=float,
        )
        self.live_combined = np.full(
            (ps_steps, number_steps),
            np.nan,
        )
        self._live_ch0_sums = np.zeros(
            (ps_steps, number_steps),
            dtype=float,
        )
        self._live_ch1_sums = np.zeros(
            (ps_steps, number_steps),
        )
        self._live_samples_per_step = np.zeros(
            (ps_steps, number_steps),
            dtype=int,
        )

        self.live_ps_label.setText(
            f"Pulse-sequence step: 1/{ps_steps}"
        )

        if self._live_reference_channels == 1:
            self.live_ch0_plot.setTitle(
                "CH0 Mean — First READ"
            )
            self.live_ch1_plot.setTitle(
                "CH1 Mean — unavailable"
            )
            self.live_normalized_plot.setTitle(
                "Normalized Difference — requires CH0 and CH1"
            )
            self.live_combined_plot.setTitle(
                "Single Reference Channel"
            )
            self.live_combined_plot.setLabel(
                "left",
                "CH0 (V)",
            )
        else:
            self.live_ch0_plot.setTitle(
                "CH0 Mean — First READ"
            )
            self.live_ch1_plot.setTitle(
                "CH1 Mean — Second READ"
            )
            self.live_normalized_plot.setTitle(
                "Normalized Difference"
            )
            self.live_combined_plot.setTitle(
                "Combined Mean"
            )
            self.live_combined_plot.setLabel(
                "left",
                "(CH0 + CH1) / 2 (V)",
            )
        self.live_ch0_curve.setData(
            [],
            [],
        )
        self.live_ch1_curve.setData(
            [],
            [],
        )

        self.live_normalized_curve.setData(
            [],
            [],
        )
        self.live_combined_curve.setData(
            [],
            [],
        )

        for attribute in (
            "_last_freq",
            "_last_ch0",
            "_last_ch1",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)

        self._live_measurement_index = 0

        # Start only after every live-data buffer is ready.
        self.process.start(cmd, args)

        # flip to Live tab so you can watch logs
        self.tabs.setCurrentIndex(1)


    def _clear_waiting_room(self):
        """Remove every file and subdirectory from ~/.qupyt/waiting_room."""
        wait_dir = Path.home() / ".qupyt" / "waiting_room"
        wait_dir.mkdir(parents=True, exist_ok=True)

        errors = []

        for path in wait_dir.iterdir():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as e:
                errors.append(f"{path.name}: {e}")

        if errors and hasattr(self, "log_output"):
            self.log_output.appendPlainText(
                "Waiting-room cleanup errors:\n" + "\n".join(errors)
            )

    def _deploy_yaml_and_run(self):
        exp_name = self.exp_combo.currentText()
        desktop_yaml = Path.home() / 'Desktop' / f"{exp_name}.yaml"
        if not desktop_yaml.exists():
            QMessageBox.critical(self, "Deployment Error", f"Could not find {desktop_yaml}")
            return
        
        # atomic copy into waiting room
        wait_dir = Path.home() / '.qupyt' / 'waiting_room'
        wait_dir.mkdir(parents=True, exist_ok=True)
        target = wait_dir / f"{exp_name}.yaml"

        temporary_target = target.with_suffix(
            ".yaml.tmp"
        )

        if temporary_target.exists():
            temporary_target.unlink()

        shutil.copyfile(
            desktop_yaml,
            temporary_target,
        )
        os.replace(
            temporary_target,
            target,
        )

        QMessageBox.information(
            self,
            "Deployed",
            f"{target.name} deployed—starting run now."
        )


    def make_widget_for(self, p: dict):
        kind = p["type"]
        if kind == "int":
            w = QSpinBox()
            w.setRange(p["min"], p["max"])
            w.setValue(p["default"])
            return w
        elif kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(p["min"], p["max"])
            w.setValue(p["default"])
            if "unit" in p:
                w.setSuffix(f" {p['unit']}")
            return w
        elif kind == "choice":
            w = QComboBox()
            w.addItems(p["choices"])
            w.setCurrentText(p["default"])
            return w
        else:
            raise ValueError(f"Unknown parameter type {kind!r}")    

    def _start(self):
        frames = self.frames_input.value()
        reference_channels = self.refch_input.value()

        if frames % reference_channels != 0:
            QMessageBox.warning(
                self,
                "Invalid acquisition settings",
                (
                    f"Frames ({frames}) must be divisible by Ref channels "
                    f"({reference_channels})."
                )
            )
            return

        try:
            ps_sweep = self._collect_ps_sweep_config()
        except ValueError as error:
            QMessageBox.warning(
                self,
                "Invalid pulse-sequence sweep",
                str(error),
            )
            return

        self._active_pulse_sequence_steps = (
            ps_sweep["steps"]
        )

        exp_name = self.exp_combo.currentText()
        desc     = self.experiment_descs[exp_name]

        vals = {
            "experiment_type": exp_name,
            "apd_input":       "Dev1/ai0",
            "address":         "COM3", 
            **CHANNEL_MAPPING,

            # static GUI inputs:
            "averages":         self.avg_input.value(),
            "frames":           self.frames_input.value(),
            "n_dynamic_steps":  self.dynamic_input.value(),
            "freq_start":       self.start_input.value() * 1e9,
            "freq_stop":        self.stop_input.value()  * 1e9,
            "power":            self.power_input.value(),
            "mw_device_type":   self.mw_dev_combo.currentText(),
            "mw_output":        self.mw_out_combo.currentText(),
            "mode":             self.mode_input.currentText(),
            "ref_channels":     self.refch_input.value(),
            "pulse_sequence_steps": ps_sweep["steps"],
            "sweep_param":      ps_sweep["parameter"],
            "sweep_values":     ps_sweep["values"],
            "sweep_unit":       ps_sweep["unit"],
            "address":          "COM3",
            "ps_path":          desc.get("pulse_generator",""),

            # pulse sequence timings:
            "mw_duration":      self.mw_dur.value() * self.time_factor,
            "laser_time":       self.las_dur.value() * self.time_factor,
            "read_time":        self.read_dur.value() * self.time_factor,
            "max_rate":         self.rate.value(),
        }

        # grab every dynamic parameter:
        for name, w in self.param_widgets.items():
            if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                value = w.value()

                if name in {
                    "mw_duration",
                    "read_time",
                    "laserduration",
                    "start_pulse_dur",
                    "I_pulse",
                    "Q_pulse",
                    "tau",
                }:
                    value *= self.time_factor

                vals[name] = value
            else:
                vals[name] = w.currentText()

        # generic_generator uses "laserduration", while odmr_yaml uses
        # the older key "laser_time".
        vals["laser_time"] = vals["laserduration"]

        # odmr_yaml expects lowercase I/Q duration keys.
        vals["i_pulse"] = vals.get("I_pulse", 0.0)
        vals["q_pulse"] = vals.get("Q_pulse", 0.0)

        # generate the low-level Python module exactly once
        desc_dict = yaml.safe_load((self.experiments_dir / f"{exp_name}.yaml").read_text())
        pulse_py = GUI_DIR / "user_pulse_seq.py"
        generate_from_descriptor(desc_dict, vals, str(pulse_py))

        # now write the experiment YAML, pointing at that Python module
        vals['ps_path'] = str(pulse_py)
        desktop_yaml = Path.home() / 'Desktop' / f"{exp_name}.yaml"
        render_experiment_yaml(vals, desktop_yaml)

        # Prevent the file-selector from auto-jumping us
        self.file_selector.blockSignals(True)

        # snapshot the current GUI values:
        self._write_last_config()

        # spawn the watcher/process as usual
        self._start_watcher()

        # Force ourselves to the Live tab
        self.tabs.setCurrentIndex(1)

    def _on_stdout(self):
        try:
            raw = bytes(self.process.readAll()).decode('utf-8', errors='ignore')
        except Exception as e:
            QMessageBox.warning(self, "Read Error", str(e))
            return

        self.log_output.appendPlainText(raw)

        for line in raw.splitlines():
            m = re.search(r"\|\s*(\d+)/(\d+)\b", line)
            if m:
                step, total = map(int, m.groups())
                # Ignore QuPyt's inner averages progress bar.
                if total == self.dynamic_input.value():
                    self.total_steps = total
                    self.step_label.setText(f"Step {step}/{total}")
                    pct = int(100 * step / total)
                    self.sweep_bar.setValue(pct)
                    self.count_gauge.setMaximum(total)
                    self.count_gauge.setValue(step)

            # ——— now the state‐machine for live plotting ———
            # Frequency may be absent because verbose output is disabled.
            if freq_m := re.search(
                rf"frequency.*?({NUMBER_PATTERN})",
                line,
                flags=re.IGNORECASE
            ):
                self._last_freq = float(freq_m.group(1)) / 1e9

            if ch0_match := re.search(
                rf"DAQ_CH0:\s*({NUMBER_PATTERN})",
                line,
                flags=re.IGNORECASE,
            ):
                self._last_ch0 = float(
                    ch0_match.group(1)
                )

            if ch1_match := re.search(
                rf"DAQ_CH1:\s*({NUMBER_PATTERN})",
                line,
                flags=re.IGNORECASE,
            ):
                self._last_ch1 = float(
                    ch1_match.group(1)
                )

            # APD voltage printed by the DAQ reader.
            if voltage_m := re.search(
                rf"DAQ_VOLTAGE:\s*({NUMBER_PATTERN})",
                line,
                flags=re.IGNORECASE,
            ):
                voltage = float(voltage_m.group(1))
                self.voltage_label.setText(
                    f"APD voltage: {voltage:.6f} V"
                )

            reference_channels = max(
                1,
                getattr(
                    self,
                    "_live_reference_channels",
                    self.refch_input.value(),
                ),
            )

            # A one-channel acquisition needs CH0.
            # Referenced acquisitions need both CH0 and CH1.
            required_attributes = (
                ("_last_ch0",)
                if reference_channels == 1
                else (
                    "_last_ch0",
                    "_last_ch1",
                )
            )

            if all(
                hasattr(self, attribute)
                for attribute in required_attributes
            ):
                ch0 = self._last_ch0
                ch1 = (
                    self._last_ch1
                    if reference_channels >= 2
                    else np.nan
                )

                averages = max(
                    1,
                    self.avg_input.value(),
                )

                number_steps = max(
                    1,
                    self.dynamic_input.value(),
                )

                ps_steps = max(
                    1,
                    getattr(
                        self,
                        "_live_ps_steps",
                        1,
                    ),
                )

                acquisition_index = (
                    self._live_measurement_index
                    // averages
                )

                ps_index = min(
                    acquisition_index
                    // number_steps,
                    ps_steps - 1,
                )

                step_index = (
                    acquisition_index
                    % number_steps
                )

                if hasattr(self, "_last_freq"):
                    self.live_freqs[
                        ps_index,
                        step_index,
                    ] = self._last_freq

                frequency = self.live_freqs[
                    ps_index,
                    step_index,
                ]

                self._live_ch0_sums[
                    ps_index,
                    step_index,
                ] += ch0

                if reference_channels >= 2:
                    self._live_ch1_sums[
                        ps_index,
                        step_index,
                    ] += ch1

                self._live_samples_per_step[
                    ps_index,
                    step_index,
                ] += 1

                sample_count = (
                    self._live_samples_per_step[
                        ps_index,
                        step_index,
                    ]
                )

                average_ch0 = (
                    self._live_ch0_sums[
                        ps_index,
                        step_index,
                    ]
                    / sample_count
                )

                if reference_channels >= 2:
                    average_ch1 = (
                        self._live_ch1_sums[
                            ps_index,
                            step_index,
                        ]
                        / sample_count
                    )
                else:
                    average_ch1 = np.nan

                if reference_channels >= 2:
                    denominator = (
                        average_ch0
                        + average_ch1
                    )

                    if (
                        np.isfinite(denominator)
                        and abs(denominator)
                        > np.finfo(float).eps
                    ):
                        normalized = (
                            average_ch0
                            - average_ch1
                        ) / denominator
                    else:
                        normalized = np.nan

                    combined = (
                        average_ch0
                        + average_ch1
                    ) / 2.0
                else:
                    normalized = np.nan
                    combined = average_ch0

                self.live_ch0[
                    ps_index,
                    step_index,
                ] = average_ch0

                self.live_ch1[
                    ps_index,
                    step_index,
                ] = average_ch1

                self.live_normalized[
                    ps_index,
                    step_index,
                ] = normalized

                self.live_combined[
                    ps_index,
                    step_index,
                ] = combined

                valid_steps = (
                    self._live_samples_per_step[
                        ps_index
                    ]
                    > 0
                )

                frequency_array = (
                    self.live_freqs[ps_index]
                )

                self.live_ch0_curve.setData(
                    frequency_array[valid_steps],
                    self.live_ch0[
                        ps_index
                    ][valid_steps],
                )

                if reference_channels >= 2:
                    self.live_ch1_curve.setData(
                        frequency_array[valid_steps],
                        self.live_ch1[
                            ps_index
                        ][valid_steps],
                    )
                else:
                    self.live_ch1_curve.setData(
                        [],
                        [],
                    )

                self.live_normalized_curve.setData(
                    frequency_array[valid_steps],
                    self.live_normalized[
                        ps_index
                    ][valid_steps],
                )

                self.live_combined_curve.setData(
                    frequency_array[valid_steps],
                    self.live_combined[
                        ps_index
                    ][valid_steps],
                )


                self.freq_label.setText(
                    f"Frequency: {frequency:.6f} GHz"
                )

                self.live_ps_label.setText(
                    "Pulse-sequence step: "
                    f"{ps_index + 1}/{ps_steps}"
                )

                self.ch0_label.setText(
                    f"CH0 mean: {average_ch0:.6f} V"
                )

                if reference_channels >= 2:
                    self.ch1_label.setText(
                        f"CH1 mean: {average_ch1:.6f} V"
                    )
                    self.normalized_label.setText(
                        f"Normalized: {normalized:.6g}"
                    )
                else:
                    self.ch1_label.setText(
                        "CH1: N/A"
                    )
                    self.normalized_label.setText(
                        "Normalized: N/A"
                    )

                self.combined_label.setText(
                    f"Combined: {combined:.6f} V"
                )

                self._live_measurement_index += 1

                for attribute in (
                    "_last_freq",
                    "_last_ch0",
                    "_last_ch1",
                ):
                    if hasattr(self, attribute):
                        delattr(
                            self,
                            attribute,
                        )

    def _stop(self):
        try:
            stop_pulse_blaster()
        except Exception as e:
            self.log_output.appendPlainText(
                f"PulseBlaster stop warning: {e}"
            )

        if self.process and self.process.state() == QtCore.QProcess.ProcessState.Running:
            # prevent on_finished() from auto‐switching to Results
            try:
                self.process.finished.disconnect(self._on_finished)
            except (TypeError, RuntimeError):
                pass

            self.process.terminate()
            self.process.waitForFinished(1000)

            if self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
                self.process.kill()
                self.process.waitForFinished(1000)

            pid = self.process.processId()
            print(f"Terminated QuPyt watcher (PID {pid})")

        else:
            print("No running process to stop.")

        # Remove YAML files and stale _running markers.
        self._clear_waiting_room()        

        self._suppress_auto_switch = False
        self.file_selector.blockSignals(False)
        self.tabs.setCurrentIndex(1)

    def _double_stop(self):
        """Stop QuPyt and send a second PulseBlaster stop command."""
        self._stop()
        QtCore.QTimer.singleShot(50, self._stop)
        
    def _open_power_supply_dialog(self):
        dlg = PowerSupplyDialog(self)
        dlg.exec()

    def _on_finished(self, *_):
        self._suppress_auto_switch = False

        # Prevent duplicate loading while rebuilding the dropdown.
        self.file_selector.blockSignals(True)

        self._populate_file_selector()
        filename = ""
        
        if self.file_selector.count():
            self.file_selector.setCurrentIndex(self.file_selector.count() - 1)
            filename = self.file_selector.currentText()

        self.file_selector.blockSignals(False)

        if filename:
            self._on_file_selected(filename)

        self.tabs.setCurrentIndex(2)

    def _on_view_data(self):
        # compute full summary and show in a dialog
        data = getattr(self, "data", None)
        path = getattr(self, "current_file", None)
        if data is None or path is None:
            QMessageBox.information(self, "No data", "No data file loaded.")
            return

        stats = {
            "Loaded file": path.name,
            "Shape": data.shape,
            "Dtype": data.dtype,
            "Total elements": data.size,
            "Memory (bytes)": data.nbytes,
            "Min": data.min(),
            "Max": data.max(),
            "Mean": data.mean(),
            "Median": np.median(data),
            "Std dev": data.std(),
            "Zero count": int(np.count_nonzero(data == 0)),
            "NaN count": int(np.count_nonzero(np.isnan(data))),
            "Inf count": int(np.count_nonzero(np.isinf(data))),
            "Unique values": int(np.unique(data).size),
        }
        msg = "\n".join(f"{k}: {v}" for k, v in stats.items())
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Raw Data Summary")
        dlg.setText(msg)
        dlg.exec()

    def _smooth_result_spectrum(self, values):
        values = np.asarray(values, dtype=float)
        requested_window = int(
            self.smooth_input.value()
        )

        if requested_window < 3 or values.size < 3:
            return values.copy()

        maximum_window = (
            values.size
            if values.size % 2 == 1
            else values.size - 1
        )

        window = min(
            requested_window,
            maximum_window,
        )

        if window % 2 == 0:
            window -= 1

        if window < 3:
            return values.copy()

        finite = np.isfinite(values)

        if np.count_nonzero(finite) < window:
            return values.copy()

        indices = np.arange(values.size)
        filled_values = np.interp(
            indices,
            indices[finite],
            values[finite],
        )

        smoothed = savgol_filter(
            filled_values,
            window_length=window,
            polyorder=min(2, window - 1),
        )

        smoothed[~finite] = np.nan
        return smoothed
    
    def _fit_result_spectrum(
        self,
        x_hz,
        y,
        model,
        table,
        plot_widget,
        normalized=False,
    ):
        """
        Fit one displayed spectrum and populate its parameter table.
        """
        center = np.nan
        fwhm_mhz = np.nan
        contrast = np.nan
        r2 = 0.0

        try:
            x_hz = np.asarray(x_hz, dtype=float)
            y = np.asarray(y, dtype=float)

            valid = np.isfinite(x_hz) & np.isfinite(y)
            x_valid = x_hz[valid]
            y_valid = y[valid]

            if x_valid.size < 4:
                raise ValueError(
                    "Not enough finite points for fitting"
                )

            span = x_valid.max() - x_valid.min()
            signal_range = np.ptp(y_valid)

            if not np.isfinite(span) or span <= 0:
                raise ValueError(
                    "Frequency sweep has zero or invalid span"
                )

            if (
                not np.isfinite(signal_range)
                or signal_range <= 0
            ):
                raise ValueError(
                    "Spectrum is flat and cannot be fitted"
                )

            # ODMR dip: initial center is the minimum measured point.
            center_guess = x_valid[np.argmin(y_valid)]
            baseline_guess = np.nanmedian(y_valid)
            amplitude_guess = (
                y_valid.min() - baseline_guess
            )

            minimum_width = max(
                span * 1e-9,
                np.finfo(float).eps
            )
            width_guess = max(
                span / 20,
                minimum_width * 10
            )

            initial_parameters = [
                center_guess,
                width_guess,
                amplitude_guess,
                baseline_guess,
            ]

            bounds = (
                [
                    x_valid.min(),
                    minimum_width,
                    -np.inf,
                    -np.inf,
                ],
                [
                    x_valid.max(),
                    span,
                    0.0,
                    np.inf,
                ],
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    OptimizeWarning
                )

                fitted_parameters, _ = curve_fit(
                    model,
                    x_valid,
                    y_valid,
                    p0=initial_parameters,
                    bounds=bounds,
                    maxfev=20_000,
                )

            if model is lorentzian:
                x0, gamma, amplitude, baseline = (
                    fitted_parameters
                )
                fwhm_hz = 2 * gamma
            else:
                x0, sigma, amplitude, baseline = (
                    fitted_parameters
                )
                fwhm_hz = 2.355 * sigma

            fitted_y = model(
                x_valid,
                *fitted_parameters
            )

            residuals = y_valid - fitted_y
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum(
                (y_valid - np.mean(y_valid)) ** 2
            )

            r2 = (
                1 - ss_res / ss_tot
                if ss_tot != 0
                else np.nan
            )

            center = x0 / 1e9
            fwhm_mhz = abs(fwhm_hz) / 1e6

            if normalized:
                contrast = 100 * abs(amplitude)
            else:
                contrast = (
                    100 * abs(amplitude) / abs(baseline)
                    if baseline != 0
                    else np.nan
                )

            # Draw the fitted curve over the measured points.
            order = np.argsort(x_valid)
            plot_widget.plot(
                x_valid[order] / 1e9,
                fitted_y[order],
                pen=pg.mkPen('y', width=2),
            )

        except Exception as error:
            logging.warning(
                "Spectrum fit failed: %s",
                error
            )

        displayed_values = [
            f"{center:.4f}",
            f"{fwhm_mhz:.2f}",
            f"{contrast:.1f}",
            f"{r2:.3f}",
        ]

        for row, value in enumerate(displayed_values):
            table.setItem(
                row,
                1,
                QTableWidgetItem(value)
            )

    def _show_results(self, *_):
        if not hasattr(self, "data") or self.data is None:
            return

        config, metadata_path = (
            self._load_result_configuration()
        )

        canonical_data = (
            self._canonical_result_data(
                self.data,
                config,
            )
        )

        try:
            analysis_data = (
                self._scale_result_data_for_analysis(
                    canonical_data,
                    config,
                )
            )
        except ValueError as error:
            QMessageBox.warning(
                self,
                "Result scaling error",
                str(error),
            )
            return

        if canonical_data.ndim < 3:
            QMessageBox.warning(
                self,
                "Invalid result shape",
                (
                    "Expected at least channel, pulse-sequence, "
                    f"and dynamic axes. Received {canonical_data.shape}."
                ),
            )
            return

        channel_count = canonical_data.shape[0]
        ps_steps = canonical_data.shape[1]
        dynamic_steps = canonical_data.shape[2]

        self.result_ps_step.blockSignals(True)
        self.result_ps_step.setRange(
            0,
            max(0, ps_steps - 1),
        )
        self.result_ps_step.setValue(
            min(
                self.result_ps_step.value(),
                ps_steps - 1,
            )
        )
        self.result_ps_step.blockSignals(False)

        ps_index = self.result_ps_step.value()

        frequencies_hz = (
            self._frequency_axis_from_config(
                config,
                dynamic_steps,
            )
        )
        frequencies_ghz = (
            frequencies_hz / 1e9
        )

        (
            ps_parameter,
            ps_unit,
            ps_values,
        ) = self._pulse_sequence_axis_from_config(
            config,
            ps_steps,
        )

        trailing_axes = tuple(
            range(3, canonical_data.ndim)
        )

        if trailing_axes:
            channel_means_all = np.nanmean(
                analysis_data,
                axis=trailing_axes,
            )
            channel_stds_all = np.nanstd(
                analysis_data,
                axis=trailing_axes,
            )
        else:
            channel_means_all = np.asarray(
                analysis_data,
                dtype=float,
            )
            channel_stds_all = np.zeros_like(
                channel_means_all,
                dtype=float,
            )

        channel_means = channel_means_all[
            :,
            ps_index,
            :,
        ]
        channel_stds = channel_stds_all[
            :,
            ps_index,
            :,
        ]

        if channel_count >= 2:
            # Normalize corresponding CH0/CH1 measurements before
            # averaging the frame and sensor dimensions.
            #
            # Shape after selecting the channel:
            #   [pulse_sequence_step, dynamic_step, frame, sensor...]
            ch0_pairwise = np.asarray(
                analysis_data[0],
                dtype=float,
            )
            ch1_pairwise = np.asarray(
                analysis_data[1],
                dtype=float,
            )

            pairwise_denominator = (
                ch0_pairwise
                + ch1_pairwise
            )

            finite_denominator = np.abs(
                pairwise_denominator[
                    np.isfinite(pairwise_denominator)
                ]
            )

            denominator_scale = (
                np.nanmedian(finite_denominator)
                if finite_denominator.size
                else 1.0
            )

            minimum_denominator = max(
                denominator_scale * 1e-9,
                np.finfo(float).eps,
            )

            with np.errstate(
                divide="ignore",
                invalid="ignore",
            ):
                pairwise_normalized = np.where(
                    np.abs(pairwise_denominator)
                    > minimum_denominator,
                    (
                        ch0_pairwise
                        - ch1_pairwise
                    )
                    / pairwise_denominator,
                    np.nan,
                )

            # Preserve axes 0 and 1:
            #   axis 0 = pulse-sequence step
            #   axis 1 = dynamic/frequency step
            #
            # Average only frame and sensor dimensions.
            pairwise_average_axes = tuple(
                range(
                    2,
                    pairwise_normalized.ndim,
                )
            )

            if pairwise_average_axes:
                normalized_all = np.nanmean(
                    pairwise_normalized,
                    axis=pairwise_average_axes,
                )
            else:
                normalized_all = (
                    pairwise_normalized
                )

            combined_all = (
                channel_means_all[0]
                + channel_means_all[1]
            ) / 2.0
        else:
            normalized_all = channel_means_all[0]
            combined_all = channel_means_all[0]

        using_normalized_difference = (
            self.sub_input.isChecked()
            and channel_count >= 2
        )

        if using_normalized_difference:
            processed_all = normalized_all
        else:
            processed_all = channel_means_all[0]

        selected_y = self._smooth_result_spectrum(
            processed_all[ps_index]
        )
        combined_y = self._smooth_result_spectrum(
            combined_all[ps_index]
        )

        self.result_canonical_data = canonical_data
        self.result_analysis_data = analysis_data
        self.result_config = config
        self.result_freqs_hz = frequencies_hz
        self.result_freqs_ghz = frequencies_ghz
        self.result_ps_parameter = ps_parameter
        self.result_ps_unit = ps_unit
        self.result_ps_values = ps_values
        self.result_channel_means_all = channel_means_all
        self.result_channel_stds_all = channel_stds_all
        self.result_channel_means = channel_means
        self.result_channel_stds = channel_stds
        self.result_processed_all = processed_all
        self.result_combined_all = combined_all
        self.result_selected_spectrum = selected_y
        self.result_combined_spectrum = combined_y

        selected_value_text = (
            f"{ps_parameter} = "
            f"{ps_values[ps_index]:.8g}"
        )

        if ps_unit:
            selected_value_text += f" {ps_unit}"

        self.result_ps_value_label.setText(
            selected_value_text
        )

        metadata_source = (
            str(metadata_path)
            if metadata_path is not None
            else "Current GUI values"
        )

        timestamp = (
            QtCore.QDateTime.currentDateTime()
            .toString()
        )

        self.meta_text.setPlainText(
            f"Time: {timestamp}\n"
            f"Metadata: {metadata_source}\n"
            f"Raw shape: {self.data.shape}\n"
            f"Canonical shape: {canonical_data.shape}\n"
            f"Dynamic steps: {dynamic_steps}\n"
            f"Pulse-sequence steps: {ps_steps}\n"
            f"Selected pulse-sequence step: "
            f"{ps_index + 1}/{ps_steps}\n"
            f"{selected_value_text}\n"
            f"Averages: {config.get('averages', '--')}"
        )
   
        self.proc_plot.clear()
        self.proc_plot.setLabel(
            'bottom',
            'Frequency (GHz)'
        )

        if using_normalized_difference:
            self.proc_box.setTitle(
                'Processed ODMR Spectrum — '
                '(CH0 - CH1) / (CH0 + CH1)'
            )

            self.proc_plot.setLabel(
                'left',
                '(CH0 - CH1) / (CH0 + CH1)'
            )
            self.proc_plot.plot(
                frequencies_ghz,
                selected_y,
                pen='r',
                symbol='x'
            )
        else:
            self.proc_box.setTitle(
                'Processed ODMR Spectrum — CH0'
            )

            self.proc_plot.setLabel(
                'left',
                'CH0 mean signal'
            )
            self.proc_plot.plot(
                frequencies_ghz,
                selected_y,
                pen='b',
                symbol='o'
            )

        # ------------------------------------------------------------
        # Plot 2: combined-channel mean
        # ------------------------------------------------------------
        self.mean_proc_plot.clear()
        self.mean_proc_plot.setLabel(
            'bottom',
            'Frequency (GHz)'
        )
        self.mean_proc_plot.setLabel(
            'left',
            '(CH0 + CH1) / 2'
        )


        self.mean_proc_plot.plot(
            frequencies_ghz,
            combined_y,
            pen='b',
            symbol='o'
        )


        # ------------------------------------------------------------
        # Fit both plots independently
        # ------------------------------------------------------------
        model = (
            lorentzian
            if self.fit_input.currentText() == 'Lorentzian'
            else gaussian
        )

        self._fit_result_spectrum(
            x_hz=frequencies_hz,
            y=selected_y,
            model=model,
            table=self.fit_table,
            plot_widget=self.proc_plot,
            normalized=using_normalized_difference,
        )

        self._fit_result_spectrum(
            x_hz=frequencies_hz,
            y=combined_y,
            model=model,
            table=self.mean_fit_table,
            plot_widget=self.mean_proc_plot,
            normalized=False,
        )

        # update the little summary label
        self._draw_result_heatmap()
        self._update_results_view_visibility()

        d = self.data
        summary = (
            f"{d.shape}, dtype={d.dtype}, "
            f"min={np.nanmin(d):.3g}, "
            f"max={np.nanmax(d):.3g}, "
            f"PS step={ps_index + 1}/{ps_steps}"
        )
        self.summary_label.setText(summary)

    def _export(self):
        """Export the result file currently selected in the Results tab."""
        data = getattr(self, "data", None)

        if data is None:
            QMessageBox.warning(
                self,
                "Export",
                "No result file is currently loaded."
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export data",
            "",
            "NumPy (*.npy);;CSV (*.csv);;PNG (*.png)"
        )

        if not path:
            return

        path = Path(path)

        if "NumPy" in selected_filter:
            path = path.with_suffix(".npy")
            np.save(path, data)

        elif "CSV" in selected_filter:
            path = path.with_suffix(".csv")

            if not hasattr(
                self,
                "result_channel_means_all",
            ):
                self._show_results()

            channel_means = (
                self.result_channel_means_all
            )
            processed = self.result_processed_all
            combined = self.result_combined_all
            frequencies = self.result_freqs_ghz
            ps_values = self.result_ps_values

            rows = []

            for ps_index in range(
                channel_means.shape[1]
            ):
                for dynamic_index in range(
                    channel_means.shape[2]
                ):
                    row = [
                        ps_index,
                        ps_values[ps_index],
                        frequencies[dynamic_index],
                    ]

                    row.extend(
                        channel_means[
                            :,
                            ps_index,
                            dynamic_index,
                        ].tolist()
                    )

                    row.extend(
                        [
                            processed[
                                ps_index,
                                dynamic_index,
                            ],
                            combined[
                                ps_index,
                                dynamic_index,
                            ],
                        ]
                    )

                    rows.append(row)

            parameter_column = re.sub(
                r"[^A-Za-z0-9_]+",
                "_",
                self.result_ps_parameter,
            ).strip("_")

            if not parameter_column:
                parameter_column = (
                    "pulse_sequence_value"
                )

            if self.result_ps_unit:
                unit_column = re.sub(
                    r"[^A-Za-z0-9_]+",
                    "_",
                    self.result_ps_unit,
                ).strip("_")

                if unit_column:
                    parameter_column += (
                        f"_{unit_column}"
                    )

            header_fields = [
                "pulse_sequence_step",
                parameter_column,
                "frequency_GHz",
            ]

            header_fields.extend(
                f"channel_{index}"
                for index in range(
                    channel_means.shape[0]
                )
            )

            header_fields.extend(
                [
                    "processed_top_spectrum",
                    "combined_channel_mean",
                ]
            )

            np.savetxt(
                path,
                np.asarray(rows, dtype=float),
                delimiter=",",
                header=",".join(header_fields),
                comments=""
            )

        else:
            path = path.with_suffix(".png")

            if (
                self.result_view_combo.currentText()
                == "2D heatmap"
            ):
                matrix = (
                    self._selected_heatmap_matrix()
                )

                figure, axis = plt.subplots(
                    figsize=(9, 6)
                )

                image = axis.imshow(
                    matrix,
                    origin="lower",
                    aspect="auto",
                    extent=[
                        self.result_freqs_ghz[0],
                        self.result_freqs_ghz[-1],
                        -0.5,
                        matrix.shape[0] - 0.5,
                    ],
                )

                tick_count = min(
                    12,
                    len(self.result_ps_values),
                )
                tick_indices = np.unique(
                    np.linspace(
                        0,
                        len(self.result_ps_values) - 1,
                        tick_count,
                        dtype=int,
                    )
                )

                axis.set_yticks(tick_indices)
                axis.set_yticklabels(
                    [
                        f"{self.result_ps_values[index]:.6g}"
                        for index in tick_indices
                    ]
                )

                y_label = self.result_ps_parameter

                if self.result_ps_unit:
                    y_label += (
                        f" ({self.result_ps_unit})"
                    )

                axis.set_xlabel("Frequency (GHz)")
                axis.set_ylabel(y_label)
                axis.set_title(
                    self.heatmap_signal_combo.currentText()
                )
                figure.colorbar(
                    image,
                    ax=axis,
                    label="Detector signal",
                )
            else:
                figure, axis = plt.subplots(
                    figsize=(8, 5)
                )

                channel_means = (
                    self.result_channel_means
                )

                for channel_index in range(
                    channel_means.shape[0]
                ):
                    axis.plot(
                        self.result_freqs_ghz,
                        channel_means[channel_index],
                        "o-",
                        label=f"Channel {channel_index}",
                    )

                axis.set_xlabel("Frequency (GHz)")
                axis.set_ylabel(
                    "Mean detector signal"
                )
                axis.legend()
                axis.grid(True, alpha=0.3)
    
            figure.tight_layout()
            figure.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(figure)

        QMessageBox.information(
            self,
            "Export complete",
            f"Exported data to:\n{path}"
        )
      

    def _save_plot(self, widget):
        """Save the displayed plot as PNG or vector SVG."""
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG (*.png);;SVG (*.svg)"
        )
        if not path:
            return

        path = Path(path)

        if "SVG" in selected_filter:
            path = path.with_suffix(".svg")
            exporter = pyqtgraph.exporters.SVGExporter(
                widget.plotItem
            )
            exporter.export(str(path))
        else:
            path = path.with_suffix(".png")

            if not widget.grab().save(str(path)):
                QMessageBox.warning(
                    self,
                    "Save failed",
                    f"Could not save plot to:\n{path}"
                )

    def _save_channel_plots(self):
        """
        Save every stored reference channel as an independent PNG plot.

        Expected data layout:
            [reference_channel, pulse_sequence_step,
             dynamic_step, frames, sensor_dimensions...]
        """
        data = getattr(self, "data", None)

        if data is None:
            QMessageBox.information(
                self,
                "No data",
                "Load a result file before saving channel plots."
            )
            return

        if data.ndim < 2:
            QMessageBox.warning(
                self,
                "Invalid data shape",
                f"Expected at least two dimensions, but received {data.shape}."
            )
            return

        canonical_data = getattr(
            self,
            "result_canonical_data",
            None,
        )

        if canonical_data is None:
            self._show_results()
            canonical_data = getattr(
                self,
                "result_canonical_data",
                None,
            )

        if canonical_data is None:
            QMessageBox.warning(
                self,
                "Result processing error",
                "Could not interpret the loaded result shape.",
            )
            return

        channel_count = canonical_data.shape[0]

        if channel_count < 2:
            QMessageBox.warning(
                self,
                "Only one stored channel",
                (
                    f"The loaded array has shape {data.shape}, so it contains "
                    "only one stored reference channel.\n\n"
                    "Run the experiment with Ref channels = 2 to save two "
                    "separate channel plots."
                )
            )
            return

        # Use the same values calculated in _show_results().
        channel_means = getattr(self, "result_channel_means", None)
        channel_stds = getattr(self, "result_channel_stds", None)
        freqs_ghz = getattr(self, "result_freqs_ghz", None)

        if channel_means is None or channel_stds is None:
            QMessageBox.warning(
                self,
                "Result processing error",
                "Channel statistics are unavailable.",
            )
            return

        if (
            freqs_ghz is None
            or len(freqs_ghz)
            != channel_means.shape[1]
        ):
            freqs_ghz = np.linspace(
                self.start_input.value(),
                self.stop_input.value(),
                channel_means.shape[1],
            )

        current_file = getattr(self, "current_file", None)
        default_stem = (
            current_file.stem
            if current_file is not None
            else self.exp_combo.currentText()
        )

        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Separate Channel Plots",
            f"{default_stem}_channels.png",
            "PNG (*.png)"
        )

        if not selected_path:
            return

        base_path = Path(selected_path)
        if base_path.suffix.lower() != ".png":
            base_path = base_path.with_suffix(".png")

        saved_paths = []

        for channel_index in range(channel_count):
            output_path = base_path.with_name(
                f"{base_path.stem}_channel_{channel_index}{base_path.suffix}"
            )

            figure, axis = plt.subplots(figsize=(8, 5))

            if self.errb_input.isChecked():
                axis.errorbar(
                    freqs_ghz,
                    channel_means[channel_index],
                    yerr=channel_stds[channel_index],
                    fmt="o-",
                    capsize=3
                )
            else:
                axis.plot(
                    freqs_ghz,
                    channel_means[channel_index],
                    "o-"
                )

            axis.set_xlabel("Frequency (GHz)")
            axis.set_ylabel("Mean detector signal")
            axis.set_title(
                f"{default_stem} — "
                f"PS step {self.result_ps_step.value() + 1} — "
                f"Reference channel {channel_index}"
            )
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(figure)

            saved_paths.append(str(output_path))

        QMessageBox.information(
            self,
            "Channel Plots Saved",
            "Saved separate channel plots:\n\n" + "\n".join(saved_paths)
        )

    def _save_config(self):
        """Save current setup parameters to a CSV file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "CSV (*.csv)")
        if not path:
            return
        fields = [
            'time_unit', 'sweep_start','sweep_stop','power',
            'averages','frames','dynamic_steps',
            'mode','ref_channels',
            'mw_duration','read_time','laser_time','max_rate',
            'mw_device_type','mw_output',
            'ps_sweep_enabled',
            'ps_sweep_parameter',
            'ps_sweep_mode',
            'ps_sweep_steps',
            'ps_sweep_start',
            'ps_sweep_stop',
            'ps_sweep_values',
        ]
        values = [
            self.unit_combo.currentText(),
            self.start_input.value(),
            self.stop_input.value(),
            self.power_input.value(),
            self.avg_input.value(),
            self.frames_input.value(),
            self.dynamic_input.value(),
            self.mode_input.currentText(),
            self.refch_input.value(),
            self.mw_dur.value(),
            self.read_dur.value(),
            self.las_dur.value(),
            self.rate.value(),
            self.mw_dev_combo.currentText(),
            self.mw_out_combo.currentText(),
            int(self.ps_sweep_enable.isChecked()),
            self.ps_sweep_param.currentData() or "",
            self.ps_sweep_mode.currentText(),
            self.ps_steps_input.value(),
            self.ps_start_input.value(),
            self.ps_stop_input.value(),
            self.ps_values_input.text(),
        ]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerow(values)
        QMessageBox.information(self, "Saved", f"Configuration saved to:\n{path}")

    def _load_config(self):
        """Load setup parameters from a CSV file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "CSV (*.csv)")

        if not path:
            return
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader, None)

        if not row:
            QMessageBox.warning(self, "Error", "No data found in the file.")
            return
        
        # Apply loaded values
        try:
            self.unit_combo.setCurrentText(
                row.get('time_unit', 'µs')
            )
            self.start_input.setValue(float(row['sweep_start']))
            self.stop_input .setValue(float(row['sweep_stop']))
            self.power_input.setValue(float(row['power']))
            self.avg_input  .setValue(int  (row['averages']))
            self.frames_input.setValue(int  (row['frames']))
            self.dynamic_input.setValue(int(row['dynamic_steps']))
            self.mode_input .setCurrentText(row['mode'])
            self.refch_input.setValue(int  (row['ref_channels']))
            self.mw_dur     .setValue(float(row['mw_duration']))
            self.read_dur   .setValue(float(row['read_time']))
            self.las_dur    .setValue(float(row['laser_time']))
            self.rate       .setValue(int  (row['max_rate']))
            self.mw_dev_combo.setCurrentText(
                row.get(
                    'mw_device_type',
                    'WindFreak'
                )
            )
            self.mw_out_combo.setCurrentText(
                row.get('mw_output', 'B')
            )

            ps_parameter = row.get(
                "ps_sweep_parameter",
                "",
            )

            ps_index = self.ps_sweep_param.findData(
                ps_parameter
            )

            if ps_index >= 0:
                self.ps_sweep_param.setCurrentIndex(
                    ps_index
                )

            self.ps_sweep_mode.setCurrentText(
                row.get(
                    "ps_sweep_mode",
                    "Start / Stop",
                )
            )
            self.ps_steps_input.setValue(
                int(
                    row.get(
                        "ps_sweep_steps",
                        1,
                    )
                )
            )
            self.ps_start_input.setValue(
                float(
                    row.get(
                        "ps_sweep_start",
                        0,
                    )
                )
            )
            self.ps_stop_input.setValue(
                float(
                    row.get(
                        "ps_sweep_stop",
                        0,
                    )
                )
            )
            self.ps_values_input.setText(
                row.get(
                    "ps_sweep_values",
                    "",
                )
            )
            self.ps_sweep_enable.setChecked(
                str(
                    row.get(
                        "ps_sweep_enabled",
                        "0",
                    )
                ).lower()
                in ("1", "true", "yes")
            )
            self._update_ps_sweep_controls()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not parse configuration:\n{e}")

            return 
        # on successful parse & apply, let the user know
        QMessageBox.information(self, "Loaded", f"Configuration loaded from:\n{path}")

    def _load_defaults(self):
        """Restore all defaults for the selected experiment."""
        self._apply_descriptor_defaults(
            self.exp_combo.currentText()
        )

        self.ps_sweep_enable.setChecked(False)
        self.ps_steps_input.setValue(1)
        self.ps_values_input.clear()
        self.ps_preview_step.setValue(0)

    def _restore_last_config(self):
        """Load last‐used JSON snapshot if present."""
        try:
            with open(LAST_CFG_PATH, 'r') as f:
                cfg = json.load(f)
        except FileNotFoundError:
            return
        # apply values back into the widgets
        try:
            self.unit_combo.setCurrentText(
                cfg.get('time_unit', 'µs')
            )

            self.start_input .setValue(cfg['sweep_start'])
            self.stop_input  .setValue(cfg['sweep_stop'])
            self.power_input .setValue(cfg['power'])
            self.avg_input   .setValue(cfg['averages'])
            self.frames_input.setValue(cfg['frames'])
            self.dynamic_input.setValue(cfg['dynamic_steps'])
            self.mode_input  .setCurrentText(cfg['mode'])
            self.refch_input .setValue(cfg['ref_channels'])
            self.mw_dur      .setValue(cfg['mw_duration'])
            self.read_dur    .setValue(cfg['read_time'])
            self.las_dur     .setValue(cfg['laser_time'])
            self.rate        .setValue(cfg['max_rate'])
            self.mw_dev_combo.setCurrentText(
                cfg.get(
                    'mw_device_type',
                    'WindFreak'
                )
            )
            self.mw_out_combo.setCurrentText(
                cfg.get('mw_output', 'B')
            )

            parameter_name = cfg.get(
                "ps_sweep_parameter"
            )

            parameter_index = (
                self.ps_sweep_param.findData(
                    parameter_name
                )
            )

            if parameter_index >= 0:
                self.ps_sweep_param.setCurrentIndex(
                    parameter_index
                )

            self.ps_sweep_mode.setCurrentText(
                cfg.get(
                    "ps_sweep_mode",
                    "Start / Stop",
                )
            )
            self.ps_steps_input.setValue(
                int(
                    cfg.get(
                        "ps_sweep_steps",
                        1,
                    )
                )
            )
            self.ps_start_input.setValue(
                float(
                    cfg.get(
                        "ps_sweep_start",
                        0,
                    )
                )
            )
            self.ps_stop_input.setValue(
                float(
                    cfg.get(
                        "ps_sweep_stop",
                        0,
                    )
                )
            )
            self.ps_values_input.setText(
                cfg.get(
                    "ps_sweep_values",
                    "",
                )
            )
            self.ps_sweep_enable.setChecked(
                bool(
                    cfg.get(
                        "ps_sweep_enabled",
                        False,
                    )
                )
            )
            self._update_ps_sweep_controls()
            
        except KeyError:
            # silently skip if schema mismatch
            pass

    def _write_last_config(self):
        """Dump current setup into a JSON file for ‘last used’ recall."""
        cfg = {
            'time_unit':      self.unit_combo.currentText(),
            'sweep_start':   self.start_input.value(),
            'sweep_stop':    self.stop_input.value(),
            'power':         self.power_input.value(),
            'averages':      self.avg_input.value(),
            'frames':        self.frames_input.value(),
            'dynamic_steps': self.dynamic_input.value(),
            'mode':          self.mode_input.currentText(),
            'ref_channels':  self.refch_input.value(),
            'mw_duration':   self.mw_dur.value(),
            'read_time':     self.read_dur.value(),
            'laser_time':    self.las_dur.value(),
            'max_rate':      self.rate.value(),
            'mw_device_type': self.mw_dev_combo.currentText(),
            'mw_output':     self.mw_out_combo.currentText(),
            'ps_sweep_enabled': self.ps_sweep_enable.isChecked(),
            'ps_sweep_parameter': self.ps_sweep_param.currentData(),
            'ps_sweep_mode': self.ps_sweep_mode.currentText(),
            'ps_sweep_steps': self.ps_steps_input.value(),
            'ps_sweep_start': self.ps_start_input.value(),
            'ps_sweep_stop': self.ps_stop_input.value(),
            'ps_sweep_values': self.ps_values_input.text(),
        }
        LAST_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LAST_CFG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)

    def _init_pulse_diagram(self):
        # make the PlotWidget
        self.pulse_plot = pg.PlotWidget(title="Pulse Diagram")
        self.pulse_plot.setLabel('bottom', 'Time')
        self.pulse_plot.getViewBox().invertY(True)   # so lane 0 is at top

        # Preserve useful vertical space for the pulse diagram.
        self.pulse_plot.setMinimumHeight(170)

        # style: white background with a thin black border
        self.pulse_plot.setBackground('w')
        self.pulse_plot.setStyleSheet("border:1px solid black;")

        # define channel→lane mapping
        self.channel_lanes = {
            'LASER':  0,
            'MW':     1,
            'READ':   2,
            'START':  3,
            'I':      4,
            'Q':      5,
        }

        # define colors per channel
        self.channel_colors = {
            'LASER': (255,  50,  50, 200),
            'MW':    (50,  50, 255, 200),
            'READ':  (50, 255,  50, 200),
            'START': (200, 100,  0, 200),
            'I':     (255,  50,255, 200),
            'Q':     (50, 255, 255, 200),
        }

        # Add the pulse diagram below the two-column setup region,
        # spanning the entire width of the Setup tab.
        self.pulse_section_widget = QtWidgets.QWidget()
        pulse_section_form = QFormLayout(
            self.pulse_section_widget
        )
        pulse_section_form.setContentsMargins(
            0,
            0,
            0,
            0,
        )
        pulse_section_form.addRow(
            'Pulse diagram:',
            self.pulse_plot,
        )

        self.setup_root_layout.addWidget(
            self.pulse_section_widget
        )

        #  hook redraw to *all* pulse-timing spin-boxes
        for sb in (
            self.las_dur, self.mw_dur,
            self.read_dur, self.I_pulse_dur, self.Q_pulse_dur,
            self.tau_input, self.blocks_input, self.start_pulse_dur
        ):
            sb.valueChanged.connect(self._update_pulse_diagram)

        # initial draw
        self._update_pulse_diagram()

    def _update_pulse_diagram(self, *_):
        # This method can be reached while _build_ui() is still creating
        # the pulse-sequence sweep controls.
        if not hasattr(self, "pulse_plot"):
            return
        # clear old bars
        try:
            # clear old bars
            self.pulse_plot.clear()
        except Exception as e:
            QMessageBox.warning(self, "Pulse Diagram Error", str(e))
            return

        # build a list of every pulse event
        pulses = self._get_all_pulses()

        total_time = max((s + d) for (_c, s, d) in pulses) if pulses else 1.0

        # 3) draw baseline *only* in the gaps between pulses
        for chan, lane in self.channel_lanes.items():
            colour = self.channel_colors[chan]
            pen    = pg.mkPen(colour, width=1)

            # gather this channel's pulse intervals
            intervals = sorted((s, s + d) for (c, s, d) in pulses if c == chan)

            # merge any overlapping intervals
            merged = []
            for s, e in intervals:
                if not merged or s > merged[-1][1]:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)

            # now draw the baseline in the "gaps" between merged pulses
            start0 = 0.0
            for s, e in merged:
                if start0 < s:
                    self.pulse_plot.plot([start0, s], [lane, lane], pen=pen)
                start0 = e
            # final tail after last pulse
            if start0 < total_time:
                self.pulse_plot.plot([start0, total_time], [lane, lane], pen=pen)

        # 4) draw each pulse as an up–over–down “box”
        pulse_h = 0.8  # how tall above the baseline the pulse goes
        for chan, start, dur in pulses:
            lane   = self.channel_lanes[chan]
            colour = self.channel_colors[chan]
            pen    = pg.mkPen(colour, width=2)

            x0, x1 = start, start + dur
            y0, y1 = lane, lane - pulse_h

            # vertical rising edge
            self.pulse_plot.plot([x0, x0], [y0, y1], pen=pen)
            # horizontal top
            self.pulse_plot.plot([x0, x1], [y1, y1], pen=pen)
            # vertical falling edge
            self.pulse_plot.plot([x1, x1], [y1, y0], pen=pen)

        # 5) relabel Y axis
        ticks = [(v, k) for k, v in self.channel_lanes.items()]
        self.pulse_plot.getAxis('left').setTicks([ticks])

        # 6) auto-scale nicely
        self.pulse_plot.setXRange(0, total_time * 1.05)
        max_lane = max(self.channel_lanes.values())
        self.pulse_plot.setYRange(-0.5, max_lane + 0.5)


    def _get_all_pulses(self):
        exp_type = self.exp_combo.currentText()

        if not exp_type:
            return []

        desc_path = self.experiments_dir / f"{exp_type}.yaml"

        if not desc_path.exists():
            return []

        try:
            desc = yaml.safe_load(desc_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        ctx = {}

        known_timing_names = {
            "mw_duration",
            "read_time",
            "laserduration",
            "start_pulse_dur",
            "I_pulse",
            "Q_pulse",
            "tau",
        }

        # Only these widgets are converted by the global unit selector.
        # Other descriptor widgets retain their own configured units.
        timing_names = known_timing_names

        # GUI values are in the selected display unit.
        # Convert timing parameters to internal µs before Jinja evaluation.
        for name, widget in self.param_widgets.items():
            if hasattr(widget, "value"):
                value = widget.value()

                if name in timing_names:
                    value *= self.time_factor

                ctx[name] = value
            else:
                ctx[name] = widget.currentText()

        # Apply the pulse-sequence sweep value selected by Preview step.
        if (
            hasattr(self, "ps_sweep_enable")
            and self.ps_sweep_enable.isChecked()
        ):
            try:
                sweep = self._collect_ps_sweep_config()

                preview_index = min(
                    self.ps_preview_step.value(),
                    sweep["steps"] - 1,
                )

                ctx[sweep["parameter"]] = (
                    sweep["values"][preview_index]
                )
            except ValueError:
                # An incomplete explicit list should not crash
                # the ordinary pulse preview.
                pass

        # Descriptor constants are assumed to use internal µs.
        ctx["constants"] = {}

        for key, value in desc.get("constants", {}).items():
            try:
                converted = float(value)
            except (TypeError, ValueError):
                converted = value

            ctx["constants"][key] = converted
            ctx[key] = converted

        # generic_generator.py always creates START separately and ignores
        # descriptor START entries. Mirror that behavior in the preview.
        start_duration_us = float(
            ctx.get("start_pulse_dur", 1.0)
        )

        pulses = [
            (
                "START",
                0.0,
                start_duration_us / self.time_factor,
            )
        ]

        for pulse in desc.get("pulses", []):
            channel = pulse.get("channel")

            if (
                channel == "START"
                or channel not in self.channel_lanes
            ):
                continue

            try:
                start_us = float(
                    Template(str(pulse["start"])).render(ctx)
                )
                duration_us = float(
                    Template(str(pulse["duration"])).render(ctx)
                )
            except (KeyError, TypeError, ValueError):
                continue

            # Convert internal µs back to the selected display unit.
            start_display = start_us / self.time_factor
            duration_display = duration_us / self.time_factor

            pulses.append(
                (channel, start_display, duration_display)
            )

        return pulses
            
    
    def _update_time_units(self, unit):
        new_factor = {
            'ns': 1e-3,
            'µs': 1.0,
            'ms': 1e3,
        }[unit]

        old_factor = getattr(self, "time_factor", 1.0)

        if hasattr(self, "ps_sweep_param"):
            parameter_name = (
                self.ps_sweep_param.currentData()
            )

            if parameter_name in TIMING_PARAMETER_NAMES:
                for widget in (
                    self.ps_start_input,
                    self.ps_stop_input,
                ):
                    value_us = (
                        widget.value()
                        * old_factor
                    )

                    widget.blockSignals(True)
                    widget.setValue(
                        value_us / new_factor
                    )
                    widget.setSuffix(f" {unit}")
                    widget.blockSignals(False)

        timing_widgets = (
            self.las_dur,
            self.mw_dur,
            self.read_dur,
            self.I_pulse_dur,
            self.Q_pulse_dur,
            self.tau_input,
            self.start_pulse_dur,
        )

        for sb in timing_widgets:
            # Convert old displayed value to internal µs.
            value_us = sb.value() * old_factor

            # Convert internal µs to the new display unit.
            new_display_value = value_us / new_factor

            sb.blockSignals(True)
            sb.setRange(0.0, 1e6 / new_factor)
            sb.setDecimals(0 if unit == 'ns' else 3)
            sb.setSuffix(f' {unit}')
            sb.setValue(new_display_value)
            sb.blockSignals(False)

        self.time_factor = new_factor
        self.pulse_plot.setLabel('bottom', f'Time ({unit})')
        self._update_pulse_diagram()
        
    def _refresh_experiment_list(self):
        self.exp_list.clear()

        for fn in Path(self.experiments_dir).glob("*.yaml"):
            # safe_load might return None or something unexpected
            try:
                desc = yaml.safe_load(fn.read_text())
            except Exception as e:
                print(f"Warning: could not parse {fn.name}: {e}")
                continue

            if not isinstance(desc, dict) or "experiment_type" not in desc:
                print(f"Warning: skipping invalid descriptor {fn.name}")
                continue

            self.exp_list.addItem(desc["experiment_type"])

    # ——— Buttons for Add/Edit/Delete ———
    def _on_add_experiment(self):
        dlg = ExperimentEditor(parent=self, experiments_dir=self.experiments_dir)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # save new descriptor
            dlg.save_descriptor()
            self._refresh_experiment_list()
            self._reload_factory()

            # if we just added/edited the experiment we're viewing, cycle away & back
            current = self.exp_combo.currentText()

            if current == dlg.le_name.text().strip():
                # pick any other experiment
                for other in self.experiment_descs:
                    if other != current:
                        self.exp_combo.setCurrentText(other)
                        break

                # then switch back to force reload
                self.exp_combo.setCurrentText(current)


    def _on_edit_experiment(self):
        item = self.exp_list.currentItem()

        if not item:
            return
        
        name = item.text()
        path = self.experiments_dir / f"{name}.yaml"
        dlg = ExperimentEditor(parent=self, descriptor_path=path, experiments_dir=self.experiments_dir)
        
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            dlg.save_descriptor()
            self._refresh_experiment_list()
            self._reload_factory()

            # if we just edited the experiment we're viewing, cycle away & back
            name = dlg.le_name.text().strip()

            if self.exp_combo.currentText() == name:
                for other in self.experiment_descs:
                    if other != name:
                        self.exp_combo.setCurrentText(other)
                        break

                self.exp_combo.setCurrentText(name)

    def _on_remove_experiment(self):
        item = self.exp_list.currentItem()

        if not item:
            return

        reply = QMessageBox.question(
            self,
            "Delete?",
            f"Remove experiment “{item.text()}”?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # delete
        (self.experiments_dir / f"{item.text()}.yaml").unlink()
        self._refresh_experiment_list()
        self._reload_factory()

    def _reload_factory(self):
        # re-load descriptors and update combo
        self.experiment_descs = load_experiments(self.experiments_dir)
        self.exp_combo.clear()
        self.exp_combo.addItems(self.experiment_descs.keys())

    def _apply_descriptor_defaults(self, exp_name: str):
        desc_path = self.experiments_dir / f"{exp_name}.yaml"
        try:
            desc = yaml.safe_load(
                desc_path.read_text(encoding="utf-8")
            )
        except Exception:
            return

        defaults = {
            p["name"]: p.get("default")
            for p in desc.get("parameters", [])
        }

        def get_default(*names, fallback):
            for name in names:
                value = defaults.get(name)
                if value is not None:
                    return value
            return fallback

        # Static ODMR/acquisition defaults.
        # Sweep values here are expected to be in GHz.
        self.start_input.setValue(
            float(
                get_default(
                    "freq_start",
                    "sweep_start",
                    fallback=2.65
                )
            )
        )

        self.stop_input.setValue(
            float(
                get_default(
                    "freq_stop",
                    "sweep_stop",
                    fallback=3.15
                )
            )
        )

        self.power_input.setValue(
            float(
                get_default(
                    "power",
                    "rf_power",
                    fallback=1.0
                )
            )
        )

        self.avg_input.setValue(
            int(
                get_default(
                    "averages",
                    fallback=5
                )
            )
        )

        self.frames_input.setValue(
            int(
                get_default(
                    "frames",
                    fallback=20
                )
            )
        )

        self.dynamic_input.setValue(
            int(
                get_default(
                    "n_dynamic_steps",
                    "dynamic_steps",
                    fallback=20
                )
            )
        )

        self.rate.setValue(
            int(
                get_default(
                    "max_rate",
                    "max_framerate",
                    fallback=16_000
                )
            )
        )

        timing_names = {
            "mw_duration",
            "read_time",
            "laserduration",
            "start_pulse_dur",
            "I_pulse",
            "Q_pulse",
            "tau",
        }

        for p in desc.get("parameters", []):
            name = p["name"]
            default = p.get("default")
            w = self.param_widgets.get(name)
            if w is None:
                continue

            if hasattr(w, "setValue"):
                value = default

                if name in timing_names:
                    value = float(default) / self.time_factor

                w.setValue(value)

            else:
                w.setCurrentText(str(default))

        self._update_pulse_diagram()

    def _clear_live(self):
        # Stop an active run before deleting its waiting-room files.
        if (
            self.process
            and self.process.state()
            == QtCore.QProcess.ProcessState.Running
        ):
            self._stop()
        else:
            self._clear_waiting_room()
            
        # Reset current detector labels.
        self.live_ps_label.setText(
            "Pulse-sequence step: 1/1"
        )
        self.freq_label.setText("Frequency: -- GHz")
        self.voltage_label.setText("APD voltage: -- V")
        self.ch0_label.setText("CH0: -- V")
        self.ch1_label.setText("CH1: -- V")
        self.normalized_label.setText(
            "Normalized: --"
        )
        self.combined_label.setText(
            "Combined: -- V"
        )

        # Clear all four live plots.
        number_steps = max(1, self.dynamic_input.value())

        ps_steps = max(
            1,
            int(
                getattr(
                    self,
                    "_active_pulse_sequence_steps",
                    1,
                )
            ),
        )

        frequency_axis = np.linspace(
            self.start_input.value(),
            self.stop_input.value(),
            number_steps
        )

        self.live_freqs = np.tile(
            frequency_axis,
            (ps_steps, 1),
        )
        self.live_ch0 = np.full(
            (ps_steps, number_steps),
            np.nan,
            dtype=float,
        )
        self.live_ch1 = np.full(
            (ps_steps, number_steps),
            np.nan,
            dtype=float,
        )
        self.live_normalized = np.full(
            (ps_steps, number_steps),
            np.nan,
        )
        self.live_combined = np.full(
            (ps_steps, number_steps),
            np.nan,
            dtype=float,
        )
        self._live_ch0_sums = np.zeros(
            (ps_steps, number_steps),
        )
        self._live_ch1_sums = np.zeros(
            (ps_steps, number_steps),
            dtype=float,
        )
        self._live_samples_per_step = np.zeros(
            (ps_steps, number_steps),
            dtype=int,
        )

        self.live_ch0_curve.setData([], [])
        self.live_ch1_curve.setData([], [])
        self.live_normalized_curve.setData([], [])
        self.live_combined_curve.setData([], [])
        self._live_measurement_index = 0

        # Remove any partially received DAQ values from the previous run.
        for attribute in (
            "_last_freq",
            "_last_ch0",
            "_last_ch1",
        ):
            if hasattr(self, attribute):
                delattr(
                    self,
                    attribute,
                )

        # Clear terminal log
        self.log_output.clear()

        # Reset status
        self.status_led.setStyleSheet("background-color: red; border-radius: 8px;")
        self.status_label.setText("Idle")

        # Reset progress bars & step label
        self.step_label.setText("Step 0/0")
        self.sweep_bar.setValue(0)
        self.count_gauge.setValue(0)

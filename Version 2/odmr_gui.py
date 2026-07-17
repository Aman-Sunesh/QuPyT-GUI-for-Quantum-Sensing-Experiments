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

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
LAST_CFG_PATH  = PROJECT_ROOT / '.qupyt' / 'last_config.json'

class ODMRGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.experiments_dir = Path.home() / 'Desktop' / 'QuPyt-master' / 'GUI' / 'experiments'
        self.output_dir = Path.home() / 'Desktop' / 'QuPyt-master'
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
        self.live_counts = []
        self.live_voltages = []

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

    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Setup Tab ---
        setup = QtWidgets.QWidget()
        self.tabs.addTab(setup, 'Setup')
        form = QtWidgets.QFormLayout(setup)
        self.setup_form = form

        # Watcher button
        self.start_watcher_btn = QtWidgets.QPushButton("Start watcher")
        self.start_watcher_btn.clicked.connect(self._start_watcher)
        form.addRow(self.start_watcher_btn)

        # Experiment type
        self.exp_combo = QtWidgets.QComboBox()
        self.exp_combo.addItems(self.experiment_descs.keys())

        self.exp_combo.currentTextChanged.connect(lambda name: (self._apply_descriptor_defaults(name),
                                                                self._update_pulse_diagram()))

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
        self.power_input.setRange(1.0, 8.0)
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
        self.refch_input.setRange(1, 99)
                
        self.mode_input = QtWidgets.QComboBox(); self.mode_input.addItems(['spread', 'sum'])

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

        self._init_pulse_diagram()
        self._update_time_units(self.unit_combo.currentText())

        # Processing & Display
        self.sub_input = QtWidgets.QCheckBox(
            'Top result plot: normalized difference (S-R)/(S+R)'
        )

        self.sub_input.setChecked(True)

        self.smooth_input = QtWidgets.QSpinBox()
        self.fit_input = QtWidgets.QComboBox(); self.fit_input.addItems(['Lorentzian', 'Gaussian'])
        self.errb_input = QtWidgets.QCheckBox('Show error bars')
       
        self.sub_input.stateChanged.connect(
            lambda *_: self._show_results()
        )
        self.fit_input.currentTextChanged.connect(
            lambda *_: self._show_results()
        )

        form.addRow(self.sub_input)
        form.addRow('Smoothing window:', self.smooth_input)
        form.addRow('Fit type:', self.fit_input)
        form.addRow(self.errb_input)

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
        form.addRow(self.powersupply_btn)

        h.addWidget(self.defaults_btn)
        h.addWidget(self.start_setup_btn)
        h.addWidget(self.stop_btn)
        h.addWidget(self.save_cfg_btn)
        h.addWidget(self.load_cfg_btn)
        form.addRow(h)

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

        # ——— Live ODMR Spectrum Plot ———
        self.live_plot = pg.PlotWidget()
        self.live_curve = self.live_plot.plot([], [], pen=None, symbol='o')
        self.live_plot.setLabel('bottom', 'Frequency (GHz)')
        self.live_plot.setLabel('left', 'Counts')

        # ——— Live APD Voltage Plot ———
        self.voltage_plot = pg.PlotWidget()
        self.voltage_curve = self.voltage_plot.plot(
            [],
            [],
            pen=None,
            symbol='o'
        )
        self.voltage_plot.setLabel('bottom', 'Frequency (GHz)')
        self.voltage_plot.setLabel('left', 'APD voltage (V)')

        # ——— Current values display ———
        hl = QtWidgets.QHBoxLayout()
        self.freq_label = QLabel("Frequency: -- GHz")
        self.voltage_label = QLabel("APD voltage: -- V")
        self.count_label = QLabel("Counts: --")

        for w in (
            self.freq_label,
            self.voltage_label,
            self.count_label
        ):
            w.setStyleSheet("font-size: 11pt; font-weight: bold;")
        hl.addWidget(self.freq_label)
        hl.addStretch()
        hl.addWidget(self.voltage_label)
        hl.addStretch()
        hl.addWidget(self.count_label)
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
        self.count_gauge.setFormat("Counts: %v/%m")

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
        live_splitter.addWidget(self.live_plot)
        live_splitter.addWidget(self.voltage_plot)
        live_splitter.addWidget(self.log_output)
        
        # give the plot a weight of 2 and the console 3
        live_splitter.setStretchFactor(0, 2)
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
        self.exp_combo.currentTextChanged.connect(
            lambda *_: self._populate_file_selector()
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
        fit_box = QGroupBox('Fit & Parameters')
        fit_layout = QVBoxLayout()
        self.fit_table = QTableWidget(4, 2)
        self.fit_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        params = ['Center (GHz)', 'FWHM (MHz)', 'Contrast (%)', 'R²']
        for i, p in enumerate(params):
            self.fit_table.setItem(i, 0, QTableWidgetItem(p))
        fit_layout.addWidget(self.fit_table)
        fit_box.setLayout(fit_layout)

        # 3b. Combined-channel ODMR spectrum: (CH0 + CH1) / 2
        mean_proc_box = QGroupBox(
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
        mean_proc_box.setLayout(mean_proc_layout)

        # Fit parameters for the combined-channel spectrum
        mean_fit_box = QGroupBox(
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
        mean_fit_box.setLayout(mean_fit_layout)

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
        splitter.addWidget(fit_box)
        splitter.addWidget(mean_proc_box)
        splitter.addWidget(mean_fit_box)

        layout = QtWidgets.QVBoxLayout(res)
        layout.addWidget(self.file_selector)
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

        if self.process and self.process.state() == QtCore.QProcess.ProcessState.Running:
            self.process.kill()

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.started.connect(self._on_started)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.finished.connect(self._on_finished)

        cmd  = sys.executable
        args = ['-u', '-m', 'qupyt.main']

        project_root = Path.home() / 'Desktop' / 'QuPyt-master'
        self.process.setWorkingDirectory(str(project_root))

        max_steps = self.dynamic_input.value()  # total number of steps
        self.count_gauge.setMinimum(0)
        self.count_gauge.setMaximum(max_steps)
        self.count_gauge.setValue(0)

        self.max_live_points = max_steps

        self.sweep_bar.setValue(0)

        # One plotted point per dynamic frequency step. Each point is the
        # running average of the acquisitions collected at that frequency.
        number_steps = max(1, self.dynamic_input.value())

        self.live_freqs = np.linspace(
            self.start_input.value(),
            self.stop_input.value(),
            number_steps
        ).tolist()

        self.live_counts = [np.nan] * number_steps
        self.live_voltages = [np.nan] * number_steps
        self._live_count_sums = np.zeros(number_steps, dtype=float)
        self._live_voltage_sums = np.zeros(number_steps, dtype=float)
        self._live_samples_per_step = np.zeros(number_steps, dtype=int)

        self.live_curve.setData([], [])
        self.voltage_curve.setData([], [])

        for attribute in (
            "_last_freq",
            "_last_count",
            "_last_voltage",
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

        if target.exists():
            target.unlink()

        shutil.copyfile(desktop_yaml, target)

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

        if self.sub_input.isChecked() and reference_channels < 2:
            QMessageBox.warning(
                self,
                "Normalized difference unavailable",
                (
                    "Normalized difference requires at least two reference "
                    "channels. Set Ref channels to 2 or disable normalization."
                )
            )
            return

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
        pulse_py  = PROJECT_ROOT / 'user_pulse_seq.py'
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

            # 2) catch any count line
            if count_m := re.search(
                rf"Counts:\s*({NUMBER_PATTERN})",
                line,
                flags=re.IGNORECASE
            ):
                self._last_count = float(count_m.group(1))

            # APD voltage printed by the DAQ reader.
            if voltage_m := re.search(
                rf"DAQ_VOLTAGE:\s*({NUMBER_PATTERN})",
                line,
                flags=re.IGNORECASE
            ):
                self._last_voltage = float(voltage_m.group(1))

            # Plot once both detector outputs have arrived.
            if (
                hasattr(self, "_last_count")
                and hasattr(self, "_last_voltage")
            ):
                c = self._last_count

                voltage = self._last_voltage

                number_steps = len(self.live_freqs)
                averages = max(1, self.avg_input.value())

                # QuPyt acquires once per average. Therefore every
                # `averages` acquisition pairs belong to one frequency.
                step_index = min(
                    self._live_measurement_index // averages,
                    number_steps - 1
                )

                if hasattr(self, "_last_freq"):
                    self.live_freqs[step_index] = self._last_freq

                f = self.live_freqs[step_index]

                self._live_count_sums[step_index] += c
                self._live_voltage_sums[step_index] += voltage
                self._live_samples_per_step[step_index] += 1

                sample_count = self._live_samples_per_step[step_index]

                self.live_counts[step_index] = (
                    self._live_count_sums[step_index] / sample_count
                )
                self.live_voltages[step_index] = (
                    self._live_voltage_sums[step_index] / sample_count
                )

                # Update both live plots.
                valid_steps = self._live_samples_per_step > 0
                frequency_array = np.asarray(self.live_freqs)
                count_array = np.asarray(self.live_counts)
                voltage_array = np.asarray(self.live_voltages)

                self.live_curve.setData(
                    frequency_array[valid_steps],
                    count_array[valid_steps]
                )
                self.voltage_curve.setData(
                    frequency_array[valid_steps],
                    voltage_array[valid_steps]
                )

                self.freq_label.setText(
                    f"Frequency: {f:.6f} GHz"
                )
                self.voltage_label.setText(
                    f"APD voltage: {voltage:.6f} V"
                )
                self.count_label.setText(
                    f"Counts: {c:.6g}"
                )

                self._live_measurement_index += 1

                for attribute in (
                    "_last_freq",
                    "_last_count",
                    "_last_voltage",
                ):
                    if hasattr(self, attribute):
                        delattr(self, attribute)

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

    def _on_finished(self):
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

    def _show_results(self):
        if not hasattr(self, "data") or self.data is None:
            return
        # 1. Metadata (try waiting_room YAML, otherwise fall back to GUI values)
        ts = QtCore.QDateTime.currentDateTime().toString()

        exp_name = self.exp_combo.currentText()
        yaml_path = (
            Path.home()
            / '.qupyt'
            / 'waiting_room'
            / f"{exp_name}.yaml"
        )

        if yaml_path.exists():
            with open(yaml_path,'r') as f:
                cfg = yaml.safe_load(f)
            meta_str = (
                f"Time: {ts}\n"
                f"Sweep: {cfg['dynamic_devices']['mw_source']['config']['frequency']} Hz\n"
                f"Steps: {cfg['dynamic_steps']}, Averages: {cfg['averages']}\n"
                f"MW dur: {cfg['pulse_sequence']['mw_duration']} μs, Readout: {cfg['pulse_sequence']['readout_time']} μs"
            )

        else:
            # build a minimal cfg from the current GUI values
            freq_start = self.start_input.value() * 1e9
            freq_stop  = self.stop_input.value()  * 1e9
            steps      = self.dynamic_input.value()
            refch      = self.refch_input.value()
            mw_dur     = self.mw_dur.value() * self.time_factor
            rd_dur     = self.read_dur.value() * self.time_factor
            cfg = {
                'dynamic_devices': {'mw_source': {'config': {'frequency': [freq_start, freq_stop]}}},
                'dynamic_steps': steps,
                'data': {'reference_channels': refch},
                'pulse_sequence': {'mw_duration': mw_dur, 'readout_time': rd_dur}
            }
            meta_str = (
                f"Time: {ts}\n"
                "No waiting_room YAML found; using GUI inputs\n"
                f"Sweep: [{freq_start}, {freq_stop}] Hz\n"
                f"Steps: {steps}, Ref chans: {refch}\n"
                f"MW dur: {mw_dur} μs, Readout: {rd_dur} μs"
            )

        self.meta_text.setPlainText(meta_str)

        # 2. Processed Spectrum
        # Average over frame/sensor dimensions while preserving:
        # axis 0 = reference channel
        # axis 1 = dynamic sweep step
        average_axes = tuple(range(2, self.data.ndim))
        arr_mean = np.nanmean(self.data, axis=average_axes)  # [channels, steps]
        arr_std  = np.nanstd(self.data, axis=average_axes)   # [channels, steps]

        frequency_cfg = cfg[
            'dynamic_devices'
        ]['mw_source']['config']['frequency']

        # Support both QuPyt formats:
        #   [start_frequency, stop_frequency]
        #   [channel, [start_frequency, stop_frequency]]
        if (
            isinstance(frequency_cfg, (list, tuple))
            and len(frequency_cfg) == 2
            and isinstance(frequency_cfg[1], (list, tuple))
        ):
            freq_start_hz, freq_stop_hz = map(
                float,
                frequency_cfg[1]
            )
        else:
            freq_start_hz, freq_stop_hz = map(
                float,
                frequency_cfg
            )

        freqs_hz = np.linspace(
            freq_start_hz,
            freq_stop_hz,
            arr_mean.shape[1]
        )

        freqs_ghz = freqs_hz / 1e9

        # Save the exact processed arrays for channel-plot exporting.
        self.result_freqs_ghz = freqs_ghz
        self.result_channel_means = arr_mean
        self.result_channel_stds = arr_std

        # ------------------------------------------------------------
        # Plot 1: normalized channel difference
        # ------------------------------------------------------------
        self.proc_plot.clear()
        self.proc_plot.setLabel(
            'bottom',
            'Frequency (GHz)'
        )

        using_normalized_difference = (
            self.sub_input.isChecked()
            and arr_mean.shape[0] >= 2
        )

        if using_normalized_difference:
            self.proc_box.setTitle(
                'Processed ODMR Spectrum — '
                '(CH0 - CH1) / (CH0 + CH1)'
            )
            denominator = arr_mean[0] + arr_mean[1]

            with np.errstate(
                divide='ignore',
                invalid='ignore'
            ):
                selected_y = np.where(
                    denominator != 0,
                    (
                        arr_mean[0] - arr_mean[1]
                    ) / denominator,
                    np.nan
                )

            self.proc_plot.setLabel(
                'left',
                '(CH0 - CH1) / (CH0 + CH1)'
            )
            self.proc_plot.plot(
                freqs_ghz,
                selected_y,
                pen='r',
                symbol='x'
            )
        else:
            self.proc_box.setTitle(
                'Processed ODMR Spectrum — CH0'
            )
            selected_y = arr_mean[0]

            self.proc_plot.setLabel(
                'left',
                'CH0 mean signal'
            )
            self.proc_plot.plot(
                freqs_ghz,
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

        if arr_mean.shape[0] >= 2:
            combined_y = (
                arr_mean[0] + arr_mean[1]
            ) / 2.0
        else:
            # Graceful fallback for old one-channel result files.
            combined_y = arr_mean[0]

        self.mean_proc_plot.plot(
            freqs_ghz,
            combined_y,
            pen='b',
            symbol='o'
        )

        # Preserve both processed arrays for later exporting/debugging.
        self.result_selected_spectrum = selected_y
        self.result_combined_spectrum = combined_y

        # ------------------------------------------------------------
        # Fit both plots independently
        # ------------------------------------------------------------
        model = (
            lorentzian
            if self.fit_input.currentText() == 'Lorentzian'
            else gaussian
        )

        self._fit_result_spectrum(
            x_hz=freqs_hz,
            y=selected_y,
            model=model,
            table=self.fit_table,
            plot_widget=self.proc_plot,
            normalized=using_normalized_difference,
        )

        self._fit_result_spectrum(
            x_hz=freqs_hz,
            y=combined_y,
            model=model,
            table=self.mean_fit_table,
            plot_widget=self.mean_proc_plot,
            normalized=False,
        )

        # update the little summary label
        d = self.data
        summary = f"{d.shape}, dtype={d.dtype}, min={d.min():.3g}, max={d.max():.3g}"
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

            average_axes = tuple(range(2, data.ndim))
            channel_means = np.nanmean(data, axis=average_axes)

            freqs_ghz = getattr(self, "result_freqs_ghz", None)
            if freqs_ghz is None or len(freqs_ghz) != data.shape[1]:
                freqs_ghz = np.linspace(
                    self.start_input.value(),
                    self.stop_input.value(),
                    data.shape[1]
                )

            export_array = np.column_stack(
                [freqs_ghz, *channel_means]
            )

            header = ",".join(
                ["frequency_GHz"]
                + [
                    f"channel_{index}"
                    for index in range(channel_means.shape[0])
                ]
            )

            np.savetxt(
                path,
                export_array,
                delimiter=",",
                header=header,
                comments=""
            )

        else:
            path = path.with_suffix(".png")

            average_axes = tuple(range(2, data.ndim))
            channel_means = np.nanmean(data, axis=average_axes)

            freqs_ghz = getattr(self, "result_freqs_ghz", None)
            if freqs_ghz is None or len(freqs_ghz) != data.shape[1]:
                freqs_ghz = np.linspace(
                    self.start_input.value(),
                    self.stop_input.value(),
                    data.shape[1]
                )

            figure, axis = plt.subplots(figsize=(8, 5))

            for channel_index in range(channel_means.shape[0]):
                axis.plot(
                    freqs_ghz,
                    channel_means[channel_index],
                    "o-",
                    label=f"Channel {channel_index}"
                )

            axis.set_xlabel("Frequency (GHz)")
            axis.set_ylabel("Mean detector signal")
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
            [reference_channel, dynamic_step, frames, sensor_dimensions...]

        Example:
            (2, 35, 55, 1)

        produces one mean spectrum for channel 0 and one for channel 1.
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

        channel_count = data.shape[0]

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

        # Fallback in case _show_results() has not yet populated them.
        if (
            channel_means is None
            or channel_stds is None
            or channel_means.shape[0] != channel_count
            or channel_means.shape[1] != data.shape[1]
            or channel_stds.shape != channel_means.shape
        ):
            average_axes = tuple(range(2, data.ndim))
            channel_means = np.nanmean(data, axis=average_axes)
            channel_stds = np.nanstd(data, axis=average_axes)

        if freqs_ghz is None or len(freqs_ghz) != data.shape[1]:
            freqs_ghz = np.linspace(
                self.start_input.value(),
                self.stop_input.value(),
                data.shape[1]
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
                f"{default_stem} — Reference channel {channel_index}"
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
            'mw_device_type','mw_output'
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
        }
        LAST_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LAST_CFG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)

    def _init_pulse_diagram(self):
        # make the PlotWidget
        self.pulse_plot = pg.PlotWidget(title="Pulse Diagram")
        self.pulse_plot.setLabel('bottom', 'Time')
        self.pulse_plot.getViewBox().invertY(True)   # so lane 0 is at top
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

        # add to the form 
        self.tabs.widget(0).layout().addRow('Pulse diagram:', self.pulse_plot)

        #  hook redraw to *all* pulse-timing spin-boxes
        for sb in (
            self.las_dur, self.mw_dur,
            self.read_dur, self.I_pulse_dur, self.Q_pulse_dur,
            self.tau_input, self.blocks_input, self.start_pulse_dur
        ):
            sb.valueChanged.connect(self._update_pulse_diagram)

        # initial draw
        self._update_pulse_diagram()

    def _update_pulse_diagram(self):
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

        # Descriptor constants are assumed to use internal µs.
        ctx["constants"] = {}

        for key, value in desc.get("constants", {}).items():
            try:
                converted = float(value)
            except (TypeError, ValueError):
                converted = value

            ctx["constants"][key] = converted
            ctx[key] = converted

        pulses = []

        for pulse in desc.get("pulses", []):
            channel = pulse.get("channel")

            if channel not in self.channel_lanes:
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
            
        # Reset frequency/count labels
        self.freq_label.setText("Frequency: -- GHz")
        self.voltage_label.setText("APD voltage: -- V")
        self.count_label.setText("Counts: --")

        # Clear both live plots.
        number_steps = max(1, self.dynamic_input.value())

        self.live_freqs = np.linspace(
            self.start_input.value(),
            self.stop_input.value(),
            number_steps
        ).tolist()

        self.live_counts = [np.nan] * number_steps
        self.live_voltages = [np.nan] * number_steps

        self._live_count_sums = np.zeros(number_steps, dtype=float)
        self._live_voltage_sums = np.zeros(number_steps, dtype=float)
        self._live_samples_per_step = np.zeros(number_steps, dtype=int)

        self.live_curve.setData([], [])
        self.voltage_curve.setData([], [])
        self._live_measurement_index = 0

        # Clear terminal log
        self.log_output.clear()

        # Reset status
        self.status_led.setStyleSheet("background-color: red; border-radius: 8px;")
        self.status_label.setText("Idle")

        # Reset progress bars & step label
        self.step_label.setText("Step 0/0")
        self.sweep_bar.setValue(0)
        self.count_gauge.setValue(0)

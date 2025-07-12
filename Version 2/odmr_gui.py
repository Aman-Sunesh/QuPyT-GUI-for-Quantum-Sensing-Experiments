# odmr_gui.py

import sys
import os
import shutil
import glob
import yaml
import csv
import json
import re
import time 
import warnings
import numpy as np
import pyqtgraph as pg
import importlib
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from jinja2 import Template
from scipy.optimize import curve_fit
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (QFileDialog, QPlainTextEdit, QMessageBox, QTableWidget, QTableWidgetItem, 
                             QTextEdit, QSplitter, QLabel, QGroupBox, QVBoxLayout, QFormLayout,
                             QSpinBox, QDoubleSpinBox)

from utils import lorentzian, gaussian
from GUI.experiment_yaml import render_experiment_yaml
from channels import CHANNEL_MAPPING
from experiment_factory import load_experiments
from experiment_editor import ExperimentEditor
from generic_generator import generate_from_descriptor
from stop_pb import stop_pulse_blaster
from GUI.power_supply import PowerSupplyDialog

warnings = getattr(sys, 'warnoptions', None)

# Global exception hook
logging.basicConfig(level=logging.ERROR)

def excepthook(exc_type, exc_value, exc_tb):
    QMessageBox.critical(None, "Unhandled Error", str(exc_value))
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

sys.excepthook = excepthook

warnings = getattr(sys, 'warnoptions', None)

from PyQt6.QtWidgets import QComboBox

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
LAST_CFG_PATH  = PROJECT_ROOT / '.qupyt' / 'last_config.json'

class ODMRGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.experiments_dir = Path.home() / 'Desktop' / 'QuPyt-master' / 'GUI' / 'experiments'
        self.experiment_descs = load_experiments(self.experiments_dir)
        self._suppress_auto_switch = False
        self.setWindowTitle('QuPyt Experiment GUI')
        self.process = None
        self.param_widgets = {}
        self._build_ui()
        self._restore_last_config()

        # populate the Experiments tab list
        self._refresh_experiment_list()

        # for live-plot data
        self.live_freqs = []
        self.live_counts = []

        # load all descriptors
        self.experiment_descs = load_experiments(self.experiments_dir)
        self.exp_combo.clear()
        self.exp_combo.addItems(self.experiment_descs.keys())
        self.exp_combo.currentTextChanged.emit(self.exp_combo.currentText())

        # force the “apply preset” step so the Setup tab reflects any edits:
        self.exp_combo.currentTextChanged.emit(self.exp_combo.currentText())

        # now override with your last‐used JSON, if it exists
        self._restore_last_config()  

        watcher = QtCore.QFileSystemWatcher([os.getcwd()], self)
        watcher.directoryChanged.connect(self._populate_file_selector)
        self._populate_file_selector() 
        self.tabs.setCurrentIndex(0)

    
    def _populate_file_selector(self):
        files = sorted(glob.glob('ODMR_*.npy'), key=os.path.getmtime)
        self.file_selector.clear()
        self.file_selector.addItems(files)

        if files:
            self.file_selector.setCurrentIndex(len(files)-1)

    def _on_started(self):
        # called when QProcess starts
        self.status_led.setStyleSheet("background-color: green; border-radius: 8px;")
        self.status_label.setText("Running")
    
    def _on_file_selected(self, filename: str):
        if self._suppress_auto_switch:
            self._suppress_auto_switch = False
            return
        
        if not filename:
            return
        try:
            self.data = np.load(filename)
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
        skip = {'mw_duration', 'read_time', 'laserduration', 'frames'}

        for p in desc.get("parameters", []):
            if p["name"] in skip:
                continue

            w = self.make_widget_for(p)
            form.addRow(f"{p['label']}:", w)
            self.param_widgets[p['name']] = w

        # Time-unit selector
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['ns', 'µs', 'ms'])
        form.addRow('Time unit:', self.unit_combo)

        # Sweep & Power
        self.start_input = QtWidgets.QDoubleSpinBox(); self.start_input.setSuffix(' GHz')
        self.stop_input = QtWidgets.QDoubleSpinBox();  self.stop_input.setSuffix(' GHz')
        self.steps_input = QtWidgets.QSpinBox()
        self.power_input = QtWidgets.QDoubleSpinBox(); self.power_input.setSuffix(' dBm')
        form.addRow('Sweep start:', self.start_input)
        form.addRow('Sweep stop:', self.stop_input)
        form.addRow('RF power:', self.power_input)

        # Averaging & Acquisition
        self.avg_input = QtWidgets.QSpinBox()
        self.frames_input = QtWidgets.QSpinBox()
        self.param_widgets['frames'] = self.frames_input
        self.dynamic_input = QtWidgets.QSpinBox()
        self.avg_input   .setRange(0, 9999)
        self.frames_input.setRange(0, 9999)
        self.dynamic_input.setRange(0, 9999)
        
        self.mode_input = QtWidgets.QComboBox(); self.mode_input.addItems(['spread', 'sum'])
        self.refch_input = QtWidgets.QSpinBox()
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
        self.sub_input = QtWidgets.QCheckBox('Baseline subtraction')
        self.smooth_input = QtWidgets.QSpinBox()
        self.fit_input = QtWidgets.QComboBox(); self.fit_input.addItems(['Lorentzian', 'Gaussian'])
        self.errb_input = QtWidgets.QCheckBox('Show error bars')
        form.addRow(self.sub_input)
        form.addRow('Smoothing window:', self.smooth_input)
        form.addRow('Fit type:', self.fit_input)
        form.addRow(self.errb_input)

        # Buttons
        h = QtWidgets.QHBoxLayout()
        self.defaults_btn    = QtWidgets.QPushButton('Load Defaults')
        self.defaults_btn.clicked.connect(self._load_defaults)
        self.start_setup_btn = QtWidgets.QPushButton('Start')
        self.stop_btn = QtWidgets.QPushButton('Stop')

        # Save/Load configuration buttons
        self.save_cfg_btn    = QtWidgets.QPushButton('Save Config…')
        self.load_cfg_btn    = QtWidgets.QPushButton('Load Config…')

        h.addWidget(self.defaults_btn)
        h.addWidget(self.start_setup_btn)
        h.addWidget(self.stop_btn)
        h.addWidget(self.save_cfg_btn)
        h.addWidget(self.load_cfg_btn)

        # Power-Supply Settings 
        self.powersupply_btn = QtWidgets.QPushButton("Power Supply…")
        self.powersupply_btn.clicked.connect(self._open_power_supply_dialog)
        form.addRow(self.powersupply_btn)

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

        # ——— Current values display ———
        hl = QtWidgets.QHBoxLayout()
        self.freq_label  = QLabel("Frequency: -- GHz")
        self.count_label = QLabel("Counts: --")
        for w in (self.freq_label, self.count_label):
            w.setStyleSheet("font-size: 11pt; font-weight: bold;")
        hl.addWidget(self.freq_label)   # left‐aligned
        hl.addStretch()                 # pushes the next widget to the right
        hl.addWidget(self.count_label)  # right‐aligned
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
        live_layout.addWidget(prog_box)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)

        # Put plot and console into a splitter for adjustable space
        live_splitter = QSplitter(QtCore.Qt.Orientation.Vertical)
        live_splitter.addWidget(self.live_plot)
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

        splitter = QSplitter(QtCore.Qt.Orientation.Vertical, res)

        # 1. Summary & Metadata
        meta_box = QGroupBox('Summary & Metadata')
        meta_layout = QVBoxLayout()
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        meta_layout.addWidget(self.meta_text)
        meta_box.setLayout(meta_layout)

        # 2. Processed ODMR Spectrum
        proc_box = QGroupBox('Processed ODMR Spectrum')
        proc_layout = QVBoxLayout()
        self.proc_plot = pg.PlotWidget()
        proc_layout.addWidget(self.proc_plot)

        # Add a "Save Spectrum" button
        save_spec_btn = QtWidgets.QPushButton("Save Spectrum…")
        save_spec_btn.clicked.connect(lambda: self._save_plot(self.proc_plot))
        proc_layout.addWidget(save_spec_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        proc_box.setLayout(proc_layout)

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

        # 4. Data Summary & “View” Button
        summary_box = QGroupBox("Data Summary")
        summary_layout = QVBoxLayout()
        # placeholder label — we’ll update this in _show_results()
        self.summary_label = QLabel("No data loaded")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        btn = QtWidgets.QPushButton("View Data…")
        btn.clicked.connect(self._on_view_data)
        summary_layout.addWidget(btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        summary_box.setLayout(summary_layout)

        splitter.addWidget(summary_box)
        splitter.addWidget(meta_box)
        splitter.addWidget(proc_box)
        splitter.addWidget(fit_box)

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
        args = ['-u', '-m', 'qupyt.main', '--verbose']

        project_root = Path.home() / 'Desktop' / 'QuPyt-master'
        self.process.setWorkingDirectory(str(project_root))

        max_steps = self.dynamic_input.value()  # total number of steps
        self.count_gauge.setMinimum(0)
        self.count_gauge.setMaximum(max_steps)
        self.count_gauge.setValue(0)

        self.max_live_points = max_steps

        self.sweep_bar.setValue(0)
        self.process.start(cmd, args)

        # reset our live‐plot buffers
        self.live_freqs.clear()
        self.live_counts.clear()
        self.live_curve.setData([], [])
        self.last_freq = None
        self.last_count = None

        # flip to Live tab so you can watch logs
        self.tabs.setCurrentIndex(1)


    def _clear_waiting_room(self):
        """Delete every file in ~/.qupyt/waiting_room."""
        wait_dir = Path.home() / '.qupyt' / 'waiting_room'

        if not wait_dir.exists():
            QMessageBox.information(self, "Nothing to clear", "Waiting room directory doesn’t exist.")
            return

        # Remove each file
        for f in wait_dir.glob('*'):
            try:
                f.unlink()
            except Exception as e:
                # skip any that can’t be removed
                print(f"Couldn’t delete {f}: {e}")

    def _deploy_yaml_and_run(self):
        desktop_yaml = Path.home() / 'Desktop' / 'ODMR.yaml'

        if not desktop_yaml.exists():
            QMessageBox.critical(self, "Deployment Error", f"Could not find {desktop_yaml}")
            return

        wait_dir = Path.home() / '.qupyt' / 'waiting_room'
        wait_dir.mkdir(parents=True, exist_ok=True)
        target = wait_dir / 'ODMR.yaml'

        # 1) initial atomic deploy
        tmp1 = target.with_suffix('.tmp')
        shutil.copy(desktop_yaml, tmp1)
        os.replace(tmp1, target)

        # 2) after 1 s clear out the waiting room
        QtCore.QTimer.singleShot(1000, lambda: self._phase_two(deploy_path=desktop_yaml, target=target))
 

    def _phase_two(self, deploy_path: Path, target: Path):
        # clear everything in waiting_room
        self._clear_waiting_room()

        # 3) after another 1 s, redeploy
        def do_redeploy():
            tmp2 = target.with_suffix('.tmp')
            shutil.copy(deploy_path, tmp2)
            os.replace(tmp2, target)
            QMessageBox.information(self, "Deployed", f"{target.name} deployed—starting run now.")

        QtCore.QTimer.singleShot(1000, do_redeploy)


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
            "mode":             self.mode_input.currentText(),
            "ref_channels":     self.refch_input.value(),
            "address":          "COM3",
            "ps_path":          desc.get("pulse_generator",""),

            # pulse sequence timings:
            "mw_duration":      self.mw_dur.value()   * self.time_factor,
            "laserduration":    self.las_dur.value()  * self.time_factor,
            "read_time":        self.read_dur.value() * self.time_factor,
            "max_rate":         self.rate.value(),
            "time_unit":     self.unit_combo.currentText(),
        }

        # now grab all other dynamic params, converting any timing ones
        for name, w in self.param_widgets.items():
            if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                val = w.value()
                # if this is one of your Jinja timing parameters, convert to µs
                if name in ("start_pulse_dur", "I_pulse", "Q_pulse", "tau"):
                    val *= self.time_factor
                vals[name] = val
            else:
                vals[name] = w.currentText()

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
                # only update if total matches what we expect
                if total != getattr(self, 'total_steps', total):
                    self.total_steps = total

                # try to pull the RF frequency (in Hz) out of the same line for the step label
                freq_match = re.search(r"frequency.*?([0-9]+(?:\.[0-9]+)?)", line)
                if freq_match:
                    freq_ghz = float(freq_match.group(1)) / 1e9
                    self.step_label.setText(f"Step {step}/{total} @ {freq_ghz:.3f} GHz")
                else:
                    self.step_label.setText(f"Step {step}/{total}")

                # update sweep progress bar
                pct = int(100 * step/total)
                self.sweep_bar.setValue(pct)

                # update count gauge as step count gauge
                self.count_gauge.setMaximum(total)
                self.count_gauge.setValue(step)

            # parse percentage progress if printed by tqdm (e.g. ' 30%|')
            if p := re.search(r"(\d+)%\|", line):
                self.sweep_bar.setValue(int(p.group(1)))

            # ——— now the state‐machine for live plotting ———
            # 1) catch any frequency line
            if freq_m := re.search(r"frequency.*?([0-9]+(?:\.[0-9]+)?)", line):
                self._last_freq = float(freq_m.group(1)) / 1e9

            # 2) catch any count line
            if count_m := re.search(r"Counts:\s*(\d+)", line):
                self._last_count = int(count_m.group(1))

            # 3) once we have both, plot and reset
            if hasattr(self, '_last_freq') and hasattr(self, '_last_count'):
                f = self._last_freq
                c = self._last_count

                # append and trim
                self.live_freqs.append(f)
                self.live_counts.append(c)
                if len(self.live_freqs) > self.max_live_points:
                    self.live_freqs.pop(0)
                    self.live_counts.pop(0)

                # update live curve
                self.live_curve.setData(self.live_freqs, self.live_counts)

                # update the little current-values label
                self.freq_label .setText(f"Frequency: {f:.3f} GHz")
                self.count_label.setText(f"Counts: {c}")

                # clear for next pair
                del self._last_freq, self._last_count

    def _stop(self):
        try:
            stop_pulse_blaster()
        except Exception as e:
            pass
        
        if self.process and self.process.state() == QtCore.QProcess.ProcessState.Running:
            # prevent on_finished() from auto‐switching to Results
            try:
                self.process.finished.disconnect(self._on_finished)
            except (TypeError, RuntimeError):
                pass

            self.process.terminate()
            self.process.waitForFinished(100)

            if self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
                self.process.kill()

            pid = self.process.processId()
            print(f"Terminated QuPyt watcher (PID {pid})")

        else:
            print("No running process to stop.")

        self.tabs.setCurrentIndex(1)

    def _double_stop(self):
        # first kill
        self._stop()
        # then schedule a second one 50 ms later to 
        # ensure that pulses are not sent by the synchroniser
        QtCore.QTimer.singleShot(50, self._stop)

    def _open_power_supply_dialog(self):
        dlg = PowerSupplyDialog(self)
        dlg.exec()

    def _on_finished(self):
        # refresh the dropdown list…
        self._populate_file_selector()
        # …and if there’s at least one file, pick the newest
        if self.file_selector.count():
            self.file_selector.setCurrentIndex(self.file_selector.count() - 1)

        self._show_results()
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

    def _show_results(self):
        # 1. Metadata (try waiting_room YAML, otherwise fall back to GUI values)
        ts = QtCore.QDateTime.currentDateTime().toString()
        yaml_path = Path.home() / '.qupyt' / 'waiting_room' / 'ODMR.yaml'
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
            mw_dur     = self.mw_dur.value()
            rd_dur     = self.read_dur.value()
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
        arr_mean = self.data.mean(axis=tuple(range(2,self.data.ndim)))  # [ch, steps]
        freqs = np.linspace(cfg['dynamic_devices']['mw_source']['config']['frequency'][0],
                            cfg['dynamic_devices']['mw_source']['config']['frequency'][1],
                            arr_mean.shape[1]) / 1e9  # GHz
        self.proc_plot.clear() 
        self.proc_plot.plot(freqs, arr_mean[0], pen='b', symbol='o')
        if cfg['data']['reference_channels']>1:
            diff = (arr_mean[0]-arr_mean[1])/(arr_mean[0]+arr_mean[1])
            self.proc_plot.plot(freqs, diff, pen='r', symbol='x')

        # 3. Fit to chosen lineshape
        y = arr_mean[0]
        x = freqs  # in Hz

        # choose model
        model = lorentzian if self.fit_input.currentText() == 'Lorentzian' else gaussian

        # initial guesses: center at min(y), width from span, amplitude and offset
        x0_guess = x[y.argmin()]
        span = x.max() - x.min()
        p0 = [x0_guess, span/20, (y.max()-y.min()), y.min()]

        try:
            popt, pcov = curve_fit(model, x, y, p0=p0)
            # extract params
            if model is lorentzian:
                x0, gamma, A, y0 = popt
                fwhm = 2*gamma
            else:
                mu, sigma, A, y0 = popt
                x0, fwhm = mu, 2.355*sigma  # FWHM of Gaussian = 2.355 σ

            # contrast as amplitude relative to baseline y0
            contrast = 100 * abs(A) / (abs(y0) + abs(A))
            # compute R²
            residuals = y - model(x, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot

            # format into human units
            center = x0/1e9
            fwhm_mhz = fwhm/1e6
        except Exception as e:
            # fallback if fit fails
            center, fwhm_mhz, contrast, r2 = np.nan, np.nan, np.nan, 0.0

        vals = [
            f"{center:.4f}",        # GHz
            f"{fwhm_mhz:.2f}",      # MHz
            f"{contrast:.1f}",      # %
            f"{r2:.3f}"
        ]
        for i,val in enumerate(vals):
            self.fit_table.setItem(i,1,QTableWidgetItem(val))


        # remember which file we just loaded
        self.current_file = Path(self.file_selector.currentText())

        # update the little summary label
        d = self.data
        summary = f"{d.shape}, dtype={d.dtype}, min={d.min():.3g}, max={d.max():.3g}"
        self.summary_label.setText(summary)


    def _export(self):
        files = glob.glob('*.npy')
        if not files:
            QMessageBox.warning(self, 'Export', 'No .npy files found.')
            return
        latest = max(files, key=os.path.getmtime)
        data = np.load(latest)
        path, _ = QFileDialog.getSaveFileName(self, 'Export data', '', 'NumPy (*.npy);;CSV (*.csv);;PNG (*.png)')
        if not path:
            return
        if path.endswith('.npy'):
            np.save(path, data)
        elif path.endswith('.csv'):
            arr = data.mean(axis=tuple(range(2, data.ndim)))
            np.savetxt(path, arr, delimiter=',')
        else:
            arr = data.mean(axis=tuple(range(2, data.ndim)))
            freqs = np.linspace(
                self.start_input.value()*1e9,
                self.stop_input.value()*1e9,
                arr.shape[1]
            )
            plt.figure()
            plt.errorbar(freqs, arr[0], yerr=arr[0].std(), fmt='o-')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Mean counts')
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        print(f"Exported data to {path}")

    def _save_plot(self, widget):
        """Prompt for filename and save given widget (PNG or SVG)."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        widget.grab().save(path)

    def _save_config(self):
        """Save current setup parameters to a CSV file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "CSV (*.csv)")
        if not path:
            return
        fields = [
            'sweep_start','sweep_stop','power',
            'averages','frames','dynamic_steps',
            'mode','ref_channels',
            'mw_duration','read_time',
            'laserduration', 'time_unit', 'max_rate'
        ]
        values = [
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
            self.unit_combo.currentText(), 
            self.rate.value(),
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
            self.las_dur    .setValue(float(row['laserduration']))

            unit = row.get('time_unit', 'µs')
            self.unit_combo.setCurrentText(unit)
            
            self.rate       .setValue(int  (row['max_rate']))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not parse configuration:\n{e}")

            return 
        # on successful parse & apply, let the user know
        QMessageBox.information(self, "Loaded", f"Configuration loaded from:\n{path}")

    def _load_defaults(self):
        """Reset MW, Readout and Laser durations to the YAML defaults."""
        desc = self.experiment_descs[self.exp_combo.currentText()]
        defaults = { p["name"]: p["default"] for p in desc.get("parameters", []) }
        self.mw_dur.setValue(defaults.get("mw_duration", 0.0))
        self.read_dur.setValue(defaults.get("read_time",    0.0))
        self.las_dur.setValue(defaults.get("laserduration",0.0))
        
    def _restore_last_config(self):
        """Load last‐used JSON snapshot if present."""
        try:
            with open(LAST_CFG_PATH, 'r') as f:
                cfg = json.load(f)
        except FileNotFoundError:
            return
        # apply values back into the widgets
        try:
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
            self.las_dur     .setValue(cfg['laserduration'])

            unit = cfg.get('time_unit', 'µs')
            self.unit_combo.setCurrentText(unit)

            self.rate        .setValue(cfg['max_rate'])
        except KeyError:
            # silently skip if schema mismatch
            pass

    def _write_last_config(self):
        """Dump current setup into a JSON file for ‘last used’ recall."""
        cfg = {
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
            'laserduration':    self.las_dur.value(),
            'time_unit':     self.unit_combo.currentText(),
            'max_rate':      self.rate.value(),
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

        desc = yaml.safe_load(desc_path.read_text())

        # build Jinja context from GUI widgets
        ctx = {}

        # --- parameters ---
        for name, widget in self.param_widgets.items():
            if hasattr(widget, 'value'):
                ctx[name] = widget.value()
            else:
                ctx[name] = widget.currentText()

        # --- constants ---
        ctx["constants"] = {}
        for k, v in desc.get("constants", {}).items():
            # constants are strings in the YAML; convert to float/int
            try:    ctx["constants"][k] = float(v)
            except: ctx["constants"][k] = int(v)

        # also expose each constant at top level
        for k, v in ctx["constants"].items():
            ctx[k] = v


        # render each pulse

        # start with the START trigger from the GUI
        start_dur = ctx.get("start_pulse_dur", 1.0)
        pulses = [("START", 0.0, start_dur)]
 
        # then add all the pulses from your descriptor
        for p in desc.get("pulses", []):
            if p.get("channel") == "START":
                continue

            ch = p["channel"]
            s_expr = p["start"]
            d_expr = p["duration"]

            try:
                s = float(Template(p["start"]).render(ctx)) * self.time_factor
                d = float(Template(p["duration"]).render(ctx)) * self.time_factor
            except Exception as e:
                # skip bad rows
                continue
            
            pulses.append((ch, s, d))

        return pulses
        
    
    def _update_time_units(self, unit):
        # convert suffix to µs-space: ns→1e-3, µs→1, ms→1e3
        factor  = {'ns': 1e-3, 'µs': 1.0, 'ms': 1e3}[unit]

        # store so we can convert back to µs when generating pulses/YAML
        self.time_factor = factor

        # update the X-axis label
        self.pulse_plot.setLabel('bottom', f'Time ({unit})')

        # choose a sane “max” in µs and scale into current unit
        max_val = 1e6 / factor   # e.g. 1 s = 1e6 µs

        for sb in (
             self.las_dur,
             self.mw_dur,
             self.read_dur,
             self.I_pulse_dur,
             self.Q_pulse_dur,
             self.tau_input,
             self.start_pulse_dur 
         ):   
            sb.blockSignals(True)
            sb.setRange(0.0, max_val)
            sb.setDecimals(0 if unit == 'ns' else 3)
            sb.setSuffix(f' {unit}')
            sb.blockSignals(False)

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
            desc = yaml.safe_load(desc_path.read_text())
        except Exception:
            return

        for p in desc.get("parameters", []):
            name = p["name"]
            default = p.get("default")
            w = self.param_widgets.get(name)
            if w is None:
                continue
            # spinboxes use setValue, combos use setCurrentText
            if hasattr(w, "setValue"):
                w.setValue(default)
            else:
                w.setCurrentText(str(default))

        params = {p["name"]: p.get("default") for p in desc.get("parameters", [])}

        if "I_pulse" in params:
            self.I_pulse_dur.setValue(params["I_pulse"])
        if "Q_pulse" in params:
            self.Q_pulse_dur.setValue(params["Q_pulse"])
        if "tau" in params:
            self.tau_input.setValue(params["tau"])
        
        self._update_pulse_diagram()

    def _clear_live(self):
        # Reset frequency/count labels
        self.freq_label .setText("Frequency: -- GHz")
        self.count_label.setText("Counts: --")

        # Clear live plot
        self.live_curve.setData([], [])

        # Clear terminal log
        self.log_output.clear()

        # Reset status
        self.status_led.setStyleSheet("background-color: red; border-radius: 8px;")
        self.status_label.setText("Idle")

        # Reset progress bars & step label
        self.step_label.setText("Step 0/0")
        self.sweep_bar.setValue(0)
        self.count_gauge.setValue(0)

        # Also clear out the waiting room directory
        self._clear_waiting_room()

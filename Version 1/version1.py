# ────────────────────────────────────────────────────────────────
# QuPyt ODMR GUI
# 
# This script provides the main GUI for configuring, running, and 
# analyzing Optically Detected Magnetic Resonance (ODMR) experiments 
# using NV centers. It integrates:
#  • Experiment setup (frequency sweep, MW power, pulse timings)
#  • Real-time data acquisition (NI-DAQ, APD signals)
#  • Pulse sequence deployment (PulseBlaster synchronization)
#  • Power supply control via VISA
#  • Live plotting and result visualization
# 
# Key technologies: PyQt6, PyQtGraph, NI-DAQmx, spinapi (PulseBlaster)
# and external YAML-based experiment templates.
# ────────────────────────────────────────────────────────────────


import sys
import os
import shutil
import glob
import yaml
import csv
import json
import pyvisa
import re
import time
import numpy as np
import subprocess
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (QFileDialog, QPlainTextEdit, QMessageBox, QTableWidget, QTableWidgetItem, QTextEdit, QSplitter,
                              QLabel, QGroupBox, QVBoxLayout, QFormLayout, QDialog, QDoubleSpinBox, QPushButton)
from string import Template
import pyqtgraph as pg
import nidaqmx
from nidaqmx.constants import AcquisitionType
from stop_pb import stop_pulse_blaster

# Suppress warnings unless explicitly enabled
warnings = getattr(sys, 'warnoptions', None)


# ────────────────────────────────────────────────────────────────
# Utility functions for curve fitting
# These are used to fit Lorentzian or Gaussian functions to ODMR dips
# ────────────────────────────────────────────────────────────────
def lorentzian(x, x0, gamma, A, y0):
    """Lorentzian line shape for fitting ODMR dips.
    Args:
        x: Frequency values (GHz)
        x0: Center frequency
        gamma: Half-width at half-maximum (HWHM)
        A: Amplitude (negative for dip)
        y0: Baseline offset
    Returns:
        Lorentzian curve evaluated at x
    """
    return y0 + A * (gamma**2 / ((x - x0)**2 + gamma**2))

def gaussian(x, mu, sigma, A, y0):
    """Gaussian line shape for alternative dip fitting.
    Args:
        x: Frequency values
        mu: Mean (center)
        sigma: Standard deviation
        A: Amplitude
        y0: Baseline offset
    Returns:
        Gaussian curve evaluated at x
    """
    return y0 + A * np.exp(-((x - mu)**2) / (2*sigma**2))


# ────────────────────────────────────────────────────────────────
# Predefined experiment presets for quick configuration
# ────────────────────────────────────────────────────────────────
PRESETS = {
    'ODMR': {  # Basic ODMR experiment
        'sweep_start': 2.80,  # GHz
        'sweep_stop': 2.95,   # GHz
        'power': 10,          # dBm
        'averages': 1,
        'frames': 20,
        'dynamic_steps': 10,  # frequency steps
        'mode': 'spread',
        'ref_channels': 2,    # reference channel count
        'mw_duration': 20,    # μs
        'read_time': 8,       # μs
        'laserduration': 150, # μs
        'max_rate': 10000,    # Hz
    },
    'XY8': {   # Advanced pulse sequence for decoherence measurements
        'sweep_start': 0,
        'sweep_stop': 0,
        'power': 10,
        'averages': 4,
        'frames': 50,
        'dynamic_steps': 10,
        'mode': 'sum',
        'ref_channels': 0,
        'mw_duration': 0.5,
        'read_time': 10,
        'laserduration': 100,
        'max_rate': 20000,
    }
    # Add more as needed
}

# Paths for saving configs
LAST_CFG_PATH = Path.home() / '.qupyt' / 'last_config.json'
PS_CFG_PATH = Path.home() / '.qupyt' / 'power_supply_config.json'

# ────────────────────────────────────────────────────────────────
# Main GUI Class
# ────────────────────────────────────────────────────────────────
class ODMRGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._suppress_auto_switch = False
        self.setWindowTitle('QuPyt Experiment GUI')
        self.process = None  # Will hold QProcess for experiment runner

        # Power Supply settings button
        self.powersupply_btn = QtWidgets.QPushButton("Power Supply")
        self.powersupply_btn.clicked.connect(self._open_power_supply_dialog)

        # Build GUI components
        self._build_ui()

        # Buffers for live plotting data
        self.live_freqs = []   # X-axis for live ODMR plot
        self.live_counts = []  # Counts from APD
        self.live_daq = []     # Voltage readings from DAQ
        self._daq_buffer = []  # Temporary buffer for DAQ averaging

        # load the default preset into all the widgets
        self._load_preset(self.exp_combo.currentText())

        # now override with your last‐used JSON, if it exists
        self._restore_last_config()  

        # Watch for new result files
        watcher = QtCore.QFileSystemWatcher([os.getcwd()], self)
        watcher.directoryChanged.connect(self._populate_file_selector)
        self._populate_file_selector() 
        self.tabs.setCurrentIndex(0) # Default to Setup tab


    # Populate results dropdown with available .npy files
    def _populate_file_selector(self):
        files = sorted(glob.glob('ODMR_*.npy'), key=os.path.getmtime)
        self.file_selector.clear()
        self.file_selector.addItems(files)

        if files:
            self.file_selector.setCurrentIndex(len(files)-1)
          
    # Update status when watcher process starts
    def _on_started(self):
        # called when QProcess starts
        self.status_led.setStyleSheet("background-color: green; border-radius: 8px;")
        self.status_label.setText("Running")

    # Load selected result file and show
    def _on_file_selected(self, filename: str):
        if self._suppress_auto_switch:
            self._suppress_auto_switch = False
            return
        
        if not filename:
            return
        try:
            self.data = np.load(filename)   # Load experiment data
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Could not load {filename}:\n{e}")
            return

        # and redraw everything
        self._show_results()   

  
    # ────────────────────────────────────────────────────────────────
    # Build GUI layout (Setup, Live, Results tabs)
    # ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Setup Tab ---
        setup = QtWidgets.QWidget()
        self.tabs.addTab(setup, 'Setup')
        form = QtWidgets.QFormLayout(setup)


        # Watcher button
        self.start_watcher_btn = QtWidgets.QPushButton("Start watcher")
        self.start_watcher_btn.clicked.connect(self._start_watcher)
        form.addRow(self.start_watcher_btn)

        # Experiment type
        self.exp_combo = QtWidgets.QComboBox()
        self.exp_combo.addItems(PRESETS.keys())
        form.addRow('Experiment:', self.exp_combo)
        self.exp_combo.currentTextChanged.connect(self._load_preset)

        # Sweep & Power
        self.start_input = QtWidgets.QDoubleSpinBox(); self.start_input.setSuffix(' GHz')
        self.stop_input = QtWidgets.QDoubleSpinBox();  self.stop_input.setSuffix(' GHz')
        self.steps_input = QtWidgets.QSpinBox()
        self.power_input = QtWidgets.QDoubleSpinBox(); self.power_input.setSuffix(' dBm')
        form.addRow('Sweep start:', self.start_input)
        form.addRow('Sweep stop:', self.stop_input)
        form.addRow('RF power:', self.power_input)

        self.start_input.setRange(0, 1000)      # allow 0 – 1000 GHz
        self.stop_input.setRange(0, 1000)       # allow 0 – 1000 GHz

        # Averaging & Acquisition
        self.avg_input = QtWidgets.QSpinBox()
        self.frames_input = QtWidgets.QSpinBox()
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
        form.addRow('MW duration:', self.mw_dur)
        form.addRow('Readout duration:', self.read_dur)
        form.addRow('Laser duration:', self.las_dur)
        form.addRow('Max rate:', self.rate)

        for sb in (self.mw_dur, self.read_dur, self.las_dur):
            sb.setMinimum(0.0)
            sb.setMaximum(9999999.0)
            sb.setDecimals(2)
    
        self.rate.setMinimum(1)
        self.rate.setMaximum(100_000)

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
        self.start_setup_btn = QtWidgets.QPushButton('Start')
        self.stop_btn        = QtWidgets.QPushButton('Stop')

        # Save/Load configuration buttons
        self.save_cfg_btn    = QtWidgets.QPushButton('Save Config…')
        self.load_cfg_btn    = QtWidgets.QPushButton('Load Config…')

        h.addWidget(self.defaults_btn)
        h.addWidget(self.start_setup_btn)
        h.addWidget(self.stop_btn)
        h.addWidget(self.save_cfg_btn)
        h.addWidget(self.load_cfg_btn)
        form.addRow(h)

        form.addRow(self.powersupply_btn)

        self.defaults_btn.clicked.connect(lambda: self._load_preset(self.exp_combo.currentText()))
        self.start_setup_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.clicked.connect(self._stop)
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

        # ——— Live ODMR Spectrum Plot ———
        self.live_plot = pg.PlotWidget(title="ODMR Spectrum")
        self.live_curve = self.live_plot.plot([], [], pen=None, symbol='o')
        self.live_plot.setLabel('bottom', 'Frequency (GHz)')
        self.live_plot.setLabel('left', 'Counts')

        # ——— Live DAQ Voltage Plot ———
        self.daq_plot = pg.PlotWidget(title="DAQ Voltage")
        self.daq_curve = self.daq_plot.plot([], [], pen=None, symbol='x')
        self.daq_plot.setLabel('bottom', 'Frequency (GHz)')
        self.daq_plot.setLabel('left', 'Voltage (V)')

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

        # create the label, then style it
        self.step_label = QLabel("Step 0/0")
        self.step_label.setStyleSheet("font-size: 14pt; font-weight: bold;")

        # now create the bars
        self.sweep_bar  = QtWidgets.QProgressBar()
        self.sweep_bar.setFormat("Sweep %p%")
        self.count_gauge= QtWidgets.QProgressBar()
        self.count_gauge.setFormat("Counts: %v/%m")

        # add to layout
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
        live_splitter.addWidget(self.daq_plot)
        live_splitter.addWidget(self.log_output)
        
        # give the plot a weight of 2 and the console 3
        live_splitter.setStretchFactor(0, 3)
        live_splitter.setStretchFactor(1, 3)
        live_splitter.setStretchFactor(2, 3)
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


    def _start_watcher(self):
        """Start the QuPyt backend watcher process that listens for new YAML files 
        in the waiting_room directory and runs the corresponding experiment."""

        # Block auto-switching of file selector while watcher is active
        self._suppress_auto_switch = True

        # If an old watcher process is still running, kill it to avoid conflicts
        if self.process and self.process.state() == QtCore.QProcess.ProcessState.Running:
            self.process.kill()

        self.process = QtCore.QProcess(self)  # Create a new QProcess for running the QuPyt backend
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)  # Merge stdout and stderr channels so logs show in one place

        # Connect process signals to handlers for UI updates
        self.process.started.connect(self._on_started)   # Set green LED + Running label
        self.process.readyReadStandardOutput.connect(self._on_stdout)   # Parse logs for progress
        self.process.finished.connect(self._on_finished)   # Switch to Results tab when done

        # Build command: Python interpreter + QuPyt main module with verbose logging
        cmd  = sys.executable
        args = ['-u', '-m', 'qupyt.main', '--verbose']

        # Set working directory to project root (assumes QuPyt installed on Desktop)
        project_root = Path.home() / 'Desktop' / 'QuPyt-master'
        self.process.setWorkingDirectory(str(project_root))

        max_steps = self.dynamic_input.value()  # total number of steps
        self.count_gauge.setMinimum(0)
        self.count_gauge.setMaximum(max_steps)
        self.count_gauge.setValue(0)

        self.max_live_points = max_steps

        # Reset sweep progress bar
        self.sweep_bar.setValue(0)

        # Start the QuPyt watcher process
        self.process.start(cmd, args)

        # reset our live‐plot buffers
        self.live_freqs.clear()
        self.live_counts.clear()
        self.live_curve.setData([], [])
        self.last_freq = None
        self.last_count = None

        # flip to Live tab so you can watch logs
        self.tabs.setCurrentIndex(1)


    # Delete all files from the waiting_room directory (used for experiment YAML files).
    def _clear_waiting_room(self):
        """Delete every file in ~/.qupyt/waiting_room."""
        wait_dir = Path.home() / '.qupyt' / 'waiting_room'

        # If directory does not exist, show info message
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

    # Deploy the generated ODMR.yaml into waiting_room so QuPyt watcher picks it up.
    def _deploy_yaml_and_run(self):
        desktop_yaml = Path.home() / 'Desktop' / 'ODMR.yaml'

        # # Validate that the YAML file exists
        if not desktop_yaml.exists():
            QMessageBox.warning(self, "Missing YAML", f"Could not find {desktop_yaml}")
            return

        # Ensure waiting_room exists
        wait_dir = Path.home() / '.qupyt' / 'waiting_room'
        wait_dir.mkdir(parents=True, exist_ok=True)

        # First copy (to trigger event)
        target = wait_dir / 'ODMR.yaml'
        tmp = target.with_suffix('.tmp')
        shutil.copy(desktop_yaml, tmp)
        os.replace(tmp, target)

        # Clear the waiting room immediately (simulate deletion + re-add for guaranteed event)
        self._clear_waiting_room()

        # Second copy (actually starts the experiment)
        tmp2 = target.with_suffix('.tmp')
        shutil.copy(desktop_yaml, tmp2)
        os.replace(tmp2, target)

        # Inform user
        QMessageBox.information(self, "Deployed", "ODMR.yaml deployed—starting run now.")


    # Load predefined experiment settings into GUI fields (e.g., ODMR, XY8).
    def _load_preset(self, name: str):
        p = PRESETS[name]
      
        # Update all relevant UI fields with preset values
        self.start_input.setValue(p['sweep_start'])
        self.stop_input.setValue(p['sweep_stop'])
        self.power_input.setValue(p['power'])
        self.avg_input.setValue(p['averages'])
        self.frames_input.setValue(p['frames'])
        self.dynamic_input.setValue(p['dynamic_steps'])
        self.mode_input.setCurrentText(p['mode'])
        self.refch_input.setValue(p['ref_channels'])
        self.mw_dur.setValue(p['mw_duration'])
        self.read_dur.setValue(p['read_time'])
        self.las_dur.setValue(p['laserduration'])
        self.rate.setValue(p['max_rate'])
        self.smooth_input.setValue(1)
        self.fit_input.setCurrentText('Lorentzian')
        self.errb_input.setChecked(True)

    def _start(self):
        """
        Generate a fully‐templated ODMR.yaml in waiting_room
        with anchors, comments and exact layout.
        """
        vals = {
            'averages': self.avg_input.value(),
            'nframes': self.frames_input.value(),
            'apd_input': 'Dev1/ai0',
            'MW': 3,
            'LASER': 2,
            'READ': 0,
            'START': 1,
            'n_dynamic_steps': self.dynamic_input.value(),
            'address': 'COM3',  
            'freq_start': self.start_input.value() * 1e9,
            'freq_stop': self.stop_input.value() * 1e9,
            'power': self.power_input.value(),
            'mode': self.mode_input.currentText(),
            'ref_channels': self.refch_input.value(),
            'ps_path': str(Path(__file__).parent / 'odmr_sample_pulse_sequence.py'),
            'mw_duration': self.mw_dur.value(),
            'laserduration': self.las_dur.value(),
            'read_time': self.read_dur.value(),
            'max_rate': self.rate.value(),
        }

        tpl = Template(r'''    # Template for the ODMR experiment
experiment_type: ODMR

averages: $averages

# Defines the sensor used in the experiment.
sensor:
  type: "DAQ"
  config:
    number_measurements: &nframes $nframes
    apd_input: "$apd_input"


# This defines the synchronising device for our experiment.
synchroniser:
  type: "PulseBlaster"
  config:
  channel_mapping:
    MW:    $MW
    LASER: $LASER
    READ:  $READ
    START: $START
dynamic_steps: &n_dynamic_steps $n_dynamic_steps


dynamic_devices:
  mw_source:
    device_type: "WindFreak"
    address: "$address"
    config:
      frequency:
        - $freq_start
        - $freq_stop
      amplitude:
        - ["channel_0", [$power, $power]]

static_devices: {}


data:
  averaging_mode: '$mode'
  dynamic_steps: *n_dynamic_steps
  compress: false
  reference_channels: $ref_channels

ps_path: '$ps_path'

pulse_sequence:
  mw_duration: $mw_duration
  laserduration: $laserduration
  readout_time: $read_time
  referenced_measurements: *nframes
  max_framerate: $max_rate
''')
        
        desktop_dir = Path.home() / 'Desktop'
        desktop_dir.mkdir(parents=True, exist_ok=True)
        desktop_yaml = desktop_dir / 'ODMR.yaml'
        with open(desktop_yaml, 'w', encoding='utf-8') as f:
            f.write(tpl.substitute(vals))

        # Prevent the file-selector from auto-jumping us
        self.file_selector.blockSignals(True)

        # snapshot the current GUI values:
        self._write_last_config()

        # spawn the watcher/process as usual
        self._start_watcher()

        # Force ourselves to the Live tab
        self.tabs.setCurrentIndex(1)

        # After a short delay (once the watcher is truly up), re-enable combo signals
        QtCore.QTimer.singleShot(200, lambda: self.file_selector.blockSignals(False))


    def _on_stdout(self):
        """Handle new output from the QuPyt watcher process.
        Parses stdout for progress updates, frequency, counts, and DAQ voltage readings.
        Updates the live plots and UI labels in real-time.
        """

        # Read all available output from the process (stdout and stderr merged)
        raw = bytes(self.process.readAll()).decode('utf-8', errors='ignore')
        self.log_output.appendPlainText(raw)

        # Extract any DAQ voltage readings (printed as DAQ_VOLTAGE: <value>)
        for vstr in re.findall(r"DAQ_VOLTAGE:\s*([0-9.+-eE]+)", raw):
            self._daq_buffer.append(float(vstr))

        # Process each line of output
        for line in raw.splitlines():
            # Match pattern like: "| step/total" to update progress bars
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
            # catch any frequency line
            if freq_m := re.search(r"frequency.*?([0-9]+(?:\.[0-9]+)?)", line):
                self._last_freq = float(freq_m.group(1)) / 1e9

            # catch any count line
            if count_m := re.search(r"Counts:\s*(\d+)", line):
                self._last_count = int(count_m.group(1))

            # once we have both, plot counts—and also collapse DAQ buffer into one voltage point
            if hasattr(self, '_last_freq') and hasattr(self, '_last_count'):
                f = self._last_freq
                c = self._last_count

                # —— plot counts ——
                self.live_freqs.append(f)
                self.live_counts.append(c)
                if len(self.live_freqs) > self.max_live_points:
                    self.live_freqs.pop(0)
                    self.live_counts.pop(0)
                self.live_curve.setData(self.live_freqs, self.live_counts)

                # —— plot DAQ voltage ——
                mean_v = None
                if self._daq_buffer:
                    mean_v = sum(self._daq_buffer) / len(self._daq_buffer)
                else:
                    # no raw samples this step?  default to NaN or 0
                    mean_v = float('nan')

                self.live_daq.append(mean_v)
                # clear buffer for next step
                self._daq_buffer.clear()

                # keep voltages in sync
                if len(self.live_daq) > len(self.live_freqs):
                    self.live_daq = self.live_daq[-len(self.live_freqs):]
                self.daq_curve.setData(self.live_freqs, self.live_daq)

                # update labels
                self.freq_label .setText(f"Frequency: {f:.3f} GHz")
                self.count_label.setText(f"Counts: {c}")

                # clear for next pair
                del self._last_freq, self._last_count

        # collect raw DAQ samples into our buffer for the next step
        for vstr in re.findall(r"DAQ_VOLTAGE:\s*([0-9.+-eE]+)", raw):
            self._daq_buffer.append(float(vstr))

    # Stop the QuPyt watcher process and PulseBlaster safely.
    def _stop(self):
        if self.process and self.process.state() == QtCore.QProcess.ProcessState.Running:
            # prevent on_finished() from auto‐switching to Results
            try:
                self.process.finished.disconnect(self._on_finished)
            except (TypeError, RuntimeError):
                pass

            # Attempt graceful termination
            self.process.terminate()
            self.process.waitForFinished(100)

            # Force kill if still running
            if self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
                self.process.kill()

            pid = self.process.processId()
            print(f"Terminated QuPyt watcher (PID {pid})")

        else:
            print("No running process to stop.")

        # Stop PulseBlaster timing outputs to avoid MW/laser stuck ON
        stop_pulse_blaster()

        self.tabs.setCurrentIndex(1)

    def _on_finished(self):
        """Called when QuPyt process finishes.
        Refresh file selector, load latest results, and switch to Results tab.
        """
      
        # refresh the dropdown list…
        self._populate_file_selector()
        # …and if there’s at least one file, pick the newest
        if self.file_selector.count():
            self.file_selector.setCurrentIndex(self.file_selector.count() - 1)

        self._show_results()
        self.tabs.setCurrentIndex(2)

    # Read a single DAQ voltage sample (blocking call).
    def _read_daq_voltage_single(self):
        """Read one sample from the DAQ."""
        try:
            # this will block until one sample is returned
            val = self.daq_task.read()
            return float(val)
        except Exception as e:
            # if something goes wrong, just propagate or log
            print("DAQ read error:", e)
            return 0.0

    # Display detailed statistics of the currently loaded .npy dataset in a popup dialog.
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


    # Load experiment metadata, plot averaged ODMR spectrum, fit resonance, and update UI.
    def _show_results(self):
        #   Get current timestamp
        ts = QtCore.QDateTime.currentDateTime().toString()

        # Try to read YAML metadata from waiting_room
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
            # Fallback: reconstruct metadata from GUI inputs
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

        # Display metadata
        self.meta_text.setPlainText(meta_str)

        # Compute averaged data across frames and channels
        arr_mean = self.data.mean(axis=tuple(range(2,self.data.ndim)))  # [ch, steps]

        # Generate frequency axis in GHz
        freqs = np.linspace(cfg['dynamic_devices']['mw_source']['config']['frequency'][0],
                            cfg['dynamic_devices']['mw_source']['config']['frequency'][1],
                            arr_mean.shape[1]) / 1e9 # GHz

        # # Plot primary channel
        self.proc_plot.clear()
        self.proc_plot.plot(freqs, arr_mean[0], pen='b', symbol='o')

        # # If reference channel exists, plot normalized difference
        if cfg['data']['reference_channels']>1:
            diff = (arr_mean[0]-arr_mean[1])/(arr_mean[0]+arr_mean[1])
            self.proc_plot.plot(freqs, diff, pen='r', symbol='x')

        # Fit to chosen lineshape
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


    # Export the most recent experiment data in .npy, .csv, or .png format.
    def _export(self):
        # Find all .npy files in the current directory
        files = glob.glob('*.npy')
        if not files:
            QMessageBox.warning(self, 'Export', 'No .npy files found.')
            return

        # Identify the latest .npy file by modification time
        latest = max(files, key=os.path.getmtime)
        data = np.load(latest)

        # Ask user for destination file path and format
        path, _ = QFileDialog.getSaveFileName(self, 'Export data', '', 'NumPy (*.npy);;CSV (*.csv);;PNG (*.png)')
        if not path:
            return # User canceled save dialog

        # Export in the chosen format
        if path.endswith('.npy'):
            np.save(path, data) # Save raw NumPy data
          
        elif path.endswith('.csv'):
            # Compute mean across extra dimensions (averaging over frames)
            arr = data.mean(axis=tuple(range(2, data.ndim)))
            # Save as comma-separated values
            np.savetxt(path, arr, delimiter=',')
        else:  # Export as PNG plot
            # Average the data and compute frequency axis
            arr = data.mean(axis=tuple(range(2, data.ndim)))
            freqs = np.linspace(
                self.start_input.value()*1e9,
                self.stop_input.value()*1e9,
                arr.shape[1]
            )

            # Create plot with error bars (standard deviation as error)
            plt.figure()
            plt.errorbar(freqs, arr[0], yerr=arr[0].std(), fmt='o-')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Mean counts')
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
          
        print(f"Exported data to {path}")


    # Capture and save a snapshot of the given plot widget in PNG or SVG format.
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

        # Define configuration fields and corresponding values
        fields = [
            'sweep_start','sweep_stop','power',
            'averages','frames','dynamic_steps',
            'mode','ref_channels',
            'mw_duration','read_time','laserduration','max_rate'
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
            self.rate.value(),
        ]

        # Write values into CSV file
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
            self.rate       .setValue(int  (row['max_rate']))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not parse configuration:\n{e}")

            return 
        # on successful parse & apply, let the user know
        QMessageBox.information(self, "Loaded", f"Configuration loaded from:\n{path}")

    # Restore last used configuration from JSON snapshot (if available).
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
            self.rate        .setValue(cfg['max_rate'])
        except KeyError:
            # silently skip if schema mismatch
            pass


    # Save current GUI configuration into a JSON file for future restoration.
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
            'max_rate':      self.rate.value(),
        }
        LAST_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LAST_CFG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)


    # Reset live plotting UI and clear temporary experiment data.
    def _clear_live(self):
        # Reset frequency/count labels
        self.freq_label .setText("Frequency: -- GHz")
        self.count_label.setText("Counts: --")

        # Clear live plots
        self.live_curve.setData([], [])
        self.live_daq.clear()    
        self.daq_curve.setData([], [])

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

    def _open_power_supply_dialog(self):
        dlg = PowerSupplyDialog(self)
        dlg.exec()
 
class PowerSupplyDialog(QDialog):
    """
    Dialog window for controlling and monitoring a multi-channel power supply unit (PSU).
    Provides options to set voltage/current for channels, start/stop output, and display actual readings.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Power-Supply Settings")

        # Form layout to organize input fields and buttons
        layout = QFormLayout(self)
      
        # Load previously saved PSU configuration if available
        try:
            cfg = json.load(open(PS_CFG_PATH))   # Load saved voltage/current setpoints
        except:
            cfg = {}  # If no config exists, start with empty settings

        # Initialize VISA resource manager for PSU communication
        rm = pyvisa.ResourceManager('@py')
        self.inst = rm.open_resource('ASRL4::INSTR')
        self.inst.write_termination = '\n'
        self.inst.read_termination  = '\n'
        self.inst.timeout = 2000

        # Try querying PSU identification string to confirm connection
        try:
            idn = self.inst.query('*IDN?').strip()
            print(f"[PSU] IDN: {idn}")
        except Exception as e:
            QMessageBox.warning(self, "PSU Error", f"Could not query PSU IDN:\n{e}")

        self.controls = {}  # Dictionary to store controls for each channel

        # Create UI for 4 PSU channels
        for ch in (1,2,3,4):
            # Load saved values or default to zero
            v0 = cfg.get(f"VSET{ch}", 0.0)
            i0 = cfg.get(f"ISET{ch}", 0.0)
            v_spin = QDoubleSpinBox(); v_spin.setSuffix(" V"); v_spin.setRange(0,30); v_spin.setValue(v0)
            i_spin = QDoubleSpinBox(); i_spin.setSuffix(" A"); i_spin.setDecimals(6); i_spin.setRange(1e-5, 50); i_spin.setSingleStep(1e-5);  i_spin.setValue(i0)
            v_act  = QLabel("-- V")
            i_act  = QLabel("-- A")
            start  = QPushButton(f"Start CH{ch}")
            stop   = QPushButton(f"Stop CH{ch}")

            row = QtWidgets.QHBoxLayout()
            row.addWidget(QLabel(f"CH{ch} Set:"))
            row.addWidget(v_spin)
            row.addWidget(i_spin)
            row.addSpacing(20)
            row.addWidget(QLabel("Actual:"))
            row.addWidget(v_act)
            row.addWidget(i_act)
            row.addSpacing(20)
            row.addWidget(start)
            row.addWidget(stop)
            layout.addRow(row)
          
            self.controls[ch] = {
                "v_spin":v_spin, "i_spin":i_spin,
                "v_act":v_act,   "i_act":i_act,
                "start":start,   "stop":stop
            }

            # Connect start/stop buttons to respective handlers
            start.clicked .connect(lambda _,c=ch: self._start_channel(c))
            stop.clicked  .connect(lambda _,c=ch: self._stop_channel(c))

        # Buttons for saving configuration and closing dialog
        self.save_btn  = QPushButton("Save Settings")
        self.close_btn = QPushButton("Close")
        layout.addRow(self.save_btn, self.close_btn)
        self.save_btn.clicked .connect(self._save_settings)
        self.close_btn.clicked.connect(self.accept)

        # Timer to periodically update actual voltage/current readings
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_actuals)
        self.timer.start(500)

    # Start power supply output on the specified channel with the set voltage/current.
    def _start_channel(self, ch):
        vs   = self.controls[ch]["v_spin"].value()
        iset = self.controls[ch]["i_spin"].value()

        # put PSU into independent‐channel mode
        self.inst.write('TRACK0')

        # use colon syntax for setpoints
        cmd_v = f"VSET{ch}:{vs:.2f}"
        cmd_i = f"ISET{ch}:{iset:.3f}"
        cmd_o = f"OUT{ch}:1"

        # Push your values out
        print(f"[PSU] → {cmd_v}")
        print(f"[PSU] → {cmd_i}")
        print(f"[PSU] → {cmd_o}")
        self.inst.write(cmd_v)
        self.inst.write(cmd_i)
        self.inst.write(cmd_o)

        # Give it a moment to settle
        time.sleep(0.5)

        # Read back actuals immediately and show in the dialog
        v_act = self.inst.query(f"VOUT{ch}?").strip()
        i_act = self.inst.query(f"IOUT{ch}?").strip()
        self.controls[ch]["v_act"].setText(f"{v_act} V")
        self.controls[ch]["i_act"].setText(f"{i_act} A")

  
    # Stop output on the specified channel.
    def _stop_channel(self, ch):
        self.inst.write(f"OUT{ch}:0")


    # Periodically update actual voltage/current values for all channels.
    def _update_actuals(self):
        for ch,ctrl in self.controls.items():
            v = self.inst.query(f"VOUT{ch}?").strip()
            i = self.inst.query(f"IOUT{ch}?").strip()
            ctrl["v_act"].setText(f"{v} V")
            ctrl["i_act"].setText(f"{i} A")

    # Save the current voltage/current setpoints to a JSON config file.
    def _save_settings(self):
        os.makedirs(PS_CFG_PATH.parent, exist_ok=True)
        data = {}
        for ch,ctrl in self.controls.items():
            data[f"VSET{ch}"] = ctrl["v_spin"].value()
            data[f"ISET{ch}"] = ctrl["i_spin"].value()
        with open(PS_CFG_PATH,'w') as f:
            json.dump(data, f, indent=2)
        QMessageBox.information(self, "Saved", "Power-supply settings saved.")

    def closeEvent(self, ev):
        """
        Clean up resources when dialog is closed:
        - Stop the update timer
        - Close PSU VISA connection
        """
        self.timer.stop()
        self.inst.close()
        super().closeEvent(ev)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ODMRGui()
    win.resize(1000, 1100)
    win.show()
    sys.exit(app.exec())

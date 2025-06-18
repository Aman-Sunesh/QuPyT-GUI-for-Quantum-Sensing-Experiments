import sys
import os
import shutil
import glob
import yaml
import numpy as np
import subprocess
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from pathlib import Path
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QFileDialog, QPlainTextEdit, QMessageBox, QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QLabel, QGroupBox, QVBoxLayout
from string import Template
import pyqtgraph as pg
import re

warnings = getattr(sys, 'warnoptions', None)

def lorentzian(x, x0, gamma, A, y0):
    return y0 + A * (gamma**2 / ((x - x0)**2 + gamma**2))

def gaussian(x, mu, sigma, A, y0):
    return y0 + A * np.exp(-((x - mu)**2) / (2*sigma**2))

# Predefined experiment presets
PRESETS = {
    'ODMR': {
        'sweep_start': 2.80,
        'sweep_stop': 2.95,
        'power': 10,
        'averages': 1,
        'frames': 20,
        'dynamic_steps': 10,
        'mode': 'spread',
        'ref_channels': 2,
        'mw_duration': 20,
        'read_time': 8,
        'laser_time': 150,
        'max_rate': 10000,
    },
    'XY8': {
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
        'laser_time': 100,
        'max_rate': 20000,
    }
}

class ODMRGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._suppress_auto_switch = False
        self.setWindowTitle('QuPyt Experiment GUI')
        self.process = None
        self._build_ui()
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

        # switch to the Results tab
        self.tabs.setCurrentIndex(2)
        # and redraw everything
        self._show_results()

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
            sb.setMaximum(999.0)
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
        h.addWidget(self.defaults_btn)
        h.addWidget(self.start_setup_btn)
        h.addWidget(self.stop_btn)
        form.addRow(h)

        self.defaults_btn.clicked.connect(lambda: self._load_preset(self.exp_combo.currentText()))
        self.start_setup_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)


        # --- Live Tab ---
        live = QtWidgets.QWidget()
        self.tabs.addTab(live, 'Live')
        live_layout = QtWidgets.QVBoxLayout(live)
        self.run_live_btn = QtWidgets.QPushButton("Run experiment")
        self.run_live_btn.clicked.connect(self._deploy_yaml_and_run)
        live_layout.addWidget(self.run_live_btn)
        
        # Clear waiting-room button
        self.clear_waiting_btn = QtWidgets.QPushButton("Clear waiting room")
        self.clear_waiting_btn.clicked.connect(self._clear_waiting_room)
        live_layout.addWidget(self.clear_waiting_btn)

        # ——— Status Section ———
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

        # ——— Progress Section ———
        prog_box = QGroupBox("Progress")
        pb = QtWidgets.QVBoxLayout()
        self.step_label = QtWidgets.QLabel('Step 0/0 @ 0 Hz')
        self.step_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.sweep_bar  = QtWidgets.QProgressBar()
        self.sweep_bar.setFormat("Sweep %p%")
        self.count_gauge= QtWidgets.QProgressBar()
        self.count_gauge.setFormat("Counts: %v/%m")
        pb.addWidget(self.step_label)
        pb.addWidget(self.sweep_bar)
        pb.addWidget(self.count_gauge)
        prog_box.setLayout(pb)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        live_layout.addWidget(self.log_output)

        # add to live tab
        live_layout.addWidget(status_box)
        live_layout.addWidget(prog_box)

        # Results Tab
        res = QtWidgets.QWidget()
        self.tabs.addTab(res, 'Results')

        # ─── File-selector dropdown ───────────────────────────────────────────────────────
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

        # 2. Raw Data Explorer
        raw_box = QGroupBox('Raw Data Explorer')
        raw_layout = QVBoxLayout()
        self.raw_view = pg.ImageView()
        raw_layout.addWidget(self.raw_view)
        raw_box.setLayout(raw_layout)

        # 3. Processed ODMR Spectrum
        proc_box = QGroupBox('Processed ODMR Spectrum')
        proc_layout = QVBoxLayout()
        self.proc_plot = pg.PlotWidget()
        proc_layout.addWidget(self.proc_plot)
        proc_box.setLayout(proc_layout)

        # 4. Fit & Parameter Readout
        fit_box = QGroupBox('Fit & Parameters')
        fit_layout = QVBoxLayout()
        self.fit_table = QTableWidget(4, 2)
        self.fit_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        params = ['Center (GHz)', 'FWHM (MHz)', 'Contrast (%)', 'R²']
        for i, p in enumerate(params):
            self.fit_table.setItem(i, 0, QTableWidgetItem(p))
        fit_layout.addWidget(self.fit_table)
        fit_box.setLayout(fit_layout)

        # 5. Data Summary & “View” Button
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
        splitter.addWidget(raw_box)
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

        # Load defaults
        self._load_preset(self.exp_combo.currentText())

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

        self.sweep_bar.setValue(0)
        self.process.start(cmd, args)

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

        QMessageBox.information(self, "Done", "Waiting room has been cleared.")

    def _deploy_yaml_and_run(self):
        desktop_yaml = Path.home() / 'Desktop' / 'ODMR.yaml'
        if not desktop_yaml.exists():
            QMessageBox.warning(self, "Missing YAML", f"Could not find {desktop_yaml}")
            return

        # atomic copy into waiting room
        wait_dir = Path.home() / '.qupyt' / 'waiting_room'
        wait_dir.mkdir(parents=True, exist_ok=True)
        target = wait_dir / 'ODMR.yaml'
        tmp = target.with_suffix('.tmp')
        shutil.copy(desktop_yaml, tmp)
        os.replace(tmp, target)

        QMessageBox.information(self, "Deployed", "ODMR.yaml deployed—starting run now.")


    def _load_preset(self, name: str):
        p = PRESETS[name]
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
        self.las_dur.setValue(p['laser_time'])
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
            'laser_time': self.las_dur.value(),
            'read_time': self.read_dur.value(),
            'max_rate': self.rate.value(),
        }

        tpl = Template(r'''
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
  laserduration: $laser_time
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

        # spawn the watcher/process as usual
        self._start_watcher()

        # Force ourselves to the Live tab
        self.tabs.setCurrentIndex(1)

        # After a short delay (once the watcher is truly up), re-enable combo signals
        QtCore.QTimer.singleShot(200, lambda: self.file_selector.blockSignals(False))


    def _on_stdout(self):
        raw = bytes(self.process.readAll()).decode('utf-8', errors='ignore')
        # dump the raw text into your Live log
        self.log_output.appendPlainText(raw)       
    
        for line in raw.splitlines():
            # parse step progress: look for patterns like '3/10'
            m = re.search(r"\s*(\d+)/(\d+)\b", line)

            if m:
                step, total = map(int, m.groups())
                self.total_steps = total
                
                if total != getattr(self, 'total_steps', None):
                    continue

                # update step label
                # try to pull the RF frequency (in Hz) out of the same line:
                freq_match = re.search(r"frequency.*?([\d\.]+)", line)
                if freq_match:
                    # convert to GHz
                    freq_ghz = float(freq_match.group(1)) / 1e9
                    self.step_label.setText(f"Step {step}/{total} @ {freq_ghz:.3f} GHz")
                else:
                    # fallback if no freq in this line
                    self.step_label.setText(f"Step {step}/{total}")


                # update sweep progress bar
                pct = int(100 * step/total)
                self.sweep_bar.setValue(pct)
                # update count gauge as step count gauge
                self.count_gauge.setMaximum(total)
                self.count_gauge.setValue(step)

            # parse percentage progress if printed by tqdm (e.g. ' 30%|')
            p = re.search(r"(\d+)%\|", line)
            if p:
                pct = int(p.group(1))
                self.sweep_bar.setValue(pct)

            # placeholder for live count readout parsing
            # e.g., if output contains 'Counts: 1234', update gauge
            c = re.search(r"Counts?:\s*(\d+)", line)
            if c:
                count = int(c.group(1))
                # assume max counts corresponds to 100% of gauge
                self.count_gauge.setValue(count)

    def _stop(self):
        # if the watcher is running, kill it
        if self.process and self.process.state() == QtCore.QProcess.ProcessState.Running:
            self.process.terminate()
            pid = self.process.processId()
            print(f"Terminated QuPyt watcher (PID {pid})")

        # now reset the Live‐tab UI
        # 1) Status LED back to red + idle label
        self.status_led.setStyleSheet("background-color: red; border-radius: 8px;")
        self.status_label.setText("Idle")

        # 2) Progress indicators back to zero
        self.step_label.setText("Step 0/0 @ 0 Hz")
        self.step_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.sweep_bar.setValue(0)
        self.count_gauge.setValue(0)

        # 3) Clear the log output
        self.log_output.clear()

    def _on_finished(self):
        # refresh the dropdown list…
        self._populate_file_selector()
        # …and if there’s at least one file, pick the newest
        if self.file_selector.count():
            self.file_selector.setCurrentIndex(self.file_selector.count() - 1)

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

        # 2. Raw Data (append first channel)
        # Make sure we actually have at least 2D data
        arr_full = self.data
        try:
            arr = arr_full[0, ...].squeeze()
        except Exception:
            arr = arr_full.squeeze()

        if arr.ndim >= 2:
            # transpose so X/Y map correctly
            self.raw_view.setImage(arr.T, autoLevels=True)
        else:
            # clear or show a placeholder if there's no image‐like data
            self.raw_view.clear()
            print(f"Skipping raw‐view (array is {arr.ndim}-D)")

        # 3. Processed Spectrum
        arr_mean = self.data.mean(axis=tuple(range(2,self.data.ndim)))  # [ch, steps]
        freqs = np.linspace(cfg['dynamic_devices']['mw_source']['config']['frequency'][0],
                            cfg['dynamic_devices']['mw_source']['config']['frequency'][1],
                            arr_mean.shape[1])
        self.proc_plot.clear()
        self.proc_plot.plot(freqs, arr_mean[0], pen='b', symbol='o')
        if cfg['data']['reference_channels']>1:
            diff = (arr_mean[0]-arr_mean[1])/(arr_mean[0]+arr_mean[1])
            self.proc_plot.plot(freqs, diff, pen='r', symbol='x')

        # 4. Fit to chosen lineshape
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

        # switch to results tab
        self.tabs.setCurrentIndex(2)


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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = ODMRGui()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())
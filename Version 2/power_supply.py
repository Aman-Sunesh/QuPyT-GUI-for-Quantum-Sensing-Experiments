# power_supply.py

# ────────────────────────────────────────────────────────────────
# Controls a Korad KC3405 Programmable DC Power Supply via PyVISA.
# Provides a Qt dialog for setting voltage/current on 4 channels,
# monitoring actual outputs, and saving/restoring setpoints.
# ────────────────────────────────────────────────────────────────

# Controls a Korad KC3405 Programmable DC Power Supply via PyVISA.
import os, json, time
from pathlib import Path
import pyvisa
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QDoubleSpinBox, QLabel,
    QPushButton, QMessageBox
)


# Path for persisting last-used voltage/current setpoints
PS_CFG_PATH = Path.home() / '.qupyt' / 'power_supply_config.json'

class PowerSupplyDialog(QDialog):
    """
    Dialog for configuring and monitoring a multi-channel Korad KC3405
    DC power supply. Allows user to set voltage/current per channel,
    start/stop each output, view actual readings, and save settings.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Power-Supply Settings")

        # Main form layout
        layout = QFormLayout(self)

        # Load previously saved setpoints if available
        try:
            cfg = json.load(open(PS_CFG_PATH))
        except:
            cfg = {}

        # Initialize VISA resource for PSU communication
        rm = pyvisa.ResourceManager('@py')
        self.inst = rm.open_resource('ASRL4::INSTR')
        self.inst.write_termination = '\n'
        self.inst.read_termination  = '\n'
        self.inst.timeout = 2000

        # Query PSU identification to verify connection
        try:
            idn = self.inst.query('*IDN?').strip()
            print(f"[PSU] IDN: {idn}")
        except Exception as e:
            QMessageBox.warning(self, "PSU Error", f"Could not query PSU IDN:\n{e}")

        # Dictionary to store per-channel controls
        self.controls = {}

        # Create UI elements for each of the 4 channels
        for ch in (1,2,3,4):
            # Load last setpoints or default to zero
            v0 = cfg.get(f"VSET{ch}", 0.0)
            i0 = cfg.get(f"ISET{ch}", 0.0)
            v_spin = QDoubleSpinBox(); v_spin.setSuffix(" V"); v_spin.setRange(0,30); v_spin.setValue(v0)
            i_spin = QDoubleSpinBox(); i_spin.setSuffix(" A"); i_spin.setDecimals(6); i_spin.setRange(1e-5, 50); i_spin.setSingleStep(1e-5);  i_spin.setValue(i0)

            # Labels to display actual measured outputs
            v_act  = QLabel("-- V")
            i_act  = QLabel("-- A")
            
            # Buttons to start/stop each channel
            start  = QPushButton(f"Start CH{ch}")
            stop   = QPushButton(f"Stop CH{ch}")

            # Horizontal layout grouping controls
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

            # Store references for event handlers
            self.controls[ch] = {
                "v_spin":v_spin, "i_spin":i_spin,
                "v_act":v_act,   "i_act":i_act,
                "start":start,   "stop":stop
            }

            # Connect buttons to channel handlers
            start.clicked .connect(lambda _,c=ch: self._start_channel(c))
            stop.clicked  .connect(lambda _,c=ch: self._stop_channel(c))

        # Buttons to save settings and close dialog
        self.save_btn  = QPushButton("Save Settings")
        self.close_btn = QPushButton("Close")
        layout.addRow(self.save_btn, self.close_btn)
        self.save_btn.clicked .connect(self._save_settings)
        self.close_btn.clicked.connect(self.accept)

        # Timer to periodically refresh actual readings
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_actuals)
        self.timer.start(500)

    # Enable output on channel `ch` with the user-set voltage/current.
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

    # Disable output on channel `ch`.
    def _stop_channel(self, ch):
        self.inst.write(f"OUT{ch}:0")

    # Periodically query and update actual voltage/current for all channels.
    def _update_actuals(self):
        for ch,ctrl in self.controls.items():
            v = self.inst.query(f"VOUT{ch}?").strip()
            i = self.inst.query(f"IOUT{ch}?").strip()
            ctrl["v_act"].setText(f"{v} V")
            ctrl["i_act"].setText(f"{i} A")

    # Persist current setpoints to disk for recall in future sessions.
    def _save_settings(self):
        os.makedirs(PS_CFG_PATH.parent, exist_ok=True)
        data = {}
        for ch,ctrl in self.controls.items():
            data[f"VSET{ch}"] = ctrl["v_spin"].value()
            data[f"ISET{ch}"] = ctrl["i_spin"].value()
        with open(PS_CFG_PATH,'w') as f:
            json.dump(data, f, indent=2)
        QMessageBox.information(self, "Saved", "Power-supply settings saved.")

    # Clean up timer and VISA connection when dialog is closed.
    def closeEvent(self, ev):
        self.timer.stop()
        self.inst.close()
        super().closeEvent(ev)

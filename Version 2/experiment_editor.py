import csv
import yaml
import pyqtgraph as pg
from pathlib import Path
from jinja2 import Template

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QTableWidget,
    QPushButton,
    QHBoxLayout,
    QDialogButtonBox,
    QLineEdit,
    QMessageBox,
    QLabel,
    QHeaderView,
    QSizePolicy,
    QFileDialog,
)
from channels import CHANNEL_MAPPING

class ExperimentEditor(QtWidgets.QDialog):
    def __init__(self, parent=None, descriptor_path=None, experiments_dir=None):
        super().__init__(parent)
        self.resize(800, 900)
        self.setMinimumSize(800, 900)
        self.setWindowTitle("Add Experiment" if descriptor_path is None else "Edit Experiment")
        self.experiments_dir = Path(experiments_dir)
        self.descriptor_path = Path(descriptor_path) if descriptor_path else None

        # channel/lane + color mappings (copy-pasted from odmr_gui._init_pulse_diagram)
        self.channel_lanes = {
            'LASER':  0,
            'MW':     1,
            'READ':   2,
            'START':  3,
            'I':      4,
            'Q':      5,
        }
        self.channel_colors = {
            'LASER': (255,  50,  50, 200),
            'MW':    (50,   50, 255, 200),
            'READ':  (50,  255,  50, 200),
            'START': (200, 100,   0, 200),
            'I':     (255,  50, 255, 200),
            'Q':     (50,  255, 255, 200),
        }

        # main layout
        v = QtWidgets.QVBoxLayout(self)
        form = QFormLayout()
        v.addLayout(form)

        # experiment_type
        self.le_name = QtWidgets.QLineEdit()
        form.addRow("Type:", self.le_name)
        # pulse_generator
        self.le_gen = QtWidgets.QLineEdit()
        form.addRow("Generator func:", self.le_gen)

        # parameters table
        self.tbl = QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(
            ["name","label","type","default","min","max","unit/choices"]
        )

        # allow vertical expansion
        self.tbl.setSizePolicy(
            QtWidgets.QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )

        # auto-resize columns
        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)

        v.addWidget(self.tbl, 3)

        # ── constants table ──
        self.const_tbl = QTableWidget(0, 2)
        self.const_tbl.setHorizontalHeaderLabels(["name", "value"])

        self.const_tbl.setSizePolicy(
            QtWidgets.QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )

        # auto-resize columns
        chdr = self.const_tbl.horizontalHeader()
        chdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        chdr.setStretchLastSection(True)

        v.addWidget(QLabel("Constants:"))
        v.addWidget(self.const_tbl, 2)

        hb3 = QHBoxLayout()
        btn_cadd = QPushButton("Add Constant")
        btn_crm  = QPushButton("Remove Constant")
        hb3.addWidget(btn_cadd); hb3.addWidget(btn_crm)
        v.addLayout(hb3)
        btn_cadd.clicked.connect(lambda: self._add_row(self.const_tbl))
        btn_crm .clicked.connect(lambda: self.const_tbl.removeRow(self.const_tbl.currentRow()))

        # ── pulse‐sequence table
        self.pulse_tbl = QTableWidget(0, 4)
        self.pulse_tbl.setHorizontalHeaderLabels(
            ["channel", "start-expr", "duration-expr", "blocks (comma-list)"]
        )

        self.pulse_tbl.setSizePolicy(
            QtWidgets.QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )

        # auto-resize columns
        phdr = self.pulse_tbl.horizontalHeader()
        phdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        phdr.setStretchLastSection(True)

        v.addWidget(QLabel("Pulse Sequence:"))
        v.addWidget(self.pulse_tbl, 3)

        self.le_seq_order   = QLineEdit()
        self.le_seq_repeats = QLineEdit()
        form.addRow("Sequence order:",   self.le_seq_order)
        form.addRow("Sequence repeats:", self.le_seq_repeats)

        # add/remove pulse rows
        hb2 = QHBoxLayout()
        btn_padd = QPushButton("Add Pulse")
        btn_prm  = QPushButton("Remove Pulse")
        hb2.addWidget(btn_padd); hb2.addWidget(btn_prm)
        v.addLayout(hb2)
        btn_padd.clicked.connect(lambda: self._add_row(self.pulse_tbl))
        btn_prm.clicked.connect(self._remove_pulse_row)

        # add/remove parameter
        hb = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("Add Param")
        btn_rm  = QtWidgets.QPushButton("Remove Param")
        hb.addWidget(btn_add); hb.addWidget(btn_rm)
        v.addLayout(hb)
        btn_add.clicked.connect(self._add_param_row)
        btn_rm .clicked.connect(lambda: self.tbl.removeRow(self.tbl.currentRow()))

        # ── SAVE/LOAD buttons for CSV ──
        hb_csv = QHBoxLayout()
        btn_save_csv = QPushButton("Save to CSV…")
        btn_load_csv = QPushButton("Load from CSV…")
        btn_template = QPushButton("Download Template…")
        hb_csv.addWidget(btn_save_csv)
        hb_csv.addWidget(btn_load_csv)
        hb_csv.addWidget(btn_template)
        v.addLayout(hb_csv)

        btn_save_csv.clicked.connect(self._save_to_csv)
        btn_load_csv.clicked.connect(self._load_from_csv)
        btn_template.clicked.connect(self._download_template)

        bb = QDialogButtonBox()
        bb.setStandardButtons(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        v.addWidget(bb)

        bb.accepted.connect(self._on_save)
        bb.rejected.connect(self._on_cancel)

        self.preview = pg.PlotWidget(title="Pulse Preview")
        self.preview.getViewBox().invertY(True)

        # create the standalone preview dialog (no close button)
        self.preview_window = QtWidgets.QDialog(self)
        self.preview_window.setWindowTitle("Pulse Preview")
        flags = self.preview_window.windowFlags()
        self.preview_window.setWindowFlags(flags & ~QtCore.Qt.WindowType.WindowCloseButtonHint)
       
        # embed the same plot widget
        pw_layout = QVBoxLayout(self.preview_window)
        pw_layout.addWidget(self.preview)
        self.preview_window.resize(600, 400)

        # load the descriptor first (so parameters/tables are populated),
        # then hook up signals & do the initial draw
        if self.descriptor_path and self.descriptor_path.exists():
            self._load_descriptor()

        # now that the tables are filled, hook every QLineEdit to redraw
        def connect_all(table):
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    w = table.cellWidget(r, c)
                    if hasattr(w, "textChanged"):
                        w.textChanged.connect(self._update_preview)

        self.le_seq_order.textChanged.connect(self._update_preview)
        self.le_seq_repeats.textChanged.connect(self._update_preview)
        connect_all(self.pulse_tbl)
        connect_all(self.const_tbl)

        # initial draw
        self._update_preview()
        self.preview_window.show()

    def closeEvent(self, event):
        self.preview_window.close()
        super().closeEvent(event)

    def _add_row(self, table: QTableWidget):
        r = table.rowCount()
        table.insertRow(r)
        for c in range(table.columnCount()):
            le = QLineEdit()
            table.setCellWidget(r, c, le)
            le.textChanged.connect(self._update_preview)

    def _add_param_row(self):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for c in range(self.tbl.columnCount()):
            self.tbl.setCellWidget(r, c, QtWidgets.QLineEdit())

    def _load_descriptor(self):
        # force UTF-8 when reading
        with open(self.descriptor_path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)

        self.le_name.setText(d["experiment_type"])
        self.le_gen .setText(d.get("pulse_generator",""))

        for p in d.get("parameters",[]):
            r = self.tbl.rowCount(); self.tbl.insertRow(r)
            vals = [p["name"], p.get("label",""), p["type"],
                    str(p.get("default","")), str(p.get("min","")), str(p.get("max","")),
                    ",".join(p.get("choices",[])) if p["type"]=="choice" else p.get("unit","")]
            for c,v in enumerate(vals):
                le = QtWidgets.QLineEdit(v)
                self.tbl.setCellWidget(r,c,le)

        for p in d.get("pulses", []):
            r = self.pulse_tbl.rowCount(); self.pulse_tbl.insertRow(r)
            self.pulse_tbl.setCellWidget(r,0, QLineEdit(p.get("channel","")))
            self.pulse_tbl.setCellWidget(r,1, QLineEdit(str(p.get("start",""))))
            self.pulse_tbl.setCellWidget(r,2, QLineEdit(str(p.get("duration",""))))
            blocks = ",".join(p.get("blocks", []))
            self.pulse_tbl.setCellWidget(r,3, QLineEdit(blocks))

        # ── load constants ──
        for k, v in d.get("constants", {}).items():
            r = self.const_tbl.rowCount(); self.const_tbl.insertRow(r)
            self.const_tbl.setCellWidget(r, 0, QLineEdit(k))
            self.const_tbl.setCellWidget(r, 1, QLineEdit(str(v)))

    def _on_save(self):
        # close the floating preview
        if hasattr(self, 'preview_window') and self.preview_window.isVisible():
            self.preview_window.close()
        # validate & accept
        self._validate_and_accept()

    def _on_cancel(self):
        # close the floating preview
        if hasattr(self, 'preview_window') and self.preview_window.isVisible():
            self.preview_window.close()
        # reject this dialog
        super().reject()

    def _validate_and_accept(self):
        name = self.le_name.text().strip()
        if not name:
            QMessageBox.critical(self, "Error", "Experiment type is required"); return
        path = self.experiments_dir / f"{name}.yaml"
        if self.descriptor_path is None and path.exists():
            QMessageBox.critical(self, "Error", f"{name} already exists"); return
        self.descriptor_path = path
        super().accept()

    def save_descriptor(self):
        desc = {
            "experiment_type": self.le_name.text().strip(),
            "pulse_generator": self.le_gen.text().strip(),
            "parameters": [],
            "pulses": [],          
            "sequence": {}         
        }
        for r in range(self.tbl.rowCount()):
            cols = [ self.tbl.cellWidget(r,c).text().strip() for c in range(7) ]
            name,label,typ,default,minv,maxv,unitc = cols
            p = {"name": name, "label": label or name, "type": typ}
            if typ in ("int","float"):
                p["default"] = float(default) if typ=="float" else int(default)
                p["min"]     = float(minv) if typ=="float" else int(minv)
                p["max"]     = float(maxv) if typ=="float" else int(maxv)
                if unitc and typ=="float": p["unit"] = unitc
            elif typ=="choice":
                p["choices"] = [c.strip() for c in unitc.split(",")]
                p["default"] = default
            desc["parameters"].append(p)

        # ── collect all pulses ──
        for r in range(self.pulse_tbl.rowCount()):
            chan     = self.pulse_tbl.cellWidget(r,0).text().strip()
            start    = self.pulse_tbl.cellWidget(r,1).text().strip()
            duration = self.pulse_tbl.cellWidget(r,2).text().strip()
            blocks   = [
                b.strip() for b in
                self.pulse_tbl.cellWidget(r,3).text().split(",")
                if b.strip()
            ]
            entry = {"channel": chan, "start": start, "duration": duration}
            if blocks:
                entry["blocks"] = blocks
            desc["pulses"].append(entry)

        # ── collect constants ──
        if self.const_tbl.rowCount():
            desc["constants"] = {}
            for r in range(self.const_tbl.rowCount()):
                k = self.const_tbl.cellWidget(r, 0).text().strip()
                v = self.const_tbl.cellWidget(r, 1).text().strip()
                desc["constants"][k] = v

        # ── optional sequence.order & repeats ──
        order_txt   = self.le_seq_order.text().strip()
        repeats_txt = self.le_seq_repeats.text().strip()
        if order_txt:
            desc["sequence"]["order"]   = [s.strip() for s in order_txt.split(",")]
        if repeats_txt:
            desc["sequence"]["repeats"] = [s.strip() for s in repeats_txt.split(",")]

        # ── drop sequence: if still empty ──
        if not desc["sequence"]:
            del desc["sequence"]

        if hasattr(self, "le_seq_order"):
            order_txt   = self.le_seq_order .text().strip()
            repeats_txt = self.le_seq_repeats.text().strip()
            if order_txt:
                desc["sequence"]["order"]   = [s.strip() for s in order_txt.split(",")]
            if repeats_txt:
                desc["sequence"]["repeats"] = [s.strip() for s in repeats_txt.split(",")]

        # force UTF-8 when writing so that µ, π, etc. survive
        with open(self.descriptor_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(desc, f, sort_keys=False)

        parent = self.parent()
        if parent and hasattr(parent, '_reload_factory'):
            parent._reload_factory()
            parent.exp_combo.setCurrentText(self.le_name.text().strip())

        # if we're editing in‐place, immediately refresh the main GUI so that
        # its pulse‐diagram and parameter defaults reflect this change:
        if self.parent() and hasattr(self.parent(), '_reload_factory'):
            # reload descriptors, refill combo, and reapply defaults
            self.parent()._reload_factory()
            self.parent().exp_combo.currentTextChanged.emit(
                self.parent().exp_combo.currentText()
            )

    def _build_context(self):
        params = {}
        constants = {}

        # ── collect parameters into `params` ──
        for r in range(self.tbl.rowCount()):
            name = self.tbl.cellWidget(r, 0).text().strip()
            val  = self.tbl.cellWidget(r, 3).text().strip()
            try:
                params[name] = float(val)
            except:
                try:
                    params[name] = int(val)
                except:
                    params[name] = val

        # ── collect constants into `constants` ──
        for r in range(self.const_tbl.rowCount()):
            name = self.const_tbl.cellWidget(r, 0).text().strip()
            val  = self.const_tbl.cellWidget(r, 1).text().strip()
            try:
                constants[name] = float(val)
            except:
                try:
                    constants[name] = int(val)
                except:
                    constants[name] = val

        # expose both: flatten constants into top-level AND keep them under "constants"
        ctx = {}
        ctx.update(params)
        ctx.update(constants)          
        ctx["constants"] = constants   
        return ctx

    def _update_preview(self):
        self.preview.clear()

        # render each start/duration expression via Jinja
        ctx    = self._build_context()
        lanes  = CHANNEL_MAPPING
        pulses = []

        # build a list of (channel, start, dur) from the table:
        for r in range(self.pulse_tbl.rowCount()):
            chan       = self.pulse_tbl.cellWidget(r,0).text().strip()
            start_expr = self.pulse_tbl.cellWidget(r,1).text().strip()
            dur_expr   = self.pulse_tbl.cellWidget(r,2).text().strip()

            try:
                s = float(Template(start_expr).render(ctx))
                d = float(Template(dur_expr).render(ctx))
            except:
                continue

            pulses.append((chan, s, d))

        total_time = max((s + d) for (_c, s, d) in pulses) if pulses else 1.0

        # 1) draw baseline *only* in the gaps between pulses
        for chan, lane in self.channel_lanes.items():
            pen = pg.mkPen(self.channel_colors[chan], width=1)
            intervals = sorted((s, s + d) for (c, s, d) in pulses if c == chan)

            # merge overlaps
            merged = []
            for s0, e0 in intervals:
                if not merged or s0 > merged[-1][1]:
                    merged.append([s0, e0])
                else:
                    merged[-1][1] = max(merged[-1][1], e0)

            start0 = 0.0
            for s0, e0 in merged:
                if start0 < s0:
                    self.preview.plot([start0, s0], [lane, lane], pen=pen)
                start0 = e0
            if start0 < total_time:
                self.preview.plot([start0, total_time], [lane, lane], pen=pen)

        # 2) draw each pulse as a little box
        pulse_h = 0.8
        for chan, start, dur in pulses:
            lane = self.channel_lanes[chan]
            pen  = pg.mkPen(self.channel_colors[chan], width=2)
            x0, x1 = start, start + dur
            y0, y1 = lane, lane - pulse_h

            # rising edge
            self.preview.plot([x0, x0], [y0, y1], pen=pen)
            # top
            self.preview.plot([x0, x1], [y1, y1], pen=pen)
            # falling edge
            self.preview.plot([x1, x1], [y1, y0], pen=pen)

        # 3) relabel & rescale
        ticks = [(v, k) for k, v in self.channel_lanes.items()]
        self.preview.getAxis('left').setTicks([ticks])
        self.preview.setXRange(0, total_time * 1.05)
        max_lane = max(self.channel_lanes.values())
        self.preview.setYRange(-0.5, max_lane + 0.5)

    def _remove_pulse_row(self):
        row = self.pulse_tbl.currentRow()
        if row >= 0:
            self.pulse_tbl.removeRow(row)
            # re-draw the preview so the removed pulse vanishes
            self._update_preview()

    def _save_to_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Descriptor", "", "CSV (*.csv)")
        if not path:
            return

        import csv
        sections = [
            ("# PARAMETERS", self.tbl, ["name","label","type","default","min","max","unit"]),
            ("# CONSTANTS",  self.const_tbl, ["name","value"]),
            ("# PULSES",     self.pulse_tbl, ["channel","start","duration","blocks"])
        ]

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for header, table, cols in sections:
                writer.writerow([header])
                writer.writerow(cols)
                for r in range(table.rowCount()):
                    row = []
                    for c,_ in enumerate(cols):
                        w = table.cellWidget(r, c)
                        row.append(w.text().strip() if hasattr(w, "text") else "")
                    writer.writerow(row)
                writer.writerow([])  # blank line

        QMessageBox.information(self, "Saved", f"Descriptor exported to:\n{path}")

    def _load_from_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Descriptor", "", "CSV (*.csv)")
        if not path:
            return

        import csv
        mode = None
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f)
            self.tbl.setRowCount(0)
            self.const_tbl.setRowCount(0)
            self.pulse_tbl.setRowCount(0)

            for row in reader:
                if not row: 
                    continue
                if row[0].startswith("#"):
                    mode = row[0]
                    # skip next header line
                    next(reader, None)
                    continue

                if mode == "# PARAMETERS":
                    self._add_row(self.tbl)
                    for c, val in enumerate(row):
                        self.tbl.cellWidget(self.tbl.rowCount()-1, c).setText(val)
                elif mode == "# CONSTANTS":
                    self._add_row(self.const_tbl)
                    self.const_tbl.cellWidget(self.const_tbl.rowCount()-1, 0).setText(row[0])
                    self.const_tbl.cellWidget(self.const_tbl.rowCount()-1, 1).setText(row[1])
                elif mode == "# PULSES":
                    self._add_row(self.pulse_tbl)
                    for c, val in enumerate(row):
                        self.pulse_tbl.cellWidget(self.pulse_tbl.rowCount()-1, c).setText(val)

        # refresh preview
        self._update_preview()
        QMessageBox.information(self, "Loaded", f"Descriptor loaded from:\n{path}")

    def _download_template(self):
        """Export a blank, fully‐commented CSV scaffold for new experiments."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV Template", "experiment_template.csv", "CSV (*.csv)"
        )
        if not path:
            return
        
        template = [
            # ─── PARAMETERS ───────────────────────────────────────────────
            ["# PARAMETERS"],
            ["name","label","type","default","min","max","unit"],
            # example defaults—edit or delete these as needed:
            ["mw_duration","MW π-pulse duration","float","0.5","0.0","1000.0","µs"],
            ["tau","Inter-pulse delay","float","2.0","0.0","1000.0","µs"],
            ["laserduration","Laser repolarisation","float","100.0","0.0","1000.0","µs"],
            ["read_time","Readout window","float","10.0","0.0","1000.0","µs"],
            ["frames","Number of repetitions","int","1","1","9999",""],
            ["I_pulse","I-pulse duration","float","0.25","0.0","1000.0","µs"],
            ["Q_pulse","Q-pulse duration","float","0.25","0.0","1000.0","µs"],
            [],

            # ─── CONSTANTS ────────────────────────────────────────────────
            ["# CONSTANTS"],
            ["name","value"],
            # example constants—edit or delete these as needed:
            ["buffer_between_pulses","1"],
            ["readout_and_repol_gap","2"],
            ["read_trigger_duration","2"],
            [],

            # ─── PULSES ──────────────────────────────────────────────────
            ["# PULSES"],
            ["channel","start","duration","blocks"],
            # example pulses—edit or delete these as needed:
            ["START","0","1","wait_loop"],
            ["MW","{{ buffer_between_pulses }}","{{ mw_duration }}","wait_loop"],
            ["LASER","{{ buffer_between_pulses*2 + mw_duration }}","{{ laserduration }}","wait_loop"],
            [],
            # Now add your own pulses below:
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in template:
                writer.writerow(row)

        QMessageBox.information(self, "Template Saved", f"CSV template written to:\n{path}")
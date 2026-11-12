# tab_helpers.py

# ────────────────────────────────────────────────────────────────
# Small widgets shared by the RF and Controls tabs (Version 4).
# ────────────────────────────────────────────────────────────────

from PyQt6 import QtCore, QtWidgets

DASH = "—"


def spin(value, lo, hi, decimals=3, step=0.1, suffix=""):
    """Double spin box that reports a change only when the edit is finished."""
    box = QtWidgets.QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(" " + suffix)
    box.setKeyboardTracking(False)
    return box


def int_spin(value, lo, hi, step=1):
    box = QtWidgets.QSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


def note(text):
    """Gray, word-wrapped explanatory line."""
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: gray;")
    return label


class Readout(QtWidgets.QGroupBox):
    """Titled box of 'name: value' rows. Values can be selected and copied."""

    def __init__(self, title, rows, parent=None):
        super().__init__(title, parent)
        form = QtWidgets.QFormLayout(self)
        self._values = {}
        for key, name in rows:
            value = QtWidgets.QLabel(DASH)
            value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setWordWrap(True)
            form.addRow(name, value)
            self._values[key] = value

    def set(self, key, text):
        value = self._values[key]
        value.setText(str(text))
        # A wrapped label changes height with its text; tell the form to lay the rows out again.
        value.updateGeometry()
        self.layout().invalidate()

    def text(self, key):
        return self._values[key].text()

    def clear(self):
        for value in self._values.values():
            value.setText(DASH)


def side_by_side(left, right, left_width=360):
    """Inputs and readouts on the left (scrollable), plots on the right."""
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setWidget(left)
    scroll.setMinimumWidth(left_width)
    scroll.setMaximumWidth(left_width + 80)

    holder = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(scroll)
    row.addWidget(right, 1)
    return holder


def missing_backend(exc):
    """Shown in place of a tab when the qzabre package cannot be imported."""
    label = QtWidgets.QLabel(
        "This tab needs the qzabre package (the folder 'qzabre' next to this file, "
        "or 'pip install -e' of the QZabre repository).\n\n"
        f"Import error: {exc}"
    )
    label.setWordWrap(True)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    return label

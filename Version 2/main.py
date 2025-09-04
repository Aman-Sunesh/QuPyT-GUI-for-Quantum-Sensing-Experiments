# main.py

# ────────────────────────────────────────────────────────────────
# Entrypoint for the QuPyt ODMR GUI application.
# Initializes the Qt application, launches the main window,
# and handles any startup errors gracefully.
# ────────────────────────────────────────────────────────────────

import sys
import pathlib

# Ensure the local package root is on sys.path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import logging
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6 import QtCore, QtGui

def main
    """
    Launch the QuPyt ODMR GUI.

    - Creates the QApplication.
    - Instantiates the ODMRGui main window.
    - Sets a default size and shows the window.
    - Enters the Qt event loop.
    - On any exception during startup, shows a critical dialog
      and logs the error before exiting.
    """
    try:
        try:
            QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
                QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        except AttributeError:
            pass

        # Create the Qt application object
        app = QApplication(sys.argv)

        # Import GUI code only after the app exists
        from GUI import odmr_gui
        sys.excepthook = odmr_gui.excepthook

        # Instantiate and configure the main window
        win = odmr_gui.ODMRGui()

        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            width_frac  = 0.55  
            height_frac = 0.95  
            w = int(geo.width() * width_frac)
            h = int(geo.height() * height_frac)
            win.resize(w, h)
            win.setMaximumHeight(h)

        win.show()
        sys.exit(app.exec())   # Start the Qt event loop

    except Exception as e:
        logging.error("Failed to start GUI", exc_info=True)

        # Only show a QMessageBox if a QApplication exists
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Startup Error", str(e))
        else:
            print(f"Startup Error: {e}", file=sys.stderr)

        sys.exit(1)

if __name__ == '__main__':
    main()




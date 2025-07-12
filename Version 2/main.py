# main.py

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import logging
from PyQt6.QtWidgets import QApplication, QMessageBox
from GUI.odmr_gui import ODMRGui

def main():
    try:
        app = QApplication(sys.argv)
        win = ODMRGui()
        win.resize(1000, 1100)
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", str(e))
        logging.error("Failed to start GUI", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

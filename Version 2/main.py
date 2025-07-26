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
from GUI.odmr_gui import ODMRGui

def main():
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
        # Create the Qt application object
        app = QApplication(sys.argv)
        
        # Instantiate and configure the main window
        win = ODMRGui()
        win.resize(1000, 1100)  # Default window dimensions
        win.show()              # Default window dimensions

        # Start the Qt event loop
        sys.exit(app.exec())
        
    except Exception as e:
        # Show a critical error message if startup fails
        QMessageBox.critical(None, "Startup Error", str(e))
        logging.error("Failed to start GUI", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

"""Launch only the offline Antenna response tab."""

import sys

from PyQt6 import QtWidgets

from antenna_response_tab import AntennaResponseTab


def create_window():
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Antenna response")
    window.setCentralWidget(AntennaResponseTab())
    window.resize(1400, 800)
    return window


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = create_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

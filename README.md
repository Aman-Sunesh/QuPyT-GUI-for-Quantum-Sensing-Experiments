# QuPyt GUI

This repository contains a graphical user interface for running quantum sensing experiments using the QuPyt framework. It is designed to streamline the experimental workflow by eliminating repetitive terminal commands and providing visual tools for experiment setup, execution, and data visualization.

--- 

## Repository Structure

```
QuPyt-GUI-for-Quantum-Sensing-Experiments/
├── Modified QuPyt Files/                        # Replacements for QuPyt source files to enable GUI integration
│   ├── SequenceDesigner.py                      # Updated for GUI-based sequence generation
│   ├── sensors.py                               # Modified to interface with GUI sensor configuration
│   └── yaml_sequence.py                         # Adapted to support dynamic YAML-based sequence creation

├── Version 1/                                   # Legacy scripts and standalone helpers (deprecated in v2)
│   ├── user_pulse_seq.py                        # Hardcoded pulse sequence example (used in earlier prototypes)
│   └── stop_pb.py                               # Script to stop PulseBlaster pulse generation manually

├── Version 2/                                   # Main GUI application source directory
│   ├── __init__.py                              # Marks folder as a Python package
│   ├── channels.py                              # Defines TTL channel labels and hardware mappings
│   ├── experiment_editor.py                     # GUI dialog for editing experiment parameters and pulse sequences
│   ├── experiment_factory.py                    # Backend logic to generate experiment YAML files
│   ├── experiment_yaml.py                       # YAML construction helper module
│   ├── generic_generator.py                     # Shared logic for generating pulse sequence structures
│   ├── main.py                                  # GUI launcher (entry point of the application)
│   ├── odmr_gui.py                              # Core GUI logic for ODMR experiment integration
│   ├── odmr_sample_pulse_sequence.py            # Example pulse sequence for an ODMR run
│   ├── power_supply.py                          # Korad KC3405 power supply control interface
│   ├── stop_pb.py                               # Used internally to safely stop ongoing PB activity
│   └── utils.py                                 # Utility functions (file I/O, formatting, YAML validators)

├── experiments/                                 # All experiment YAML descriptors (editable via GUI)
│   ├── ODMR.yaml                                 # Descriptor for Optically Detected Magnetic Resonance scan
│   └── XY8.yaml                                  # Descriptor for XY8 decoupling pulse sequence

├── QuPyt GUI User Manual.pdf                    # Comprehensive user manual detailing all GUI features
├── requirements.txt                             # List of required Python packages for setting up environment
└── README.md                                    # Project overview, setup instructions, and launch guide
```

---

## Installation

### 1. Clone Repository
Download this repository or clone it using:
```bash
git clone https://github.com/Aman-Sunesh/QuPyT-GUI-for-Quantum-Sensing-Experiments.git
```

### 2. Setup Virtual Environment & Dependencies
```bash
python -m venv venv
.\venv\Scripts\activate

cd "<path-to-QuPyt-master>"
pip install -e .

pip install matplotlib nidaqmx numpy pulsestreamer pypylon pyserial pyvisa pyvisa-py PyYAML termcolor tqdm watchdog PyQt6 pyqtgraph pydantic windfreak harvester harvesters jinja2 scipy
```

---

## Launching the GUI

### Option 1: Terminal
```bash
cd /d "C:/path/to/QuPyt-master"
.env\Scripts\Activate.ps1
python -m GUI.main
```

### Option 2: Desktop Shortcut
Create a `.bat` file with:
```bat
@echo off
cd /d "C:/path/to/QuPyt-master"
set PYTHONPATH=%CD%
"%CD%\venv\Scripts\pythonw.exe" -m GUI.main
```

---

## Modified Core Files

The GUI requires changes to three QuPyt source files:
- `SequenceDesigner.py`
- `sensors.py`
- `yaml_sequence.py`

Make sure to replace these with the versions found in the `Modified QuPyt Files/` folder.

---

## Full Manual

See `QuPyt_GUI_User_Manual.pdf` (included in the repository) for:
- GUI component breakdown
- Experiment configuration
- YAML descriptor structure
- Pulse sequence code generation
- Troubleshooting tips

---

## References

- [QuPyt (Original)](https://github.com/KarDB/QuPyt/tree/master/source)
- [QuPyt-GUI by Aman Sunesh](https://github.com/Aman-Sunesh/QuPyT-GUI-for-Quantum-Sensing-Experiments)
- [PyQt6 Documentation](https://pypi.org/project/PyQt6/)

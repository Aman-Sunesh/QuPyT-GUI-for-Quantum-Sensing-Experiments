# Version 4

Version 3 plus two new tabs. Start it the same way: `python main.py` from this folder.

## RF tab

| Page | What it does | Kind of result |
|---|---|---|
| Microstrip | Trace impedance, effective permittivity, width for a target impedance, quarter wavelength | Model (Hammerstad-Jensen, quasi-static) |
| Loop antenna | Incident power to loop current, field, Rabi frequency and pi-pulse time. Series-C plus shunt-C match and its S11 curve | Model. The loop resistance is an estimate |
| S11 viewer | Opens a VNA `.s1p` file. Reports the resonance, minimum S11, bandwidth and loaded Q | From the file you load |

## Controls tab

Simulation only. Nothing in this tab drives the microwave source.

| Page | What it does |
|---|---|
| Dip tracking | Probes both sides of an ODMR dip and steers the microwave back to the center. Shows tracking error under field drift and photon shot noise |
| Lock-in | Dithers the microwave frequency, demodulates, and closes a PI loop. Shows whether the lock holds for a given drift rate and start offset |

Use these to choose loop gains before building a real loop. Every number is a model value for the inputs shown, not a measurement. The dip width, contrast and photon rate defaults are placeholders: use values fitted from your own spectra.

## Files added in this version

- `rf_tab.py`, `controls_tab.py`: the two tabs.
- `tab_helpers.py`: small widgets both tabs share.
- `qzabre/`: the math the tabs call (NV relations, ODMR lineshape, microstrip, loop antenna, S11 analysis, tracking and lock-in simulators).
- `odmr_gui.py`: two imports and the lines that add the tabs. Nothing else changed from Version 3.
- `experiments/`: the four experiment descriptors from Version 2 (AOM_Test, NV-ESR, ODMR, XY8), copied unchanged. The GUI needs this folder to start.

No new packages are needed beyond `requirements.txt` (PyQt6, pyqtgraph, numpy, scipy).

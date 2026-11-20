# Version 4

Version 3 plus three new tabs. Start it the same way: `python main.py` from this folder.

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

## Antenna response tab

Reads saved EduControl Rabi run folders only; it does not contact hardware. Add one run folder or scan a parent folder containing several runs. The tab reads the executed microwave settings from `seq0/src/rabi.seq`, using `$freq + $df` as the driven frequency, and fits `seq0/eval/rabi.h5` with `qzabre.pulsed_io`.

The table compares the fitted Rabi frequency with EduControl's `freq0` from `scalarData.h5`. A fit is excluded when its relative uncertainty exceeds 10% or the scan window contains less than one fitted period; excluded points remain visible in gray. Fits with 5–10% relative uncertainty or only one to two periods receive a warning. A difference above 10% from EduControl also produces a warning. R² is displayed as information only. The amplitude-response plot connects only runs with the same driven frequency and acquisition date; the frequency-response plot connects only runs with the same amplitude and date. The table can be exported to CSV.

## Files added in this version

- `rf_tab.py`, `controls_tab.py`, `antenna_response_tab.py`: the three tabs.
- `tab_helpers.py`: small widgets the added tabs share.
- `qzabre/`: the math the tabs call (NV relations, ODMR lineshape, pulsed I/O and Rabi fitting, microstrip, loop antenna, S11 analysis, tracking and lock-in simulators).
- `odmr_gui.py`: imports and the lines that add the tabs. Nothing else changed from Version 3.
- `experiments/`: the four experiment descriptors from Version 2 (AOM_Test, NV-ESR, ODMR, XY8), copied unchanged. The GUI needs this folder to start.

The added tabs use PyQt6, pyqtgraph, NumPy and SciPy. The Antenna response tab also needs h5py to read EduControl files.

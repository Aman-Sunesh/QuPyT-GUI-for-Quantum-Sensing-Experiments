# channels.py

# ────────────────────────────────────────────────────────────────
# This module defines the hardware channel mapping for the QuPyt
# ODMR experiment control. Each key corresponds to a logical signal
# name used throughout the code, mapped to the numeric output channel
# index on the PulseBlaster (or other timing hardware).
# ────────────────────────────────────────────────────────────────

CHANNEL_MAPPING = {
    'READ':  0,  # Channel 0: APD readout/reset line
    'START': 1,  # Channel 1: Experiment start trigger
    'LASER': 2,  # Channel 2: Laser on/off control
    'MW':    3,  # Channel 3: Microwave pulse output
    'I':     4,  # Channel 4: IQ mixer I-component
    'Q':     5,  # Channel 5: IQ mixer Q-component
}

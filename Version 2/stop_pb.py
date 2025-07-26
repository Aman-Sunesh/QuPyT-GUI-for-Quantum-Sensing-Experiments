# stop_pb.py

# ────────────────────────────────────────────────────────────────
# Utility for safely halting and closing the PulseBlaster device.
# Ensures any running pulse programs are stopped, the board is
# optionally reset, and resources are freed.
# ────────────────────────────────────────────────────────────────

import spinapi

# Index of the PulseBlaster board to control
BOARD = 0

def stop_pulse_blaster():
    """
    Initialize the PulseBlaster API, stop any active pulse sequence,
    attempt a reset if supported, and close the connection.
    """

    # Initialize the API; pb_init returns 0 on success
    rv = spinapi.pb_init()
    
    if rv != 0:
        # Warn but continue even if initialization failed
        print(f"Warning: pb_init() returned {rv} - continuing anyway")

    # Select the board to operate on
    spinapi.pb_select_board(BOARD)

    # Immediately stop any running pulse program
    spinapi.pb_stop()

    # Some spinapi versions support pb_reset; ignore AttributeError if not present
    try:
        spinapi.pb_reset()
    except AttributeError:
        # Older API versions do not implement pb_reset()
        pass

    # Close the API session and free resources
    spinapi.pb_close()

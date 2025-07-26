# This script provides a utility function to safely stop any running PulseBlaster
# pulse sequence. It is typically called when the user presses the "Stop" button
# in the GUI or if the program needs to abort an experiment. It ensures the device
# is halted, reset, and properly closed to prevent hardware lock-ups.

import spinapi

BOARD = 0

def stop_pulse_blaster():
    """
    Stop the PulseBlaster safely:
    1. Initialize connection to the board.
    2. Select the target board.
    3. Stop any running pulse program immediately.
    4. Attempt to reset the board for a clean state.
    5. Close the API connection.
    """

    # Initialize the PulseBlaster API
    rv = spinapi.pb_init()
    if rv != 0:
        # If initialization fails, print a warning but continue attempting cleanup
        print(f"Warning: pb_init() returned {rv} – continuing anyway")

    # Select the board (index = BOARD) and stop any running program
    spinapi.pb_select_board(BOARD) 
    spinapi.pb_stop()

    try:
        # Reset the board to a clean state (some API versions may lack this method)
        spinapi.pb_reset()
    except AttributeError:
        # If pb_reset() is not available, ignore the error
        pass

    # Close the connection to release resources
    spinapi.pb_close()

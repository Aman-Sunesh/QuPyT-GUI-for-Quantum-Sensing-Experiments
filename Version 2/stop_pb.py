# stop_pb.py

import spinapi

BOARD = 0

def stop_pulse_blaster():
    rv = spinapi.pb_init()
    if rv != 0:
        print(f"Warning: pb_init() returned {rv} - continuing anyway")
    spinapi.pb_select_board(BOARD)
    spinapi.pb_stop()

    try:
        spinapi.pb_reset()
    except AttributeError:
        pass

    spinapi.pb_close()

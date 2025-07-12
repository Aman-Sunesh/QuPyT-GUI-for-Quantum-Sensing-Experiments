"""
Auto-generated pulse sequence module for the QuPyt ODMR GUI.
Such a file is created each time you press “Start” in the GUI to reflect
your current settings (START, MW, LASER, READ pulses, loops, etc.).
Do not edit manually—re-run from the GUI to regenerate.
"""

import logging
from qupyt.pulse_sequences.yaml_sequence import YamlSequence

from qupyt import set_up
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_sequence(params: dict) -> dict:
    logging.debug("Generating pulse sequence with %s", params)
    seq = YamlSequence(duration=1000000.0)
    seq.add_pulse("START", 0.0, 1000.0, sequence_blocks=["wait_loop"])
    seq.add_pulse("MW", 1000.0, 1000.0, sequence_blocks=['wait_loop', 'block_0'])
    seq.add_pulse("LASER", 3000.0, 5000.0, sequence_blocks=['wait_loop', 'block_0'])
    seq.add_pulse("LASER", 10000.0, 5000.0, sequence_blocks=['wait_loop', 'block_0'])
    seq.add_pulse("READ", 2000.0, 2000.0, sequence_blocks=['block_0'])

    seq.sequencing_order   = ['wait_loop', 'block_0']
    seq.sequencing_repeats = [1, 50]
    # write the pulse-sequence YAML into the sequence directory
    seq_dir  = set_up.get_seq_dir()
    seq_file = seq_dir / "sequence.yaml"
    seq.write(seq_file)
    logging.info("Pulse sequence generated.")
    return {}

"""
Generate a sample ODMR pulse sequence file.
This pulse sequence is not suited to be used directly in a measurement.
Instead, the pulse sequence needs to be adjusted to the particularities
of the hardware used.
"""

# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=logging-not-lazy

import logging
from qupyt.pulse_sequences.yaml_sequence import YamlSequence

# Set up logging for this module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_sequence(params: dict):
    logging.debug("Generating pulse sequence with parameters: %s", params)
    return gen_esr(
        params.get('mw_duration', 10),
        params.get('laserduration', 100),
        params.get('readout_time', 50),
        params.get('referenced_measurements', 100),
        params.get('max_framerate', 1000)
    )

def gen_esr(mw_duration: float, laserduration: float, readout_time: float, referenced_measurements: int, max_framerate: float = 10000) -> dict:
    logging.debug("Starting gen_esr with laserduration=%f, readout_time=%f, referenced_measurements=%d, max_framerate=%f",
                  laserduration, readout_time, referenced_measurements, max_framerate)
    
    # Define basic timing parameters
    readout_and_repol_gap = 2
    buffer_between_pulses = 1
    read_trigger_duration = 2

    # Compute the half-block time using only the laser and gap.
    time_half = buffer_between_pulses * 3 + mw_duration + laserduration + readout_and_repol_gap
    time_half = max(time_half, 1/max_framerate * 1e6)
    total_time = 2 * time_half
    logging.debug("Computed time_half=%.2f and total_time=%.2f", time_half, total_time)

    # Create the pulse sequence object with the total duration.
    esr = YamlSequence(duration=total_time)

    # Add a START pulse that spans the entire measurement period.
    esr.add_pulse("START", 0.0, 10, sequence_blocks=['wait_loop'])
    logging.debug("Added START pulse spanning 0.0 to %.2f", total_time)

    # Add the microwave pulse at the beginning of the pulse sequece
    # starting after one buffer time. There is only one microwave pulse
    # per sequence block.
    esr.add_pulse(
        "MW",  # pulse channel, see YAML config file.
        buffer_between_pulses,  # Starting time of the pulse.
        mw_duration,  # Pulse duration.
        # This pulse appears in two sequence blocks.
        sequence_blocks=['wait_loop', 'block_0']
    )

    # Add the LASER and READ pulses in two blocks.
    for i in range(2):
        logging.debug("Adding pulses for block %d", i)
        # LASER pulse for readout: placed after a small delay.
        esr.add_pulse(
            "LASER",
            i * time_half + 2 * buffer_between_pulses
            + mw_duration,
            readout_time,
            sequence_blocks=['wait_loop', 'block_0']
        )

        esr.add_pulse(
            "LASER",
            i * time_half + 2 * buffer_between_pulses + mw_duration + readout_time + readout_and_repol_gap,
            laserduration - readout_time,
            sequence_blocks=['wait_loop', 'block_0']
        )

        # READ pulse to trigger DAQ readout.
        esr.add_pulse(
            "READ",
            i * time_half + buffer_between_pulses + mw_duration,
            read_trigger_duration,
            sequence_blocks=['block_0']
        )
    esr.sequencing_order = ['wait_loop', 'block_0']
    esr.sequencing_repeats = [100, int(referenced_measurements/2) + 10]
    logging.debug("Sequencing order: %s, sequencing repeats: %s", esr.sequencing_order, esr.sequencing_repeats)
    
    esr.write()
    logging.info("Pulse sequence generated successfully.")
    return {}

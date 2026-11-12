"""Auto-generated QuPyt pulse-sequence module."""

import logging
from qupyt.pulse_sequences.yaml_sequence import YamlSequence

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

SEQUENCE_SPECS = [{'duration': 2400.0,
  'pulses': [{'channel': 'START', 'start': 0.0, 'duration': 10.0, 'blocks': ['wait_loop']},
             {'channel': 'MW', 'start': 1176.0, 'duration': 10.0, 'blocks': ['block_0']},
             {'channel': 'LASER', 'start': 0.0, 'duration': 2400.0, 'blocks': ['block_0']},
             {'channel': 'READ', 'start': 1187.0, 'duration': 2.0, 'blocks': ['block_0']},
             {'channel': 'READ', 'start': 2280.0, 'duration': 2.0, 'blocks': ['block_0']}],
  'order': ['wait_loop', 'block_0'],
  'repeats': [1, 100]}]

def generate_sequence(params: dict) -> dict:
    requested_steps = int(params.get("pulse_sequence_steps", len(SEQUENCE_SPECS)))
    if requested_steps != len(SEQUENCE_SPECS):
        raise ValueError('Generated sequence count does not match pulse_sequence_steps')

    for ps_step, spec in enumerate(SEQUENCE_SPECS):
        seq = YamlSequence(duration=spec["duration"])

        for pulse in spec["pulses"]:
            seq.add_pulse(
                pulse["channel"],
                pulse["start"],
                pulse["duration"],
                sequence_blocks=pulse["blocks"],
            )

        seq.sequencing_order = spec["order"]
        seq.sequencing_repeats = spec["repeats"]
        seq.write(ps_step)

    logging.info("Generated %d pulse sequence(s).", len(SEQUENCE_SPECS))
    return {}

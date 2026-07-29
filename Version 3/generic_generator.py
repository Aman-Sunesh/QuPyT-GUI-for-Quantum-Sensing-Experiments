# generic_generator.py

# ────────────────────────────────────────────────────────────────
# Utility for generating PulseBlaster-compatible Python modules
# from a descriptor dictionary. Uses Jinja2 for templated expressions
# and Qupyt’s YamlSequence to write out a pulse-sequence YAML file.
# ────────────────────────────────────────────────────────────────

import logging
import pprint
from copy import deepcopy
import numpy as np
from jinja2 import Template

def generate_from_descriptor(desc: dict, params: dict, output_path: str = "user_pulse_seq.py"):
    """
    Generate a Python pulse-sequence module from a descriptor.

        The generated module writes:

        ~/.qupyt/sequences/sequence_0.yaml
        ~/.qupyt/sequences/sequence_1.yaml
        ...

    depending on pulse_sequence_steps.

    """
    
    def coerce_constant(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value

        if number.is_integer():
            return int(number)

        return number

    def render_number(expression, context):
        rendered = Template(str(expression)).render(context)
        return float(rendered)

    pulse_sequence_steps = int(
        params.get("pulse_sequence_steps", 1)
    )

    if pulse_sequence_steps < 1:
        raise ValueError(
            "pulse_sequence_steps must be at least 1"
        )

    sweep_param_raw = params.get("sweep_param")
    sweep_param = (
        str(sweep_param_raw).strip()
        if sweep_param_raw not in (None, "")
        else None
    )

    raw_sweep_values = params.get("sweep_values", [])

    if isinstance(raw_sweep_values, np.ndarray):
        raw_sweep_values = raw_sweep_values.tolist()
    elif not isinstance(raw_sweep_values, (list, tuple)):
        raw_sweep_values = [raw_sweep_values]

    raw_sweep_values = list(raw_sweep_values)

    if sweep_param is None:
        sweep_values = [None] * pulse_sequence_steps
    elif len(raw_sweep_values) == pulse_sequence_steps:
        sweep_values = raw_sweep_values
    elif len(raw_sweep_values) == 2:
        sweep_values = np.linspace(
            float(raw_sweep_values[0]),
            float(raw_sweep_values[1]),
            pulse_sequence_steps,
        ).tolist()
    elif pulse_sequence_steps == 1 and raw_sweep_values:
        sweep_values = [raw_sweep_values[0]]
    else:
        raise ValueError(
            "sweep_values must contain either two endpoints or exactly "
            "pulse_sequence_steps values"
        )

    constants = {
        key: coerce_constant(value)
        for key, value in desc.get("constants", {}).items()
    }

    sequence_specs = []

    for ps_step in range(pulse_sequence_steps):
        step_params = deepcopy(params)

        if sweep_param is not None:
            step_params[sweep_param] = sweep_values[ps_step]

        context = {}
        context.update(step_params)
        context.update(constants)
        context["constants"] = constants

        pulses = []

        # Preserve the GUI's dedicated START-pulse behavior.
        start_duration = float(
            context.get("start_pulse_dur", 1.0)
        )

        if start_duration <= 0:
            raise ValueError(
                "start_pulse_dur must be greater than zero"
            )

        pulses.append(
            {
                "channel": "START",
                "start": 0.0,
                "duration": start_duration,
                "blocks": ["wait_loop"],
            }
        )

        maximum_end = start_duration

        for pulse in desc.get("pulses", []):
            channel = str(pulse.get("channel", "")).strip()

            if not channel or channel == "START":
                continue

            start = render_number(
                pulse.get("start", 0),
                context,
            )
            duration = render_number(
                pulse.get("duration", 0),
                context,
            )

            if start < 0:
                raise ValueError(
                    f"{channel} pulse start cannot be negative"
                )

            if duration < 0:
                raise ValueError(
                    f"{channel} pulse duration cannot be negative"
                )

            blocks = pulse.get("blocks", ["block_0"])

            if not blocks:
                blocks = ["block_0"]
            elif isinstance(blocks, str):
                blocks = [blocks]
            else:
                blocks = list(blocks)

            pulses.append(
                {
                    "channel": channel,
                    "start": start,
                    "duration": duration,
                    "blocks": blocks,
                }
            )

            maximum_end = max(
                maximum_end,
                start + duration,
            )

        if "total_duration" in desc:
            total_duration = render_number(
                desc["total_duration"],
                context,
            )
        else:
            total_duration = maximum_end

        total_duration = max(
            float(total_duration),
            maximum_end,
            start_duration,
        )

        sequence_description = desc.get(
            "sequence",
            {},
        )

        raw_order = sequence_description.get(
            "order",
            ["wait_loop", "block_0"],
        )

        if isinstance(raw_order, str):
            order = [
                value.strip()
                for value in raw_order.split(",")
                if value.strip()
            ]
        else:
            order = [
                str(value)
                for value in raw_order
            ]

        repeat_expressions = sequence_description.get(
            "repeats",
            [],
        )

        if not isinstance(
            repeat_expressions,
            (list, tuple),
        ):
            repeat_expressions = [
                repeat_expressions
            ]

        if repeat_expressions:
            repeats = [
                int(
                    float(
                        Template(str(expression)).render(
                            context
                        )
                    )
                )
                for expression in repeat_expressions
            ]
        else:
            frames = int(
                context.get("frames", 1)
            )
            reference_channels = int(
                context.get("ref_channels", 1)
            )

            if reference_channels < 1:
                raise ValueError(
                    "ref_channels must be at least 1."
                )

            if frames % reference_channels != 0:
                raise ValueError(
                    "frames must be divisible by ref_channels."
                )

            cycles_per_acquisition = (
                frames // reference_channels
            )
            repeats = [
                (
                    1
                    if block == "wait_loop"
                    else cycles_per_acquisition
                )
                for block in order
            ]

        if len(order) != len(repeats):
            raise ValueError(
                "sequence.order and sequence.repeats must have "
                "the same number of entries"
            )

        if any(repeat < 1 for repeat in repeats):
            raise ValueError(
                "Every sequence repeat count must be at least 1."
            )

        available_blocks = {
            block
            for pulse in pulses
            for block in pulse["blocks"]
        }

        missing_blocks = [
            block
            for block in order
            if block not in available_blocks
        ]

        if missing_blocks:
            raise ValueError(
                "sequence.order references blocks with no pulses: "
                + ", ".join(missing_blocks)
            )

        if total_duration <= 0:
            raise ValueError(
                "Pulse-sequence total duration must be positive."
            )


        sequence_specs.append(
            {
                "duration": total_duration,
                "pulses": pulses,
                "order": order,
                "repeats": repeats,
            }
        )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(
            '"""Auto-generated QuPyt pulse-sequence module."""\n\n'
        )
        file.write("import logging\n")
        file.write(
            "from qupyt.pulse_sequences.yaml_sequence "
            "import YamlSequence\n\n"
        )
        file.write(
            "logging.basicConfig("
            "level=logging.DEBUG, "
            'format="%(asctime)s - %(levelname)s - %(message)s"'
            ")\n\n"
        )

        file.write("SEQUENCE_SPECS = ")
        file.write(
            pprint.pformat(
                sequence_specs,
                width=100,
                sort_dicts=False,
            )
        )
        file.write("\n\n")

        file.write(
            "def generate_sequence(params: dict) -> dict:\n"
        )
        file.write(
            "    requested_steps = int("
            'params.get("pulse_sequence_steps", len(SEQUENCE_SPECS))'
            ")\n"
        )
        file.write(
            "    if requested_steps != len(SEQUENCE_SPECS):\n"
        )
        file.write(
            "        raise ValueError("
            "'Generated sequence count does not match "
            "pulse_sequence_steps'"
            ")\n\n"
        )
        file.write(
            "    for ps_step, spec in enumerate(SEQUENCE_SPECS):\n"
        )
        file.write(
            "        seq = YamlSequence("
            'duration=spec["duration"]'
            ")\n\n"
        )
        file.write(
            '        for pulse in spec["pulses"]:\n'
        )
        file.write(
            "            seq.add_pulse(\n"
        )
        file.write(
            '                pulse["channel"],\n'
        )
        file.write(
            '                pulse["start"],\n'
        )
        file.write(
            '                pulse["duration"],\n'
        )
        file.write(
            '                sequence_blocks=pulse["blocks"],\n'
        )
        file.write(
            "            )\n\n"
        )
        file.write(
            '        seq.sequencing_order = spec["order"]\n'
        )
        file.write(
            '        seq.sequencing_repeats = spec["repeats"]\n'
        )
        file.write(
            "        seq.write(ps_step)\n\n"
        )
        file.write(
            '    logging.info("Generated %d pulse sequence(s).", '
            "len(SEQUENCE_SPECS))\n"
        )
        file.write("    return {}\n")

    logging.info(
        "Wrote pulse-sequence module to %s",
        output_path,
    )

    return {}
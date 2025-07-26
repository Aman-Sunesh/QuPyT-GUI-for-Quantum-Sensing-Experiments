# generic_generator.py

# ────────────────────────────────────────────────────────────────
# Utility for generating PulseBlaster-compatible Python modules
# from a descriptor dictionary. Uses Jinja2 for templated expressions
# and Qupyt’s YamlSequence to write out a pulse-sequence YAML file.
# ────────────────────────────────────────────────────────────────

import logging
from jinja2 import Template
from qupyt.pulse_sequences.yaml_sequence import YamlSequence

def generate_from_descriptor(desc: dict, params: dict, output_path: str = "user_pulse_seq.py"):
    """
    Generate a Python pulse-sequence module from a descriptor.

    Args:
        desc: Descriptor dict defining 'constants', 'pulses', and 'sequence' metadata.
        params: Runtime parameters including 'time_unit', 'frames', etc.
        output_path: Path to write the generated .py module.

    Returns:
        An empty dict (placeholder for possible future metadata).
    """
    
    # Build Jinja context from params
    ctx = params.copy()

    # Determine time unit and scaling factor
    unit        = params.get('time_unit', 'µs')
    unit_factor = {'ns': 1e-3, 'µs': 1.0, 'ms': 1e3}[unit]

    # Render and coerce descriptor constants into numeric values
    ctx["constants"] = {
        k: float(v) if str(v).replace('.','',1).isdigit() else int(v)
        for k,v in desc.get("constants",{}).items()
    }

    # Helper to render expressions and convert to seconds (or chosen unit)
    def R(expr):
        return float(Template(str(expr)).render(ctx)) * unit_factor

    # Compute total sequence duration if not explicitly given
    total = desc.get(
        "total_duration",
        max(R(p["start"]) + R(p["duration"]) for p in desc["pulses"])
    )

    # Ensure at least the START pulse duration is covered
    start_dur = float(ctx.get("start_pulse_dur", 1.0)) * unit_factor
    total = max(total, start_dur)

   # Begin writing the output Python module
    with open(output_path, 'w', encoding='utf-8') as f:
        # Module docstring
        f.write('\"\"\"\n')
        f.write('Generate a sample ODMR pulse sequence file.\n')
        f.write('This pulse sequence is not suited to be used directly in a measurement.\n')
        f.write('…adjust to your hardware…\n')
        f.write('\"\"\"\n\n')

        # Imports and logger setup
        f.write('import logging\n')
        f.write('from qupyt.pulse_sequences.yaml_sequence import YamlSequence\n\n')
        f.write('from qupyt import set_up\n')
        f.write('logging.basicConfig(level=logging.DEBUG, '
                'format="%(asctime)s - %(levelname)s - %(message)s")\n\n')

        # Define main generation function
        f.write('def generate_sequence(params: dict) -> dict:\n')
        f.write('    logging.debug("Generating pulse sequence with %s", params)\n')
        f.write(f'    seq = YamlSequence(duration={total})\n')

        # Add the START trigger pulse
        start_dur = ctx.get("start_pulse_dur", 1.0)

        # Emit each non-START pulse from the descriptor
        f.write(f'    seq.add_pulse("START", 0.0, {start_dur}, sequence_blocks=["wait_loop"])\n')
        for p in desc["pulses"]:
            if p["channel"] == "START":
                continue
            s = R(p["start"]); d = R(p["duration"])
            blocks = p.get("blocks", ["block_0"])
            f.write(f'    seq.add_pulse("{p["channel"]}", {s}, {d}, sequence_blocks={blocks})\n')
        f.write('\n')

        # Handle sequencing order and repeats
        seq_desc = desc.get("sequence", {})
        order   = seq_desc.get("order", [])
        repeats = [int(Template(str(r)).render(ctx)) for r in seq_desc.get("repeats", [])]

        # Provide sensible defaults if missing
        if not order:
            order = ["wait_loop", "block_0"]
        if not repeats:
            repeats = [1, int(ctx.get("frames", 1))]

        f.write(f'    seq.sequencing_order   = {order}\n')
        f.write(f'    seq.sequencing_repeats = {repeats}\n')

        # Write out the YAML file to the Qupyt sequence directory
        f.write('    # write the pulse-sequence YAML into the sequence directory\n')
        f.write('    seq_dir  = set_up.get_seq_dir()\n')
        f.write('    seq_file = seq_dir / "sequence.yaml"\n')
        f.write('    seq.write(seq_file)\n')        
        f.write('    logging.info("Pulse sequence generated.")\n')
        f.write('    return {}\n')

    # Log completion
    logging.info("Wrote pulse‐sequence module to %s", output_path)
    
    return {}

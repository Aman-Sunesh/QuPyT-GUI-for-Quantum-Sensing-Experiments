import logging
from jinja2 import Template
from qupyt.pulse_sequences.yaml_sequence import YamlSequence

def generate_from_descriptor(desc: dict, params: dict, output_path: str = "user_pulse_seq.py"):
    # build Jinja context
    ctx = params.copy()
    unit        = params.get('time_unit', 'µs')
    unit_factor = {'ns': 1e-3, 'µs': 1.0, 'ms': 1e3}[unit]

    ctx["constants"] = {
        k: float(v) if str(v).replace('.','',1).isdigit() else int(v)
        for k,v in desc.get("constants",{}).items()
    }

    def R(expr):
        return float(Template(str(expr)).render(ctx)) * unit_factor

    # compute total duration
    total = desc.get(
        "total_duration",
        max(R(p["start"]) + R(p["duration"]) for p in desc["pulses"])
    )

    # ensure total covers the START trigger
    start_dur = float(ctx.get("start_pulse_dur", 1.0)) * unit_factor
    total = max(total, start_dur)

    # open the .py module and start writing
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\"\"\"\n')
        f.write('Generate a sample ODMR pulse sequence file.\n')
        f.write('This pulse sequence is not suited to be used directly in a measurement.\n')
        f.write('…adjust to your hardware…\n')
        f.write('\"\"\"\n\n')
        f.write('import logging\n')
        f.write('from qupyt.pulse_sequences.yaml_sequence import YamlSequence\n\n')
        f.write('from qupyt import set_up\n')
        f.write('logging.basicConfig(level=logging.DEBUG, '
                'format="%(asctime)s - %(levelname)s - %(message)s")\n\n')
        f.write('def generate_sequence(params: dict) -> dict:\n')
        f.write('    logging.debug("Generating pulse sequence with %s", params)\n')
        f.write(f'    seq = YamlSequence(duration={total})\n')

        start_dur = ctx.get("start_pulse_dur", 1.0)
        f.write(f'    seq.add_pulse("START", 0.0, {start_dur}, sequence_blocks=["wait_loop"])\n')

        # emit each pulse
        for p in desc["pulses"]:
            if p["channel"] == "START":
                continue
            s = R(p["start"]); d = R(p["duration"])
            blocks = p.get("blocks", ["block_0"])
            f.write(f'    seq.add_pulse("{p["channel"]}", {s}, {d}, sequence_blocks={blocks})\n')
        f.write('\n')

        # sequencing order & repeats
        seq_desc = desc.get("sequence", {})
        order   = seq_desc.get("order", [])
        repeats = [int(Template(str(r)).render(ctx)) for r in seq_desc.get("repeats", [])]

        # defaults if descriptor didn’t specify
        if not order:
            order = ["wait_loop", "block_0"]
        if not repeats:
            repeats = [1, int(ctx.get("frames", 1))]

        f.write(f'    seq.sequencing_order   = {order}\n')
        f.write(f'    seq.sequencing_repeats = {repeats}\n')
        f.write('    # write the pulse-sequence YAML into the sequence directory\n')
        f.write('    seq_dir  = set_up.get_seq_dir()\n')
        f.write('    seq_file = seq_dir / "sequence.yaml"\n')
        f.write('    seq.write(seq_file)\n')        
        f.write('    logging.info("Pulse sequence generated.")\n')
        f.write('    return {}\n')

    logging.info("Wrote pulse‐sequence module to %s", output_path)
    
    return {}
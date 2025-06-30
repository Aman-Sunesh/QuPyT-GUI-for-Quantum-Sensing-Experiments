import logging
from jinja2 import Template
from qupyt.pulse_sequences.yaml_sequence import YamlSequence

def generate_from_descriptor(desc: dict, params: dict, output_path: str = "user_pulse_seq.py"):
    # build Jinja context
    ctx = params.copy()
    ctx["constants"] = {
        k: float(v) if str(v).replace('.','',1).isdigit() else int(v)
        for k,v in desc.get("constants",{}).items()
    }

    def R(expr):
        return float(Template(str(expr)).render(ctx))

    # compute total duration
    total = desc.get(
        "total_duration",
        max(R(p["start"]) + R(p["duration"]) for p in desc["pulses"])
    )

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

        # emit each pulse
        for p in desc["pulses"]:
            s = R(p["start"]); d = R(p["duration"])
            blocks = p.get("blocks", ["block_0"])
            f.write(f'    seq.add_pulse("{p["channel"]}", {s}, {d}, sequence_blocks={blocks})\n')
        f.write('\n')

        # sequencing order & repeats
        order   = desc["sequence"]["order"]
        repeats = [int(Template(str(r)).render(ctx)) for r in desc["sequence"]["repeats"]]
        f.write(f'    seq.sequencing_order   = {order}\n')
        f.write(f'    seq.sequencing_repeats = {repeats}\n')
        f.write('    # write the pulse‐sequence YAML into the sequence directory\n')
        f.write('    seq_dir  = set_up.get_seq_dir()\n')
        f.write('    seq_file = seq_dir / "sequence.yaml"\n')
        f.write('    seq.write(seq_file)\n')        
        f.write('    logging.info("Pulse sequence generated.")\n')
        f.write('    return {}\n')

    logging.info("Wrote pulse‐sequence module to %s", output_path)
    
    return {}
# experiment_yaml.py

# ────────────────────────────────────────────────────────────────
# Renders a complete ODMR experiment YAML file from a set of
# GUI-supplied values. Writes atomically to avoid partial files.
# ────────────────────────────────────────────────────────────────

import os
import yaml
from pathlib import Path
from channels import CHANNEL_MAPPING
from yaml.dumper import SafeDumper
from yaml.nodes import ScalarNode
from yaml.representer import SafeRepresenter


# force flow-style sequences like [a, b] 
class FlowSeq(list):
    pass

def _flow_seq_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowSeq, _flow_seq_representer, Dumper=SafeDumper)

class QuotedStr(str):
    """Render this string in quotes in YAML."""
    pass

def _quotedstr_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

yaml.add_representer(QuotedStr, _quotedstr_representer, Dumper=SafeDumper)


_default_float_repr = SafeRepresenter.represent_float

def _float_representer(dumper, value: float):
    # Use compact scientific for large magnitudes; otherwise default.
    if abs(value) >= 1e6:
        sig = 3  # mantissa significant digits
        s = f"{float(value):.{sig}e}"       # e.g., '2.820e+09'
        mant, exp = s.split('e')
        mant = mant.rstrip('0').rstrip('.') or '0'

        if '.' not in mant:
            mant += '.0'                    # '2.0' style
        exp = str(int(exp))                 # drop '+' and leading zeros
        txt = f"{mant}e{exp}"               # e.g., '2.82e9'

        # Return a PLAIN scalar with implicit type so no !!float tag appears
        node = ScalarNode('tag:yaml.org,2002:float', txt, style=None)
        node.implicit = (True, True)

        return node
    
    return _default_float_repr(dumper, value)

yaml.add_representer(float, _float_representer, Dumper=SafeDumper)

def render_experiment_yaml(vals: dict, output_path: str):
    """
    Build a YAML dict from vals and write it atomically to output_path.
    Expects vals to contain at least the keys:
      - experiment_type, averages, frames, apd_input,
        MW, LASER, READ, START, n_dynamic_steps,
        address, freq_start, freq_stop, power,
        mode, ref_channels, ps_path,
        mw_duration, laserduration, read_time, max_rate
    Optionally: I_pulse, Q_pulse, tau, blocks.
    """
    # convert GUI times to µs
    unit        = vals.get('time_unit', 'µs')
    unit_factor = {'ns': 1e-3, 'µs': 1.0, 'ms': 1e3}[unit]
    mw_dur    = vals['mw_duration']   * unit_factor
    laser_dur = vals['laserduration'] * unit_factor
    read_dur  = vals['read_time']     * unit_factor

    # normalise output path & some inputs
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    freq_start = float(vals['freq_start'])
    freq_stop  = float(vals['freq_stop'])
    mode = vals.get('mode', 'spread')
    if mode not in ('spread', 'sum', 'interleaved'):
        mode = 'spread'

    # amplitude shape: allow constant, [start, stop], or per-channel dict
    power = vals.get('power')
    amp_cfg = []

    if isinstance(power, dict):
        for ch, rng in power.items():
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                start, stop = float(rng[0]), float(rng[1])
            else:
                start = stop = float(rng)
            # each amplitude row as a flow-seq: ['channel_x', [start, stop]]
            amp_cfg.append(FlowSeq([QuotedStr(str(ch)), FlowSeq([start, stop])]))

    elif isinstance(power, (list, tuple)) and len(power) == 2:
        amp_cfg.append(FlowSeq([QuotedStr('channel_0'), FlowSeq([float(power[0]), float(power[1])])]))
    
    else:
        amp = float(power)
        amp_cfg.append(FlowSeq([QuotedStr('channel_0'), FlowSeq([amp, amp])]))

    mw_device_type = vals.get('mw_device_type', 'WindFreakSHDMini')

    # quick sanity checks (fail fast if misconfigured)
    assert freq_stop >= freq_start, "freq_stop must be ≥ freq_start"
    assert vals['frames'] % max(1, int(vals['ref_channels'])) == 0, \
        "number_measurements (frames) should be a multiple of reference_channels"

    # Base structure
    cfg = {
        'experiment_type': vals['experiment_type'],
        'averages': vals['averages'],
        'sensor': {
            'type': 'DAQ',
            'config': {
                'number_measurements': vals['frames'],
                'apd_input': vals['apd_input']
            }
        },
        'synchroniser': {
            'type': 'PulseBlaster',
            'config': {},
            'channel_mapping': CHANNEL_MAPPING.copy()
        },
        'dynamic_steps': vals['n_dynamic_steps'],
        'dynamic_devices': {
            'mw_source': {
                'device_type': mw_device_type,
                'address': vals['address'],
                'config': {
                    'amplitude': amp_cfg,
                    'frequency': FlowSeq([freq_start, freq_stop]),
                }
            }
        },
        'static_devices': {},
        'data': {
            'averaging_mode': mode,
            'dynamic_steps': vals['n_dynamic_steps'],
            'compress': False,
            'reference_channels': vals['ref_channels']
        },
        'ps_path': str(vals['ps_path']),
        'pulse_sequence': {
            'mw_duration': mw_dur,
            'laserduration': laser_dur,
            'readout_time': read_dur,
            'referenced_measurements': vals['frames'],
            'max_framerate': vals['max_rate']
        }
    }

    # Inject the extra channels into the synchroniser mapping if they’re in vals
    for chan in ('I', 'Q'):
        key = f'{chan}_pulse'
        if key in vals and vals[key] > 0:
            idx_key = f'{chan}_channel'
            if idx_key in vals:
                cfg['synchroniser']['channel_mapping'][chan] = vals[idx_key]

            # also record duration in the pulse_sequence block
            cfg['pulse_sequence'][f'{chan.lower()}_pulse_duration'] = vals[key] * unit_factor

    # And the other parameters (tau, blocks)
    if 'tau' in vals:
        cfg['pulse_sequence']['tau'] = vals['tau'] * unit_factor
    if 'blocks' in vals:
        cfg['pulse_sequence']['blocks'] = vals['blocks']

    # Write
    tmp = output_path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)  # SafeDumper is used by safe_dump
    os.replace(tmp, output_path)

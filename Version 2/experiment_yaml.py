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
from yaml.representer import SafeRepresenter
import re


# force flow-style sequences like [a, b] 
class FlowSeq(list):
    pass

def _flow_seq_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

class FlowDumper(SafeDumper):
    """Custom dumper to keep floats plain (no !!float) and FlowSeq in flow style."""
    pass

yaml.add_representer(FlowSeq, _flow_seq_representer, Dumper=FlowDumper)

class QuotedStr(str):
    """Render this string in quotes in YAML."""
    pass

def _quotedstr_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

yaml.add_representer(QuotedStr, _quotedstr_representer, Dumper=FlowDumper)


_default_float_repr = SafeRepresenter.represent_float

def _float_representer(dumper, value: float):
    """
    Emit large magnitudes in compact scientific notation like '2.0e9'
    as a *plain* scalar so no quotes or !!float tag are added.
    """
    if abs(value) >= 1e6:
        sig = 3  # mantissa significant digits
        s = f"{float(value):.{sig}e}"      # e.g., '2.820e+09'
        mant, exp = s.split('e')
        mant = mant.rstrip('0').rstrip('.') or '0'
        if '.' not in mant:
            mant += '.0'                   # ensure '2.0' style
        exp = str(int(exp))                # drop '+' and leading zeros
        txt = f"{mant}e{exp}"              # e.g., '2.82e9'
        return dumper.represent_scalar('tag:yaml.org,2002:float', txt)
    return _default_float_repr(dumper, value)

# Use our float representer on the custom dumper
yaml.add_representer(float, _float_representer, Dumper=FlowDumper)

# Ensure plain scalars like 2.82e9 are recognized as floats (so no explicit !!float)
FlowDumper.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$'),
    list('-+0123456789.')
)

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

    # choose RF port token based on device type:
    # - WindFreak (Official):  use 'channel_0' / 'channel_1'
    # - WindFreakSHDMini:      use 0 / 1 (ints)
    mw_device_type = vals.get('mw_device_type', 'WindFreakSHDMini')
    mw_out = str(vals.get('mw_output', 'A')).strip()
    if mw_device_type == 'WindFreak':
        ch_token = QuotedStr('channel_1') if mw_out in ('B','1','channel_1') else QuotedStr('channel_0')
    else:
        ch_token = 1 if mw_out in ('B','1','channel_1') else 0

    # amplitude shape: allow constant, [start, stop], or per-channel dict
    power = vals.get('power')
    amp_cfg = []

    if isinstance(power, dict):
        for ch, rng in power.items():
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                start, stop = float(rng[0]), float(rng[1])
            else:
                start = stop = float(rng)
            # honor explicit per-channel dict as-is (assume caller matched device type)
            if mw_device_type == 'WindFreak':
                key = QuotedStr(str(ch))
            else:
                # accept 0/1 (int), or "0"/"1" (str). Leave other types as-is.
                if isinstance(ch, int):
                    key = ch
                elif isinstance(ch, str) and ch.isdigit():
                    key = int(ch)
                else:
                    key = ch
            amp_cfg.append(FlowSeq([key, FlowSeq([start, stop])]))

    elif isinstance(power, (list, tuple)) and len(power) == 2:
        amp_cfg.append(FlowSeq([ch_token, FlowSeq([float(power[0]), float(power[1])])]))
    
    else:
        amp = float(power)
        amp_cfg.append(FlowSeq([ch_token, FlowSeq([amp, amp])]))

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
                    # include channel with frequency range for WindFreak Official,
                    # otherwise plain ints for SHDMini
                    'frequency': (
                        FlowSeq([QuotedStr('channel_1' if mw_out in ('B','1','channel_1') else 'channel_0'),
                                 FlowSeq([freq_start, freq_stop])])
                        if mw_device_type == 'WindFreak'
                        else FlowSeq([ (1 if mw_out in ('B','1','channel_1') else 0),
                                       FlowSeq([freq_start, freq_stop])])
                    ),
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
        yaml.dump(cfg, f, Dumper=FlowDumper, sort_keys=False)  
    os.replace(tmp, output_path)

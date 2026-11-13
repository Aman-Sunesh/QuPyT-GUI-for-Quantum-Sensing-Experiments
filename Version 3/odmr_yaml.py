# odmr_yaml.py

# ────────────────────────────────────────────────────────────────
# Renders a complete ODMR experiment YAML file from a set of
# GUI-supplied values. Writes atomically to avoid partial files.
# ────────────────────────────────────────────────────────────────

import os
import yaml
import numpy as np
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
    Emit large magnitudes in scientific notation while retaining
    enough precision for narrow microwave-frequency sweeps.
    """
    if abs(value) >= 1e6:
        decimal_places = 12
        s = f"{float(value):.{decimal_places}e}"
        mant, exp = s.split('e')
        mant = mant.rstrip('0').rstrip('.') or '0'
        if '.' not in mant:
            mant += '.0'
        exp = str(int(exp))
        txt = f"{mant}e{exp}"
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
        I, Q, LASER, READ, START, n_dynamic_steps,
        address, freq_start, freq_stop, power,
        mode, ref_channels, ps_path,
        mw_duration, laserduration, read_time, max_rate

    Dynamic sweep modes:

      frequency:
        Sweep freq_start -> freq_stop while holding power fixed.

      power:
        Sweep power_start -> power_stop while holding
        fixed_frequency constant.

    Optionally: I_pulse, Q_pulse, tau, blocks.
    """
    mw_dur = float(vals['mw_duration'])
    laser_dur = float(vals['laserduration'])
    read_dur = float(vals['read_time'])

    ps_steps = int(
        vals.get("pulse_sequence_steps", 1)
    )

    if ps_steps < 1:
        raise ValueError(
            "pulse_sequence_steps must be at least 1"
        )

    sweep_param_raw = vals.get("sweep_param")
    sweep_param = (
        str(sweep_param_raw).strip()
        if sweep_param_raw not in (None, "")
        else None
    )

    sweep_values = vals.get("sweep_values", [])

    if sweep_values is None:
        sweep_values = []
    elif not isinstance(sweep_values, (list, tuple)):
        sweep_values = [sweep_values]
    else:
        sweep_values = list(sweep_values)

    if sweep_param is not None:
        if len(sweep_values) not in (2, ps_steps):
            raise ValueError(
                "sweep_values must contain either two endpoints "
                "or exactly pulse_sequence_steps values"
            )
        
    # Normalise output path and dynamic-sweep inputs.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_steps = int(vals["n_dynamic_steps"])

    if dynamic_steps < 1:
        raise ValueError(
            "n_dynamic_steps must be at least 1"
        )

    dynamic_sweep_type = str(
        vals.get(
            "dynamic_sweep_type",
            "frequency",
        )
    ).strip().lower()

    dynamic_sweep_aliases = {
        "frequency": "frequency",
        "frequency sweep": "frequency",
        "power": "power",
        "power sweep": "power",
        "microwave power": "power",
        "microwave power sweep": "power",
    }

    try:
        dynamic_sweep_type = (
            dynamic_sweep_aliases[
                dynamic_sweep_type
            ]
        )
    except KeyError as error:
        raise ValueError(
            "dynamic_sweep_type must be either "
            "'frequency' or 'power'"
        ) from error

    freq_start = float(vals['freq_start'])
    freq_stop  = float(vals['freq_stop'])

    fixed_frequency = float(
        vals.get(
            "fixed_frequency",
            freq_start,
        )
    )

    fixed_power = float(
        vals.get(
            "power",
            0.0,
        )
    )

    power_start = float(
        vals.get(
            "power_start",
            fixed_power,
        )
    )

    power_stop = float(
        vals.get(
            "power_stop",
            fixed_power,
        )
    )

    if dynamic_sweep_type == "power":
        if power_stop < power_start:
            raise ValueError(
                "power_stop must be greater than or "
                "equal to power_start"
            )

        # The frequency remains fixed at every dynamic step.
        freq_start = fixed_frequency
        freq_stop = fixed_frequency

        # Existing DynamicDeviceHandler interprets this pair as
        # a linear amplitude sweep.
        power = [
            power_start,
            power_stop,
        ]

        dynamic_axis_parameter = (
            "microwave_power"
        )
        dynamic_axis_unit = "dBm"
        dynamic_axis_values = np.linspace(
            power_start,
            power_stop,
            dynamic_steps,
        ).tolist()

    else:
        if freq_stop < freq_start:
            raise ValueError(
                "freq_stop must be greater than or "
                "equal to freq_start"
            )

        power = fixed_power

        dynamic_axis_parameter = "frequency"
        dynamic_axis_unit = "Hz"
        dynamic_axis_values = np.linspace(
            freq_start,
            freq_stop,
            dynamic_steps,
        ).tolist()

    mode = vals.get('mode', 'spread')
    if mode not in ('spread', 'sum'):
        mode = 'spread'

    # choose RF port token based on device type:
    # - WindFreak (Official):  use 'channel_0' / 'channel_1'
    # - WindFreakSHDMini:      use '0' / '1' 
    mw_device_type = vals.get('mw_device_type', 'WindFreak')
    mw_out = str(vals.get('mw_output', 'B')).strip().upper()
    if mw_device_type == 'WindFreak':
        ch_token = (
            QuotedStr('channel_1')
            if mw_out in ('B', '1', 'CHANNEL_1')
            else QuotedStr('channel_0')
        )
    else:
        ch_token = (
            QuotedStr('1')
            if mw_out in ('B', '1', 'CHANNEL_1')
            else QuotedStr('0')
        )

    # amplitude shape: allow constant, [start, stop], or per-channel dict
    amp_cfg = []

    if isinstance(power, dict):
        for ch, rng in power.items():
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                start, stop = float(rng[0]), float(rng[1])
            else:
                start = stop = float(rng)
            # force channel key to string for both drivers
            key = QuotedStr(str(ch))
            amp_cfg.append(FlowSeq([key, FlowSeq([start, stop])]))

    elif isinstance(power, (list, tuple)) and len(power) == 2:
        amp_cfg.append(
            FlowSeq([
                ch_token,
                FlowSeq([
                    float(power[0]),
                    float(power[1])
                ])
            ])
        )
    
    else:
        amp = float(power)
        amp_cfg.append(
            FlowSeq([
                ch_token,
                FlowSeq([amp, amp])
            ])
        )

    reference_channels = max(
        1,
        int(vals["ref_channels"]),
    )

    if int(vals["frames"]) % reference_channels != 0:
        raise ValueError(
            "number_measurements (frames) must be divisible by "
            "reference_channels"
        )

    # Base structure
    cfg = {
        'experiment_type': vals['experiment_type'],
        'averages': vals['averages'],
        'pulse_sequence_steps': ps_steps,
        'result_schema_version': 2,
        'data_layout': 'channel_ps_dynamic_frame_roi',
        'dynamic_sweep_type': dynamic_sweep_type,
        'dynamic_axis': {
            'parameter': dynamic_axis_parameter,
            'values': [
                float(value)
                for value in dynamic_axis_values
            ],
            'unit': dynamic_axis_unit,
        },
        'sensor': {
            'type': 'DAQ',
            'config': {
                'number_measurements': vals['frames'],
                'apd_input': vals['apd_input'],
                'reference_channels': vals['ref_channels']
            }
        },
        'synchroniser': {
            'type': 'PulseBlaster',
            'config': {},
            'channel_mapping': CHANNEL_MAPPING.copy()
        },
        'dynamic_steps': dynamic_steps,
        'dynamic_devices': {
            'mw_source': {
                'device_type': mw_device_type,
                'address': vals['address'],
                'config': {
                    'amplitude': amp_cfg,
                    'frequency': FlowSeq([
                        ch_token,
                        FlowSeq([
                            freq_start,
                            freq_stop
                        ])
                    ]),
                }
            }
        },
        'static_devices': {},
        'data': {
            'averaging_mode': mode,
            'dynamic_steps': dynamic_steps,
            'ps_steps': ps_steps,
            'live_compression': False,
            'reference_channels': vals['ref_channels']
        },
        'ps_path': str(vals['ps_path']),
        'pulse_sequence': {
            'pulse_sequence_steps': ps_steps,
            'mw_duration': mw_dur,
            'laserduration': laser_dur,
            'readout_time': read_dur,
            'referenced_measurements': vals['frames'],
            'max_framerate': vals['max_rate']
        }
    }

    if sweep_param is not None:
        cfg['pulse_sequence']['sweep_param'] = sweep_param
        cfg['pulse_sequence']['sweep_values'] = [
            float(value)
            for value in sweep_values
        ]

        cfg['pulse_sequence_axis'] = {
            'parameter': sweep_param,
            'values': [
                float(value)
                for value in sweep_values
            ],
            'unit': str(
                vals.get("sweep_unit", "")
            ),
        }

    else:
        cfg['pulse_sequence_axis'] = {
            'parameter': 'step',
            'values': list(range(ps_steps)),
            'unit': '',
        }

    # Inject the extra channels into the synchroniser mapping if they’re in vals
    for chan in ('I', 'Q'):
        key = f'{chan}_pulse'
        if key in vals and vals[key] > 0:
            idx_key = f'{chan}_channel'
            if idx_key in vals:
                cfg['synchroniser']['channel_mapping'][chan] = vals[idx_key]

            # also record duration in the pulse_sequence block
            cfg['pulse_sequence'][f'{chan.lower()}_pulse_duration'] = float(vals[key])

    # And the other parameters (tau, blocks)
    if 'tau' in vals:
        cfg['pulse_sequence']['tau'] = float(vals['tau'])
    if 'blocks' in vals:
        cfg['pulse_sequence']['blocks'] = vals['blocks']

    # Write
    tmp = output_path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, Dumper=FlowDumper, sort_keys=False)  
    os.replace(tmp, output_path)
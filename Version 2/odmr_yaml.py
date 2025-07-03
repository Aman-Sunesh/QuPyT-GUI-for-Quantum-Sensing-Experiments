# experiment_yaml.py
import os
import yaml
from pathlib import Path
from channels import CHANNEL_MAPPING

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
                'device_type': 'WindFreak',
                'address': vals['address'],
                'config': {
                    'frequency': [vals['freq_start'], vals['freq_stop']],
                    'amplitude': [["channel_0", [vals['power'], vals['power']]]]
                }
            }
        },
        'static_devices': {},
        'data': {
            'averaging_mode': vals['mode'],
            'dynamic_steps': vals['n_dynamic_steps'],
            'compress': False,
            'reference_channels': vals['ref_channels']
        },
        'ps_path': vals['ps_path'],
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
        yaml.safe_dump(cfg, f, sort_keys=False)
    os.replace(tmp, output_path)

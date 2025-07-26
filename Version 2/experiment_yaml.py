# experiment_yaml.py

# ────────────────────────────────────────────────────────────────
# Renders a complete ODMR experiment YAML file from a set of
# GUI-supplied values. Writes atomically to avoid partial files.
# ────────────────────────────────────────────────────────────────

import os
import yaml
from pathlib import Path
from channels import CHANNEL_MAPPING

def render_experiment_yaml(vals: dict, output_path: str):
    """
    Build an ODMR experiment configuration dict from GUI values
    and atomically write it to the given YAML file path.

    Args:
        vals: Dictionary containing experiment parameters. Required keys:
            - experiment_type, averages, frames, apd_input,
            - MW, LASER, READ, START, n_dynamic_steps,
            - address, freq_start, freq_stop, power,
            - mode, ref_channels, ps_path,
            - mw_duration, laserduration, read_time, max_rate
          Optional keys:
            - I_pulse, Q_pulse, I_channel, Q_channel, tau, blocks
            - time_unit (ns, µs, ms)
        output_path: Path object where YAML is written.
    """
    
    # Determine time unit conversion factor (default µs)
    unit        = vals.get('time_unit', 'µs')
    unit_factor = {'ns': 1e-3, 'µs': 1.0, 'ms': 1e3}[unit]

    # Convert pulse durations to chosen unit (µs)
    mw_dur    = vals['mw_duration']   * unit_factor
    laser_dur = vals['laserduration'] * unit_factor
    read_dur  = vals['read_time']     * unit_factor

    # Base YAML structure
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

    # Inject I/Q pulses if specified
    for chan in ('I', 'Q'):
        key = f'{chan}_pulse'
        if key in vals and vals[key] > 0:
            # Map the channel number if provided
            idx_key = f'{chan}_channel'
            if idx_key in vals:
                cfg['synchroniser']['channel_mapping'][chan] = vals[idx_key]

            # Record pulse duration
            cfg['pulse_sequence'][f'{chan.lower()}_pulse_duration'] = vals[key] * unit_factor

    # And the other parameters (tau, blocks)
    if 'tau' in vals:
        cfg['pulse_sequence']['tau'] = vals['tau'] * unit_factor
    if 'blocks' in vals:
        cfg['pulse_sequence']['blocks'] = vals['blocks']

    # Atomically write the YAML to avoid partial files
    tmp = output_path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    os.replace(tmp, output_path)

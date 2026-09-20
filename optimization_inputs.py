"""Fixed physical inputs shared by the launcher and optimizer CLI."""
import math
import re

INPUTS = {
    'detuning_mhz': 'nominal Delta/(2*pi), in MHz',
    'alpha_mhz': 'nominal anharmonicity alpha/(2*pi), in MHz (negative)',
    't1_1_ns': 'T1(1), in ns',
    't1_2_ns': 'T1(2), in ns',
    'tphi_1_ns': 'Tphi(1), in ns',
    'tphi_2_ns': 'Tphi(2), in ns',
    'tg_ns': 'fixed gate duration Tg, in ns (10-60, on the 0.2 ns grid)',
}


def add_input_arguments(parser):
    parser.add_argument('--non-interactive', action='store_true',
                        help='use configuration/CLI inputs without input() prompts')
    import argparse
    parser.add_argument('--nsga', action=argparse.BooleanOptionalAction, default=None,
                        help='enable/disable NSGA-III numerical search')
    parser.add_argument('--grape', action=argparse.BooleanOptionalAction, default=None,
                        help='enable/disable GRAPE before final Monte Carlo')
    for name, help_text in INPUTS.items():
        parser.add_argument('--' + name.replace('_', '-'), type=float, help=help_text)


def apply_input_arguments(config, args):
    if getattr(args, 'nsga', None) is not None:
        config['nsga_enabled'] = args.nsga
    if getattr(args, 'grape', None) is not None:
        config.setdefault('grape', {})['enabled'] = args.grape
    for name in INPUTS:
        value = getattr(args, name)
        if value is not None:
            if name == 'tg_ns':
                config['gate_duration_ns'] = value
            else:
                config['device'][name] = value


def input_command_arguments(args):
    result = []
    for method in ('nsga', 'grape'):
        enabled = getattr(args, method, None)
        if enabled is not None:
            result.append('--' + ('' if enabled else 'no-') + method)
    for name in INPUTS:
        value = getattr(args, name)
        if value is not None:
            result.extend(['--' + name.replace('_', '-'), str(value)])
    return result


def parse_physical_input(raw, name):
    """Convert explicit frequency/time units to the project's MHz/ns convention."""
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Zµμ]*)\s*", raw)
    if not match:
        raise ValueError('enter a number, optionally followed by units (e.g. 300000 Hz or 20 us)')
    value = float(match.group(1))
    unit = match.group(2).lower().replace('µ', 'u').replace('μ', 'u')
    units = ({'': 1, 'hz': 1e-6, 'khz': 1e-3, 'mhz': 1, 'ghz': 1e3}
             if name in ('detuning_mhz', 'alpha_mhz') else
             {'': 1, 'ns': 1, 'us': 1e3, 'ms': 1e6, 's': 1e9})
    if unit not in units:
        raise ValueError('frequency inputs require Hz/kHz/MHz/GHz; time inputs require ns/us/ms/s')
    if name == 'detuning_mhz' and not unit and abs(value) > 1000:
        raise ValueError('large detuning: specify units explicitly; 300000 Hz = 0.3 MHz, whereas 300000 MHz = 300 GHz')
    return value*units[unit]


def prompt_input_arguments(config, args):
    """Read seven fixed inputs once; blank lines keep the displayed defaults."""
    if args.non_interactive:
        return
    print("Enter fixed physical inputs. Press Enter to keep each default.")
    print("Default units: MHz and ns. You can type units, e.g. 300000 Hz or 30 us.")
    for name, label in INPUTS.items():
        default = getattr(args, name)
        if default is None:
            default = (config['gate_duration_ns'] if name == 'tg_ns'
                       else config['device'][name])
        while True:
            raw = input(f"{label} [{default:g}]: ").strip()
            try:
                value = parse_physical_input(raw, name) if raw else float(default)
                if not math.isfinite(value):
                    raise ValueError('enter a finite number')
                if name == 'alpha_mhz' and value >= 0:
                    raise ValueError('anharmonicity must be negative')
                if name.startswith(('t1_', 'tphi_')) and value <= 0:
                    raise ValueError('lifetime must be positive')
                if name == 'tg_ns' and (not 10 <= value <= 60 or
                        not math.isclose(round(value/.2)*.2, value, abs_tol=1e-10, rel_tol=0)):
                    raise ValueError('Tg must be 10-60 ns, in steps of 0.2 ns')
            except ValueError as error:
                print(f"Invalid input: {error}. Try again.")
                continue
            unit = 'MHz' if name in ('detuning_mhz', 'alpha_mhz') else 'ns'
            print(f"  Using {value:g} {unit}")
            setattr(args, name, value)
            break

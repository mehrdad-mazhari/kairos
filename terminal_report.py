"""Print completed numerical results and the PDF-required technical report.

Uses only the standard library, so saved results can be inspected without
loading the simulation environment. CSV rows and plot contents stay in files.
"""
import json
import csv
import math
from pathlib import Path
import re
from pulse_coefficients import pulse_coefficients


def _value(value):
    if isinstance(value, bool):
        return 'PASS' if value else 'FAIL'
    if isinstance(value, float):
        return f'{value:.10g}'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_value(item) for item in value) + ']'
    return str(value)


def _metrics(data, indent='  '):
    for key, value in data.items():
        label = key.replace('_', ' ')
        if isinstance(value, dict):
            print(f'{indent}{label}:')
            _metrics(value, indent + '  ')
        else:
            print(f'{indent}{label:<43} = {_value(value)}')


def print_detailed_results(manifest, output=None):
    """Print every saved final metric, plus report prose without image embeds."""
    print('\n' + '=' * 78 + '\nFINAL RESULTS (PDF pages 8-9)\n' + '=' * 78)
    print('Fidelity/leakage values below are fractions: 1 = 100%.')
    print('Frequency units: MHz; time: ns; angular drive: rad/ns; slew: rad/ns^2.')
    print('Confidence intervals are [lower, upper]; sampled extremes are not guarantees.')
    if manifest.get('selected_pulse_spec'):
        from pulse_methods import parameter_rows
        print('SELECTED PULSE PARAMETERS')
        for label, value in parameter_rows(manifest['selected_pulse_spec']):
            print(f'  {label}: {value}')
    final = manifest['variants']['robust_selected']
    p = final['parameters']
    if 'gate_duration_ns' in manifest['config']:
        print('\nFIXED USER INPUTS (not optimized)')
        _metrics({**manifest['config']['device'],
                  'gate_duration_ns': manifest['config']['gate_duration_ns']})
        print('NSGA-III, when enabled, optimizes sigma/Tg, DRAG beta, amplitude scale and I/Q phase ramp.')
        print('Scenario detuning is an additive error around the fixed nominal input.')
    device = manifest['config']['device']
    delta = device.get('detuning_mhz', 0.) * 2 * math.pi * .001
    alpha = device['alpha_mhz'] * 2 * math.pi * .001
    print('\nHAMILTONIAN: PDF page 4, rotating frame, rotating-wave approximation')
    print('  a = |0><1| + sqrt(2)*|1><2|; n = a^dagger*a; hbar = 1')
    print('  H0 = Delta*n + alpha*n*(n-1)/2')
    print('  Hctrl(t) = I(t)*(a+a^dagger)/2 - i*Q(t)*(a-a^dagger)/2')
    print('  H(t) = [[0, c, 0], [c*, Delta, sqrt(2)*c],')
    print('          [0, sqrt(2)*c*, 2*Delta+alpha]], c=(I-iQ)/2; * means conjugate')
    print(f'  Nominal H0 diagonal (rad/ns) = [0, {delta:.10g}, {2*delta+alpha:.10g}]')
    print('  Noise: T1/Tphi Lindblad, using the challenge PDF page-5 operators.')
    print('  L10=sqrt(1/T1(1))*|0><1|; L21=sqrt(1/T1(2))*|1><2|')
    print('  Lphi1=sqrt(2/Tphi(1))*|1><1|; Lphi2=sqrt(2/Tphi(2))*|2><2|')
    print('  T1/Tphi are scaled per manufacturing sample; no explicit bath is modeled.')
    if 'drag_reference' in manifest:
        print('  DRAG reference: arXiv:1008.2554 Eq. (14), beta=lambda^2/4=0.5')
        print('  Optimized beta is fitted; the challenge fixes lambda=sqrt(2).')
    refined = manifest.get('grape', {}).get('selected', manifest.get('grape', {}).get('accepted', False))
    if refined:
        print('\nFINAL PULSE REPRESENTATION: GRAPE-refined I/Q samples in pulse.csv')
        print('  Tg and physical inputs are fixed. Use the exact recipe above; the formulas below are Gaussian references.')
        print('  The refined waveform is not represented exactly by a Gaussian A/B pair.')
    print('\nGAUSSIAN/DRAG REFERENCE FAMILY (use selected recipe above for the actual pulse)' if manifest.get('selected_pulse_spec') else '\n' + ('SEED' if refined else 'FINAL') + ' PULSE: tapered Gaussian-DRAG with I/Q phase ramp')
    print('  g(t) = exp(-(t-T/2)^2/(2*sigma^2)) * sin(pi*t/T)^2')
    print('  C = amplitude*pi/trapezoidal_integral(g, sampled times)')
    print('  I(t)+iQ(t) = C*(g(t)-i*beta*g\'(t)/alpha_nominal)')
    print('                 * exp(i*2*pi*phase_ramp_mhz*0.001*(t-T/2))')
    print('  alpha_nominal uses rad/ns; analytic derivative; endpoints forced to zero.')
    print('  The applied waveform linearly interpolates the exported 0.2-ns samples'
          if manifest['config']['device']['dt_ns'] == .2 else
          '  The applied waveform linearly interpolates samples at configured dt.')
    _metrics(p)
    print(f"  Gaussian sigma ns                           = {p['duration_ns']*p['sigma_fraction']:.10g}")
    print(f"  sample step ns                              = {manifest['config']['device']['dt_ns']}")
    print('\nGAUSSIAN REFERENCE ENVELOPE A AND B' if manifest.get('selected_pulse_spec') else '\n' + ('PRE-GRAPE SEED' if refined else 'FINAL') + ' ENVELOPE A AND B')
    print('  Applied pre-phase envelope: I0=A*(exp(...)-B)*sin(pi*t/T)^2; B=0.')
    print('  A is the normalization coefficient, not the peak of the phase-rotated I/Q pulse.')
    print('  Paper reference: Ipi=A*(exp(...)-B); B is dimensionless.')
    print('  Equivalently Ipi=A*exp(...)-B_offset, where B_offset=A*B (rad/ns).')
    print('  Reference coefficients do not describe the exported tapered waveform.')
    _metrics(manifest.get('pulse_coefficients') or pulse_coefficients(p, device['dt_ns']))
    peak = final['hardware']['peak_amplitude_mhz']
    scale = 1 + manifest['config']['manufacturing_ranges']['amplitude_fraction']
    print(f'  peak MHz at maximum manufacturing scale     = {peak*scale:.10g}')
    print('\nFINAL PULSE AND BASELINE COMPARISONS')
    for name, variant in manifest['variants'].items():
        print('\n' + name.upper().replace('_', ' '))
        for section, values in variant.items():
            label = 'seed parameters (not final waveform)' if refined and name == 'robust_selected' and section == 'parameters' else section.replace('_', ' ')
            print('  ' + label + ':')
            if section == 'nominal':
                _metrics({k: v for k, v in values.items() if not k.startswith('state_')}, '    ')
                print('    Input state      X-target fidelity       Leakage to |2>')
                for state, fidelity, leakage in zip(
                        ('|0>', '|1>', '|+>', '|->', '|+i>', '|-i>'),
                        values['state_fidelities'], values['state_leakages']):
                    print(f'    {state:<12} {fidelity:18.10f} {leakage:20.10g}')
            else:
                _metrics(values, '    ')
    if 'grape' in manifest:
        print('\nGRAPE REFINEMENT')
        _metrics(manifest['grape'])
    print('\nMONTE CARLO SELECTION (selection-set intervals are descriptive)')
    _metrics(manifest['mc_selection'])
    print('  Pass fractions use PDF thresholds F >= 0.990 and leakage <= 0.020.')
    print('\nACCEPTANCE AND NUMERICAL CONSISTENCY')
    print('  PDF nominal numerical thresholds            = ' +
          ('PASS' if final['nominal']['fidelity'] >= .990 and final['nominal']['leakage'] <= .020 else 'FAIL'))
    _metrics({k: manifest[k] for k in ('nominal_acceptance_passed', 'quality_passed',
                                      'numerical_checks')})
    print('  nominal acceptance above uses the stricter configured engineering targets.')
    print('\nSEARCH AND REPRODUCIBILITY')
    _metrics({k: manifest[k] for k in ('optimizer', 'unique_optimization_candidates',
                                      'training_channel_simulations', 'selection_loss', 'versions')})
    _metrics({'configuration': manifest['config']})
    if output is not None:
        report = Path(output) / 'report.md'
        print('\nTECHNICAL REPORT (plots remain in the results directory)\n')
        prose = report.read_text(encoding='utf-8')
        for line in prose.splitlines():
            if not line.lstrip().startswith('!['):
                line = re.sub(r'^#+\s*', '', line).replace('**', '').replace('`', '')
                print(line)
    print('\nSuggested extensions: measured drive-chain filtering, higher transmon levels,')
    print('correlated fabrication errors, and larger search/validation budgets.')
    print('\nEND OF FINAL RESULTS\n', flush=True)


def print_results(manifest, output=None):
    """Compact default; complete scientific details remain in saved reports."""
    from pulse_methods import METHOD_LABELS as labels, parameter_rows
    final = manifest['variants']['robust_selected']
    p = final['parameters']
    print("\nSTAGE RESULTS (probabilities as fractions)")
    print(f"{'Stage':<20} {'Time ns':>9} {'Nominal 1-F':>13} {'Nominal leak':>13} {'MC mean 1-F':>13} {'MC mean leak':>13} {'MC P05 F':>11}")
    for name, row in manifest['variants'].items():
        n, m = row['nominal'], row['monte_carlo']
        print(f"{labels.get(name, name):<20} {row['hardware']['duration_ns']:9.3g} {n['infidelity']:13.6g} {n['leakage']:13.6g} {1-m['mean_fidelity']:13.6g} {m['mean_leakage']:13.6g} {m['p05_fidelity']:11.7f}")
    g = manifest.get('grape', {})
    print(f"\nNSGA-III enabled: {manifest['config'].get('nsga_enabled', True)}; "
          f"selected method: {manifest.get('selected_method', 'legacy')}; "
          f"MC winner: {manifest['mc_selection']['candidate']+1}; GRAPE: "
          + ('accepted' if g.get('accepted') else 'not retained' if g.get('enabled') else 'disabled/not run'))
    if g.get('comparison_checks'):
        failed = [key.replace('_', ' ') for key, passed in g['comparison_checks'].items() if not passed]
        print('GRAPE trade-off checks: ' + ('PASS' if not failed else 'FAIL: ' + '; '.join(failed)))
    if g.get('history') and 'a_coefficients' in g['history'][-1]:
        coeff = g['history'][-1]
        print('GRAPE Fourier a: ' + ', '.join(f'{v:.6g}' for v in coeff['a_coefficients']))
        print('GRAPE Fourier b: ' + ', '.join(f'{v:.6g}' for v in coeff['b_coefficients']))
    print(f"Tg={p['duration_ns']:g} ns; peak={final['hardware']['peak_amplitude_mhz']:.6f} MHz; "
          f"quality={'PASS' if manifest['quality_passed'] else 'FAIL'}; "
          f"P05 95% lower bound={final['monte_carlo']['p05_fidelity_lcb95']:.7f}")
    if manifest.get('selected_pulse_spec'):
        print('\nSELECTED PULSE PARAMETERS (pulse.csv is authoritative)')
        for label, value in parameter_rows(manifest['selected_pulse_spec']):
            print(f'  {label}: {value}')
        print('\nMETHOD ELIGIBILITY')
        for name, variant in manifest['variants'].items():
            selection = variant.get('selection')
            if selection:
                print(f"  {labels.get(name, name)}: " + ('eligible' if selection['selection_feasible'] else '; '.join(selection['rejection_reasons'])))
    else:
        label = 'Seed' if g.get('selected', g.get('accepted')) else 'Final'
        print(f"{label} parameters: sigma={p['duration_ns']*p['sigma_fraction']:.7g} ns, "
              f"beta={p['beta']:.7g}, amplitude={p['amplitude']:.7g}, phase ramp={p['phase_ramp_mhz']:.7g} MHz")
        coefficients = manifest.get('pulse_coefficients') or pulse_coefficients(p, manifest['config']['device']['dt_ns'])
        print(f"{label} A/B (I0=A*(exp-B)*taper): A={coefficients['applied_tapered_envelope']['A_rad_per_ns']:.8g} rad/ns, B=0")
        for key, label in [('shifted_gaussian_same_sigma_reference','Gaussian reference, same sigma'),
                           ('paper_eq12_sigma_T_over_2_pi_area_reference','Paper reference, sigma=Tg/2')]:
            row = coefficients[key]
            print(f"{label}: A={row['A_rad_per_ns']:.8g} rad/ns, B={row['B_dimensionless']:.8g}, A*B={row['B_offset_rad_per_ns']:.8g} rad/ns")
    if output:
        path = Path(output)/'robust_selected_public.csv'
        if path.exists():
            with path.open() as stream:
                rows = [r for r in csv.DictReader(stream) if float(r['amplitude_scale']) == 1 and float(r['alpha_shift_mhz']) == 0]
            print("\nFINAL DETUNING EFFECT (other device parameters nominal)")
            print(f"{'Detuning MHz':>13} {'Fidelity':>13} {'Leakage':>13}")
            for r in sorted(rows, key=lambda r: float(r['detuning_mhz'])):
                d = manifest['config']['device'].get('detuning_mhz', 0)+float(r['detuning_mhz'])
                print(f"{d:13.6g} {float(r['fidelity']):13.8f} {float(r['leakage']):13.6g}")
        if (Path(output)/'stage_comparisons.md').exists():
            print(f"\nTables and plots: {Path(output).resolve() / 'stage_comparisons.md'}")
        print(f"Full report: {Path(output).resolve() / 'report.md'}", flush=True)


def show_saved_results(output):
    output = Path(output).resolve()
    manifest = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    print(f'Saved results: {output} (no optimization rerun)')
    print_results(manifest, output)

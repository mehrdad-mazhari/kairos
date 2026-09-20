"""Four focused comparisons: stage errors, waveforms, optimization and detuning."""
import json
from itertools import cycle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from challenge_report import LABELS, COLORS
from challenge_model import MHZ_TO_RAD_NS


def detuning_rows(rows):
    # Isolate detuning: nominal amplitude, alpha and all coherence times.
    return sorted([r for r in rows if r['amplitude_scale'] == 1 and r['alpha_shift_mhz'] == 0],
                  key=lambda r: r['detuning_mhz'])


def save_stage_comparisons(output, manifest, pulses, public, history):
    output = Path(output)
    names = list(pulses)
    variants = manifest['variants']
    labels = [LABELS[n] for n in names]
    tables = ['# Stage comparisons', '',
              'All probabilities are fractions. Each stage uses the same held-out devices. '
              'GRAPE proposal is shown even if rejected; Final selected waveform is a diagnostic candidate when quality is FAIL.', '',
              '## 1. Infidelity and leakage by stage', '',
              '| Stage | Nominal infidelity | Nominal leakage | MC mean infidelity | MC mean leakage | MC P05 fidelity | MC worst leakage |',
              '|---|---:|---:|---:|---:|---:|---:|']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(names))
    for ax, metric, title in zip(axes, ['infidelity', 'leakage'], ['Infidelity by stage', 'Leakage by stage']):
        nominal = [variants[n]['nominal'][metric] for n in names]
        mc = [(1-variants[n]['monte_carlo']['mean_fidelity']) if metric == 'infidelity'
              else variants[n]['monte_carlo']['mean_leakage'] for n in names]
        ax.bar(x-.18, np.maximum(nominal, 1e-12), .36, label='Nominal')
        ax.bar(x+.18, np.maximum(mc, 1e-12), .36, label='Held-out MC mean')
        ax.set_xticks(x, labels, rotation=25, ha='right')
        ax.set(title=title, ylabel='Fraction', yscale='log'); ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(output/'stage_metrics.png', dpi=160); plt.close(fig)
    for n in names:
        v, m = variants[n]['nominal'], variants[n]['monte_carlo']
        tables.append(f"| {LABELS[n]} | {v['infidelity']:.9g} | {v['leakage']:.9g} | {1-m['mean_fidelity']:.9g} | {m['mean_leakage']:.9g} | {m['p05_fidelity']:.9g} | {m['worst_sample_leakage']:.9g} |")
    tables += ['', '### Selection eligibility', '', '| Method | Eligible | Reason |', '|---|---|---|']
    for n in names:
        selection = variants[n].get('selection')
        reason = '; '.join(selection.get('rejection_reasons', [])) if selection else 'Not shortlisted / legacy result'
        tables.append(f"| {LABELS[n]} | {selection['selection_feasible'] if selection else 'N/A'} | {reason or 'Passed selection constraints'} |")
    tables += ['', '## 2. Pulse shape by stage', '',
               '| Stage | Tg (ns) | Peak magnitude / 2pi (MHz) | Peak slew (rad/ns²) | Waveform |',
               '|---|---:|---:|---:|---|']
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)
    styles = ['-', '--', '-.', ':', '--', '-.']
    markers = ['o', 's', '^', 'D', 'v', 'x']
    for index, (n, color) in enumerate(zip(names, cycle(COLORS))):
        t, i, q = pulses[n]
        for ax, values in zip(axes, [i, q]):
            ax.plot(t, values/MHZ_TO_RAD_NS, color=color, ls=styles[index % len(styles)],
                    marker=markers[index % len(markers)], markevery=(index*2, max(7, len(t)//9)),
                    ms=4, lw=1.7, label=LABELS[n])
        h = variants[n]['hardware']
        tables.append(f"| {LABELS[n]} | {h['duration_ns']:g} | {h['peak_amplitude_mhz']:.9g} | {h['max_slew_rad_ns2']:.9g} | [{n}_pulse.csv]({n}_pulse.csv) |")
    for ax, channel in zip(axes, ['I', 'Q']):
        ax.set(title=f'{channel} channel: all stages on shared axes', ylabel=f'{channel} / 2pi (MHz)')
        ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel('Time (ns)')
    fig.suptitle('Pulse comparison (coincident curves indicate identical or very similar pulses)')
    fig.savefig(output/'pulse_stages.png', dpi=160); plt.close(fig)
    # A single time-aligned table, not separate per-stage tables.
    from pulse_methods import comparison_grid
    time, aligned = comparison_grid(pulses)
    columns, headers = [time], ['time_ns']
    for n in names:
        t, i, q = aligned[n]
        columns.extend([i/MHZ_TO_RAD_NS, q/MHZ_TO_RAD_NS])
        headers.extend([f'{n}_I_over_2pi_mhz', f'{n}_Q_over_2pi_mhz'])
    np.savetxt(output/'pulse_comparison.csv', np.column_stack(columns), delimiter=',',
               header=','.join(headers), comments='', fmt='%.12g')
    tables += ['', '[All pulse samples in one shared-time table](pulse_comparison.csv). '
               'Shorter pulses are zero-padded for this table only; simulations use actual durations. I/Q columns are drive amplitudes divided by 2pi, in MHz.']
    if (output/'pre_grape_pulse.csv').exists() and 'grape_proposal' in pulses:
        seed = np.loadtxt(output/'pre_grape_pulse.csv', delimiter=',', skiprows=1)[:, 1:]
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
        tables += ['', '| Change relative to pre-GRAPE seed | Peak I/Q change (MHz) | RMS I/Q change (MHz) |', '|---|---:|---:|']
        for name, style in [('grape_proposal', '-'), ('robust_selected', '--')]:
            difference = (np.column_stack(pulses[name][1:])-seed)/MHZ_TO_RAD_NS
            for column, ax in enumerate(axes):
                ax.plot(pulses[name][0], difference[:, column], style, label=LABELS[name])
            peak = np.linalg.norm(difference, axis=1).max()
            rms = np.sqrt(np.mean(np.sum(difference**2, axis=1)))
            tables.append(f'| {LABELS[name]} | {peak:.9g} | {rms:.9g} |')
        for ax, label in zip(axes, ['Delta I / 2pi (MHz)', 'Delta Q / 2pi (MHz)']):
            ax.set_ylabel(label); ax.legend(); ax.grid(alpha=.2)
        axes[-1].set_xlabel('Time (ns)')
        fig.suptitle('Pulse changes relative to the pre-GRAPE seed')
        fig.savefig(output/'pulse_changes.png', dpi=160); plt.close(fig)
        tables += ['', '![Pulse changes relative to pre-GRAPE seed](pulse_changes.png)',
                   'A zero final difference means the pre-GRAPE seed was retained; the GRAPE proposal is shown separately.']
    tables += ['', '## 3. NSGA-III, Monte Carlo selection and GRAPE', '',
               '| NSGA seed | Evaluations | Final archive hypervolume |', '|---|---:|---:|']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for seed in sorted({r['seed'] for r in history}):
        h = [r for r in history if r['seed'] == seed]
        axes[0].plot([r['generation'] for r in h], [r['archive_hypervolume'] for r in h], label=f'Seed {seed}')
        tables.append(f"| {seed} | {h[-1]['evaluations']} | {h[-1]['archive_hypervolume']:.9g} |")
    axes[0].set(title='NSGA-III search' if history else 'NSGA-III disabled', xlabel='Generation', ylabel='Archive hypervolume')
    if history:
        axes[0].legend()
    rankings = json.loads((output/'mc_selection_ranking.json').read_text())
    tables += ['', '| MC candidate | Selection P05 fidelity | Mean infidelity | Worst leakage | Eligible |', '|---|---:|---:|---:|---|']
    for r in rankings:
        m = r['statistics']
        tables.append(f"| {r['candidate']+1} | {m['p05_fidelity']:.9g} | {1-m['mean_fidelity']:.9g} | {m['worst_sample_leakage']:.9g} | {r['selection_feasible']} |")
    axes[1].plot([r['candidate']+1 for r in rankings], [100*r['statistics']['p05_fidelity'] for r in rankings], 'o-')
    axes[1].axvline(manifest['mc_selection']['candidate']+1, ls=':', color='k', label='MC winner')
    axes[1].set(title='Monte Carlo candidate selection', xlabel='Candidate', ylabel='P05 fidelity (%)'); axes[1].legend()
    gh = manifest.get('grape', {}).get('history', [])
    if gh: axes[2].plot([r['iteration'] for r in gh], [r['loss'] for r in gh], 'o-')
    axes[2].set(title=f"GRAPE (accepted: {manifest.get('grape', {}).get('accepted', False)})", xlabel='Iteration', ylabel='Training surrogate loss')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(output/'optimizer_stages.png', dpi=160); plt.close(fig)
    tables += ['', '| GRAPE iteration | Training loss |', '|---|---:|']
    tables.extend(f"| {r['iteration']} | {r['loss']:.9g} |" for r in gh)
    from grape_gradient_plot import save_grape_gradient_plot
    if save_grape_gradient_plot(output, gh):
        tables += ['', '![GRAPE Fourier-basis gradients](grape_gradients.png)',
                   '[Gradient data](grape_gradients.csv). Analytic derivatives of the full training loss '
                   '(including leakage and smoothness) with respect to dimensionless Fourier a/b coefficients. '
                   'These are objective gradients, not constrained optimality residuals; '
                   'active amplitude constraints can leave nonzero gradients at convergence.']
    tables += ['', '## 4. Detuning effect', '',
               'Only detuning varies; amplitude, anharmonicity and coherence are nominal. '
               'Five PDF grid points are connected as a visual guide, not a dense sweep.', '',
               '| Stage | Actual detuning (MHz) | Fidelity | Leakage |', '|---|---:|---:|---:|']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for n, color in zip(names, cycle(COLORS)):
        rows = detuning_rows(public[n])
        detuning = [manifest['config']['device'].get('detuning_mhz', 0)+r['detuning_mhz'] for r in rows]
        axes[0].plot(detuning, [100*r['fidelity'] for r in rows], 'o-', color=color, label=LABELS[n])
        axes[1].plot(detuning, [100*r['leakage'] for r in rows], 'o-', color=color, label=LABELS[n])
        tables.extend(f"| {LABELS[n]} | {d:g} | {r['fidelity']:.9g} | {r['leakage']:.9g} |" for d, r in zip(detuning, rows))
    for ax, ylabel in zip(axes, ['Fidelity (%)', 'Leakage (%)']):
        ax.set(xlabel='Actual detuning / 2pi (MHz)', ylabel=ylabel); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(output/'detuning_effect.png', dpi=160); plt.close(fig)
    for name in ['stage_metrics', 'pulse_stages', 'optimizer_stages', 'detuning_effect']:
        tables += ['', f'![{name}]({name}.png)']
    (output/'stage_comparisons.md').write_text('\n'.join(tables)+'\n')
    with (output/'report.md').open('a') as stream:
        stream.write('\n\nSee [all four stage comparison tables and plots](stage_comparisons.md).\n')

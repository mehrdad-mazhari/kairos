"""Post-optimization pulse plots; duration sweeps do not alter the selected gate."""
import argparse
from itertools import cycle
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from qutip import mesolve

from challenge_model import Device, Scenario, KETS, MHZ_TO_RAD_NS, hamiltonian, collapse_operators


def rabi_chevron(pulse, device, durations, detunings):
    """Stretch both final envelopes in time, keeping amplitude and Q/I unchanged.

    Every grid point starts in |0>. Detunings are actual rotating-frame detunings,
    not offsets from the device input. This is a diagnostic duration sweep, not
    a family of recalibrated DRAG gates.
    """
    t, i, q = pulse
    populations = np.empty((len(detunings), len(durations), 3))
    for row, detuning in enumerate(detunings):
        scenario = Scenario(detuning_mhz=float(detuning-device.detuning_mhz))
        for col, duration in enumerate(durations):
            if duration == 0:
                populations[row, col] = [1, 0, 0]
                continue
            stretched = t * duration/t[-1]
            result = mesolve(hamiltonian(stretched, i, q, device, scenario),
                             KETS[0].proj(), [0, float(duration)],
                             c_ops=collapse_operators(device, scenario),
                             e_ops=[k.proj() for k in KETS],
                             options={'atol': 1e-10, 'rtol': 1e-8,
                                      'max_step': min(device.dt_ns, duration/(len(t)-1))/2,
                                      'nsteps': 100000, 'normalize_output': False})
            populations[row, col] = [values[-1] for values in result.expect]
    if np.max(np.abs(populations.sum(axis=2)-1)) > 2e-6 or populations.min() < -2e-6:
        raise RuntimeError('Nonphysical Rabi chevron populations')
    return populations


def save_pulse_diagnostics(output, manifest, pulses):
    output = Path(output)
    device = Device(**manifest['config']['device'])
    t, i, q = pulses['robust_selected']
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for values, label in [(i, 'I'), (q, 'Q'), (np.hypot(i, q), '|I + iQ|')]:
        axes[0].plot(t, values/MHZ_TO_RAD_NS, label=label)
    from challenge_report import LABELS, COLORS
    for (name, (time, pi, pq)), color in zip(pulses.items(), cycle(COLORS)):
        axes[1].plot(time, (pi+pq)/MHZ_TO_RAD_NS, label=LABELS[name], color=color,
                     linestyle='--' if name == 'robust_selected' else '-')
    axes[0].set_title('Final pulse: I and Q on the same axes')
    axes[1].set_title('Literal I + Q sum: all stages (not the complex-envelope magnitude)')
    for ax in axes:
        ax.set_ylabel('Amplitude / 2π (MHz)'); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.2)
    axes[1].set_xlabel('Time (ns)')
    fig.savefig(output/'pulse_i_plus_q.png', dpi=160); plt.close(fig)
    from pulse_methods import comparison_grid
    comparison_time, aligned = comparison_grid(pulses)
    np.savetxt(output/'pulse_i_plus_q.csv', np.column_stack([comparison_time]+[(p[1]+p[2])/MHZ_TO_RAD_NS for p in aligned.values()]),
               delimiter=',', header=','.join(['time_ns']+[f'{n}_I_plus_Q_over_2pi_mhz' for n in pulses]), comments='')
    # Include the exact selected duration and nominal detuning in the sweep.
    durations = np.unique(np.r_[np.linspace(0, 4*t[-1], 61), t[-1]])
    span = max(40., 2*np.trapezoid(np.hypot(i, q), t)/t[-1]/MHZ_TO_RAD_NS)
    detunings = np.unique(np.r_[np.linspace(-span, span, 51), device.detuning_mhz])
    populations = rabi_chevron((t, i, q), device, durations, detunings)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, level, title in zip(axes, [1, 2], ['Rabi chevron: excited-state population P₁', 'Leakage population P₂']):
        mesh = ax.pcolormesh(durations, detunings, populations[:, :, level], shading='auto',
                             vmin=0, vmax=1 if level == 1 else None, cmap='viridis')
        fig.colorbar(mesh, ax=ax, label=f'P{level} (fraction)')
        ax.axvline(t[-1], color='white', ls='--', lw=1, label='Optimized Tg')
        ax.axhline(device.detuning_mhz, color='white', ls=':', lw=1)
        ax.set(title=title, xlabel='Pulse duration (ns)', ylabel='Actual detuning / 2π (MHz)')
        ax.legend(fontsize=8)
    fig.suptitle('Final pulse stretched in time at fixed amplitude; initial state |0>; T1/Tphi noise')
    fig.savefig(output/'rabi_chevron.png', dpi=160); plt.close(fig)
    dd, tt = np.meshgrid(detunings, durations, indexing='ij')
    np.savetxt(output/'rabi_chevron.csv', np.column_stack([tt.ravel(), dd.ravel(), populations.reshape(-1, 3)]),
               delimiter=',', header='duration_ns,actual_detuning_mhz,P0,P1,P2', comments='')
    text = ('\n\n## Rabi chevron and combined I/Q\n\n'
            '![Combined I/Q and literal sum](pulse_i_plus_q.png)\n\n'
            'I + Q is the literal sum; the complex-envelope magnitude is sqrt(I² + Q²). '
            '[Sum samples](pulse_i_plus_q.csv).\n\n'
            '![Rabi chevron](rabi_chevron.png)\n\n'
            'Each point starts in |0> and scales the final pulse duration while preserving its I/Q amplitudes. '
            'T1, Tphi and anharmonicity stay fixed. Q is not recalibrated as a DRAG derivative during this diagnostic sweep. '
            'The vertical line marks the optimized Tg; the horizontal line marks nominal detuning. '
            'P1 is population, not six-state gate fidelity. This sweep does not change optimization or select a new gate. '
            '[Chevron data](rabi_chevron.csv).\n')
    (output/'pulse_diagnostics.md').write_text(text.lstrip())
    for filename in ['report.md', 'stage_comparisons.md']:
        path = output/filename
        if path.exists():
            content = path.read_text()
            link = '\n\n[Additional Rabi chevron and combined I/Q plots](pulse_diagnostics.md).\n'
            if link not in content:
                path.write_text(content+link)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Existing optimization result directory')
    args = parser.parse_args()
    manifest = json.loads((args.output/'summary.json').read_text())
    pulses = {name: tuple(np.loadtxt(args.output/f'{name}_pulse.csv', delimiter=',', skiprows=1).T)
              for name in manifest['variants']}
    save_pulse_diagnostics(args.output, manifest, pulses)

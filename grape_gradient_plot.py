"""Plot recorded analytic Fourier-GRAPE objective gradients."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_grape_gradient_plot(output, history):
    rows = [r for r in history if 'coefficient_gradient' in r]
    if not rows:
        return False  # Older runs did not record gradients; do not invent them.
    output = Path(output)
    gradients = np.asarray([r['coefficient_gradient'] for r in rows])
    harmonics = gradients.shape[1]//2
    labels = [f'{channel}{j+1}' for channel in ['a', 'b'] for j in range(harmonics)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for restart in sorted({r['restart'] for r in rows}):
        selected = [r for r in rows if r['restart'] == restart]
        x = [r['iteration'] for r in selected]
        g = np.asarray([r['coefficient_gradient'] for r in selected])
        axes[0].semilogy(x, np.maximum(np.linalg.norm(g, axis=1), 1e-16), label=f'Start {restart}')
        for j, label in enumerate(labels):
            ax = axes[1 if j < harmonics else 2]
            ax.plot(x, g[:, j], color=f'C{j % harmonics}',
                    label=label if restart == rows[0]['restart'] else None)
        for ax in axes:
            ax.axvline(x[0], color='gray', ls=':', alpha=.4)
    axes[0].set_ylabel('Gradient L2 norm'); axes[0].legend()
    axes[1].set_ylabel('∂ loss / ∂ a'); axes[2].set_ylabel('∂ loss / ∂ b')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[1].legend(ncol=harmonics); axes[2].legend(ncol=harmonics)
    axes[2].set_xlabel('Recorded optimizer step (dotted lines mark restart starts)')
    fig.suptitle('GRAPE Fourier-basis gradients of the training objective')
    fig.savefig(output/'grape_gradients.png', dpi=160); plt.close(fig)
    np.savetxt(output/'grape_gradients.csv',
               np.column_stack([[r['iteration'] for r in rows], [r['restart'] for r in rows],
                                np.linalg.norm(gradients, axis=1), gradients]), delimiter=',',
               header=','.join(['step', 'restart', 'gradient_l2_norm']+[f'dloss_d{v}' for v in labels]), comments='')
    return True

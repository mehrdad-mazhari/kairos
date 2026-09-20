import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, destroy, mesolve, qeye
from core import build_transmon, device_metrics
from calibration import grid_minimize

LEVELS = 3
RABI_RATE_GHZ = 0.025
T1_NS, T2_NS = 30_000.0, 20_000.0
RAMSEY_DETUNING_GHZ = 0.001


def collapse_operators(lowering, number, t1_ns, t2_ns):
    gamma_phi = max(0.0, 1.0 / t2_ns - 1.0 / (2.0 * t1_ns))
    operators = [np.sqrt(1.0 / t1_ns) * lowering]
    if gamma_phi > 0:
        operators.append(np.sqrt(2.0 * gamma_phi) * number)
    return operators


def driven_hamiltonian(detuning, omega_rabi, alpha, lowering, number):
    identity = qeye(LEVELS)
    return (-detuning * number
            + 0.5 * alpha * number * (number - identity)
            + 0.5 * omega_rabi * (lowering + lowering.dag()))


def simulate_chevron(detunings, times, omega_rabi, alpha, c_ops):
    lowering = destroy(LEVELS)
    number = lowering.dag() * lowering
    initial = basis(LEVELS, 0)
    p1, p2 = basis(LEVELS, 1).proj(), basis(LEVELS, 2).proj()
    excited = np.empty((len(times), len(detunings)))
    leakage = np.empty_like(excited)
    for column, detuning in enumerate(detunings):
        hamiltonian = driven_hamiltonian(detuning, omega_rabi, alpha, lowering, number)
        result = mesolve(hamiltonian, initial, times, c_ops=c_ops, e_ops=[p1, p2])
        excited[:, column] = np.real(result.expect[0])
        leakage[:, column] = np.real(result.expect[1])
    return excited, leakage


def optimize_calibration(omega_rabi, alpha, c_ops, points_per_axis=21):
    """Optimize square-pulse duration (ns) and detuning (GHz) for final P1."""
    lowering = destroy(LEVELS)
    number = lowering.dag() * lowering
    pi_time = np.pi / omega_rabi

    def objective(parameters):
        duration, detuning_ghz = parameters
        hamiltonian = driven_hamiltonian(
            2 * np.pi * detuning_ghz, omega_rabi, alpha, lowering, number)
        result = mesolve(hamiltonian, basis(LEVELS, 0), [0, duration],
                         c_ops=c_ops, e_ops=[basis(LEVELS, 1).proj()])
        return max(0.0, 1 - float(np.real(result.expect[0][-1])))

    return grid_minimize(objective, [(0.7 * pi_time, 1.3 * pi_time), (-0.01, 0.01)],
                    x0=[pi_time, 0], points_per_axis=points_per_axis)


def main():
    metrics = device_metrics(build_transmon())
    alpha = 2 * np.pi * metrics["alpha"]
    omega_rabi = 2 * np.pi * RABI_RATE_GHZ
    pi_time = np.pi / omega_rabi

    detunings_ghz = np.linspace(-0.12, 0.12, 61)
    detunings = 2 * np.pi * detunings_ghz
    times = np.linspace(0, 4 * pi_time, 180)
    lowering = destroy(LEVELS)
    number = lowering.dag() * lowering
    c_ops = collapse_operators(lowering, number, T1_NS, T2_NS)
    optimization = optimize_calibration(omega_rabi, alpha, c_ops)
    best_time, best_detuning = optimization.x
    population, leakage = simulate_chevron(detunings, times, omega_rabi, alpha, c_ops)
    resonance_index = np.argmin(np.abs(detunings))

    t1_times = np.linspace(0, 3 * T1_NS, 250)
    t1_signal = np.exp(-t1_times / T1_NS)
    ramsey_times = np.linspace(0, 3 * T2_NS, 500)
    ramsey_signal = 0.5 * (1 + np.exp(-ramsey_times / T2_NS)
                           * np.cos(2 * np.pi * RAMSEY_DETUNING_GHZ * ramsey_times))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    image = axes[0, 0].pcolormesh(detunings_ghz, times, population,
                                  shading="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].plot(best_detuning, best_time, "r*", markersize=12)
    axes[0, 0].set(title="Rabi chevron", xlabel="Drive detuning (GHz)",
                   ylabel="Pulse duration (ns)")
    fig.colorbar(image, ax=axes[0, 0], label=r"$P_1$")

    axes[0, 1].plot(times, population[:, resonance_index], label=r"$P_1$")
    axes[0, 1].plot(times, leakage[:, resonance_index], label=r"$P_2$ leakage")
    axes[0, 1].axvline(pi_time, color="black", ls="--", label=fr"$t_\pi={pi_time:.2f}$ ns")
    axes[0, 1].set(title="On-resonance calibration", xlabel="Pulse duration (ns)",
                   ylabel="Population", ylim=(-0.02, 1.02))
    axes[0, 1].legend(fontsize=8)

    leak_image = axes[0, 2].pcolormesh(detunings_ghz, times, 100 * leakage,
                                       shading="auto", cmap="magma")
    axes[0, 2].set(title="Leakage map", xlabel="Drive detuning (GHz)",
                   ylabel="Pulse duration (ns)")
    fig.colorbar(leak_image, ax=axes[0, 2], label=r"$P_2$ (%)")

    axes[1, 0].plot(t1_times / 1e3, t1_signal, color="mediumseagreen")
    axes[1, 0].axvline(T1_NS / 1e3, color="black", ls="--",
                       label=fr"$T_1={T1_NS/1e3:.1f}\ \mu s$")
    axes[1, 0].set(title=r"Energy relaxation ($T_1$)", xlabel=r"Delay ($\mu$s)", ylabel=r"$P_1$")
    axes[1, 0].legend()

    axes[1, 1].plot(ramsey_times / 1e3, ramsey_signal, color="orchid")
    axes[1, 1].plot(ramsey_times / 1e3, 0.5 * (1 + np.exp(-ramsey_times / T2_NS)),
                    "k--", alpha=0.5, label=fr"$T_2^*={T2_NS/1e3:.1f}\ \mu s$")
    axes[1, 1].set(title=r"Ramsey ($T_2^*$)", xlabel=r"Delay ($\mu$s)",
                   ylabel="Excited probability")
    axes[1, 1].legend()

    evaluations = np.arange(1, len(optimization.values) + 1)
    axes[1, 2].semilogy(evaluations, np.maximum(optimization.values, 1e-10), ".")
    axes[1, 2].semilogy(evaluations, np.maximum(optimization.best_so_far, 1e-10))
    axes[1, 2].set(title="Grid pulse calibration", xlabel="Simulation evaluation",
                   ylabel="Transfer error (1 - P1)")
    pi_index = np.argmin(np.abs(times - pi_time))
    max_pi_leakage = leakage[pi_index, resonance_index]
    summary = (f"f01: {metrics['f01']:.5f} GHz\n"
               f"anharmonicity: {metrics['alpha'] * 1e3:.2f} MHz\n"
               f"Rabi rate: {RABI_RATE_GHZ * 1e3:.1f} MHz\n"
               f"pi-pulse time: {pi_time:.2f} ns\n"
               f"simulated pi leakage: {100 * max_pi_leakage:.3f}%\n"
               f"T1: {T1_NS / 1e3:.1f} us\nT2*: {T2_NS / 1e3:.1f} us")
    summary += (f"\nGrid duration: {best_time:.4f} ns"
                f"\nGrid detuning: {best_detuning * 1e3:.4f} MHz"
                f"\nTransfer error: {optimization.values[0]:.6g} -> {optimization.fun:.6g}")
    fig.suptitle("Three-level transmon control and coherence benchmark", fontsize=16)
    for ax in axes.flat[:5]:
        ax.grid(alpha=0.2)
    print(summary)
    plt.show()


if __name__ == "__main__":
    main()

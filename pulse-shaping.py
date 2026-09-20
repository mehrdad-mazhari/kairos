import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, destroy, mesolve, qeye

from core import build_transmon, device_metrics
from pulse_shapes import (integrate, normalize_pi_area, square_pulse,
                          gaussian_pulse, drag_pulse)


LEVELS = 3
GATE_TIME_NS = 20.0
GAUSSIAN_SIGMA_NS = GATE_TIME_NS / 4.0
TIME_POINTS = 301
T1_NS = 30_000.0
T2_NS = 20_000.0


def collapse_operators(lowering, number):
    gamma_phi = max(0.0, 1.0 / T2_NS - 1.0 / (2.0 * T1_NS))
    result = [np.sqrt(1.0 / T1_NS) * lowering]
    if gamma_phi > 0:
        result.append(np.sqrt(2.0 * gamma_phi) * number)
    return result


def simulate_pulse(times, in_phase, quadrature, alpha_rad_ns,
                   detuning_rad_ns=0.0, amplitude_scale=1.0):
    lowering = destroy(LEVELS)
    number = lowering.dag() * lowering
    identity = qeye(LEVELS)
    static = (-detuning_rad_ns * number
              + 0.5 * alpha_rad_ns * number * (number - identity))
    drive_x = 0.5 * (lowering + lowering.dag())
    drive_y = -0.5j * (lowering - lowering.dag())
    hamiltonian = [static,
                   [drive_x, amplitude_scale * in_phase],
                   [drive_y, amplitude_scale * quadrature]]
    projectors = [basis(LEVELS, level).proj() for level in range(LEVELS)]
    return mesolve(hamiltonian, basis(LEVELS, 0), times,
                   c_ops=collapse_operators(lowering, number), e_ops=projectors)


def final_metrics(result):
    populations = np.real(np.asarray(result.expect))
    p0, p1, p2 = populations[:, -1]
    return {"p0": p0, "p1": p1, "p2": p2,
            "transfer_error": max(0.0, 1.0 - p1)}, populations


def optimize_drag(times, alpha_rad_ns, beta_values):
    scores = []
    leakages = []
    for beta in beta_values:
        pulse_i, pulse_q = drag_pulse(times, alpha_rad_ns, beta)
        metrics, _ = final_metrics(simulate_pulse(times, pulse_i, pulse_q, alpha_rad_ns))
        scores.append(metrics["transfer_error"])
        leakages.append(metrics["p2"])
    best_index = int(np.argmin(scores))
    return beta_values[best_index], np.asarray(scores), np.asarray(leakages)


def robustness_scan(times, pulses, alpha_rad_ns, scan_values, mode):
    results = {name: [] for name in pulses}
    for name, (pulse_i, pulse_q) in pulses.items():
        for value in scan_values:
            kwargs = ({"amplitude_scale": value} if mode == "amplitude"
                      else {"detuning_rad_ns": 2 * np.pi * value})
            metrics, _ = final_metrics(
                simulate_pulse(times, pulse_i, pulse_q, alpha_rad_ns, **kwargs)
            )
            results[name].append(metrics["p1"])
    return results


def main():
    device = device_metrics(build_transmon())
    alpha_rad_ns = 2 * np.pi * device["alpha"]
    times = np.linspace(0.0, GATE_TIME_NS, TIME_POINTS)

    beta_values = np.linspace(-2, 2, 81)
    best_beta, drag_errors, drag_leakages = optimize_drag(times, alpha_rad_ns, beta_values)
    best_sigma, best_amplitude = GAUSSIAN_SIGMA_NS, 1.0
    gaussian_i = gaussian_pulse(times)
    drag_i, drag_q = drag_pulse(times, alpha_rad_ns, best_beta, best_sigma)
    pulses = {
        "Square": (square_pulse(times), np.zeros_like(times)),
        "Gaussian": (gaussian_i, np.zeros_like(times)),
        "DRAG": (drag_i, drag_q),
    }

    metrics_by_pulse = {}
    populations_by_pulse = {}
    for name, (pulse_i, pulse_q) in pulses.items():
        metrics_by_pulse[name], populations_by_pulse[name] = final_metrics(
            simulate_pulse(times, pulse_i, pulse_q, alpha_rad_ns)
        )

    amplitude_scales = np.linspace(0.85, 1.15, 31)
    detuning_ghz = np.linspace(-0.025, 0.025, 31)
    amplitude_robustness = robustness_scan(
        times, pulses, alpha_rad_ns, amplitude_scales, "amplitude"
    )
    detuning_robustness = robustness_scan(
        times, pulses, alpha_rad_ns, detuning_ghz, "detuning"
    )

    colors = {"Square": "tab:blue", "Gaussian": "mediumseagreen", "DRAG": "orchid"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    axes[0, 0].plot(times, pulses["Square"][0] / (2 * np.pi),
                    label="Square", color=colors["Square"])
    axes[0, 0].plot(times, pulses["Gaussian"][0] / (2 * np.pi),
                    label="Gaussian", color=colors["Gaussian"])
    drag_magnitude = np.sqrt(drag_i**2 + drag_q**2)
    axes[0, 0].plot(times, drag_magnitude / (2 * np.pi),
                    label=r"DRAG $|I+iQ|$", color=colors["DRAG"])
    axes[0, 0].plot(times, drag_q / (2 * np.pi), "--",
                    label="DRAG Q", color="darkviolet", alpha=0.85)
    axes[0, 0].set(title="Pulse-envelope magnitude", xlabel="Time (ns)",
                   ylabel="Drive amplitude (GHz)")
    axes[0, 0].legend(fontsize=8)

    for name, populations in populations_by_pulse.items():
        axes[0, 1].plot(times, populations[1], label=fr"{name} $P_1$", color=colors[name])
        axes[0, 1].plot(times, populations[2], ":", color=colors[name], alpha=0.8)
    axes[0, 1].set(title=r"Population transfer (dotted: $P_2$)", xlabel="Time (ns)",
                   ylabel="Population", ylim=(-0.02, 1.02))
    axes[0, 1].legend(fontsize=8)

    names = list(pulses)
    x_positions = np.arange(len(names))
    axes[0, 2].bar(x_positions - 0.18,
                   [100 * metrics_by_pulse[name]["transfer_error"] for name in names],
                   width=0.36, label="Transfer error")
    axes[0, 2].bar(x_positions + 0.18,
                   [100 * metrics_by_pulse[name]["p2"] for name in names],
                   width=0.36, label=r"$P_2$ leakage")
    axes[0, 2].set_xticks(x_positions, names)
    axes[0, 2].set(title="Final-state errors", ylabel="Percent", yscale="log")
    axes[0, 2].legend(fontsize=8)

    for name in names:
        axes[1, 0].plot(100 * (amplitude_scales - 1), amplitude_robustness[name],
                        label=name, color=colors[name])
    axes[1, 0].set(title="Amplitude-error robustness", xlabel="Amplitude error (%)",
                   ylabel=r"Final $P_1$", ylim=(0, 1.02))

    for name in names:
        axes[1, 1].plot(1e3 * detuning_ghz, detuning_robustness[name],
                        label=name, color=colors[name])
    axes[1, 1].set(title="Frequency-error robustness", xlabel="Detuning (MHz)",
                   ylabel=r"Final $P_1$", ylim=(0, 1.02))

    axes[1, 2].semilogy(beta_values, 100 * np.maximum(drag_errors, 1e-10), label="Transfer error")
    axes[1, 2].semilogy(beta_values, 100 * np.maximum(drag_leakages, 1e-10), label="Leakage")
    axes[1, 2].set(title="DRAG grid calibration", xlabel="DRAG beta", ylabel="Error (%)")
    axes[1, 2].legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"Transmon pulse shaping — gate={GATE_TIME_NS:.1f} ns, "
        f"anharmonicity={device['alpha'] * 1e3:.1f} MHz", fontsize=15
    )

    print("Pulse-shaping comparison")
    print(f"  f01 = {device['f01']:.6f} GHz")
    print(f"  anharmonicity = {device['alpha'] * 1e3:.3f} MHz")
    print(f"  optimized DRAG beta = {best_beta:.3f}")
    print(f"  optimized sigma = {best_sigma:.3f} ns, amplitude = {best_amplitude:.5f}")
    for name in names:
        values = metrics_by_pulse[name]
        print(f"  {name:8s}: P1={values['p1']:.6f}, "
              f"leakage={100 * values['p2']:.4f}%, "
              f"transfer error={100 * values['transfer_error']:.4f}%")
    plt.show()


if __name__ == "__main__":
    main()

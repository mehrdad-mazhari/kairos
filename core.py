import matplotlib.pyplot as plt
import numpy as np
import scqubits as scq

EJ_GHZ, EC_GHZ, NG, NCUT = 15.0, 0.30, 0.0, 30
RESONATOR_GHZ, COUPLING_GHZ, LEVELS = 7.0, 0.10, 6


def build_transmon(ej=EJ_GHZ, ec=EC_GHZ, ng=NG, ncut=NCUT):
    return scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=ncut)


def device_metrics(transmon, resonator_ghz=RESONATOR_GHZ, coupling_ghz=COUPLING_GHZ):
    energies = np.asarray(transmon.eigenvals(evals_count=LEVELS))
    transitions = np.diff(energies)
    f01, f12 = transitions[:2]
    alpha = f12 - f01
    detuning = f01 - resonator_ghz
    chi = coupling_ghz**2 * alpha / (detuning * (detuning + alpha))
    n_crit = detuning**2 / (4 * coupling_ghz**2)
    return {"energies": energies, "transitions": transitions, "f01": f01,
            "alpha": alpha, "ej_over_ec": transmon.EJ / transmon.EC,
            "detuning": detuning, "chi": chi, "n_crit": n_crit}


def charge_dispersion(transmon, ng_values, levels=LEVELS):
    original_ng = transmon.ng
    spectra = []
    try:
        for ng in ng_values:
            transmon.ng = ng
            spectra.append(transmon.eigenvals(evals_count=levels))
    finally:
        transmon.ng = original_ng
    return np.asarray(spectra)


def plot_device_dashboard(transmon, metrics):
    ng_values = np.linspace(-0.5, 0.5, 121)
    spectra = charge_dispersion(transmon, ng_values)
    relative_spectra = spectra - spectra[:, [0]]
    dispersion_khz = np.ptp(spectra[:, 1] - spectra[:, 0]) * 1e6

    ratios = np.linspace(10, 100, 100)
    f01_ratio, alpha_ratio = np.empty_like(ratios), np.empty_like(ratios)
    for index, ratio in enumerate(ratios):
        trial_energies = build_transmon(ej=ratio * transmon.EC,
                                        ec=transmon.EC).eigenvals(evals_count=3)
        f01_ratio[index] = trial_energies[1] - trial_energies[0]
        alpha_ratio[index] = trial_energies[2] - 2 * trial_energies[1] + trial_energies[0]

    charge_matrix = np.abs(transmon.matrixelement_table("n_operator", evals_count=LEVELS))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for level in range(LEVELS):
        axes[0, 0].plot(ng_values, relative_spectra[:, level], label=fr"$|{level}\rangle$")
    axes[0, 0].set(title="Charge dispersion", xlabel=r"Offset charge $n_g$",
                   ylabel="Energy above ground (GHz)")
    axes[0, 0].legend(ncol=2, fontsize=8)

    axes[0, 1].plot(ratios, f01_ratio, label=r"$f_{01}$")
    axes[0, 1].plot(ratios, np.abs(alpha_ratio), label=r"$|\alpha|$")
    axes[0, 1].axvline(metrics["ej_over_ec"], color="black", ls="--", alpha=0.6)
    axes[0, 1].set(title="Design trade-off", xlabel=r"$E_J/E_C$", ylabel="Frequency (GHz)")
    axes[0, 1].legend()

    indices = np.arange(len(metrics["transitions"]))
    axes[1, 0].bar(indices, metrics["transitions"], color="mediumseagreen")
    axes[1, 0].set_xticks(indices, [fr"$f_{{{i}{i+1}}}$" for i in indices])
    axes[1, 0].set(title="Adjacent transitions", ylabel="Frequency (GHz)")

    image = axes[1, 1].imshow(charge_matrix, origin="lower", cmap="magma")
    axes[1, 1].set(title=r"Charge coupling $|\langle i|n|j\rangle|$",
                   xlabel="State j", ylabel="State i")
    fig.colorbar(image, ax=axes[1, 1], shrink=0.85)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Transmon — EJ={transmon.EJ:.2f} GHz, EC={transmon.EC:.2f} GHz, "
                 f"EJ/EC={metrics['ej_over_ec']:.1f}", fontsize=15)
    return fig, dispersion_khz


def main():
    transmon = build_transmon()
    metrics = device_metrics(transmon)
    _, dispersion_khz = plot_device_dashboard(transmon, metrics)
    print("Transmon device metrics")
    print(f"  f01               = {metrics['f01']:.6f} GHz")
    print(f"  anharmonicity     = {metrics['alpha'] * 1e3:.3f} MHz")
    print(f"  EJ/EC             = {metrics['ej_over_ec']:.1f}")
    print(f"  charge dispersion = {dispersion_khz:.3f} kHz")
    print(f"  qubit-res detuning= {metrics['detuning']:.4f} GHz")
    print(f"  dispersive chi    = {metrics['chi'] * 1e3:.3f} MHz (approx.)")
    print(f"  critical photons  = {metrics['n_crit']:.1f} (approx.)")
    plt.show()


if __name__ == "__main__":
    main()

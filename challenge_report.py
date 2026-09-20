"""Static scientific figures and a self-contained Markdown technical report."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qutip import mesolve

from challenge_model import (KETS, MHZ_TO_RAD_NS, STATE_NAMES, TEST_STATES,
                             collapse_operators, hamiltonian, make_pulse)

from pulse_methods import METHOD_LABELS as LABELS, parameter_rows
from itertools import cycle
COLORS = ['#708090', '#c98239', '#5367b0', '#00847b', '#b54885', '#8b6c42', '#725bd6', '#222222']


def save_plots(output, manifest, pareto, search_history, variants, mc_rows,
               public_rows, device):
    output = Path(output)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    final = variants["robust_selected"]
    times, pulse_i, pulse_q = np.loadtxt(output / "pulse.csv", delimiter=",", skiprows=1).T
    axes[0, 0].plot(times, pulse_i / MHZ_TO_RAD_NS, label="I")
    axes[0, 0].plot(times, pulse_q / MHZ_TO_RAD_NS, label="Q")
    axes[0, 0].plot(times, np.hypot(pulse_i, pulse_q) / MHZ_TO_RAD_NS, "k--", label="Magnitude")
    axes[0, 0].axhline(device.amplitude_limit_mhz, color="red", ls=":", label="Limit")
    axes[0, 0].set(title="Final exported pulse", xlabel="Time (ns)", ylabel="Amplitude / 2 pi (MHz)")
    axes[0, 0].legend(fontsize=8)

    if pareto:
        f = np.array([r["objectives"] for r in pareto])
        sc = axes[0, 1].scatter(f[:, 1], f[:, 0], c=np.log10(np.maximum(f[:, 2], 1e-12)),
                               s=28, cmap="viridis")
        axes[0, 1].set(title="Candidate Pareto set", xlabel="Training worst gate infidelity",
                       ylabel="Training mean gate infidelity", yscale="log")
        from matplotlib.ticker import MaxNLocator
        axes[0, 1].xaxis.set_major_locator(MaxNLocator(4))
        axes[0, 1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=True)
        fig.colorbar(sc, ax=axes[0, 1], label="log10 worst training leakage")

    else:
        axes[0, 1].text(.5, .5, 'No feasible Pareto archive', ha='center', transform=axes[0, 1].transAxes)
    for seed in sorted({h["seed"] for h in search_history}):
        history = [h for h in search_history if h["seed"] == seed]
        axes[0, 2].plot([h["generation"] for h in history],
                        [h["archive_hypervolume"] for h in history], label=f"Seed {seed}")
    axes[0, 2].set(title="NSGA-III search diagnostics", xlabel="Generation",
                   ylabel="Feasible archive hypervolume (fixed scales)")
    if search_history:
        axes[0, 2].legend(fontsize=8)
    else:
        axes[0, 2].set_title('NSGA-III disabled')

    positions = np.arange(len(variants))
    nominal_errors = [1-manifest["variants"][name]["nominal"]["fidelity"] for name in variants]
    nominal_leakage = [manifest["variants"][name]["nominal"]["leakage"] for name in variants]
    axes[1, 0].bar(positions-.18, np.maximum(nominal_errors, 1e-12), width=.36, label="Gate infidelity")
    axes[1, 0].bar(positions+.18, np.maximum(nominal_leakage, 1e-12), width=.36, label="Mean leakage")
    axes[1, 0].set_xticks(positions, [LABELS[n] for n in variants], rotation=25, ha="right")
    axes[1, 0].set(title="Nominal six-state metrics", ylabel="Error / population", yscale="log")
    axes[1, 0].legend(fontsize=8)

    for (name, rows), color in zip(mc_rows.items(), cycle(COLORS)):
        fidelity = np.sort([r["fidelity"] for r in rows])
        axes[1, 1].step(fidelity, np.arange(1, len(rows)+1)/len(rows), where="post",
                        label=LABELS[name], color=color)
    axes[1, 1].axvline(.990, color="black", ls=":", label="PDF nominal minimum")
    axes[1, 1].set(title="Held-out Monte Carlo", xlabel="Average X-gate fidelity",
                   ylabel="Empirical cumulative probability")
    axes[1, 1].legend(fontsize=7, loc="upper left")

    rows = [r for r in public_rows["robust_selected"] if r["alpha_shift_mhz"] == 0]
    grid = np.array([r["fidelity"] for r in rows]).reshape(3, 5)
    im = axes[1, 2].imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    axes[1, 2].set_xticks(range(5), [device.detuning_mhz + d for d in [-3, -1, 0, 1, 3]])
    axes[1, 2].set_yticks(range(3), [.97, 1, 1.03])
    axes[1, 2].set(title=f"Public-grid slice: alpha / 2 pi = {device.alpha_mhz:g} MHz",
                   xlabel="Residual detuning (MHz)", ylabel="Amplitude scale")
    fig.colorbar(im, ax=axes[1, 2], label="Average X-gate fidelity")
    for ax in axes.flat[:5]:
        ax.grid(alpha=.2)
    fig.suptitle("Three-level X gate: method comparison and manufacturing robustness", fontsize=15)
    fig.savefig(output / "optimization_dashboard.png", dpi=170)
    fig.savefig(output / "optimization_dashboard.svg")
    plt.close(fig)

    if "mc_selection" in manifest:
        rankings = json.loads((output / "mc_selection_ranking.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        positions = np.arange(1, len(rankings)+1)
        for metric, label, marker in [("mean_fidelity", "Mean fidelity", "o"),
                                       ("p05_fidelity", "5th-percentile fidelity", "s"),
                                       ("worst_sample_fidelity", "Lowest sampled fidelity", "v")]:
            values = [100*r["statistics"][metric] for r in rankings]
            axes[0].plot(positions, values, marker=marker, ms=5, label=label)
        winner = manifest["mc_selection"]["candidate"] + 1
        axes[0].axvline(winner, color="black", ls=":", label=f"Selected candidate {winner}")
        axes[0].set_xticks(positions)
        axes[0].set(title="Manufacturing-range candidate selection",
                     xlabel="Candidate (ordered by optimizer training loss)", ylabel="X-gate fidelity (%)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=.2)
        rows = mc_rows["robust_selected"]
        scatter = axes[1].scatter([device.detuning_mhz + r["detuning_mhz"] for r in rows],
                                  [100*r["fidelity"] for r in rows],
                                  c=[100*(r["amplitude_scale"]-1) for r in rows],
                                  s=12, alpha=.65, cmap="coolwarm")
        axes[1].set(title="Selected pulse: independent manufactured-device samples",
                     xlabel="Residual detuning (MHz)", ylabel="X-gate fidelity (%)")
        fig.colorbar(scatter, ax=axes[1], label="Amplitude error (%)")
        axes[1].grid(alpha=.2)
        fig.savefig(output / "manufacturing_selection.png", dpi=170)
        plt.close(fig)

    h = hamiltonian(times, pulse_i, pulse_q, device)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, state, name in zip(axes.flat, TEST_STATES, STATE_NAMES):
        result = mesolve(h, state, times, c_ops=collapse_operators(device),
                         e_ops=[ket.proj() for ket in KETS],
                         options={"atol": 1e-10, "rtol": 1e-8, "max_step": .1,
                                  "normalize_output": False})
        for level, population in enumerate(result.expect):
            ax.plot(times, population, label=f"P{level}")
        ax.set(title=f"Input |{name}>", xlabel="Time (ns)", ylabel="Population", ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Final pulse: all six input-state population trajectories", fontsize=15)
    fig.savefig(output / "six_state_dynamics.png", dpi=170)
    plt.close(fig)



def write_report(output, manifest):
    """Describe the actual NSGA-III run without asserting global convergence."""
    output = Path(output)
    config = manifest["config"]
    final = manifest["variants"]["robust_selected"]
    mc = final["monte_carlo"]
    quality = config["quality_targets"]
    selection = manifest["mc_selection"]
    p = final["parameters"]
    from pulse_coefficients import pulse_coefficients
    coefficients = manifest.get("pulse_coefficients") or pulse_coefficients(p, config["device"]["dt_ns"])
    coefficient_table = ["| Envelope | A (rad/ns) | B (dimensionless) | A*B offset (rad/ns) |",
                         "|---|---:|---:|---:|"]
    for name, row in coefficients.items():
        coefficient_table.append(f"| {name} | {row['A_rad_per_ns']:.10g} | {row['B_dimensionless']:.10g} | {row['B_offset_rad_per_ns']:.10g} |")
    histories = json.loads((output / "nsga_history.json").read_text())
    table = ["| Pulse | Duration ns | Nominal F | Nominal leakage | Validation mean F | Validation P05 F | P05 95% lower bound |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in manifest["variants"].items():
        m = r["monte_carlo"]
        table.append(f"| {LABELS[name]} | {r['parameters']['duration_ns']:.1f} | "
                     f"{r['nominal']['fidelity']:.8f} | {r['nominal']['leakage']:.3e} | "
                     f"{m['mean_fidelity']:.8f} | {m['p05_fidelity']:.8f} | {m['p05_fidelity_lcb95']:.8f} |")
    search_table = ["| Search seed | Generations | Evaluations | Archive HV | Relative HV gain over last 10 generations |",
                    "|---|---:|---:|---:|---:|"]
    for seed in sorted({h["seed"] for h in histories}):
        last = [h for h in histories if h["seed"] == seed][-1]
        gain = last["hypervolume_gain_last_10"]
        search_table.append(f"| {seed} | {last['generation']} | {last['evaluations']} | "
                            f"{last['archive_hypervolume']:.6f} | {gain if gain is not None else 'insufficient history'} |")
    grape = manifest.get('grape', {})
    grape_note = ("GRAPE selected: final pulse.csv contains refined I/Q samples. Use the exact selected pulse parameters and nested seed specification above. The exported waveform is the final representation."
                  if grape.get('selected', grape.get('accepted')) else "GRAPE refinement was not accepted or was disabled; a reference/search pulse remains final; its exact family is specified above.")
    status = "PASS" if manifest["quality_passed"] else "FAIL - exported pulse is a diagnostic candidate"
    reference = manifest["variants"].get("nsga_selected", manifest["variants"]["drag_baseline"])["monte_carlo"][config["mc_selection_criterion"]]
    comparison = ("The independent validation result did not improve on the training-selected reference."
                  if mc[config["mc_selection_criterion"]] <= reference else
                  "The independent validation statistic was higher than the training-selected reference.")
    selected_parameters = ''
    if manifest.get('selected_pulse_spec'):
        selected_parameters = '## Selected pulse: exact parameters\n\n' + '\n'.join(
            f'- **{label}:** {value}' for label, value in parameter_rows(manifest['selected_pulse_spec']))
        selected_parameters += '\n\nFull precision: [selected_pulse_parameters.json](selected_pulse_parameters.json).\n'
        selected_parameters += '\n## Selection eligibility\n\n| Method | Eligible | Reason |\n|---|---|---|\n'
        for name, variant in manifest['variants'].items():
            selection = variant.get('selection')
            reason = '; '.join(selection.get('rejection_reasons', [])) if selection else 'Not shortlisted'
            selected_parameters += f"| {LABELS[name]} | {selection['selection_feasible'] if selection else 'N/A'} | {reason or 'Passed selection constraints'} |\n"
        selected_parameters += '\nThe six analytic references are fixed designs, not independently optimized. Higher-order DRAG uses a second-order Hermite envelope. Wah-Wah is tested only on this single-transmon model; spectator leakage is not modeled. BB1 and CORPSE use DRAG-shaped segments and longer hardware-feasible durations. They are comparison-only for fixed-Tg selection; each is simulated for its actual duration.\n'
    report = f"""# Transmon X-gate method comparison

**Strict engineering quality: {status}.**

Reference problem: Dr. Tabatabaei.pdf, PolyHaQ PQO-2026-02, pages 3-9.
Workflow: baselines -> optional NSGA-III -> optional GRAPE -> final Monte Carlo selection -> independent validation.
NSGA enabled: {config.get("nsga_enabled", True)}. Selected method: {manifest.get("selected_method", "legacy run")}.

{selected_parameters}

{grape_note}

GRAPE adapts the supplied Rowland & Jones paper (arXiv:1203.6260,
Sections 1(b)-1(d), 2(a)) and Wright et al., Phys. Rev. Applied 25, 054037
(2026), Eq. (2), Eq. (4) and Appendix B. I uses odd sine harmonics and Q uses
even sine harmonics, with five coefficients each by default. Exact Frechet
propagator derivatives are mapped to coefficient gradients by the chain rule.
Constrained SLSQP uses analytic amplitude-constraint gradients and multiple
smooth starting pulses. Training averages over a structured amplitude/detuning
grid plus fixed training stress scenarios; an optional tail weight and explicit
leakage/smoothness terms are adaptations, not the paper's exact objective.
The challenge's X gate, six-state dissipative fidelity, T1/Tphi inputs and 0.2 ns
linear sample interpolation are retained. We do not copy the paper's 112/128 ns
X_pi/2 pulse coefficients into a different device or gate duration.
After all methods finish, common Monte Carlo selection samples rank the completed
candidate waveforms by the configured fidelity statistic, subject to nominal,
hardware and sampled-leakage limits. GRAPE does not receive Monte Carlo samples
during training. Independent validation assesses the frozen winner and never
changes it. If none is eligible, the best diagnostic candidate is exported with
quality FAIL. Training does not guarantee improvement or global convergence.
GRAPE settings: `{config.get('grape', {})}`.
GRAPE acceptance: `{grape.get('accepted', False)}`.

## Physical model

`H = Delta*n + alpha*n*(n-1)/2 + I*(a+a.dag())/2 - i*Q*(a-a.dag())/2`.
Times are ns; MHz is converted to rad/ns with `2*pi*1e-3`. Detuning has the PDF's
positive sign. Device settings: `{config['device']}`.

The PDF has inconsistent collapse factors on pages 2 and 5. This code follows
the detailed page-5 equations: `sqrt(1/T1_j)` for each separate relaxation transition,
and `sqrt(2/Tphi_j)` for each excited-level projector. In particular rho01 decays
at `1/(2*T1_1) + 1/Tphi_1`. This modeling choice is retained explicitly.

QuTiP propagates the full Lindblad channel once per device scenario. Fidelity is
the mean of `<X psi|rho_out|X psi>` for the six Pauli eigenstates
`|0>, |1>, |+>, |->, |+i>, |-i>`. Leakage is the mean final level-2 population.
The six-state average equals the qubit Haar average for this linear channel up
to integration error. Leaked output is never renormalized. No free virtual-Z
correction is fitted.

## Gaussian/DRAG reference representation and pulse limits

`g(t)=exp(-(t-T/2)^2/(2*sigma^2))*sin(pi*t/T)^2`, normalized to pi area before
amplitude scaling; `Q=-beta*dI/dt/alpha_nominal`. The derivative is analytic.
The derivative quadrature follows arXiv:1008.2554 Eq. (14). Its anharmonicity
Delta maps to our alpha, and its dimensionless alpha maps to our beta.
The challenge's lambda=sqrt(2) gives reference beta=lambda^2/4=0.5; this is
used for the DRAG baseline, while the optimizer fits beta. We retain the smooth
taper for zero I/Q endpoints. This is an adaptation of the fixed-frequency
control, not the full cubic/chirped correction of Eq. (13).
The paper's phase-qubit Ohmic environment is not used: noise is determined
solely by the input T1/Tphi values and their manufacturing variations.
A centered linear phase ramp is encoded in I/Q. The waveform is never retuned
to the perturbed device during robustness evaluation.

Duration Tg = {config["gate_duration_ns"]} ns is a fixed user input, validated
on the 0.2 ns grid in [10,60] ns. The optimizer never changes it. Both endpoints of I and
Q are zero. Linear interpolation of exported samples prevents intersample
amplitude overshoot. The interpolant is continuous, with slope changes at sample
knots; the PDF supplies no numerical slew/bandwidth constraint. Peak sample slew
is reported. Optimization headroom is {config['amplitude_headroom_scale']}; the
final selection additionally checks the full configured amplitude range.

Gaussian/DRAG reference parameters (not a definition of other families): `{p}`.
Reference Gaussian sigma: **{p['duration_ns']*p['sigma_fraction']:.8f} ns**.
Peak amplitude / 2 pi: **{final['hardware']['peak_amplitude_mhz']:.6f} MHz**.
Peak sample slew: **{final['hardware']['max_slew_rad_ns2']:.6g} rad/ns^2**.

## Gaussian reference envelope A and B

For the analytic seed pre-phase envelope, `I0=A*(exp(...)-B)*sin(pi*t/T)^2`, B=0
and A includes the optimized amplitude scale and sampled area normalization.
These coefficients apply to the Gaussian reference only. For Hermite DRAG, Wah-Wah, BB1 or GRAPE use the exact selected pulse parameters above; its amplitude and shape can differ.
For the paper's Eq. (12) reference, `Ipi=A*(exp(...)-B)`,
`B=exp(-T^2/(8*sigma^2))` and
`A=area/[sigma*sqrt(2*pi)*erf(T/(2*sqrt(2)*sigma))-T*B]`.
If the offset is written outside A, its value is `B_offset=A*B` rad/ns.
The same-sigma reference uses the optimized sigma and amplitude*pi area;
the literal paper reference uses sigma=T/2 and pi area. Neither reference
replaces the exported tapered pulse. A is not necessarily the final I/Q peak.

{chr(10).join(coefficient_table)}

## Optional NSGA-III search

Enabled for this run: {config.get("nsga_enabled", True)}. The following settings apply only when enabled.

pymoo implements NSGA-III with normalized reference-direction survival, bounded
SBX crossover (probability 0.9, eta 15), polynomial mutation (eta 20, probability
1/d per variable), bound repair and duplicate elimination. Three objectives are
minimized separately: mean stress-case gate infidelity, worst stress-case gate
infidelity, and worst stress-case leakage. Duration and all nominal device
parameters are fixed inputs. Only sigma/Tg, beta, amplitude scale and I/Q phase
ramp are optimized. The phase ramp changes the waveform, not the input detuning. Strict nominal quality and
hardware limits are constraints; candidates failing nominal quality skip the
remaining stress simulations and are not eligible for final selection.

Das-Dennis reference partitions: {config['reference_partitions']}.
Population: {config['population_size']}; generations per search: {config['generations']}.
Independent search seeds: {config['search_seeds']}. Each seed starts with a Latin
hypercube and the same analytic Gaussian/DRAG seed pulses. The cache is shared,
but the later search is not initialized from the earlier search's winners.

The 10 fixed training scenarios cover nominal, individual PDF uncertainty
limits, reduced coherence and two joint stress cases. Manufacturing ranges are
used in final Monte Carlo selection and validation. Training loss weights
`{config['selection_weights']}` shortlist candidates; they do not determine the
final winner. These are engineering preferences, not the PDF's scoring formula.

{chr(10).join(search_table)}

Hypervolume uses a fixed reference `[0.02, 0.02, 0.0001]` in objective units.
It is computed from each search's feasible archive. The recent gain is a
convergence diagnostic, not an early-stopping rule or global-convergence proof.
Changing from NSGA-II to NSGA-III alone does not guarantee superior accuracy.

## Manufacturing variation and selection

Uniform, independent quasi-static variations: `{config['manufacturing_ranges']}`.
Amplitude varies around scale 1. Detuning errors are added to the input
Delta/(2*pi) = {config["device"]["detuning_mhz"]} MHz; anharmonicity shifts are
added to the input alpha/(2*pi) = {config["device"]["alpha_mhz"]} MHz.
The four lifetimes vary independently in [{config['coherence_min']},1] times nominal.
The PDF gives uncertainty ranges but not probability laws or a quantified
coherence degradation range; these distribution and lifetime assumptions remain explicit.

Up to {config['mc_selection_candidates']} search finalists, plus baseline and GRAPE comparisons, see the same
{config['mc_selection_samples']} manufactured-device samples, seed
{config['mc_selection_seed']}. The winner maximizes **{config['mc_selection_criterion']}**;
for p05_fidelity this favors performance toward the weaker end of the device range.
Nominal quality, sampled leakage, and amplitude-range limits determine eligibility.
Selected candidate index (zero-based): {selection['candidate']}; selection score:
**{selection['score']:.8f}**. This is best among the tested shortlist, not among
all possible pulses. Selection-set confidence intervals are descriptive because
the same observations were used to select the pulse.

## Independent validation and stricter targets

The frozen pulse is tested on {mc['n_samples']} independent samples using seed
{config['mc_seed']}. All comparison pulses receive identical validation
devices. A separate 45-case public grid covers the PDF's amplitude, detuning
and anharmonicity offsets around the supplied device inputs, with nominal
coherence. This matches the absolute PDF grid only at the PDF nominal inputs.
Primitive baselines use the fixed requested Tg. BB1 and CORPSE comparisons use longer explicitly reported durations and cannot win fixed-Tg selection.

Engineering targets: `{quality}`. These are stricter, configurable choices;
the PDF's nominal minimum remains F>=0.990 and leakage<=0.020.
PASS requires strict nominal fidelity/leakage, sampled validation/public leakage
below the target, the validation P05 lower confidence bound above its target,
and the numerical consistency checks. A failed run retains diagnostic outputs
but does not claim an accepted design.

The P05 confidence bound is a distribution-free, one-sided 95% order-statistic
bound: choose k with `P(Binomial(n,0.05) < k) <= 0.05`, then use the kth smallest
fidelity. If n is too small, the bound is zero. This is confidence in the
distribution's 5th percentile under iid draws, not a guarantee for every device.

{chr(10).join(table)}

{comparison} Selection remains frozen after validation; no new winner is chosen
using validation data. Near-tied selection scores are subject to sampling error.
Mean fidelity 95% Student-t CI: `{mc['mean_fidelity_ci95']}`.
Lowest sampled fidelity: {mc['worst_sample_fidelity']:.8f}; worst sampled leakage:
{mc['worst_sample_leakage']:.3e}. Sampled extremes are not guaranteed worst cases.
The pass-fraction diagnostic in summary.json uses the original PDF thresholds.

## Numerical checks and limitations

Default solver atol=1e-10, rtol=1e-8, max_step=dt/2; strict checks use atol=1e-12,
rtol=1e-10, max_step=dt/4. The sample grid stays at the PDF-required 0.2 ns.
Traces, positivity and metric bounds are checked on the six output states.
The exported CSV is reloaded and compared to the simulated samples. Nominal
and worst sampled cases are recomputed strictly, with allowed absolute metric
difference {quality['solver_metric_tolerance']}.
Observed differences: `{manifest['numerical_checks']}`.

This finite-budget result is not a global optimum, a hardware certification,
or proof that NSGA-III dominates another algorithm. It omits higher levels,
drive-chain filtering, correlated fabrication errors and time-dependent colored
noise. Validation uncertainty does not capture those modeling omissions.

## Reproduce and inspect

`python3 run_all.py --config {output.as_posix()}/config.json --output results/reproduction_nsga3`

Use a new directory per run. summary.json records source hashes, dependency
versions, configuration and measured metrics. pulse.csv contains time_ns,
omega_i_rad_ns and omega_q_rad_ns. Monte Carlo ranking, device samples, full
archives and per-seed histories are saved alongside it.

![Optimization and validation](optimization_dashboard.png)

![Manufacturing selection](manufacturing_selection.png)

![Six-state dynamics](six_state_dynamics.png)

## Method sources and library roles

- [pymoo NSGA-III](https://pymoo.org/algorithms/moo/nsga3.html): optimizer and reference-direction survival.
- [QuTiP propagators](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-propagator.html): Lindblad channel integration.
- [Fidelity of Single Qubit Maps](https://arxiv.org/abs/quant-ph/0201106): few-state fidelity evaluation.
- NumPy/SciPy: arrays, sampling and statistical bounds; Matplotlib: static plots.
- scqubits supports the separate original device diagnostics, not the PDF's fixed device parameters.
"""
    (output / "report.md").write_text(report)

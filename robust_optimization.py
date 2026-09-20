"""Compare baselines -> optional NSGA-III -> optional GRAPE -> final Monte Carlo.

Usage: python robust_optimization.py --config challenge_config.json
Outputs include exact pulse samples, a full evaluation archive, configuration,
nominal/public/Monte Carlo metrics, convergence/Pareto plots and a short report.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t, binom
from qutip.solver.integrator.integrator import IntegratorException

from pulse_shapes import PAPER_DRAG_BETA
from pulse_coefficients import pulse_coefficients
from pulse_methods import reference_pulses, analytic_spec, fourier_spec, parameter_rows
from nsga3 import minimize as nsga_minimize, nondominated_fronts
from challenge_model import (Device, PulseParameters, Scenario, evaluate_waveform,
                             hardware_metrics, make_pulse, monte_carlo_scenarios,
                             public_scenarios, training_scenarios)

PARAMETER_NAMES = ("sigma_fraction", "beta", "amplitude", "phase_ramp_mhz")
OBJECTIVE_NAMES = ("training_mean_infidelity", "training_worst_infidelity", "training_worst_leakage")


class TrainingEvaluator:
    """Cache exact quantized parameters; reuse fixed scenarios for every candidate."""

    def __init__(self, config):
        self.device = Device(**config["device"])
        self.duration_ns = config["gate_duration_ns"]
        self.scenarios = training_scenarios(config["coherence_min"])
        self.headroom = config["amplitude_headroom_scale"]
        self.weights = config["selection_weights"]
        self.quality = config["quality_targets"]
        self.cache = {}
        self.simulations = 0

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (4,) or not np.all(np.isfinite(x)):
            raise ValueError("optimize four pulse controls; Tg is a fixed input")
        p = PulseParameters(self.duration_ns, **dict(zip(PARAMETER_NAMES, x)))
        key = tuple(p.as_array())
        if key in self.cache:
            return self.cache[key]
        pulse = make_pulse(p, self.device)
        record = self.evaluate_samples(pulse, asdict(p))
        self.cache[key] = record
        return record

    def evaluate_samples(self, pulse, parameters):
        """Same objective and constraints for arbitrary reference I/Q samples."""
        hardware = hardware_metrics(*pulse, self.device)
        cv = max(0.0, self.headroom * hardware["peak_amplitude_mhz"]
                 / self.device.amplitude_limit_mhz - 1)
        record = {"parameters": parameters, "hardware": hardware, "violation": 1 + cv if cv > 0 else 0.0}
        if cv > 0:
            record.update(objectives=[1.0, 1.0, 1.0], selection_loss=10 + cv,
                          nominal=None, training=None)
        else:
            nominal = evaluate_waveform(*pulse, self.device, self.scenarios[0])
            self.simulations += 1
            nominal_cv = (max(0., (self.quality["nominal_fidelity_min"] - nominal["fidelity"])
                              / (1 - self.quality["nominal_fidelity_min"]))
                          + max(0., nominal["leakage"] / self.quality["leakage_max"] - 1))
            if nominal_cv > 0:
                # Keep hardware-feasible candidates ahead of all hardware failures.
                # Raw nominal violations can greatly exceed tiny amplitude violations.
                record.update(violation=nominal_cv/(1+nominal_cv), nominal=nominal, training=None,
                              objectives=[nominal["infidelity"], nominal["infidelity"],
                                          nominal["leakage"]],
                              selection_loss=10 + nominal_cv)
                return record
            metrics = [nominal] + [evaluate_waveform(*pulse, self.device, s) for s in self.scenarios[1:]]
            self.simulations += len(metrics) - 1
            errors = np.array([m["infidelity"] for m in metrics])
            leaks = np.array([m["leakage"] for m in metrics])
            summary = {"mean_infidelity": float(errors.mean()),
                       "worst_infidelity": float(errors.max()),
                       "worst_leakage": float(leaks.max())}
            record.update(objectives=[summary["mean_infidelity"], summary["worst_infidelity"],
                                      summary["worst_leakage"]],
                          selection_loss=sum(self.weights[k] * v for k, v in summary.items()),
                          nominal=metrics[0], training=summary)
        return record

    def objective(self, x):
        record = self.evaluate(x)
        return record["objectives"], record["violation"]

    def scalar(self, x):
        return self.evaluate(x)["selection_loss"]


def passes_nominal(record, quality=None):
    quality = quality or {"nominal_fidelity_min": .999, "leakage_max": .0001}
    return (record["violation"] == 0 and record["nominal"] is not None
            and record["nominal"]["fidelity"] >= quality["nominal_fidelity_min"]
            and record["nominal"]["leakage"] <= quality["leakage_max"])


def lower_tail_confidence_bound(fidelities, quantile=.05, confidence=.95):
    """Distribution-free one-sided lower bound from a binomial order statistic.

    For continuous iid samples, P(X_(k) <= true q-quantile) >= confidence.
    A sample too small for a nontrivial bound returns the physical bound zero.
    """
    values = np.sort(np.asarray(fidelities, dtype=float))
    if not len(values) or not 0 < quantile < 1 or not 0 < confidence < 1:
        raise ValueError("invalid sample, quantile or confidence")
    k = int(binom.ppf(1-confidence, len(values), quantile))
    return float(values[k-1]) if k >= 1 else 0.0


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_trials(rows):
    """Empirical tails; mean CI is a Student-t approximation, yield CI is Wilson."""
    fidelity = np.array([r["fidelity"] for r in rows])
    leakage = np.array([r["leakage"] for r in rows])
    n = len(rows)
    se = float(fidelity.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    half = float(student_t.ppf(0.975, n - 1) * se) if n > 1 else 0.0
    successes = (fidelity >= .990) & (leakage <= .020)
    rate = float(successes.mean())
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (rate + z*z/(2*n)) / denominator
    radius = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / denominator
    return {"n_samples": n, "mean_fidelity": float(fidelity.mean()),
            "std_fidelity": float(fidelity.std(ddof=1)) if n > 1 else 0.0,
            "mean_fidelity_ci95": [max(0., float(fidelity.mean()-half)),
                                    min(1., float(fidelity.mean()+half))],
            "p05_fidelity": float(np.quantile(fidelity, .05)),
            "p05_fidelity_lcb95": lower_tail_confidence_bound(fidelity),
            "worst_sample_fidelity": float(fidelity.min()),
            "mean_leakage": float(leakage.mean()),
            "p95_leakage": float(np.quantile(leakage, .95)),
            "worst_sample_leakage": float(leakage.max()),
            "threshold_pass_fraction": rate,
            "threshold_pass_fraction_wilson_ci95": [max(0., center-radius), min(1., center+radius)]}


def evaluate_set(pulse, scenarios, device, label):
    rows = []
    for i, scenario in enumerate(scenarios):
        m = evaluate_waveform(*pulse, device, scenario)
        rows.append({"sample": i, **asdict(scenario),
                     "actual_detuning_mhz": device.detuning_mhz + scenario.detuning_mhz,
                     "actual_alpha_mhz": device.alpha_mhz + scenario.alpha_shift_mhz,
                     **{k: m[k] for k in ["fidelity", "infidelity", "leakage",
                                          "max_trace_error", "min_output_eigenvalue"]}})
        if (i + 1) % 2000 == 0:
            print(f"  {label}: {i+1}/{len(scenarios)} samples", flush=True)
    return rows


def validate_config(config):
    from pulse_methods import settings as method_settings
    method_settings(config)
    bounds = np.asarray(config["bounds"], dtype=float)
    if (bounds.shape != (4, 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 1] <= bounds[:, 0])
            or bounds[0, 0] <= 0):
        raise ValueError("provide four increasing bounds: sigma_fraction, beta, amplitude, phase_ramp_mhz; Tg is fixed")
    device = Device(**config["device"])
    if (not np.all(np.isfinite(list(asdict(device).values()))) or device.alpha_mhz >= 0
            or min(device.t1_1_ns, device.t1_2_ns, device.tphi_1_ns,
                   device.tphi_2_ns, device.amplitude_limit_mhz) <= 0
            or not np.isclose(device.dt_ns, .2, atol=1e-12, rtol=0)):
        raise ValueError("require negative alpha, positive physical parameters and PDF dt=0.2 ns")
    from grape import settings
    g = settings(config)
    if (g['iterations'] < 1 or g['samples'] < 2 or g['substeps'] < 1
            or not 0 < g['tail_fraction'] <= 1
            or not np.all(np.isfinite([g['leakage_weight'], g['smoothness_weight'], g['restart_scale'], g['ftol']]))
            or min(g['leakage_weight'], g['smoothness_weight']) < 0
            or g['harmonics'] < 1 or g['restarts'] < 1 or g['restart_scale'] < 0
            or g['amplitude_grid_points'] < 2 or g['detuning_grid_points'] < 2
            or not 0 <= g['tail_weight'] <= 1 or g['ftol'] <= 0
            or g['seed'] in (config['mc_seed'], config['mc_selection_seed'])):
        raise ValueError('invalid GRAPE settings; training seed must differ from MC selection/validation')
    duration = config["gate_duration_ns"]
    if (not np.isfinite(duration) or not 10 <= duration <= 60
            or not np.isclose(round(duration/device.dt_ns)*device.dt_ns, duration,
                              atol=1e-10, rtol=0)):
        raise ValueError("fixed Tg must be 10-60 ns and an exact multiple of 0.2 ns")
    if (config["population_size"] < 4 or config["population_size"] % 2
            or config["generations"] < 1 or config["monte_carlo_samples"] < 2
            or not 0 < config["coherence_min"] <= 1
            or config["amplitude_headroom_scale"] < 1):
        raise ValueError("invalid optimization budget or robustness configuration")
    expected = {"mean_infidelity", "worst_infidelity", "worst_leakage"}
    weights = config["selection_weights"]
    if (set(weights) != expected or not all(np.isfinite(w) and w >= 0 for w in weights.values())
            or weights["mean_infidelity"] <= 0):
        raise ValueError("selection_weights must be finite nonnegative with positive mean error weight")
    if (config["mc_selection_samples"] < 2 or config["mc_selection_candidates"] < 1
            or config["mc_selection_seed"] == config["mc_seed"]
            or config["mc_selection_criterion"] not in
            {"p05_fidelity", "mean_fidelity", "worst_sample_fidelity"}):
        raise ValueError("invalid MC selection settings; selection and validation seeds must differ")
    from math import comb
    if (config["reference_partitions"] < 1
            or config["population_size"] < comb(config["reference_partitions"]+2, 2)
            or not config["search_seeds"] or len(set(config["search_seeds"])) != len(config["search_seeds"])):
        raise ValueError("require distinct search seeds and enough population for reference directions")
    q = config["quality_targets"]
    if (not .990 <= q["nominal_fidelity_min"] < 1 or not 0 < q["leakage_max"] <= .020
            or not 0 < q["p05_validation_lcb_min"] < 1
            or not 0 < q["solver_metric_tolerance"] <= 2e-5):
        raise ValueError("quality targets must be at least as strict as the PDF minimums")
    ranges = config['manufacturing_ranges']
    if (not all(np.isfinite(v) and v >= 0 for v in ranges.values())
            or ranges['amplitude_fraction'] >= 1):
        raise ValueError('invalid manufacturing ranges')


def candidate_pulse(record, device):
    """Preserve sampled GRAPE controls; analytic parameters only describe its seed."""
    if 'waveform' in record:
        return np.asarray(record['waveform'], dtype=float)
    return np.asarray(make_pulse(PulseParameters(**record['parameters']), device))


def select_robust_candidate(candidates, config, device, output, *, diagnostic_fallback=False):
    """Compare fixed pulses on common perturbed devices, not the luckiest device.

    Shortlist by optimizer training loss, then rank by the requested Monte Carlo
    fidelity statistic. Maximum sampled leakage and amplitude remain constraints.
    All candidates see identical draws. Independent validation cannot change this choice.
    """
    shortlist, seen = [], set()
    for candidate in sorted(candidates, key=lambda r: r["selection_loss"]):
        key = candidate_pulse(candidate, device).tobytes()
        if key not in seen:
            shortlist.append(candidate)
            seen.add(key)
        if len(shortlist) == config["mc_selection_candidates"]:
            break
    scenarios = monte_carlo_scenarios(config["mc_selection_samples"], config["mc_selection_seed"],
                                      config["coherence_min"], **config["manufacturing_ranges"])
    criterion = config["mc_selection_criterion"]
    write_json(output / "mc_selection_scenarios.json", [asdict(s) for s in scenarios])
    rankings = []
    for index, candidate in enumerate(shortlist):
        pulse = candidate_pulse(candidate, device)
        rows = evaluate_set(pulse, scenarios, device, f"candidate {index+1}")
        stats = summarize_trials(rows)
        max_amplitude = candidate["hardware"]["peak_amplitude_mhz"] * (
            1 + config["manufacturing_ranges"]["amplitude_fraction"])
        reasons = []
        if not np.isclose(pulse[0,-1], config["gate_duration_ns"]):
            reasons.append("comparison only: actual duration exceeds requested gate duration")
        nominal = candidate.get('nominal')
        if nominal is None:
            reasons.append('nominal qualification skipped because training hardware headroom failed')
        elif nominal['fidelity'] < config['quality_targets']['nominal_fidelity_min']:
            reasons.append('nominal fidelity below target')
        if nominal is not None and nominal['leakage'] > config['quality_targets']['leakage_max']:
            reasons.append('nominal leakage above limit')
        if stats['worst_sample_leakage'] > config['quality_targets']['leakage_max']:
            reasons.append('selection MC leakage above limit')
        if max_amplitude > device.amplitude_limit_mhz + 1e-10:
            reasons.append('amplitude exceeds hardware limit with manufacturing headroom')
        if candidate['violation'] != 0 and not reasons:
            reasons.append('training hardware/quality constraints failed')
        feasible = not reasons
        rankings.append({"candidate": index, "method": candidate.get("method", "NSGA-III"), "parameters": candidate["parameters"],
                         "training_loss": candidate["selection_loss"], "statistics": stats,
                         "waveform_sha256": hashlib.sha256(pulse.tobytes()).hexdigest(),
                         "rejection_reasons": reasons,
                         "selection_feasible": feasible, "score": stats[criterion]})
        write_csv(output / f"candidate_{index:02d}_mc_selection.csv", rows)
        print(f"  candidate {index+1}/{len(shortlist)}: {criterion}="
              f"{stats[criterion]:.8f}, eligible={feasible}", flush=True)
    write_json(output / "mc_selection_ranking.json", rankings)
    eligible = [r for r in rankings if r["selection_feasible"]]
    if not eligible and not diagnostic_fallback:
        raise OptimizationFailure("No shortlisted pulse passes the nominal, sampled-leakage and amplitude "
                           "constraints. Selection diagnostics saved; no final pulse exported.")
    duration_matched = [r for r in rankings if np.isclose(r["parameters"]["duration_ns"], config["gate_duration_ns"])]
    if not eligible and not duration_matched:
        raise OptimizationFailure("No pulse matches requested gate duration")
    winner = max(eligible or duration_matched, key=lambda r: (r["score"], -r["training_loss"]))
    selected = dict(shortlist[winner["candidate"]])
    selected["mc_selection"] = winner
    return selected


def print_final_summary(manifest, output=None):
    """Print all final numerical results and PDF-required report content."""
    from terminal_report import print_results
    print_results(manifest, output)


class OptimizationFailure(RuntimeError):
    """A completed search did not find an acceptable design, not a software crash."""


def report_search_failure(candidates, config, output):
    measured = [r for r in candidates if r["nominal"] is not None]
    closest = min(measured or candidates, key=lambda r: r["violation"])
    diagnostic = {"status": "NO_ACCEPTABLE_PULSE", "config": config,
                  "candidates_tested": len(candidates), "nominal_simulations": len(measured),
                  "closest_candidate": closest,
                  "highest_fidelity_candidate": max(measured, key=lambda r: r["nominal"]["fidelity"]) if measured else None,
                  "lowest_leakage_candidate": min(measured, key=lambda r: r["nominal"]["leakage"]) if measured else None}
    write_json(Path(output) / "failure_summary.json", diagnostic)
    print("\nNO ACCEPTABLE PULSE FOUND (search result, not a dependency error)", flush=True)
    print(f"  Candidates tested: {len(candidates)}; nominally simulated: {len(measured)}")
    print(f"  Fixed device inputs: {config['device']}; Tg={config['gate_duration_ns']} ns")
    print(f"  Closest tested candidate parameters: {closest['parameters']}")
    print(f"  Peak amplitude including headroom: {closest['hardware']['peak_amplitude_mhz']*config['amplitude_headroom_scale']:.8f} MHz "
          f"(limit {config['device']['amplitude_limit_mhz']} MHz)")
    if closest["nominal"]:
        m = closest["nominal"]
        q = config['quality_targets']
        print(f"  Closest candidate fidelity: {100*m['fidelity']:.6f}% (required >= {100*q['nominal_fidelity_min']:.6f}%)")
        print(f"  Closest candidate leakage: {100*m['leakage']:.6f}% (required <= {100*q['leakage_max']:.6f}%)")
        print(f"  Highest tested fidelity: {100*diagnostic['highest_fidelity_candidate']['nominal']['fidelity']:.6f}%")
        print(f"  Lowest tested leakage: {100*diagnostic['lowest_leakage_candidate']['nominal']['leakage']:.6f}% (may be a different candidate)")
    print("  Your inputs and quality targets were preserved. No accepted final pulse was exported.")
    print("  This finite search does not prove impossibility. Review the search bounds/budget,")
    print("  or choose a different Tg/device input if your design allows it.")
    print(f"  Diagnostics: {(Path(output)/'failure_summary.json').resolve()}", flush=True)


def run(config, output):
    from grape import settings
    config['grape'] = settings(config)
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new --output")
    write_json(output / "config.json", config)
    evaluator = TrainingEvaluator(config)
    device = evaluator.device
    print("Fixed physical inputs (not optimized):", flush=True)
    print(f"  detuning = {device.detuning_mhz:g} MHz; alpha = {device.alpha_mhz:g} MHz")
    print(f"  T1(1) = {device.t1_1_ns:g} ns; T1(2) = {device.t1_2_ns:g} ns")
    print(f"  Tphi(1) = {device.tphi_1_ns:g} ns; Tphi(2) = {device.tphi_2_ns:g} ns")
    print(f"  Tg = {config['gate_duration_ns']:g} ns", flush=True)
    bounds = np.asarray(config["bounds"])
    write_json(output / "training_scenarios.json", [asdict(s) for s in evaluator.scenarios])

    # All reference families share device inputs, total Tg and sample spacing.
    references = reference_pulses(config, device)
    baseline_records = []
    for key, reference in references.items():
        print(f"  Building and evaluating {reference['method']} reference", flush=True)
        record = evaluator.evaluate_samples(candidate_pulse(reference, device), reference['parameters'])
        reference.update(record)
        baseline_records.append(reference)
    all_history = []
    nsga_enabled = config.get('nsga_enabled', True)
    print(f"Stage 1/4: baseline comparison; NSGA-III {'enabled' if nsga_enabled else 'disabled'}", flush=True)
    if nsga_enabled:
        seeds = [np.array([sigma, beta, 1., 0.])
                 for sigma in [.25, .3, .35] for beta in [0., .4, PAPER_DRAG_BETA, 1.]]
        seeds = [s for s in seeds if np.all(s >= bounds[:, 0]) and np.all(s <= bounds[:, 1])]
        for search_seed in config['search_seeds']:
            def progress(h):
                if h['generation'] == 1 or h['generation'] == config['generations'] or h['generation'] % 8 == 0:
                    print(f"  seed {search_seed}, generation {h['generation']}/{config['generations']}: "
                          f"{h['feasible_count']} feasible", flush=True)
            result = nsga_minimize(
                evaluator.objective, bounds, population_size=config['population_size'],
                generations=config['generations'], seed=search_seed,
                reference_partitions=config['reference_partitions'], n_objectives=3,
                hv_reference=[.02, .02, .0001],
                initial_points=seeds[:config['population_size']] or None, callback=progress)
            all_history.extend(result.history)
            write_json(output / f'search_seed_{search_seed}.json', result.history)
        np.savetxt(output / 'reference_directions.csv', result.reference_directions, delimiter=',')
    write_json(output / 'nsga_history.json', all_history)
    records = []
    if nsga_enabled:
        for record in evaluator.cache.values():
            spec = analytic_spec(PulseParameters(**record['parameters']), device, 'NSGA-III')
            records.append(dict(record, method='NSGA-III', pulse_spec=spec))
    baseline_waveforms = {candidate_pulse(r, device).tobytes() for r in baseline_records}
    records = baseline_records + [r for r in records if candidate_pulse(r, device).tobytes() not in baseline_waveforms]
    write_json(output / 'optimization_archive.json', records)
    if nsga_enabled:
        write_json(output / 'nsga_archive.json', records)
    fixed_duration = [r for r in records if np.isclose(r['parameters']['duration_ns'], config['gate_duration_ns'])]
    feasible = [r for r in fixed_duration if passes_nominal(r, config['quality_targets'])]
    pareto = ([feasible[i] for i in nondominated_fronts(np.array([r['objectives'] for r in feasible]))[0]]
              if feasible else [])
    write_json(output / 'pareto.json', pareto)
    seed_record = min(feasible or fixed_duration, key=lambda r: r['selection_loss'])
    seed_pulse = candidate_pulse(seed_record, device)
    nsga_records = [r for r in records if r['method'] == 'NSGA-III']
    nsga_best = min(nsga_records, key=lambda r: r['selection_loss']) if nsga_records else seed_record
    # Limit numerical finalists while always comparing every reference family.
    candidates = sorted(records, key=lambda r: r['selection_loss'])[:config['mc_selection_candidates']]
    for record in baseline_records:
        if record not in candidates:
            candidates.append(record)
    from grape import refine
    g = config['grape']
    grape_result = {'enabled': g['enabled'], 'accepted': False, 'seed_method': seed_record['method']}
    if g['enabled']:
        print('Stage 2/4: GRAPE refinement before any Monte Carlo sampling', flush=True)
        np.savetxt(output / 'pre_grape_pulse.csv', seed_pulse.T, delimiter=',',
                   header='time_ns,omega_i_rad_ns,omega_q_rad_ns', comments='', fmt='%.17g')
        ranges = config['manufacturing_ranges']
        # Fixed stress cases and a deterministic grid: no Monte Carlo before GRAPE.
        training = [Scenario(amplitude_scale=float(a), detuning_mhz=float(d))
                    for a in np.linspace(1-ranges['amplitude_fraction'], 1+ranges['amplitude_fraction'], g['amplitude_grid_points'])
                    for d in np.linspace(-ranges['detuning_mhz'], ranges['detuning_mhz'], g['detuning_grid_points'])]
        training += evaluator.scenarios
        write_json(output / 'grape_training_scenarios.json', [asdict(s) for s in training])
        proposal, history = refine(seed_pulse, device, training, g,
                                  max(config['amplitude_headroom_scale'], 1+ranges['amplitude_fraction']))
        proposal = np.asarray(proposal)
        write_json(output / 'grape_history.json', history)
        np.savetxt(output / 'grape_proposal.csv', proposal.T, delimiter=',',
                   header='time_ns,omega_i_rad_ns,omega_q_rad_ns', comments='', fmt='%.17g')
        nominal = evaluate_waveform(*proposal, device)
        hardware = hardware_metrics(*proposal, device)
        q = config['quality_targets']
        violation = 0. if (hardware['feasible'] and nominal['fidelity'] >= q['nominal_fidelity_min']
                           and nominal['leakage'] <= q['leakage_max']) else 1.
        grape_candidate = dict(parameters=seed_record['parameters'], method='GRAPE',
                               waveform=proposal.tolist(), nominal=nominal, hardware=hardware,
                               violation=violation, selection_loss=seed_record['selection_loss'],
                               pulse_spec=fourier_spec(proposal, history, seed_record['pulse_spec'], device))
        candidates.append(grape_candidate)
        grape_result.update(nominal=nominal, hardware=hardware, history=history,
                            method='Fourier-GRAPE, constrained SLSQP, multistart',
                            representation='Fourier I/Q samples; analytic parameters describe the seed only')
    else:
        print('Stage 2/4: GRAPE disabled; proceeding to final Monte Carlo', flush=True)
    print('Stage 3/4: final Monte Carlo selection of completed method candidates', flush=True)
    selection_config = {**config, 'mc_selection_candidates': len(candidates)}
    best = select_robust_candidate(candidates, selection_config, device, output, diagnostic_fallback=True)
    chosen = PulseParameters(**best['parameters'])
    selected_spec = {**best['pulse_spec'], 'waveform_file': 'pulse.csv',
                     'peak_amplitude_mhz': best['hardware']['peak_amplitude_mhz']}
    write_json(output / 'selected_pulse_parameters.json', selected_spec)
    final_pulse = candidate_pulse(best, device)
    grape_result['accepted'] = best['method'] == 'GRAPE' and best['mc_selection']['selection_feasible']
    grape_result['selected'] = best['method'] == 'GRAPE'
    write_json(output / 'grape_summary.json', grape_result)
    # Freeze selection and export before generating/looking at held-out MC data.
    write_json(output / "selected_pulse.json", {**best, "grape": grape_result, "final_waveform_file": "pulse.csv"})
    np.savetxt(output / "pulse.csv", np.column_stack(final_pulse), delimiter=",",
               header="time_ns,omega_i_rad_ns,omega_q_rad_ns", comments="", fmt="%.17g")
    reloaded = np.loadtxt(output / "pulse.csv", delimiter=",", skiprows=1).T
    np.testing.assert_allclose(reloaded, np.array(final_pulse), rtol=1e-14, atol=1e-14)

    print("Stage 4/4: frozen-pulse public grid and independent Monte Carlo validation", flush=True)
    mc_scenarios = monte_carlo_scenarios(config["monte_carlo_samples"], config["mc_seed"],
                                        config["coherence_min"], **config["manufacturing_ranges"])
    public = public_scenarios()
    stage_records = dict(references)
    if nsga_enabled:
        stage_records['nsga_selected'] = nsga_best
    if g['enabled']:
        stage_records['grape_proposal'] = grape_candidate
    stage_records['robust_selected'] = best
    variants = {name: PulseParameters(**record['parameters']) for name, record in stage_records.items()}
    stage_pulses = {name: candidate_pulse(record, device) for name, record in stage_records.items()}
    rankings = json.loads((output/'mc_selection_ranking.json').read_text())
    validation_cache = {}
    summaries, mc_rows, public_rows = {}, {}, {}
    for name, parameters in variants.items():
        print(f"  Validating {name} on common public and held-out MC devices", flush=True)
        pulse = stage_pulses[name]
        np.savetxt(output / f"{name}_pulse.csv", np.column_stack(pulse), delimiter=",",
                   header="time_ns,omega_i_rad_ns,omega_q_rad_ns", comments="", fmt="%.17g")
        waveform_key = np.array(pulse).tobytes()
        if waveform_key not in validation_cache:
            validation_cache[waveform_key] = (evaluate_waveform(*pulse, device),
                evaluate_set(pulse, public, device, name + ' public'),
                evaluate_set(pulse, mc_scenarios, device, name + ' MC'))
        nominal, public_rows[name], mc_rows[name] = validation_cache[waveform_key]
        write_csv(output / f"{name}_public.csv", public_rows[name])
        write_csv(output / f"{name}_monte_carlo.csv", mc_rows[name])
        public_f = [r["fidelity"] for r in public_rows[name]]
        public_l = [r["leakage"] for r in public_rows[name]]
        matches = [r for r in rankings if r.get('waveform_sha256') == hashlib.sha256(np.asarray(pulse).tobytes()).hexdigest()]
        selection = matches[0] if matches else None
        summaries[name] = {"parameters": asdict(parameters), "nominal": nominal,
                           "pulse_spec": stage_records[name]['pulse_spec'],
                           "selection": selection,
                           "hardware": hardware_metrics(*pulse, device),
                           "monte_carlo": summarize_trials(mc_rows[name]),
                           "public": {"n_samples": len(public),
                                      "mean_fidelity": float(np.mean(public_f)),
                                      "worst_fidelity": float(np.min(public_f)),
                                      "worst_leakage": float(np.max(public_l))}}


    # Independent tighter solves on the same exported waveform; no resampling.
    worst_index = int(np.argmin([r["fidelity"] for r in mc_rows["robust_selected"]]))
    strict_nominal = evaluate_waveform(*reloaded, device, strict=True)
    strict_worst = evaluate_waveform(*reloaded, device, mc_scenarios[worst_index], strict=True)
    numerical_checks = {
        "nominal_fidelity_difference": abs(strict_nominal["fidelity"] - summaries["robust_selected"]["nominal"]["fidelity"]),
        "nominal_leakage_difference": abs(strict_nominal["leakage"] - summaries["robust_selected"]["nominal"]["leakage"]),
        "worst_mc_fidelity_difference": abs(strict_worst["fidelity"] - mc_rows["robust_selected"][worst_index]["fidelity"]),
        "worst_mc_leakage_difference": abs(strict_worst["leakage"] - mc_rows["robust_selected"][worst_index]["leakage"])}
    if max(numerical_checks.values()) > config["quality_targets"]["solver_metric_tolerance"]:
        raise RuntimeError("Solver convergence check failed; inspect results and tighten tolerances")
    q = config["quality_targets"]
    accepted = bool(best["mc_selection"]["selection_feasible"] and passes_nominal(best, q) and strict_nominal["fidelity"] >= q["nominal_fidelity_min"]
                    and strict_nominal["leakage"] <= q["leakage_max"])
    quality_passed = bool(accepted
                          and summaries["robust_selected"]["monte_carlo"]["p05_fidelity_lcb95"] >= q["p05_validation_lcb_min"]
                          and summaries["robust_selected"]["monte_carlo"]["worst_sample_leakage"] <= q["leakage_max"]
                          and summaries["robust_selected"]["public"]["worst_leakage"] <= q["leakage_max"])
    manifest = {"config": config, "variants": summaries, "numerical_checks": numerical_checks,
                "nominal_acceptance_passed": accepted, "quality_passed": quality_passed,
                "optimizer": "pymoo NSGA-III" if nsga_enabled else "Analytic baselines",
                "selected_method": best["method"],
                "selected_pulse_spec": selected_spec,
                "workflow": ["baselines", "optional_nsga", "optional_grape", "mc_selection", "mc_validation"],
                "noise_model": "T1/Tphi Lindblad (challenge PDF page 5)",
                "grape": grape_result,
                "pulse_coefficients": pulse_coefficients(asdict(chosen), device.dt_ns),
                "pulse_coefficients_scope": "Gaussian reference only; selected_pulse_spec defines the actual final pulse",
                "drag_reference": {"source": "arXiv:1008.2554 Eq. (14)",
                                   "lambda": float(np.sqrt(2)), "baseline_beta": PAPER_DRAG_BETA},
                "training_channel_simulations": evaluator.simulations,
                "unique_optimization_candidates": len(evaluator.cache),
                "selection_loss": best["selection_loss"],
                "mc_selection": best["mc_selection"],
                "versions": {n: importlib.metadata.version(n) for n in
                             ["numpy", "scipy", "matplotlib", "qutip", "scqubits", "pymoo"]},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")}}
    write_json(output / "summary.json", manifest)
    parameter_text = '# Selected pulse parameters\n\n' + '\n'.join(
        f'- **{label}:** {value}' for label, value in parameter_rows(selected_spec))
    (output / 'selected_pulse_parameters.md').write_text(parameter_text+'\n')
    from challenge_report import save_plots, write_report
    save_plots(output, manifest, pareto, all_history, variants, mc_rows, public_rows, device)
    write_report(output, manifest)
    from stage_report import save_stage_comparisons
    save_stage_comparisons(output, manifest, stage_pulses, public_rows, all_history)
    from pulse_diagnostics import save_pulse_diagnostics
    save_pulse_diagnostics(output, manifest, stage_pulses)
    print_final_summary(manifest, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("challenge_config.json"))
    parser.add_argument("--output", type=Path, default=Path("results/nsga3"))
    parser.add_argument("--quick", action="store_true", help="small end-to-end smoke run, not convergence evidence")
    from optimization_inputs import add_input_arguments, apply_input_arguments, prompt_input_arguments
    add_input_arguments(parser)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        prompt_input_arguments(config, args)
    except EOFError:
        parser.error("Input ended before completion; supply seven answers or use --non-interactive.")
    except KeyboardInterrupt:
        raise SystemExit(130)
    apply_input_arguments(config, args)
    if args.quick:
        config['grape'] = {**config.get('grape', {}), 'iterations': 3, 'samples': 2, 'restarts': 1, 'amplitude_grid_points': 3, 'detuning_grid_points': 3}
        config.update(population_size=12, generations=3, reference_partitions=2,
                      search_seeds=[7], monte_carlo_samples=128,
                      mc_selection_samples=32, mc_selection_candidates=3)
    try:
        manifest = run(config, args.output)
    except IntegratorException as error:
        diagnostic = {"status": "SOLVER_FAILURE", "config": config, "error": str(error)}
        if args.output.is_dir():
            write_json(args.output / "solver_failure.json", diagnostic)
        print(f"\nNumerical integration failed: {error}", flush=True)
        print(f"Input detuning: {config['device']['detuning_mhz']:g} MHz; "
              f"anharmonicity: {config['device']['alpha_mhz']:g} MHz; "
              f"Tg: {config['gate_duration_ns']:g} ns.")
        print("Check frequency units first: 300000 Hz = 0.3 MHz, not 300000 MHz.")
        print("If units are intentional, the solver settings/model require investigation; no valid final result was produced.")
        raise SystemExit(3)
    except OptimizationFailure as error:
        print(f"\nOptimization stopped: {error}", flush=True)
        raise SystemExit(2)
    if not manifest["quality_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

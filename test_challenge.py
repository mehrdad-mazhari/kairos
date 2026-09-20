"""Physics and optimizer checks independent of the production search results."""

import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
from qutip import Qobj, expect, liouvillian, mesolve, qeye, to_super

from challenge_model import (Device, KETS, MHZ_TO_RAD_NS, PulseParameters, Scenario,
                             TARGET_X, TEST_STATES, channel_metrics, collapse_operators,
                             evaluate_waveform, hamiltonian, hardware_metrics, make_pulse,
                             monte_carlo_scenarios, public_scenarios)
from robust_optimization import summarize_trials, validate_config, select_robust_candidate, lower_tail_confidence_bound
from nsga3 import minimize, nondominated_fronts


class ModelTests(unittest.TestCase):
    def test_x_identity_and_phase_sensitive_fidelity(self):
        ideal = TARGET_X + KETS[2].proj()
        self.assertAlmostEqual(channel_metrics(to_super(ideal))["fidelity"], 1)
        self.assertAlmostEqual(channel_metrics(to_super(qeye(3)))["fidelity"], 1/3)
        # Y flips both basis populations but is not the intended X gate.
        y = -1j*KETS[0]*KETS[1].dag() + 1j*KETS[1]*KETS[0].dag() + KETS[2].proj()
        metric = channel_metrics(to_super(y))
        np.testing.assert_allclose(metric["state_fidelities"][:2], [1, 1])
        self.assertAlmostEqual(metric["fidelity"], 1/3)
        self.assertAlmostEqual(metric["leakage"], 0)

    def test_leakage_is_not_renormalized_away(self):
        swap12 = Qobj([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        metrics = channel_metrics(to_super(swap12))
        self.assertAlmostEqual(metrics["leakage"], .5)
        self.assertLess(metrics["fidelity"], .5)

    def test_pdf_sign_and_collapse_factors(self):
        device = Device()
        h = hamiltonian(np.array([0., 1.]), np.zeros(2), np.zeros(2),
                        device, Scenario(detuning_mhz=3))
        np.testing.assert_allclose(h(0).diag(), np.array([0, 3, 6-220])*MHZ_TO_RAD_NS)
        dissipator = liouvillian(0*KETS[0].proj(), collapse_operators(device))
        coherence = KETS[0]*KETS[1].dag()
        decay = dissipator(coherence).full()[0, 1]
        self.assertAlmostEqual(decay.real, -(1/(2*device.t1_1_ns)+1/device.tphi_1_ns))
        excited2_derivative = dissipator(KETS[2].proj())
        self.assertAlmostEqual(expect(KETS[2].proj(), excited2_derivative), -1/device.t1_2_ns)

    def test_pdf_full_hamiltonian_matrix(self):
        # Independent explicit 3x3 oracle checks Q sign, both couplings, units,
        # uncertainty offsets and the chosen interpolation between samples.
        device = Device(detuning_mhz=1.5, alpha_mhz=-225)
        scenario = Scenario(amplitude_scale=1.03, detuning_mhz=-.4,
                            alpha_shift_mhz=3)
        times = np.array([0., .2, .4])
        pulse_i = np.array([.1, .3, -.2])
        pulse_q = np.array([-.05, .08, .04])
        h = hamiltonian(times, pulse_i, pulse_q, device, scenario)
        delta = 1.1 * MHZ_TO_RAD_NS
        alpha = -222 * MHZ_TO_RAD_NS
        for time in [0., .1, .2, .3, .4]:
            i = np.interp(time, times, pulse_i)*1.03
            q = np.interp(time, times, pulse_q)*1.03
            c = (i-1j*q)/2
            expected = np.array([[0, c, 0],
                                 [c.conjugate(), delta, np.sqrt(2)*c],
                                 [0, np.sqrt(2)*c.conjugate(), 2*delta+alpha]])
            actual = h(time).full()
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
            np.testing.assert_allclose(actual, actual.conj().T, atol=1e-14, rtol=0)

    def test_envelope_coefficients_match_area_and_applied_pulse(self):
        from dataclasses import asdict
        from scipy.integrate import quad
        from pulse_coefficients import pulse_coefficients
        parameters = PulseParameters(16, .3, .5, 1.01, 0.)
        values = pulse_coefficients(asdict(parameters), .2)
        times, i, _ = make_pulse(parameters)
        applied = values['applied_tapered_envelope']
        sigma = 16*.3
        reconstructed = applied['A_rad_per_ns'] * np.exp(-.5*((times-8)/sigma)**2) * np.sin(np.pi*times/16)**2
        np.testing.assert_allclose(i, reconstructed, atol=1e-14, rtol=0)
        self.assertEqual(applied['B_dimensionless'], 0)
        for name in ('shifted_gaussian_same_sigma_reference',
                     'paper_eq12_sigma_T_over_2_pi_area_reference'):
            row = values[name]
            envelope = lambda t: row['A_rad_per_ns']*(np.exp(-.5*((t-8)/row['sigma_ns'])**2)-row['B_dimensionless'])
            self.assertAlmostEqual(envelope(0), 0, places=13)
            self.assertAlmostEqual(envelope(16), 0, places=13)
            self.assertAlmostEqual(quad(envelope, 0, 16)[0], row['area_rad'], places=12)
            self.assertAlmostEqual(row['B_offset_rad_per_ns'], row['A_rad_per_ns']*row['B_dimensionless'])

    def test_pulse_export_geometry_and_frozen_controls(self):
        p = PulseParameters.from_array([23.91, .3, .7, 1.01, 2.0])
        self.assertAlmostEqual(p.duration_ns, 24)
        times, i, q = make_pulse(p)
        np.testing.assert_allclose(np.diff(times), .2)
        np.testing.assert_array_equal([i[0], q[0], i[-1], q[-1]], np.zeros(4))
        self.assertTrue(hardware_metrics(times, i, q)["feasible"])
        dense = np.linspace(0, times[-1], 10_001)
        self.assertLessEqual(np.hypot(np.interp(dense, times, i), np.interp(dense, times, q)).max(),
                             np.hypot(i, q).max()+1e-12)
        h1 = hamiltonian(times, i, q)
        h2 = hamiltonian(times, i, q, scenario=Scenario(alpha_shift_mhz=8))
        # Uncertainty changes the static level-2 energy, never the pulse's DRAG Q.
        np.testing.assert_allclose((h2(10)-h1(10)).full(),
                                   np.diag([0, 0, 8*MHZ_TO_RAD_NS]), atol=1e-14)

    def test_channel_matches_six_independent_master_equations(self):
        device = Device()
        pulse = make_pulse(PulseParameters(20, .3, .5, 1.0, .5), device)
        scenario = Scenario(.98, 2, -5, .9, .95, .85, .92)
        metrics = evaluate_waveform(*pulse, device, scenario, strict=True)
        h = hamiltonian(*pulse, device, scenario)
        f, leakage = [], []
        for state in TEST_STATES:
            target = (TARGET_X*state).proj()
            result = mesolve(h, state, [0, pulse[0][-1]], c_ops=collapse_operators(device, scenario),
                             e_ops=[target, KETS[2].proj()],
                             options={"atol": 1e-11, "rtol": 1e-9, "max_step": .05,
                                      "nsteps": 100_000, "normalize_output": False})
            f.append(result.expect[0][-1])
            leakage.append(result.expect[1][-1])
        np.testing.assert_allclose(metrics["state_fidelities"], f, atol=2e-6, rtol=0)
        np.testing.assert_allclose(metrics["state_leakages"], leakage, atol=2e-6, rtol=0)

    def test_sampling_and_configuration(self):
        samples = monte_carlo_scenarios(100, 25)
        self.assertEqual(samples, monte_carlo_scenarios(100, 25))
        self.assertNotEqual(samples, monte_carlo_scenarios(100, 26))
        self.assertEqual(len(public_scenarios()), 45)
        for s in samples:
            self.assertTrue(.97 <= s.amplitude_scale <= 1.03)
            self.assertTrue(-3 <= s.detuning_mhz <= 3)
            self.assertTrue(-8 <= s.alpha_shift_mhz <= 8)
            self.assertTrue(.8 <= s.tphi_2_scale <= 1)
        config = json.loads(Path(__file__).with_name("challenge_config.json").read_text())
        validate_config(config)
        config["device"]["dt_ns"] = 1
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_fixed_inputs_and_four_control_search(self):
        from robust_optimization import TrainingEvaluator
        config = json.loads(Path(__file__).with_name("challenge_config.json").read_text())
        config["gate_duration_ns"] = 24.0
        config["device"]["detuning_mhz"] = 1.25
        config["device"]["alpha_mhz"] = -230.0
        config["device"]["t1_1_ns"] = 42000.0
        validate_config(config)
        evaluator = TrainingEvaluator(config)
        measured = dict(fidelity=.9999, infidelity=.0001, leakage=.00001)
        with patch("robust_optimization.evaluate_waveform", return_value=measured) as solve:
            record = evaluator.evaluate([.3, .5, 1.0, .2])
        self.assertEqual(record["parameters"]["duration_ns"], 24.0)
        self.assertEqual(len(record["objectives"]), 3)
        self.assertEqual(record["parameters"]["phase_ramp_mhz"], .2)
        for call in solve.call_args_list:
            self.assertEqual(call.args[0][-1], 24.0)
            self.assertEqual(call.args[3].detuning_mhz, 1.25)
            self.assertEqual(call.args[3].t1_1_ns, 42000.0)
        device = evaluator.device
        times = np.array([0., .2, .4])
        h = hamiltonian(times, np.zeros(3), np.zeros(3), device,
                        Scenario(detuning_mhz=.75, alpha_shift_mhz=4))
        np.testing.assert_allclose(h(0).diag(), np.array([0, 2, 4-226])*MHZ_TO_RAD_NS)
        with self.assertRaises(ValueError):
            evaluator.evaluate([24, .3, .5, 1, 0])
        config["gate_duration_ns"] = 24.1
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_hardware_failures_cannot_dominate_nominal_search(self):
        from robust_optimization import TrainingEvaluator
        config = json.loads(Path(__file__).with_name("challenge_config.json").read_text())
        evaluator = TrainingEvaluator(config)
        limit = config['device']['amplitude_limit_mhz']/config['amplitude_headroom_scale']
        with patch('robust_optimization.hardware_metrics', return_value={'peak_amplitude_mhz': limit*(1+1e-8)}):
            over = evaluator.evaluate([.3, .5, 1., 0.])
        with patch('robust_optimization.hardware_metrics', return_value={'peak_amplitude_mhz': limit*.99}), patch('robust_optimization.evaluate_waveform', return_value={'fidelity': .95, 'infidelity': .05, 'leakage': .01}):
            allowed = evaluator.evaluate([.3, .6, 1., 0.])
        self.assertGreater(over['violation'], 1)
        self.assertGreater(allowed['violation'], 0)
        self.assertLess(allowed['violation'], over['violation'])

    def test_grape_analytic_gradient_and_constraints(self):
        from grape import EnsembleGRAPE, DEFAULTS, refine
        times = np.linspace(0, 1., 6)
        controls = np.column_stack([.1*np.sin(np.pi*times), .03*np.sin(2*np.pi*times)])
        options = {**DEFAULTS, 'iterations': 2, 'samples': 2, 'harmonics': 2, 'restarts': 1}
        device = Device()
        scenarios = [Scenario(), Scenario(.98, 2, -4, .9, .9, .9, .9)]
        model = EnsembleGRAPE(times, device, scenarios, options)
        value, gradient = model.value_gradient(controls)
        for index in [(1, 0), (2, 1), (4, 0)]:
            plus, minus = controls.copy(), controls.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            finite = (model.value_gradient(plus)[0]-model.value_gradient(minus)[0])/2e-6
            self.assertAlmostEqual(gradient[index], finite, delta=2e-8)
        with patch('builtins.print'):
            pulse, history = refine((times, controls[:, 0], controls[:, 1]), device, scenarios, options, 1.03)
        self.assertLessEqual(history[-1]['loss'], value)
        np.testing.assert_array_equal([pulse[1][0], pulse[1][-1], pulse[2][0], pulse[2][-1]], [0]*4)
        self.assertLessEqual(np.hypot(pulse[1], pulse[2]).max()*1.03, 80*MHZ_TO_RAD_NS+1e-14)

    def test_grape_propagation_matches_independent_lindblad_solver(self):
        from grape import EnsembleGRAPE, DEFAULTS
        pulse = make_pulse(PulseParameters(16, .3, .5, 1.01, .2))
        controls = np.column_stack(pulse[1:])
        scenario = Scenario(1.02, 2, -5, .85, .9, .95, .8)
        reference = evaluate_waveform(*pulse, Device(), scenario, strict=True)
        target = reference['infidelity']+2*reference['leakage']
        errors = []
        for substeps in (2, 4):
            model = EnsembleGRAPE(pulse[0], Device(), [scenario], {**DEFAULTS, 'substeps': substeps})
            value, gradient = model.scenario_value_gradient(controls, model.models[0])
            errors.append(abs(value-target))
            direction = np.random.default_rng(9).normal(size=controls.shape)
            direction[[0,-1]] = 0
            direction /= np.linalg.norm(direction)
            epsilon = 1e-5
            plus = model.scenario_value_gradient(controls+epsilon*direction, model.models[0])[0]
            minus = model.scenario_value_gradient(controls-epsilon*direction, model.models[0])[0]
            self.assertAlmostEqual(np.sum(gradient*direction), (plus-minus)/(2*epsilon), delta=2e-8)
        self.assertLess(errors[0], 5e-7)
        self.assertLess(errors[1], errors[0]/3)

    def test_fourier_grape_chain_rule_and_amplitude_constraint(self):
        from grape import FourierControls, EnsembleGRAPE, DEFAULTS
        times = np.arange(81)*.2
        basis = FourierControls(times, 5, 80*MHZ_TO_RAD_NS)
        coefficients = np.array([.6, -.1, .03, 0, .01, .05, -.02, .01, 0, 0])
        controls = basis.waveform(coefficients)
        np.testing.assert_allclose(controls[:, 0], controls[::-1, 0], atol=1e-14)
        np.testing.assert_allclose(controls[:, 1], -controls[::-1, 1], atol=1e-14)
        np.testing.assert_array_equal(controls[[0,-1]], np.zeros((2,2)))
        model = EnsembleGRAPE(times, Device(), [Scenario()], DEFAULTS)
        _, sample_gradient = model.value_gradient(controls)
        gradient = basis.gradient(sample_gradient)
        direction = np.random.default_rng(3).normal(size=10)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-6
        plus, minus = coefficients+epsilon*direction, coefficients-epsilon*direction
        finite = (model.value_gradient(basis.waveform(plus))[0]-model.value_gradient(basis.waveform(minus))[0])/(2*epsilon)
        self.assertAlmostEqual(gradient@direction, finite, delta=2e-8)
        limit = 80*MHZ_TO_RAD_NS/1.03
        finite_constraint = (basis.constraints(plus,limit)-basis.constraints(minus,limit))/(2*epsilon)
        np.testing.assert_allclose(basis.constraint_gradient(coefficients,limit)@direction, finite_constraint, atol=2e-9)

    def test_custom_manufacturing_ranges(self):
        samples = monte_carlo_scenarios(100, 25, .95, amplitude_fraction=.01,
                                       detuning_mhz=.5, alpha_shift_mhz=2)
        for s in samples:
            self.assertTrue(.99 <= s.amplitude_scale <= 1.01)
            self.assertTrue(abs(s.detuning_mhz) <= .5)
            self.assertTrue(abs(s.alpha_shift_mhz) <= 2)
            self.assertTrue(.95 <= s.t1_1_scale <= 1)

    def test_mc_selection_favors_stability_over_nominal_and_mean(self):
        config = json.loads(Path(__file__).with_name("challenge_config.json").read_text())
        config.update(mc_selection_samples=20, mc_selection_candidates=2)
        # A has higher nominal and average fidelity, but a weak lower tail.
        config["gate_duration_ns"] = 24.0
        candidates = [dict(parameters=dict(duration_ns=24, sigma_fraction=.3, beta=beta,
                                           amplitude=1, phase_ramp_mhz=0),
                           nominal=dict(fidelity=nominal, leakage=.00001), violation=0,
                           hardware=dict(peak_amplitude_mhz=50), selection_loss=loss)
                      for beta, nominal, loss in [(1, .9999, .001), (.5, .999, .002)]]
        distributions = [[.94]*2 + [.999]*18, [.992]*20]
        seen_scenarios = []

        def evaluate(pulse, scenarios, device, label):
            seen_scenarios.append(scenarios)
            return [dict(fidelity=f, leakage=.00001) for f in distributions[len(seen_scenarios)-1]]

        with tempfile.TemporaryDirectory() as directory, patch(
                "robust_optimization.evaluate_set", side_effect=evaluate), patch("builtins.print"):
            selected = select_robust_candidate(candidates, config, Device(), Path(directory))
        self.assertEqual(selected["parameters"]["beta"], .5)
        self.assertEqual(seen_scenarios[0], seen_scenarios[1])
        self.assertNotEqual(seen_scenarios[0], monte_carlo_scenarios(20, config["mc_seed"]))
        config["mc_selection_seed"] = config["mc_seed"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_statistics_do_not_claim_certainty_after_zero_failures(self):
        summary = summarize_trials([{"fidelity": .999, "leakage": .001}]*100)
        self.assertEqual(summary["threshold_pass_fraction"], 1)
        self.assertLess(summary["threshold_pass_fraction_wilson_ci95"][0], .99)
        self.assertAlmostEqual(summary["threshold_pass_fraction_wilson_ci95"][1], 1)

    def test_tail_confidence_bound_has_binomial_coverage(self):
        from scipy.stats import binom
        values = np.linspace(.98, 1, 1024)
        bound = lower_tail_confidence_bound(values)
        k = int(np.flatnonzero(values == bound)[0]) + 1
        self.assertLessEqual(binom.cdf(k-1, len(values), .05), .05)
        self.assertLessEqual(bound, np.quantile(values, .05))
        self.assertEqual(lower_tail_confidence_bound(values[:10]), 0)


class NSGATests(unittest.TestCase):
    def test_nsga3_four_objective_dtlz2(self):
        from pymoo.problems import get_problem
        problem = get_problem("dtlz2", n_var=6, n_obj=4)
        result = minimize(lambda x: (problem.evaluate(x), 0), [(0, 1)]*6,
                          population_size=56, generations=100, reference_partitions=5,
                          n_objectives=4, seed=8, hv_reference=[2]*4)
        # DTLZ2's known Pareto surface is the unit sphere in the positive orthant.
        radial_error = np.linalg.norm(result.objectives, axis=1) - 1
        self.assertLess(np.mean(radial_error), .025)
        self.assertEqual(len(result.reference_directions), 56)
        self.assertTrue(np.all(np.diff([h["archive_hypervolume"] for h in result.history]) >= -1e-10))

    def test_fronts_and_constraint_priority(self):
        f = np.array([[0, 1], [.5, .5], [1, 0], [1, 1], [2, 2]])
        fronts = nondominated_fronts(f)
        np.testing.assert_array_equal(fronts[0], [0, 1, 2])
        np.testing.assert_array_equal(fronts[1], [3])
        np.testing.assert_array_equal(fronts[2], [4])
        constrained = nondominated_fronts(f, [1, 1, 1, 0, 0])
        np.testing.assert_array_equal(constrained[0], [3])

    def test_zdt1_convergence_and_reproducibility(self):
        def objective(x):
            g = 1 + 9*x[1]
            return [x[0], g*(1-np.sqrt(x[0]/g))], 0

        kwargs = dict(population_size=24, generations=80, seed=12,
                      n_objectives=2, reference_partitions=12)
        result = minimize(objective, [(0, 1), (0, 1)], **kwargs)
        repeat = minimize(objective, [(0, 1), (0, 1)], **kwargs)
        np.testing.assert_array_equal(result.points, repeat.points)
        self.assertEqual(len(result.archive_points), 24*80)
        self.assertTrue(np.all(result.points >= 0) and np.all(result.points <= 1))
        self.assertLess(np.mean(result.points[:, 1]), .01)
        self.assertGreater(np.ptp(result.points[:, 0]), .9)


if __name__ == "__main__":
    unittest.main()

"""Workflow contract tests: method toggles, MC ordering and sampled-waveform export."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import robust_optimization as workflow
from challenge_model import Scenario
from optimization_ui import build_config


class WorkflowTests(unittest.TestCase):
    def config(self):
        config = json.loads(Path(__file__).with_name('challenge_config.json').read_text())
        config.update(mc_selection_samples=2, monte_carlo_samples=2, search_seeds=[7])
        return config

    def exercise(self, nsga, grape, feasible=True):
        config = self.config()
        config['nsga_enabled'] = nsga
        config['grape']['enabled'] = grape
        events = []
        original_mc = workflow.monte_carlo_scenarios
        proposals = []
        metric = dict(fidelity=.9999 if feasible else .95, infidelity=.0001 if feasible else .05,
                      leakage=1e-6, max_trace_error=0., min_output_eigenvalue=0.)
        def fake_nsga(objective, bounds, **kwargs):
            events.append('nsga')
            objective([.32, .52, 1., 0.])
            return SimpleNamespace(history=[], reference_directions=np.eye(3))
        def fake_refine(pulse, *args):
            events.append('grape')
            proposal = np.array(pulse).copy()
            proposal[2] *= 1.01
            proposals.append(proposal)
            return proposal, []
        def sample(*args, **kwargs):
            events.append('mc')
            return original_mc(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            nsga_mock = stack.enter_context(patch.object(workflow, 'nsga_minimize', side_effect=fake_nsga))
            grape_mock = stack.enter_context(patch('grape.refine', side_effect=fake_refine))
            stack.enter_context(patch.object(workflow, 'monte_carlo_scenarios', side_effect=sample))
            stack.enter_context(patch.object(workflow, 'evaluate_waveform', return_value=metric))
            stack.enter_context(patch.object(workflow, 'public_scenarios', return_value=[Scenario()]))
            for name in ('challenge_report.save_plots', 'challenge_report.write_report',
                         'stage_report.save_stage_comparisons', 'pulse_diagnostics.save_pulse_diagnostics',
                         'robust_optimization.print_final_summary', 'builtins.print'):
                stack.enter_context(patch(name))
            result = workflow.run(config, directory)
            self.assertEqual(nsga_mock.call_count, int(nsga))
            self.assertEqual(grape_mock.call_count, int(grape))
            if grape:
                self.assertLess(events.index('grape'), events.index('mc'))
                stored = np.loadtxt(Path(directory)/'grape_proposal.csv', delimiter=',', skiprows=1).T
                np.testing.assert_allclose(stored, proposals[0])
            from pulse_methods import REFERENCE_METHODS, waveform_from_spec
            self.assertTrue(set(REFERENCE_METHODS).issubset(result['variants']))
            self.assertAlmostEqual(result['variants']['robust_selected']['hardware']['duration_ns'], config['gate_duration_ns'])
            for name in ('composite_bb1','composite_corpse'):
                row=result['variants'][name]['selection']
                self.assertFalse(row['selection_feasible'])
                self.assertTrue(any('duration' in reason for reason in row['rejection_reasons']))
            self.assertTrue((Path(directory)/'selected_pulse_parameters.json').is_file())
            self.assertTrue((Path(directory)/'selected_pulse_parameters.md').is_file())
            spec = result['selected_pulse_spec']
            if spec['representation'] != 'sampled_iq':
                np.testing.assert_allclose(waveform_from_spec(spec), np.loadtxt(
                    Path(directory)/'pulse.csv', delimiter=',', skiprows=1).T, atol=1e-13)
            self.assertEqual('nsga_selected' in result['variants'], nsga)
            self.assertEqual('grape_proposal' in result['variants'], grape)
            if not nsga:
                self.assertFalse((Path(directory)/'reference_directions.csv').exists())
                self.assertFalse((Path(directory)/'nsga_archive.json').exists())
            if not feasible:
                self.assertFalse(result['quality_passed'])
                self.assertFalse(result['mc_selection']['selection_feasible'])
            return result

    def test_all_method_switch_combinations(self):
        for nsga in (False, True):
            for grape in (False, True):
                with self.subTest(nsga=nsga, grape=grape):
                    self.exercise(nsga, grape)

    def test_no_eligible_method_still_exports_diagnostic_comparison(self):
        self.exercise(False, False, feasible=False)

    def test_grape_sampled_waveform_is_used_in_selection(self):
        config = self.config()
        config['mc_selection_candidates'] = 2
        device = workflow.Device()
        parameters = dict(duration_ns=15.6, sigma_fraction=.3, beta=.5, amplitude=1., phase_ramp_mhz=0.)
        seed = np.asarray(workflow.make_pulse(workflow.PulseParameters(**parameters), device))
        refined = seed.copy()
        refined[2] *= 1.1
        common = dict(parameters=parameters, nominal=dict(fidelity=.9999, leakage=1e-6),
                      violation=0., hardware=dict(peak_amplitude_mhz=50.), selection_loss=.001)
        candidates = [dict(common, method='DRAG'), dict(common, method='GRAPE', waveform=refined.tolist())]
        def evaluate(pulse, *args):
            fidelity = .9999 if np.array_equal(pulse, refined) else .999
            return [dict(fidelity=fidelity, leakage=1e-6)]*2
        with tempfile.TemporaryDirectory() as directory, patch.object(workflow, 'evaluate_set', side_effect=evaluate), patch('builtins.print'):
            selected = workflow.select_robust_candidate(candidates, config, device, Path(directory))
        self.assertEqual(selected['method'], 'GRAPE')
        np.testing.assert_array_equal(workflow.candidate_pulse(selected, device), refined)

    def test_ui_configuration_units_and_invalid_inputs(self):
        config = self.config()
        before = copy.deepcopy(config)
        values = {key: str(config['gate_duration_ns'] if key == 'tg_ns' else config['device'][key])
                  for key in ('detuning_mhz', 'alpha_mhz', 't1_1_ns', 't1_2_ns', 'tphi_1_ns', 'tphi_2_ns', 'tg_ns')}
        values['t1_1_ns'] = '30 us'
        actual = build_config(config, values, False, True)
        self.assertEqual(actual['device']['t1_1_ns'], 30000.)
        self.assertFalse(actual['nsga_enabled'])
        self.assertTrue(actual['grape']['enabled'])
        self.assertEqual(config, before)
        for key, invalid in [('tg_ns', '15.7'), ('alpha_mhz', '220'), ('t1_1_ns', '-1')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_config(config, {**values, key: invalid}, False, False)


if __name__ == '__main__':
    unittest.main()

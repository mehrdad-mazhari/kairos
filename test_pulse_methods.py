"""Independent envelope, composite-rotation and exported-parameter checks."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import expm
from challenge_model import Device, MHZ_TO_RAD_NS
from pulse_shapes import integrate
from pulse_methods import (reference_pulses, waveform_from_spec, fourier_spec, settings,
                           parameter_rows, REFERENCE_METHODS, _modulated)
from grape import FourierControls


class PulseMethodTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name('challenge_config.json').read_text())
        self.device = Device(**self.config['device'])
        self.references = reference_pulses(self.config,self.device)

    def test_all_reference_waveforms_reproduce_and_share_grid(self):
        self.assertEqual(tuple(self.references),REFERENCE_METHODS)
        grid=None
        for name,record in self.references.items():
            with self.subTest(method=name):
                pulse=np.asarray(record['waveform'])
                self.assertTrue(np.isfinite(pulse).all())
                np.testing.assert_allclose(pulse[1:,[0,-1]],0,atol=1e-14)
                np.testing.assert_allclose(np.diff(pulse[0]),self.device.dt_ns,atol=1e-12)
                np.testing.assert_allclose(waveform_from_spec(record['pulse_spec']),pulse,atol=1e-14,rtol=1e-14)
                if not name.startswith('composite_'):
                    self.assertAlmostEqual(integrate(pulse[1],pulse[0]),np.pi,places=12)
        self.assertLess(self.references['composite_bb1']['pulse_spec']['peak_amplitude_mhz'],80.)

    def test_zero_extra_correction_recovers_drag(self):
        config=json.loads(json.dumps(self.config))
        config['pulse_methods'].update(hermite_h2=0.,wah_modulation_depth=0.)
        pulses=reference_pulses(config,self.device)
        for key in ('higher_order_drag','wah_wah'):
            np.testing.assert_allclose(pulses[key]['waveform'],pulses['drag_baseline']['waveform'],atol=1e-14)

    def test_corrected_quadrature_matches_independent_derivative(self):
        times=np.linspace(0.,20.,20001)
        alpha=-220*MHZ_TO_RAD_NS
        for h2,depth in ((.5,0.),(0.,.3)):
            pulse,_=_modulated(times,6.,alpha,.5,1.,h2=h2,depth=depth,frequency=25.)
            derivative=np.gradient(pulse[1],times,edge_order=2)
            np.testing.assert_allclose(pulse[2,2:-2],-.5*derivative[2:-2]/alpha,atol=2e-9)

    def test_bb1_segment_area_phase_and_ideal_error_cancellation(self):
        record=self.references['composite_bb1']
        pulse=np.asarray(record['waveform'])
        sx=np.array([[0,1],[1,0]],complex)
        sy=np.array([[0,-1j],[1j,0]],complex)
        def gate(error):
            unitary=np.eye(2,dtype=complex)
            for segment in record['pulse_spec']['segments']:
                start,stop=segment['start_sample'],segment['end_sample']
                z=(pulse[1,start:stop+1]+1j*pulse[2,start:stop+1])*np.exp(-1j*segment['phase_rad'])
                self.assertAlmostEqual(integrate(z.imag,pulse[0,start:stop+1]),0.,places=12)
                area=integrate(z.real,pulse[0,start:stop+1])
                self.assertAlmostEqual(area,segment['rotation_rad'],places=12)
                axis=np.cos(segment['phase_rad'])*sx+np.sin(segment['phase_rad'])*sy
                unitary=expm(-.5j*(1+error)*area*axis)@unitary
            return unitary
        ideal=gate(0.)
        self.assertAlmostEqual(abs(np.trace(sx@ideal))/2,1.,places=12)
        error=.03
        bb1_error=1-abs(np.trace(sx@gate(error))/2)**2
        plain_error=1-abs(np.trace(sx@expm(-.5j*np.pi*(1+error)*sx))/2)**2
        self.assertLess(bb1_error,plain_error/1000)
        self.assertNotIn('sigma_ns',record['pulse_spec'])

    def test_composites_real_transmon_and_duration(self):
        from challenge_model import evaluate_waveform
        for name in ('composite_bb1','composite_corpse'):
            record=self.references[name]
            pulse=np.asarray(record['waveform'])
            self.assertGreater(pulse[0,-1],self.config['gate_duration_ns'])
            result=evaluate_waveform(*pulse,self.device)
            self.assertGreater(result['fidelity'],.99)
            self.assertLess(result['leakage'],.002)

    def test_corpse_canonical_square_sequence_cancels_small_detuning(self):
        sx=np.array([[0,1],[1,0]],complex)
        sz=np.diag([1.,-1.])
        def gate(offset):
            unitary=np.eye(2,dtype=complex)
            for angle,sign in [(7*np.pi/3,1),(5*np.pi/3,-1),(np.pi/3,1)]:
                unitary=expm(-.5j*angle*(sign*sx+offset*sz))@unitary
            return 1-abs(np.trace(sx@unitary)/2)**2
        self.assertLess(abs(gate(0)),1e-12)
        self.assertLess(gate(.01),1e-7)

    def test_fourier_recipe_recreates_actual_final_controls(self):
        t=np.asarray(self.references['drag_baseline']['waveform'])[0]
        a,b=[.65,-.1,.02],[.06,-.01,.003]
        scale=.5
        controls=FourierControls(t,3,scale).waveform(np.r_[a,b])
        pulse=np.array([t,controls[:,0],controls[:,1]])
        spec=fourier_spec(pulse,[dict(a_coefficients=a,b_coefficients=b,scale_rad_ns=scale)],
                          self.references['drag_baseline']['pulse_spec'],self.device)
        np.testing.assert_allclose(waveform_from_spec(spec),pulse,rtol=1e-13,atol=1e-13)
        self.assertNotIn('sigma_ns',spec)
        rows=dict(parameter_rows(spec))
        self.assertIn('I Fourier coefficient 3',rows)
        self.assertIn('Seed / Sigma',rows)

    def test_invalid_family_settings_rejected(self):
        for key,value in [('hermite_h2',2.),('wah_modulation_frequency_mhz',float('nan')),
                          ('bb1_edge_fraction',0.),('amplitude_scale',-1.)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                settings({'pulse_methods':{key:value}})


if __name__ == '__main__':
    unittest.main()

"""Constrained Fourier GRAPE with a dissipative ensemble objective.

Frechet derivatives of Liouvillian exponentials give analytic control gradients.
Linear sample interpolation is integrated with midpoint substeps; acceptance
uses the production QuTiP solver, not this optimization approximation.
"""
import numpy as np
from scipy.linalg import expm_frechet
from scipy.optimize import minimize
from qutip import liouvillian
from challenge_model import (DRIVE_X, DRIVE_Y, KETS, TEST_STATES, TARGET_X,
                             Scenario, collapse_operators, hamiltonian, MHZ_TO_RAD_NS)

DEFAULTS = dict(enabled=True, iterations=40, samples=8, seed=2030,
                substeps=4, tail_fraction=.1, leakage_weight=2.,
                smoothness_weight=.001,
                harmonics=5, restarts=2, restart_scale=.08, tail_weight=0.,
                amplitude_grid_points=3, detuning_grid_points=5, ftol=1e-9)


def settings(config):
    resolved = {**DEFAULTS, **config.get('grape', {})}
    resolved.pop('initial_step', None)
    resolved.pop('backtracks', None)
    return resolved


def _vec(matrix):
    return matrix.full().reshape(-1, order='F')


class EnsembleGRAPE:
    def __init__(self, times, device, scenarios, options):
        self.times, self.options = np.asarray(times), options
        self.initial = np.column_stack([_vec(s.proj()) for s in TEST_STATES])
        self.target = np.column_stack([_vec((TARGET_X*s).proj()) for s in TEST_STATES])
        self.leak = np.column_stack([_vec(KETS[2].proj())]*6)
        self.models = []
        for scenario in scenarios:
            static = hamiltonian(times, np.zeros(len(times)), np.zeros(len(times)), device, scenario)(0)
            self.models.append((liouvillian(static, collapse_operators(device, scenario)).full(),
                liouvillian(DRIVE_X*scenario.amplitude_scale, []).full(),
                liouvillian(DRIVE_Y*scenario.amplitude_scale, []).full()))

    def scenario_value_gradient(self, controls, model):
        base, di, dq = model
        state = self.initial.copy()
        history = []
        substeps = self.options['substeps']
        for j, interval in enumerate(np.diff(self.times)):
            dt = interval/substeps
            for sub in range(substeps):
                w = (sub+.5)/substeps
                u = (1-w)*controls[j]+w*controls[j+1]
                generator = dt*(base+u[0]*di+u[1]*dq)
                propagator, gi = expm_frechet(generator, dt*di)
                gq = expm_frechet(generator, dt*dq, compute_expm=False)
                history.append((j, w, state, propagator, gi, gq))
                state = propagator@state
        costate = (-self.target+self.options['leakage_weight']*self.leak)/6
        value = 1+np.vdot(costate, state).real
        gradient = np.zeros_like(controls)
        for j, w, forward, propagator, gi, gq in reversed(history):
            derivative = np.array([np.vdot(costate, gi@forward).real,
                                   np.vdot(costate, gq@forward).real])
            gradient[j] += (1-w)*derivative
            gradient[j+1] += w*derivative
            costate = propagator.conj().T@costate
        return float(value), gradient

    def value_gradient(self, controls):
        evaluated = [self.scenario_value_gradient(controls, model) for model in self.models]
        values = np.array([v for v, _ in evaluated])
        gradients = np.array([g for _, g in evaluated])
        count = max(1, int(np.ceil(self.options['tail_fraction']*len(values))))
        tail = np.argsort(values)[-count:]
        tail_weight = self.options.get('tail_weight', 0.)
        value = tail_weight*values[tail].mean()+(1-tail_weight)*values.mean()
        gradient = tail_weight*gradients[tail].mean(axis=0)+(1-tail_weight)*gradients.mean(axis=0)
        differences = np.diff(controls, axis=0)
        weight = self.options['smoothness_weight']/len(differences)
        value += weight*np.sum(differences**2)
        gradient[:-1] -= 2*weight*differences
        gradient[1:] += 2*weight*differences
        return float(value), gradient


class FourierControls:
    """Wright et al. Eq. (2): odd sine harmonics for I, even for Q.

    dimensionless coefficients multiply the nominal hardware scale (rad/ns).
    Sample values use linear interpolation in the challenge solver.
    """
    def __init__(self, times, harmonics, scale):
        phase = np.pi*(np.asarray(times)-times[0])/(times[-1]-times[0])
        n = np.arange(1, harmonics+1)
        self.i = scale*np.sin(phase[:, None]*(2*n-1))
        self.q = scale*np.sin(phase[:, None]*(2*n))
        self.i[[0, -1]] = 0.
        self.q[[0, -1]] = 0.
        self.harmonics = harmonics

    def waveform(self, coefficients):
        return np.column_stack([self.i@coefficients[:self.harmonics],
                                self.q@coefficients[self.harmonics:]])

    def gradient(self, sample_gradient):
        return np.concatenate([self.i.T@sample_gradient[:, 0], self.q.T@sample_gradient[:, 1]])

    def fit(self, controls):
        return np.concatenate([np.linalg.lstsq(self.i, controls[:, 0], rcond=None)[0],
                               np.linalg.lstsq(self.q, controls[:, 1], rcond=None)[0]])

    def constraints(self, coefficients, limit):
        u = self.waveform(coefficients)
        return 1-np.sum(u*u, axis=1)/limit**2

    def constraint_gradient(self, coefficients, limit):
        u = self.waveform(coefficients)
        return np.column_stack([-2*u[:, [0]]*self.i, -2*u[:, [1]]*self.q])/limit**2


def refine(pulse, device, scenarios, options, amplitude_scale):
    """Constrained Fourier-GRAPE with analytic chain-rule gradients and restarts.

    Eq. (2), ensemble averaging Eq. (4) and Appendix B of Wright et al.;
    smooth sinusoidal initialization/restarts from Rowland & Jones Sec. 1(d).
    We retain dissipative six-state X fidelity rather than copying their closed
    system X_pi/2 trace objective, and enforce the challenge's joint I/Q bound.
    """
    times, i, q = pulse
    controls = np.column_stack([i, q])
    model = EnsembleGRAPE(times, device, scenarios, options)
    scale = device.amplitude_limit_mhz*MHZ_TO_RAD_NS
    limit = scale/amplitude_scale
    basis = FourierControls(times, options['harmonics'], scale)
    start = basis.fit(controls)
    def feasible_start(c):
        peak = np.linalg.norm(basis.waveform(c), axis=1).max()
        return c*min(1., .999*limit/max(peak, 1e-30))
    start = feasible_start(start)
    original_loss = model.value_gradient(controls)[0]
    history = [{'iteration': 0, 'loss': original_loss, 'kind': 'pre-GRAPE seed'}]
    rng = np.random.default_rng(options['seed'])
    best_value, best_coefficients = float('inf'), start.copy()
    for restart in range(options['restarts']):
        initial = start if restart == 0 else feasible_start(start+rng.normal(0, options['restart_scale'], len(start)))
        cache = {}
        def objective(c):
            nonlocal best_value, best_coefficients
            key = c.tobytes()
            if key not in cache:
                value, gradient = model.value_gradient(basis.waveform(c))
                cache.clear()
                cache[key] = value, basis.gradient(gradient)
                if np.min(basis.constraints(c, limit)) >= -1e-10 and value < best_value:
                    best_value, best_coefficients = value, c.copy()
            return cache[key]
        def progress(c):
            value, gradient = objective(c)
            history.append({'iteration': len(history), 'restart': restart+1, 'loss': float(value),
                            'coefficient_gradient': gradient.tolist(),
                            'gradient_norm': float(np.linalg.norm(gradient)),
                            'coefficients': c.tolist()})
            if len(history) % 5 == 0:
                print(f"  Fourier-GRAPE start {restart+1}/{options['restarts']}: loss={value:.9g}", flush=True)
        progress(initial)
        history[-1]['kind'] = 'Fourier start'
        result = minimize(objective, initial, method='SLSQP', jac=True,
                          constraints={'type': 'ineq',
                            'fun': lambda c: basis.constraints(c, limit),
                            'jac': lambda c: basis.constraint_gradient(c, limit)},
                          callback=progress,
                          options={'maxiter': options['iterations'], 'ftol': options['ftol']})
        value, gradient = objective(result.x)
        history.append({'iteration': len(history), 'restart': restart+1,
                        'loss': float(result.fun), 'optimizer_success': bool(result.success),
                        'message': str(result.message),
                        'coefficient_gradient': gradient.tolist(),
                        'gradient_norm': float(np.linalg.norm(gradient)),
                        'coefficients': result.x.tolist()})
    history.append({'iteration': len(history), 'loss': best_value, 'kind': 'best feasible Fourier proposal',
                    'a_coefficients': best_coefficients[:options['harmonics']].tolist(),
                    'b_coefficients': best_coefficients[options['harmonics']:].tolist(),
                    'scale_rad_ns': scale})
    u = basis.waveform(best_coefficients)
    return (np.asarray(times).copy(), u[:, 0], u[:, 1]), history

"""Shared pulse envelopes; angular amplitudes are in rad/ns."""

import numpy as np

GAUSSIAN_SIGMA_NS = 5.0
# arXiv:1008.2554 Eq. (14): beta=lambda^2/4. The challenge fixes lambda=sqrt(2).
PAPER_DRAG_BETA = 0.5


def integrate(values, times):
    """trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, times)
    return np.trapz(values, times)


def normalize_pi_area(envelope, times):
    """rotation area is pi."""
    return np.pi * envelope / integrate(envelope, times)


def square_pulse(times):
    return normalize_pi_area(np.ones_like(times), times)


def gaussian_pulse(times, sigma=GAUSSIAN_SIGMA_NS):
    center = 0.5 * (times[0] + times[-1])
    raw = np.exp(-0.5 * ((times - center) / sigma) ** 2)
    raw -= raw[0]
    return normalize_pi_area(raw, times)


def drag_pulse(times, alpha_rad_ns, beta=1.0, sigma=GAUSSIAN_SIGMA_NS):
    """Return I/Q envelopes using Q = -beta * dI/dt / anharmonicity."""
    in_phase = gaussian_pulse(times, sigma)
    derivative = np.gradient(in_phase, times)
    quadrature = -beta * derivative / alpha_rad_ns
    return in_phase, quadrature


def smooth_drag_pulse(times, alpha_rad_ns, beta, sigma, amplitude=1.0,
                      phase_ramp_mhz=0.0):
    """Gaussian with sin^2 taper: both I and derivative-based Q end at zero.

    A centered linear phase ramp is encoded in I/Q, so no extra detuning
    control or unreported virtual-Z correction is used in the Hamiltonian.
    Uses the fixed-frequency derivative control of arXiv:1008.2554 Eq. (14):
    Q=-beta*dI/dt/alpha_nominal. The paper's Delta is our anharmonicity alpha;
    its dimensionless alpha is our beta. beta=0.5 for the challenge's sqrt(2)
    coupling; beta remains tunable. The taper enforces the challenge endpoints
    instead of copying Eq. (12)'s shifted Gaussian. A constant phase ramp is
    an extra optimized control, not the full time-dependent DRAG of Eq. (13).
    DRAG is computed once from nominal alpha, never retuned for test samples.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 3 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0) or not np.isfinite(sigma) or sigma <= 0
            or not np.isfinite(alpha_rad_ns) or alpha_rad_ns == 0
            or not np.all(np.isfinite([beta, amplitude, phase_ramp_mhz]))):
        raise ValueError("require increasing finite times, positive sigma, nonzero alpha")
    duration = times[-1] - times[0]
    offset = times - (times[0] + times[-1]) / 2
    angle = np.pi * (times - times[0]) / duration
    gaussian = np.exp(-0.5 * (offset / sigma)**2)
    raw = gaussian * np.sin(angle)**2
    derivative = gaussian * (np.pi / duration * np.sin(2 * angle)
                             - offset / sigma**2 * np.sin(angle)**2)
    factor = amplitude * np.pi / integrate(raw, times)
    z = factor * (raw - 1j * beta * derivative / alpha_rad_ns)
    z *= np.exp(2j * np.pi * phase_ramp_mhz * 1e-3 * offset)
    z[[0, -1]] = 0
    return z.real.copy(), z.imag.copy()


"""Envelope coefficients, with applied and paper-reference shapes kept distinct."""
import math


def shifted_gaussian_coefficients(duration_ns, sigma_ns, area_rad=math.pi):
    """Eq. (12) convention: I(t)=A*(exp(-(t-T/2)^2/(2*sigma^2))-B).

    B is dimensionless. If written I=A*exp(...)-B_offset instead, B_offset=A*B.
    Uses the paper's continuous area condition, not sampled normalization.
    """
    b = math.exp(-duration_ns**2/(8*sigma_ns**2))
    integral = sigma_ns*math.sqrt(2*math.pi)*math.erf(duration_ns/(2*math.sqrt(2)*sigma_ns)) - duration_ns*b
    a = area_rad/integral
    return {'sigma_ns': sigma_ns, 'A_rad_per_ns': a,
            'A_over_2pi_mhz': a/(2*math.pi*.001), 'B_dimensionless': b,
            'B_offset_rad_per_ns': a*b, 'area_rad': area_rad}


def pulse_coefficients(parameters, dt_ns):
    """Coefficients for the exported tapered envelope and Eq. (12) references."""
    t = parameters['duration_ns']
    sigma = t*parameters['sigma_fraction']
    steps = round(t/dt_ns)
    values = [math.exp(-.5*((i*dt_ns-t/2)/sigma)**2)*math.sin(math.pi*i*dt_ns/t)**2
              for i in range(steps+1)]
    integral = sum((values[i]+values[i+1])*dt_ns/2 for i in range(steps))
    area = parameters['amplitude']*math.pi
    a = area/integral
    return {
        'applied_tapered_envelope': {'A_rad_per_ns': a,
            'A_over_2pi_mhz': a/(2*math.pi*.001), 'B_dimensionless': 0.0,
            'B_offset_rad_per_ns': 0.0, 'pre_phase_sampled_area_rad': area},
        'shifted_gaussian_same_sigma_reference': shifted_gaussian_coefficients(t, sigma, area),
        'paper_eq12_sigma_T_over_2_pi_area_reference': shifted_gaussian_coefficients(t, t/2),
    }

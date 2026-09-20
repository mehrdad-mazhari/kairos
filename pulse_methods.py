"""Reproducible reference families; composite comparisons use explicit longer durations.

Higher-order DRAG uses the second-order Hermite envelope in Rigetti's
hrm_gaussian definition, with this project's smooth taper and DRAG sign.
Wah-Wah follows Vesterinen et al. arXiv:1405.0450 Eqs. (1)-(2), tapered.
BB1 uses the amplitude-error correcting sequence with phi=acos(-1/4),
implemented with DRAG-shaped subpulses at explicitly extended durations.
Legacy raised-cosine BB1 recipes remain reconstructible.
"""
from dataclasses import asdict
import numpy as np
from challenge_model import MHZ_TO_RAD_NS, PulseParameters, make_pulse, hardware_metrics
from pulse_shapes import PAPER_DRAG_BETA, integrate, smooth_drag_pulse
from pulse_coefficients import pulse_coefficients

METHOD_LABELS = {
    'gaussian_baseline': 'Gaussian', 'drag_baseline': 'DRAG',
    'higher_order_drag': 'Higher-order DRAG', 'wah_wah': 'Wah-Wah',
    'composite_bb1': 'Composite BB1', 'composite_corpse': 'Composite CORPSE', 'nsga_selected': 'NSGA-III',
    'grape_proposal': 'GRAPE', 'mc_selected': 'MC selection (legacy)',
    'robust_selected': 'Final selected pulse',
}
REFERENCE_METHODS = tuple(list(METHOD_LABELS)[:6])
DEFAULTS = dict(sigma_fraction=.3, amplitude_scale=1., beta=PAPER_DRAG_BETA,
                hermite_h2=.5, wah_modulation_depth=.3,
                wah_modulation_frequency_mhz=25., bb1_edge_fraction=.1)


def settings(config):
    options = {**DEFAULTS, **config.get('pulse_methods', {})}
    if (not all(np.isfinite(v) for v in options.values())
            or options['sigma_fraction'] <= 0 or options['amplitude_scale'] <= 0
            or not 0 <= options['hermite_h2'] <= 1
            or not 0 <= options['wah_modulation_depth'] < 1
            or not 0 < options['wah_modulation_frequency_mhz'] < 2500
            or not 0 < options['bb1_edge_fraction'] <= .5):
        raise ValueError('invalid pulse_methods parameters (positive sigma/amplitude; H2 in [0,1]; modulation depth in [0,1); positive sub-Nyquist frequency; edge fraction in (0,.5])')
    return options


def analytic_spec(parameters, device, method):
    p = asdict(parameters)
    return dict(method=method, representation='tapered_gaussian_drag',
                duration_ns=p['duration_ns'], dt_ns=device.dt_ns,
                sigma_ns=p['sigma_fraction']*p['duration_ns'],
                sigma_fraction=p['sigma_fraction'], amplitude_scale=p['amplitude'],
                beta=p['beta'], phase_ramp_mhz=p['phase_ramp_mhz'],
                alpha_mhz=device.alpha_mhz,
                **pulse_coefficients(p, device.dt_ns)['applied_tapered_envelope'],
                formula='I+iQ = A*(g - i*beta*g_dot/alpha)*exp(i*2*pi*phase_ramp_mhz*0.001*(t-T/2)); g=exp(-(t-T/2)^2/(2*sigma^2))*sin(pi*t/T)^2',
                notes='Sampled I/Q, linearly interpolated. Amplitude scale is dimensionless; A and peak amplitude are separate quantities.')


def _modulated(times, sigma, alpha, beta, amplitude, *, h2=0., depth=0., frequency=0.):
    x = times - (times[0]+times[-1])/2
    angle = np.pi*(times-times[0])/(times[-1]-times[0])
    gauss = np.exp(-.5*(x/sigma)**2)
    taper = np.sin(angle)**2
    base = gauss*taper
    derivative = gauss*(np.pi/(times[-1]-times[0])*np.sin(2*angle)-x/sigma**2*taper)
    w = frequency*MHZ_TO_RAD_NS
    modulation = (1-h2*x*x/(2*sigma*sigma))*(1-depth*np.cos(w*x))
    modulation_dot = (-h2*x/sigma**2*(1-depth*np.cos(w*x))
                      +(1-h2*x*x/(2*sigma*sigma))*depth*w*np.sin(w*x))
    raw = base*modulation
    norm = amplitude*np.pi/integrate(raw, times)
    i = norm*raw
    q = -beta*norm*(derivative*modulation+base*modulation_dot)/alpha
    i[[0,-1]] = 0.; q[[0,-1]] = 0.
    return np.array([times, i, q]), float(norm)


def _bb1(times, edge_fraction, amplitude_scale):
    # Chronological target pi followed by the BB1 identity correction.
    angles = np.array([np.pi, np.pi, 2*np.pi, np.pi])
    phi = float(np.arccos(-.25))
    phases = [0., phi, 3*phi, phi]
    total_steps = len(times)-1
    boundaries = np.r_[0, np.rint(np.cumsum(angles)/angles.sum()*total_steps).astype(int)]
    pulse = np.zeros((3, len(times)))
    pulse[0] = times
    segments = []
    for start, stop, angle, phase in zip(boundaries[:-1], boundaries[1:], angles, phases):
        t = times[start:stop+1]
        u = (t-t[0])/(t[-1]-t[0])
        edge = np.minimum(u, 1-u)/edge_fraction
        envelope = np.sin(.5*np.pi*np.minimum(edge, 1.))**2
        envelope[[0,-1]] = 0.
        peak = amplitude_scale*angle/integrate(envelope,t)
        pulse[1,start:stop+1] = peak*envelope*np.cos(phase)
        pulse[2,start:stop+1] = peak*envelope*np.sin(phase)
        segments.append(dict(start_sample=int(start), end_sample=int(stop),
                             start_ns=float(t[0]), duration_ns=float(t[-1]-t[0]),
                             rotation_rad=float(angle), phase_rad=float(phase),
                             phase_deg=float(np.degrees(phase)), amplitude_rad_ns=float(peak),
                             amplitude_mhz=float(peak/MHZ_TO_RAD_NS)))
    return pulse, segments


def _composite(segments, dt, sigma_fraction, beta, alpha, amplitude):
    pieces = []
    for segment in segments:
        t = np.arange(round(segment['duration_ns']/dt)+1)*dt
        i, q = smooth_drag_pulse(t, alpha*MHZ_TO_RAD_NS, beta,
                                sigma_fraction*t[-1], amplitude*segment['rotation_rad']/np.pi)
        z = (i+1j*q)*np.exp(1j*segment['phase_rad'])
        pieces.append(z if not pieces else z[1:])
    z = np.concatenate(pieces)
    return np.array([np.arange(len(z))*dt, z.real, z.imag])


def composite_reference(kind, requested, options, device, headroom):
    phi = float(np.arccos(-.25))
    angles, phases = ((np.pi*np.array([1.,1.,2.,1.]), [0.,phi,3*phi,phi])
                      if kind == 'composite_bb1' else
                      (np.pi*np.array([7/3,5/3,1/3]), [0.,np.pi,0.]))
    # Every segment gets at least the requested primitive duration. Longer
    # rotations retain approximately the primitive drive strength.
    durations = requested*np.maximum(angles/np.pi, 1.)
    for attempt in range(1000):
        steps = np.ceil(durations/device.dt_ns-1e-10).astype(int)
        boundaries = np.r_[0, np.cumsum(steps)]
        segments = [dict(start_sample=int(a), end_sample=int(b), start_ns=float(a*device.dt_ns),
                         duration_ns=float((b-a)*device.dt_ns), rotation_rad=float(angle),
                         phase_rad=float(phase), phase_deg=float(np.degrees(phase)))
                    for a,b,angle,phase in zip(boundaries[:-1],boundaries[1:],angles,phases)]
        pulse = _composite(segments, device.dt_ns, options['sigma_fraction'], options['beta'],
                           device.alpha_mhz, options['amplitude_scale'])
        peak = hardware_metrics(*pulse,device)['peak_amplitude_mhz']
        if peak*headroom <= device.amplitude_limit_mhz+1e-10:
            break
        durations *= max(1.01, peak*headroom/device.amplitude_limit_mhz)
    else:
        raise ValueError('Unable to construct hardware-feasible composite pulse')
    for segment in segments:
        a,b=segment['start_sample'],segment['end_sample']
        segment.update(sigma_ns=options['sigma_fraction']*segment['duration_ns'],
                       beta=options['beta'], amplitude_mhz=float(np.hypot(*pulse[1:,a:b+1]).max()/MHZ_TO_RAD_NS))
    spec = dict(method=METHOD_LABELS[kind], representation='composite_drag',
                duration_ns=float(pulse[0,-1]), requested_duration_ns=requested,
                dt_ns=device.dt_ns, sigma_fraction=options['sigma_fraction'], beta=options['beta'],
                alpha_mhz=device.alpha_mhz, amplitude_scale=options['amplitude_scale'],
                segments=segments, total_rotation_area_rad=float(sum(angles)*options['amplitude_scale']),
                formula='BB1: (pi,0),(pi,phi),(2*pi,3*phi),(pi,phi), phi=acos(-1/4). CORPSE: (7*pi/3,0),(5*pi/3,pi),(pi/3,0). Each segment is a tapered Gaussian with local DRAG quadrature, then phase rotated.',
                source='https://www.nature.com/articles/s41598-020-58823-9',
                notes='Extended-duration comparison only for the fixed-Tg task. Each segment lasts at least requested Tg; durations extend further for hardware headroom. Shaped/DRAG adaptation of canonical composite rotations; ideal square-pulse error-cancellation guarantees do not automatically carry over. Fidelity includes decoherence throughout actual duration.')
    return pulse, spec


def comparison_grid(pulses):
    """Zero padding is for exported plots only, never for physical evaluation."""
    time = np.unique(np.concatenate([p[0] for p in pulses.values()]))
    return time, {name: np.array([time, np.interp(time,p[0],p[1],left=0,right=0),
                                 np.interp(time,p[0],p[2],left=0,right=0)]) for name,p in pulses.items()}


def reference_pulses(config, device):
    """Six distinct designs; all are fixed references, not extra numerical searches."""
    o = settings(config)
    result = {}
    for key in REFERENCE_METHODS:
        p = PulseParameters(config['gate_duration_ns'], sigma_fraction=o['sigma_fraction'],
                            beta=0. if key in ('gaussian_baseline','composite_bb1') else o['beta'],
                            amplitude=o['amplitude_scale'])
        pulse = np.asarray(make_pulse(p, device))
        spec = analytic_spec(p, device, METHOD_LABELS[key])
        if key in ('higher_order_drag','wah_wah'):
            h2 = o['hermite_h2'] if key == 'higher_order_drag' else 0.
            depth = o['wah_modulation_depth'] if key == 'wah_wah' else 0.
            frequency = o['wah_modulation_frequency_mhz'] if key == 'wah_wah' else 0.
            pulse, norm = _modulated(pulse[0], spec['sigma_ns'], device.alpha_mhz*MHZ_TO_RAD_NS,
                                     p.beta, p.amplitude, h2=h2, depth=depth, frequency=frequency)
            spec.update(representation='hermite_drag' if key == 'higher_order_drag' else 'wah_wah',
                        A_rad_per_ns=norm, A_over_2pi_mhz=norm/MHZ_TO_RAD_NS,
                        formula='I=A*g*m; Q=-beta*A*(g_dot*m+g*m_dot)/alpha; g=exp(-x^2/(2*sigma^2))*sin(pi*t/T)^2; x=t-T/2')
            if key == 'higher_order_drag':
                spec.update(hermite_h2=h2, modulation='m=1-H2*x^2/(2*sigma^2)',
                            source='https://pyquil-docs.rigetti.com/en/stable/quilt_waveforms.html#hrm-gaussian',
                            notes='Second-order Hermite DRAG with smooth taper; not the fifth-order/chirped DRAG expansion. Fixed reference settings, not independently optimized.')
            else:
                spec.update(modulation_depth=depth, modulation_frequency_mhz=frequency,
                            modulation='m=1-depth*cos(2*pi*frequency_mhz*0.001*x)',
                            source='https://arxiv.org/abs/1405.0450',
                            notes='Wah-Wah envelope with smooth taper, evaluated on one transmon. No spectator-qubit crosstalk model; this cannot assess external leakage protection. Fixed reference settings, not independently optimized.')
        elif key in ('composite_bb1', 'composite_corpse'):
            pulse, spec = composite_reference(key, config['gate_duration_ns'], o, device,
                max(config.get('amplitude_headroom_scale', 1.), 1 + config.get('manufacturing_ranges', {}).get('amplitude_fraction', .05)))
            p = PulseParameters(float(pulse[0,-1]), sigma_fraction=o['sigma_fraction'],
                                beta=o['beta'], amplitude=o['amplitude_scale'])
        spec.update(peak_amplitude_mhz=hardware_metrics(*pulse, device)['peak_amplitude_mhz'],
                    sample_count=len(pulse[0]), interpolation='linear')
        result[key] = dict(parameters=asdict(p), method=METHOD_LABELS[key],
                           waveform=pulse.tolist(), pulse_spec=spec)
    return result


def fourier_spec(pulse, history, seed_spec, device):
    spec = dict(method='GRAPE', representation='sampled_iq', duration_ns=float(pulse[0][-1]),
                dt_ns=device.dt_ns, sample_count=len(pulse[0]), interpolation='linear',
                peak_amplitude_mhz=hardware_metrics(*pulse, device)['peak_amplitude_mhz'],
                seed=seed_spec,
                notes='No single final sigma or beta: those belong only to the starting seed. Final pulse.csv contains the authoritative I/Q samples.')
    if history and 'a_coefficients' in history[-1]:
        row = history[-1]
        spec.update(representation='fourier', a_coefficients=row['a_coefficients'],
                    b_coefficients=row['b_coefficients'], scale_rad_ns=row['scale_rad_ns'],
                    formula='I=scale*sum(a_n*sin((2*n-1)*pi*t/T)); Q=scale*sum(b_n*sin(2*n*pi*t/T)); n=1..N. Sample then linearly interpolate.')
    return spec


def parameter_rows(spec):
    """Shared human-readable output for the desktop, terminal and Markdown report."""
    fields = [('method','Method',''), ('representation','Representation',''),
              ('duration_ns','Actual gate duration','ns'), ('requested_duration_ns','Requested gate duration','ns'), ('dt_ns','Sample interval','ns'),
              ('peak_amplitude_mhz','Peak |I+iQ| / 2pi','MHz'),
              ('amplitude_scale','Amplitude scale','dimensionless'),
              ('A_rad_per_ns','Envelope coefficient A','rad/ns'), ('A_over_2pi_mhz','A / 2pi','MHz'),
              ('B_dimensionless','Envelope offset B','dimensionless'),
              ('sigma_ns','Sigma','ns'), ('sigma_fraction','Sigma / Tg',''),
              ('beta','DRAG beta','dimensionless'), ('alpha_mhz','Nominal alpha / 2pi','MHz'),
              ('phase_ramp_mhz','I/Q phase ramp','MHz'), ('hermite_h2','Hermite H2',''),
              ('modulation_depth','Sideband modulation depth',''),
              ('modulation_frequency_mhz','Sideband modulation frequency','MHz'),
              ('edge_fraction','BB1 edge fraction',''), ('phi_rad','BB1 phase phi','rad'),
              ('total_rotation_area_rad','Total rotation area','rad'),
              ('scale_rad_ns','Fourier amplitude scale','rad/ns'), ('sample_count','Samples',''),
              ('interpolation','Interpolation',''), ('waveform_file','Waveform file','')]
    rows = []
    for key,label,unit in fields:
        if key in spec:
            value=spec[key]
            text=f'{value:.10g}' if isinstance(value,(float,int)) else str(value)
            rows.append((label, text+(' '+unit if unit else '')))
    for key,channel in [('a_coefficients','I'),('b_coefficients','Q')]:
        for n,value in enumerate(spec.get(key,[]),1):
            rows.append((f'{channel} Fourier coefficient {n}',f'{value:.12g}'))
    for n,segment in enumerate(spec.get('segments',[]),1):
        rows.append((f'Segment {n}', f"t={segment['start_ns']:.8g} ns; duration={segment['duration_ns']:.8g} ns; angle={segment['rotation_rad']:.10g} rad; phase={segment['phase_deg']:.10g} deg; amp={segment['amplitude_mhz']:.10g} MHz"))
        if 'sigma_ns' in segment:
            rows.append((f'Segment {n} shape', f"sigma={segment['sigma_ns']:.10g} ns; beta={segment['beta']:.10g}; amplitude above is peak |I+iQ|/2pi"))
    for key in ('formula','modulation','notes','source'):
        if key in spec:
            rows.append((key.capitalize(),spec[key]))
    if 'seed' in spec:
        rows.append(('Starting seed (not final waveform)', spec['seed'].get('method','unknown')))
        rows.extend(('Seed / '+label, value) for label,value in parameter_rows(spec['seed']))
    return rows


def selected_spec_from_manifest(manifest):
    """Also expose honest pulse parameters when opening pre-feature saved runs."""
    if manifest.get('selected_pulse_spec'):
        return manifest['selected_pulse_spec']
    from challenge_model import Device
    device = Device(**manifest['config']['device'])
    final = manifest['variants']['robust_selected']
    if final.get('pulse_spec'):
        return {**final['pulse_spec'], 'waveform_file': 'pulse.csv'}
    method = manifest.get('selected_method', 'Gaussian/DRAG family (legacy)')
    spec = analytic_spec(PulseParameters(**final['parameters']), device, method)
    grape = manifest.get('grape', {})
    if grape.get('selected', grape.get('accepted', False)):
        spec['method'] = grape.get('seed_method', 'Gaussian/DRAG seed')
        times = np.arange(round(spec['duration_ns']/device.dt_ns)+1)*device.dt_ns
        spec = fourier_spec((times,np.zeros_like(times),np.zeros_like(times)), grape.get('history', []), spec, device)
    spec.update(peak_amplitude_mhz=final['hardware']['peak_amplitude_mhz'], waveform_file='pulse.csv')
    return spec


def waveform_from_spec(spec):
    """Rebuild the sampled I/Q waveform from selected_pulse_parameters.json."""
    from challenge_model import Device
    t = np.arange(round(spec['duration_ns']/spec['dt_ns'])+1)*spec['dt_ns']
    kind = spec['representation']
    if kind == 'tapered_gaussian_drag':
        p = PulseParameters(spec['duration_ns'], spec['sigma_fraction'], spec['beta'],
                            spec['amplitude_scale'], spec['phase_ramp_mhz'])
        return np.asarray(make_pulse(p, Device(alpha_mhz=spec['alpha_mhz'], dt_ns=spec['dt_ns'])))
    if kind in ('hermite_drag','wah_wah'):
        return _modulated(t, spec['sigma_ns'], spec['alpha_mhz']*MHZ_TO_RAD_NS,
                          spec['beta'], spec['amplitude_scale'], h2=spec.get('hermite_h2',0.),
                          depth=spec.get('modulation_depth',0.),
                          frequency=spec.get('modulation_frequency_mhz',0.))[0]
    if kind == 'composite_drag':
        return _composite(spec['segments'], spec['dt_ns'], spec['sigma_fraction'], spec['beta'],
                          spec['alpha_mhz'], spec['amplitude_scale'])
    if kind == 'bb1':
        return _bb1(t, spec['edge_fraction'], spec['amplitude_scale'])[0]
    if kind == 'fourier':
        from grape import FourierControls
        basis = FourierControls(t,len(spec['a_coefficients']),spec['scale_rad_ns'])
        controls = basis.waveform(np.r_[spec['a_coefficients'],spec['b_coefficients']])
        return np.array([t, controls[:,0], controls[:,1]])
    raise ValueError('This legacy sampled waveform requires loading pulse.csv; no analytic coefficients were saved.')

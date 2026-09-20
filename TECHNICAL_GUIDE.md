# Kairos desktop application

Kairos opens with a black screen that types its name, then reveals the pulse optimization interface.

Launch the software UI from this directory:

```bash
python3 optimization_ui.py
```

The launcher uses the existing project `.venv` when available. Tk support is
required (on Debian/Ubuntu, the `python3-tk` system package).
Enter the same seven physical inputs as the CLI, including optional explicit
units. **NSGA-III is unchecked by default and does not run unless selected.**
GRAPE has a separate checkbox. Gaussian, DRAG, higher-order DRAG (Hermite H2), Wah-Wah and composite BB1/CORPSE references are always compared.
Monte Carlo selection and independent validation run last, after the chosen
optimization stages. The UI remains responsive during computation and supports
stopping a run, loading configuration JSON and reopening saved `summary.json`.

Results show the best eligible method found under the configured MC selection
criterion, nominal fidelity, held-out MC mean/P05 fidelity, leakage, and quality
status. A failed target is shown explicitly; finite searches do not establish a
global optimum. Other existing settings (budgets, bounds, tolerances) can be
changed in a configuration JSON and loaded with **Load configuration**.

CLI method switches are also available:

```bash
python3 run_all.py --non-interactive --no-nsga --grape
python3 run_all.py --non-interactive --nsga --no-grape
```

The CLI preserves the existing NSGA-enabled default when no switch is supplied.

# Transmon X gate: NSGA-III and manufacturing robustness

Run everything with:

```bash
python3 run_all.py
```

The launcher asks for the seven fixed physical inputs using Python `input()`:
detuning, anharmonicity, T1(1), T1(2), Tphi(1), Tphi(2), and Tg. Press Enter
to keep a displayed default. Invalid values are requested again. Times are ns;
frequencies are MHz. Prompts appear once, before the tests and optimization.
Use `--non-interactive` to run automatically from configuration/CLI values.
`--show-results` displays saved results without prompting.

The launcher uses the current Python environment, or finds the existing project
`.venv`, runs the tests, then runs **baselines -> optional NSGA-III -> optional GRAPE -> final Monte Carlo
selection -> independent validation**. Each run gets a new timestamped directory under `results/`. The
saved reports cover the PDF pages 8-9 deliverables: the pulse formula and parameters,
gate duration, all nominal six-state fidelity/leakage results, baseline comparison
metrics, public-grid and Monte Carlo statistics (including confidence intervals),
hardware and numerical checks, configuration, and the full technical report text.
CSV samples and plots remain files; JSON and Markdown copies are retained for
reproducibility. Fractions in the detailed terminal tables use 1 = 100%.

Display a completed run immediately, without repeating optimization:

```bash
python3 run_all.py --show-results results/nsga3_strict
```

For a short integration check:

```bash
python3 run_all.py --quick
```

The quick budget checks execution only; it is not evidence of optimization
convergence. You can choose `--config challenge_config.json` and
`--output results/my_new_run`. Existing results are not overwritten.

## Fixed physical inputs

Set `device.detuning_mhz`, `device.alpha_mhz`, `device.t1_1_ns`,
`device.t1_2_ns`, `device.tphi_1_ns`, `device.tphi_2_ns`, and
`gate_duration_ns` in `challenge_config.json`, or override them on the command line:

```bash
python3 run_all.py --detuning-mhz 0 --alpha-mhz -220 \
  --t1-1-ns 30000 --t1-2-ns 18000 \
  --tphi-1-ns 40000 --tphi-2-ns 25000 --tg-ns 15.6
```

Frequencies are Delta/(2*pi) and alpha/(2*pi) in MHz. All times are ns
(30 microseconds = 30000 ns). Defaults retain the PDF device and use Tg=15.6 ns.
Tg must be 10-60 ns on the 0.2 ns sample grid; invalid inputs are rejected, never
silently rounded or optimized. No supplied device parameter is optimized.
Only `sigma_fraction` (sigma/Tg), `beta`, `amplitude`, and `phase_ramp_mhz`
are searched, in that order in the four-row `bounds` array. The phase ramp is
an I/Q waveform control; nominal Hamiltonian detuning remains your input.

Manufacturing detuning and anharmonicity errors are added to your nominal
inputs; the four lifetime scales multiply your input lifetimes. Tg is fixed
throughout selection and validation. Primitive references use the requested Tg; composite comparisons report their longer actual durations.
CSV scenario `detuning_mhz` is an additive error; `actual_detuning_mhz` and
`actual_alpha_mhz` give the resulting device values. The public grid uses PDF
offsets around your inputs, matching the PDF absolute grid only at its nominal
values. Some input combinations may not meet the quality targets; the program
reports failure instead of changing Tg or the device to make a pulse pass.
Saved older runs remain viewable with `--show-results`; new optimization runs
require the current four-control configuration format.

## Optimizer and tighter defaults

The workflow uses **pymoo NSGA-III**, with three separate objectives:

1. Mean gate infidelity on the fixed training stress scenarios.
2. Worst gate infidelity on those scenarios.
3. Worst average leakage on those scenarios.

NSGA-III uses normalized reference directions to preserve trade-offs. The
configuration has 55 Das-Dennis directions (`reference_partitions=9`), a
population of 56 and 32 generations for each of two independent seeds, 7 and 17.
This allows up to 3,584 candidate evaluations, versus the original 512-evaluation
NSGA-II search. A shared exact-candidate cache avoids repeated simulations.
Analytic Gaussian/DRAG pulses and a Latin hypercube initialize each search;
the second seed does not start from the first seed's winners.

The implementation includes bounded SBX crossover, polynomial mutation,
elitist reference-direction survival, bound repair and duplicate elimination.
A fixed-reference archive hypervolume and its recent gain are recorded for each
seed. They are diagnostics, not proof of convergence. NSGA-III does not guarantee
better accuracy merely because its version number is higher.

There is no Bayesian optimizer or Bayesian refinement stage. `robust_optimization.py`
is the coordinator called by `run_all.py`. `hybrid_optimization.py` is only a
compatibility entry point for old commands.

## Quality targets

The new configurable engineering targets in `quality_targets` are:

| Metric | Target |
|---|---:|
| Nominal six-state X-gate fidelity | >= 99.9% |
| Nominal and maximum sampled leakage | <= 0.01% |
| Validation 5th-percentile fidelity, one-sided 95% lower confidence bound | >= 99.5% |
| Difference under tighter solver settings | <= 0.000002 absolute |

These targets are stricter than the PDF's minimum nominal fidelity 99.0% and
leakage 2.0%. Candidates that fail nominal/hardware constraints are excluded
from selection. If none is eligible, comparisons are saved as diagnostics with quality FAIL. If a
selected pulse fails independent validation, results are still saved but marked
**FAIL**, not reported as an accepted design.

Default integration tolerances are `atol=1e-10`, `rtol=1e-8`, `max_step=dt/2`.
Strict rechecks use `atol=1e-12`, `rtol=1e-10`, `max_step=dt/4`. The waveform grid
remains the PDF's required **0.2 ns**; solver steps and pulse sampling are different.

## Manufacturing tolerance selection

Optional NSGA-III produces candidate pulses using 10 fixed stress scenarios.
Optional GRAPE then refines the best training candidate. Up to 12 search finalists,
plus all six analytic references and the GRAPE proposal, then see the **same 1,024
manufacturing samples**.
The final pulse maximizes **5th-percentile fidelity** (`p05_fidelity`), favoring
performance toward the weaker end of the device distribution. It chooses a fixed
pulse for the range, not the most favorable manufactured-device realization.

After freezing that choice, **2,048 independent validation samples** assess it.
Selection and validation seeds are distinct (2028 and 2029). All comparison
pulses see identical validation devices. Selection-set statistics are descriptive;
the independent validation set supplies the reported performance estimates.

`manufacturing_ranges` controls uniform variations around nominal:

- `amplitude_fraction`: +/-3% by default around scale 1.
- `detuning_mhz`: +/-3 MHz around the supplied nominal detuning.
- `alpha_shift_mhz`: +/-8 MHz around nominal anharmonicity.
- `coherence_min`: four independent lifetime scales in [0.8,1.0] by default.

Independent uniform distributions and 80-100% lifetime scales are assumptions;
the PDF does not specify probability laws or a numerical coherence-degradation
range. Modify these values to represent your expected fabrication and calibration
tolerances. Training scenarios remain the PDF stress cases; Monte Carlo selection
and validation use the configured ranges. Errors are constant during each gate.

The P05 lower confidence bound uses a binomial order statistic. With too few
samples it returns zero. It bounds the distribution's 5th percentile under iid
sampling, not the fidelity of every possible device. Sampled worst cases are
not guaranteed worst-case bounds. Selection scores that are close can change
order on independent samples.

## Physical model

The three-level Hamiltonian uses the PDF's positive detuning sign and your
nominal inputs (default alpha/2pi = -220 MHz). We follow the detailed **page-5 collapse operators**:
`sqrt(1/T1_j)` for each separate relaxation transition and `sqrt(2/Tphi_j)` for
excited-level projectors. **Page 2 has conflicting factors**; the report documents
this modeling choice.

`challenge_model.hamiltonian` explicitly implements PDF page 4:
`a=|0><1|+sqrt(2)|1><2|`, `n=a†a`,
`H0=Delta*n+alpha*n*(n-1)/2`, and
`Hctrl=I(t)*(a+a†)/2-i*Q(t)*(a-a†)/2`.
The basis is `|0>, |1>, |2>`; units are rad/ns with hbar=1.
The full Hamiltonian matrix and nominal H0 diagonal print in the terminal.
All optimizer, selection, validation and dynamics plots use this same function.

Fidelity is averaged over `|0>, |1>, |+>, |->, |+i>, |-i>` against X-gate targets.
Leaked output is not renormalized and no free virtual-Z correction is fitted.
The shared pulse family is a sin-squared-tapered Gaussian with analytic DRAG
quadrature and optional I/Q phase ramp. Both I/Q endpoints are zero. Duration is
a fixed input in 10-60 ns, validated on the 0.2 ns grid. Linear sample interpolation prevents amplitude
overshoot. The configured 1.03 training headroom covers +3% amplitude error; final
selection also checks the complete configured amplitude range. Peak slew is
reported because the PDF specifies no numerical slew/bandwidth limit.

## Outputs and files

| File | Purpose |
|---|---|
| `run_all.py` | One-command launcher and checks |
| `robust_optimization.py` | NSGA-III runs, robust selection, validation and exports |
| `nsga3.py` | pymoo adapter, reference directions and hypervolume diagnostics |
| `challenge_model.py` | PDF model, channel fidelity and perturbed-device sampling |
| `pulse_shapes.py` | Shared Gaussian/DRAG pulse construction |
| `challenge_report.py` | Scientific plots and technical report |
| `challenge_config.json` | Physical settings, budgets, tolerances and quality targets |
| `test_challenge.py` | Physics, gate-phase/leakage, DTLZ2/ZDT1 and statistics checks |

Each results directory contains `pulse.csv`, `selected_pulse.json`, `summary.json`,
`config.json`, `reference_directions.csv`, per-seed histories, NSGA/optimization
archives, the Pareto set, Monte Carlo rankings, exact sampled device parameters,
public-grid and validation CSVs, three plot panels and `report.md`.
`summary.json` includes source hashes, dependency versions and quality status.
All comparisons use the PDF model with your supplied nominal device inputs.
Old results directories are historical records and are not rewritten.

Install dependencies in a separate environment with:

```bash
python -m pip install -r requirements.txt
```

The existing `core.py`, `pulse-shaping.py` and `analysis.py` remain optional
original-device diagnostic dashboards. Their calibration is now deterministic
grid search (`calibration.py`); they do not run during the PDF challenge workflow.
Old configurations containing Bayesian budgets should be replaced using the
current `challenge_config.json` template.

Method references: [pymoo NSGA-III](https://pymoo.org/algorithms/moo/nsga3.html),
[QuTiP channel propagation](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-propagator.html).

## DRAG reference and noise model

The derivative control uses Eq. (14) of the supplied `1008.2554v1.pdf`:
`Q=-beta*dI/dt/alpha_nominal`. The paper's anharmonicity Delta is our alpha;
its dimensionless alpha is our beta. For the challenge's fixed coupling
lambda=sqrt(2), the reference beta=lambda^2/4 is 0.5. The DRAG baseline uses
0.5, and NSGA-III optimizes beta along with the other pulse controls. The
challenge's smooth taper is retained; Eq. (13)'s full cubic/chirped correction
is not claimed. The optional constant phase ramp remains a fitted control.

Noise uses only your four T1/Tphi inputs through the challenge page-5 Lindblad
operators. Monte Carlo scales these lifetimes for manufacturing robustness.
No explicit bosonic/Ohmic environment, temperature, cutoff or f01 is needed.

Final output also includes envelope A and B. The applied tapered envelope has
B=0 and A equal to its sampled normalization coefficient, including amplitude
scale. Eq. (12) shifted-Gaussian references are printed separately for the same
optimized sigma and for the paper's sigma=Tg/2, pi-area choice. With the
convention I=A*(exp(...)-B), B is dimensionless; the equivalent subtractive
offset A*B is also printed in rad/ns. Reference coefficients do not change the
exported pulse. Detailed results and explanations are retained in the saved reports.

If no candidate meets the fixed inputs and strict targets, the final comparison
is saved with quality FAIL and the pulse is labeled diagnostic. The process exits
with status 2. It does not relax the targets or claim the design is accepted.

Input prompts accept explicit units: `300000 Hz` becomes `0.3 MHz`, `-0.18 GHz`
becomes `-180 MHz`, and `20 us` becomes `20000 ns`. Bare numbers still mean
MHz for frequencies and ns for times. Each parsed value is echoed before
optimization. Large bare detuning entries require explicit units to help catch
unit mistakes. Numerical integration failures are reported separately from
quality failures, with `solver_failure.json` and exit status 3.

## Optional GRAPE before final Monte Carlo

GRAPE runs only when enabled. It refines the best training candidate from the
analytic baselines and optional NSGA search. Its Fourier controls, analytic
Frechet/adjoint gradients, constrained SLSQP and multistart optimizer are retained.
The ensemble now consists of a deterministic amplitude/detuning grid and fixed
stress scenarios. No Monte Carlo samples are generated before refinement finishes.
The legacy GRAPE `samples` and `seed` settings are retained for configuration
compatibility; they no longer generate a random training ensemble.

After the GRAPE decision and refinement, the completed candidate waveforms enter
final Monte Carlo selection. All six analytic references and the GRAPE proposal are included
alongside the numerical finalists. Eligibility still requires nominal fidelity,
leakage and amplitude limits. The configured MC fidelity statistic selects the
winner; separate held-out MC samples assess it without changing the selection.
This ranking replaces the old post-MC GRAPE non-regression acceptance rule.
When no candidate is eligible, reports and comparisons are still saved with
quality FAIL and the exported pulse clearly marked as a diagnostic candidate.

`pulse.csv` is authoritative. If GRAPE is selected, analytic Gaussian/DRAG
parameters and A/B describe its starting seed, not the final Fourier waveform.
GRAPE history, waveform and gradient plots are retained. Quick mode uses a
reduced search/refinement budget and is not convergence evidence.

## Focused final comparisons

The terminal now prints a compact stage table (nominal and held-out MC
infidelity/leakage, MC P05 fidelity), final parameters and A/B, GRAPE acceptance,
quality status and final detuning table. Detailed explanations and all numerical
results remain in the saved reports and JSON/CSV files.

`stage_comparisons.md` contains the four requested comparison groups:
1. Infidelity/leakage for Gaussian, DRAG, Hermite DRAG, Wah-Wah, composite BB1/CORPSE, optional NSGA-III and optional GRAPE proposal
   and the final waveform (`stage_metrics.png`), marked diagnostic if quality fails.
2. I/Q pulse shapes for every stage (`pulse_stages.png` and per-stage pulse CSVs).
3. NSGA convergence, MC candidate ranking and GRAPE loss (`optimizer_stages.png`).
4. Fidelity and leakage versus detuning at nominal amplitude/alpha/coherence,
   using the five PDF detuning-grid points (`detuning_effect.png`).

A rejected GRAPE proposal is explicitly labeled and still compared; it never
replaces the final pulse after held-out validation. Identical waveforms share
cached validation results on exactly the same device samples.

Pulse comparison uses shared I and Q axes with every stage overlaid and distinct
line/marker styles. `pulse_comparison.csv` contains a single shared time column
and I/Q columns for all stages in MHz (drive divided by 2pi). Coincident curves
can be identical, especially when a GRAPE proposal was rejected.

The custom GRAPE implementation uses constrained SLSQP on Fourier coefficients,
with analytic Frechet/adjoint derivatives and a dissipative ensemble objective.
Its mean error/leakage objective (with optional tail weighting) is a surrogate
for the final MC selection criterion; lower training loss need not improve P05.
Tests compare directional derivatives to finite differences and its approximate
propagation to an independent high-accuracy QuTiP Lindblad solve. This validates
the tested gradients/dynamics, not global convergence or universal improvement.

`pulse_changes.png` magnifies Delta I and Delta Q relative to the pre-GRAPE
seed; the common comparison table also reports peak/RMS changes in MHz. A
rejected proposal can differ from the seed while the final curve is identical.

Every completed run also saves `pulse_diagnostics.md`, `rabi_chevron.png` and
`pulse_i_plus_q.png`, with CSV data. The chevron sweeps actual detuning and
0–4 Tg pulse duration from |0>, stretching the final I/Q envelopes at fixed
amplitude with the same T1/Tphi noise. These diagnostic pulses are not
reoptimized gates. The combined plot shows final I, Q and envelope magnitude,
plus the literal I + Q sum for all stages on shared axes.
To generate these plots for an existing result without rerunning optimization:

```bash
python pulse_diagnostics.py results/<run-directory>
```

GRAPE also saves `grape_gradients.png` and `grape_gradients.csv`: the analytic
training-loss gradient norm and signed derivatives for every Fourier a/b
coefficient, recorded at each optimizer callback with restart boundaries.
These are objective gradients, not constrained optimality residuals. Older
runs without recorded gradients require a new optimization to produce this plot.


## Selected pulse parameters and additional families

The **Selected pulse** tab displays amplitude, sigma, beta and the controls that
actually define the selected family, with units and a copy button. Each new run
also saves `selected_pulse_parameters.json` (full numeric precision) and
`selected_pulse_parameters.md`. The same parameters print in the terminal and
appear in the technical report. `pulse.csv` remains the authoritative waveform.
The JSON can regenerate analytic/Fourier samples:

```python
import json
from pulse_methods import waveform_from_spec
spec = json.load(open("results/<run>/selected_pulse_parameters.json"))
time, pulse_i, pulse_q = waveform_from_spec(spec)
```

- **Gaussian / DRAG:** total duration, sample interval, sigma and sigma/Tg,
  dimensionless amplitude scale, envelope normalization A (rad/ns and MHz), B,
  DRAG beta, nominal alpha, phase ramp and peak combined I/Q amplitude.
- **Higher-order DRAG:** the second-order Hermite envelope correction H2, as
  defined by [Rigetti's HRM Gaussian](https://pyquil-docs.rigetti.com/en/stable/quilt_waveforms.html#hrm-gaussian),
  with the project's sin-squared taper, sampled pi-area normalization and
  analytic derivative quadrature. This is a specific higher-order envelope
  variant, not the fifth-order/chirped DRAG expansion. H2 defaults to 0.5.
- **Wah-Wah:** sideband depth and frequency, in addition to the Gaussian/DRAG
  controls. The modulation is `1-depth*cos(2*pi*f_mhz*0.001*(t-T/2))`, following
  [Vesterinen et al., Eqs. (1)-(2)](https://arxiv.org/abs/1405.0450), with the
  existing taper. Defaults are depth 0.3 and 25 MHz; these are reference settings,
  not fitted device-specific values. The present one-transmon model can compare
  local gate fidelity/leakage only, not protection of neighboring qubits.
- **Composite BB1:** four chronological segments `(pi,0), (pi,phi),
  (2*pi,3*phi), (pi,phi)`, with `phi=acos(-1/4)`, using the
  [BB1 rotation sequence](https://pmc.ncbi.nlm.nih.gov/articles/PMC7005697/).
  Each segment uses an area-normalized tapered Gaussian with local DRAG
  quadrature, rotated by its phase. Its duration is at least requested Tg;
  larger rotation angles get proportionally longer durations. Durations extend
  further if needed to satisfy hardware amplitude and manufacturing headroom.
- **Composite CORPSE:** chronological rotations `(420 degrees,0)`,
  `(300 degrees,180 degrees)`, `(60 degrees,0)`, following the
  [canonical CORPSE sequence](https://www.nature.com/articles/s41598-020-58823-9).
  Uses the same DRAG-shaped segment construction and duration policy as BB1.
  These are shaped adaptations: the ideal square-pulse detuning cancellation
  guarantee does not automatically apply. Simulated fidelity is authoritative.
  Both composites export every segment's duration, sigma, beta, phase, angle
  and peak amplitude, plus exact reconstructible samples.
- **GRAPE:** final Fourier amplitude scale and every I/Q coefficient. Sigma and
  beta are shown only under the starting seed, since they do not define the
  final Fourier pulse. The actual seed may now be any eligible reference or
  numerical-search pulse.

All six references use the same device and sample interval. Composite pulses
have explicitly reported longer durations and include decoherence throughout
those durations. They are **comparison only** for a fixed-Tg task: neither
selection (including diagnostic fallback) nor the GRAPE seed can use them.
Other methods keep the requested total gate duration. Common Monte Carlo
scenarios are shared across methods, after optional optimizers finish.
Combined waveform CSVs zero-pad shorter pulses for display only; each physical
simulation uses its native duration. Legacy saved BB1 recipes still reconstruct
with their original raised-cosine shape; rerun to obtain corrected comparisons.

The comparison table includes selection eligibility. Click a row to see why it
was excluded (nominal fidelity, leakage or hardware amplitude). Selection and
held-out validation still use distinct common Monte Carlo devices, after all
pulse construction and optional optimizers. A good validation number cannot
retroactively select a rejected pulse. Existing saved runs remain historical;
start a new run to obtain the additional method comparisons.

## Appearance and font

Kairos uses a dark charcoal theme with gold accents and the supplied logo in
the header, window icon, and animated startup screen. Keep the `assets/` folder
with the Python files. The startup wordmark uses original dot-matrix lettering drawn directly in Tk,
so it needs no external font file or installation. Only the startup uses this
style; the main interface uses a readable sans-serif (Noto Sans, DejaVu Sans,
or Arial depending on availability).

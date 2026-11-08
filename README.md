# Kairos — Robust Quantum Optimal Control for Superconducting Qubits

Kairos optimizes superconducting transmon X gates under uncertainty. It provides conventional GRAPE, fixed Risk-Aware GRAPE, CVaR continuation, calibrated DRAG references, Fourier controls, five-level Lindblad propagation, Monte Carlo robustness analysis, optional JAX acceleration, and statistical fidelity certification. CPU QuTiP remains the authoritative production solver.

**Public release is blocked pending an owner-approved license and citation metadata.** This is a prepared research release, not a claim of journal publication.

## Physical model

The confirmatory study uses a five-level Duffing oscillator in the rotating-wave approximation with Lindblad relaxation/dephasing, 15.6 ns gates and an 80 MHz joint I/Q amplitude constraint including 1.03 manufacturing headroom. Fidelity and leakage average six computational input states without renormalization. The API retains legacy three-level defaults: explicitly use `Device(levels=5)` for this study. Uncertainty is quasi-static; model details and exact distributions are frozen in the protocol.

## Installation and CPU examples

Linux and Python 3.12 are the tested environment. Other platforms and Python versions are unverified. From this repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kairos-mpl
python examples/demo.py simulate
python examples/demo.py evaluate
python examples/demo.py grape
python examples/demo.py risk
python examples/demo.py continuation
python examples/demo.py robustness
python -m unittest discover -s tests -v
(cd reproducibility/original && python -m unittest discover -p 'test_*.py')
python scripts/verify_release.py
```

Examples are intentionally tiny demonstrations, not converged research optimizations or published results. They reuse a frozen starting pulse and a separate demonstration seed. They never regenerate the independent cohort.

For NVIDIA GPUs on Linux, install `python -m pip install -r requirements-gpu.txt` into a separate environment after CPU installation. See [GPU guidance](docs/JAX_ACCELERATION.md). Set the optimizer's `acceleration` options to `backend: jax_gpu, precision: float64`; retain accuracy qualification and CPU fallback. Ordinary optional execution falls back to CPU when GPU initialization fails; strict research settings fail closed. No GPU is required for CPU use. RTX 2080 Super FP64 was tested historically; FP32 failed the frozen numerical tolerance and is not qualified. Other hardware is unverified. Historical speedups depend on batch size and workload; they are not universal guarantees.

## Independent confirmatory results

Exactly 2,000 devices and 10,000 candidate-device production evaluations were completed using five fixed waveforms, including one ineligible diagnostic. The four primary candidates were:

| Method | Mean fidelity | P05 fidelity | CVaR95 infidelity | Engineering |
|---|---:|---:|---:|---|
| Conventional GRAPE | 0.998653219 | 0.997398629 | 0.0030237417 | PASS |
| CVaR continuation | 0.998626354 | 0.997388012 | 0.00303112568 | PASS |
| Fixed Risk-Aware GRAPE | 0.998498366 | 0.997058879 | 0.00338305312 | PASS |
| Calibrated DRAG | 0.998480563 | 0.997004501 | 0.00340873105 | PASS |

Conventional GRAPE had the lowest observed CVaR95. Continuation did not demonstrate statistically significant superiority over conventional GRAPE; this does not establish equivalence. Both beat calibrated DRAG in prespecified independent comparisons. Fixed-risk GRAPE performed worse than conventional GRAPE here, not universally. All four eligible frozen pulses passed engineering certification. The excluded diagnostic violated leakage on 416/2,000 devices and remains reported.

The overall reporting convention returned **PARTIALLY CONFIRMED** (four of six qualified favorable contrasts). Individual prespecified contrasts remain authoritative. Statistical significance, engineering certification, numerical reliability and publication readiness are separate assessments. See [confirmatory report](results/confirmatory/confirmatory_validation_report.md) and [six corrected intervals](results/confirmatory/bootstrap_confidence_intervals.csv).

The original 27-trial development study had nonbinding resource limits and identical waveforms across budget regimes. The separate 54-trial corrected study used actual resource limits. Neither establishes universal efficiency superiority. The earlier approximately 21% exploratory improvement over conventional GRAPE was not independently confirmed for the selected pulses. Read [results summary](docs/results_summary.md).

## Scientific limitations

These are simulated devices, not hardware measurements. Duffing/RWA truncation, Markovian Lindblad noise and quasi-static uncertainty are model assumptions. Monte Carlo samples and numerical repeat checks are finite. Development optimizations were budget-limited rather than convergence-certified. Computational costs include different kinds of solver work; cost comparisons require genuinely matched budgets. Fixed-pulse validation is not an independent replication of aggregate optimizer performance.

## Reproducibility

[Reproducibility status](docs/reproducibility_status.md) explains preserved source, frozen hashes, protocol, seeds and checkpoints. Original relative paths map to `reproducibility/original/`; immutable files keep their original bytes. The installable package qualifies internal imports and has its own hashes. The confirmatory cohort seed 941206 has already been used; bootstrap seed is 941207. Do not redraw the cohort, rerun the campaign into a fresh output, or tune using validation outcomes. Existing records support read-only verification without expensive recomputation.

## Citation and licensing

Use [CITATION.cff](CITATION.cff) with the exact version/commit used. Author metadata and a permanent identifier await owner confirmation; no DOI or published paper is claimed. See [licensing and citation status](docs/licensing_and_citation.md) before distribution. Dependencies retain their own licenses.

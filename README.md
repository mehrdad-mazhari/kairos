# Kairos

A Python desktop application for simulating and comparing transmon X-gate
pulses, with optional NSGA-III optimization and GRAPE refinement.

Compare Gaussian, DRAG, higher-order DRAG, Wah-Wah, BB1 and CORPSE pulses.
Results include nominal fidelity, independent Monte Carlo fidelity, leakage,
selection eligibility, and reproducible pulse parameters and waveforms.
Monte Carlo selection and validation run after the enabled optimizers.

On launch, a black startup screen types `kairos` before the main interface appears.

## Install and launch

Tested with Python 3.12 on Linux. Open a terminal in this folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python optimization_ui.py
```

On Ubuntu/Debian, if virtual environments or Tkinter are missing, install
`python3-venv` and `python3-tk` using your system package manager first.
On Windows, use `py -3.12 -m venv .venv` and activate with
`.venv\Scripts\activate` in Command Prompt, then run the same pip and launch
commands. Windows has not been verified in this release.

Enter the device parameters, choose whether to enable NSGA-III and GRAPE,
and click **Run optimization**. NSGA-III is unchecked by default in the UI.
Generated results are saved in `results/` and are excluded from version control.
Full optimization can take substantial time. Quick mode is an execution check,
not evidence that the physical quality targets are satisfied.

## Verify the code

```bash
python -m unittest test_challenge test_workflow test_pulse_methods
```

## Understanding the comparison

BB1 and CORPSE use DRAG-shaped segments and explicitly longer durations.
They are comparison-only for a fixed-duration task, not eligible winners or
GRAPE seeds. Their simulated fidelities include their full actual durations.
The shaped CORPSE implementation is an adaptation of the canonical rotation
sequence, not a guarantee of ideal square-pulse error cancellation.
The program reports the best eligible candidate it evaluated, not a proven
global optimum. This is simulation software, not experimental validation.

## Files

- `optimization_ui.py`: desktop application.
- `run_all.py`: command-line launcher and test runner.
- `robust_optimization.py`, `nsga3.py`, `grape.py`: optimization and selection.
- `challenge_model.py`, `pulse_methods.py`, `pulse_shapes.py`: physical model and pulses.
- `challenge_config.json`: default device, optimization and quality settings.
- `test_*.py`: physics, pulse reconstruction and workflow tests.
- `core.py`, `analysis.py`, `calibration.py`, `pulse-shaping.py`: original exploratory scripts.
- [Technical guide](TECHNICAL_GUIDE.md): detailed model, controls, outputs and references.
- [First GitHub upload](GITHUB_FIRST_UPLOAD.md): beginner instructions.

No project license has been selected in this prepared copy.


## Appearance and font

Kairos uses a dark charcoal theme with gold accents and the supplied logo in
the header, window icon, and animated startup screen. Keep the `assets/` folder
with the Python files. The startup wordmark uses original dot-matrix lettering drawn directly in Tk,
so it needs no external font file or installation. Only the startup uses this
style; the main interface uses a readable sans-serif (Noto Sans, DejaVu Sans,
or Arial depending on availability).


---

> **Note:** This code was originally written for a hackathon — the UI was added later. I'm happy to fix any issues, so feel free to contact me at **mehrdadm@tutamail.com**.

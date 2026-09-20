#!/usr/bin/env python3
"""One command for validation, NSGA-III pulse optimization, plots and robustness."""

import argparse
import json
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small end-to-end check instead of the full search")
    parser.add_argument("--config", type=Path, default=project / "challenge_config.json")
    parser.add_argument("--output", type=Path, help="new results directory; defaults to a timestamped project folder")
    parser.add_argument("--show-results", type=Path,
                        help="print all results from an existing directory without rerunning")
    from optimization_inputs import add_input_arguments, input_command_arguments, prompt_input_arguments
    add_input_arguments(parser)
    args = parser.parse_args()
    if args.show_results:
        if args.quick or args.output or input_command_arguments(args):
            parser.error("--show-results cannot be combined with --quick, --output or physical input overrides")
        from terminal_report import show_saved_results
        try:
            show_saved_results(args.show_results)
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"Cannot display saved results: {error}")
        return 0
    config = args.config.resolve()
    if not config.is_file():
        parser.error(f"Configuration file not found: {config}")
    output = (args.output.resolve() if args.output else
              project / "results" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be a new or empty directory: {output}")

    try:
        prompt_input_arguments(json.loads(config.read_text()), args)
    except EOFError:
        parser.error("Input ended before completion. Supply all seven answers or use --non-interactive.")
    except KeyboardInterrupt:
        print("\nInput cancelled.", file=sys.stderr)
        return 130

    # Honor an activated Python environment. Otherwise find the existing project
    # environment so ./run_all.py works with the system's Python launcher too.
    python = Path(sys.executable)
    if sys.prefix == sys.base_prefix:
        for candidate in [project / ".venv/bin/python", project.parent / ".venv/bin/python"]:
            if candidate.is_file():
                python = candidate
                break

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["OPENBLAS_NUM_THREADS"] = "1"
    print(f"Python: {python}\nResults: {output}", flush=True)
    with tempfile.TemporaryDirectory(prefix="transmon-mpl-") as cache:
        env["MPLCONFIGDIR"] = cache
        try:
            print("Checking physics and optimizers...", flush=True)
            subprocess.run([str(python), "-m", "unittest", "test_challenge.py", "test_workflow.py", "test_pulse_methods.py"],
                           cwd=project, env=env, check=True)
            command = [str(python), "-u", str(project / "robust_optimization.py"),
                       "--config", str(config), "--output", str(output), "--non-interactive"]
            command.extend(input_command_arguments(args))
            if args.quick:
                command.append("--quick")
            print("Running baseline comparison, optional NSGA-III, "
                  "optional GRAPE, then final Monte Carlo selection and validation...", flush=True)
            subprocess.run(command, cwd=project, env=env, check=True)
        except subprocess.CalledProcessError as error:
            if error.returncode == 3:
                print("Stopped because numerical integration failed. See the input/solver diagnostics above.", file=sys.stderr)
                return 3
            if error.returncode == 2:
                print("No accepted design was produced. See the quality/diagnostic report above.", file=sys.stderr)
                return 2
            print(f"Stopped: a required step failed (exit {error.returncode}). "
                  "See the error above. Dependencies are listed in requirements.txt.", file=sys.stderr)
            return error.returncode
        except KeyboardInterrupt:
            print("Run interrupted.", file=sys.stderr)
            return 130
    print(f"Final waveform: {output / 'pulse.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

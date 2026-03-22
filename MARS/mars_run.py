#!/usr/bin/env python3
"""
MARS Run — Orchestrator Script

Reads mars_params.json, calls mars_numerics.py via subprocess,
then calls mars_latex.py to generate mars_values.tex.

Usage:
    python mars_run.py

Workflow:
    1. Read mars_params.json
    2. Build CLI argument list from JSON values
    3. Call mars_numerics.py via subprocess
    4. Call mars_latex.generate() to create mars_values.tex
"""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Get script directory (for relative paths)
    script_dir = Path(__file__).parent

    params_file = script_dir / "mars_params.json"
    numerics_script = script_dir / "mars_numerics.py"

    # 1. Read parameters
    if not params_file.exists():
        print(f"Error: {params_file} not found")
        sys.exit(1)

    with open(params_file) as f:
        params = json.load(f)

    fixed = params.get("fixed", {})
    design = params.get("design", {})

    # 2. Build CLI args for mars_numerics.py
    args = [
        sys.executable, str(numerics_script),
        f"--n_arms={design.get('n_arms', 2)}",
        f"--R_blade_m={design.get('R_blade_m', 1.6)}",
        f"--chord_m={design.get('chord_m', 0.16)}",
        f"--m_blade_kg={design.get('m_blade_kg', 0.25)}",
        f"--m_plate_kg={design.get('m_plate_kg', 0.25)}",
        f"--tau_motor_Nm={design.get('tau_motor_Nm', 60.0)}",
        f"--delta_r_m={design.get('delta_r_m', 0.8)}",
        f"--r0_carrier_m={design.get('r0_carrier_m', 0.8)}",
        f"--z_lift_mm={design.get('z_lift_mm', 5.0)}",
        f"--t_lift_s={design.get('t_lift_s', 0.5)}",
        f"--F_actuator_N={design.get('F_actuator_N', 20.0)}",
        f"--u_c_mm_s={fixed.get('u_c_mm_s', 0.1)}",
        f"--output={script_dir / 'mars_results.json'}",
    ]

    # 3. Run mars_numerics.py
    result = subprocess.run(args, cwd=script_dir)
    if result.returncode != 0:
        print(f"Error: mars_numerics.py failed with code {result.returncode}")
        sys.exit(result.returncode)

    # 4. Generate LaTeX
    print("Generating LaTeX macros...")
    import mars_latex
    mars_latex.generate(
        params_file=str(params_file),
        results_file=str(script_dir / "mars_results.json"),
        output_file=str(script_dir / "mars_values.tex"),
    )

    print()
    print("Done. Files generated:")
    print(f"  - {script_dir / 'mars_results.json'}")
    print(f"  - {script_dir / 'mars_values.tex'}")


if __name__ == "__main__":
    main()

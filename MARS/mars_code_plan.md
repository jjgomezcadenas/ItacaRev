# MARS Code Architecture Plan

## Overview

Three Python scripts with clear separation of concerns:

| Script | Purpose | I/O |
|--------|---------|-----|
| `mars_numerics.py` | Standalone computation | CLI args → terminal + JSON |
| `mars_run.py` | Orchestrator | JSON → subprocess → JSON + LaTeX |
| `mars_latex.py` | LaTeX generator (internal) | JSON → `.tex` |

---

## File Structure

```
MARS/
├── mars_params_defs.json    # Reference defaults (never modified)
├── mars_params.json         # Input parameters (edit to change)
├── mars_results.json        # Output from mars_numerics.py
├── mars_values.tex          # LaTeX macros (generated)
├── mars_numerics.py         # Standalone computation script
├── mars_run.py              # Orchestrator
└── mars_latex.py            # Internal LaTeX generator module
```

---

## `mars_numerics.py` — Standalone Computation

### Usage

```bash
# With defaults:
python mars_numerics.py

# With CLI overrides:
python mars_numerics.py --n_arms 8 --z_lift_mm 6.0 --t_lift_s 0.4
```

### CLI Arguments (all with hardwired defaults)

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_arms` | 2 | Number of arms |
| `--R_blade_m` | 1.6 | Blade length [m] |
| `--chord_m` | 0.16 | Blade chord [m] |
| `--m_blade_kg` | 0.25 | Blade mass [kg] |
| `--m_plate_kg` | 0.25 | Carrier plate mass [kg] |
| `--tau_motor_Nm` | 60.0 | Motor torque [N·m] |
| `--delta_r_m` | 0.8 | Carrier slide distance [m] |
| `--r0_carrier_m` | 0.8 | Initial carrier position [m] |
| `--z_lift_mm` | 5.0 | Lift height [mm] |
| `--t_lift_s` | 0.5 | Lift duration [s] |
| `--F_actuator_N` | 20.0 | Actuator force [N] |
| `--u_c_mm_s` | 0.1 | Quiescence cutoff velocity [mm/s] |
| `--output` | `mars_results.json` | Output JSON file |

### Fixed Constants (hardwired, no CLI)

```python
P_bar = 15.0          # Pressure [bar]
T_K = 300.0           # Temperature [K]
gamma_Xe = 5/3        # Adiabatic index
R_cyl_m = 1.6         # Vessel radius [m]
H_cyl_m = 1.5         # Vessel height [m]
L_drift_m = 1.5       # Drift length [m]
v_drift_mm_s = 100.0  # Ion drift velocity [mm/s]
sigma_diff_mm = 1.0   # Diffusion sigma [mm]
```

### Output

**Terminal:** Clean, readable summary:
```
════════════════════════════════════════════════════════════════════
MARS FLOW ANALYSIS — n = 2 arms
════════════════════════════════════════════════════════════════════

GAS PROPERTIES (Xe at 15 bar, 300 K)
  ρ = 79.0 kg/m³    ν = 2.94×10⁻⁷ m²/s    c_s = 178 m/s

PHASE 1: BLADE ROTATION (180°)
  t_rot = 395 ms    V_tip = 25.4 m/s
  Ma = 0.14 (incompressible)    Re = 1.4×10⁷ (turbulent)
  δ_blade = 2.21 mm    max_reach = 3.25 mm

PHASE 2: CARRIER SLIDE (0.8 m)
  t_slide = 200 ms    V_slide = 8.0 m/s
  δ_carrier = 2.78 mm    max_reach = 4.00 mm

PHASE 3: VERTICAL LIFT (5 mm)
  t_lift = 500 ms    V_lift = 10 mm/s
  Re_lift = 330 (laminar)    δ_Stokes = 0.39 mm

SUMMARY
  t_total = 1.10 s
  max_reach (worst case) = 4.00 mm
  z_lift = 5.00 mm    margin = 1.00 mm ✓
  dead_zone = 110 mm (7.3% of drift)
```

**JSON:** `mars_results.json` with all computed quantities:
- Gas properties (ρ, ν, c_s)
- Phase 1: kinematics, flow regime, BL, eddy diffusion, max_reach
- Phase 2: kinematics, BL, max_reach
- Phase 3: kinematics, Stokes layer, Re_lift
- 6 mechanisms: key parameters for each (for LaTeX documentation)
- Summary: t_total, max_reach, margin, dead_zone

### Computations

1. **Gas properties**: ρ, ν, c_s from P, T, γ
2. **Phase 1 (rotation)**:
   - sweep_angle = 360° / n_arms
   - Bang-bang kinematics: I_total, t_rot, ω_max, V_tip
   - Flow regime: Ma, Re
   - Boundary layer: δ, θ, δ*, Cf, u_τ
   - Eddy diffusion: ν_t, t_decay, ℓ_turb, ℓ_mol, max_reach
3. **Phase 2 (slide)**:
   - Bang-bang kinematics: t_slide, V_slide
   - Boundary layer: δ_carrier
   - Eddy diffusion: max_reach_carrier
4. **Phase 3 (lift)**:
   - Kinematics: V_lift = 2 × z_lift / t_lift (bang-bang peak)
   - Re_lift (confirm laminar)
   - Stokes layer: δ_Stokes = √(ν × t_lift)
5. **6 Mechanisms** (all zero effect, but compute for documentation):
   - M1: BL (max_reach, t_decay)
   - M2: Bulk swirl (τ_drag, δ_Ekman, t_Ekman_decay)
   - M3: Potential flow (t_acoustic)
   - M4: Pressure pulse (Δp, t_traverse)
   - M5: Secondary flows (a_cent)
   - M6: Vortex shedding (St, f_shed)
6. **Summary**:
   - t_total = t_rot + t_slide + t_lift
   - max_reach = max(max_reach_blade, max_reach_carrier)
   - margin = z_lift - max_reach
   - dead_zone = v_drift × t_total

---

## `mars_run.py` — Orchestrator

### Usage

```bash
python mars_run.py
```

### Workflow

1. Read `mars_params.json`
2. Build CLI argument list from JSON values
3. Call `mars_numerics.py` via subprocess with those arguments
4. Call `mars_latex.py` (as module) to generate `mars_values.tex`

### Code Structure

```python
#!/usr/bin/env python3
"""Orchestrator: runs mars_numerics.py with params from JSON."""

import json
import subprocess
from pathlib import Path

def main():
    # 1. Read parameters
    with open("mars_params.json") as f:
        params = json.load(f)

    # 2. Build CLI args
    design = params["design"]
    fixed = params["fixed"]

    args = [
        "python", "mars_numerics.py",
        f"--n_arms={design['n_arms']}",
        f"--R_blade_m={design['R_blade_m']}",
        f"--chord_m={design['chord_m']}",
        # ... all design params
        f"--u_c_mm_s={fixed['u_c_mm_s']}",
    ]

    # 3. Run mars_numerics.py
    subprocess.run(args, check=True)

    # 4. Generate LaTeX
    import mars_latex
    mars_latex.generate()

if __name__ == "__main__":
    main()
```

---

## `mars_latex.py` — LaTeX Generator (Internal Module)

### Function

```python
def generate(
    params_file="mars_params.json",
    results_file="mars_results.json",
    output_file="mars_values.tex"
):
    """Generate LaTeX macros from JSON files."""
```

### Output Format

`mars_values.tex`:
```latex
% MARS parameters - auto-generated, do not edit
% Generated by mars_latex.py

% Fixed parameters
\newcommand{\Pbar}{15.0}
\newcommand{\TK}{300}
\newcommand{\Rcyl}{1.6}

% Design parameters
\newcommand{\narms}{2}
\newcommand{\Rblade}{1.6}
\newcommand{\chord}{0.16}
\newcommand{\zlift}{5.0}
\newcommand{\tlift}{0.5}

% Gas properties
\newcommand{\rhoxe}{79.0}
\newcommand{\nuxe}{2.94e-7}
\newcommand{\csound}{178}

% Phase 1 results
\newcommand{\trot}{395}
\newcommand{\Vtip}{25.4}
\newcommand{\Matip}{0.14}
\newcommand{\Retip}{1.4e7}
\newcommand{\deltablade}{2.21}
\newcommand{\maxreachblade}{3.25}

% Phase 2 results
\newcommand{\tslide}{200}
\newcommand{\Vslide}{8.0}
\newcommand{\deltacarrier}{2.78}
\newcommand{\maxreachcarrier}{4.00}

% Phase 3 results
\newcommand{\Vlift}{10}
\newcommand{\Relift}{330}
\newcommand{\deltaStokes}{0.39}

% Summary
\newcommand{\ttotal}{1.10}
\newcommand{\maxreach}{4.00}
\newcommand{\margin}{1.00}
\newcommand{\deadzone}{110}
\newcommand{\deadzonepct}{7.3}
```

---

## `mars_params.json` Structure

```json
{
  "fixed": {
    "P_bar": 15.0,
    "T_K": 300.0,
    "gamma_Xe": 1.667,
    "R_cyl_m": 1.6,
    "H_cyl_m": 1.5,
    "L_drift_m": 1.5,
    "v_drift_mm_s": 100.0,
    "sigma_diff_mm": 1.0,
    "u_c_mm_s": 0.1
  },
  "design": {
    "n_arms": 2,
    "R_blade_m": 1.6,
    "chord_m": 0.16,
    "m_blade_kg": 0.25,
    "m_plate_kg": 0.25,
    "tau_motor_Nm": 60.0,
    "delta_r_m": 0.8,
    "r0_carrier_m": 0.8,
    "z_lift_mm": 5.0,
    "t_lift_s": 0.5,
    "F_actuator_N": 20.0
  }
}
```

---

## `mars_results.json` Structure

```json
{
  "input_params": {
    "n_arms": 2,
    "R_blade_m": 1.6,
    "...": "..."
  },
  "gas_properties": {
    "rho_kg_m3": 79.0,
    "nu_m2_s": 2.94e-7,
    "c_s_m_s": 178.0
  },
  "phase1": {
    "sweep_angle_deg": 180.0,
    "t_rot_s": 0.395,
    "t_rot_ms": 395,
    "omega_max_rad_s": 15.9,
    "V_tip_m_s": 25.4,
    "Ma_tip": 0.14,
    "Re_tip": 1.4e7,
    "delta_mm": 2.21,
    "t_decay_ms": 3.5,
    "max_reach_mm": 3.25
  },
  "phase2": {
    "t_slide_s": 0.2,
    "t_slide_ms": 200,
    "V_slide_m_s": 8.0,
    "delta_mm": 2.78,
    "t_decay_ms": 14,
    "max_reach_mm": 4.00
  },
  "phase3": {
    "z_lift_mm": 5.0,
    "t_lift_s": 0.5,
    "t_lift_ms": 500,
    "V_lift_mm_s": 10.0,
    "Re_lift": 330,
    "delta_Stokes_mm": 0.39
  },
  "mechanisms": {
    "m1_boundary_layer": {
      "delta_blade_mm": 2.21,
      "delta_carrier_mm": 2.78,
      "max_reach_mm": 4.00,
      "effect": "zero (lift separates)"
    },
    "m2_bulk_swirl": {
      "tau_drag_Nm": 48,
      "delta_Ekman_mm": 0.14,
      "t_Ekman_decay_ms": 70,
      "effect": "zero (horizontal)"
    },
    "m3_potential_flow": {
      "t_acoustic_ms": 9,
      "effect": "zero (vanishes)"
    },
    "m4_pressure_pulse": {
      "dp_kPa": 25,
      "t_traverse_ms": 18,
      "effect": "zero (transient)"
    },
    "m5_secondary_flows": {
      "a_cent_m_s2": 200,
      "effect": "zero (confined)"
    },
    "m6_vortex_shedding": {
      "St": 0.21,
      "f_shed_Hz": 270,
      "effect": "zero (suppressed)"
    }
  },
  "summary": {
    "t_total_s": 1.10,
    "t_total_ms": 1100,
    "max_reach_mm": 4.00,
    "z_lift_mm": 5.00,
    "margin_mm": 1.00,
    "dead_zone_mm": 110,
    "dead_zone_pct": 7.3
  }
}
```

---

## Implementation Order

1. **`mars_numerics.py`** — core computation + CLI + terminal + JSON output
2. **`mars_latex.py`** — LaTeX generator module
3. **`mars_run.py`** — orchestrator
4. **Update `mars_params.json`** — add `u_c_mm_s` to fixed section

---

## `mars_numerics.py` — Cleanup from Old Code

### What Gets REMOVED

| Old Code | Why Remove |
|----------|------------|
| `--drag` option | Analytical no-drag is sufficient; drag adds complexity for <1% effect |
| `--numeric` option | Analytical eddy diffusion is sufficient; PDE solver is overkill |
| `t_wait` parameter | Obsolete — no physics reason for wait time |
| `sweep_angle` as CLI input | Derived: `sweep_angle = 360° / n_arms` |
| `z_obs` / `t_obs` parameters | Legacy — we use `z_lift` directly |
| `u_c` "quiescence" profile scan | Replaced by `max_reach` calculation |
| Numerical ODE solvers for kinematics | Analytical bang-bang is exact without drag |
| `compute_bangbang_kinematics_numerical()` | Removed (drag option gone) |
| `compute_carrier_slide_numerical()` | Removed (drag option gone) |
| `mechanism_eddy_diffusion_numerical()` | Removed (PDE solver gone) |
| Velocity profile arrays | Only `max_reach` needed, not full profiles |
| `compute_velocity_profile_blade()` | Replaced by simpler `max_reach` calc |
| `compute_velocity_profile_slide()` | Replaced by simpler `max_reach` calc |
| `compute_velocity_profile_lift()` | Not needed (lift is vertical, no lateral effect) |
| Verbose multi-page terminal output | Replace with clean, concise summary |
| Separate `propeller_results_n2.json` / `n8.json` | Single `mars_results.json` |
| `propeller_analysis_export.py` | Functionality merged into `mars_numerics.py` |

### What Gets KEPT (core physics)

| Function | Purpose |
|----------|---------|
| `compute_gas_properties()` | ρ, ν, c_s from P, T, γ |
| `compute_bangbang_kinematics_analytical()` | Phase 1 rotation: I, t_rot, ω_max, V_tip |
| `compute_carrier_slide_analytical()` | Phase 2 slide: t_slide, V_slide |
| `compute_lift_kinematics()` | Phase 3 lift: V_lift |
| `compute_flow_regime()` | Ma, Re |
| `compute_boundary_layer()` | δ, θ, δ*, Cf, u_τ |
| `compute_carrier_boundary_layer()` | δ_carrier |
| `compute_stokes_layer_thickness()` | δ_Stokes = √(ν × t_lift) |
| Eddy diffusion (analytical) | ν_t, t_decay, ℓ_turb, ℓ_mol, max_reach |
| 6 mechanism calculations | For documentation (all zero effect) |

### What Gets SIMPLIFIED

| Old | New |
|-----|-----|
| `mechanism_eddy_diffusion()` with profile dict | Simple `max_reach = δ + ℓ_turb + ℓ_mol` |
| `carrier_eddy_diffusion_analytical()` | Merge into single eddy diffusion function |
| Complex profile scanning for z_q | Just compute `max_reach` directly |
| Multiple wrapper functions | Direct computation, fewer layers |

### Key Formulas (analytical, no iteration)

**Gas properties:**
```python
rho = P * M_Xe / (R_gas * T)
nu = mu_Xe / rho
c_s = sqrt(gamma * R_gas * T / M_Xe)
```

**Phase 1 — Blade rotation (bang-bang, no drag):**
```python
sweep_angle = 2 * pi / n_arms
I_total = n_arms * (m_blade * R² / 3 + m_plate * r0²)
t_rot = 2 * sqrt(sweep_angle * I_total / tau_motor)
omega_max = sqrt(sweep_angle * tau_motor / I_total)
V_tip = omega_max * R_blade
```

**Phase 1 — Boundary layer:**
```python
Re_c = V_tip * chord / nu
delta = 0.37 * chord * Re_c**(-0.2)
delta_star = delta / 8
u_tau = V_tip * sqrt(0.0592 * Re_c**(-0.2) / 2)
```

**Phase 1 — Eddy diffusion / max_reach:**
```python
nu_t = 0.018 * V_tip * delta_star
u_prime = 0.05 * V_tip
t_eddy = delta / u_prime
t_decay = 2 * t_eddy
ell_turb = sqrt(nu_t * t_decay)
ell_mol = sqrt(nu * t_decay)  # use t_decay as reference time
max_reach_blade = delta + ell_turb + ell_mol
```

**Phase 2 — Carrier slide (bang-bang, no drag):**
```python
t_slide = 2 * sqrt(delta_r * m_plate / F_actuator)
V_slide = sqrt(delta_r * F_actuator / m_plate)
# Then same BL + eddy diffusion as Phase 1
```

**Phase 3 — Lift:**
```python
V_lift = 2 * z_lift / t_lift  # bang-bang peak
Re_lift = V_lift * z_lift / nu  # confirm laminar
delta_Stokes = sqrt(nu * t_lift)
```

**Summary:**
```python
t_total = t_rot + t_slide + t_lift
max_reach = max(max_reach_blade, max_reach_carrier)
margin = z_lift - max_reach
dead_zone = v_drift * t_total
```

---

## Notes

- `mars_numerics.py` is fully standalone — can run without any JSON files
- `mars_run.py` is the "user-friendly" entry point for the full pipeline
- All 6 mechanisms computed for documentation, but only BL max_reach matters operationally
- No velocity profile arrays in JSON — just max_reach values

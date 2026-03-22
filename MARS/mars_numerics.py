#!/usr/bin/env python3
"""
MARS Flow Analysis — Standalone Computation Script

Computes gas disturbance parameters for the MARS 3-phase motion sequence:
  Phase 1: Blade rotation (bang-bang, 360°/n_arms sweep)
  Phase 2: Carrier radial slide
  Phase 3: Vertical plate lift

All 6 mechanisms analyzed have zero effect on ions:
  M1: Boundary layer (horizontal, lift separates)
  M2: Bulk swirl (horizontal)
  M3: Potential flow (vanishes in ~9 ms)
  M4: Pressure pulse (transient ~20 ms)
  M5: Secondary flows (confined to BL)
  M6: Vortex shedding (suppressed for streamlined body)

Usage:
    python mars_numerics.py                    # With defaults
    python mars_numerics.py --n_arms 8         # Override parameters
    python mars_numerics.py --output my.json   # Custom output file

Output:
    - Terminal: Clean summary
    - JSON: mars_results.json (all computed quantities)
"""

import argparse
import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════════
# Fixed Physical Constants (hardwired)
# ═══════════════════════════════════════════════════════════════════════════
M_Xe = 0.13129       # kg/mol, molar mass of Xe
R_gas = 8.314        # J/(mol·K), universal gas constant
mu_Xe = 2.32e-5      # Pa·s, dynamic viscosity of Xe at 300 K

# Vessel/detector constants
P_bar = 15.0         # bar, gas pressure
T_K = 300.0          # K, gas temperature
gamma_Xe = 5.0 / 3.0 # adiabatic index (monatomic)
R_cyl_m = 1.6        # m, vessel radius
H_cyl_m = 1.5        # m, vessel height
L_drift_m = 1.5      # m, drift length
v_drift_mm_s = 100.0 # mm/s, ion drift velocity
sigma_diff_mm = 1.0  # mm, diffusion sigma

# Carrier plate dimensions (fixed geometry)
CARRIER_WIDTH_m = 0.160   # 160 mm
CARRIER_HEIGHT_m = 0.005  # 5 mm


# ═══════════════════════════════════════════════════════════════════════════
# CLI Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════
def validate_n_arms(value: str) -> int:
    """Validate that n_arms is a positive even integer."""
    ivalue = int(value)
    if ivalue < 2 or ivalue % 2 != 0:
        raise argparse.ArgumentTypeError(
            f"n_arms must be a positive even integer (2, 4, 6, ...), got {value}"
        )
    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MARS flow analysis: compute gas disturbance for 3-phase motion."
    )
    # Design parameters (CLI with defaults)
    parser.add_argument("--n_arms", type=validate_n_arms, default=2,
                        help="Number of arms (must be even: 2,4,6,...) [default: 2]")
    parser.add_argument("--R_blade_m", type=float, default=1.6,
                        help="Blade length [m] [default: 1.6]")
    parser.add_argument("--chord_m", type=float, default=0.16,
                        help="Blade chord [m] [default: 0.16]")
    parser.add_argument("--m_blade_kg", type=float, default=0.25,
                        help="Blade mass [kg] [default: 0.25]")
    parser.add_argument("--m_plate_kg", type=float, default=0.25,
                        help="Carrier plate mass [kg] [default: 0.25]")
    parser.add_argument("--tau_motor_Nm", type=float, default=60.0,
                        help="Motor torque [N·m] [default: 60.0]")
    parser.add_argument("--delta_r_m", type=float, default=0.8,
                        help="Carrier slide distance [m] [default: 0.8]")
    parser.add_argument("--r0_carrier_m", type=float, default=0.8,
                        help="Initial carrier position [m] [default: 0.8]")
    parser.add_argument("--z_lift_mm", type=float, default=5.0,
                        help="Lift height [mm] [default: 5.0]")
    parser.add_argument("--t_lift_s", type=float, default=0.5,
                        help="Lift duration [s] [default: 0.5]")
    parser.add_argument("--F_actuator_N", type=float, default=20.0,
                        help="Actuator force [N] [default: 20.0]")
    parser.add_argument("--u_c_mm_s", type=float, default=0.1,
                        help="Quiescence cutoff velocity [mm/s] [default: 0.1]")
    parser.add_argument("--output", type=str, default="mars_results.json",
                        help="Output JSON file [default: mars_results.json]")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Core Physics Functions
# ═══════════════════════════════════════════════════════════════════════════
def compute_gas_properties() -> dict:
    """Compute xenon gas properties at P, T."""
    P_Pa = P_bar * 1e5
    rho = P_Pa * M_Xe / (R_gas * T_K)
    nu = mu_Xe / rho
    c_s = np.sqrt(gamma_Xe * R_gas * T_K / M_Xe)
    return {
        "rho_kg_m3": rho,
        "nu_m2_s": nu,
        "c_s_m_s": c_s,
    }


def compute_phase1_rotation(n_arms: int, R_blade_m: float, chord_m: float,
                            m_blade_kg: float, m_plate_kg: float,
                            tau_motor_Nm: float, r0_carrier_m: float,
                            nu: float, c_s: float) -> dict:
    """
    Phase 1: Blade rotation (bang-bang, no drag).

    sweep_angle = 2π / n_arms
    I_total = n_arms × (m_blade × R²/3 + m_plate × r0²)
    t_rot = 2 × √(θ × I / τ)
    ω_max = √(θ × τ / I)
    V_tip = ω_max × R
    """
    sweep_angle_rad = 2.0 * np.pi / n_arms
    sweep_angle_deg = 360.0 / n_arms

    # Moment of inertia
    I_blades = n_arms * (1.0 / 3.0) * m_blade_kg * R_blade_m**2
    I_plates = n_arms * m_plate_kg * r0_carrier_m**2
    I_total = I_blades + I_plates

    # Bang-bang kinematics
    t_rot_s = 2.0 * np.sqrt(sweep_angle_rad * I_total / tau_motor_Nm)
    omega_max = np.sqrt(sweep_angle_rad * tau_motor_Nm / I_total)
    V_tip_m_s = omega_max * R_blade_m

    # Flow regime
    Ma_tip = V_tip_m_s / c_s
    Re_tip = V_tip_m_s * chord_m / nu

    # Turbulent boundary layer (flat-plate correlation)
    Re_c = Re_tip
    delta_m = 0.37 * chord_m * Re_c**(-0.2)
    delta_star_m = delta_m / 8.0
    theta_m = (7.0 / 72.0) * delta_m
    Cf = 0.0592 * Re_c**(-0.2)
    u_tau = V_tip_m_s * np.sqrt(Cf / 2.0)

    # Eddy diffusion / max_reach
    nu_t = 0.018 * V_tip_m_s * delta_star_m
    u_prime = 0.05 * V_tip_m_s
    t_eddy = delta_m / u_prime
    t_decay_s = 2.0 * t_eddy
    ell_turb_m = np.sqrt(nu_t * t_decay_s)
    ell_mol_m = np.sqrt(nu * t_decay_s)
    max_reach_m = delta_m + ell_turb_m + ell_mol_m

    return {
        "sweep_angle_deg": sweep_angle_deg,
        "sweep_angle_rad": sweep_angle_rad,
        "I_blades_kg_m2": I_blades,
        "I_plates_kg_m2": I_plates,
        "I_total_kg_m2": I_total,
        "t_rot_s": t_rot_s,
        "t_rot_ms": t_rot_s * 1000,
        "omega_max_rad_s": omega_max,
        "V_tip_m_s": V_tip_m_s,
        "Ma_tip": Ma_tip,
        "Re_tip": Re_tip,
        "delta_m": delta_m,
        "delta_mm": delta_m * 1000,
        "delta_star_m": delta_star_m,
        "theta_m": theta_m,
        "Cf": Cf,
        "u_tau_m_s": u_tau,
        "nu_t_m2_s": nu_t,
        "u_prime_m_s": u_prime,
        "t_eddy_s": t_eddy,
        "t_decay_s": t_decay_s,
        "t_decay_ms": t_decay_s * 1000,
        "ell_turb_mm": ell_turb_m * 1000,
        "ell_mol_mm": ell_mol_m * 1000,
        "max_reach_m": max_reach_m,
        "max_reach_mm": max_reach_m * 1000,
    }


def compute_phase2_slide(m_plate_kg: float, F_actuator_N: float,
                         delta_r_m: float, nu: float, c_s: float) -> dict:
    """
    Phase 2: Carrier radial slide (bang-bang, no drag).

    t_slide = 2 × √(Δr × m / F)
    V_slide = √(Δr × F / m)
    """
    # Bang-bang kinematics
    t_slide_s = 2.0 * np.sqrt(delta_r_m * m_plate_kg / F_actuator_N)
    V_slide_m_s = np.sqrt(delta_r_m * F_actuator_N / m_plate_kg)

    # Flow regime
    Ma_slide = V_slide_m_s / c_s
    Re_slide = V_slide_m_s * CARRIER_WIDTH_m / nu

    # Boundary layer on carrier (turbulent if Re > 5e5)
    L_carrier = CARRIER_WIDTH_m
    Re_L = V_slide_m_s * L_carrier / nu

    if Re_L < 5e5:
        # Laminar
        delta_m = 5.0 * L_carrier / np.sqrt(Re_L) if Re_L > 0 else 0.0
        regime = "laminar"
    else:
        # Turbulent
        delta_m = 0.37 * L_carrier * Re_L**(-0.2)
        regime = "turbulent"

    delta_star_m = delta_m / 8.0 if regime == "turbulent" else delta_m / 3.0

    # Eddy diffusion / max_reach
    nu_t = 0.018 * V_slide_m_s * delta_star_m
    u_prime = 0.05 * V_slide_m_s
    t_eddy = delta_m / u_prime if u_prime > 0 else float('inf')
    t_decay_s = 2.0 * t_eddy if t_eddy < float('inf') else 0.0
    ell_turb_m = np.sqrt(nu_t * t_decay_s) if t_decay_s > 0 else 0.0
    ell_mol_m = np.sqrt(nu * t_decay_s) if t_decay_s > 0 else 0.0
    max_reach_m = delta_m + ell_turb_m + ell_mol_m

    return {
        "delta_r_m": delta_r_m,
        "t_slide_s": t_slide_s,
        "t_slide_ms": t_slide_s * 1000,
        "V_slide_m_s": V_slide_m_s,
        "Ma_slide": Ma_slide,
        "Re_slide": Re_slide,
        "regime": regime,
        "delta_m": delta_m,
        "delta_mm": delta_m * 1000,
        "delta_star_m": delta_star_m,
        "nu_t_m2_s": nu_t,
        "t_decay_s": t_decay_s,
        "t_decay_ms": t_decay_s * 1000,
        "ell_turb_mm": ell_turb_m * 1000,
        "ell_mol_mm": ell_mol_m * 1000,
        "max_reach_m": max_reach_m,
        "max_reach_mm": max_reach_m * 1000,
    }


def compute_phase3_lift(z_lift_mm: float, t_lift_s: float, nu: float) -> dict:
    """
    Phase 3: Vertical plate lift.

    V_lift = 2 × z_lift / t_lift (bang-bang peak velocity)
    Re_lift = V_lift × z_lift / ν
    δ_Stokes = √(ν × t_lift)
    """
    z_lift_m = z_lift_mm / 1000.0

    # Bang-bang peak velocity
    V_lift_m_s = 2.0 * z_lift_m / t_lift_s
    V_lift_mm_s = V_lift_m_s * 1000.0

    # Reynolds number (should be laminar)
    Re_lift = V_lift_m_s * z_lift_m / nu
    regime = "laminar" if Re_lift < 2000 else "turbulent"

    # Stokes layer thickness
    delta_Stokes_m = np.sqrt(nu * t_lift_s)

    return {
        "z_lift_mm": z_lift_mm,
        "z_lift_m": z_lift_m,
        "t_lift_s": t_lift_s,
        "t_lift_ms": t_lift_s * 1000,
        "V_lift_m_s": V_lift_m_s,
        "V_lift_mm_s": V_lift_mm_s,
        "Re_lift": Re_lift,
        "regime": regime,
        "delta_Stokes_m": delta_Stokes_m,
        "delta_Stokes_mm": delta_Stokes_m * 1000,
    }


def compute_6_mechanisms(phase1: dict, phase2: dict, rho: float, nu: float,
                         c_s: float, R_blade_m: float, chord_m: float) -> dict:
    """
    Compute parameters for all 6 mechanisms (for documentation).
    All have zero effect on ions.
    """
    V_tip = phase1["V_tip_m_s"]
    delta_blade = phase1["delta_m"]
    t_rot = phase1["t_rot_s"]

    # M1: Boundary layer (already computed in phase1/phase2)
    m1 = {
        "delta_blade_mm": phase1["delta_mm"],
        "delta_carrier_mm": phase2["delta_mm"],
        "max_reach_blade_mm": phase1["max_reach_mm"],
        "max_reach_carrier_mm": phase2["max_reach_mm"],
        "effect": "zero (horizontal, lift separates)"
    }

    # M2: Bulk swirl
    # Drag torque (approximate)
    C_d = 0.1
    d_wake = 0.12 * chord_m
    tau_drag = rho * C_d * d_wake * R_blade_m**4 * (V_tip / R_blade_m)**2 / 4.0

    # Ekman layer
    Omega_wake = 0.5 * V_tip / R_blade_m
    delta_Ekman_m = np.sqrt(nu / Omega_wake)
    t_Ekman_decay_s = H_cyl_m / np.sqrt(nu * Omega_wake)

    m2 = {
        "tau_drag_Nm": tau_drag,
        "delta_Ekman_mm": delta_Ekman_m * 1000,
        "t_Ekman_decay_ms": t_Ekman_decay_s * 1000,
        "effect": "zero (horizontal tangential flow)"
    }

    # M3: Potential (displacement) flow
    t_acoustic_s = R_blade_m / c_s
    m3 = {
        "t_acoustic_ms": t_acoustic_s * 1000,
        "effect": "zero (vanishes after motion stops)"
    }

    # M4: Pressure pulse
    dp_Pa = rho * V_tip**2
    t_traverse_s = R_cyl_m / c_s
    m4 = {
        "dp_Pa": dp_Pa,
        "dp_kPa": dp_Pa / 1000,
        "t_traverse_ms": t_traverse_s * 1000,
        "effect": "zero (transient, ions arrive >1s later)"
    }

    # M5: Secondary flows
    u_wake = 0.5 * V_tip
    a_cent = u_wake**2 / R_blade_m
    m5 = {
        "a_cent_m_s2": a_cent,
        "effect": "zero (horizontal, confined to BL)"
    }

    # M6: Vortex shedding
    # For streamlined NACA 0012 at α=0, shedding is suppressed
    # If any: St ≈ 0.21
    St = 0.21
    t_max = 0.12 * chord_m
    f_shed = St * V_tip / t_max if t_max > 0 else 0
    m6 = {
        "St": St,
        "f_shed_Hz": f_shed,
        "effect": "zero (suppressed for streamlined body at α=0)"
    }

    return {
        "m1_boundary_layer": m1,
        "m2_bulk_swirl": m2,
        "m3_potential_flow": m3,
        "m4_pressure_pulse": m4,
        "m5_secondary_flows": m5,
        "m6_vortex_shedding": m6,
    }


def compute_velocity_profiles(phase1: dict, phase2: dict, phase3: dict,
                               z_points_mm: list = None) -> dict:
    """
    Compute velocity profiles u_θ(z), u_r(z), u_z(z) for all three phases.

    Uses 1/7 power law for turbulent BLs (phases 1 & 2):
        u(z) = V × (z/δ)^{1/7}    for z < δ

    Phase 1 (blade rotation): u_θ(z) from blade BL, u_r = u_z = 0
    Phase 2 (carrier slide):  u_r(z) from carrier BL, u_θ = u_z = 0
    Phase 3 (vertical lift):  u_z(z) from Stokes layer, u_θ = u_r = 0

    Args:
        phase1: Phase 1 results dict (contains V_tip_m_s, delta_m)
        phase2: Phase 2 results dict (contains V_slide_m_s, delta_m)
        phase3: Phase 3 results dict (contains V_lift_m_s, delta_Stokes_m)
        z_points_mm: Heights at which to evaluate [mm]. Default: 0.1 to 5 mm.

    Returns:
        Dictionary with z_mm array and u_theta, u_r, u_z velocity arrays.
    """
    if z_points_mm is None:
        z_points_mm = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    z_points_m = np.array(z_points_mm) / 1000.0

    # Phase 1: Blade rotation → u_θ(z)
    V_tip = phase1["V_tip_m_s"]
    delta_blade = phase1["delta_m"]
    u_theta = []
    for z_m in z_points_m:
        if z_m < delta_blade:
            u = V_tip * (z_m / delta_blade) ** (1.0 / 7.0)
        else:
            u = 0.0
        u_theta.append(u)

    # Phase 2: Carrier slide → u_r(z)
    V_slide = phase2["V_slide_m_s"]
    delta_carrier = phase2["delta_m"]
    u_r = []
    for z_m in z_points_m:
        if z_m < delta_carrier:
            u = V_slide * (z_m / delta_carrier) ** (1.0 / 7.0)
        else:
            u = 0.0
        u_r.append(u)

    # Phase 3: Vertical lift → u_z(z)
    # Stokes layer: u_z decays exponentially from surface
    # u_z(z) ≈ V_lift × exp(-z/δ_Stokes) for laminar oscillatory BL
    V_lift = phase3["V_lift_m_s"]
    delta_Stokes = phase3["delta_Stokes_m"]
    u_z = []
    for z_m in z_points_m:
        # Exponential decay in Stokes layer
        u = V_lift * np.exp(-z_m / delta_Stokes)
        u_z.append(u)

    return {
        "z_mm": z_points_mm,
        "delta_blade_mm": delta_blade * 1000,
        "delta_carrier_mm": delta_carrier * 1000,
        "delta_Stokes_mm": delta_Stokes * 1000,
        "V_tip_m_s": V_tip,
        "V_slide_m_s": V_slide,
        "V_lift_m_s": V_lift,
        "u_theta_m_s": u_theta,
        "u_r_m_s": u_r,
        "u_z_m_s": u_z,
        "note": "u_θ, u_r: 1/7 power law; u_z: Stokes exponential decay"
    }


def compute_summary(phase1: dict, phase2: dict, phase3: dict) -> dict:
    """Compute summary metrics."""
    t_total_s = phase1["t_rot_s"] + phase2["t_slide_s"] + phase3["t_lift_s"]
    max_reach_mm = max(phase1["max_reach_mm"], phase2["max_reach_mm"])
    z_lift_mm = phase3["z_lift_mm"]
    margin_mm = z_lift_mm - max_reach_mm
    dead_zone_mm = v_drift_mm_s * t_total_s
    dead_zone_pct = 100.0 * dead_zone_mm / (L_drift_m * 1000)

    return {
        "t_total_s": t_total_s,
        "t_total_ms": t_total_s * 1000,
        "max_reach_mm": max_reach_mm,
        "z_lift_mm": z_lift_mm,
        "margin_mm": margin_mm,
        "margin_ok": margin_mm > 0,
        "dead_zone_mm": dead_zone_mm,
        "dead_zone_pct": dead_zone_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output Functions
# ═══════════════════════════════════════════════════════════════════════════
def print_velocity_table(profiles: dict) -> None:
    """Print velocity profile table for u_θ(z), u_r(z), u_z(z)."""
    print("VELOCITY PROFILES vs HEIGHT z (1/7 power law for u_θ, u_r; Stokes for u_z)")
    print("─" * 80)
    print(f"Phase 1 (blade):   V_tip = {profiles['V_tip_m_s']:.1f} m/s,   δ_blade = {profiles['delta_blade_mm']:.2f} mm")
    print(f"Phase 2 (carrier): V_slide = {profiles['V_slide_m_s']:.1f} m/s, δ_carrier = {profiles['delta_carrier_mm']:.2f} mm")
    print(f"Phase 3 (lift):    V_lift = {profiles['V_lift_m_s']*1000:.1f} mm/s, δ_Stokes = {profiles['delta_Stokes_mm']:.2f} mm")
    print()
    print("  z (mm)  │  u_θ (m/s)  │  u_r (m/s)  │  u_z (mm/s)  │  Direction")
    print("──────────┼─────────────┼─────────────┼──────────────┼─────────────────")

    for i, z_mm in enumerate(profiles['z_mm']):
        u_theta = profiles['u_theta_m_s'][i]
        u_r = profiles['u_r_m_s'][i]
        u_z = profiles['u_z_m_s'][i] * 1000  # Convert to mm/s for display

        # Determine which components are active
        directions = []
        if u_theta > 0:
            directions.append("tangential")
        if u_r > 0:
            directions.append("radial")
        if u_z > 0.01:  # threshold for display
            directions.append("vertical")
        direction_str = ", ".join(directions) if directions else "quiescent"

        print(f"  {z_mm:5.1f}   │    {u_theta:7.3f}   │    {u_r:7.3f}   │     {u_z:7.2f}   │  {direction_str}")

    print()


def print_terminal_output(gas: dict, phase1: dict, phase2: dict, phase3: dict,
                          summary: dict, n_arms: int) -> None:
    """Print clean terminal summary."""
    print("═" * 70)
    print(f"MARS FLOW ANALYSIS — n = {n_arms} arms")
    print("═" * 70)
    print()

    print(f"GAS PROPERTIES (Xe at {P_bar:.0f} bar, {T_K:.0f} K)")
    print(f"  ρ = {gas['rho_kg_m3']:.1f} kg/m³    "
          f"ν = {gas['nu_m2_s']:.2e} m²/s    "
          f"c_s = {gas['c_s_m_s']:.0f} m/s")
    print()

    print(f"PHASE 1: BLADE ROTATION ({phase1['sweep_angle_deg']:.0f}°)")
    print(f"  t_rot = {phase1['t_rot_ms']:.0f} ms    V_tip = {phase1['V_tip_m_s']:.1f} m/s")
    print(f"  Ma = {phase1['Ma_tip']:.2f} (incompressible)    Re = {phase1['Re_tip']:.1e} (turbulent)")
    print(f"  δ_blade = {phase1['delta_mm']:.2f} mm    max_reach = {phase1['max_reach_mm']:.2f} mm")
    print()

    print(f"PHASE 2: CARRIER SLIDE ({phase2['delta_r_m']:.1f} m)")
    print(f"  t_slide = {phase2['t_slide_ms']:.0f} ms    V_slide = {phase2['V_slide_m_s']:.1f} m/s")
    print(f"  δ_carrier = {phase2['delta_mm']:.2f} mm    max_reach = {phase2['max_reach_mm']:.2f} mm")
    print()

    print(f"PHASE 3: VERTICAL LIFT ({phase3['z_lift_mm']:.0f} mm)")
    print(f"  t_lift = {phase3['t_lift_ms']:.0f} ms    V_lift = {phase3['V_lift_mm_s']:.0f} mm/s")
    print(f"  Re_lift = {phase3['Re_lift']:.0f} ({phase3['regime']})")
    print(f"  δ_Stokes = {phase3['delta_Stokes_mm']:.2f} mm")
    print()

    print("SUMMARY")
    print(f"  t_total = {summary['t_total_s']:.2f} s")
    print(f"  max_reach (worst case) = {summary['max_reach_mm']:.2f} mm")
    margin_status = "✓" if summary["margin_ok"] else "✗ WARNING"
    print(f"  z_lift = {summary['z_lift_mm']:.2f} mm    margin = {summary['margin_mm']:.2f} mm {margin_status}")
    print(f"  dead_zone = {summary['dead_zone_mm']:.0f} mm ({summary['dead_zone_pct']:.1f}% of drift)")
    print()


def write_json_output(output_file: str, args: argparse.Namespace, gas: dict,
                      phase1: dict, phase2: dict, phase3: dict,
                      mechanisms: dict, summary: dict,
                      velocity_profiles: dict) -> None:
    """Write all results to JSON file."""
    results = {
        "input_params": {
            "n_arms": args.n_arms,
            "R_blade_m": args.R_blade_m,
            "chord_m": args.chord_m,
            "m_blade_kg": args.m_blade_kg,
            "m_plate_kg": args.m_plate_kg,
            "tau_motor_Nm": args.tau_motor_Nm,
            "delta_r_m": args.delta_r_m,
            "r0_carrier_m": args.r0_carrier_m,
            "z_lift_mm": args.z_lift_mm,
            "t_lift_s": args.t_lift_s,
            "F_actuator_N": args.F_actuator_N,
            "u_c_mm_s": args.u_c_mm_s,
        },
        "fixed_params": {
            "P_bar": P_bar,
            "T_K": T_K,
            "gamma_Xe": gamma_Xe,
            "R_cyl_m": R_cyl_m,
            "H_cyl_m": H_cyl_m,
            "L_drift_m": L_drift_m,
            "v_drift_mm_s": v_drift_mm_s,
            "sigma_diff_mm": sigma_diff_mm,
        },
        "gas_properties": gas,
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "mechanisms": mechanisms,
        "summary": summary,
        "velocity_profiles": velocity_profiles,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"Results written to: {output_file}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()

    # Compute gas properties
    gas = compute_gas_properties()
    rho = gas["rho_kg_m3"]
    nu = gas["nu_m2_s"]
    c_s = gas["c_s_m_s"]

    # Phase 1: Blade rotation
    phase1 = compute_phase1_rotation(
        n_arms=args.n_arms,
        R_blade_m=args.R_blade_m,
        chord_m=args.chord_m,
        m_blade_kg=args.m_blade_kg,
        m_plate_kg=args.m_plate_kg,
        tau_motor_Nm=args.tau_motor_Nm,
        r0_carrier_m=args.r0_carrier_m,
        nu=nu,
        c_s=c_s,
    )

    # Phase 2: Carrier slide
    phase2 = compute_phase2_slide(
        m_plate_kg=args.m_plate_kg,
        F_actuator_N=args.F_actuator_N,
        delta_r_m=args.delta_r_m,
        nu=nu,
        c_s=c_s,
    )

    # Phase 3: Vertical lift
    phase3 = compute_phase3_lift(
        z_lift_mm=args.z_lift_mm,
        t_lift_s=args.t_lift_s,
        nu=nu,
    )

    # 6 Mechanisms (for documentation)
    mechanisms = compute_6_mechanisms(
        phase1=phase1,
        phase2=phase2,
        rho=rho,
        nu=nu,
        c_s=c_s,
        R_blade_m=args.R_blade_m,
        chord_m=args.chord_m,
    )

    # Summary
    summary = compute_summary(phase1, phase2, phase3)

    # Velocity profiles u_θ(z), u_r(z), u_z(z)
    velocity_profiles = compute_velocity_profiles(phase1, phase2, phase3)

    # Output
    print_terminal_output(gas, phase1, phase2, phase3, summary, args.n_arms)
    print_velocity_table(velocity_profiles)
    write_json_output(args.output, args, gas, phase1, phase2, phase3, mechanisms,
                      summary, velocity_profiles)


if __name__ == "__main__":
    main()

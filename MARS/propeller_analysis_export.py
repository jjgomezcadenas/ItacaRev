#!/usr/bin/env python3
"""
Export propeller flow analysis results to JSON for LaTeX integration.

Runs analysis for both N=2 and N=8 configurations with numeric PDE solver,
capturing all parameters, boundary layer properties, and velocity profiles.

Usage:
    python propeller_analysis_export.py

Outputs:
    propeller_results_n2.json
    propeller_results_n8.json
"""

import json
import numpy as np
from pathlib import Path


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Import functions from propeller_flow_numerics
from propeller_flow_numerics import (
    # Constants
    M_Xe, R_gas, P, T, gamma, mu_Xe, R_blade, Delta_D, kappa,
    CARRIER_WIDTH, CARRIER_HEIGHT, CARRIER_FRONTAL_AREA,
    # Functions
    compute_gas_properties,
    compute_bangbang_kinematics,
    compute_flow_regime,
    compute_boundary_layer,
    compute_geometry,
    compute_carrier_slide,
    compute_carrier_flow_regime,
    compute_carrier_boundary_layer,
    carrier_eddy_diffusion_analytical,
    mechanism_eddy_diffusion,
    mechanism_bulk_swirl,
    mechanism_potential_flow,
    mechanism_pressure_pulse,
    mechanism_secondary_flows,
    mechanism_vortex_shedding,
)


def run_analysis(n_arms: int, numeric: bool = True, drag: bool = False) -> dict:
    """
    Run full propeller flow analysis for given configuration.

    Parameters:
        n_arms: Number of arms (2 or 8)
        numeric: Use numerical PDE solver for eddy diffusion
        drag: Include aerodynamic drag in kinematics

    Returns:
        Dictionary with all parameters and results
    """
    # Fixed parameters
    c = 0.16  # chord [m]
    m_blade = 0.25  # kg
    m_plate = 0.25  # kg (same as m_carrier - it's the same carrier plate)
    tau_motor = 60.0  # N·m
    z_obs = 0.005  # m
    t_obs = 0.5  # s

    # Phase 2 parameters (carrier slide)
    F_actuator = 20.0  # N
    m_carrier = m_plate  # same carrier plate mass
    Cd_carrier = 0.1
    r0_carrier = 0.8  # m (R/2)

    # Derived
    sweep_angle = 2.0 * np.pi / n_arms

    # Gas properties
    rho, nu, c_s = compute_gas_properties()

    # Build results dictionary
    results = {
        "metadata": {
            "n_arms": n_arms,
            "numeric_solver": numeric,
            "drag_model": drag,
        },
        "fixed_parameters": {
            "R_blade_m": R_blade,
            "chord_m": c,
            "m_blade_kg": m_blade,
            "m_plate_kg": m_plate,
            "tau_motor_Nm": tau_motor,
            "z_obs_m": z_obs,
            "z_obs_mm": z_obs * 1000,
            "t_obs_s": t_obs,
            "P_Pa": P,
            "P_bar": P / 1e5,
            "T_K": T,
            "Delta_D_m": Delta_D,
        },
        "carrier_parameters": {
            "F_actuator_N": F_actuator,
            "m_carrier_kg": m_carrier,
            "Cd_carrier": Cd_carrier,
            "r0_carrier_m": r0_carrier,
            "carrier_width_m": CARRIER_WIDTH,
            "carrier_height_m": CARRIER_HEIGHT,
            "carrier_frontal_area_m2": CARRIER_FRONTAL_AREA,
        },
        "gas_properties": {
            "rho_kg_m3": rho,
            "mu_Pa_s": mu_Xe,
            "nu_m2_s": nu,
            "c_s_m_s": c_s,
            "gamma": gamma,
        },
    }

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: BLADE ROTATION
    # ═══════════════════════════════════════════════════════════════

    # Kinematics
    bb = compute_bangbang_kinematics(
        n_arms, m_blade, m_plate, tau_motor, sweep_angle,
        rho, c, drag=drag, r0_carrier=r0_carrier
    )
    t_rot = bb["t_rot"]
    omega_max = bb["omega_max"]
    V_tip = bb["V_tip_max"]

    results["phase1_kinematics"] = {
        "sweep_angle_rad": sweep_angle,
        "sweep_angle_deg": np.degrees(sweep_angle),
        "I_blades_kg_m2": bb["I_blades"],
        "I_plates_kg_m2": bb["I_plates"],
        "I_total_kg_m2": bb["I_total"],
        "t_rot_s": t_rot,
        "t_rot_ms": t_rot * 1000,
        "omega_max_rad_s": omega_max,
        "V_tip_max_m_s": V_tip,
        "method": bb["method"],
    }

    # Flow regime
    Ma_tip, Re_tip = compute_flow_regime(c, V_tip, c_s, nu)
    results["phase1_flow_regime"] = {
        "Ma_tip": Ma_tip,
        "Re_tip": Re_tip,
        "compressible": Ma_tip >= 0.3,
        "turbulent": Re_tip > 5e5,
    }

    # Boundary layer
    delta, theta, delta_star, H, Cf, u_tau, Cd, Re_c = compute_boundary_layer(c, V_tip, nu)
    results["phase1_boundary_layer"] = {
        "delta_m": delta,
        "delta_mm": delta * 1000,
        "theta_m": theta,
        "delta_star_m": delta_star,
        "H": H,
        "Cf": Cf,
        "u_tau_m_s": u_tau,
        "Cd": Cd,
        "Re_c": Re_c,
    }

    # Geometry
    t_max, z_blade_top, z_BL_top, z_observation, gap, R_cyl, tip_gap = compute_geometry(
        c, delta, z_obs
    )
    results["geometry"] = {
        "t_max_m": t_max,
        "t_max_mm": t_max * 1000,
        "z_blade_top_mm": z_blade_top * 1000,
        "z_BL_top_mm": z_BL_top * 1000,
        "gap_mm": gap * 1000,
        "R_cyl_m": R_cyl,
        "tip_gap_mm": tip_gap * 1000,
    }

    # Mechanism 1: Eddy diffusion
    m1 = mechanism_eddy_diffusion(
        V_tip, c, delta, delta_star, u_tau, nu, z_obs, t_obs, numeric=numeric
    )
    results["phase1_eddy_diffusion"] = {
        "method": m1["method"],
        "nu_t_outer_m2_s": m1["nu_t_outer"],
        "nu_t_over_nu": m1["nu_t_over_nu"],
        "u_prime_m_s": m1["u_prime"],
        "t_eddy_s": m1["t_eddy"],
        "t_eddy_ms": m1["t_eddy"] * 1000,
        "t_decay_s": m1["t_decay"],
        "t_decay_ms": m1["t_decay"] * 1000,
        "ell_turb_m": m1["ell_turb"],
        "ell_turb_mm": m1["ell_turb"] * 1000,
        "ell_mol_m": m1["ell_mol"],
        "ell_mol_mm": m1["ell_mol"] * 1000,
        "max_reach_m": m1["max_reach"],
        "max_reach_mm": m1["max_reach"] * 1000,
        "mom_initial": m1["mom_initial"],
        "mom_final": m1["mom_final"],
        "mom_fraction": m1["mom_fraction"],
        "u_at_obs_m_s": m1["u_at_obs"],
        "u_at_obs_cm_s": m1["u_at_obs"] * 100,
        "reaches_z_obs": m1["max_reach"] >= z_obs,
    }

    # Velocity profile (convert keys to strings for JSON)
    results["phase1_velocity_profile"] = {
        "z_mm": list(m1["profile"].keys()),
        "u_m_s": list(m1["profile"].values()),
        "u_cm_s": [v * 100 for v in m1["profile"].values()],
    }

    # Other mechanisms
    m2 = mechanism_bulk_swirl(
        n_arms, rho, nu, omega_max, c, Cd, t_rot, delta, z_obs, t_obs, V_tip
    )
    results["phase1_bulk_swirl"] = {
        "torque_Nm": m2["torque"],
        "J_angular_Nm_s": m2["J_angular"],
        "gap_mm": m2["gap"] * 1000,
        "t_visc_s": m2["t_visc"],
        "delta_E_mol_mm": m2["delta_E_mol"] * 1000,
        "w_E_mol_um_s": m2["w_E_mol"] * 1e6,
        "delta_E_turb_mm": m2["delta_E_turb"] * 1000,
        "w_E_turb_cm_s": m2["w_E_turb"] * 100,
        "z_Ekman_turb_mm": m2["z_Ekman_turb"] * 1000,
        "t_turb_life_ms": m2["t_turb_life"] * 1000,
    }

    m3 = mechanism_potential_flow(V_tip, c, z_obs, c_s)
    results["phase1_potential_flow"] = {
        "t_max_mm": m3["t_max"] * 1000,
        "blockage_ratio": m3["blockage_ratio"],
        "w_disp_during_sweep_m_s": m3["w_disp_during_sweep"],
        "t_acoustic_ms": m3["t_acoustic"] * 1000,
    }

    m4 = mechanism_pressure_pulse(rho, V_tip, c_s, z_obs)
    results["phase1_pressure_pulse"] = {
        "dp_Pa": m4["dp"],
        "dp_fraction": m4["dp_fraction"],
        "du_acoustic_m_s": m4["du_acoustic"],
        "t_arrive_us": m4["t_arrive"] * 1e6,
    }

    m5 = mechanism_secondary_flows(rho, V_tip, nu)
    results["phase1_secondary_flows"] = {
        "tip_gap_mm": m5["tip_gap"] * 1000,
        "a_cent_m_s2": m5["a_cent"],
        "t_cent_s": m5["t_cent"],
        "dp_cent_Pa": m5["dp_cent"],
    }

    m6 = mechanism_vortex_shedding(V_tip, c, nu)
    results["phase1_vortex_shedding"] = {
        "Re_d": m6["Re_d"],
        "St": m6["St"],
        "f_shed_Hz": m6["f_shed"],
    }

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: CARRIER SLIDE
    # ═══════════════════════════════════════════════════════════════

    cs = compute_carrier_slide(
        m_carrier, F_actuator, r0_carrier, Cd_carrier,
        rho, delta_r=r0_carrier, drag=drag
    )
    t_slide = cs["t_slide"]
    V_slide_max = cs["V_max"]

    results["phase2_kinematics"] = {
        "delta_r_m": cs["delta_r"],
        "t_slide_s": t_slide,
        "t_slide_ms": t_slide * 1000,
        "V_max_m_s": V_slide_max,
        "method": cs["method"],
    }

    # Flow regime
    Ma_slide, Re_slide = compute_carrier_flow_regime(V_slide_max, c_s, nu)
    results["phase2_flow_regime"] = {
        "Ma": Ma_slide,
        "Re": Re_slide,
        "compressible": Ma_slide >= 0.3,
        "turbulent": Re_slide > 5e5,
    }

    # Boundary layer
    bl_carrier = compute_carrier_boundary_layer(V_slide_max, nu)
    delta_c = bl_carrier["delta"]
    results["phase2_boundary_layer"] = {
        "L_carrier_mm": bl_carrier["L_carrier"] * 1000,
        "Re_L": bl_carrier["Re_L"],
        "regime": bl_carrier["regime"],
        "delta_m": delta_c,
        "delta_mm": delta_c * 1000,
        "theta_m": bl_carrier["theta"],
        "delta_star_m": bl_carrier["delta_star"],
        "H": bl_carrier["H"],
        "Cf": bl_carrier["Cf"],
        "u_tau_m_s": bl_carrier["u_tau"],
    }

    # Eddy diffusion
    m1_c = carrier_eddy_diffusion_analytical(
        V_slide_max, delta_c, bl_carrier["delta_star"], nu, z_obs, t_obs
    )
    results["phase2_eddy_diffusion"] = {
        "method": m1_c["method"],
        "nu_t_outer_m2_s": m1_c["nu_t_outer"],
        "nu_t_over_nu": m1_c["nu_t_over_nu"],
        "u_prime_m_s": m1_c["u_prime"],
        "t_eddy_s": m1_c["t_eddy"] if m1_c["t_eddy"] < float("inf") else None,
        "t_eddy_ms": m1_c["t_eddy"] * 1000 if m1_c["t_eddy"] < float("inf") else None,
        "t_decay_s": m1_c["t_decay"] if m1_c["t_decay"] < float("inf") else None,
        "t_decay_ms": m1_c["t_decay"] * 1000 if m1_c["t_decay"] < float("inf") else None,
        "ell_turb_mm": m1_c["ell_turb"] * 1000,
        "ell_mol_mm": m1_c["ell_mol"] * 1000,
        "max_reach_mm": m1_c["max_reach"] * 1000,
        "u_at_obs_m_s": m1_c["u_at_obs"],
        "u_at_obs_cm_s": m1_c["u_at_obs"] * 100,
        "reaches_z_obs": m1_c["max_reach"] >= z_obs,
    }

    # ═══════════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ═══════════════════════════════════════════════════════════════

    t_total = t_rot + t_slide
    results["summary"] = {
        "t_rot_ms": t_rot * 1000,
        "t_slide_ms": t_slide * 1000,
        "t_total_ms": t_total * 1000,
        "V_tip_max_m_s": V_tip,
        "V_slide_max_m_s": V_slide_max,
        "phase1_delta_mm": delta * 1000,
        "phase2_delta_mm": delta_c * 1000,
        "phase1_max_reach_mm": m1["max_reach"] * 1000,
        "phase2_max_reach_mm": m1_c["max_reach"] * 1000,
        "z_obs_mm": z_obs * 1000,
        "phase1_u_at_obs_m_s": m1["u_at_obs"],
        "phase2_u_at_obs_m_s": m1_c["u_at_obs"],
        "gas_quiescent_at_z_obs": (m1["u_at_obs"] < 1e-6) and (m1_c["u_at_obs"] < 1e-6),
    }

    return results


def main():
    """Run analysis for N=2 and N=8, export to JSON."""
    output_dir = Path(__file__).parent

    for n_arms in [2, 8]:
        print(f"Running analysis for N = {n_arms}...")
        results = run_analysis(n_arms, numeric=True, drag=False)

        output_file = output_dir / f"propeller_results_n{n_arms}.json"
        # Convert numpy types for JSON serialization
        results = convert_numpy_types(results)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Saved to {output_file}")
        print(f"  Phase 1: t_rot = {results['phase1_kinematics']['t_rot_ms']:.1f} ms, "
              f"V_tip = {results['phase1_kinematics']['V_tip_max_m_s']:.2f} m/s")
        print(f"  Phase 2: t_slide = {results['phase2_kinematics']['t_slide_ms']:.1f} ms, "
              f"V_max = {results['phase2_kinematics']['V_max_m_s']:.2f} m/s")
        print(f"  u(z_obs) Phase 1: {results['phase1_eddy_diffusion']['u_at_obs_m_s']:.2e} m/s")
        print(f"  u(z_obs) Phase 2: {results['phase2_eddy_diffusion']['u_at_obs_m_s']:.2e} m/s")
        print(f"  Gas quiescent: {results['summary']['gas_quiescent_at_z_obs']}")
        print()

    print("Done. JSON files ready for LaTeX integration.")


if __name__ == "__main__":
    main()

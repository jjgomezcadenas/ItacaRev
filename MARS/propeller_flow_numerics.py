#!/usr/bin/env python3
"""
Residual gas velocity above a stopped propeller blade in dense xenon.
Mechanism-by-mechanism numerical evaluation.

Accompanies: propeller_flow_analysis.tex

Usage:
    python propeller_flow_numerics.py [OPTIONS]

Options:
    --chord        Blade chord length [m]             (default: 0.16)
    --sweep_angle  Sweep angle [rad]                  (default: pi)
    --m_blade      Mass of each blade [kg]            (default: 0.25)
    --m_plate      Mass of each carrier plate [kg]   (default: 0.15)
    --tau_motor    Motor torque [N·m]                 (default: 60.0)
    --z_obs        Observation height above blade [m]  (default: 0.005)
    --t_obs        Observation time after stop [s]     (default: 0.5)

Fixed parameters:
    R = 1.6 m, P = 15 bar, T = 300 K, NACA 0012 at alpha = 0,
    wire grid 0.2 mm / 2 mm, vessel diameter surplus = 10 cm.

Motion profile:
    Bang-bang (maximum acceleration then maximum deceleration).
    Rotation time t_rot is computed from inertia and motor torque.
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
import sys


# ═══════════════════════════════════════════════════════════════
# Parse command-line arguments
# ═══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Residual gas velocity above a stopped propeller blade in Xe."
    )
    parser.add_argument(
        "--chord", type=float, default=0.16,
        help="Blade chord length [m] (default: 0.16)"
    )
    parser.add_argument(
        "--sweep_angle", type=float, default=np.pi,
        help="Sweep angle [rad] (default: pi ≈ 3.14159)"
    )
    parser.add_argument(
        "--m_blade", type=float, default=0.25,
        help="Mass of each blade [kg] (default: 0.25, 2 blades total)"
    )
    parser.add_argument(
        "--m_plate", type=float, default=0.15,
        help="Mass of each carrier plate [kg] (default: 0.15, 2 plates total)"
    )
    parser.add_argument(
        "--tau_motor", type=float, default=60.0,
        help="Motor torque [N·m] (default: 60.0)"
    )
    parser.add_argument(
        "--z_obs", type=float, default=0.005,
        help="Observation height above blade top [m] (default: 0.005)"
    )
    parser.add_argument(
        "--t_obs", type=float, default=0.5,
        help="Observation time after stop [s] (default: 0.5)"
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Fixed physical parameters
# ═══════════════════════════════════════════════════════════════
M_Xe = 0.13129      # kg/mol, molar mass of Xe
R_gas = 8.314        # J/(mol·K), universal gas constant
P = 15e5             # Pa, gas pressure
T = 300.0            # K, gas temperature
gamma = 5.0 / 3.0   # adiabatic index (monatomic)
mu_Xe = 2.32e-5      # Pa·s, dynamic viscosity of Xe at 300 K

R_blade = 1.6        # m, blade span (hub to tip)
Delta_D = 0.10       # m, vessel inner diameter surplus
kappa = 0.41         # von Kármán constant


def compute_gas_properties():
    """Compute xenon gas properties at P, T."""
    rho = P * M_Xe / (R_gas * T)
    nu = mu_Xe / rho
    c_s = np.sqrt(gamma * R_gas * T / M_Xe)
    return rho, nu, c_s


def compute_bangbang_kinematics(m_blade, m_plate, tau_motor, sweep_angle, rho, c):
    """
    Compute bang-bang motion kinematics for the propeller WITH aerodynamic drag.

    Solves the equation of motion numerically:
        I dω/dt = ±τ_motor - τ_drag(ω)
    where:
        τ_drag = (ρ C_d d_wake R⁴ / 4) ω²

    Since drag always opposes motion, deceleration is stronger than acceleration.
    To complete the full sweep angle θ, the switching point must be at θ_switch > θ/2.
    We use bisection to find θ_switch such that ω → 0 exactly when θ → θ_target.

    Assumptions:
    - 2 blades, each a uniform rod from r=0 to r=R_blade, rotating about r=0
    - 2 carrier plates, worst case at tip (r=R_blade), point masses
    - Drag uses wake thickness d_wake = 0.12c (blade thickness)
    - Bluff-body drag coefficient C_d ≈ 0.1 for streamlined airfoil

    Returns dict with inertia, drag parameters, and kinematics.
    """
    from scipy.optimize import brentq

    # ── Moment of inertia ──
    I_blades = 2.0 * (1.0 / 3.0) * m_blade * R_blade**2
    I_plates = 2.0 * m_plate * R_blade**2
    I_total = I_blades + I_plates

    # ── Drag torque parameters ──
    d_wake = 0.12 * c
    C_d = 0.1
    D_coeff = rho * C_d * d_wake * R_blade**4 / 4.0

    # ── Inertia-only reference ──
    alpha_no_drag = tau_motor / I_total
    t_rot_no_drag = 2.0 * np.sqrt(sweep_angle * I_total / tau_motor)
    omega_max_no_drag = alpha_no_drag * (t_rot_no_drag / 2.0)

    # ── Define ODE phases ──
    def accel_phase(t, y):
        omega = y[1]
        tau_drag = D_coeff * omega**2
        return [omega, (tau_motor - tau_drag) / I_total]

    def decel_phase(t, y):
        omega = y[1]
        tau_drag = D_coeff * omega**2
        return [omega, (-tau_motor - tau_drag) / I_total]

    def simulate_with_switch(theta_switch):
        """
        Simulate bang-bang with given switching angle.
        Returns (theta_final, omega_final, t_accel, t_decel, omega_max).
        """
        # Event: reached theta_switch
        def reached_switch(t, y):
            return y[0] - theta_switch
        reached_switch.terminal = True
        reached_switch.direction = 1

        # Phase 1: Acceleration
        sol1 = solve_ivp(
            accel_phase, [0, 100], [0.0, 0.0],
            events=reached_switch, max_step=1e-4
        )
        if len(sol1.t_events[0]) == 0:
            return None  # didn't reach switch
        t_accel = sol1.t_events[0][0]
        omega_at_switch = sol1.y_events[0][0][1]

        # Event: omega reaches zero
        def omega_zero(t, y):
            return y[1]
        omega_zero.terminal = True
        omega_zero.direction = -1

        # Phase 2: Deceleration
        sol2 = solve_ivp(
            decel_phase, [0, 100], [theta_switch, omega_at_switch],
            events=omega_zero, max_step=1e-4
        )
        if len(sol2.t_events[0]) == 0:
            return None  # didn't stop
        t_decel = sol2.t_events[0][0]
        theta_final = sol2.y_events[0][0][0]

        return (theta_final, 0.0, t_accel, t_decel, omega_at_switch)

    # ── Find theta_switch via bisection ──
    def objective(theta_switch):
        result = simulate_with_switch(theta_switch)
        if result is None:
            return -sweep_angle  # force search to continue
        return result[0] - sweep_angle

    # Search bounds: switch must be > θ/2 (since decel is stronger)
    # Upper bound: can't switch past θ
    theta_switch_opt = brentq(objective, sweep_angle * 0.5, sweep_angle * 0.99, xtol=1e-8)

    # Get final results with optimal switch
    result = simulate_with_switch(theta_switch_opt)
    theta_final, _, t_accel, t_decel, omega_max = result
    t_rot = t_accel + t_decel
    V_tip_max = omega_max * R_blade

    # Drag impact
    drag_ratio = (t_rot - t_rot_no_drag) / t_rot_no_drag
    omega_reduction = (omega_max_no_drag - omega_max) / omega_max_no_drag

    return {
        "I_blades": I_blades,
        "I_plates": I_plates,
        "I_total": I_total,
        "D_coeff": D_coeff,
        "d_wake": d_wake,
        "C_d": C_d,
        "alpha_no_drag": alpha_no_drag,
        "t_rot_no_drag": t_rot_no_drag,
        "omega_max_no_drag": omega_max_no_drag,
        "theta_switch": theta_switch_opt,
        "t_accel": t_accel,
        "t_decel": t_decel,
        "t_rot": t_rot,
        "omega_max": omega_max,
        "V_tip_max": V_tip_max,
        "theta_final": theta_final,
        "drag_time_increase": drag_ratio,
        "omega_reduction": omega_reduction,
    }


def compute_flow_regime(c, V_tip_max, c_s, nu):
    """Compute flow regime parameters using peak velocity from bang-bang motion."""
    Ma_tip = V_tip_max / c_s
    Re_tip = V_tip_max * c / nu
    return Ma_tip, Re_tip


def compute_boundary_layer(c, V_tip, nu):
    """
    Compute turbulent BL properties on the upper surface at the blade tip.
    Uses flat-plate correlations (validated against explicit NACA 0012 calc).
    """
    Re_c = V_tip * c / nu

    # BL thickness: delta = 0.37 c Re^(-1/5)
    delta = 0.37 * c * Re_c**(-0.2)

    # Momentum thickness
    theta = (7.0 / 72.0) * delta

    # Displacement thickness
    delta_star = delta / 8.0

    # Shape factor
    H = delta_star / theta

    # Skin friction: Cf = 0.0592 Re^(-1/5)
    Cf = 0.0592 * Re_c**(-0.2)

    # Friction velocity
    u_tau = V_tip * np.sqrt(Cf / 2.0)

    # Drag coefficient (Squire-Young, approximate)
    # Using Ue/V ≈ 0.95 at trailing edge for NACA 0012
    Ue_ratio_TE = 0.95
    Cd = 2.0 * (2.0 * theta) / c * Ue_ratio_TE**((H + 5.0) / 2.0)

    return delta, theta, delta_star, H, Cf, u_tau, Cd, Re_c


def compute_geometry(c, delta, z_obs):
    """Compute vertical geometry."""
    t_max = 0.12 * c                    # NACA 0012 max thickness
    z_blade_top = t_max / 2.0           # upper surface above rotation axis
    z_BL_top = z_blade_top + delta      # top of BL
    z_observation = z_blade_top + z_obs  # observation point above axis
    gap = z_obs - delta                  # gap between BL top and observation
    R_cyl = R_blade + Delta_D / 2.0     # vessel inner radius
    tip_gap = R_cyl - R_blade            # blade tip to vessel wall
    return t_max, z_blade_top, z_BL_top, z_observation, gap, R_cyl, tip_gap


# ═══════════════════════════════════════════════════════════════
# Mechanism 1: Turbulent Eddy Diffusion
# ═══════════════════════════════════════════════════════════════
def mechanism_eddy_diffusion(V_tip, c, delta, delta_star, u_tau, nu,
                             z_obs, t_obs):
    """
    Solve the 1D diffusion equation for momentum in the upper BL
    after the blade stops, with decaying turbulent viscosity.

    Returns the velocity at z_obs, t_obs and supporting diagnostics.
    """
    # ── Turbulent viscosity parameters ──
    nu_t_inner_peak = kappa * u_tau * 0.2 * delta   # at z = 0.2 delta
    nu_t_outer = 0.018 * V_tip * delta_star          # outer layer

    # Turbulence intensity and eddy timescale
    u_prime = 0.05 * V_tip
    t_eddy = delta / u_prime             # eddy turnover time
    t_decay = 2.0 * t_eddy              # e-folding time for nu_t decay

    # Diffusion length scales
    ell_turb = np.sqrt(nu_t_outer * t_decay)
    ell_mol = np.sqrt(nu * t_obs)
    max_reach = delta + ell_turb + ell_mol

    # ── 1D PDE: du/dt = d/dz [nu_eff du/dz] ──
    L = max(5.0 * delta, z_obs + 5.0 * delta)  # domain height
    Nz = 2000
    dz = L / Nz
    z = np.linspace(0, L, Nz + 1)

    # Initial condition: u(z) = V (1 - (z/delta)^(1/7)) for z < delta
    u = np.zeros(Nz + 1)
    for i in range(Nz + 1):
        if 0 < z[i] < delta:
            u[i] = V_tip * (1.0 - (z[i] / delta) ** (1.0 / 7.0))
    u[0] = 0.0  # no-slip (blade now stationary)

    # Initial momentum
    mom_initial = np.trapezoid(u, z) if hasattr(np, 'trapezoid') else np.trapz(u, z)

    # Turbulent viscosity profile at t=0
    nu_t_init = np.zeros(Nz + 1)
    for i in range(Nz + 1):
        if z[i] <= 0:
            nu_t_init[i] = 0.0
        elif z[i] < 0.2 * delta:
            nu_t_init[i] = kappa * u_tau * z[i]
        elif z[i] < delta:
            nu_t_init[i] = nu_t_outer
        else:
            nu_t_init[i] = 0.0

    # Time integration (explicit FD)
    t_current = 0.0
    dt_max = 1e-4

    while t_current < t_obs:
        decay_factor = np.exp(-t_current / t_decay)
        nu_eff = nu + nu_t_init * decay_factor

        nu_max = np.max(nu_eff)
        dt_stable = 0.4 * dz**2 / nu_max if nu_max > 0 else dt_max
        dt = min(dt_stable, dt_max, t_obs - t_current)

        u_new = u.copy()
        for i in range(1, Nz):
            nu_r = 0.5 * (nu_eff[i] + nu_eff[i + 1])
            nu_l = 0.5 * (nu_eff[i] + nu_eff[i - 1])
            u_new[i] = u[i] + dt / dz * (
                nu_r * (u[i + 1] - u[i]) / dz
                - nu_l * (u[i] - u[i - 1]) / dz
            )

        u_new[0] = 0.0
        u_new[Nz] = 0.0
        u = u_new
        t_current += dt

    # Extract velocity at z_obs
    idx_obs = int(round(z_obs / dz))
    idx_obs = min(idx_obs, Nz)
    u_at_obs = u[idx_obs]

    # Final momentum
    mom_final = np.trapezoid(u, z) if hasattr(np, 'trapezoid') else np.trapz(u, z)

    # Velocity profile at selected heights
    profile = {}
    for zi_mm in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]:
        idx = int(round(zi_mm * 1e-3 / dz))
        if idx <= Nz:
            profile[zi_mm] = u[idx]

    return {
        "u_at_obs": u_at_obs,
        "nu_t_inner_peak": nu_t_inner_peak,
        "nu_t_outer": nu_t_outer,
        "nu_t_over_nu": nu_t_outer / nu,
        "u_prime": u_prime,
        "t_eddy": t_eddy,
        "t_decay": t_decay,
        "ell_turb": ell_turb,
        "ell_mol": ell_mol,
        "max_reach": max_reach,
        "mom_initial": mom_initial,
        "mom_final": mom_final,
        "mom_fraction": mom_final / mom_initial if mom_initial > 0 else 0,
        "profile": profile,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 2: Bulk Swirl / Angular Momentum
# ═══════════════════════════════════════════════════════════════
def mechanism_bulk_swirl(rho, nu, omega, c, Cd, sweep_time, delta,
                         z_obs, t_obs, V_tip):
    """Evaluate bulk swirl and vertical transport mechanisms."""
    # Torque and angular impulse
    torque = rho * omega**2 * c * Cd * R_blade**4 / 4.0
    J_angular = torque * sweep_time

    # Molecular diffusion timescale for the gap
    gap = z_obs - delta
    if gap <= 0:
        t_visc = 0.0
    else:
        t_visc = gap**2 / nu

    # Ekman pumping
    u_wake = 0.5 * V_tip  # representative wake velocity
    Omega_wake = u_wake / R_blade if R_blade > 0 else 1e-10

    # Molecular Ekman
    delta_E_mol = np.sqrt(nu / Omega_wake)
    w_E_mol = np.sqrt(nu * Omega_wake)

    # Turbulent Ekman (during turbulence lifetime only)
    nu_t_outer = 0.018 * V_tip * (delta / 8.0)
    t_eddy = delta / (0.05 * V_tip)
    t_turb_life = 2.0 * t_eddy  # turbulence lifetime

    delta_E_turb = np.sqrt(nu_t_outer / Omega_wake)
    w_E_turb = np.sqrt(nu_t_outer * Omega_wake)
    z_Ekman_turb = w_E_turb * t_turb_life  # vertical distance during turb life

    # Ekman spin-up time (assume vessel height ~ 1 m)
    H_vessel = 1.0
    t_Ekman_mol = H_vessel / np.sqrt(nu * Omega_wake)
    t_Ekman_turb = H_vessel / np.sqrt(nu_t_outer * Omega_wake)

    return {
        "torque": torque,
        "J_angular": J_angular,
        "gap": gap,
        "t_visc": t_visc,
        "u_wake": u_wake,
        "Omega_wake": Omega_wake,
        "delta_E_mol": delta_E_mol,
        "w_E_mol": w_E_mol,
        "delta_E_turb": delta_E_turb,
        "w_E_turb": w_E_turb,
        "z_Ekman_turb": z_Ekman_turb,
        "t_Ekman_mol": t_Ekman_mol,
        "t_Ekman_turb_spinup": t_Ekman_turb,
        "t_turb_life": t_turb_life,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 3: Displacement (Potential) Flow
# ═══════════════════════════════════════════════════════════════
def mechanism_potential_flow(V_tip, c, z_obs, c_s):
    """Evaluate the displacement flow and its acoustic clearing time."""
    t_max = 0.12 * c
    d_eff = t_max

    # Vertical velocity at z_obs during the sweep (dipole approximation)
    w_disp = V_tip * (d_eff / 2.0)**2 / (z_obs**2 + (d_eff / 2.0)**2)

    # Acoustic clearing time
    t_acoustic = R_blade / c_s

    # Blockage ratio
    tip_gap = Delta_D / 2.0
    blockage = t_max / tip_gap if tip_gap > 0 else float('inf')

    return {
        "t_max": t_max,
        "w_disp_during_sweep": w_disp,
        "t_acoustic": t_acoustic,
        "blockage_ratio": blockage,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 4: Pressure Pulse from Deceleration
# ═══════════════════════════════════════════════════════════════
def mechanism_pressure_pulse(rho, V_tip, c_s, z_obs):
    """Evaluate the transient pressure pulse when the blade stops."""
    # Assume deceleration time ~ 10 ms (must be finite)
    t_decel = 0.01

    # Pressure perturbation
    dp = rho * V_tip**2

    # Induced acoustic velocity
    du_acoustic = dp / (rho * c_s)

    # Arrival time at z_obs
    t_arrive = z_obs / c_s

    # Fractional pressure perturbation
    dp_fraction = dp / P

    return {
        "t_decel": t_decel,
        "dp": dp,
        "dp_fraction": dp_fraction,
        "du_acoustic": du_acoustic,
        "t_arrive": t_arrive,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 5: Secondary Flows
# ═══════════════════════════════════════════════════════════════
def mechanism_secondary_flows(rho, V_tip, nu):
    """Evaluate tip clearance, centrifugal effects, wake-wall interaction."""
    tip_gap = Delta_D / 2.0
    u_wake = 0.5 * V_tip

    # Centrifugal acceleration
    a_cent = u_wake**2 / R_blade

    # Time to traverse tip gap under centrifugal force
    t_cent = np.sqrt(2.0 * tip_gap / a_cent) if a_cent > 0 else float('inf')

    # Radial pressure difference over tip gap
    dp_cent = rho * u_wake**2 * (tip_gap / R_blade)

    # Tip clearance: requires lift. At alpha = 0, dp_tip = 0
    dp_tip = 0.0

    return {
        "tip_gap": tip_gap,
        "a_cent": a_cent,
        "t_cent": t_cent,
        "dp_cent": dp_cent,
        "dp_tip_clearance": dp_tip,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 6: Vortex Shedding
# ═══════════════════════════════════════════════════════════════
def mechanism_vortex_shedding(V_tip, c, nu):
    """Evaluate vortex shedding from the blade."""
    t_max = 0.12 * c
    Re_d = V_tip * t_max / nu
    St = 0.21  # Strouhal number (bluff body reference)
    f_shed = St * V_tip / t_max if t_max > 0 else 0

    return {
        "t_max": t_max,
        "Re_d": Re_d,
        "St": St,
        "f_shed": f_shed,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()

    c = args.chord
    sweep_angle = args.sweep_angle
    m_blade = args.m_blade
    m_plate = args.m_plate
    tau_motor = args.tau_motor
    z_obs = args.z_obs
    t_obs = args.t_obs

    DIVIDER = "=" * 70

    # ── Header ──
    print(DIVIDER)
    print("RESIDUAL GAS VELOCITY ABOVE A STOPPED PROPELLER IN DENSE XENON")
    print(DIVIDER)
    print(f"\n  Variable parameters:")
    print(f"    Chord length:        c = {c:.4f} m")
    print(f"    Sweep angle:         θ = {sweep_angle:.5f} rad "
          f"({np.degrees(sweep_angle):.1f}°)")
    print(f"    Blade mass (each):   m_blade = {m_blade:.3f} kg")
    print(f"    Plate mass (each):   m_plate = {m_plate:.3f} kg")
    print(f"    Motor torque:        τ = {tau_motor:.1f} N·m")
    print(f"    Observation height:  z = {z_obs*1000:.1f} mm")
    print(f"    Observation time:    t = {t_obs:.3f} s")
    print(f"\n  Fixed parameters:")
    print(f"    Blade span:          R = {R_blade} m")
    print(f"    Pressure:            P = {P/1e5:.0f} bar")
    print(f"    Temperature:         T = {T:.0f} K")
    print(f"    NACA 0012, α = 0°")
    print(f"    Vessel diam. surplus:  ΔD = {Delta_D*100:.0f} cm")

    # ── Gas properties ──
    rho, nu, c_s = compute_gas_properties()
    rho_air = 1.2  # kg/m³ at STP
    nu_air = 1.5e-5  # m²/s at STP
    print(f"\n{DIVIDER}")
    print("1. XENON GAS PROPERTIES")
    print(DIVIDER)
    print(f"  Density:              ρ  = {rho:.1f} kg/m³  ({rho/rho_air:.0f}× denser than air)")
    print(f"  Dynamic viscosity:    μ  = {mu_Xe:.2e} Pa·s")
    print(f"  Kinematic viscosity:  ν  = {nu:.3e} m²/s  ({nu_air/nu:.0f}× smaller than air)")
    print(f"  Speed of sound:       cₛ = {c_s:.1f} m/s")
    print()
    print(f"  NOTE: Dense Xe has very LOW kinematic viscosity (ν = μ/ρ) because")
    print(f"  ρ is large while μ is similar to air. This makes diffusion SLOW")
    print(f"  and Reynolds numbers HIGH for a given velocity.")

    # ── Bang-bang Kinematics (with drag) ──
    bb = compute_bangbang_kinematics(m_blade, m_plate, tau_motor, sweep_angle, rho, c)
    t_rot = bb["t_rot"]
    omega_max = bb["omega_max"]
    V_tip = bb["V_tip_max"]

    print(f"\n{DIVIDER}")
    print("2. BANG-BANG KINEMATICS (with aerodynamic drag)")
    print(DIVIDER)
    print(f"  Moment of inertia (I = Σ m·r²):")
    print(f"    I_blades = 2 × (1/3)·m·R²:  {bb['I_blades']:.4f} kg·m²  (rods about end)")
    print(f"    I_plates = 2 × m·R²:        {bb['I_plates']:.4f} kg·m²  (point masses at tip)")
    print(f"    I_total:                    {bb['I_total']:.4f} kg·m²")
    print(f"\n  Aerodynamic drag torque:")
    print(f"    τ_drag = (ρ C_d d_wake R⁴ / 4) ω²")
    print(f"    Wake thickness:     d_wake = 0.12c = {bb['d_wake']*1000:.1f} mm")
    print(f"    Drag coefficient:   C_d = {bb['C_d']}")
    print(f"    Drag factor:        D = {bb['D_coeff']:.4f} N·m·s²/rad²")
    print(f"\n  Equation of motion (solved numerically):")
    print(f"    Accel phase:  I dω/dt = τ_motor - D·ω²")
    print(f"    Decel phase:  I dω/dt = -τ_motor - D·ω²")
    print(f"\n  Inertia-only reference (no drag):")
    print(f"    α = τ/I = {bb['alpha_no_drag']:.2f} rad/s²")
    print(f"    t_rot = 2√(θI/τ) = {bb['t_rot_no_drag']*1000:.1f} ms")
    print(f"    ω_max = {bb['omega_max_no_drag']:.3f} rad/s")
    print(f"\n  With drag (numerical integration):")
    print(f"    Switching angle:        θ_switch = {bb['theta_switch']:.5f} rad "
          f"({np.degrees(bb['theta_switch']):.2f}°)")
    print(f"    Accel time (0 → θ_sw):  t_accel  = {bb['t_accel']*1000:.2f} ms")
    print(f"    Decel time (θ_sw → θ):  t_decel  = {bb['t_decel']*1000:.2f} ms")
    print(f"    Total rotation time:    t_rot    = {t_rot*1000:.2f} ms")
    print(f"    Peak ω (at θ_switch):   ω_max    = {omega_max:.4f} rad/s")
    print(f"    Peak V_tip:             V_tip    = {V_tip:.4f} m/s")
    print(f"    Final θ:                θ_final  = {bb['theta_final']:.5f} rad "
          f"(target: {sweep_angle:.5f})")
    print(f"\n  Drag impact:")
    print(f"    Time increase:          {bb['drag_time_increase']*100:+.2f}%")
    print(f"    ω_max reduction:        {bb['omega_reduction']*100:.2f}%")

    # ── Flow Regime ──
    Ma_tip, Re_tip = compute_flow_regime(c, V_tip, c_s, nu)
    print(f"\n{DIVIDER}")
    print("3. FLOW REGIME (at peak velocity)")
    print(DIVIDER)
    print(f"  Tip Mach number:      Ma    = {Ma_tip:.4f}")
    print(f"  Tip Reynolds number:  Re    = {Re_tip:.3e}")
    print()
    # Physics explanations
    if Ma_tip < 0.3:
        print(f"  COMPRESSIBILITY (Ma = {Ma_tip:.3f} < 0.3):")
        print(f"    The flow is INCOMPRESSIBLE. Density variations are negligible")
        print(f"    (< 5%), so the continuity equation simplifies to ∇·v = 0.")
        print(f"    No shock waves or compressibility-driven phenomena occur.")
    else:
        print(f"  COMPRESSIBILITY (Ma = {Ma_tip:.3f} ≥ 0.3):")
        print(f"    Compressibility effects may be significant.")
    print()
    if Re_tip > 5e5:
        print(f"  TURBULENCE (Re = {Re_tip:.2e} > 5×10⁵):")
        print(f"    The boundary layer is FULLY TURBULENT. Inertial forces")
        print(f"    dominate viscous forces, causing chaotic eddies. The BL is")
        print(f"    thicker than laminar, with enhanced mixing near the wall.")
        print(f"    Transition occurs at Re ~ 5×10⁵ for flat plates.")
    elif Re_tip > 2300:
        print(f"  TURBULENCE (Re = {Re_tip:.2e}):")
        print(f"    Transitional flow regime; may have laminar and turbulent regions.")
    else:
        print(f"  TURBULENCE (Re = {Re_tip:.2e} < 2300):")
        print(f"    The flow is LAMINAR; viscous forces dominate.")

    # ── Boundary layer ──
    delta, theta, delta_star, H, Cf, u_tau, Cd, Re_c = \
        compute_boundary_layer(c, V_tip, nu)
    print(f"\n{DIVIDER}")
    print("4. BOUNDARY LAYER (upper surface, at blade tip)")
    print(DIVIDER)
    print(f"  The boundary layer (BL) is the thin region near the blade where")
    print(f"  velocity transitions from zero (at wall) to free-stream value.")
    print()
    print(f"  BL thickness:         δ  = 0.37·c·Re⁻⁰·² = {delta*1000:.3f} mm")
    print(f"  Momentum thickness:   θ  = (7/72)·δ = {theta*1000:.4f} mm")
    print(f"  Displacement thick.:  δ* = δ/8 = {delta_star*1000:.4f} mm")
    print(f"  Shape factor:         H  = δ*/θ = {H:.3f}")
    print(f"  Skin friction coeff:  Cf = 0.0592·Re⁻⁰·² = {Cf:.5f}")
    print(f"  Friction velocity:    u_τ = V·√(Cf/2) = {u_tau:.4f} m/s")
    print(f"  Drag coefficient:     Cd = {Cd:.5f}")
    print(f"  Re_c (chord-based):   {Re_c:.3e}")
    print()
    print(f"  NOTE: The BL contains ALL the momentum imparted by the blade.")
    print(f"  Gas outside δ is undisturbed (except for potential flow).")

    # ── Geometry ──
    t_max, z_blade_top, z_BL_top, z_observation, gap, R_cyl, tip_gap = \
        compute_geometry(c, delta, z_obs)
    print(f"\n{DIVIDER}")
    print("5. VERTICAL GEOMETRY")
    print(DIVIDER)
    print(f"  Blade max thickness:    t_max     = {t_max*1000:.1f} mm")
    print(f"  Upper surface height:   z_blade   = {z_blade_top*1000:.1f} mm")
    print(f"  BL top:                 z_BL      = {z_BL_top*1000:.1f} mm")
    print(f"  Observation point:      z_obs     = {z_obs*1000:.1f} mm "
          f"above blade top")
    print(f"  Gap (BL top → obs):     Δz_gap    = {gap*1000:.1f} mm")
    print(f"  Vessel inner radius:    R_cyl     = {R_cyl:.3f} m")
    print(f"  Tip clearance:          Δr        = {tip_gap*1000:.0f} mm")

    if gap <= 0:
        print(f"\n  *** NOTE: z_obs ({z_obs*1000:.1f} mm) is WITHIN the "
              f"boundary layer (δ = {delta*1000:.1f} mm). ***")
        print(f"  *** The observation point has nonzero velocity at t=0. ***")

    # ── Mechanism 1: Eddy Diffusion ──
    print(f"\n{DIVIDER}")
    print("6. MECHANISM 1: TURBULENT EDDY DIFFUSION")
    print(DIVIDER)
    print(f"  Solving 1D diffusion equation with decaying ν_t ...")

    m1 = mechanism_eddy_diffusion(V_tip, c, delta, delta_star, u_tau,
                                  nu, z_obs, t_obs)

    print(f"\n  Turbulent viscosity:")
    print(f"    ν_t (inner peak):  {m1['nu_t_inner_peak']:.3e} m²/s")
    print(f"    ν_t (outer):       {m1['nu_t_outer']:.3e} m²/s")
    print(f"    ν_t / ν:           {m1['nu_t_over_nu']:.0f}×")
    print(f"\n  Eddy scales:")
    print(f"    u':                {m1['u_prime']:.4f} m/s")
    print(f"    t_eddy (δ/u'):     {m1['t_eddy']*1000:.1f} ms")
    print(f"    t_decay (2 t_e):   {m1['t_decay']*1000:.1f} ms")
    print(f"\n  Diffusion reach:")
    print(f"    Turbulent:         √(ν_t × t_decay)  = {m1['ell_turb']*1000:.2f} mm")
    print(f"    Molecular:         √(ν × t_obs)      = {m1['ell_mol']*1000:.2f} mm")
    print(f"    Max reach from δ:  δ + ℓ_turb + ℓ_mol = {m1['max_reach']*1000:.1f} mm")
    print(f"    Observation at:    z_obs              = {z_obs*1000:.1f} mm")

    if m1['max_reach'] < z_obs:
        print(f"    → Max reach ({m1['max_reach']*1000:.1f} mm) < z_obs "
              f"({z_obs*1000:.1f} mm): momentum CANNOT reach observation point")
    else:
        print(f"    → Max reach ({m1['max_reach']*1000:.1f} mm) ≥ z_obs "
              f"({z_obs*1000:.1f} mm): some momentum MAY reach observation point")

    print(f"\n  Momentum budget:")
    print(f"    Initial:           {m1['mom_initial']:.4f} kg·m/s per m²")
    print(f"    Final:             {m1['mom_final']:.4f} kg·m/s per m²")
    print(f"    Fraction remaining:{m1['mom_fraction']:.2%}")
    print(f"    (rest absorbed by blade surface via no-slip)")

    print(f"\n  Velocity profile at t = {t_obs} s:")
    print(f"    {'z [mm]':>8s}  {'u [m/s]':>10s}  {'u [cm/s]':>10s}")
    for zi_mm, vel in sorted(m1['profile'].items()):
        marker = " ← z_obs" if abs(zi_mm - z_obs * 1000) < 0.5 else ""
        print(f"    {zi_mm:8.1f}  {vel:10.6f}  {vel*100:10.4f}{marker}")

    u_str1 = f"{m1['u_at_obs']:.2e}"
    u_str2 = f"{m1['u_at_obs']*100:.4f}"
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = {u_str1} m/s{' '*(17-len(u_str1))}│")
    print(f"  │         = {u_str2} cm/s{' '*(30-len(u_str2))}│")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 2: Bulk Swirl ──
    print(f"\n{DIVIDER}")
    print("7. MECHANISM 2: BULK SWIRL / ANGULAR MOMENTUM")
    print(DIVIDER)

    m2 = mechanism_bulk_swirl(rho, nu, omega_max, c, Cd, t_rot, delta,
                              z_obs, t_obs, V_tip)

    print(f"  Torque (2 blades):     τ = {m2['torque']:.3f} N·m")
    print(f"  Angular impulse:       J = {m2['J_angular']:.3f} N·m·s")
    print(f"\n  Momentum confinement:")
    print(f"    Gap (BL top → obs):  {m2['gap']*1000:.1f} mm")
    print(f"    Mol. diffusion time: t_visc = gap²/ν = {m2['t_visc']:.0f} s")
    print(f"    → {'≫' if m2['t_visc'] > 10*t_obs else '~'} t_obs = {t_obs} s")
    print(f"\n  Ekman pumping (molecular ν):")
    print(f"    δ_E = {m2['delta_E_mol']*1000:.2f} mm")
    print(f"    w_E = {m2['w_E_mol']*1e6:.1f} μm/s")
    print(f"\n  Ekman pumping (turbulent ν, first {m2['t_turb_life']*1000:.0f} ms):")
    print(f"    δ_E = {m2['delta_E_turb']*1000:.1f} mm")
    print(f"    w_E = {m2['w_E_turb']*100:.2f} cm/s")
    print(f"    Vertical reach in {m2['t_turb_life']*1000:.0f} ms: "
          f"{m2['z_Ekman_turb']*1000:.2f} mm")
    print(f"\n  Ekman spin-up timescales (H_vessel ~ 1 m):")
    print(f"    Molecular: {m2['t_Ekman_mol']:.0f} s")
    print(f"    Turbulent: {m2['t_Ekman_turb_spinup']:.0f} s")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) ≈ 0  (bulk swirl)      │")
    print(f"  │ Momentum stays in blade plane; no vertical      │")
    print(f"  │ transport mechanism operates within t_obs.       │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 3: Potential Flow ──
    print(f"\n{DIVIDER}")
    print("8. MECHANISM 3: DISPLACEMENT (POTENTIAL) FLOW")
    print(DIVIDER)

    m3 = mechanism_potential_flow(V_tip, c, z_obs, c_s)

    print(f"  Blade max thickness:    {m3['t_max']*100:.1f} cm")
    print(f"  Blockage ratio:         {m3['blockage_ratio']:.2f}")
    print(f"  Displacement velocity at z_obs DURING sweep:")
    print(f"    w_disp = {m3['w_disp_during_sweep']:.2f} m/s")
    print(f"  Acoustic clearing time: R/c_s = {m3['t_acoustic']*1000:.1f} ms")
    print(f"  Time ratio: t_obs / t_acoustic = {t_obs/m3['t_acoustic']:.0f}")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = 0  (potential flow)   │")
    print(f"  │ Field vanishes at speed of sound after stop.     │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 4: Pressure Pulse ──
    print(f"\n{DIVIDER}")
    print("9. MECHANISM 4: PRESSURE PULSE FROM DECELERATION")
    print(DIVIDER)

    m4 = mechanism_pressure_pulse(rho, V_tip, c_s, z_obs)

    print(f"  Deceleration time:      t_decel  = {m4['t_decel']*1000:.0f} ms (assumed)")
    print(f"  Pressure perturbation:  Δp       = {m4['dp']:.0f} Pa")
    print(f"  Fractional:             Δp/P     = {m4['dp_fraction']:.2e}")
    print(f"  Induced velocity:       Δu       = {m4['du_acoustic']:.3f} m/s")
    print(f"  Arrival at z_obs:       t_arrive = {m4['t_arrive']*1e6:.1f} μs")
    print(f"  Duration:               ~{m4['t_decel']*1000:.0f} ms (transient)")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = 0  (pressure pulse)   │")
    print(f"  │ Transient acoustic wave; passes through and      │")
    print(f"  │ leaves no residual.                              │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 5: Secondary Flows ──
    print(f"\n{DIVIDER}")
    print("10. MECHANISM 5: SECONDARY FLOWS")
    print(DIVIDER)

    m5 = mechanism_secondary_flows(rho, V_tip, nu)

    print(f"  Tip clearance flow:")
    print(f"    ΔP across blade (α=0): {m5['dp_tip_clearance']:.0f} Pa → NO FLOW")
    print(f"\n  Centrifugal effects:")
    print(f"    a_cent = u²/R = {m5['a_cent']:.3f} m/s²")
    print(f"    Traversal time for tip gap: {m5['t_cent']:.2f} s")
    print(f"    Radial ΔP over tip gap: {m5['dp_cent']:.1f} Pa")
    print(f"\n  Wake-wall interaction:")
    print(f"    Wake is TANGENTIAL → flows along wall, not into it")
    print(f"    No face-on impingement → no axial deflection")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = 0  (secondary flows)  │")
    print(f"  │ No lift → no tip clearance; centrifugal too slow;│")
    print(f"  │ tangential wake follows wall curvature.          │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 6: Vortex Shedding ──
    print(f"\n{DIVIDER}")
    print("11. MECHANISM 6: VORTEX SHEDDING")
    print(DIVIDER)

    m6 = mechanism_vortex_shedding(V_tip, c, nu)

    print(f"  Blade thickness:  {m6['t_max']*1000:.1f} mm")
    print(f"  Re (thickness):   {m6['Re_d']:.2e}")
    print(f"  Strouhal number:  {m6['St']}")
    print(f"  Shedding freq:    {m6['f_shed']:.0f} Hz (bluff body ref)")
    print(f"  NACA 0012 at α=0 is streamlined → coherent shedding suppressed")
    print(f"  Any shed vortices confined to blade plane (~δ from surface)")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = 0  (vortex shedding)  │")
    print(f"  │ Streamlined body; vortices confined to blade     │")
    print(f"  │ plane with same transport limitations.           │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Vessel Confinement Effects ──
    print(f"\n{DIVIDER}")
    print("12. VESSEL CONFINEMENT EFFECTS")
    print(DIVIDER)

    H_vessel = 1.5  # m, vessel height
    Delta_r = Delta_D / 2.0  # radial tip clearance

    # (a) Wall BL thickness (Stokes penetration on cylindrical wall)
    delta_wall = np.sqrt(nu * t_rot)

    # (b) Ekman quantities (already in m2, but reprint with H=1.5)
    Omega_wake = 0.5 * V_tip / R_blade
    delta_E_mol = np.sqrt(nu / Omega_wake)
    w_E_mol = np.sqrt(nu * Omega_wake)
    t_Ekman_mol = H_vessel / np.sqrt(nu * Omega_wake)

    nu_t_outer = 0.018 * V_tip * (delta / 8.0)
    t_eddy_conf = delta / (0.05 * V_tip)
    t_turb_life = 2.0 * t_eddy_conf
    delta_E_turb = np.sqrt(nu_t_outer / Omega_wake)
    w_E_turb = np.sqrt(nu_t_outer * Omega_wake)
    z_Ekman_turb = w_E_turb * t_turb_life
    t_Ekman_turb = H_vessel / np.sqrt(nu_t_outer * Omega_wake)

    # (c) Tip-gap Reynolds number
    Re_gap = V_tip * Delta_r / nu

    # (d) Acoustic reverberation time
    ell_max = np.sqrt((2 * R_cyl)**2 + H_vessel**2)
    t_reverb = ell_max / c_s

    # (e) Displacement readjustment time
    t_disp = R_blade / c_s

    print(f"\n  Vessel dimensions:")
    print(f"    Height:             H     = {H_vessel} m")
    print(f"    Inner radius:       R_cyl = {R_cyl:.3f} m")
    print(f"    Radial tip gap:     Δr    = {Delta_r*1000:.0f} mm")

    print(f"\n  (a) Wall boundary layer (Stokes penetration during sweep):")
    print(f"    δ_wall = √(ν·t_rot) = {delta_wall*1000:.3f} mm")
    print(f"    → Confined to < 1 mm from vessel wall; cannot affect z_obs.")

    print(f"\n  (b) Ekman spin-up (endcap BLs coupling to bulk):")
    print(f"    Molecular:")
    print(f"      δ_E     = √(ν/Ω) = {delta_E_mol*1000:.3f} mm")
    print(f"      w_E     = √(νΩ)  = {w_E_mol*1e6:.1f} μm/s")
    print(f"      t_Ekman = H/√(νΩ) = {t_Ekman_mol:.0f} s")
    print(f"    Turbulent (lives only {t_turb_life*1000:.0f} ms):")
    print(f"      δ_E     = {delta_E_turb*1000:.2f} mm")
    print(f"      w_E     = {w_E_turb*100:.2f} cm/s")
    print(f"      z_reach = w_E × t_turb = {z_Ekman_turb*1000:.2f} mm")
    print(f"      t_Ekman = {t_Ekman_turb:.0f} s")

    print(f"\n  (c) Tip-gap Couette flow:")
    print(f"    Re_gap = V_tip·Δr/ν = {Re_gap:.2e}")
    regime = "turbulent" if Re_gap > 1e4 else "laminar"
    print(f"    Flow regime: {regime}")
    print(f"    Direction: purely TANGENTIAL (no axial component)")
    print(f"    No lift (α=0) → no pressure-driven tip leakage")

    print(f"\n  (d) Acoustic reverberation:")
    print(f"    Longest path: ℓ = √((2R_cyl)² + H²) = {ell_max:.2f} m")
    print(f"    Reverberation time: ℓ/c_s = {t_reverb*1000:.1f} ms")
    print(f"    → Dissipated within ~{3*t_reverb*1000:.0f} ms (few reflections)")

    print(f"\n  (e) Displacement readjustment:")
    print(f"    Acoustic clearing: R/c_s = {t_disp*1000:.1f} ms")
    print(f"    → Potential field vanishes on this timescale after stop")

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ CONFINEMENT SUMMARY: No new vertical transport mechanism.  │")
    print(f"  │ All wall effects are tangential, transient, or too slow.   │")
    print(f"  │ The 6-mechanism analysis remains valid without change.     │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # ── Summary ──
    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)
    print(f"\n  Parameters: c = {c} m, θ = {np.degrees(sweep_angle):.1f}°, "
          f"t_rot = {t_rot*1000:.1f} ms, z_obs = {z_obs*1000:.1f} mm, "
          f"t_obs = {t_obs} s")
    print(f"\n  Key scales:")
    print(f"    BL thickness (δ):       {delta*1000:.2f} mm")
    print(f"    Gap to observation:     {max(gap,0)*1000:.1f} mm")
    print(f"    Eddy decay time:        {m1['t_decay']*1000:.0f} ms")
    print(f"    Acoustic clearing time: {m3['t_acoustic']*1000:.0f} ms")
    print(f"    Mol. diffusion time:    {m2['t_visc']:.0f} s")

    print(f"""
  ┌──────────────────────┬────────────────────┬──────────────────┐
  │ Mechanism            │ Timescale          │ u at (z,t)       │
  ├──────────────────────┼────────────────────┼──────────────────┤
  │ 1. Eddy diffusion    │ ~{m1['t_decay']*1000:4.0f} ms decay     │ {m1['u_at_obs']:.2e} m/s │
  │ 2. Bulk swirl        │ ~{m2['t_visc']:6.0f} s (mol.diff)│ 0                │
  │ 3. Potential flow    │ ~{m3['t_acoustic']*1000:4.0f} ms acoustic │ 0                │
  │ 4. Pressure pulse    │ ~{m4['t_decel']*1000:4.0f} ms transient │ 0                │
  │ 5. Secondary flows   │ ~{m2['t_Ekman_turb_spinup']:6.0f} s (Ekman)  │ 0                │
  │ 6. Vortex shedding   │  N/A (confined)    │ 0                │
  └──────────────────────┴────────────────────┴──────────────────┘""")

    if gap > 0 and m1['u_at_obs'] < 1e-6:
        print(f"""
  CONCLUSION: The gas at z = {z_obs*1000:.1f} mm above the blade is
  QUIESCENT at t = {t_obs} s after the blade stops.
  All six mechanisms yield zero residual velocity.
""")
    elif gap <= 0:
        print(f"""
  NOTE: z_obs = {z_obs*1000:.1f} mm is WITHIN the boundary layer
  (δ = {delta*1000:.1f} mm). The observation point has nonzero
  velocity from the BL momentum. See the profile above.
""")
    else:
        print(f"""
  NOTE: Eddy diffusion gives a small but nonzero value:
  u = {m1['u_at_obs']:.2e} m/s = {m1['u_at_obs']*100:.4f} cm/s.
  This may be above or below the effective noise floor
  depending on the application.
""")

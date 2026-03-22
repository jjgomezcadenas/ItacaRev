#!/usr/bin/env python3
"""
Residual gas velocity above a stopped propeller blade in dense xenon.
Mechanism-by-mechanism numerical evaluation.

Accompanies: propeller_flow_analysis.tex

Usage:
    python propeller_flow_numerics.py [OPTIONS]

Options:
    --n_arms       Number of arms (must be even: 2,4,6,...) (default: 2)
    --chord        Blade chord length [m]             (default: 0.16)
    --m_blade      Mass of each blade [kg]            (default: 0.25)
    --m_plate      Mass of each carrier plate [kg]   (default: 0.25)
    --tau_motor    Motor torque [N·m]                 (default: 60.0)
    --numeric      Run full numerical PDE solver      (default: analytical)
    --drag         Include aerodynamic drag in kinematics (default: no drag)

Fixed parameters:
    R = 1.6 m, P = 15 bar, T = 300 K, NACA 0012 at alpha = 0,
    wire grid 0.2 mm / 2 mm, vessel diameter surplus = 10 cm.

Motion profile:
    Bang-bang (maximum acceleration then maximum deceleration).
    Rotation time t_rot is computed from inertia and motor torque.
    Sweep angle is computed as 2π / n_arms (e.g., 2 arms → π, 4 arms → π/2).

Three-phase motion sequence:
    Phase 1: Blade rotation (t_rot)
    Phase 2: Carrier radial slide (t_slide)
    Phase 3: Vertical plate lift (t_lift) + wait (t_wait)
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc
import sys


# ═══════════════════════════════════════════════════════════════
# Parse command-line arguments
# ═══════════════════════════════════════════════════════════════
def validate_n_arms(value):
    """Validate that n_arms is a positive even integer."""
    ivalue = int(value)
    if ivalue < 2 or ivalue % 2 != 0:
        raise argparse.ArgumentTypeError(
            f"n_arms must be a positive even integer (2, 4, 6, ...), got {value}"
        )
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Residual gas velocity above a stopped propeller blade in Xe."
    )
    parser.add_argument(
        "--n_arms", type=validate_n_arms, default=2,
        help="Number of arms/blades (must be even: 2,4,6,...) (default: 2)"
    )
    parser.add_argument(
        "--chord", type=float, default=0.16,
        help="Blade chord length [m] (default: 0.16)"
    )
    parser.add_argument(
        "--m_blade", type=float, default=0.25,
        help="Mass of each blade [kg] (default: 0.25)"
    )
    parser.add_argument(
        "--m_plate", type=float, default=0.25,
        help="Mass of each carrier plate [kg] (default: 0.25)"
    )
    parser.add_argument(
        "--tau_motor", type=float, default=60.0,
        help="Motor torque [N·m] (default: 60.0)"
    )
    parser.add_argument(
        "--numeric", action="store_true",
        help="Run full numerical PDE solver (default: analytical estimate only)"
    )
    parser.add_argument(
        "--drag", action="store_true",
        help="Include aerodynamic drag in kinematics (default: analytical no-drag)"
    )
    # Phase 2: Carrier slide parameters
    parser.add_argument(
        "--F_actuator", type=float, default=20.0,
        help="Linear actuator force [N] (default: 20.0)"
    )
    parser.add_argument(
        "--Cd_carrier", type=float, default=0.1,
        help="Carrier plate drag coefficient (default: 0.1)"
    )
    parser.add_argument(
        "--r0_carrier", type=float, default=0.8,
        help="Initial radial position of carrier [m] (default: 0.8 = R/2)"
    )
    # Phase 3: Vertical plate lift parameters
    parser.add_argument(
        "--z_lift", type=float, default=0.005,
        help="Height of lifted plate above blade [m] (default: 0.005 = 5mm)"
    )
    parser.add_argument(
        "--t_lift", type=float, default=1.0,
        help="Lift duration [s] (default: 1.0)"
    )
    parser.add_argument(
        "--t_wait", type=float, default=0.5,
        help="Wait time after lift [s] (default: 0.5)"
    )
    parser.add_argument(
        "--u_c", type=float, default=0.0001,
        help="Quiescence cutoff velocity [m/s] (default: 0.0001 = 0.1 mm/s)"
    )
    parser.add_argument(
        "--z_max", type=float, default=0.020,
        help="Safety maximum height for profile [m] (default: 0.020 = 20mm)"
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
Delta_D = 0.01       # m, vessel inner diameter surplus
kappa = 0.41         # von Kármán constant


def compute_gas_properties():
    """Compute xenon gas properties at P, T."""
    rho = P * M_Xe / (R_gas * T)
    nu = mu_Xe / rho
    c_s = np.sqrt(gamma * R_gas * T / M_Xe)
    return rho, nu, c_s


def compute_bangbang_kinematics_analytical(n_arms, m_blade, m_plate, tau_motor, sweep_angle,
                                           r0_carrier=None):
    """
    Compute bang-bang motion kinematics WITHOUT aerodynamic drag (fast analytical).

    For pure bang-bang motion with constant acceleration/deceleration:
        α = τ/I                    (angular acceleration)
        θ_switch = θ/2             (switch at midpoint)
        ω_max = √(θ·τ/I)           (peak angular velocity)
        t_rot = 2√(θ·I/τ)          (total rotation time)
        V_tip = ω_max × R          (peak tip velocity)

    Parameters:
    - n_arms: Number of blades/arms (must be even: 2, 4, 6, ...)
    - m_blade: Mass of each blade [kg]
    - m_plate: Mass of each carrier plate [kg]
    - tau_motor: Motor torque [N·m]
    - sweep_angle: Sweep angle [rad] (typically 2π / n_arms)
    - r0_carrier: Initial radial position of carrier [m] (default: R_blade)

    Returns dict with inertia and kinematics.
    """
    # ── Moment of inertia (scales with n_arms) ──
    # Plates are at r0_carrier, not at blade tip
    r0 = r0_carrier if r0_carrier is not None else R_blade
    I_blades = n_arms * (1.0 / 3.0) * m_blade * R_blade**2
    I_plates = n_arms * m_plate * r0**2
    I_total = I_blades + I_plates

    # ── Closed-form kinematics (no drag) ──
    alpha = tau_motor / I_total
    t_rot = 2.0 * np.sqrt(sweep_angle * I_total / tau_motor)
    omega_max = np.sqrt(sweep_angle * tau_motor / I_total)
    theta_switch = sweep_angle / 2.0
    t_accel = t_rot / 2.0
    t_decel = t_rot / 2.0
    V_tip_max = omega_max * R_blade

    return {
        "n_arms": n_arms,
        "I_blades": I_blades,
        "I_plates": I_plates,
        "I_total": I_total,
        "alpha": alpha,
        "theta_switch": theta_switch,
        "t_accel": t_accel,
        "t_decel": t_decel,
        "t_rot": t_rot,
        "omega_max": omega_max,
        "V_tip_max": V_tip_max,
        "theta_final": sweep_angle,
        "method": "analytical",
    }


def compute_bangbang_kinematics_numerical(n_arms, m_blade, m_plate, tau_motor, sweep_angle,
                                          rho, c, r0_carrier=None):
    """
    Compute bang-bang motion kinematics for the propeller WITH aerodynamic drag.

    Solves the equation of motion numerically:
        I dω/dt = ±τ_motor - τ_drag(ω)
    where:
        τ_drag = n_arms × (ρ C_d d_wake R⁴ / 4) ω²

    Since drag always opposes motion, deceleration is stronger than acceleration.
    To complete the full sweep angle θ, the switching point must be at θ_switch > θ/2.
    We use bisection to find θ_switch such that ω → 0 exactly when θ → θ_target.

    Parameters:
    - n_arms: Number of blades/arms (must be even: 2, 4, 6, ...)
    - m_blade: Mass of each blade [kg]
    - m_plate: Mass of each carrier plate [kg]
    - tau_motor: Motor torque [N·m]
    - sweep_angle: Sweep angle [rad] (typically 2π / n_arms)
    - rho: Gas density [kg/m³]
    - c: Blade chord length [m]
    - r0_carrier: Initial radial position of carrier [m] (default: R_blade)

    Assumptions:
    - n_arms blades, each a uniform rod from r=0 to r=R_blade, rotating about r=0
    - n_arms carrier plates at r0_carrier, point masses
    - Drag uses wake thickness d_wake = 0.12c (blade thickness)
    - Effective drag coefficient C_d ≈ 0.1 for blade+plate assembly

    Returns dict with inertia, drag parameters, and kinematics.
    """
    from scipy.optimize import brentq

    # ── Moment of inertia (scales with n_arms) ──
    # Plates are at r0_carrier, not at blade tip
    r0 = r0_carrier if r0_carrier is not None else R_blade
    I_blades = n_arms * (1.0 / 3.0) * m_blade * R_blade**2
    I_plates = n_arms * m_plate * r0**2
    I_total = I_blades + I_plates

    # ── Drag torque parameters (scales with n_arms) ──
    d_wake = 0.12 * c
    C_d = 0.1
    # Drag coefficient per blade, then multiply by n_arms
    D_coeff_per_blade = rho * C_d * d_wake * R_blade**4 / 4.0
    D_coeff = n_arms * D_coeff_per_blade

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
        "n_arms": n_arms,
        "I_blades": I_blades,
        "I_plates": I_plates,
        "I_total": I_total,
        "D_coeff": D_coeff,
        "D_coeff_per_blade": D_coeff_per_blade,
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
        "method": "numerical",
    }


def compute_bangbang_kinematics(n_arms, m_blade, m_plate, tau_motor, sweep_angle,
                                 rho, c, drag=False, r0_carrier=None):
    """
    Wrapper for bang-bang kinematics calculation.

    Parameters:
        drag: If True, run numerical ODE solver with aerodynamic drag.
              If False (default), use closed-form analytical solution.
        r0_carrier: Initial radial position of carrier [m] (default: R_blade)

    Both methods return compatible dict structures.
    """
    if drag:
        return compute_bangbang_kinematics_numerical(
            n_arms, m_blade, m_plate, tau_motor, sweep_angle, rho, c, r0_carrier
        )
    else:
        return compute_bangbang_kinematics_analytical(
            n_arms, m_blade, m_plate, tau_motor, sweep_angle, r0_carrier
        )


def compute_flow_regime(c, V_tip_max, c_s, nu):
    """Compute flow regime parameters using peak velocity from bang-bang motion."""
    Ma_tip = V_tip_max / c_s
    Re_tip = V_tip_max * c / nu
    return Ma_tip, Re_tip


# ═══════════════════════════════════════════════════════════════
# Phase 2: Carrier Slide Kinematics (Linear Bang-Bang Motion)
# ═══════════════════════════════════════════════════════════════
# Carrier geometry constants
CARRIER_WIDTH = 0.160     # m, carrier plate width (160 mm)
CARRIER_HEIGHT = 0.005    # m, carrier plate height (5 mm)
CARRIER_FRONTAL_AREA = CARRIER_WIDTH * CARRIER_HEIGHT  # m², frontal area for drag


def compute_carrier_slide_analytical(m_carrier, F_actuator, r0_carrier, delta_r=None):
    """
    Compute bang-bang linear motion kinematics for carrier slide WITHOUT drag.

    For pure bang-bang motion with constant acceleration/deceleration:
        a = F/m                    (linear acceleration)
        x_switch = Δr/2            (switch at midpoint)
        V_max = √(Δr·a)            (peak velocity)
        t_slide = 2√(Δr·m/F)       (total slide time)

    Parameters:
    - m_carrier: Mass of carrier plate [kg]
    - F_actuator: Linear actuator force [N]
    - r0_carrier: Initial radial position [m]
    - delta_r: Travel distance [m] (default: r0_carrier, i.e., slide to center)

    Returns dict with kinematics.
    """
    if delta_r is None:
        delta_r = r0_carrier  # worst case: slide from r0 to center

    # ── Closed-form kinematics (no drag) ──
    a = F_actuator / m_carrier
    t_slide = 2.0 * np.sqrt(delta_r * m_carrier / F_actuator)
    V_max = np.sqrt(delta_r * a)
    x_switch = delta_r / 2.0
    t_accel = t_slide / 2.0
    t_decel = t_slide / 2.0

    return {
        "m_carrier": m_carrier,
        "F_actuator": F_actuator,
        "r0_carrier": r0_carrier,
        "delta_r": delta_r,
        "a": a,
        "x_switch": x_switch,
        "t_accel": t_accel,
        "t_decel": t_decel,
        "t_slide": t_slide,
        "V_max": V_max,
        "method": "analytical",
    }


def compute_carrier_slide_numerical(m_carrier, F_actuator, r0_carrier, Cd_carrier,
                                    rho, delta_r=None):
    """
    Compute bang-bang linear motion kinematics for carrier slide WITH aerodynamic drag.

    Solves the equation of motion numerically:
        m dx²/dt² = ±F_actuator - F_drag(V)
    where:
        F_drag = ½ ρ Cd A V²

    Parameters:
    - m_carrier: Mass of carrier plate [kg]
    - F_actuator: Linear actuator force [N]
    - r0_carrier: Initial radial position [m]
    - Cd_carrier: Drag coefficient of carrier plate
    - rho: Gas density [kg/m³]
    - delta_r: Travel distance [m] (default: r0_carrier)

    Returns dict with kinematics and drag effects.
    """
    from scipy.optimize import brentq

    if delta_r is None:
        delta_r = r0_carrier

    # ── Drag parameters ──
    A_frontal = CARRIER_FRONTAL_AREA
    D_coeff = 0.5 * rho * Cd_carrier * A_frontal  # F_drag = D_coeff * V²

    # ── Inertia-only reference ──
    a_no_drag = F_actuator / m_carrier
    t_slide_no_drag = 2.0 * np.sqrt(delta_r * m_carrier / F_actuator)
    V_max_no_drag = a_no_drag * (t_slide_no_drag / 2.0)

    # ── Define ODE phases (state: [x, V]) ──
    def accel_phase(t, y):
        V = y[1]
        F_drag = D_coeff * V**2
        return [V, (F_actuator - F_drag) / m_carrier]

    def decel_phase(t, y):
        V = y[1]
        F_drag = D_coeff * V**2
        return [V, (-F_actuator - F_drag) / m_carrier]

    def simulate_with_switch(x_switch):
        """
        Simulate bang-bang with given switching position.
        Returns (x_final, V_final, t_accel, t_decel, V_max).
        """
        # Event: reached x_switch
        def reached_switch(t, y):
            return y[0] - x_switch
        reached_switch.terminal = True
        reached_switch.direction = 1

        # Phase 1: Acceleration
        sol1 = solve_ivp(
            accel_phase, [0, 100], [0.0, 0.0],
            events=reached_switch, max_step=1e-5
        )
        if len(sol1.t_events[0]) == 0:
            return None
        t_accel = sol1.t_events[0][0]
        V_at_switch = sol1.y_events[0][0][1]

        # Event: V reaches zero
        def V_zero(t, y):
            return y[1]
        V_zero.terminal = True
        V_zero.direction = -1

        # Phase 2: Deceleration
        sol2 = solve_ivp(
            decel_phase, [0, 100], [x_switch, V_at_switch],
            events=V_zero, max_step=1e-5
        )
        if len(sol2.t_events[0]) == 0:
            return None
        t_decel = sol2.t_events[0][0]
        x_final = sol2.y_events[0][0][0]

        return (x_final, 0.0, t_accel, t_decel, V_at_switch)

    # ── Find x_switch via bisection ──
    def objective(x_switch):
        result = simulate_with_switch(x_switch)
        if result is None:
            return -delta_r
        return result[0] - delta_r

    # Search bounds: switch must be > Δr/2 (since decel is stronger with drag)
    x_switch_opt = brentq(objective, delta_r * 0.5, delta_r * 0.99, xtol=1e-8)

    # Get final results with optimal switch
    result = simulate_with_switch(x_switch_opt)
    x_final, _, t_accel, t_decel, V_max = result
    t_slide = t_accel + t_decel

    # Drag impact
    drag_time_increase = (t_slide - t_slide_no_drag) / t_slide_no_drag
    V_reduction = (V_max_no_drag - V_max) / V_max_no_drag

    # Peak drag force
    F_drag_max = D_coeff * V_max**2

    return {
        "m_carrier": m_carrier,
        "F_actuator": F_actuator,
        "r0_carrier": r0_carrier,
        "delta_r": delta_r,
        "D_coeff": D_coeff,
        "Cd_carrier": Cd_carrier,
        "A_frontal": A_frontal,
        "a_no_drag": a_no_drag,
        "t_slide_no_drag": t_slide_no_drag,
        "V_max_no_drag": V_max_no_drag,
        "x_switch": x_switch_opt,
        "t_accel": t_accel,
        "t_decel": t_decel,
        "t_slide": t_slide,
        "V_max": V_max,
        "F_drag_max": F_drag_max,
        "drag_time_increase": drag_time_increase,
        "V_reduction": V_reduction,
        "method": "numerical",
    }


def compute_carrier_slide(m_carrier, F_actuator, r0_carrier, Cd_carrier,
                          rho, delta_r=None, drag=False):
    """
    Wrapper for carrier slide kinematics calculation.

    Parameters:
        drag: If True, run numerical ODE solver with aerodynamic drag.
              If False (default), use closed-form analytical solution.

    Both methods return compatible dict structures.
    """
    if drag:
        return compute_carrier_slide_numerical(
            m_carrier, F_actuator, r0_carrier, Cd_carrier, rho, delta_r
        )
    else:
        return compute_carrier_slide_analytical(
            m_carrier, F_actuator, r0_carrier, delta_r
        )


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


def compute_carrier_boundary_layer(V_slide, nu):
    """
    Compute turbulent BL properties on the carrier plate upper surface.

    The carrier plate is a 160×160 mm flat plate moving at V_slide through Xe.
    Uses flat-plate correlations with the carrier width as characteristic length.
    """
    L_carrier = CARRIER_WIDTH  # characteristic length = plate width
    Re_L = V_slide * L_carrier / nu

    # Handle low Reynolds number (laminar) case
    if Re_L < 5e5:
        # Laminar BL: delta = 5.0 L / sqrt(Re)
        delta = 5.0 * L_carrier / np.sqrt(Re_L) if Re_L > 0 else 0.0
        # Laminar skin friction
        Cf = 1.328 / np.sqrt(Re_L) if Re_L > 0 else 0.0
        regime = "laminar"
    else:
        # Turbulent BL: delta = 0.37 L Re^(-1/5)
        delta = 0.37 * L_carrier * Re_L**(-0.2)
        # Turbulent skin friction
        Cf = 0.0592 * Re_L**(-0.2)
        regime = "turbulent"

    # Momentum thickness
    theta = (7.0 / 72.0) * delta if regime == "turbulent" else delta / 5.0

    # Displacement thickness
    delta_star = delta / 8.0 if regime == "turbulent" else delta / 3.0

    # Shape factor
    H = delta_star / theta if theta > 0 else 0.0

    # Friction velocity
    u_tau = V_slide * np.sqrt(Cf / 2.0) if Cf > 0 else 0.0

    return {
        "L_carrier": L_carrier,
        "Re_L": Re_L,
        "regime": regime,
        "delta": delta,
        "theta": theta,
        "delta_star": delta_star,
        "H": H,
        "Cf": Cf,
        "u_tau": u_tau,
    }


def compute_carrier_flow_regime(V_slide, c_s, nu):
    """Compute flow regime parameters for carrier slide motion."""
    Ma = V_slide / c_s
    Re = V_slide * CARRIER_WIDTH / nu
    return Ma, Re


def carrier_eddy_diffusion_analytical(V_slide, delta, delta_star, nu, z_obs, t_obs):
    """
    Analytical estimate for eddy diffusion above stopped carrier plate.

    Same physics as blade case but with carrier-specific parameters.
    """
    # ── Turbulent viscosity parameters ──
    nu_t_outer = 0.018 * V_slide * delta_star

    # Turbulence intensity and eddy timescale
    u_prime = 0.05 * V_slide
    t_eddy = delta / u_prime if u_prime > 0 else float('inf')
    t_decay = 2.0 * t_eddy

    # Diffusion length scales
    ell_turb = np.sqrt(nu_t_outer * t_decay) if t_decay < float('inf') else 0.0
    ell_mol = np.sqrt(nu * t_obs)
    max_reach = delta + ell_turb + ell_mol

    # Gap between BL top and observation point
    gap = z_obs - delta

    # ── Analytical estimate ──
    if gap <= 0:
        # Observation point is within the BL
        zeta = z_obs / delta if delta > 0 else 0
        u_initial = V_slide * (1.0 - zeta ** (1.0 / 7.0)) if zeta < 1 else 0
        nu_eff_avg = nu + nu_t_outer * 0.5
        t_char = delta**2 / nu_eff_avg if nu_eff_avg > 0 else float('inf')
        decay = np.exp(-t_obs / t_char) if t_char < float('inf') else 0
        u_at_obs = u_initial * decay
    elif z_obs > max_reach:
        u_at_obs = 0.0
    else:
        # Observation point is in the diffusion tail
        if t_obs < t_decay:
            ell_diff = np.sqrt(4.0 * (nu + nu_t_outer * 0.5) * t_obs)
        else:
            ell_diff = np.sqrt(4.0 * nu_t_outer * t_decay + 4.0 * nu * (t_obs - t_decay))

        if ell_diff > 0:
            eta = gap / ell_diff
            u_at_obs = V_slide * 0.5 * erfc(eta)
        else:
            u_at_obs = 0.0

        absorption_factor = np.exp(-t_obs / t_decay) if t_decay < float('inf') else 0
        u_at_obs *= absorption_factor

    # Momentum budget
    mom_initial = V_slide * delta * 7.0 / 8.0 if delta > 0 else 0
    mom_fraction = np.exp(-t_obs / t_decay) if t_decay < float('inf') else 0

    return {
        "u_at_obs": u_at_obs,
        "nu_t_outer": nu_t_outer,
        "nu_t_over_nu": nu_t_outer / nu if nu > 0 else 0,
        "u_prime": u_prime,
        "t_eddy": t_eddy,
        "t_decay": t_decay,
        "ell_turb": ell_turb,
        "ell_mol": ell_mol,
        "max_reach": max_reach,
        "mom_initial": mom_initial,
        "mom_final": mom_initial * mom_fraction,
        "mom_fraction": mom_fraction,
        "method": "analytical",
    }


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
def mechanism_eddy_diffusion_analytical(V_tip, delta, delta_star, nu, z_obs, t_obs):
    """
    Analytical estimate for eddy diffusion (fast).

    Uses diffusion length scales to determine if momentum can reach z_obs,
    and if so, estimates the velocity using error function solutions.

    The key insight: momentum initially confined to 0 < z < δ diffuses outward.
    If z_obs > δ + ℓ_turb + ℓ_mol, momentum cannot reach → u ≈ 0.
    Otherwise, use complementary error function for the diffusing front.
    """

    # ── Turbulent viscosity parameters ──
    nu_t_outer = 0.018 * V_tip * delta_star

    # Turbulence intensity and eddy timescale
    u_prime = 0.05 * V_tip
    t_eddy = delta / u_prime
    t_decay = 2.0 * t_eddy

    # Diffusion length scales
    ell_turb = np.sqrt(nu_t_outer * t_decay)
    ell_mol = np.sqrt(nu * t_obs)
    max_reach = delta + ell_turb + ell_mol

    # Gap between BL top and observation point
    gap = z_obs - delta

    # ── Analytical estimate ──
    if gap <= 0:
        # Observation point is within the BL
        # Use 1/7 power law profile, decayed by turbulence + no-slip absorption
        zeta = z_obs / delta
        u_initial = V_tip * (1.0 - zeta ** (1.0 / 7.0))
        # Decay factor: turbulence decays, and no-slip absorbs momentum
        # Approximate: momentum decays as exp(-t/t_char) where t_char ~ delta²/nu_eff
        nu_eff_avg = nu + nu_t_outer * 0.5  # time-averaged
        t_char = delta**2 / nu_eff_avg
        decay = np.exp(-t_obs / t_char)
        u_at_obs = u_initial * decay
    elif z_obs > max_reach:
        # Momentum cannot reach observation point
        u_at_obs = 0.0
    else:
        # Observation point is in the diffusion tail
        # Use complementary error function for diffusion from a step
        # The BL momentum diffuses outward from z = δ
        # u(z, t) ~ V₀ × erfc((z - δ) / √(4 ν_eff t)) × decay_factors

        # Diffusion length from BL edge
        if t_obs < t_decay:
            # During turbulent phase
            ell_diff = np.sqrt(4.0 * (nu + nu_t_outer * 0.5) * t_obs)
        else:
            # Turbulent phase contribution + molecular phase
            ell_diff = np.sqrt(4.0 * nu_t_outer * t_decay + 4.0 * nu * (t_obs - t_decay))

        # Complementary error function gives the diffusing front
        if ell_diff > 0:
            eta = gap / ell_diff
            # erfc decays rapidly for eta > 1
            u_at_obs = V_tip * 0.5 * erfc(eta)
        else:
            u_at_obs = 0.0

        # Additional decay from no-slip absorption at z = 0
        # Momentum is absorbed by the stationary blade surface
        absorption_factor = np.exp(-t_obs / t_decay)
        u_at_obs *= absorption_factor

    # Build velocity profile at selected heights (analytical)
    profile = {}
    for zi_mm in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]:
        zi = zi_mm * 1e-3
        gap_i = zi - delta
        if zi <= 0:
            profile[zi_mm] = 0.0
        elif gap_i <= 0:
            zeta = zi / delta
            u_init = V_tip * (1.0 - zeta ** (1.0 / 7.0))
            nu_eff_avg = nu + nu_t_outer * 0.5
            t_char = delta**2 / nu_eff_avg
            profile[zi_mm] = u_init * np.exp(-t_obs / t_char)
        elif zi > max_reach:
            profile[zi_mm] = 0.0
        else:
            if t_obs < t_decay:
                ell_diff = np.sqrt(4.0 * (nu + nu_t_outer * 0.5) * t_obs)
            else:
                ell_diff = np.sqrt(4.0 * nu_t_outer * t_decay + 4.0 * nu * (t_obs - t_decay))
            if ell_diff > 0:
                eta = gap_i / ell_diff
                u_val = V_tip * 0.5 * erfc(eta) * np.exp(-t_obs / t_decay)
            else:
                u_val = 0.0
            profile[zi_mm] = u_val

    # Estimate momentum (analytical)
    # Initial momentum in BL: integral of V(1 - (z/δ)^(1/7)) from 0 to δ = V × δ × 7/8
    mom_initial = V_tip * delta * 7.0 / 8.0
    # Final momentum decays due to absorption
    mom_fraction = np.exp(-t_obs / t_decay)

    return {
        "u_at_obs": u_at_obs,
        "nu_t_outer": nu_t_outer,
        "nu_t_over_nu": nu_t_outer / nu,
        "u_prime": u_prime,
        "t_eddy": t_eddy,
        "t_decay": t_decay,
        "ell_turb": ell_turb,
        "ell_mol": ell_mol,
        "max_reach": max_reach,
        "mom_initial": mom_initial,
        "mom_final": mom_initial * mom_fraction,
        "mom_fraction": mom_fraction,
        "profile": profile,
        "method": "analytical",
    }


def mechanism_eddy_diffusion_numerical(V_tip, c, delta, delta_star, u_tau, nu,
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
        "method": "numerical",
    }


def mechanism_eddy_diffusion(V_tip, c, delta, delta_star, u_tau, nu,
                             z_obs, t_obs, numeric=False):
    """
    Wrapper for eddy diffusion calculation.

    Parameters:
        numeric: If True, run full PDE solver. If False (default), use analytical estimate.

    Both methods return the same dict structure for compatibility.
    """
    if numeric:
        return mechanism_eddy_diffusion_numerical(V_tip, c, delta, delta_star, u_tau, nu,
                                                  z_obs, t_obs)
    else:
        return mechanism_eddy_diffusion_analytical(V_tip, delta, delta_star, nu, z_obs, t_obs)


# ═══════════════════════════════════════════════════════════════
# Mechanism 2: Bulk Swirl / Angular Momentum
# ═══════════════════════════════════════════════════════════════
def mechanism_bulk_swirl(n_arms, rho, nu, omega, c, Cd, sweep_time, delta,
                         z_obs, t_obs, V_tip):
    """Evaluate bulk swirl and vertical transport mechanisms."""
    # Torque and angular impulse (scales with n_arms)
    torque = n_arms * rho * omega**2 * c * Cd * R_blade**4 / 4.0
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
# Phase 3: Vertical Plate Lift - Velocity Profiles and Quiescence
# ═══════════════════════════════════════════════════════════════
def compute_velocity_profile_blade(V_tip, delta, delta_star, nu, t_obs, z_max, u_c, dz=0.0001):
    """
    Compute velocity profile above stopped blade at time t_obs after stop.

    Uses analytical error function solution for diffusing BL momentum.
    Scans from z=0 upward until u < u_c (quiescence) or z > z_max.

    Parameters:
        V_tip: Peak blade tip velocity [m/s]
        delta: BL thickness [m]
        delta_star: Displacement thickness [m]
        nu: Kinematic viscosity [m²/s]
        t_obs: Observation time after stop [s]
        z_max: Maximum height to scan [m]
        u_c: Quiescence cutoff velocity [m/s]
        dz: Height step [m] (default 0.1 mm)

    Returns dict with:
        z_array: Heights [m]
        u_array: Velocities [m/s]
        delta_1: BL thickness [m]
        z_q1: Quiescence height [m] (where u < u_c, or z_max if never)
    """
    # Turbulent viscosity and decay parameters
    nu_t_outer = 0.018 * V_tip * delta_star
    u_prime = 0.05 * V_tip
    t_eddy = delta / u_prime if u_prime > 0 else float('inf')
    t_decay = 2.0 * t_eddy

    # Diffusion length scales
    ell_turb = np.sqrt(nu_t_outer * t_decay) if t_decay < float('inf') else 0.0
    ell_mol = np.sqrt(nu * t_obs)
    max_reach = delta + ell_turb + ell_mol

    # Build profile
    z_values = []
    u_values = []
    z_q1 = z_max  # default if never reaches quiescence
    found_peak = False  # track if we've passed the peak velocity
    u_prev = 0.0

    z = 0.0
    while z <= z_max:
        gap = z - delta

        if z <= 0:
            u = 0.0  # At blade surface (no-slip)
        elif gap <= 0:
            # Within BL: use 1/7 power law with decay
            zeta = z / delta
            u_initial = V_tip * (1.0 - zeta ** (1.0 / 7.0))
            nu_eff_avg = nu + nu_t_outer * 0.5
            t_char = delta**2 / nu_eff_avg if nu_eff_avg > 0 else float('inf')
            decay = np.exp(-t_obs / t_char) if t_char < float('inf') else 0
            u = u_initial * decay
        elif z > max_reach:
            u = 0.0
        else:
            # Above BL: diffusion tail with erfc
            if t_obs < t_decay:
                ell_diff = np.sqrt(4.0 * (nu + nu_t_outer * 0.5) * t_obs)
            else:
                ell_diff = np.sqrt(4.0 * nu_t_outer * t_decay + 4.0 * nu * (t_obs - t_decay))

            if ell_diff > 0:
                eta = gap / ell_diff
                u = V_tip * 0.5 * erfc(eta)
            else:
                u = 0.0

            # Absorption factor from no-slip
            absorption = np.exp(-t_obs / t_decay) if t_decay < float('inf') else 0
            u *= absorption

        z_values.append(z)
        u_values.append(u)
        z += dz

    # Convert to arrays
    z_arr = np.array(z_values)
    u_arr = np.array(u_values)

    # Find peak velocity (excluding z=0)
    nonzero_mask = z_arr > 0
    if np.any(nonzero_mask):
        u_max = np.max(u_arr[nonzero_mask])
        peak_idx = np.argmax(u_arr)
    else:
        u_max = 0.0
        peak_idx = 0

    # Find z_q1: first z after peak where u < u_c
    if u_max < u_c:
        # Entire profile is already quiescent; z_q1 is first z > 0
        z_q1 = z_arr[1] if len(z_arr) > 1 else z_arr[0]
    else:
        # Search after peak
        z_q1 = z_max  # default
        for i in range(peak_idx, len(u_arr)):
            if u_arr[i] < u_c:
                z_q1 = z_arr[i]
                break

    return {
        "z_array": z_arr,
        "u_array": u_arr,
        "delta_1": delta,
        "z_q1": z_q1,
        "max_reach": max_reach,
        "t_decay": t_decay,
        "nu_t_outer": nu_t_outer,
    }


def compute_velocity_profile_slide(V_slide, delta, delta_star, nu, t_obs, z_max, u_c, dz=0.0001):
    """
    Compute velocity profile above stopped carrier plate at time t_obs after slide stops.

    Uses analytical error function solution for diffusing BL momentum.
    Scans from z=0 upward until u < u_c (quiescence) or z > z_max.

    Parameters:
        V_slide: Peak carrier slide velocity [m/s]
        delta: Carrier BL thickness [m]
        delta_star: Carrier displacement thickness [m]
        nu: Kinematic viscosity [m²/s]
        t_obs: Observation time after slide stops [s]
        z_max: Maximum height to scan [m]
        u_c: Quiescence cutoff velocity [m/s]
        dz: Height step [m] (default 0.1 mm)

    Returns dict with:
        z_array: Heights [m]
        u_array: Velocities [m/s]
        delta_2: BL thickness [m]
        z_q2: Quiescence height [m] (where u < u_c, or z_max if never)
    """
    # Turbulent viscosity and decay parameters
    nu_t_outer = 0.018 * V_slide * delta_star
    u_prime = 0.05 * V_slide
    t_eddy = delta / u_prime if u_prime > 0 else float('inf')
    t_decay = 2.0 * t_eddy

    # Diffusion length scales
    ell_turb = np.sqrt(nu_t_outer * t_decay) if t_decay < float('inf') else 0.0
    ell_mol = np.sqrt(nu * t_obs)
    max_reach = delta + ell_turb + ell_mol

    # Build profile
    z_values = []
    u_values = []
    z_q2 = z_max  # default if never reaches quiescence

    z = 0.0
    while z <= z_max:
        gap = z - delta

        if z <= 0:
            u = 0.0  # At carrier surface (no-slip)
        elif gap <= 0:
            # Within BL: use 1/7 power law with decay
            zeta = z / delta
            u_initial = V_slide * (1.0 - zeta ** (1.0 / 7.0))
            nu_eff_avg = nu + nu_t_outer * 0.5
            t_char = delta**2 / nu_eff_avg if nu_eff_avg > 0 else float('inf')
            decay = np.exp(-t_obs / t_char) if t_char < float('inf') else 0
            u = u_initial * decay
        elif z > max_reach:
            u = 0.0
        else:
            # Above BL: diffusion tail with erfc
            if t_obs < t_decay:
                ell_diff = np.sqrt(4.0 * (nu + nu_t_outer * 0.5) * t_obs)
            else:
                ell_diff = np.sqrt(4.0 * nu_t_outer * t_decay + 4.0 * nu * (t_obs - t_decay))

            if ell_diff > 0:
                eta = gap / ell_diff
                u = V_slide * 0.5 * erfc(eta)
            else:
                u = 0.0

            # Absorption factor from no-slip
            absorption = np.exp(-t_obs / t_decay) if t_decay < float('inf') else 0
            u *= absorption

        z_values.append(z)
        u_values.append(u)
        z += dz

    # Convert to arrays
    z_arr = np.array(z_values)
    u_arr = np.array(u_values)

    # Find peak velocity (excluding z=0)
    nonzero_mask = z_arr > 0
    if np.any(nonzero_mask):
        u_max = np.max(u_arr[nonzero_mask])
        peak_idx = np.argmax(u_arr)
    else:
        u_max = 0.0
        peak_idx = 0

    # Find z_q2: first z after peak where u < u_c
    if u_max < u_c:
        # Entire profile is already quiescent; z_q2 is first z > 0
        z_q2 = z_arr[1] if len(z_arr) > 1 else z_arr[0]
    else:
        # Search after peak
        z_q2 = z_max  # default
        for i in range(peak_idx, len(u_arr)):
            if u_arr[i] < u_c:
                z_q2 = z_arr[i]
                break

    return {
        "z_array": z_arr,
        "u_array": u_arr,
        "delta_2": delta,
        "z_q2": z_q2,
        "max_reach": max_reach,
        "t_decay": t_decay,
        "nu_t_outer": nu_t_outer,
    }


def compute_stokes_layer_thickness(nu, t_lift):
    """
    Compute Stokes layer thickness for plate lifted over time t_lift.

    The Stokes first problem gives δ_Stokes = √(π ν t).
    For oscillating/impulsive motion, use δ ≈ √(ν t).
    """
    return np.sqrt(nu * t_lift)


def compute_velocity_profile_lift(V_lift, delta_stokes, nu, t_wait, z_lift, z_max, u_c, dz=0.0001):
    """
    Compute velocity profile above lifted plate at time t_wait after lift stops.

    The lifted plate was moving at V_lift, then stopped. The gas near the plate
    was dragged along and now diffuses. Uses the Stokes first problem solution:
    the velocity diffuses away from the plate surface with erfc profile.

    Coordinates: z = 0 at original blade surface (RF1).
    The plate is now at z = z_lift.

    Parameters:
        V_lift: Peak lift velocity [m/s]
        delta_stokes: Stokes layer thickness at end of lift [m]
        nu: Kinematic viscosity [m²/s]
        t_wait: Wait time after lift stops [s]
        z_lift: Height of lifted plate [m]
        z_max: Maximum height to scan [m]
        u_c: Quiescence cutoff velocity [m/s]
        dz: Height step [m] (default 0.1 mm)

    Returns dict with:
        z_array: Heights in RF1 [m]
        u_array: Velocities [m/s]
        delta_3: Stokes layer thickness [m]
        z_q3: Quiescence height in RF1 [m] (where u < u_c, or z_max if never)
    """
    # Total diffusion time: the Stokes layer was created during t_lift,
    # then continues to diffuse during t_wait.
    # Effective diffusion length after waiting:
    # At end of lift: delta_stokes = sqrt(nu * t_lift)
    # After additional t_wait: delta_eff = sqrt(nu * (t_lift + t_wait))
    # But we only have delta_stokes, so compute t_lift from it:
    t_lift_eff = delta_stokes**2 / nu if nu > 0 else 0
    t_total = t_lift_eff + t_wait
    delta_eff = np.sqrt(nu * t_total) if t_total > 0 else 0

    # Build profile (starting from z_lift upward)
    z_values = []
    u_values = []

    z = z_lift
    while z <= z_max:
        # Distance above the lifted plate surface
        z_above_plate = z - z_lift

        if z_above_plate <= 0:
            # At plate surface: no-slip, u = 0 (plate has stopped)
            u = 0.0
        else:
            # Stokes first problem: impulsively stopped plate
            # After plate stops, the momentum diffuses away.
            # Profile: u(z,t) = V_lift * erfc(z / (2*sqrt(nu*t)))
            # This gives max velocity near (but not at) the surface,
            # monotonically decreasing upward.
            if delta_eff > 0:
                eta = z_above_plate / (2.0 * delta_eff)
                u = V_lift * erfc(eta)
            else:
                u = 0.0

        z_values.append(z)
        u_values.append(u)
        z += dz

    # Convert to arrays
    z_arr = np.array(z_values)
    u_arr = np.array(u_values)

    # Find z_q3: height where u drops below u_c
    # Profile shape: u=0 at plate surface, peak just above, then decay
    # We want to find where u < u_c AFTER the peak (not at the surface)
    z_q3 = z_max  # default if never reaches quiescence
    u_max = np.max(u_arr) if len(u_arr) > 0 else 0

    if u_max < u_c:
        # Entire profile is already quiescent
        z_q3 = z_arr[0] if len(z_arr) > 0 else z_lift
    else:
        # Find peak index
        peak_idx = np.argmax(u_arr)
        # Search AFTER peak for where u < u_c
        for i in range(peak_idx, len(u_arr)):
            if u_arr[i] < u_c:
                z_q3 = z_arr[i]
                break

    return {
        "z_array": z_arr,
        "u_array": u_arr,
        "delta_3": delta_stokes,
        "z_q3": z_q3,
        "z_lift": z_lift,
        "delta_eff": delta_eff,
    }


def compute_lift_kinematics(z_lift, t_lift):
    """
    Compute bang-bang kinematics for vertical plate lift.

    For bang-bang motion over distance z_lift in time t_lift:
        a = 4 * z_lift / t_lift²
        V_max = 2 * z_lift / t_lift

    Returns dict with kinematics.
    """
    a_lift = 4.0 * z_lift / t_lift**2
    V_max = 2.0 * z_lift / t_lift
    t_accel = t_lift / 2.0
    t_decel = t_lift / 2.0

    return {
        "z_lift": z_lift,
        "t_lift": t_lift,
        "a_lift": a_lift,
        "V_max": V_max,
        "t_accel": t_accel,
        "t_decel": t_decel,
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

    n_arms = args.n_arms
    c = args.chord
    m_blade = args.m_blade
    m_plate = args.m_plate
    tau_motor = args.tau_motor
    numeric = args.numeric
    drag = args.drag

    # Phase 2: Carrier slide parameters (carrier plate = plate on arm, same mass)
    F_actuator = args.F_actuator
    m_carrier = m_plate  # Carrier plate is the same as plate on arm
    Cd_carrier = args.Cd_carrier
    r0_carrier = args.r0_carrier

    # Phase 3: Vertical plate lift parameters
    z_lift = args.z_lift
    t_lift = args.t_lift
    t_wait = args.t_wait
    u_c = args.u_c
    z_max = args.z_max

    # Legacy parameters for Phase 1 mechanism output (derived from Phase 3)
    # z_obs: observation height = z_lift (the lifted plate height)
    # t_obs: observation time = t_wait (evaluate at end of wait period)
    z_obs = z_lift
    t_obs = t_wait

    # Compute sweep angle from number of arms: θ = 2π / n_arms
    sweep_angle = 2.0 * np.pi / n_arms

    DIVIDER = "=" * 70

    # ── Header ──
    print(DIVIDER)
    print("RESIDUAL GAS VELOCITY ABOVE A STOPPED PROPELLER IN DENSE XENON")
    print(DIVIDER)
    print(f"\n  Variable parameters:")
    print(f"    Number of arms:      n = {n_arms}")
    print(f"    Sweep angle:         θ = 2π/{n_arms} = {sweep_angle:.5f} rad "
          f"({np.degrees(sweep_angle):.1f}°)")
    print(f"    Chord length:        c = {c:.4f} m")
    print(f"    Blade mass (each):   m_blade = {m_blade:.3f} kg  (×{n_arms} = {n_arms*m_blade:.3f} kg total)")
    print(f"    Plate mass (each):   m_plate = {m_plate:.3f} kg  (×{n_arms} = {n_arms*m_plate:.3f} kg total)")
    print(f"    Motor torque:        τ = {tau_motor:.1f} N·m")
    print(f"\n  Phase 2 (carrier slide) parameters:")
    print(f"    Carrier mass:        m_carrier = {m_carrier:.3f} kg (= m_plate)")
    print(f"    Initial position:    r₀ = {r0_carrier:.2f} m (R/2)")
    print(f"    Actuator force:      F = {F_actuator:.1f} N")
    print(f"    Drag coefficient:    Cd = {Cd_carrier}")
    print(f"\n  Phase 3 (vertical lift) parameters:")
    print(f"    Lift height:         z_lift = {z_lift*1000:.1f} mm")
    print(f"    Lift duration:       t_lift = {t_lift:.3f} s")
    print(f"    Wait time:           t_wait = {t_wait:.3f} s")
    print(f"    Quiescence cutoff:   u_c = {u_c*1000:.2f} mm/s")
    print(f"    Max profile height:  z_max = {z_max*1000:.1f} mm")
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

    # ── Bang-bang Kinematics ──
    bb = compute_bangbang_kinematics(n_arms, m_blade, m_plate, tau_motor, sweep_angle,
                                      rho, c, drag=drag, r0_carrier=r0_carrier)
    t_rot = bb["t_rot"]
    omega_max = bb["omega_max"]
    V_tip = bb["V_tip_max"]

    print(f"\n{DIVIDER}")
    if drag:
        print(f"2. BANG-BANG KINEMATICS (with aerodynamic drag, {n_arms} arms)")
    else:
        print(f"2. BANG-BANG KINEMATICS (analytical, no drag, {n_arms} arms)")
    print(DIVIDER)
    print(f"  Moment of inertia (I = Σ m·r²):")
    print(f"    I_blades = {n_arms} × (1/3)·m·R²:  {bb['I_blades']:.4f} kg·m²  (rods about end)")
    print(f"    I_plates = {n_arms} × m·r₀²:       {bb['I_plates']:.4f} kg·m²  (point masses at r₀={r0_carrier:.2f}m)")
    print(f"    I_total:                    {bb['I_total']:.4f} kg·m²")

    if drag:
        # Numerical with drag: show drag parameters and comparison
        print(f"\n  Aerodynamic drag torque:")
        print(f"    τ_drag = {n_arms} × (ρ C_d d_wake R⁴ / 4) ω²")
        print(f"    Wake thickness:     d_wake = 0.12c = {bb['d_wake']*1000:.1f} mm")
        print(f"    Drag coefficient:   C_d = {bb['C_d']}")
        print(f"    Drag per blade:     D₁ = {bb['D_coeff_per_blade']:.4f} N·m·s²/rad²")
        print(f"    Drag total ({n_arms} arms): D = {bb['D_coeff']:.4f} N·m·s²/rad²")
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
    else:
        # Analytical without drag: simpler output
        print(f"\n  Closed-form solution (I dω/dt = ±τ, no drag):")
        print(f"    α = τ/I = {bb['alpha']:.2f} rad/s²")
        print(f"    θ_switch = θ/2 = {bb['theta_switch']:.5f} rad "
              f"({np.degrees(bb['theta_switch']):.2f}°)")
        print(f"    t_accel = t_decel = {bb['t_accel']*1000:.2f} ms")
        print(f"    Total rotation time:    t_rot    = {t_rot*1000:.2f} ms")
        print(f"    Peak ω (at θ_switch):   ω_max    = {omega_max:.4f} rad/s")
        print(f"    Peak V_tip:             V_tip    = {V_tip:.4f} m/s")
        print(f"\n  NOTE: Use --drag flag for numerical solution with aerodynamic drag.")

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
    if numeric:
        print(f"  Solving 1D diffusion PDE numerically (--numeric flag set)...")
    else:
        print(f"  Using analytical estimate (use --numeric for full PDE solver)...")

    m1 = mechanism_eddy_diffusion(V_tip, c, delta, delta_star, u_tau,
                                  nu, z_obs, t_obs, numeric=numeric)

    print(f"\n  Method: {m1['method'].upper()}")
    print(f"\n  Turbulent viscosity:")
    if 'nu_t_inner_peak' in m1:
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

    m2 = mechanism_bulk_swirl(n_arms, rho, nu, omega_max, c, Cd, t_rot, delta,
                              z_obs, t_obs, V_tip)

    print(f"  Torque ({n_arms} blades):     τ = {m2['torque']:.3f} N·m")
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
    print(f"\n  Parameters: n_arms = {n_arms}, c = {c} m, θ = 2π/{n_arms} = {np.degrees(sweep_angle):.1f}°, "
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

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: CARRIER RADIAL SLIDE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PHASE 2: CARRIER RADIAL SLIDE (after blade rotation completes)")
    print("=" * 70)

    print(f"\n  The carrier plate slides radially along the blade arm after")
    print(f"  the blade rotation is complete. Motion is sequential (not simultaneous)")
    print(f"  to avoid Coriolis coupling and time-varying moment of inertia.")

    # ── Carrier Slide Kinematics ──
    cs = compute_carrier_slide(m_carrier, F_actuator, r0_carrier, Cd_carrier,
                               rho, delta_r=r0_carrier, drag=drag)
    t_slide = cs["t_slide"]
    V_slide_max = cs["V_max"]

    print(f"\n{DIVIDER}")
    if drag:
        print("13. CARRIER SLIDE KINEMATICS (with aerodynamic drag)")
    else:
        print("13. CARRIER SLIDE KINEMATICS (analytical, no drag)")
    print(DIVIDER)
    print(f"  Carrier geometry:")
    print(f"    Width:               {CARRIER_WIDTH*1000:.0f} mm")
    print(f"    Height:              {CARRIER_HEIGHT*1000:.0f} mm")
    print(f"    Frontal area:        A = {CARRIER_FRONTAL_AREA*1e4:.1f} cm²")
    print(f"    Mass:                m = {m_carrier:.3f} kg")
    print(f"\n  Motion parameters:")
    print(f"    Initial position:    r₀ = {r0_carrier:.2f} m")
    print(f"    Travel distance:     Δr = {cs['delta_r']:.2f} m (to center)")
    print(f"    Actuator force:      F = {F_actuator:.1f} N")

    if drag:
        print(f"\n  Aerodynamic drag:")
        print(f"    Drag coefficient:    Cd = {cs['Cd_carrier']}")
        print(f"    D_coeff = ½ρCdA:     {cs['D_coeff']:.4f} N·s²/m²")
        print(f"    Peak drag force:     F_drag = {cs['F_drag_max']:.3f} N")
        print(f"\n  Inertia-only reference (no drag):")
        print(f"    a = F/m = {cs['a_no_drag']:.1f} m/s²")
        print(f"    t_slide = 2√(Δr·m/F) = {cs['t_slide_no_drag']*1000:.1f} ms")
        print(f"    V_max = {cs['V_max_no_drag']:.2f} m/s")
        print(f"\n  With drag (numerical integration):")
        print(f"    Switching position:  x_switch = {cs['x_switch']:.4f} m")
        print(f"    Accel time:          t_accel  = {cs['t_accel']*1000:.2f} ms")
        print(f"    Decel time:          t_decel  = {cs['t_decel']*1000:.2f} ms")
        print(f"    Total slide time:    t_slide  = {t_slide*1000:.2f} ms")
        print(f"    Peak velocity:       V_max    = {V_slide_max:.4f} m/s")
        print(f"\n  Drag impact:")
        print(f"    Time increase:       {cs['drag_time_increase']*100:+.2f}%")
        print(f"    V_max reduction:     {cs['V_reduction']*100:.2f}%")
    else:
        print(f"\n  Closed-form solution (m dV/dt = ±F, no drag):")
        print(f"    a = F/m = {cs['a']:.1f} m/s²")
        print(f"    x_switch = Δr/2 = {cs['x_switch']:.4f} m")
        print(f"    t_accel = t_decel = {cs['t_accel']*1000:.2f} ms")
        print(f"    Total slide time:    t_slide  = {t_slide*1000:.2f} ms")
        print(f"    Peak velocity:       V_max    = {V_slide_max:.4f} m/s")
        print(f"\n  NOTE: Use --drag flag for numerical solution with aerodynamic drag.")

    # ── Carrier Flow Regime ──
    Ma_slide, Re_slide = compute_carrier_flow_regime(V_slide_max, c_s, nu)
    print(f"\n{DIVIDER}")
    print("14. CARRIER FLOW REGIME (at peak slide velocity)")
    print(DIVIDER)
    print(f"  Carrier Mach number:   Ma = {Ma_slide:.4f}")
    print(f"  Carrier Reynolds:      Re = {Re_slide:.3e}")
    print()
    if Ma_slide < 0.3:
        print(f"  COMPRESSIBILITY (Ma = {Ma_slide:.3f} < 0.3):")
        print(f"    Incompressible flow.")
    else:
        print(f"  COMPRESSIBILITY (Ma = {Ma_slide:.3f} ≥ 0.3):")
        print(f"    Compressibility effects may be significant.")
    print()
    if Re_slide > 5e5:
        print(f"  TURBULENCE (Re = {Re_slide:.2e} > 5×10⁵):")
        print(f"    Fully turbulent boundary layer on carrier.")
    elif Re_slide > 2300:
        print(f"  TURBULENCE (Re = {Re_slide:.2e}):")
        print(f"    Transitional flow; may have laminar and turbulent regions.")
    else:
        print(f"  TURBULENCE (Re = {Re_slide:.2e} < 2300):")
        print(f"    Laminar flow; viscous forces dominate.")

    # ── Carrier Boundary Layer ──
    bl_carrier = compute_carrier_boundary_layer(V_slide_max, nu)
    delta_c = bl_carrier["delta"]
    print(f"\n{DIVIDER}")
    print("15. CARRIER BOUNDARY LAYER (upper surface)")
    print(DIVIDER)
    print(f"  Characteristic length: L = {bl_carrier['L_carrier']*1000:.0f} mm (carrier width)")
    print(f"  Reynolds number:       Re_L = {bl_carrier['Re_L']:.3e}")
    print(f"  Flow regime:           {bl_carrier['regime'].upper()}")
    print()
    print(f"  BL thickness:          δ  = {bl_carrier['delta']*1000:.3f} mm")
    print(f"  Momentum thickness:    θ  = {bl_carrier['theta']*1000:.4f} mm")
    print(f"  Displacement thick.:   δ* = {bl_carrier['delta_star']*1000:.4f} mm")
    print(f"  Shape factor:          H  = {bl_carrier['H']:.3f}")
    print(f"  Skin friction coeff:   Cf = {bl_carrier['Cf']:.5f}")
    print(f"  Friction velocity:     u_τ = {bl_carrier['u_tau']:.4f} m/s")

    # ── Carrier Eddy Diffusion ──
    print(f"\n{DIVIDER}")
    print("16. CARRIER EDDY DIFFUSION (after slide stops)")
    print(DIVIDER)

    m1_c = carrier_eddy_diffusion_analytical(
        V_slide_max, delta_c, bl_carrier["delta_star"], nu, z_obs, t_obs
    )

    print(f"  Turbulent viscosity:")
    print(f"    ν_t (outer):       {m1_c['nu_t_outer']:.3e} m²/s")
    print(f"    ν_t / ν:           {m1_c['nu_t_over_nu']:.0f}×")
    print(f"\n  Eddy scales:")
    print(f"    u':                {m1_c['u_prime']:.4f} m/s")
    if m1_c['t_eddy'] < float('inf'):
        print(f"    t_eddy (δ/u'):     {m1_c['t_eddy']*1000:.1f} ms")
        print(f"    t_decay (2 t_e):   {m1_c['t_decay']*1000:.1f} ms")
    else:
        print(f"    t_eddy (δ/u'):     ∞ (V_slide ≈ 0)")
        print(f"    t_decay (2 t_e):   ∞")
    print(f"\n  Diffusion reach:")
    print(f"    Turbulent:         √(ν_t × t_decay)  = {m1_c['ell_turb']*1000:.2f} mm")
    print(f"    Molecular:         √(ν × t_obs)      = {m1_c['ell_mol']*1000:.2f} mm")
    print(f"    Max reach from δ:  δ + ℓ_turb + ℓ_mol = {m1_c['max_reach']*1000:.1f} mm")
    print(f"    Observation at:    z_obs              = {z_obs*1000:.1f} mm")

    gap_c = z_obs - delta_c
    if m1_c['max_reach'] < z_obs:
        print(f"    → Max reach ({m1_c['max_reach']*1000:.1f} mm) < z_obs "
              f"({z_obs*1000:.1f} mm): momentum CANNOT reach observation point")
    else:
        print(f"    → Max reach ({m1_c['max_reach']*1000:.1f} mm) ≥ z_obs "
              f"({z_obs*1000:.1f} mm): some momentum MAY reach observation point")

    u_str1_c = f"{m1_c['u_at_obs']:.2e}"
    u_str2_c = f"{m1_c['u_at_obs']*100:.4f}"
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = {u_str1_c} m/s{' '*(17-len(u_str1_c))}│")
    print(f"  │         = {u_str2_c} cm/s{' '*(30-len(u_str2_c))}│")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Phase 2 Summary ──
    print(f"\n{DIVIDER}")
    print("PHASE 2 SUMMARY")
    print(DIVIDER)
    print(f"\n  Carrier slide parameters:")
    print(f"    Mass:            m = {m_carrier:.3f} kg")
    print(f"    Travel:          Δr = {cs['delta_r']:.2f} m (r₀ → 0)")
    print(f"    Force:           F = {F_actuator:.1f} N")
    print(f"    Slide time:      t_slide = {t_slide*1000:.1f} ms")
    print(f"    Peak velocity:   V_max = {V_slide_max:.2f} m/s")
    print(f"\n  Boundary layer:")
    print(f"    BL thickness:    δ = {delta_c*1000:.2f} mm")
    print(f"    Gap to obs:      {max(gap_c,0)*1000:.1f} mm")
    if m1_c['t_decay'] < float('inf'):
        print(f"    Eddy decay:      {m1_c['t_decay']*1000:.0f} ms")

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │ PHASE 2 RESULT: Carrier slide velocity at z_obs, t_obs           │
  │   u = {m1_c['u_at_obs']:.2e} m/s = {m1_c['u_at_obs']*100:.4f} cm/s{' '*23}│
  └───────────────────────────────────────────────────────────────────┘""")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: VERTICAL PLATE LIFT
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PHASE 3: VERTICAL PLATE LIFT (after carrier slide completes)")
    print("=" * 70)

    print(f"\n  The carrier plate is lifted vertically by z_lift = {z_lift*1000:.1f} mm")
    print(f"  over t_lift = {t_lift:.3f} s, then waits t_wait = {t_wait:.3f} s.")
    print(f"  We compute velocity profiles to find quiescence heights.")

    # ── Lift Kinematics ──
    lift_kin = compute_lift_kinematics(z_lift, t_lift)
    V_lift = lift_kin["V_max"]

    print(f"\n{DIVIDER}")
    print("17. LIFT KINEMATICS (bang-bang vertical motion)")
    print(DIVIDER)
    print(f"  Lift parameters:")
    print(f"    Lift height:         z_lift = {z_lift*1000:.1f} mm")
    print(f"    Lift duration:       t_lift = {t_lift:.3f} s")
    print(f"\n  Bang-bang motion (a = 4·z/t², V_max = 2·z/t):")
    print(f"    Acceleration:        a = {lift_kin['a_lift']:.4f} m/s²")
    print(f"    Peak velocity:       V_max = {V_lift*1000:.3f} mm/s")
    print(f"    Accel time:          t_accel = {lift_kin['t_accel']*1000:.1f} ms")
    print(f"    Decel time:          t_decel = {lift_kin['t_decel']*1000:.1f} ms")

    # ── Stokes Layer ──
    delta_stokes = compute_stokes_layer_thickness(nu, t_lift)
    print(f"\n{DIVIDER}")
    print("18. STOKES LAYER FROM LIFT")
    print(DIVIDER)
    print(f"  Stokes first problem: plate impulsively moves through fluid.")
    print(f"  Boundary layer thickness: δ_Stokes = √(ν·t)")
    print(f"\n  Results:")
    print(f"    δ_Stokes = √({nu:.3e} × {t_lift:.3f}) = {delta_stokes*1000:.4f} mm")
    print(f"    Peak lift velocity:  V_lift = {V_lift*1000:.3f} mm/s = {V_lift*100:.5f} cm/s")

    # ── Profile 1: Blade BL diffusion ──
    # Time for Profile 1: t_obs after blade rotation stops (before slide starts)
    # But we need to use the combined time after rotation + slide + lift + wait
    # Actually, Profile 1 should be evaluated at the time when measurements are taken
    # which is t_wait after the lift completes
    # Total time since rotation stopped: t_slide + t_lift + t_wait
    t_profile1 = t_slide + t_lift + t_wait

    print(f"\n{DIVIDER}")
    print("19. PROFILE 1: BLADE BL DIFFUSION")
    print(DIVIDER)
    print(f"  Evaluating blade BL diffusion profile at t = {t_profile1:.3f} s after rotation stops.")
    print(f"  (This is: t_slide + t_lift + t_wait = {t_slide:.3f} + {t_lift:.3f} + {t_wait:.3f} s)")

    profile1 = compute_velocity_profile_blade(
        V_tip, delta, delta_star, nu, t_profile1, z_max, u_c
    )

    print(f"\n  BL properties:")
    print(f"    δ₁ (BL thickness):   {profile1['delta_1']*1000:.3f} mm")
    print(f"    Max reach:           {profile1['max_reach']*1000:.2f} mm")
    print(f"    Eddy decay time:     {profile1['t_decay']*1000:.1f} ms")
    print(f"\n  Quiescence result:")
    print(f"    Cutoff velocity:     u_c = {u_c*1000:.2f} mm/s")
    if profile1['z_q1'] < z_max:
        print(f"    z_q1 (quiescence):   {profile1['z_q1']*1000:.3f} mm")
    else:
        print(f"    z_q1 (quiescence):   > {z_max*1000:.1f} mm (not reached)")

    # Print sample profile
    print(f"\n  Velocity profile (sample points):")
    print(f"    {'z [mm]':>8s}  {'u [m/s]':>12s}  {'u [mm/s]':>10s}")
    z_arr = profile1['z_array']
    u_arr = profile1['u_array']
    for z_mm in [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]:
        z_val = z_mm * 1e-3
        if z_val <= z_max:
            idx = np.argmin(np.abs(z_arr - z_val))
            u_val = u_arr[idx]
            marker = ""
            if abs(z_arr[idx] - profile1['z_q1']) < 0.0005:
                marker = " ← z_q1"
            print(f"    {z_mm:8.1f}  {u_val:12.2e}  {u_val*1000:10.4f}{marker}")

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ PROFILE 1 RESULT:                                           │")
    print(f"  │   δ₁ = {profile1['delta_1']*1000:.3f} mm  (BL thickness)                        │")
    if profile1['z_q1'] < z_max:
        print(f"  │   z_q1 = {profile1['z_q1']*1000:.3f} mm  (quiescence height, u < {u_c*1000:.2f} mm/s)    │")
    else:
        print(f"  │   z_q1 > {z_max*1000:.1f} mm  (quiescence not reached)                │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # ── Profile 2: Carrier slide BL diffusion ──
    # Time for Profile 2: t_obs after slide stops = t_lift + t_wait
    t_profile2 = t_lift + t_wait

    print(f"\n{DIVIDER}")
    print("20. PROFILE 2: CARRIER SLIDE BL DIFFUSION")
    print(DIVIDER)
    print(f"  Evaluating carrier slide BL diffusion profile at t = {t_profile2:.3f} s after slide stops.")
    print(f"  (This is: t_lift + t_wait = {t_lift:.3f} + {t_wait:.3f} s)")

    profile2 = compute_velocity_profile_slide(
        V_slide_max, delta_c, bl_carrier["delta_star"], nu, t_profile2, z_max, u_c
    )

    print(f"\n  BL properties:")
    print(f"    δ₂ (BL thickness):   {profile2['delta_2']*1000:.3f} mm")
    print(f"    Max reach:           {profile2['max_reach']*1000:.2f} mm")
    print(f"    Eddy decay time:     {profile2['t_decay']*1000:.1f} ms")
    print(f"\n  Quiescence result:")
    print(f"    Cutoff velocity:     u_c = {u_c*1000:.2f} mm/s")
    if profile2['z_q2'] < z_max:
        print(f"    z_q2 (quiescence):   {profile2['z_q2']*1000:.3f} mm")
    else:
        print(f"    z_q2 (quiescence):   > {z_max*1000:.1f} mm (not reached)")

    # Print sample profile
    print(f"\n  Velocity profile (sample points):")
    print(f"    {'z [mm]':>8s}  {'u [m/s]':>12s}  {'u [mm/s]':>10s}")
    z_arr2 = profile2['z_array']
    u_arr2 = profile2['u_array']
    for z_mm in [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]:
        z_val = z_mm * 1e-3
        if z_val <= z_max:
            idx = np.argmin(np.abs(z_arr2 - z_val))
            u_val = u_arr2[idx]
            marker = ""
            if abs(z_arr2[idx] - profile2['z_q2']) < 0.0005:
                marker = " ← z_q2"
            print(f"    {z_mm:8.1f}  {u_val:12.2e}  {u_val*1000:10.4f}{marker}")

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ PROFILE 2 RESULT:                                           │")
    print(f"  │   δ₂ = {profile2['delta_2']*1000:.3f} mm  (BL thickness)                        │")
    if profile2['z_q2'] < z_max:
        print(f"  │   z_q2 = {profile2['z_q2']*1000:.3f} mm  (quiescence height, u < {u_c*1000:.2f} mm/s)    │")
    else:
        print(f"  │   z_q2 > {z_max*1000:.1f} mm  (quiescence not reached)                │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # ── Profile 3: Stokes layer from lift ──
    print(f"\n{DIVIDER}")
    print("21. PROFILE 3: STOKES LAYER FROM LIFT")
    print(DIVIDER)
    print(f"  Evaluating Stokes layer profile at t_wait = {t_wait:.3f} s after lift stops.")
    print(f"  Plate is now at z = z_lift = {z_lift*1000:.1f} mm (RF1 coordinates).")

    profile3 = compute_velocity_profile_lift(
        V_lift, delta_stokes, nu, t_wait, z_lift, z_max, u_c
    )

    print(f"\n  Stokes layer properties:")
    print(f"    δ₃ (Stokes thickness): {profile3['delta_3']*1000:.4f} mm")
    print(f"    Effective δ (diffused): {profile3['delta_eff']*1000:.4f} mm")
    print(f"\n  Quiescence result:")
    print(f"    Cutoff velocity:       u_c = {u_c*1000:.2f} mm/s")
    if profile3['z_q3'] < z_max:
        print(f"    z_q3 (quiescence):     {profile3['z_q3']*1000:.3f} mm (in RF1)")
    else:
        print(f"    z_q3 (quiescence):     > {z_max*1000:.1f} mm (not reached)")

    # Print sample profile
    print(f"\n  Velocity profile (sample points, z in RF1):")
    print(f"    {'z [mm]':>8s}  {'u [m/s]':>12s}  {'u [mm/s]':>10s}")
    z_arr3 = profile3['z_array']
    u_arr3 = profile3['u_array']
    for z_mm in [5, 5.5, 6, 7, 8, 10, 12, 15, 20]:
        z_val = z_mm * 1e-3
        if z_val >= z_lift and z_val <= z_max:
            idx = np.argmin(np.abs(z_arr3 - z_val))
            if idx < len(u_arr3):
                u_val = u_arr3[idx]
                marker = ""
                if abs(z_arr3[idx] - profile3['z_q3']) < 0.0005:
                    marker = " ← z_q3"
                print(f"    {z_mm:8.1f}  {u_val:12.2e}  {u_val*1000:10.4f}{marker}")

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ PROFILE 3 RESULT:                                           │")
    print(f"  │   δ₃ = {profile3['delta_3']*1000:.4f} mm  (Stokes layer thickness)              │")
    if profile3['z_q3'] < z_max:
        print(f"  │   z_q3 = {profile3['z_q3']*1000:.3f} mm  (quiescence height, u < {u_c*1000:.2f} mm/s)   │")
    else:
        print(f"  │   z_q3 > {z_max*1000:.1f} mm  (quiescence not reached)                │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # ── Phase 3 Summary ──
    print(f"\n{DIVIDER}")
    print("PHASE 3 SUMMARY")
    print(DIVIDER)
    print(f"\n  Lift parameters:")
    print(f"    Lift height:         z_lift = {z_lift*1000:.1f} mm")
    print(f"    Lift duration:       t_lift = {t_lift:.3f} s")
    print(f"    Peak velocity:       V_lift = {V_lift*1000:.3f} mm/s")
    print(f"    Wait time:           t_wait = {t_wait:.3f} s")
    print(f"\n  Quiescence cutoff:     u_c = {u_c*1000:.2f} mm/s")

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │ PHASE 3 RESULTS:                                                  │
  │                                                                   │
  │   Profile 1 (Blade rotation BL diffusion):                        │
  │     δ₁ = {profile1['delta_1']*1000:6.3f} mm                                             │""")
    if profile1['z_q1'] < z_max:
        print(f"  │     z_q1 = {profile1['z_q1']*1000:6.3f} mm                                            │")
    else:
        print(f"  │     z_q1 > {z_max*1000:5.1f} mm (not reached)                                │")
    print(f"  │                                                                   │")
    print(f"  │   Profile 2 (Carrier slide BL diffusion):                         │")
    print(f"  │     δ₂ = {profile2['delta_2']*1000:6.3f} mm                                             │")
    if profile2['z_q2'] < z_max:
        print(f"  │     z_q2 = {profile2['z_q2']*1000:6.3f} mm                                            │")
    else:
        print(f"  │     z_q2 > {z_max*1000:5.1f} mm (not reached)                                │")
    print(f"  │                                                                   │")
    print(f"  │   Profile 3 (Stokes layer from lift):                             │")
    print(f"  │     δ₃ = {profile3['delta_3']*1000:6.4f} mm                                            │")
    if profile3['z_q3'] < z_max:
        print(f"  │     z_q3 = {profile3['z_q3']*1000:6.3f} mm                                            │")
    else:
        print(f"  │     z_q3 > {z_max*1000:5.1f} mm (not reached)                                │")
    print(f"  └───────────────────────────────────────────────────────────────────┘")

    # ── Combined Timeline ──
    print(f"\n{DIVIDER}")
    print("COMBINED TIMELINE")
    print(DIVIDER)
    t_total_motion = t_rot + t_slide + t_lift
    t_total_cycle = t_rot + t_slide + t_lift + t_wait
    print(f"""
  Phase 1: Blade rotation
    t = 0 → {t_rot*1000:.1f} ms         Blade sweeps through θ = {np.degrees(sweep_angle):.0f}°
    Peak V_tip = {V_tip:.2f} m/s at midpoint

  Phase 2: Carrier slide (begins after Phase 1)
    t = {t_rot*1000:.1f} → {(t_rot+t_slide)*1000:.1f} ms   Carrier slides Δr = {cs['delta_r']:.2f} m to center
    Peak V_slide = {V_slide_max:.2f} m/s at midpoint

  Phase 3: Vertical lift (begins after Phase 2)
    t = {(t_rot+t_slide)*1000:.1f} ms → {t_total_motion:.3f} s   Plate lifts z_lift = {z_lift*1000:.1f} mm
    Peak V_lift = {V_lift*1000:.3f} mm/s at midpoint

  Wait period:
    t = {t_total_motion:.3f} → {t_total_cycle:.3f} s   Gas settles for t_wait = {t_wait:.3f} s

  TOTAL CYCLE TIME: {t_total_cycle:.3f} s = {t_total_cycle*1000:.1f} ms
""")

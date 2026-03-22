#!/usr/bin/env python3
"""
Residual gas velocity above a stopped rotating disk in dense xenon.
Mechanism-by-mechanism numerical evaluation.

Accompanies: disk_flow_analysis.tex

Usage:
    python disk_flow_numerics.py [OPTIONS]

Options:
    --h_disk       Disk thickness [m]                  (default: 0.005)
    --sweep_angle  Sweep angle [rad]                   (default: pi/8)
    --m_disk       Mass of disk [kg]                   (default: 50.0)
    --m_plate      Mass of each carrier plate [kg]     (default: 0.15)
    --N_plates     Number of carrier positions          (default: 8)
    --tau_motor    Motor torque [N·m]                   (default: 60.0)
    --z_obs        Observation height above disk [m]    (default: 0.005)
    --t_obs        Observation time after stop [s]      (default: 0.5)
    --numeric      Run full numerical PDE solver        (default: analytical)
    --drag         Include aerodynamic drag in kinematics (default: no drag)

Fixed parameters:
    R = 1.6 m, P = 15 bar, T = 300 K, vessel height H = 1.5 m,
    vessel diameter surplus = 10 mm.

Motion profile:
    Bang-bang (maximum acceleration then maximum deceleration).
    Includes skin-friction drag (turbulent rotating disk) and rim drag.
"""

import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import sys


# ═══════════════════════════════════════════════════════════════
# Parse command-line arguments
# ═══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Residual gas velocity above a stopped rotating disk in Xe."
    )
    parser.add_argument(
        "--h_disk", type=float, default=0.005,
        help="Disk thickness [m] (default: 0.005)"
    )
    parser.add_argument(
        "--sweep_angle", type=float, default=np.pi / 8,
        help="Sweep angle [rad] (default: pi/8 ≈ 0.3927)"
    )
    parser.add_argument(
        "--m_disk", type=float, default=50.0,
        help="Mass of disk [kg] (default: 50.0)"
    )
    parser.add_argument(
        "--m_plate", type=float, default=0.15,
        help="Mass of each carrier plate [kg] (default: 0.15)"
    )
    parser.add_argument(
        "--N_plates", type=int, default=8,
        help="Number of carrier positions (default: 8)"
    )
    parser.add_argument(
        "--tau_motor", type=float, default=60.0,
        help="Motor torque [N·m] (default: 60.0)"
    )
    parser.add_argument(
        "--z_obs", type=float, default=0.005,
        help="Observation height above disk surface [m] (default: 0.005)"
    )
    parser.add_argument(
        "--t_obs", type=float, default=0.5,
        help="Observation time after stop [s] (default: 0.5)"
    )
    parser.add_argument(
        "--numeric", action="store_true",
        help="Run full numerical PDE solver for BL diffusion (default: analytical)"
    )
    parser.add_argument(
        "--drag", action="store_true",
        help="Include aerodynamic drag in kinematics (default: analytical no-drag)"
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Fixed physical parameters
# ═══════════════════════════════════════════════════════════════
M_Xe = 0.13129       # kg/mol, molar mass of Xe
R_gas = 8.314        # J/(mol·K), universal gas constant
P = 15e5             # Pa, gas pressure
T = 300.0            # K, gas temperature
gamma = 5.0 / 3.0   # adiabatic index (monatomic)
mu_Xe = 2.32e-5      # Pa·s, dynamic viscosity of Xe at 300 K

R_disk = 1.6         # m, disk radius
Delta_D = 0.010      # m, vessel inner diameter surplus (10 mm)
H_vessel = 1.5       # m, vessel height
kappa = 0.41         # von Kármán constant


def compute_gas_properties():
    """Compute xenon gas properties at P, T."""
    rho = P * M_Xe / (R_gas * T)
    nu = mu_Xe / rho
    c_s = np.sqrt(gamma * R_gas * T / M_Xe)
    return rho, nu, c_s


# ═══════════════════════════════════════════════════════════════
# Kinematics: bang-bang motion
# ═══════════════════════════════════════════════════════════════
def compute_bangbang_kinematics_analytical(m_disk, m_plate, N_plates, tau_motor,
                                            sweep_angle):
    """
    Compute bang-bang motion kinematics WITHOUT aerodynamic drag (fast analytical).

    For pure bang-bang motion with constant acceleration/deceleration:
        α = τ/I                    (angular acceleration)
        θ_switch = θ/2             (switch at midpoint)
        ω_max = √(θ·τ/I)           (peak angular velocity)
        t_rot = 2√(θ·I/τ)          (total rotation time)
        V_tip = ω_max × R          (peak tip velocity)

    Returns dict with inertia and kinematics.
    """
    # ── Moment of inertia ──
    I_disk = 0.5 * m_disk * R_disk**2
    I_plates = N_plates * m_plate * R_disk**2
    I_total = I_disk + I_plates

    # ── Closed-form kinematics (no drag) ──
    alpha = tau_motor / I_total
    t_rot = 2.0 * np.sqrt(sweep_angle * I_total / tau_motor)
    omega_max = np.sqrt(sweep_angle * tau_motor / I_total)
    theta_switch = sweep_angle / 2.0
    t_accel = t_rot / 2.0
    t_decel = t_rot / 2.0
    V_tip_max = omega_max * R_disk

    return {
        "I_disk": I_disk,
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


def compute_bangbang_kinematics_numerical(m_disk, m_plate, N_plates, tau_motor,
                                           sweep_angle, rho, nu, h_disk):
    """
    Compute bang-bang motion kinematics for the rotating disk WITH
    aerodynamic drag (skin friction + rim drag).

    Drag model:
      - Skin friction (two sides): τ_skin = C_M ρ ω² R⁵
        where C_M = 0.146 Re_R^(-1/5), Re_R = ω R² / ν
        Ref: Schlichting & Gersten, Boundary-Layer Theory, 9th ed., Ch. 5
      - Rim drag: τ_rim = ½ ρ C_d,rim h_disk R⁴ ω²
        with C_d,rim ≈ 1.0 (blunt edge)
        Ref: Hoerner, Fluid-Dynamic Drag, Ch. 3
    """
    # ── Moment of inertia ──
    I_disk = 0.5 * m_disk * R_disk**2
    I_plates = N_plates * m_plate * R_disk**2
    I_total = I_disk + I_plates

    # ── Rim drag coefficient ──
    C_d_rim = 1.0

    # ── Inertia-only reference ──
    alpha_no_drag = tau_motor / I_total
    t_rot_no_drag = 2.0 * np.sqrt(sweep_angle * I_total / tau_motor)
    omega_max_no_drag = alpha_no_drag * (t_rot_no_drag / 2.0)

    # ── Drag torque as function of omega ──
    def drag_torque(omega):
        omega_abs = abs(omega)
        if omega_abs < 1e-12:
            return 0.0
        Re_R = omega_abs * R_disk**2 / nu
        C_M = 0.146 * Re_R**(-0.2)
        tau_skin = C_M * rho * omega_abs**2 * R_disk**5
        tau_rim = 0.5 * rho * C_d_rim * h_disk * R_disk**4 * omega_abs**2
        return tau_skin + tau_rim

    # ── ODE phases ──
    def accel_phase(t, y):
        omega = y[1]
        td = drag_torque(omega)
        return [omega, (tau_motor - td) / I_total]

    def decel_phase(t, y):
        omega = y[1]
        td = drag_torque(omega)
        return [omega, (-tau_motor - td) / I_total]

    def simulate_with_switch(theta_switch):
        def reached_switch(t, y):
            return y[0] - theta_switch
        reached_switch.terminal = True
        reached_switch.direction = 1

        sol1 = solve_ivp(
            accel_phase, [0, 1000], [0.0, 0.0],
            events=reached_switch, max_step=1e-4
        )
        if len(sol1.t_events[0]) == 0:
            return None
        t_accel = sol1.t_events[0][0]
        omega_at_switch = sol1.y_events[0][0][1]

        def omega_zero(t, y):
            return y[1]
        omega_zero.terminal = True
        omega_zero.direction = -1

        sol2 = solve_ivp(
            decel_phase, [0, 1000], [theta_switch, omega_at_switch],
            events=omega_zero, max_step=1e-4
        )
        if len(sol2.t_events[0]) == 0:
            return None
        t_decel = sol2.t_events[0][0]
        theta_final = sol2.y_events[0][0][0]

        return (theta_final, 0.0, t_accel, t_decel, omega_at_switch)

    # ── Find theta_switch via bisection ──
    def objective(theta_switch):
        result = simulate_with_switch(theta_switch)
        if result is None:
            return -sweep_angle
        return result[0] - sweep_angle

    theta_switch_opt = brentq(
        objective, sweep_angle * 0.5, sweep_angle * 0.99, xtol=1e-8
    )

    result = simulate_with_switch(theta_switch_opt)
    theta_final, _, t_accel, t_decel, omega_max = result
    t_rot = t_accel + t_decel
    V_tip_max = omega_max * R_disk

    # ── Drag at peak omega ──
    Re_R_peak = omega_max * R_disk**2 / nu
    C_M_peak = 0.146 * Re_R_peak**(-0.2) if Re_R_peak > 0 else 0
    tau_skin_peak = C_M_peak * rho * omega_max**2 * R_disk**5
    tau_rim_peak = 0.5 * rho * C_d_rim * h_disk * R_disk**4 * omega_max**2

    drag_ratio = (t_rot - t_rot_no_drag) / t_rot_no_drag
    omega_reduction = (omega_max_no_drag - omega_max) / omega_max_no_drag

    return {
        "I_disk": I_disk,
        "I_plates": I_plates,
        "I_total": I_total,
        "C_d_rim": C_d_rim,
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
        "Re_R_peak": Re_R_peak,
        "C_M_peak": C_M_peak,
        "tau_skin_peak": tau_skin_peak,
        "tau_rim_peak": tau_rim_peak,
        "drag_time_increase": drag_ratio,
        "omega_reduction": omega_reduction,
        "method": "numerical",
    }


def compute_bangbang_kinematics(m_disk, m_plate, N_plates, tau_motor,
                                 sweep_angle, rho, nu, h_disk, drag=False):
    """
    Wrapper for bang-bang kinematics calculation.

    Parameters:
        drag: If True, run numerical ODE solver with aerodynamic drag.
              If False (default), use closed-form analytical solution.

    Both methods return compatible dict structures.
    """
    if drag:
        return compute_bangbang_kinematics_numerical(
            m_disk, m_plate, N_plates, tau_motor, sweep_angle, rho, nu, h_disk
        )
    else:
        return compute_bangbang_kinematics_analytical(
            m_disk, m_plate, N_plates, tau_motor, sweep_angle
        )


# ═══════════════════════════════════════════════════════════════
# Boundary layer: von Kármán rotating disk
# ═══════════════════════════════════════════════════════════════
def compute_boundary_layer(omega, nu, rho):
    """
    Compute the von Kármán rotating-disk boundary layer.
    Returns properties at the disk edge (r = R).

    Laminar: δ = 5.5 √(ν/ω)   [Schlichting & Gersten, Ch. 5.2]
    Turbulent: δ = 0.526 r Re_r^(-1/5)  [Owen & Rogers, 1989]
    Transition at Re_r = ω r² / ν ≈ 2.5×10⁵  [Lingwood, 1995]
    """
    if omega < 1e-12:
        return {
            "delta_lam": 0, "delta_turb": 0, "delta": 0,
            "Re_R": 0, "r_trans": R_disk, "regime": "none",
            "w_pump_lam": 0, "Q_turb": 0, "w_pump_turb_mean": 0,
            "nu_t_outer": 0, "u_tau": 0,
        }

    Re_R = omega * R_disk**2 / nu
    Re_crit = 2.5e5  # transition Re_r

    # Transition radius
    r_trans = np.sqrt(Re_crit * nu / omega)
    if r_trans > R_disk:
        regime = "laminar"
    else:
        regime = "turbulent"

    # Laminar BL thickness (independent of r)
    delta_lam = 5.5 * np.sqrt(nu / omega)

    # Turbulent BL thickness at r = R
    Re_r_R = omega * R_disk**2 / nu
    delta_turb = 0.526 * R_disk * Re_r_R**(-0.2)

    # Use appropriate delta
    delta = delta_turb if regime == "turbulent" else delta_lam

    # Laminar axial pumping velocity (Cochran, 1934)
    w_pump_lam = 0.886 * np.sqrt(nu * omega)

    # Turbulent entrainment flow rate (Owen & Rogers, 1989)
    # Q = 0.219 Re_R^(4/5) ν R  (one side)
    Q_turb = 0.219 * Re_R**0.8 * nu * R_disk
    w_pump_turb_mean = Q_turb / (np.pi * R_disk**2)

    # Turbulent viscosity in the BL (outer layer)
    V_tip = omega * R_disk
    delta_star = delta / 8.0  # approximate displacement thickness
    nu_t_outer = 0.018 * V_tip * delta_star

    # Skin friction and friction velocity
    # Turbulent rotating disk: C_f ~ 0.0592 Re_r^(-1/5)
    Cf = 0.0592 * Re_r_R**(-0.2) if regime == "turbulent" else 0.664 / np.sqrt(Re_r_R)
    u_tau = V_tip * np.sqrt(Cf / 2.0)

    return {
        "delta_lam": delta_lam,
        "delta_turb": delta_turb,
        "delta": delta,
        "Re_R": Re_R,
        "r_trans": r_trans,
        "regime": regime,
        "w_pump_lam": w_pump_lam,
        "Q_turb": Q_turb,
        "w_pump_turb_mean": w_pump_turb_mean,
        "nu_t_outer": nu_t_outer,
        "u_tau": u_tau,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 1: Von Kármán pumping decay
# ═══════════════════════════════════════════════════════════════
def mechanism_vk_pumping(omega, nu, delta, nu_t_outer, V_tip, z_obs, t_obs):
    """
    Estimate the residual axial (pumping) velocity at z_obs, t_obs
    after the disk stops.

    The pumping velocity scales as w ~ √(ν_eff · ω_eff).
    After stop, ω_eff decays as the tangential BL decays.

    We model:
    - Turbulent phase (0 < t < 2 t_e): ω_eff decays exponentially
      with e-folding time ~ 2 t_e, ν_eff = ν_t
    - Molecular phase (t > 2 t_e): ω_eff decays via Stokes diffusion,
      ν_eff = ν

    The axial velocity at z_obs depends on whether z_obs is inside
    or outside the BL.
    """
    if omega < 1e-12:
        return {"w_at_obs": 0, "w_pump_lam_peak": 0,
                "w_pump_turb_peak": 0, "t_eddy": 0, "t_decay": 0}

    # Peak pumping velocities during rotation
    w_pump_lam_peak = 0.886 * np.sqrt(nu * omega)
    Q_turb = 0.219 * (omega * R_disk**2 / nu)**0.8 * nu * R_disk
    w_pump_turb_peak = Q_turb / (np.pi * R_disk**2)

    # Eddy timescales
    u_prime = 0.05 * V_tip
    t_eddy = delta / u_prime if u_prime > 0 else 1e10
    t_decay = 2.0 * t_eddy

    # After stop, the tangential velocity in the BL decays.
    # The pumping is driven by this tangential velocity.
    # Model: w(t) ~ w_peak * exp(-t / t_decay) for t < ~3 t_decay
    #        w(t) ~ w_lam * (t_decay / t)^(1/2) for t >> t_decay (molecular)

    # At t = t_obs:
    if t_obs < 3.0 * t_decay:
        # Still in turbulent decay phase
        w_at_obs = w_pump_turb_peak * np.exp(-t_obs / t_decay)
    else:
        # Turbulence dead; molecular diffusion phase
        # The tangential BL spreads as √(ν t), diluting the velocity.
        # Effective omega decays as ~ omega * (delta / √(ν t))²
        # (momentum conserved, spread over larger height)
        spread = np.sqrt(nu * t_obs)
        if spread > delta:
            omega_eff = omega * (delta / spread)**2
        else:
            omega_eff = omega
        w_mol = 0.886 * np.sqrt(nu * omega_eff)
        # But also the pumping mechanism requires active rotation,
        # which is gone. The residual is from diffusing momentum only.
        # The axial velocity at z_obs is driven by the radial gradient
        # of the tangential BL, which decays.
        w_at_obs = w_mol

    # If z_obs >> delta, the pumping velocity is already at its
    # asymptotic value (uniform above BL). If z_obs < delta,
    # it's smaller (linear in z near disk).
    if z_obs < delta:
        # Inside BL: w ~ w_inf * (z/delta)
        w_at_obs *= (z_obs / delta)

    return {
        "w_at_obs": w_at_obs,
        "w_pump_lam_peak": w_pump_lam_peak,
        "w_pump_turb_peak": w_pump_turb_peak,
        "t_eddy": t_eddy,
        "t_decay": t_decay,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 2: Tangential BL diffusion (same as propeller)
# ═══════════════════════════════════════════════════════════════
from scipy.special import erfc


def mechanism_tangential_diffusion_analytical(V_tip, delta, delta_star, nu, z_obs, t_obs):
    """
    Analytical estimate for tangential BL diffusion (fast).

    Uses diffusion length scales to determine if momentum can reach z_obs,
    and if so, estimates the velocity using error function solutions.
    Same approach as propeller eddy diffusion analytical.
    """
    # ── Turbulent viscosity parameters ──
    nu_t_outer = 0.018 * V_tip * delta_star

    # Turbulence intensity and eddy timescale
    u_prime = 0.05 * V_tip
    t_eddy = delta / u_prime if u_prime > 0 else 1e10
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
        zeta = z_obs / delta
        v_initial = V_tip * (1.0 - zeta ** (1.0 / 7.0))
        nu_eff_avg = nu + nu_t_outer * 0.5
        t_char = delta**2 / nu_eff_avg
        decay = np.exp(-t_obs / t_char)
        v_at_obs = v_initial * decay
    elif z_obs > max_reach:
        # Momentum cannot reach observation point
        v_at_obs = 0.0
    else:
        # Observation point is in the diffusion tail
        if t_obs < t_decay:
            ell_diff = np.sqrt(4.0 * (nu + nu_t_outer * 0.5) * t_obs)
        else:
            ell_diff = np.sqrt(4.0 * nu_t_outer * t_decay + 4.0 * nu * (t_obs - t_decay))

        if ell_diff > 0:
            eta = gap / ell_diff
            v_at_obs = V_tip * 0.5 * erfc(eta)
        else:
            v_at_obs = 0.0

        # Additional decay from no-slip absorption
        absorption_factor = np.exp(-t_obs / t_decay)
        v_at_obs *= absorption_factor

    # Build velocity profile at selected heights
    profile = {}
    for zi_mm in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50]:
        zi = zi_mm * 1e-3
        gap_i = zi - delta
        if zi <= 0:
            profile[zi_mm] = 0.0
        elif gap_i <= 0:
            zeta = zi / delta
            v_init = V_tip * (1.0 - zeta ** (1.0 / 7.0))
            nu_eff_avg = nu + nu_t_outer * 0.5
            t_char = delta**2 / nu_eff_avg
            profile[zi_mm] = v_init * np.exp(-t_obs / t_char)
        elif zi > max_reach:
            profile[zi_mm] = 0.0
        else:
            if t_obs < t_decay:
                ell_diff = np.sqrt(4.0 * (nu + nu_t_outer * 0.5) * t_obs)
            else:
                ell_diff = np.sqrt(4.0 * nu_t_outer * t_decay + 4.0 * nu * (t_obs - t_decay))
            if ell_diff > 0:
                eta = gap_i / ell_diff
                v_val = V_tip * 0.5 * erfc(eta) * np.exp(-t_obs / t_decay)
            else:
                v_val = 0.0
            profile[zi_mm] = v_val

    # Estimate momentum
    mom_initial = V_tip * delta * 7.0 / 8.0
    mom_fraction = np.exp(-t_obs / t_decay)

    return {
        "v_at_obs": v_at_obs,
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
        "profile": profile,
        "method": "analytical",
    }


def mechanism_tangential_diffusion_numerical(V_tip, delta, delta_star, u_tau, nu,
                                              z_obs, t_obs):
    """
    Solve the 1D diffusion equation for tangential momentum above
    the disk after it stops, with decaying turbulent viscosity.
    Same PDE as propeller eddy diffusion.
    """
    nu_t_inner_peak = kappa * u_tau * 0.2 * delta
    nu_t_outer = 0.018 * V_tip * delta_star

    u_prime = 0.05 * V_tip
    t_eddy = delta / u_prime if u_prime > 0 else 1e10
    t_decay = 2.0 * t_eddy

    ell_turb = np.sqrt(nu_t_outer * t_decay)
    ell_mol = np.sqrt(nu * t_obs)
    max_reach = delta + ell_turb + ell_mol

    # ── 1D PDE: dv/dt = d/dz [nu_eff dv/dz] ──
    L = max(5.0 * delta, z_obs + 5.0 * delta)
    if L < 0.01:
        L = 0.05  # minimum domain
    Nz = 2000
    dz = L / Nz
    z = np.linspace(0, L, Nz + 1)

    # Initial condition: v(z) = V_tip (1 - (z/delta)^(1/7)) for z < delta
    v = np.zeros(Nz + 1)
    for i in range(Nz + 1):
        if 0 < z[i] < delta:
            v[i] = V_tip * (1.0 - (z[i] / delta) ** (1.0 / 7.0))
    v[0] = 0.0  # no-slip on stopped disk

    mom_initial = np.trapezoid(v, z) if hasattr(np, 'trapezoid') else np.trapz(v, z)

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

        v_new = v.copy()
        for i in range(1, Nz):
            nu_r = 0.5 * (nu_eff[i] + nu_eff[i + 1])
            nu_l = 0.5 * (nu_eff[i] + nu_eff[i - 1])
            v_new[i] = v[i] + dt / dz * (
                nu_r * (v[i + 1] - v[i]) / dz
                - nu_l * (v[i] - v[i - 1]) / dz
            )

        v_new[0] = 0.0
        v_new[Nz] = 0.0
        v = v_new
        t_current += dt

    idx_obs = int(round(z_obs / dz))
    idx_obs = min(idx_obs, Nz)
    v_at_obs = v[idx_obs]

    mom_final = np.trapezoid(v, z) if hasattr(np, 'trapezoid') else np.trapz(v, z)

    profile = {}
    for zi_mm in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50]:
        idx = int(round(zi_mm * 1e-3 / dz))
        if idx <= Nz:
            profile[zi_mm] = v[idx]

    return {
        "v_at_obs": v_at_obs,
        "nu_t_inner_peak": nu_t_inner_peak,
        "nu_t_outer": nu_t_outer,
        "nu_t_over_nu": nu_t_outer / nu if nu > 0 else 0,
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


def mechanism_tangential_diffusion(V_tip, delta, delta_star, u_tau, nu,
                                    z_obs, t_obs, numeric=False):
    """
    Wrapper for tangential BL diffusion calculation.

    Parameters:
        numeric: If True, run full PDE solver. If False (default), use analytical estimate.

    Both methods return the same dict structure for compatibility.
    """
    if numeric:
        return mechanism_tangential_diffusion_numerical(V_tip, delta, delta_star, u_tau, nu,
                                                         z_obs, t_obs)
    else:
        return mechanism_tangential_diffusion_analytical(V_tip, delta, delta_star, nu, z_obs, t_obs)


# ═══════════════════════════════════════════════════════════════
# Mechanism 3: Bulk Ekman spin-down
# ═══════════════════════════════════════════════════════════════
def mechanism_bulk_ekman(rho, nu, omega, delta, V_tip, z_obs, t_obs):
    """Evaluate bulk Ekman spin-down timescales."""
    gap = z_obs - delta
    t_visc = gap**2 / nu if gap > 0 else 0.0

    Omega_wake = 0.5 * V_tip / R_disk if R_disk > 0 else 1e-10

    delta_E_mol = np.sqrt(nu / Omega_wake)
    w_E_mol = np.sqrt(nu * Omega_wake)

    nu_t_outer = 0.018 * V_tip * (delta / 8.0)
    t_eddy = delta / (0.05 * V_tip) if V_tip > 0 else 1e10
    t_turb_life = 2.0 * t_eddy

    delta_E_turb = np.sqrt(nu_t_outer / Omega_wake)
    w_E_turb = np.sqrt(nu_t_outer * Omega_wake)
    z_Ekman_turb = w_E_turb * t_turb_life

    t_Ekman_mol = H_vessel / np.sqrt(nu * Omega_wake)
    t_Ekman_turb = H_vessel / np.sqrt(nu_t_outer * Omega_wake)

    return {
        "gap": gap,
        "t_visc": t_visc,
        "Omega_wake": Omega_wake,
        "delta_E_mol": delta_E_mol,
        "w_E_mol": w_E_mol,
        "delta_E_turb": delta_E_turb,
        "w_E_turb": w_E_turb,
        "z_Ekman_turb": z_Ekman_turb,
        "t_Ekman_mol": t_Ekman_mol,
        "t_Ekman_turb": t_Ekman_turb,
        "t_turb_life": t_turb_life,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 4: Displacement (potential) flow
# ═══════════════════════════════════════════════════════════════
def mechanism_potential_flow(V_tip, h_disk, z_obs, c_s):
    """Evaluate displacement flow from rotating disk rim."""
    t_acoustic = R_disk / c_s
    Delta_r = Delta_D / 2.0
    blockage = h_disk / Delta_r if Delta_r > 0 else float('inf')

    # Displacement velocity estimate (rim as blunt body)
    w_disp = V_tip * (h_disk / 2.0)**2 / (z_obs**2 + (h_disk / 2.0)**2)

    return {
        "h_disk": h_disk,
        "w_disp_during_sweep": w_disp,
        "t_acoustic": t_acoustic,
        "blockage_ratio": blockage,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 5: Pressure pulse
# ═══════════════════════════════════════════════════════════════
def mechanism_pressure_pulse(rho, V_tip, c_s, z_obs):
    """Evaluate transient pressure pulse when disk stops."""
    t_decel = 0.01
    dp = rho * V_tip**2
    du_acoustic = dp / (rho * c_s)
    t_arrive = z_obs / c_s
    dp_fraction = dp / P

    return {
        "t_decel": t_decel,
        "dp": dp,
        "dp_fraction": dp_fraction,
        "du_acoustic": du_acoustic,
        "t_arrive": t_arrive,
    }


# ═══════════════════════════════════════════════════════════════
# Mechanism 6: Rim separation wake
# ═══════════════════════════════════════════════════════════════
def mechanism_rim_wake(V_tip, h_disk, nu):
    """Evaluate rim separation wake."""
    Re_rim = V_tip * h_disk / nu
    return {
        "h_disk": h_disk,
        "Re_rim": Re_rim,
    }


# ═══════════════════════════════════════════════════════════════
# Vessel confinement effects
# ═══════════════════════════════════════════════════════════════
def compute_confinement(nu, omega, V_tip, delta, c_s, h_disk, t_rot):
    """Compute all vessel confinement numerical estimates."""
    Delta_r = Delta_D / 2.0
    R_cyl = R_disk + Delta_r

    delta_wall = np.sqrt(nu * t_rot)

    Omega_wake = 0.5 * V_tip / R_disk if R_disk > 0 else 1e-10
    delta_E_mol = np.sqrt(nu / Omega_wake)
    w_E_mol = np.sqrt(nu * Omega_wake)
    t_Ekman_mol = H_vessel / np.sqrt(nu * Omega_wake)

    nu_t_outer = 0.018 * V_tip * (delta / 8.0)
    t_eddy = delta / (0.05 * V_tip) if V_tip > 0 else 1e10
    t_turb_life = 2.0 * t_eddy
    delta_E_turb = np.sqrt(nu_t_outer / Omega_wake)
    w_E_turb = np.sqrt(nu_t_outer * Omega_wake)
    z_Ekman_turb = w_E_turb * t_turb_life
    t_Ekman_turb = H_vessel / np.sqrt(nu_t_outer * Omega_wake)

    Re_gap = V_tip * Delta_r / nu

    ell_max = np.sqrt((2 * R_cyl)**2 + H_vessel**2)
    t_reverb = ell_max / c_s
    t_disp = R_disk / c_s

    return {
        "Delta_r": Delta_r,
        "R_cyl": R_cyl,
        "delta_wall": delta_wall,
        "delta_E_mol": delta_E_mol,
        "w_E_mol": w_E_mol,
        "t_Ekman_mol": t_Ekman_mol,
        "delta_E_turb": delta_E_turb,
        "w_E_turb": w_E_turb,
        "z_Ekman_turb": z_Ekman_turb,
        "t_Ekman_turb": t_Ekman_turb,
        "t_turb_life": t_turb_life,
        "Re_gap": Re_gap,
        "ell_max": ell_max,
        "t_reverb": t_reverb,
        "t_disp": t_disp,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()

    h_disk = args.h_disk
    sweep_angle = args.sweep_angle
    m_disk = args.m_disk
    m_plate = args.m_plate
    N_plates = args.N_plates
    tau_motor = args.tau_motor
    z_obs = args.z_obs
    t_obs = args.t_obs
    numeric = args.numeric
    drag = args.drag

    DIVIDER = "=" * 70

    # ── Header ──
    print(DIVIDER)
    print("RESIDUAL GAS VELOCITY ABOVE A STOPPED ROTATING DISK IN DENSE XENON")
    print(DIVIDER)
    print(f"\n  Variable parameters:")
    print(f"    Disk thickness:      h = {h_disk*1000:.1f} mm")
    print(f"    Sweep angle:         θ = {sweep_angle:.5f} rad "
          f"({np.degrees(sweep_angle):.1f}°)")
    print(f"    Disk mass:           m_disk = {m_disk:.1f} kg")
    print(f"    Plate mass (each):   m_plate = {m_plate:.3f} kg")
    print(f"    Number of positions: N = {N_plates}")
    print(f"    Motor torque:        τ = {tau_motor:.1f} N·m")
    print(f"    Observation height:  z = {z_obs*1000:.1f} mm")
    print(f"    Observation time:    t = {t_obs:.3f} s")
    print(f"\n  Fixed parameters:")
    print(f"    Disk radius:         R = {R_disk} m")
    print(f"    Pressure:            P = {P/1e5:.0f} bar")
    print(f"    Temperature:         T = {T:.0f} K")
    print(f"    Vessel height:       H = {H_vessel} m")
    print(f"    Vessel diam. surplus:  ΔD = {Delta_D*1000:.0f} mm")

    # ── Gas properties ──
    rho, nu, c_s = compute_gas_properties()
    rho_air = 1.2
    nu_air = 1.5e-5
    print(f"\n{DIVIDER}")
    print("1. XENON GAS PROPERTIES")
    print(DIVIDER)
    print(f"  Density:              ρ  = {rho:.1f} kg/m³  "
          f"({rho/rho_air:.0f}× denser than air)")
    print(f"  Dynamic viscosity:    μ  = {mu_Xe:.2e} Pa·s")
    print(f"  Kinematic viscosity:  ν  = {nu:.3e} m²/s  "
          f"({nu_air/nu:.0f}× smaller than air)")
    print(f"  Speed of sound:       cₛ = {c_s:.1f} m/s")

    # ── Bang-bang kinematics ──
    bb = compute_bangbang_kinematics(
        m_disk, m_plate, N_plates, tau_motor, sweep_angle, rho, nu, h_disk, drag=drag
    )
    t_rot = bb["t_rot"]
    omega_max = bb["omega_max"]
    V_tip = bb["V_tip_max"]

    print(f"\n{DIVIDER}")
    if drag:
        print("2. BANG-BANG KINEMATICS (with aerodynamic drag)")
    else:
        print("2. BANG-BANG KINEMATICS (analytical, no drag)")
    print(DIVIDER)
    print(f"  Moment of inertia:")
    print(f"    I_disk   = ½ m R²:        {bb['I_disk']:.2f} kg·m²")
    print(f"    I_plates = N m_plate R²:  {bb['I_plates']:.2f} kg·m²")
    print(f"    I_total:                  {bb['I_total']:.2f} kg·m²")

    if drag:
        print(f"\n  Aerodynamic drag at peak ω:")
        print(f"    Re_R = ω R²/ν:            {bb['Re_R_peak']:.3e}")
        print(f"    C_M (turbulent skin):      {bb['C_M_peak']:.5f}")
        print(f"    τ_skin (both sides):       {bb['tau_skin_peak']:.3f} N·m")
        print(f"    τ_rim  (blunt edge):       {bb['tau_rim_peak']:.3f} N·m")
        print(f"\n  Inertia-only reference (no drag):")
        print(f"    α = τ/I = {bb['alpha_no_drag']:.3f} rad/s²")
        print(f"    t_rot = {bb['t_rot_no_drag']*1000:.1f} ms")
        print(f"    ω_max = {bb['omega_max_no_drag']:.4f} rad/s")
        print(f"\n  With drag (numerical integration):")
        print(f"    Switching angle:        θ_switch = {bb['theta_switch']:.5f} rad "
              f"({np.degrees(bb['theta_switch']):.2f}°)")
        print(f"    Accel time:             t_accel  = {bb['t_accel']*1000:.2f} ms")
        print(f"    Decel time:             t_decel  = {bb['t_decel']*1000:.2f} ms")
        print(f"    Total rotation time:    t_rot    = {t_rot*1000:.2f} ms")
        print(f"    Peak ω:                 ω_max    = {omega_max:.5f} rad/s")
        print(f"    Peak V_tip:             V_tip    = {V_tip:.4f} m/s")
        print(f"    Final θ:                θ_final  = {bb['theta_final']:.5f} rad "
              f"(target: {sweep_angle:.5f})")
        print(f"\n  Drag impact:")
        print(f"    Time increase:          {bb['drag_time_increase']*100:+.2f}%")
        print(f"    ω_max reduction:        {bb['omega_reduction']*100:.2f}%")
    else:
        print(f"\n  Closed-form solution (I dω/dt = ±τ, no drag):")
        print(f"    α = τ/I = {bb['alpha']:.3f} rad/s²")
        print(f"    θ_switch = θ/2 = {bb['theta_switch']:.5f} rad "
              f"({np.degrees(bb['theta_switch']):.2f}°)")
        print(f"    t_accel = t_decel = {bb['t_accel']*1000:.2f} ms")
        print(f"    Total rotation time:    t_rot    = {t_rot*1000:.2f} ms")
        print(f"    Peak ω (at θ_switch):   ω_max    = {omega_max:.5f} rad/s")
        print(f"    Peak V_tip:             V_tip    = {V_tip:.4f} m/s")
        print(f"\n  NOTE: Use --drag flag for numerical solution with aerodynamic drag.")

    # ── Flow regime ──
    Ma_tip = V_tip / c_s
    Re_R = omega_max * R_disk**2 / nu
    print(f"\n{DIVIDER}")
    print("3. FLOW REGIME (at peak velocity)")
    print(DIVIDER)
    print(f"  Tip Mach number:      Ma    = {Ma_tip:.4f}")
    print(f"  Rotational Reynolds:  Re_R  = {Re_R:.3e}")
    if Ma_tip < 0.3:
        print(f"  → INCOMPRESSIBLE (Ma < 0.3)")
    else:
        print(f"  → Compressibility effects may matter")
    Re_crit = 2.5e5
    r_trans = np.sqrt(Re_crit * nu / omega_max) if omega_max > 0 else R_disk
    if r_trans < R_disk:
        print(f"  → TURBULENT (transition at r = {r_trans:.3f} m, "
              f"i.e. {r_trans/R_disk*100:.1f}% of radius)")
    else:
        print(f"  → LAMINAR over entire disk")

    # ── Boundary layer ──
    bl = compute_boundary_layer(omega_max, nu, rho)
    delta = bl["delta"]
    V_tip_for_bl = omega_max * R_disk

    print(f"\n{DIVIDER}")
    print("4. VON KÁRMÁN BOUNDARY LAYER (at disk edge, r = R)")
    print(DIVIDER)
    print(f"  Laminar BL thickness:   δ_lam  = 5.5√(ν/ω) = "
          f"{bl['delta_lam']*1000:.3f} mm")
    print(f"  Turbulent BL thickness: δ_turb = 0.526 R Re_R⁻⁰·² = "
          f"{bl['delta_turb']*1000:.2f} mm")
    print(f"  Active regime:          {bl['regime']}")
    print(f"  BL thickness used:      δ = {delta*1000:.2f} mm")
    print(f"\n  Axial pumping (during rotation):")
    print(f"    Laminar:  w∞ = 0.886√(νω)  = {bl['w_pump_lam']*1000:.3f} mm/s")
    print(f"    Turbulent (mean over disk): = {bl['w_pump_turb_mean']*100:.3f} cm/s")
    print(f"    Turbulent flow rate Q:      = {bl['Q_turb']*1000:.2f} L/s")

    # ── Geometry ──
    gap = z_obs - delta
    R_cyl = R_disk + Delta_D / 2.0
    tip_gap = Delta_D / 2.0
    print(f"\n{DIVIDER}")
    print("5. VERTICAL GEOMETRY")
    print(DIVIDER)
    print(f"  Disk thickness:         h_disk    = {h_disk*1000:.1f} mm")
    print(f"  BL thickness at edge:   δ         = {delta*1000:.2f} mm")
    print(f"  Observation height:     z_obs     = {z_obs*1000:.1f} mm above disk")
    print(f"  Gap (BL top → obs):     Δz_gap    = {gap*1000:.2f} mm")
    print(f"  Vessel inner radius:    R_cyl     = {R_cyl:.3f} m")
    print(f"  Tip clearance:          Δr        = {tip_gap*1000:.0f} mm")

    if gap <= 0:
        print(f"\n  *** WARNING: z_obs ({z_obs*1000:.1f} mm) is INSIDE the "
              f"boundary layer (δ = {delta*1000:.1f} mm). ***")
        print(f"  *** The observation point has nonzero velocity at t=0. ***")

    # ── Mechanism 1: von Kármán pumping ──
    print(f"\n{DIVIDER}")
    print("6. MECHANISM 1: VON KÁRMÁN PUMPING (axial velocity)")
    print(DIVIDER)

    m1 = mechanism_vk_pumping(omega_max, nu, delta,
                               bl["nu_t_outer"], V_tip, z_obs, t_obs)

    print(f"\n  Peak pumping velocity during rotation:")
    print(f"    Laminar:   w_lam   = {m1['w_pump_lam_peak']*1000:.3f} mm/s")
    print(f"    Turbulent: w_turb  = {m1['w_pump_turb_peak']*100:.3f} cm/s")
    print(f"\n  Eddy timescales:")
    print(f"    t_eddy:    {m1['t_eddy']*1000:.1f} ms")
    print(f"    t_decay:   {m1['t_decay']*1000:.1f} ms")
    print(f"\n  Residual at (z_obs, t_obs):")
    print(f"    w = {m1['w_at_obs']:.3e} m/s = {m1['w_at_obs']*100:.4f} cm/s")

    w_str1 = f"{m1['w_at_obs']:.2e}"
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: w(z_obs, t_obs) = {w_str1} m/s{' '*(17-len(w_str1))}│")
    print(f"  │ (axial pumping, decaying after stop)             │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 2: Tangential BL diffusion ──
    print(f"\n{DIVIDER}")
    print("7. MECHANISM 2: TANGENTIAL BL DIFFUSION")
    print(DIVIDER)
    if numeric:
        print(f"  Solving 1D diffusion PDE numerically (--numeric flag set)...")
    else:
        print(f"  Using analytical estimate (use --numeric for full PDE solver)...")

    delta_star = delta / 8.0
    m2 = mechanism_tangential_diffusion(
        V_tip, delta, delta_star, bl["u_tau"], nu, z_obs, t_obs, numeric=numeric
    )

    print(f"\n  Method: {m2.get('method', 'unknown').upper()}")
    print(f"\n  Turbulent viscosity:")
    if 'nu_t_inner_peak' in m2:
        print(f"    ν_t (inner peak):  {m2['nu_t_inner_peak']:.3e} m²/s")
    print(f"    ν_t (outer):       {m2['nu_t_outer']:.3e} m²/s")
    print(f"    ν_t / ν:           {m2['nu_t_over_nu']:.0f}×")
    print(f"\n  Eddy scales:")
    print(f"    u':                {m2['u_prime']:.4f} m/s")
    print(f"    t_eddy (δ/u'):     {m2['t_eddy']*1000:.1f} ms")
    print(f"    t_decay (2 t_e):   {m2['t_decay']*1000:.1f} ms")
    print(f"\n  Diffusion reach:")
    print(f"    Turbulent:         √(ν_t × t_decay)  = {m2['ell_turb']*1000:.2f} mm")
    print(f"    Molecular:         √(ν × t_obs)      = {m2['ell_mol']*1000:.2f} mm")
    print(f"    Max reach from δ:  δ + ℓ_turb + ℓ_mol = {m2['max_reach']*1000:.1f} mm")
    print(f"    Observation at:    z_obs              = {z_obs*1000:.1f} mm")

    if m2['max_reach'] < z_obs:
        print(f"    → Max reach ({m2['max_reach']*1000:.1f} mm) < z_obs "
              f"({z_obs*1000:.1f} mm)")
    else:
        print(f"    → Max reach ({m2['max_reach']*1000:.1f} mm) ≥ z_obs "
              f"({z_obs*1000:.1f} mm)")

    print(f"\n  Velocity profile at t = {t_obs} s:")
    print(f"    {'z [mm]':>8s}  {'v_φ [m/s]':>10s}  {'v_φ [cm/s]':>10s}")
    for zi_mm, vel in sorted(m2['profile'].items()):
        marker = " ← z_obs" if abs(zi_mm - z_obs * 1000) < 0.5 else ""
        print(f"    {zi_mm:8.1f}  {vel:10.6f}  {vel*100:10.4f}{marker}")

    v_str1 = f"{m2['v_at_obs']:.2e}"
    v_str2 = f"{m2['v_at_obs']*100:.4f}"
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: v_φ(z_obs, t_obs) = {v_str1} m/s{' '*(14-len(v_str1))}│")
    print(f"  │         = {v_str2} cm/s{' '*(30-len(v_str2))}│")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 3: Bulk Ekman ──
    print(f"\n{DIVIDER}")
    print("8. MECHANISM 3: BULK EKMAN SPIN-DOWN")
    print(DIVIDER)

    m3 = mechanism_bulk_ekman(rho, nu, omega_max, delta, V_tip, z_obs, t_obs)

    print(f"  Gap (BL top → obs):  {m3['gap']*1000:.1f} mm")
    if m3['t_visc'] > 0:
        print(f"  Mol. diffusion time: t_visc = gap²/ν = {m3['t_visc']:.0f} s")
    print(f"\n  Ekman pumping (molecular):")
    print(f"    δ_E = {m3['delta_E_mol']*1000:.2f} mm")
    print(f"    w_E = {m3['w_E_mol']*1e6:.1f} μm/s")
    print(f"    t_Ekman = {m3['t_Ekman_mol']:.0f} s")
    print(f"\n  Ekman pumping (turbulent, first {m3['t_turb_life']*1000:.0f} ms):")
    print(f"    δ_E = {m3['delta_E_turb']*1000:.1f} mm")
    print(f"    w_E = {m3['w_E_turb']*100:.2f} cm/s")
    print(f"    z_reach = {m3['z_Ekman_turb']*1000:.2f} mm")
    print(f"    t_Ekman = {m3['t_Ekman_turb']:.0f} s")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) ≈ 0  (Ekman spin-down)  │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 4: Potential flow ──
    print(f"\n{DIVIDER}")
    print("9. MECHANISM 4: DISPLACEMENT (POTENTIAL) FLOW")
    print(DIVIDER)

    m4 = mechanism_potential_flow(V_tip, h_disk, z_obs, c_s)

    print(f"  Disk thickness:         {m4['h_disk']*1000:.1f} mm")
    print(f"  Blockage ratio:         {m4['blockage_ratio']:.2f}")
    print(f"  Displacement velocity during sweep:")
    print(f"    w_disp = {m4['w_disp_during_sweep']:.4f} m/s")
    print(f"  Acoustic clearing time: R/c_s = {m4['t_acoustic']*1000:.1f} ms")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = 0  (potential flow)   │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 5: Pressure pulse ──
    print(f"\n{DIVIDER}")
    print("10. MECHANISM 5: PRESSURE PULSE")
    print(DIVIDER)

    m5 = mechanism_pressure_pulse(rho, V_tip, c_s, z_obs)

    print(f"  Pressure perturbation:  Δp = {m5['dp']:.0f} Pa  "
          f"(Δp/P = {m5['dp_fraction']:.2e})")
    print(f"  Induced velocity:       Δu = {m5['du_acoustic']:.4f} m/s")
    print(f"  Arrival at z_obs:       {m5['t_arrive']*1e6:.1f} μs")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = 0  (pressure pulse)   │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Mechanism 6: Rim wake ──
    print(f"\n{DIVIDER}")
    print("11. MECHANISM 6: RIM SEPARATION WAKE")
    print(DIVIDER)

    m6 = mechanism_rim_wake(V_tip, h_disk, nu)

    print(f"  Rim height:           {m6['h_disk']*1000:.1f} mm")
    print(f"  Re (rim):             {m6['Re_rim']:.2e}")
    print(f"  Wake is tangential, confined to disk-edge annular gap")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │ RESULT: u(z_obs, t_obs) = 0  (rim wake)         │")
    print(f"  └─────────────────────────────────────────────────┘")

    # ── Vessel confinement ──
    print(f"\n{DIVIDER}")
    print("12. VESSEL CONFINEMENT EFFECTS")
    print(DIVIDER)

    conf = compute_confinement(nu, omega_max, V_tip, delta, c_s, h_disk, t_rot)

    print(f"\n  Vessel dimensions:")
    print(f"    Height:             H     = {H_vessel} m")
    print(f"    Inner radius:       R_cyl = {conf['R_cyl']:.3f} m")
    print(f"    Radial tip gap:     Δr    = {conf['Delta_r']*1000:.0f} mm")
    print(f"\n  (a) Wall BL (Stokes penetration during sweep):")
    print(f"    δ_wall = √(ν·t_rot) = {conf['delta_wall']*1000:.3f} mm")
    print(f"\n  (b) Ekman spin-up:")
    print(f"    Molecular: t_Ekman = {conf['t_Ekman_mol']:.0f} s")
    print(f"    Turbulent (first {conf['t_turb_life']*1000:.0f} ms): "
          f"t_Ekman = {conf['t_Ekman_turb']:.0f} s")
    print(f"    Turb. vertical reach: {conf['z_Ekman_turb']*1000:.2f} mm")
    print(f"\n  (c) Tip-gap Couette:")
    print(f"    Re_gap = {conf['Re_gap']:.2e}")
    regime = "turbulent" if conf['Re_gap'] > 1e4 else "laminar"
    print(f"    Flow regime: {regime}, purely tangential")
    print(f"\n  (d) Acoustic reverberation:")
    print(f"    t_reverb = {conf['t_reverb']*1000:.1f} ms")
    print(f"\n  (e) Displacement readjustment:")
    print(f"    R/c_s = {conf['t_disp']*1000:.1f} ms")

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ CONFINEMENT SUMMARY: No new vertical transport mechanism.  │")
    print(f"  │ Same conclusion as propeller analysis.                     │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # ── Summary ──
    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)
    print(f"\n  Parameters: h = {h_disk*1000:.0f} mm, θ = {np.degrees(sweep_angle):.1f}°, "
          f"N = {N_plates}, t_rot = {t_rot*1000:.1f} ms, "
          f"z_obs = {z_obs*1000:.1f} mm, t_obs = {t_obs} s")
    print(f"\n  Key scales:")
    print(f"    BL thickness at edge (δ): {delta*1000:.2f} mm")
    print(f"    Gap to observation:       {max(gap,0)*1000:.1f} mm")
    print(f"    Eddy decay time:          {m2['t_decay']*1000:.0f} ms")
    print(f"    Acoustic clearing time:   {m4['t_acoustic']*1000:.0f} ms")
    print(f"    Peak V_tip:               {V_tip:.4f} m/s")

    print(f"""
  ┌────────────────────────────┬────────────────────┬──────────────────┐
  │ Mechanism                  │ Timescale          │ u at (z,t)       │
  ├────────────────────────────┼────────────────────┼──────────────────┤
  │ 1. Von Kármán pumping      │ ~{m1['t_decay']*1000:4.0f} ms decay     │ {m1['w_at_obs']:.2e} m/s │
  │ 2. Tangential BL diffusion │ ~{m2['t_decay']*1000:4.0f} ms decay     │ {m2['v_at_obs']:.2e} m/s │
  │ 3. Bulk Ekman spin-down    │ ~{m3['t_Ekman_mol']:6.0f} s (mol.)  │ 0                │
  │ 4. Displacement flow       │ ~{m4['t_acoustic']*1000:4.0f} ms acoustic │ 0                │
  │ 5. Pressure pulse          │ ~{m5['t_decel']*1000:4.0f} ms transient │ 0                │
  │ 6. Rim wake                │  confined to edge  │ 0                │
  └────────────────────────────┴────────────────────┴──────────────────┘""")

    # Total residual
    u_total = abs(m1['w_at_obs']) + abs(m2['v_at_obs'])

    if gap <= 0:
        print(f"""
  *** z_obs is INSIDE the boundary layer. ***
  The observation point has nonzero velocity from both tangential
  and axial components. Consider increasing z_obs or reducing ω.
""")
    elif u_total < 1e-6:
        print(f"""
  CONCLUSION: The gas at z = {z_obs*1000:.1f} mm above the disk is
  QUIESCENT at t = {t_obs} s after the disk stops.
""")
    else:
        print(f"""
  NOTE: Nonzero residual velocity detected.
  Axial (pumping):    {m1['w_at_obs']:.2e} m/s
  Tangential (diffn): {m2['v_at_obs']:.2e} m/s
  Combined magnitude: {u_total:.2e} m/s = {u_total*100:.4f} cm/s

  KEY DIFFERENCE FROM PROPELLER: The von Kármán pumping provides
  a direct axial velocity mechanism that the propeller lacks.
""")

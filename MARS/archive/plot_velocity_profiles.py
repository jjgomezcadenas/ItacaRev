#!/usr/bin/env python3
"""
Plot velocity profiles from propeller flow analysis JSON files.

Generates figures for inclusion in LaTeX document:
- velocity_profiles_comparison.pdf: N=2 vs N=8 velocity profiles
- velocity_profile_n2.pdf: Phase 1 velocity profile for N=2
- velocity_profile_n8.pdf: Phase 1 velocity profile for N=8

Usage:
    python plot_velocity_profiles.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style settings for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})


def load_results(n_arms: int) -> dict:
    """Load JSON results for given configuration."""
    path = Path(__file__).parent / f"propeller_results_n{n_arms}.json"
    with open(path) as f:
        return json.load(f)


def plot_single_profile(results: dict, output_file: Path):
    """Plot velocity profile for a single configuration."""
    n_arms = results["metadata"]["n_arms"]
    profile = results["phase1_velocity_profile"]
    z_mm = np.array(profile["z_mm"])
    u_cm_s = np.array(profile["u_cm_s"])

    # Key parameters for annotation
    delta_mm = results["phase1_boundary_layer"]["delta_mm"]
    max_reach_mm = results["phase1_eddy_diffusion"]["max_reach_mm"]
    z_obs_mm = results["fixed_parameters"]["z_obs_mm"]
    t_obs = results["fixed_parameters"]["t_obs_s"]
    V_tip = results["phase1_kinematics"]["V_tip_max_m_s"]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot velocity profile
    ax.plot(u_cm_s, z_mm, 'b-o', label='Velocity profile')

    # Mark key heights
    ax.axhline(delta_mm, color='green', linestyle='--', alpha=0.7,
               label=f'BL thickness δ = {delta_mm:.2f} mm')
    ax.axhline(max_reach_mm, color='orange', linestyle='--', alpha=0.7,
               label=f'Max reach = {max_reach_mm:.2f} mm')
    ax.axhline(z_obs_mm, color='red', linestyle='-', alpha=0.7,
               label=f'Observation z = {z_obs_mm:.1f} mm')

    ax.set_xlabel('Velocity u [cm/s]')
    ax.set_ylabel('Height z [mm]')
    ax.set_title(f'Velocity Profile at t = {t_obs} s after stop (N = {n_arms})')

    # Add text box with key parameters
    textstr = '\n'.join([
        f'V_tip = {V_tip:.2f} m/s',
        f'δ = {delta_mm:.2f} mm',
        f'max reach = {max_reach_mm:.2f} mm',
        f't_obs = {t_obs*1000:.0f} ms',
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, max(z_mm) + 1)

    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_comparison(results_n2: dict, results_n8: dict, output_file: Path):
    """Plot comparison of N=2 vs N=8 velocity profiles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, results, color, label in [
        (ax1, results_n2, 'blue', 'N = 2'),
        (ax2, results_n8, 'red', 'N = 8'),
    ]:
        n_arms = results["metadata"]["n_arms"]
        profile = results["phase1_velocity_profile"]
        z_mm = np.array(profile["z_mm"])
        u_cm_s = np.array(profile["u_cm_s"])

        delta_mm = results["phase1_boundary_layer"]["delta_mm"]
        max_reach_mm = results["phase1_eddy_diffusion"]["max_reach_mm"]
        z_obs_mm = results["fixed_parameters"]["z_obs_mm"]
        V_tip = results["phase1_kinematics"]["V_tip_max_m_s"]
        t_rot = results["phase1_kinematics"]["t_rot_ms"]

        # Plot
        ax.plot(u_cm_s, z_mm, f'{color[0]}-o', label=f'u(z) at t = 500 ms')
        ax.axhline(delta_mm, color='green', linestyle='--', alpha=0.7,
                   label=f'δ = {delta_mm:.2f} mm')
        ax.axhline(max_reach_mm, color='orange', linestyle='--', alpha=0.7,
                   label=f'max reach = {max_reach_mm:.2f} mm')
        ax.axhline(z_obs_mm, color='gray', linestyle='-', alpha=0.7,
                   label=f'z_obs = {z_obs_mm:.1f} mm')

        ax.set_xlabel('Velocity u [cm/s]')
        ax.set_title(f'{label}: V_tip = {V_tip:.1f} m/s, t_rot = {t_rot:.0f} ms')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(left=0)

    ax1.set_ylabel('Height z [mm]')
    ax1.set_ylim(0, 22)

    plt.suptitle('Residual Velocity Profiles: 2-Arm vs 8-Arm Propeller', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_summary_figure(results_n2: dict, results_n8: dict, output_file: Path):
    """Create a comprehensive summary figure with both profiles and key parameters."""
    fig = plt.figure(figsize=(14, 8))

    # Create grid: 2 rows, 3 columns
    # Row 1: velocity profiles (2 plots) + parameter table
    # Row 2: timeline diagram spanning full width

    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.25)

    # Velocity profiles
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax_table = fig.add_subplot(gs[0, 2])
    ax_timeline = fig.add_subplot(gs[1, :])

    for ax, results, color, label in [
        (ax1, results_n2, 'blue', 'N = 2'),
        (ax2, results_n8, 'red', 'N = 8'),
    ]:
        profile = results["phase1_velocity_profile"]
        z_mm = np.array(profile["z_mm"])
        u_cm_s = np.array(profile["u_cm_s"])

        delta_mm = results["phase1_boundary_layer"]["delta_mm"]
        max_reach_mm = results["phase1_eddy_diffusion"]["max_reach_mm"]
        z_obs_mm = results["fixed_parameters"]["z_obs_mm"]

        ax.plot(u_cm_s, z_mm, f'{color[0]}-o', markersize=4)
        ax.axhline(delta_mm, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(max_reach_mm, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(z_obs_mm, color='gray', linestyle='-', alpha=0.7, linewidth=1)

        ax.set_xlabel('u [cm/s]')
        ax.set_title(label)
        ax.set_xlim(left=0)

    ax1.set_ylabel('z [mm]')
    ax1.set_ylim(0, 15)
    plt.setp(ax2.get_yticklabels(), visible=False)

    # Parameter comparison table
    ax_table.axis('off')
    cell_text = [
        ['Parameter', 'N = 2', 'N = 8'],
        ['Sweep angle', '180°', '45°'],
        ['t_rot', f'{results_n2["phase1_kinematics"]["t_rot_ms"]:.0f} ms',
         f'{results_n8["phase1_kinematics"]["t_rot_ms"]:.0f} ms'],
        ['V_tip', f'{results_n2["phase1_kinematics"]["V_tip_max_m_s"]:.1f} m/s',
         f'{results_n8["phase1_kinematics"]["V_tip_max_m_s"]:.1f} m/s'],
        ['Ma_tip', f'{results_n2["phase1_flow_regime"]["Ma_tip"]:.3f}',
         f'{results_n8["phase1_flow_regime"]["Ma_tip"]:.3f}'],
        ['Re_tip', f'{results_n2["phase1_flow_regime"]["Re_tip"]:.1e}',
         f'{results_n8["phase1_flow_regime"]["Re_tip"]:.1e}'],
        ['δ', f'{results_n2["phase1_boundary_layer"]["delta_mm"]:.2f} mm',
         f'{results_n8["phase1_boundary_layer"]["delta_mm"]:.2f} mm'],
        ['max reach', f'{results_n2["phase1_eddy_diffusion"]["max_reach_mm"]:.2f} mm',
         f'{results_n8["phase1_eddy_diffusion"]["max_reach_mm"]:.2f} mm'],
        ['u(z_obs)', f'{results_n2["phase1_eddy_diffusion"]["u_at_obs_m_s"]:.1e} m/s',
         f'{results_n8["phase1_eddy_diffusion"]["u_at_obs_m_s"]:.1e} m/s'],
        ['', '', ''],
        ['t_slide', f'{results_n2["phase2_kinematics"]["t_slide_ms"]:.0f} ms',
         f'{results_n8["phase2_kinematics"]["t_slide_ms"]:.0f} ms'],
        ['V_slide', f'{results_n2["phase2_kinematics"]["V_max_m_s"]:.1f} m/s',
         f'{results_n8["phase2_kinematics"]["V_max_m_s"]:.1f} m/s'],
    ]

    table = ax_table.table(
        cellText=cell_text,
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.325, 0.325],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Header row styling
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    # Timeline diagram
    ax_timeline.set_xlim(0, 700)
    ax_timeline.set_ylim(0, 2)
    ax_timeline.set_xlabel('Time [ms]')
    ax_timeline.set_yticks([0.5, 1.5])
    ax_timeline.set_yticklabels(['N = 8', 'N = 2'])

    # N=2 timeline
    t_rot_2 = results_n2["phase1_kinematics"]["t_rot_ms"]
    t_slide_2 = results_n2["phase2_kinematics"]["t_slide_ms"]
    ax_timeline.barh(1.5, t_rot_2, left=0, height=0.4, color='blue', alpha=0.7, label='Phase 1: Blade rotation')
    ax_timeline.barh(1.5, t_slide_2, left=t_rot_2, height=0.4, color='green', alpha=0.7, label='Phase 2: Carrier slide')
    ax_timeline.axvline(t_rot_2 + t_slide_2 + 500, color='red', linestyle='--', alpha=0.5)

    # N=8 timeline
    t_rot_8 = results_n8["phase1_kinematics"]["t_rot_ms"]
    t_slide_8 = results_n8["phase2_kinematics"]["t_slide_ms"]
    ax_timeline.barh(0.5, t_rot_8, left=0, height=0.4, color='blue', alpha=0.7)
    ax_timeline.barh(0.5, t_slide_8, left=t_rot_8, height=0.4, color='green', alpha=0.7)
    ax_timeline.axvline(t_rot_8 + t_slide_8 + 500, color='red', linestyle='--', alpha=0.5)

    ax_timeline.legend(loc='upper right')
    ax_timeline.set_title('Motion Timeline (red dashed = observation at t_obs = 500 ms after Phase 2)')

    plt.suptitle('Propeller Flow Analysis: Residual Velocity in Dense Xenon', fontsize=14, y=0.98)
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_phase3_profiles(results: dict, output_file: Path):
    """Plot Phase 3 velocity profiles (blade BL, carrier slide BL, and Stokes layer)."""
    from propeller_flow_numerics import (
        compute_velocity_profile_blade,
        compute_velocity_profile_slide,
        compute_velocity_profile_lift,
        compute_stokes_layer_thickness,
        compute_gas_properties,
        compute_boundary_layer,
        compute_carrier_boundary_layer,
    )

    n_arms = results["metadata"]["n_arms"]
    V_tip = results["phase1_kinematics"]["V_tip_max_m_s"]
    delta = results["phase1_boundary_layer"]["delta_m"]
    delta_star = results["phase1_boundary_layer"]["delta_star_m"]
    t_slide = results["phase2_kinematics"]["t_slide_s"]
    V_slide = results["phase2_kinematics"]["V_max_m_s"]

    # Phase 3 parameters from results
    z_lift = results["phase3_parameters"]["z_lift_m"]
    t_lift = results["phase3_parameters"]["t_lift_s"]
    t_wait = results["phase3_parameters"]["t_wait_s"]
    u_c = results["phase3_parameters"]["u_c_m_s"]
    z_max = results["phase3_parameters"]["z_max_m"]

    V_lift = results["phase3_kinematics"]["V_lift_max_m_s"]

    # Gas properties
    _, nu, _ = compute_gas_properties()

    # Carrier BL parameters
    bl_carrier = compute_carrier_boundary_layer(V_slide, nu)
    delta_c = bl_carrier["delta"]
    delta_star_c = bl_carrier["delta_star"]

    # Compute profiles
    t_eval_profile1 = t_slide + t_lift + t_wait
    profile1 = compute_velocity_profile_blade(V_tip, delta, delta_star, nu, t_eval_profile1, z_max, u_c)

    t_eval_profile2 = t_lift + t_wait
    profile2 = compute_velocity_profile_slide(V_slide, delta_c, delta_star_c, nu, t_eval_profile2, z_max, u_c)

    delta_stokes = compute_stokes_layer_thickness(nu, t_lift)
    profile3 = compute_velocity_profile_lift(V_lift, delta_stokes, nu, t_wait, z_lift, z_max, u_c)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Units: all inputs from profile functions are in SI (m, m/s)
    # Convert to mm and mm/s for plotting
    u_c_mm_s = u_c * 1000  # m/s -> mm/s
    z_max_mm = z_max * 1000  # m -> mm

    # Profile 1: Blade BL diffusion
    z1_m = profile1["z_array"]
    u1_m_s = profile1["u_array"]
    z_q1_m = profile1["z_q1"]
    z_q1_mm = z_q1_m * 1000
    u1_max = np.max(u1_m_s)

    # Check if profile is essentially zero (fully decayed)
    if u1_max < u_c:
        # Profile has fully decayed - show text instead of meaningless plot
        ax1.text(0.5, 0.5, 'Profile fully decayed\n(u < u_c everywhere)',
                 transform=ax1.transAxes, ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 1)
    else:
        # Truncate at quiescence height
        if z_q1_m < z_max:
            mask1 = z1_m <= z_q1_m
        else:
            mask1 = u1_m_s > u_c * 0.01  # plot down to 1% of cutoff

        z1_mm = z1_m[mask1] * 1000
        u1_mm_s = u1_m_s[mask1] * 1000

        ax1.plot(z1_mm, u1_mm_s, 'b-', linewidth=2, label='Velocity profile')
        ax1.axhline(u_c_mm_s, color='red', linestyle='--', alpha=0.7,
                    label=f'u_c = {u_c_mm_s:.1f} mm/s')
        if z_q1_m < z_max:
            ax1.axvline(z_q1_mm, color='green', linestyle='--', alpha=0.7,
                        label=f'z_q1 = {z_q1_mm:.2f} mm')
        ax1.axvline(delta * 1000, color='orange', linestyle=':', alpha=0.7,
                    label=f'δ₁ = {delta*1000:.2f} mm')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_xlim(0, min(z_q1_mm + 1, 10) if z_q1_m < z_max else 5)
        ax1.set_ylim(0, max(u1_mm_s) * 1.1 if len(u1_mm_s) > 0 else 1)

    ax1.set_xlabel('Height z [mm]')
    ax1.set_ylabel('Velocity u [mm/s]')
    ax1.set_title(f'Profile 1: Blade BL\nt = {t_eval_profile1:.2f} s')
    ax1.grid(True, alpha=0.3)

    # Profile 2: Carrier slide BL diffusion
    z2_m = profile2["z_array"]
    u2_m_s = profile2["u_array"]
    z_q2_m = profile2["z_q2"]

    # Truncate at quiescence height
    if z_q2_m < z_max:
        mask2 = z2_m <= z_q2_m
    else:
        mask2 = u2_m_s > 1e-12

    z2_mm = z2_m[mask2] * 1000
    u2_mm_s = u2_m_s[mask2] * 1000
    z_q2_mm = z_q2_m * 1000

    ax2.plot(z2_mm, u2_mm_s, 'g-', linewidth=2, label='Velocity profile')
    ax2.axhline(u_c_mm_s, color='red', linestyle='--', alpha=0.7,
                label=f'u_c = {u_c_mm_s:.1f} mm/s')
    if z_q2_m < z_max:
        ax2.axvline(z_q2_mm, color='green', linestyle='--', alpha=0.7,
                    label=f'z_q2 = {z_q2_mm:.2f} mm')
    ax2.axvline(delta_c * 1000, color='orange', linestyle=':', alpha=0.7,
                label=f'δ₂ = {delta_c*1000:.2f} mm')

    ax2.set_xlabel('Height z [mm]')
    ax2.set_ylabel('Velocity u [mm/s]')
    ax2.set_title(f'Profile 2: Carrier Slide BL\nt = {t_eval_profile2:.2f} s')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0, min(z_q2_mm + 1, 10) if z_q2_m < z_max else 8)
    ax2.set_ylim(0, max(u2_mm_s) * 1.1 if len(u2_mm_s) > 0 and max(u2_mm_s) > 0 else 1)
    ax2.grid(True, alpha=0.3)

    # Profile 3: Stokes layer from lift
    z3_m = profile3["z_array"]
    u3_m_s = profile3["u_array"]
    z_q3_m = profile3["z_q3"]
    z_lift_mm = z_lift * 1000

    # Truncate at quiescence height
    if z_q3_m < z_max:
        mask3 = z3_m <= z_q3_m
    else:
        mask3 = u3_m_s > 1e-12

    z3_mm = z3_m[mask3] * 1000
    u3_mm_s = u3_m_s[mask3] * 1000
    z_q3_mm = z_q3_m * 1000

    ax3.plot(z3_mm, u3_mm_s, 'r-', linewidth=2, label='Velocity profile')
    ax3.axhline(u_c_mm_s, color='red', linestyle='--', alpha=0.7,
                label=f'u_c = {u_c_mm_s:.1f} mm/s')
    if z_q3_m < z_max:
        ax3.axvline(z_q3_mm, color='green', linestyle='--', alpha=0.7,
                    label=f'z_q3 = {z_q3_mm:.2f} mm')
    ax3.axvline(z_lift_mm, color='blue', linestyle=':', alpha=0.7,
                label=f'z_lift = {z_lift_mm:.1f} mm')
    ax3.axvline(z_lift_mm + delta_stokes * 1000, color='orange', linestyle=':', alpha=0.7,
                label=f'δ₃ = {delta_stokes*1000:.3f} mm')

    ax3.set_xlabel('Height z [mm] (RF1)')
    ax3.set_ylabel('Velocity u [mm/s]')
    ax3.set_title(f'Profile 3: Stokes Layer\nt_wait = {t_wait:.1f} s')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(z_lift_mm - 0.5, min(z_q3_mm + 1, 15) if z_q3_m < z_max else 15)
    ax3.set_ylim(0, max(u3_mm_s) * 1.1 if len(u3_mm_s) > 0 and max(u3_mm_s) > 0 else 1)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f'Phase 3 Velocity Profiles (N = {n_arms})', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_phase3_comparison(results_n2: dict, results_n8: dict, output_file: Path):
    """Plot comparison of Phase 3 quiescence heights for N=2 vs N=8."""
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = ['N = 2', 'N = 8']
    x = np.arange(len(labels))
    width = 0.25

    z_q1_n2 = results_n2["phase3_profile1"]["z_q1_mm"]
    z_q1_n8 = results_n8["phase3_profile1"]["z_q1_mm"]
    z_q2_n2 = results_n2["phase3_profile2"]["z_q2_mm"]
    z_q2_n8 = results_n8["phase3_profile2"]["z_q2_mm"]
    z_q3_n2 = results_n2["phase3_profile3"]["z_q3_mm"]
    z_q3_n8 = results_n8["phase3_profile3"]["z_q3_mm"]

    z_q1_vals = [z_q1_n2, z_q1_n8]
    z_q2_vals = [z_q2_n2, z_q2_n8]
    z_q3_vals = [z_q3_n2, z_q3_n8]

    rects1 = ax.bar(x - width, z_q1_vals, width, label='z_q1 (Blade BL)', color='blue', alpha=0.7)
    rects2 = ax.bar(x, z_q2_vals, width, label='z_q2 (Carrier Slide BL)', color='green', alpha=0.7)
    rects3 = ax.bar(x + width, z_q3_vals, width, label='z_q3 (Stokes layer)', color='red', alpha=0.7)

    ax.set_ylabel('Quiescence height [mm]')
    ax.set_title('Phase 3: Quiescence Heights Above Lifted Plate')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Add horizontal line for z_lift
    z_lift_mm = results_n2["phase3_parameters"]["z_lift_mm"]
    ax.axhline(z_lift_mm, color='green', linestyle='--', alpha=0.5,
               label=f'z_lift = {z_lift_mm:.1f} mm')

    # Add text with timing info
    t_total_n2 = results_n2["summary"]["t_total_s"]
    t_total_n8 = results_n8["summary"]["t_total_s"]
    u_c_mm_s = results_n2["summary"]["u_c_mm_s"]

    textstr = '\n'.join([
        f'Total cycle time: {t_total_n2:.2f} s',
        f'Quiescence cutoff: {u_c_mm_s:.1f} mm/s',
        f'z_lift = {z_lift_mm:.1f} mm',
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_full_timeline(results_n2: dict, results_n8: dict, output_file: Path):
    """Plot full 3-phase timeline for both configurations."""
    fig, ax = plt.subplots(figsize=(12, 5))

    t_rot_2 = results_n2["phase1_kinematics"]["t_rot_ms"]
    t_slide_2 = results_n2["phase2_kinematics"]["t_slide_ms"]
    t_lift = results_n2["phase3_kinematics"]["t_lift_s"] * 1000  # ms
    t_wait = results_n2["phase3_kinematics"]["t_wait_s"] * 1000  # ms

    t_rot_8 = results_n8["phase1_kinematics"]["t_rot_ms"]
    t_slide_8 = results_n8["phase2_kinematics"]["t_slide_ms"]

    y_n2 = 1.5
    y_n8 = 0.5

    # N=2 timeline
    ax.barh(y_n2, t_rot_2, left=0, height=0.4, color='blue', alpha=0.7, label='Phase 1: Rotation')
    ax.barh(y_n2, t_slide_2, left=t_rot_2, height=0.4, color='green', alpha=0.7, label='Phase 2: Slide')
    ax.barh(y_n2, t_lift, left=t_rot_2 + t_slide_2, height=0.4, color='orange', alpha=0.7, label='Phase 3: Lift')
    ax.barh(y_n2, t_wait, left=t_rot_2 + t_slide_2 + t_lift, height=0.4, color='gray', alpha=0.7, label='Wait')

    # N=8 timeline
    ax.barh(y_n8, t_rot_8, left=0, height=0.4, color='blue', alpha=0.7)
    ax.barh(y_n8, t_slide_8, left=t_rot_8, height=0.4, color='green', alpha=0.7)
    ax.barh(y_n8, t_lift, left=t_rot_8 + t_slide_8, height=0.4, color='orange', alpha=0.7)
    ax.barh(y_n8, t_wait, left=t_rot_8 + t_slide_8 + t_lift, height=0.4, color='gray', alpha=0.7)

    # Total cycle times
    t_total_n2 = t_rot_2 + t_slide_2 + t_lift + t_wait
    t_total_n8 = t_rot_8 + t_slide_8 + t_lift + t_wait

    ax.axvline(t_total_n2, color='red', linestyle='--', alpha=0.7, label=f'Cycle end ({t_total_n2:.0f} ms)')

    ax.set_xlim(0, max(t_total_n2, t_total_n8) + 100)
    ax.set_ylim(0, 2)
    ax.set_xlabel('Time [ms]')
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['N = 8', 'N = 2'])
    ax.set_title('Full Motion Sequence: Rotation → Slide → Lift → Wait')

    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def main():
    """Generate all plots."""
    output_dir = Path(__file__).parent

    # Load results
    results_n2 = load_results(2)
    results_n8 = load_results(8)

    # Phase 1 & 2 plots (existing)
    plot_single_profile(results_n2, output_dir / "velocity_profile_n2.pdf")
    plot_single_profile(results_n8, output_dir / "velocity_profile_n8.pdf")
    plot_comparison(results_n2, results_n8, output_dir / "velocity_profiles_comparison.pdf")
    plot_summary_figure(results_n2, results_n8, output_dir / "propeller_analysis_summary.pdf")

    # Phase 3 plots (new)
    plot_phase3_profiles(results_n2, output_dir / "phase3_profiles_n2.pdf")
    plot_phase3_profiles(results_n8, output_dir / "phase3_profiles_n8.pdf")
    plot_phase3_comparison(results_n2, results_n8, output_dir / "phase3_quiescence_comparison.pdf")
    plot_full_timeline(results_n2, results_n8, output_dir / "full_timeline.pdf")

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()

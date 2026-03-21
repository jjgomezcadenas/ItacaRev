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


def main():
    """Generate all plots."""
    output_dir = Path(__file__).parent

    # Load results
    results_n2 = load_results(2)
    results_n8 = load_results(8)

    # Generate plots
    plot_single_profile(results_n2, output_dir / "velocity_profile_n2.pdf")
    plot_single_profile(results_n8, output_dir / "velocity_profile_n8.pdf")
    plot_comparison(results_n2, results_n8, output_dir / "velocity_profiles_comparison.pdf")
    plot_summary_figure(results_n2, results_n8, output_dir / "propeller_analysis_summary.pdf")

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()

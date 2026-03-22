# Blade Specification

## Overview

The blade is a radial arm of the propeller assembly that sweeps through the xenon gas to position carrier plates. The design uses an airfoil profile to minimize flow disturbance and drag.

## Geometry

### Overall Dimensions
| Parameter | Symbol | Value |
|-----------|--------|-------|
| Length (radius) | R | 1.6 m |
| Chord | c | 160 mm |
| Thickness | t | 19.2 mm (12% of chord) |
| Profile | — | NACA 0012 |
| Angle of attack | α | 0° |

### NACA 0012 Profile
The NACA 0012 is a symmetric airfoil with:
- Maximum thickness: 12% of chord at 30% chord position
- Zero camber (symmetric about chord line)
- Smooth pressure distribution at α = 0°

Thickness distribution:
```
y/c = 0.6 × (0.2969√(x/c) - 0.1260(x/c) - 0.3516(x/c)² + 0.2843(x/c)³ - 0.1015(x/c)⁴)
```

### Key Dimensions
| Location | x/c | Thickness (mm) |
|----------|-----|----------------|
| Leading edge | 0% | 0 (rounded) |
| 10% chord | 10% | 11.0 |
| 30% chord | 30% | 19.2 (max) |
| 50% chord | 50% | 17.8 |
| 70% chord | 70% | 13.4 |
| Trailing edge | 100% | 0 (sharp) |

## Material and Mass

| Parameter | Value |
|-----------|-------|
| Material | TBD (aluminum, carbon fiber, or titanium) |
| Mass per blade | m_blade = 0.25 kg |
| Construction | Hollow or foam-core for low mass |

### Moment of Inertia (per blade)
Treating blade as uniform rod rotating about one end:
```
I_blade = (1/3) × m_blade × R²
        = (1/3) × 0.25 × 1.6²
        = 0.213 kg·m²
```

## Configuration Options

### Number of Arms
| Config | n_arms | Sweep angle θ | I_total (blades) | I_total (+ plates) |
|--------|--------|---------------|------------------|---------------------|
| 2-arm | 2 | π (180°) | 0.427 kg·m² | 0.747 kg·m² |
| 4-arm | 4 | π/2 (90°) | 0.853 kg·m² | 1.493 kg·m² |
| 8-arm | 8 | π/4 (45°) | 1.707 kg·m² | 2.987 kg·m² |

**Note**: Sweep angle θ = 2π/n_arms ensures full coverage. Plate inertia uses I_plate = m_plate × r₀² with r₀ = R/2 = 0.8 m.

## Kinematics (Bang-Bang Profile)

### Motion Equations
For torque τ and total inertia I:
```
Angular acceleration:  α = τ/I
Switching angle:       θ_switch = θ/2
Peak angular velocity: ω_max = √(θ·τ/I)
Total rotation time:   t_rot = 2√(θ·I/τ)
Peak tip velocity:     V_tip = ω_max × R
```

### Results (τ = 60 N·m, with 250 g plates at r₀ = 0.8 m)

| Config | t_rot (ms) | ω_max (rad/s) | V_tip (m/s) |
|--------|------------|---------------|-------------|
| 2-arm | 396 | 15.9 | 25.4 |
| 8-arm | 396 | 4.0 | 6.4 |

## Aerodynamic Properties

### Flow Regime (at peak velocity)
| Parameter | 2-arm | 8-arm |
|-----------|-------|-------|
| Tip Mach number | 0.143 | 0.036 |
| Tip Reynolds number | 1.38×10⁷ | 3.46×10⁶ |
| Flow regime | Fully turbulent | Fully turbulent |

### Drag Model
For NACA 0012 at α = 0° in dense xenon:

**Wake drag**:
```
τ_drag = n_arms × (ρ C_d d_wake R⁴ / 4) ω²
```
where:
- C_d ≈ 0.01 (NACA 0012 at α = 0°)
- d_wake = 0.12 × c = 19.2 mm (wake thickness)
- ρ ≈ 80 kg/m³ (xenon at 15 bar, 300 K)

**Impact**: Drag increases rotation time by ~5–10% and reduces ω_max by ~3–5%.

### Boundary Layer
At peak velocity (2-arm case, V_tip ≈ 25.4 m/s):
| Parameter | Value |
|-----------|-------|
| BL thickness δ | ~2.2 mm |
| Displacement thickness δ* | ~0.28 mm |
| Momentum thickness θ | ~0.21 mm |
| Friction velocity u_τ | ~0.84 m/s |
| Wall shear stress τ_w | ~56 Pa |

## Vessel Constraints

| Parameter | Value |
|-----------|-------|
| Vessel inner diameter | 2R + ΔD = 3.21 m |
| Vessel height | H = 1.5 m |
| Clearance (radial) | 5 mm |
| Gas | Xenon at P = 15 bar, T = 300 K |

## Flow Mechanisms After Stop

The blade rotation induces several flow mechanisms that persist after the blade stops:

1. **Turbulent eddy diffusion**: BL momentum diffuses upward
2. **Bulk swirl**: Angular momentum transferred to gas
3. **Displacement flow**: Potential flow from blade sweep
4. **Pressure pulse**: From deceleration phase
5. **Secondary flows**: Corner vortices at blade-carrier junction
6. **Vortex shedding**: From trailing edge (minimal at α = 0°)

### Residual Velocity Summary
At z = 5 mm above blade surface, t = 500 ms after stop:
| Config | u_residual |
|--------|------------|
| 2-arm | ~3×10⁻⁷ m/s |
| 8-arm | ~3×10⁻⁵ m/s |

## Carrier Plate Interface

Each blade carries one carrier plate:
- Initial position: r₀ = R/2 = 0.8 m from center
- Plate slides radially along blade after rotation completes
- Interface: Linear rail or bearing
- Mass contribution to inertia: I_plate = m_plate × r₀² (point mass at r₀)

## Summary Table

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Length | R | 1.6 m |
| Chord | c | 160 mm |
| Profile | — | NACA 0012 |
| Thickness | t | 19.2 mm |
| Mass | m_blade | 0.25 kg |
| I per blade | I_blade | 0.213 kg·m² |
| Angle of attack | α | 0° |
| Motor torque | τ | 60 N·m |
| Configurations | n_arms | 2, 4, 6, or 8 |

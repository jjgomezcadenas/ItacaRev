# Carrier Plate Specification

## Overview

The carrier plate is a rigid platform that holds the ion detection/collection apparatus. It is mounted on a propeller blade arm and executes two sequential motions:
1. **Tangential rotation** (with the blade)
2. **Radial translation** (along the arm, after blade stops)

## Geometry

### Central Body
| Parameter | Value | Notes |
|-----------|-------|-------|
| Shape | Rectangular box | Hollow thin-wall construction |
| Surface area | 160 × 160 mm² | |
| Height | 5 mm | |
| Wall thickness | 0.8–1.0 mm | Ti-6Al-4V |

### Edge Profiles
All four edges feature **symmetric sharp leading edges** (wedge profile):
- Half-angle: ~10°
- Provides low drag coefficient C_d ≈ 0.08–0.1

**Note**: A NACA profile was considered for tangential edges but rejected because:
1. The 5 mm height is insufficient for a proper airfoil cross-section
2. Marginal drag improvement (~0.3 mN difference)
3. Symmetric edges simplify manufacturing

### Total Envelope
Including edge profiles, the carrier footprint is approximately:
- Length: ~170 mm (160 mm body + edge fairings)
- Width: ~170 mm
- Height: 5 mm

## Material

| Property | Value |
|----------|-------|
| Material | Ti-6Al-4V (Grade 5 Titanium) |
| Density | 4430 kg/m³ |
| Yield strength | 880 MPa |
| Young's modulus | 114 GPa |

## Mass Budget

| Component | Mass |
|-----------|------|
| Central box (1 mm walls) | ~240 g |
| Edge fairings | ~10 g |
| **Total (design value)** | **250 g** |

This is a conservative worst-case value. Optimization could reduce to ~150–200 g.

## Initial Position

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Radial position | r₀ = R/2 | Simplified positioning |
| For R = 1.6 m | r₀ = 0.8 m | |

This position is at the midpoint of the blade span.

## Motion Profile

### Sequence
1. **Blade rotation completes** (bang-bang profile)
2. **Plate slides radially** (separate actuator, bang-bang profile)

The motions are **sequential, not simultaneous**, to avoid:
- Coriolis torque coupling (~17% of motor torque if simultaneous)
- Centrifugal acceleration (~180 m/s² at ω_max)
- Time-varying moment of inertia (~100% change in plate contribution)

### Radial Slide Parameters
| Parameter | Value |
|-----------|-------|
| Maximum travel | 0.8 m (r₀ → 0) or 0.8 m (r₀ → R) |
| Worst case | Full travel from r₀ = 0.8 m to r = 0 |
| Actuator force | F = 20 N (linear actuator) |
| Slide time | t_slide ≈ 200 ms |
| Peak velocity | V_max ≈ 8 m/s |
| Motion profile | Bang-bang (max accel/decel) |

## Structural Analysis

### Loading Conditions

**During blade rotation** (carrier at r₀ = 0.8 m):
| Parameter | Value |
|-----------|-------|
| Angular acceleration | α ≈ 20.1 rad/s² (8-arm case) |
| Tangential acceleration | a_t = α × r₀ ≈ 16 m/s² |
| Tangential force | F_t = m × a_t ≈ 4.0 N |

**During radial slide** (blade stationary):
| Parameter | Value |
|-----------|-------|
| Linear acceleration | a_r ~ 50–100 m/s² (TBD by actuator) |
| Radial force | F_r = m × a_r ≈ 25 N |

### Stress and Deflection

For 1 mm wall thickness:
| Parameter | Value | Limit |
|-----------|-------|-------|
| Maximum bending stress | ~36 MPa | 880 MPa (yield) |
| Safety factor | 24× | >2× required |
| Maximum deflection | ~3 μm | Negligible |

## Aerodynamic Drag

### During Blade Rotation
| Parameter | Value |
|-----------|-------|
| Frontal area | A = 5 mm × 160 mm = 8×10⁻⁴ m² |
| Local velocity (at r₀ = 0.8 m, ω_max ≈ 4.0 rad/s, 8-arm) | V ≈ 3.2 m/s |
| Drag coefficient | C_d ≈ 0.1 (worst case) |
| Drag force | F_d ≈ 32 mN |

### During Radial Slide
| Parameter | Value |
|-----------|-------|
| Frontal area | A = 5 mm × 160 mm = 8×10⁻⁴ m² |
| Slide velocity | V_slide ~ 0.5–1 m/s (TBD) |
| Drag coefficient | C_d ≈ 0.1 |
| Drag force | F_d ~ 1–2 mN |

**Conclusion**: Carrier plate drag is negligible compared to blade drag.

## Interface Requirements

1. **Blade arm interface**: Sliding rail or linear bearing for radial motion
2. **Actuator connection**: Linear actuator (electric or pneumatic)
3. **Electrical feedthrough**: For ion detection instrumentation
4. **Positioning accuracy**: TBD based on ion collection requirements

## Summary

| Parameter | Value |
|-----------|-------|
| Dimensions | 160 × 160 × 5 mm³ (body) |
| Mass | 250 g (conservative) |
| Material | Ti-6Al-4V, 1 mm walls |
| Edge profile | Symmetric sharp wedge (all sides) |
| Initial position | r₀ = R/2 = 0.8 m |
| Actuator force | F = 20 N |
| Slide time | t_slide ≈ 200 ms |
| Peak velocity | V_max ≈ 8 m/s |
| Motion | Sequential: blade rotation, then radial slide |
| Drag coefficient | C_d ≈ 0.1 (worst case) |

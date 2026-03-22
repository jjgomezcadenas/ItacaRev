# ITACA Detector and MARS System Specification

## 1. Overview

**ITACA** (Ion Tracking And Collection Apparatus) is a detector for measuring ionization tracks in dense xenon gas. **MARS** (Magnetic Actuated Rotor System) is the mechanical subsystem that positions ion collection plates under ionization tracks.

## 2. Detector Geometry

### 2.1 Chamber

| Parameter | Value | Notes |
|-----------|-------|-------|
| Shape | Vertical cylinder | |
| Diameter | 3.2 m | |
| Height | H = 1.5 m | Drift length |
| Gas | Xenon | |
| Pressure | P = 15 bar | |
| Temperature | T = 300 K | |

### 2.2 Coordinate System

- **z-axis**: Vertical (along cylinder axis)
- **z = H** (1.5 m): Anode (top)
- **z = 0**: Ion plate surface (after lift)
- **x-y plane**: Horizontal (MARS operates here)
- **r, θ**: Cylindrical coordinates in x-y plane

### 2.3 Electrode Structure (from top to bottom)

```
z = H (1.5 m)   ─────────────  Anode
                     │
                  Drift region (ions produced here)
                     │
z ~ 10 mm       ─────────────  Cathode grid
                     │
z ~ 5 mm        ═════════════  Ion plate (after lift)
                     │
z = 0           ═════════════  MARS blade/carrier surface
```

### 2.4 Cathode Grid

| Parameter | Value |
|-----------|-------|
| Type | Wire mesh |
| Wire diameter | w = 200 µm |
| Pitch | M = 5 mm |
| Geometrical transparency | f = (1 - w/M)² ≈ 92% |
| Voltage | 500 V |

## 3. Xenon Gas Properties

Dense xenon at 15 bar, 300 K has unusual properties:

| Property | Symbol | Value | Comparison to air |
|----------|--------|-------|-------------------|
| Density | ρ | 79 kg/m³ | 66× denser |
| Dynamic viscosity | μ | 2.32×10⁻⁵ Pa·s | Similar |
| Kinematic viscosity | ν | 2.94×10⁻⁷ m²/s | 51× smaller |
| Speed of sound | c_s | 178 m/s | |
| Heat capacity ratio | γ | 5/3 | Monatomic |

**Key consequence:** The very low kinematic viscosity means:
- High Reynolds numbers at modest velocities
- Very slow molecular diffusion
- Long viscous timescales

## 4. Ion Physics

### 4.1 Ionization Event

An ionization event in the chamber produces electron-ion pairs along a track.

### 4.2 Electron Drift

| Parameter | Value |
|-----------|-------|
| Drift velocity | ~1500 m/s |
| Drift time (H = 1.5 m) | ~1 ms |
| Readout | Anode provides (r, θ) position |

Electrons reach the anode almost instantly and trigger MARS.

### 4.3 Ion Drift

| Parameter | Value |
|-----------|-------|
| Drift velocity | v_d = 100 mm/s = 0.1 m/s |
| Drift direction | -z (downward toward cathode) |
| Minimum drift time | ~1-2 s (ions near cathode) |
| Maximum drift time | ~15 s (ions near anode) |
| Diffusion (RMS at cathode) | σ_diff ≈ 1 mm |

### 4.4 Ion Collection

Ions pass through the cathode grid (92% transparent) and are collected on the ion plate. The electric field between cathode and ion plate accelerates ions to velocity ≥ v_d.

## 5. MARS System

### 5.1 Purpose

Position the ion plate directly under the ionization track location (r, θ) provided by the anode readout at t = 0.

### 5.2 Components

1. **Propeller blades**: N NACA 0012 airfoils rotating about central axis
2. **Carrier plates**: Ion collection plates mounted on blades
3. **Linear actuators**: Slide carriers radially along blades
4. **Lift mechanism**: Raise carrier plate vertically toward cathode

### 5.3 Blade Specification

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Length (radius) | R | 1.6 m |
| Chord | c | 160 mm |
| Profile | — | NACA 0012 |
| Thickness | t | 19.2 mm (12% of chord) |
| Angle of attack | α | 0° |
| Mass per blade | m_blade | 0.25 kg |
| Motor torque | τ | 60 N·m |

**Configurations:**

| n_arms | Sweep angle | V_tip (peak) | Re_tip |
|--------|-------------|--------------|--------|
| 2 | 180° | 25.4 m/s | 1.4×10⁷ |
| 8 | 45° | 6.4 m/s | 3.5×10⁶ |

### 5.4 Carrier Plate Specification

| Parameter | Value |
|-----------|-------|
| Dimensions | 160 × 160 × 5 mm³ |
| Mass | m_plate = 0.25 kg |
| Material | Ti-6Al-4V |
| Edge profile | Symmetric sharp wedge |
| Initial position | r₀ = R/2 = 0.8 m |
| Drag coefficient | C_d ≈ 0.1 |

### 5.5 Three-Phase Motion Sequence

All motions are **sequential** (not simultaneous) and triggered at t = 0 by anode readout.

#### Phase 1: Blade Rotation
| Parameter | Value |
|-----------|-------|
| Motion | Tangential (θ direction) |
| Profile | Bang-bang (max accel/decel) |
| Duration | t_rot ≈ 400 ms |
| Peak velocity (n=2) | V_tip = 25.4 m/s |
| Peak velocity (n=8) | V_tip = 6.4 m/s |

#### Phase 2: Carrier Slide
| Parameter | Value |
|-----------|-------|
| Motion | Radial (r direction) |
| Travel | Δr = 0.8 m |
| Duration | t_slide = 200 ms |
| Peak velocity | V_slide = 8 m/s |
| Actuator force | F = 20 N |

#### Phase 3: Vertical Lift
| Parameter | Value |
|-----------|-------|
| Motion | Vertical (+z direction) |
| Lift height | z_lift = 5 mm |
| Duration | t_lift = 1.0 s |
| Peak velocity | V_lift = 10 mm/s |
| Profile | Smooth (half-sine or trapezoidal) |

#### Wait Period
| Parameter | Value |
|-----------|-------|
| Duration | t_wait = 0.5 s (adjustable) |
| Purpose | Allow Stokes layer to decay |

#### Total Cycle
| Parameter | Value |
|-----------|-------|
| Total time | t_total ≈ 2.1 s |
| Target | Reduce to ~1 s |

## 6. Gas Disturbance Analysis

### 6.1 Two Regions of Concern

#### Region 1: Above Cathode Grid (z > z_cathode)
- **What:** The drift region where ions are produced and drift
- **Concern:** Can MARS disturbances propagate upward and affect drifting ions?
- **Requirement:** Horizontal gas velocity u_h << v_d = 100 mm/s

#### Region 2: Below Cathode Grid (z < z_cathode)
- **What:** The gap between cathode and ion plate
- **Concern:** Do ions experience lateral displacement when crossing this region?
- **Requirement:** Lateral displacement Δr < σ_diff ≈ 1 mm

### 6.2 Velocity Directions

| Motion | BL velocity direction | Effect on ions |
|--------|----------------------|----------------|
| Blade rotation | Tangential (u_θ, horizontal) | Lateral displacement |
| Carrier slide | Radial (u_r, horizontal) | Lateral displacement |
| Vertical lift | Vertical (u_z) | Timing shift only |

### 6.3 Region 1 Analysis: Propagation Above Cathode

**Question:** Can MARS disturbances reach z > z_cathode (1.5 m above)?

**Molecular diffusion:**
```
L_diff = √(ν × t) = √(3×10⁻⁷ × 15) ≈ 2 mm
```
Molecular diffusion cannot transport momentum 1.5 m in 15 s.

**Mechanisms analyzed:**

| Mechanism | Timescale | Reaches drift region? |
|-----------|-----------|----------------------|
| Bulk swirl (tangential) | t_visc ~ 26 s | No - tangential only, no vertical transport |
| Pressure pulse | t_acoustic ~ 8 ms | Transient, passes through |
| Potential flow | t_acoustic ~ 9 ms | Vanishes when motion stops |
| Eddy diffusion | t_decay ~ 3-18 ms | Confined to δ ~ 3 mm of surface |

**Conclusion:** No mechanism transports significant horizontal momentum from MARS to the drift region. Region 1 is safe.

### 6.4 Region 2 Analysis: Ion Plate to Cathode

**Geometry after lift:**
- Ion plate surface at z = z_lift = 5 mm
- Blade/slide BLs were at z = 0 (now 5 mm below plate)
- Cathode grid at z ~ 10 mm

**Three velocity profiles evaluated:**

#### Profile 1: Blade BL (horizontal, tangential)
| Parameter | n=2 | n=8 |
|-----------|-----|-----|
| Evaluation time | 1.7 s | 1.7 s |
| BL thickness δ₁ | 2.21 mm | 2.91 mm |
| Decay time | 3.5 ms | 18 ms |
| Quiescence height z_q1 | 0.1 mm | 2.9 mm |

**Status:** Fully decayed. Located at z = 0, which is 5 mm below ion plate surface. Ions don't traverse this layer.

#### Profile 2: Carrier Slide BL (horizontal, radial)
| Parameter | Value |
|-----------|-------|
| Evaluation time | 1.5 s |
| BL thickness δ₂ | 2.78 mm |
| Decay time | 14 ms |
| Quiescence height z_q2 | 2.8 mm |

**Status:** Fully decayed. Located at z = 0, which is 5 mm below ion plate surface. Ions don't traverse this layer.

#### Profile 3: Lift Stokes Layer (vertical)
| Parameter | Value |
|-----------|-------|
| Evaluation time | t_wait = 0.5 s |
| Stokes layer δ₃ | 0.54 mm |
| Quiescence height z_q3 | 7.5 mm (in RF1) |
| Height above plate | 2.5 mm |

**Status:** This is a **vertical** velocity (u_z). It does NOT cause lateral displacement - only a small timing shift:
- Transit time through layer: Δt ~ 25 ms
- Timing shift: ~ 1 ms (negligible on 1-15 s drift)

**Conclusion:** Region 2 is safe. Horizontal BLs are below the ion plate and decayed. The lift Stokes layer is vertical and doesn't displace ions laterally.

### 6.5 Summary Table

| Mechanism | Direction | Location | Status | Ion displacement |
|-----------|-----------|----------|--------|-----------------|
| Blade BL | Horizontal | z = 0 | Decayed | 0 |
| Slide BL | Horizontal | z = 0 | Decayed | 0 |
| Lift Stokes | Vertical | z = 5-7.5 mm | Active but vertical | 0 (timing only) |
| Bulk swirl | Horizontal | z < 3 mm | Can't reach ions | 0 |

## 7. t_wait Tradeoff

### 7.1 Dead Zone

Ions produced within distance d from cathode are lost because MARS is still moving:
```
d_dead = v_d × (t_MARS + t_wait)
```

| t_wait | Dead zone | Lost drift region |
|--------|-----------|-------------------|
| 0 s | 100 mm | 7% of chamber |
| 0.5 s | 150 mm | 10% of chamber |
| 1.0 s | 200 mm | 13% of chamber |

### 7.2 Optimization

Since the lift Stokes layer is vertical (no lateral displacement), t_wait can potentially be reduced to zero without affecting ion position reconstruction. The only effect is a ~1 ms timing shift.

**Recommendation:** t_wait can be minimized or eliminated. The limiting factor is the total MARS cycle time, not the Stokes layer decay.

## 8. Design Requirements Summary

### 8.1 Primary Requirement
Gas velocity perturbations from MARS must not displace ions by more than σ_diff ≈ 1 mm at the cathode.

### 8.2 Derived Requirements

1. **Horizontal BLs must decay** before ions arrive at ion plate
   - Achieved: t_decay ~ 3-18 ms << t_eval ~ 1.5 s

2. **Horizontal BLs must be below ion plate surface**
   - Achieved: BLs at z = 0, plate at z = 5 mm

3. **Vertical flows don't cause lateral displacement**
   - Confirmed: Lift creates u_z only

4. **No mechanism propagates to drift region**
   - Confirmed: Molecular diffusion reach ~ 2 mm << 1.5 m

### 8.3 Current Status

**All requirements satisfied** for both n = 2 and n = 8 configurations.

## 9. Files and Scripts

| File | Description |
|------|-------------|
| `propeller_flow_analysis.tex` | Full physics analysis with equations |
| `propeller_flow_numerics.py` | Parameterized numerical calculations |
| `propeller_analysis_export.py` | Generate JSON results |
| `plot_velocity_profiles.py` | Generate plots |
| `propeller_results_n2.json` | Results for n = 2 |
| `propeller_results_n8.json` | Results for n = 8 |

### Running the Analysis

```bash
# Run numerics (n=2 default)
python propeller_flow_numerics.py

# Run for n=8
python propeller_flow_numerics.py --n_arms 8

# Export to JSON
python propeller_analysis_export.py

# Generate plots
python plot_velocity_profiles.py
```

## 10. References

- Blade specification: `blade_specification.md`
- Carrier plate specification: `carrier_plate_specification.md`
- Detailed analysis: `propeller_flow_analysis.tex`

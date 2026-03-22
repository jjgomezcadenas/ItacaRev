# MARS Physics Mechanisms

This document describes the physical mechanisms that determine gas flow disturbances in the MARS system and their effect on ion collection.

---

## Mechanism 1: Boundary Layer Diffusion

### Formation During Motion

When the blade or carrier moves through dense xenon, a turbulent boundary layer forms:
```
δ = 0.37 × L × Re_L^(-1/5)
```

The mean flow in the BL is **horizontal**:
- Blade rotation: tangential velocity u_θ
- Carrier slide: radial velocity u_r

### After Motion Stops: Two Regimes

**1. Within BL (z < δ):** Momentum is trapped. Velocity persists because molecular diffusion is extremely slow:
```
t_diff = δ²/ν = (2.2 mm)² / (3×10⁻⁷ m²/s) ≈ 16,000 s ≈ 4.4 hours
```

**2. Above BL (δ < z < max_reach):** During motion, turbulent eddies (3D chaotic motions) transport horizontal momentum vertically. After motion stops:
- Eddies decay rapidly: t_decay = 2 × (δ/u') ≈ 3-18 ms
- Once eddies die, upward transport stops
- The horizontal momentum already transported to max_reach is "frozen" there
- erfc diffusion tail persists with horizontal velocity

### Key Parameters

| Parameter | Blade BL (n=2) | Carrier BL |
|-----------|----------------|------------|
| Peak velocity | 25.4 m/s | 8.0 m/s |
| BL thickness δ | 2.21 mm | 2.78 mm |
| Direction | Tangential (u_θ) | Radial (u_r) |
| t_decay | 3.5 ms | 14 ms |
| ℓ_turb | 0.66 mm | 0.83 mm |
| max_reach = δ + ℓ_turb + ℓ_mol | 3.3 mm | 4.0 mm |

### Why Lift Must Be ≥ 5 mm

The lift must exceed **max_reach ≈ 4 mm** (worst case: carrier BL).

With z_lift = 5 mm:
- Ion plate surface at z = 5 mm
- Persistent horizontal flow confined to z < 4 mm
- Margin: 1 mm

Ions traversing from cathode (z ~ 10 mm) to ion plate (z = 5 mm) never encounter the persistent horizontal flow.

### Geometry After Lift

```
z (mm)   Description
─────────────────────────────────────────────────────
10       Cathode grid
         ↓ Ions traverse this region (quiescent)
5        Ion plate surface (after lift)
         ══════════════════════════════════════
         GAP: 1 mm margin
         ══════════════════════════════════════
4        max_reach (carrier BL)
3.3      max_reach (blade BL, n=2)
2.8      δ (carrier BL)
2.2      δ (blade BL, n=2)
         ↓ Horizontal velocity persists (hours)
0        Blade/carrier surface (before lift)
```

### The Lift (Phase 3)

The lift moves the ion plate vertically from z = 0 to z = z_lift = 5 mm, placing it above the persistent horizontal BLs.

**Parameters:**
- z_lift = 5 mm
- t_lift = 0.5 s
- V_lift = 10 mm/s (constant velocity) or 20 mm/s peak (triangular profile)

**Flow regime:**
```
Re_lift = V_lift × z_lift / ν = (0.02 m/s × 0.005 m) / (3×10⁻⁷ m²/s) ≈ 330
```
This is **laminar** (Re << 2000) — no turbulent eddies, just a clean Stokes diffusion layer.

**Stokes layer thickness:**
```
δ_Stokes = √(ν × t_lift) = √(3×10⁻⁷ × 0.5) ≈ 0.39 mm
```

**Key point: The lift velocity is vertical (u_z).**
- It does not displace ions laterally
- It only affects ion arrival timing: Δt ~ δ_Stokes / v_d ~ 0.4 mm / 100 mm/s ~ 4 ms
- This is negligible on a 1-15 s drift timescale

**No wait time needed:**
- Horizontal BLs don't decay (persist for hours) — lift provides geometric separation
- Lift Stokes layer is vertical — no lateral effect on ions
- Therefore t_wait = 0

### Total Cycle Time

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Blade rotation | 0.4 s | 0.4 s |
| Phase 2: Carrier slide | 0.2 s | 0.6 s |
| Phase 3: Vertical lift | 0.5 s | 1.1 s |
| **Total** | **1.1 s** | |

**Dead zone** (ions lost because MARS is still moving):
```
d_dead = v_d × t_total = 100 mm/s × 1.1 s = 110 mm ≈ 7% of drift length
```

### Conclusion

- **Horizontal BLs persist** within δ for hours (molecular diffusion timescale)
- **Turbulent transport** carries horizontal momentum to max_reach ≈ 4 mm, then stops (t_decay ~ ms)
- **Lift z_lift = 5 mm** places ion plate above max_reach with 1 mm margin
- **Lift creates laminar vertical Stokes layer** — no lateral ion displacement
- **No wait time needed** — t_wait = 0
- **Result**: Ions never encounter persistent horizontal flow; total cycle 1.1 s

---

## Mechanism 2: Bulk Swirl / Ekman Pumping

The rotating blade transfers angular momentum to the gas, creating a bulk swirl (solid-body-like rotation). Question: can this swirl create vertical flow that reaches the ion collection region?

### Angular Momentum Transfer

The blade exerts tangential drag on the gas, spinning it up. The torque transferred:

```
τ_drag ≈ 48 N·m (for n=2 at peak ω)
```

This creates a bulk rotation in the gas that persists after the blade stops.

### Key Physics: Swirl is Purely Horizontal

Angular momentum is a horizontal vector. The swirl velocity is tangential (u_θ) — it has no vertical component. By itself, bulk swirl cannot transport anything vertically.

### Ekman Pumping (Potential Vertical Transport)

On horizontal boundaries (endcaps), the swirl creates Ekman layers that could pump fluid vertically. However:

**Ekman layer thickness:**

```
δ_Ekman = √(ν/Ω) = √(3×10⁻⁷ / 16) ≈ 0.14 mm
```

**Ekman pumping velocity:**

```
w_Ekman ~ ν/δ_Ekman ~ 2 mm/s (during motion)
```

**Location:** Only at endcaps (top and bottom of vessel), not near the blade.

**Decay time:** After blade stops, Ekman layer decays:

```
t_Ekman_decay ~ δ_Ekman²/ν ~ 0.07 s ≈ 70 ms
```

**Ekman spin-up time** (for completeness — how long for swirl to equilibrate through full vessel height):

```
t_spinup = H / √(ν Ω) ~ 1000 s (molecular)
```

### Conclusion

- Bulk swirl is purely horizontal (tangential velocity u_θ)
- Ekman pumping is confined to ~0.1 mm layers at endcaps, far from ion collection region
- Decays in ~70 ms after blade stops
- **Effect on ions: Zero**

---

## Mechanism 3: Potential (Displacement) Flow

When the blade sweeps through the gas, it displaces fluid. This creates a potential flow field that extends instantaneously through the incompressible region.

### During Motion

The blade thickness (t = 19.2 mm) displaces gas as it sweeps. The displacement velocity scales as:

```
w_disp ~ V_tip × (t/R) ~ 25 m/s × (19.2 mm / 1600 mm) ~ 0.3 m/s
```

This is a potential flow — irrotational, extends throughout the vessel.

### After Motion Stops

Potential flow has no memory. When the blade stops, the displacement flow **vanishes at the speed of sound**:

```
t_acoustic = R / c_s = 1.6 m / 178 m/s ≈ 9 ms
```

Within 9 ms of the blade stopping, all potential flow disturbances have propagated away.

### Key Physics

- Potential flow is instantaneous (incompressible approximation) or propagates at sound speed
- It carries no momentum — it's a kinematic constraint, not a dynamic flow
- When the source (moving blade) stops, the flow vanishes immediately

### Conclusion

- Displacement flow vanishes in ~9 ms after blade stops
- Ions arrive > 1 s after MARS stops
- **Effect on ions: Zero**

---

## Mechanism 4: Pressure Pulse

When the blade decelerates at the end of Phase 1 (bang-bang profile), it creates a pressure pulse that propagates through the gas.

### Pressure Pulse Magnitude

The deceleration creates a dynamic pressure disturbance:

```
Δp ~ ½ ρ V_tip² ~ 0.5 × 79 kg/m³ × (25 m/s)² ~ 25 kPa
```

This is ~1.7% of the ambient pressure (15 bar = 1500 kPa).

### Propagation

The pressure pulse travels at the speed of sound:

```
c_s = 178 m/s
```

It traverses the vessel in:

```
t_traverse = 2R / c_s = 3.2 m / 178 m/s ≈ 18 ms
```

### Key Physics

- Pressure pulses are **acoustic** — they propagate and dissipate
- They do not carry net momentum
- They pass through the drift region and are gone in ~20 ms
- After passage, the gas returns to its original state (no residual velocity)

### Conclusion

- Pressure pulse is transient (~20 ms)
- Ions arrive > 1 s after MARS stops
- **Effect on ions: Zero**

---

## Mechanism 5: Secondary Flows

Secondary flows arise from geometric features: blade tips, blade-carrier junctions, corners, etc.

### Tip Vortices

For an airfoil generating lift, tip vortices form due to pressure difference between upper and lower surfaces. However:

- NACA 0012 at α = 0° produces **no lift**
- No pressure difference → no tip vortex
- The blade is symmetric and operates symmetrically

### Centrifugal Effects

The rotating blade creates centrifugal acceleration in the gas:

```
a_cent = ω² × r ~ (16 rad/s)² × 0.8 m ~ 200 m/s²
```

This drives radial outflow in the blade BL. However:

- This is a **horizontal** (radial) flow
- Confined to the blade BL (z < δ)
- Does not create vertical transport

### Corner Flows at Blade-Carrier Junction

Where the carrier plate meets the blade, corner vortices could form. However:

- These are localized to the junction geometry
- Confined to within a few mm of the junction
- Decay rapidly after motion stops (same t_decay ~ ms as BL)
- Located at z = 0, below the lifted ion plate

### Conclusion

- No tip vortices (no lift at α = 0°)
- Centrifugal flow is horizontal, confined to BL
- Corner flows are localized and decay in ms
- **Effect on ions: Zero**

---

## Mechanism 6: Vortex Shedding

Bluff bodies shed vortices in their wake (von Kármán vortex street). Could the NACA 0012 blade shed vortices that affect ions?

### Strouhal Number and Shedding Frequency

For a NACA 0012 at α = 0°:

```
St ≈ 0.21 (typical for streamlined bodies)
f_shed = St × V / t = 0.21 × 25 m/s / 0.0192 m ≈ 270 Hz
```

### Key Physics

1. **Streamlined body suppresses shedding:** NACA 0012 at zero angle of attack is highly streamlined. Vortex shedding is weak or absent compared to bluff bodies.

2. **Shedding is in the blade plane:** Any shed vortices travel horizontally in the wake, not vertically. They remain confined to z < δ.

3. **Rapid decay after motion stops:** When the blade stops, there's no more flow to sustain shedding. Any existing vortices decay in t_decay ~ ms.

4. **Location:** Vortices are shed from the trailing edge at z = 0, below the lifted ion plate.

### Conclusion

- Vortex shedding is suppressed (streamlined body at α = 0°)
- Any shedding is horizontal, confined to blade plane
- Decays in ms after motion stops
- **Effect on ions: Zero**

---

## Summary

| Mechanism | Direction | Timescale | Effect on Ions |
|-----------|-----------|-----------|----------------|
| 1. Boundary Layer | Horizontal | Persists (hours) within δ | **Zero** (lift separates) |
| 2. Bulk Swirl | Horizontal | Persists | **Zero** (no vertical component) |
| 3. Potential Flow | — | ~9 ms | **Zero** (vanishes) |
| 4. Pressure Pulse | — | ~20 ms | **Zero** (transient) |
| 5. Secondary Flows | Horizontal | ~ms decay | **Zero** (confined to BL) |
| 6. Vortex Shedding | Horizontal | ~ms decay | **Zero** (confined to blade plane) |

**The only mechanism requiring mitigation is Mechanism 1 (Boundary Layer), and the 5 mm lift provides geometric separation.**

All other mechanisms either:
- Are purely horizontal and don't reach the ion collection region, or
- Decay in milliseconds, long before ions arrive (> 1 s)

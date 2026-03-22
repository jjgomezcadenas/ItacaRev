# MARS Research Status

## Current Baseline Design

**Configuration:** Propeller with N = 2 arms

| Component | Parameter | Value |
|-----------|-----------|-------|
| Blade | Profile | NACA 0012, α = 0° |
| | Length | R = 1.6 m |
| | Chord | c = 160 mm |
| | Mass | m_blade = 0.25 kg |
| Carrier | Dimensions | 160 × 160 × 5 mm³ |
| | Mass | m_plate = 0.25 kg |
| | Initial position | r₀ = 0.8 m |
| Motor | Torque | τ = 60 N·m |

## Motion Sequence

| Phase | Motion | Duration | Peak Velocity |
|-------|--------|----------|---------------|
| 1 | Blade rotation (180°) | 0.4 s | V_tip = 25.4 m/s |
| 2 | Carrier slide (0.8 m) | 0.2 s | V_slide = 8 m/s |
| 3 | Vertical lift (5 mm) | 0.5 s | V_lift = 10 mm/s |
| **Total** | | **1.1 s** | |

## Key Results

### Gas Disturbance Analysis

All 6 mechanisms analyzed — **all have zero effect on ions**:

| Mechanism | Why Zero Effect |
|-----------|-----------------|
| 1. Boundary Layer | Horizontal, confined to z < 4 mm; 5 mm lift separates |
| 2. Bulk Swirl | Purely horizontal (tangential); no vertical component |
| 3. Potential Flow | Vanishes in ~9 ms after motion stops |
| 4. Pressure Pulse | Transient (~20 ms); ions arrive > 1 s later |
| 5. Secondary Flows | Horizontal, confined to BL; decay in ms |
| 6. Vortex Shedding | Suppressed (streamlined body); horizontal if any |

### Critical Design Insight

**The lift is essential** — not because BLs decay (they don't), but because it provides geometric separation:
- Horizontal BLs persist within δ ≈ 3 mm for hours (molecular diffusion timescale)
- Turbulent transport reaches max_reach ≈ 4 mm, then stops (t_decay ~ ms)
- Lift z_lift = 5 mm places ion plate above max_reach with 1 mm margin

### Performance

| Metric | Value |
|--------|-------|
| Total cycle time | 1.1 s |
| Dead zone | 110 mm (7% of drift length) |
| Ion displacement | 0 (all mechanisms zero) |

## Completed Analysis

- [x] 6 flow mechanisms analyzed quantitatively
- [x] Boundary layer physics understood (persist vs decay)
- [x] Lift requirement established (z_lift ≥ 5 mm)
- [x] t_wait eliminated (no physics reason for it)
- [x] Cycle time optimized to 1.1 s

## Open Questions


## Next Steps


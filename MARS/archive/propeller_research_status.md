# Propeller Flow Analysis - Status

**Last updated:** 2026-03-22

## Purpose

Analyze gas quiescence after propeller motion in dense xenon (15 bar, 300K) for ion detection.

## Current State

- **3 velocity profiles** computed: rotation BL, slide BL, Stokes layer from lift
- **Key result (n=2):** z_q3 ≈ 7.5 mm for quiescence (u < 0.1 mm/s)
- **Total cycle:** ~2.1 s

## Design Decisions

- Analytical erfc solution (no PDE solver needed)
- Profile 3: erfc for impulsively stopped plate, z_q found after velocity peak
- Profile 1 fully decays by evaluation time (t_decay = 3.5 ms vs t_eval = 1.7 s)

## Pending

- Update LaTeX document with Phase 3 section

## Quick Reference

```bash
python propeller_flow_numerics.py          # Run analysis (n=2 default)
python propeller_flow_numerics.py --n_arms 8
python propeller_analysis_export.py        # Generate JSON
python plot_velocity_profiles.py           # Generate plots
```

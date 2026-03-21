# MARS / ITACA Detector — Design and Analysis

## Project Overview

MARS (Mechanism for Azimuthal Rotation and Sampling) is a subsystem of the ITACA detector. It operates inside a vertical cylindrical pressure vessel filled with xenon gas at 15 bar, 300 K. A rotating element (propeller or disk) moves carrier plates to discrete azimuthal positions. After each rotation the gas must become quiescent so that ion collection plates can measure signals without flow-induced noise.

The core physics question: **what is the residual gas velocity at a given height above the rotating element, a given time after it stops?**

## Two Competing Geometries

- **Geometry A (Propeller):** N NACA 0012 blades at zero angle of attack, symmetric, no lift, drag only.
- **Geometry B (Rotating Disk):** Solid disk with radially-sliding carrier plates.

Both geometries are described in `mars_geoms.tex`.

## Xenon Gas Properties

Dense xenon at 15 bar, 300 K has unusual properties that dominate the physics:

- ρ ≈ 79 kg/m³ (66× denser than air)
- μ ≈ 2.32×10⁻⁵ Pa·s (similar to air)
- ν = μ/ρ ≈ 2.94×10⁻⁷ m²/s (51× smaller than air)
- c_s ≈ 178 m/s

The very low kinematic viscosity means high Reynolds numbers at modest velocities and very slow molecular diffusion.

## Key Physics Results (Propeller Geometry)

Six mechanisms were analyzed for their contribution to residual velocity above the blade:

1. **Eddy diffusion** — Turbulent BL (δ ≈ 2–5 mm depending on parameters) decays in ~5–10 ms. Turbulent + molecular diffusion reach extends ~1.5 mm beyond δ. Zero at z_obs if z_obs − δ > ~2 mm; small nonzero tail if gap < ~1 mm.
2. **Bulk swirl / Ekman** — Angular momentum stays in blade plane. Ekman spin-up time ~1000 s (molecular), ~50 s (turbulent). Zero.
3. **Potential (displacement) flow** — Vanishes at speed of sound after blade stops (~9 ms). Zero.
4. **Pressure pulse** — Transient acoustic wave, passes through. Zero.
5. **Secondary flows** — No lift at α = 0 means no tip clearance flow. Centrifugal effects tangential. Zero.
6. **Vortex shedding** — Streamlined body, suppressed. Confined to blade plane. Zero.

Vessel confinement (walls, endcaps, tip gap) introduces no new vertical transport mechanism. All confinement effects are tangential, transient, or orders of magnitude too slow. Quantified numerically in the Python script (Section 12).

## Repository Structure

| File | Description |
|------|-------------|
| `propeller_flow_analysis.tex` | Full LaTeX document: mechanism-by-mechanism physics analysis with equations, including vessel confinement section |
| `propeller_flow_numerics.py` | Python script: parameterized numerics for all 6 mechanisms + confinement. CLI arguments: `--chord`, `--sweep_angle`, `--z_obs`, `--t_obs`, `--m_blade`, `--m_plate`, `--tau_motor` |
| `mars_geoms.tex` | LaTeX document: geometric description of both propeller and disk geometries, with parameter tables |
| `ITACA2.pdf` | Reference drawing of the ITACA detector |

## Goal

Design the best possible MARS system. "Best" means:

- Gas is quiescent (residual velocity ≈ 0) at the observation height within the observation time after the rotating element stops.
- All design choices are justified by quantitative analysis.
- The design space (chord, number of blades/positions, sweep angle, clearances) is explored systematically.

## Working Rules

**These rules are mandatory. Follow them in every interaction.**

1. **Always present plans before acting.** Show what you intend to do (calculations, code changes, new files) and wait for explicit approval. Never launch into long computations or code writing without an OK.

2. **Quantify everything.** Every physical effect must have a numerical estimate. Use Python or Julia scripts with parameterized inputs. No hand-waving, no "this is small" without a number.

3. **Check every formula against literature.** When introducing an equation (BL thickness, Ekman time, diffusion length, etc.), cite the source: textbook name, paper, or standard reference. If you cannot cite a source, say so explicitly.

4. **Document all analysis in LaTeX.** Physics analysis goes in `.tex` files with full derivations, equations, and discussion. Numerics go in `.py` or `.jl` scripts that can be run independently.

5. **Scripts must be parameterized.** All physical parameters that might vary are CLI arguments with sensible defaults. No magic numbers buried in code.

6. **When in doubt, compute.** If there is any question about whether an effect matters, write the estimate. A one-line calculation is always better than a qualitative argument.

7. **Be honest about uncertainties.** If a result depends on a modeling choice (e.g., turbulence decay rate), state the assumption and its basis. Flag results that are sensitive to assumptions.

8. **Distinguish zero from small.** Zero means there is no physical mechanism. Small means there is a mechanism but its magnitude is negligible for the application. These are different statements and must be distinguished.

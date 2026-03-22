# MARS / ITACA Detector — Design and Analysis

## Start Here

Read these files in order:

1. **`itaca_and_mars_spec.md`** — Complete system specification (detector geometry, ion physics, MARS motion sequence)
2. **`mars_physics.md`** — All 6 flow mechanisms analyzed, why the lift is essential
3. **`mars_research_status.md`** — Current baseline design, key results, open questions

## Project Overview

**MARS** (Mechanism for Azimuthal Rotation and Sampling) positions ion collection plates in the **ITACA** detector. It operates in dense xenon gas (15 bar, 300 K) inside a 3.2 m diameter × 1.5 m tall cylindrical vessel.

**Baseline design:** Propeller with N = 2 NACA 0012 blades at zero angle of attack.

**Core physics question:** What is the residual gas velocity at the ion collection height after MARS stops?

**Answer:** Zero — all 6 mechanisms either decay in ms or are geometrically separated by the 5 mm lift.

## Working Rules

**These rules are mandatory. Follow them in every interaction.**

1. **Always present plans before acting.** Show what you intend to do and wait for explicit approval.

2. **Quantify everything.** Every physical effect must have a numerical estimate. No hand-waving.

3. **Check formulas against literature.** Cite the source (textbook, paper, standard reference).

4. **Document analysis in LaTeX.** Physics goes in `.tex` files. Numerics go in `.py` scripts.

5. **Scripts must be parameterized.** CLI arguments with sensible defaults. No magic numbers.

6. **When in doubt, compute.** A one-line calculation is always better than a qualitative argument.

7. **Be honest about uncertainties.** State assumptions and flag results that are sensitive to them.

8. **Distinguish zero from small.** Zero = no physical mechanism. Small = mechanism exists but negligible.

## Repository Structure

| File | Description |
|------|-------------|
| `itaca_and_mars_spec.md` | Complete system specification |
| `mars_physics.md` | All 6 mechanisms analyzed |
| `mars_research_status.md` | Current status and open questions |
| `propeller_flow_numerics.py` | Parameterized numerical calculations |
| `propeller_results_n2.json` | Numerical results for baseline (n=2) |

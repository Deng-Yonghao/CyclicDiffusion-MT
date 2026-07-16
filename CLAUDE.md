# CyclicDiffusion-MT

Multi-target full-atom cyclic peptide generation via diffusion models.

## Quick Start (New Session)

When entering this project, read:
1. `docs/superpowers/specs/2026-07-14-cyclicdiffusion-mt-design.md` — complete design specification
2. `memory/MEMORY.md` — index of implementation details

Then ask the user which component they want to work on, or pick up from where the conversation left off.

## Project Status

- **Phase**: Design completed, ready for implementation
- **Current step**: Write implementation plan → scaffold project → implement modules
- **Next action**: Invoke `superpowers:writing-plans` to create the implementation plan

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/superpowers/specs/2026-07-14-cyclicdiffusion-mt-design.md` | Full design spec |
| `memory/MEMORY.md` | Memory index |
| `memory/project-overview.md` | Key decisions summary |
| `memory/se3-frame-denoiser.md` | SE(3) denoiser math |
| `memory/wrapped-normal-diffusion.md` | Angle diffusion details |
| `memory/nerf-implementation.md` | Coordinate conversion |
| `memory/data-preprocessing.md` | Data pipeline |
| `memory/non-canonical-amino-acids.md` | AA vocabulary |
| `memory/masked-discrete-diffusion.md` | Discrete diffusion |
| `memory/affinity-head.md` | Affinity regression |
| `memory/geometry-and-cyclo-losses.md` | Constraint losses |

## Tech Stack

PyTorch 2.x, Biopython, self-implemented NeRF, Rosetta (offline scoring)

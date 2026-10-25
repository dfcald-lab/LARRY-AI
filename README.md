# LARRY-AI

LARRY-AI is my personal project to build AI from the ground up and understand how its components work.

I learn by implementing systems, running experiments, investigating failures, and documenting results. My long-term goal is to build an AI system I can run locally and understand well enough to maintain and improve.

## Approach

**Understand → Implement → Test → Investigate → Improve → Document**

I aim to keep experiments understandable and base conclusions on evidence, including failed results.

## Repository guide

- `AI_PROJECT_PLAN.md` — long-term vision and learning roadmap
- `docs/PROJECT_JOURNAL.md` — detailed research history and results
- [`docs/EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md) — index of all 76 experiments
- `src/` — reusable neural-network components
- `experiments/` — numbered experiments and research scripts

## Current research

Experiments 067–076 investigate neural-network representation, adding small ReLU features to a trained XOR network, output coupling, hidden-layer gradients, and the difference between initial gradient predictions and later training behavior.

Experiment 075 found the fixed-state decomposition `g(gamma) = gamma*A + gamma^2*B`. Experiment 076 tests how that initial relationship connects to training trajectories. The results depend on the tested configuration and are not universal guarantees.

The next research direction is to test whether early activation-pattern changes and relative hidden-parameter movement can help predict instability across initializations.

## Development

Run commands from the repository root. Use `PYTHONPATH=.` when an experiment needs imports from the local `src` package. Inspect a script before running it because some experiments launch full sweeps.

Start with the [project plan](AI_PROJECT_PLAN.md) and [project journal](docs/PROJECT_JOURNAL.md). Use the [experiment index](docs/EXPERIMENT_INDEX.md) to navigate all 76 numbered experiments.

# UrbanEV Codex Rules

## Project goal

This repository is used for EV charging spatio-temporal forecasting research.
The current stage is innovation model development for the paper direction:

- EV charging prediction
- UrbanEV-based spatio-temporal forecasting
- Current focus: cold-start inductive setting and proposed model skeleton

## Read order

Before making changes, read files in this order:

1. `AGENTS.md`
2. `docs/handover.md`
3. `docs/project_status.md`
4. `docs/innovation_spec.md`
5. `code/parse.py`
6. `code/utils.py`
7. `code/train.py`

## Engineering boundaries

- Future model development should happen under `code/`.
- Do not directly modify third-party baseline source files unless explicitly requested.
- Prefer adapter-style integration under `code/models/*/*_adapter.py`.
- Keep `parse.py` as the global parameter bus.
- Keep `utils.py` as the data loading and model assembly hub.
- Keep `train.py` as the training and evaluation loop entry.

## Current priorities

1. Build the cold-start masking mechanism.
2. Verify training and evaluation behavior under the new protocol.
3. After Phase 1 is stable, start the proposed model skeleton under `code/models/Proposed/`.

## Change policy

- Explain touched files before code edits.
- Keep changes narrow and traceable.
- Record experiments in `experiments/experiment_log.md`.
- Separate measured conclusions from hypotheses.


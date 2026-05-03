# TODO

## Phase 0: Context stabilization

- [x] Create project-level `AGENTS.md`
- [x] Move AI handover context into `docs/handover.md`
- [x] Create `docs/project_status.md`
- [x] Create `docs/innovation_spec.md`
- [x] Create `experiments/experiment_log.md`

## Phase 1: Cold-start inductive protocol

- [ ] Add `--cold_start` to `code/parse.py`
- [ ] Add `--cold_ratio` to `code/parse.py`
- [ ] Generate node mask in `code/utils.py`
- [ ] Blind occupancy history for held-out nodes in `code/utils.py`
- [ ] Keep external features unchanged
- [ ] Restrict training loss to observed nodes in `code/train.py`
- [ ] Restrict test metrics to held-out nodes in `code/train.py`
- [ ] Add a minimal verification command and expected behavior note

## Phase 2: Proposed model skeleton

- [ ] Create `code/models/Proposed/`
- [ ] Add `DynamicGraphGenerator.py`
- [ ] Define input and output tensor contract
- [ ] Wire model registration into the existing model loading flow

## Phase 3: Paper-facing cleanup

- [ ] Rewrite the root `readme.md` so the paper project context appears before the upstream dataset description
- [ ] Add a reproducible experiment command section for your paper track
- [ ] Revalidate baseline scores in the current codebase


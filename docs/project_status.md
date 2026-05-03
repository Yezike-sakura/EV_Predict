# Project Status

## Current positioning

- Project topic: EV charging spatio-temporal forecasting.
- Base repository: upstream UrbanEV dataset and benchmark codebase.
- Current branch purpose: extend the upstream project for paper-oriented model research.
- Main development track: `code/`.

## What has already happened

- Several baseline models have been added or adapted.
- Part of the internal framework has already been refactored to fit the research direction.
- Adapter-based integration already exists for models such as `AGCRN`, `GWNET`, and `PDG2Seq`.
- The upstream root `readme.md` still mainly describes the dataset and public benchmark repo.

## Practical code entry points

- `code/parse.py`: global experiment arguments.
- `code/utils.py`: data loading, feature assembly, loader creation, model registration.
- `code/train.py`: training, validation, checkpointing, evaluation flow.
- `code/models/`: model implementations and adapters.

## Baselines to beat

The current handover document lists the following targets:

- `DLinear`: MAPE `75.7%`, MAE `0.361`, `seq_len=96`
- `GWNET`: MAPE `13.08%`, MAE `0.173`, `seq_len=12`
- `PatchTST`: MAPE `11.0%`, MAE `0.118`, `seq_len=96`

These numbers should be treated as working targets until revalidated in the current codebase.

## Current research phase

- The paper has moved past baseline comparison.
- The current stage is innovation implementation.
- The first engineering milestone is the cold-start inductive protocol.
- The second milestone is the proposed model skeleton.

## Immediate gaps

- No project-level `AGENTS.md` existed before now.
- No structured `docs/` directory existed before now.
- No dedicated `experiments/experiment_log.md` existed before now.
- The Gemini handover file was not in a standard project doc location.


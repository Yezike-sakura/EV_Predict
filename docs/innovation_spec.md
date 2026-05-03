# Innovation Spec

## Research direction

The current proposed direction is a hybrid spatio-temporal model for EV charging forecasting with:

- residual hybrid dynamic graph generation
- patch-based spatial-temporal coupling
- node-level cold-start inductive evaluation

## Innovation 1: Residual hybrid graph

Target:

- combine static graph structure with dynamic graph residuals
- use multimodal meta features to generate dynamic spatial relations
- improve robustness and cold-start generalization

Implementation constraints:

- static graph base comes from the existing adjacency or distance priors
- dynamic graph should be generated efficiently
- avoid naive dense `O(N^2)` fully connected reconstruction where possible
- prefer node embedding interaction or attention-style lightweight generation

Expected output:

- dynamic or hybrid graph tensor shaped like `[B, P, N, N]`

## Innovation 2: Patch-level spatial-temporal coupling

Target:

- split long sequences into patches
- perform spatial aggregation at patch level
- perform temporal attention across patches

Expected dataflow:

1. input tensor shaped like `[B, C, N, L]`
2. patch reshape to `[B, C, N, P, P_len]`
3. intra-patch spatial fusion using hybrid graph
4. inter-patch temporal modeling across `P`

## Innovation 3: Cold-start inductive setting

Target:

- mask out a subset of nodes during input construction
- keep external features available
- train on old nodes only
- evaluate on held-out nodes only

Expected Phase 1 changes:

- `code/parse.py`: add `cold_start` and `cold_ratio`
- `code/utils.py`: create mask and blind node history input
- `code/train.py`: restrict loss to observed nodes and metrics to held-out nodes

## Current implementation order

1. Finish the cold-start protocol first.
2. Verify training and evaluation logic.
3. Create `code/models/Proposed/` and start the model skeleton.


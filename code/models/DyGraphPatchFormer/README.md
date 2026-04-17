# DyGraph-PatchFormer Stage-1 Notes

## Purpose
- Build `A_hybrid` with shape `[B, N, N]`:
  - `A_hybrid = A_static + lambda * DeltaA_dynamic`
- Keep the module plug-and-play in current UrbanEV training pipeline.

## Runtime Stats
- Adapter exposes `latest_graph_stats` after each forward:
  - `lambda`
  - `adj_mean`
  - `adj_std`
  - `adj_zero_ratio`

## Suggested experiment log template
| model | fold | seq_len | pred_len | add_feat | lambda | adj_mean | adj_std | adj_zero_ratio | val_loss | test_mae | test_mape |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dygraph_patchformer | 0 | 12 | 1 | time | 0.10 | - | - | - | - | - | - |


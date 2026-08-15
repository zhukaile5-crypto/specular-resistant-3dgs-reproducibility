# Reproducibility Matrix

## Reproducibility levels

- **R1 -- Exact archived-artifact verification:** the frozen SHA-256 value is
  included and can verify the paper artifact in the original workspace. Paper
  image files are intentionally not redistributed in this package.
- **R2 -- Deterministic post-processing:** the artifact can be rebuilt from
  preserved machine-readable metrics or source images. Pixel values are
  expected to match under the recorded library/font stack; PNG byte hashes may
  vary across encoders.
- **R3 -- Metric recomputation:** the evaluator can be rerun from preserved
  rendered or processed images and compared with the frozen JSON/CSV results.
- **R4 -- Full pipeline rerun:** preprocessing, COLMAP, and/or 3DGS can be rerun
  from earlier-stage inputs. Numerical proximity is expected, but bitwise
  identity is not promised for GPU training.
- **R5 -- Archival only:** an exact final artifact is preserved, but an
  earlier provenance step is incomplete and cannot be recreated from raw data
  alone.

## Figures

| Paper item | Current paper asset | Best supported level | What can be reproduced | Limitation |
|---|---|---:|---|---|
| Figure 1 | `Fig01_photo_scene6_global_final_method_comparison.png` | R1/R2 | Exact three-panel crop from frame 0051 and preserved outputs | Requires the original, StableDelight, and final fused images |
| Figure 2 | `Fig02_photo_scene6_local_final_method_comparison.png` | R1/R2 | Exact three-panel crop from frame 0012 and preserved outputs | Same dependency as Figure 1 |
| Figure 3 | `Fig03_2D_reflection_suppression_metrics.png` | R1/R3 | Full 96-image metric recomputation with frozen thresholds and reference comparison | Same-family proxy, not independent ground truth |
| Figure 4 | historical `Fig07_raw_vs_deglared_3DGS_matched_novel_views_reflection_marked.png` | R1/R2/R4 | Exact montage and annotations from six preserved matched renders; full models can also be rerun | Full GPU reruns need not be bitwise identical |
| Figure 5 | historical `Fig08_floating_specular_artifact_suppression_marked.png` | R1/R2/R5 | Exact montage from eight archived crops and fixed annotation boxes | Screenshot camera matrices and deterministic crop provenance were not archived |
| Figure 6 | historical `Fig09_matched_orbit_reflection_evaluation.png` | R1/R2/R3/R4 | Plot from archived CSV/JSON, metrics from 90 paired renders, or full 3DGS rerun | Confidence intervals describe one trajectory, not training/scene uncertainty |

## Tables

| Paper item | Best supported level | Machine-readable source | Limitation |
|---|---:|---|---|
| Table 1: method parameters | R2 | `versions/registry.json`, `blend_strategies.py` | The table is a selected subset of the full V6 profile |
| Table 2: method components | R2 | detector, cross-view, matte/fusion, and acceleration source files | Descriptive architecture table; it is not a computed benchmark |
| Table 3: cross-scene 2D diagnostics | R3/R4 | per-scene suppression summaries and `2d_structure_fidelity_20260811/summary.json` | Same-family proxy and narrow original-only protected masks |
| Table 4: held-out fixed-pose control | R2/R3, partial R4 | two `spec_eval` JSON files and two `view_dependence` JSON files | Exact metric recalculation is supported; exact retraining is not because seeds were not archived |
| Table 5: cross-scene end-to-end 3DGS | R2/R3/R4 | matched-orbit/common-support summaries and trajectory manifests | S2/S5 use midpoint trajectories; S3/S4/S6 use supported ellipses |

## Current automation boundary

`run_reproduction.py tables` generates every table in one command.
`run_reproduction.py figures` handles archived copies and deterministic
post-processing. Heavy detector, COLMAP, and 3DGS reruns remain explicit
per-experiment commands so that expensive or destructive work is never started
implicitly.

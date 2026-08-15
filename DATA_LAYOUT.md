# Expected Repository Layout

The package does not redistribute raw captures, StableDelight weights, COLMAP
databases, or 3DGS checkpoints. The full repository should provide the
following paths when the corresponding reproduction level is requested.

```text
nvidia_project/
  StableDelight/
    stabledelight/utils/specular_detector_v3.py
  data/custom/
    photo_scene*/
    photo_scene*_delighted/
    photo_scene*_blended_evidence_matte_t0.3_v3*/
  experiment_configs/
  raw_images/
    photo_scene*/
  results/
    cross_scene_frozen_metrics_20260811/
    2d_structure_fidelity_20260811/
    end_to_end_3dgs_cross_scene_20260811/
    photo_scene6/
    final_paper_figures/
  third_party/gsplat/
  versions/registry.json
```

## Minimum inputs by task

- **Paper-output verification:** this package plus the local
  `results/final_paper_figures/` and preserved source-output paths.
- **Table export from live results:** the JSON/CSV result directories listed
  above.
- **Figures 1--2 composition:** frames 0051 and 0012 from raw images,
  StableDelight outputs, and final fused outputs.
- **Figure 3 metric recomputation:** all 96 original and final processed S6
  images plus the frozen V3 detector.
- **Figure 4 composition:** the six preserved matched novel-view renders.
- **Figure 5 composition:** the eight preserved user-selected crop images.
- **Figure 6 plot redraw:** `summary.json` and `per_view_metrics.csv` from the
  S6 matched 90-pose trajectory.
- **Full 3DGS rerun:** images, COLMAP models, the locked `third_party/gsplat`
  checkout, CUDA, and the recorded Conda environment.

## Dataset release

The existing dataset archives and their checksums are stored separately under
`release_packages/2026-08-09_upload_ready`. Keep those large files outside the
Git source history and publish them through a dataset release or external
storage service.

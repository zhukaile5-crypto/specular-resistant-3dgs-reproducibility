# Reproduction Commands

Commands below assume execution from the repository root. Replace
`/path/to/nvidia_project` with the local checkout path. The historical heavy
pipeline was validated in WSL; lightweight package commands are portable.

## Figure 1 and Figure 2

The package wrapper rebuilds only the final-method panels into a new output
directory:

```bash
python release_packages/2026-08-15_paper_reproducibility/run_reproduction.py \
  figures --project-root /path/to/nvidia_project --mode compose --ids 1 2
```

Authoritative generator:

```bash
python experiment_configs/build_v1_v6_visual_comparison.py
```

Reproduction level: exact composition from preserved images. It does not rerun
StableDelight or the proposed preprocessing.

## Figure 3

```bash
python release_packages/2026-08-15_paper_reproducibility/run_reproduction.py \
  figures --project-root /path/to/nvidia_project --mode recompute --ids 3
```

This runs the audited entry point
`experiment_configs/reproduce_fig3_specular_reduction.py`, recomputes scores
for all 96 original/final image pairs, verifies frozen source/reference hashes,
and compares the new summary with the preserved historical summary.

Reproduction level: exact metric recomputation from preserved processed
images. It does not independently validate physical reflectance removal.

## Figure 4

```bash
python release_packages/2026-08-15_paper_reproducibility/run_reproduction.py \
  figures --project-root /path/to/nvidia_project --mode compose --ids 4
```

This reconstructs the one-row montage and fixed teal annotations from six
preserved matched renders. To regenerate the underlying S6 models and renders,
use the training/evaluation stages in:

```bash
python experiment_configs/full96_novel_view_comparison.py --help
python experiment_configs/evaluate_matched_orbit_specularity.py --help
```

Reproduction level: exact montage from archived renders; full GPU rerun is
protocol-reproducible but not claimed to be bitwise identical.

## Figure 5

```bash
python release_packages/2026-08-15_paper_reproducibility/run_reproduction.py \
  figures --project-root /path/to/nvidia_project --mode compose --ids 5
```

This rebuilds the montage from eight archived crop images and applies the
fixed attention boxes and pose labels. The camera matrices and deterministic
crop provenance of the original presentation screenshots were not archived.

Reproduction level: exact from archived crops, archival-only before the crop
stage. This is intentionally an appearance demonstration rather than a
registered quantitative comparison.

## Figure 6

Redraw the chart from the archived summary and per-view CSV:

```bash
python release_packages/2026-08-15_paper_reproducibility/run_reproduction.py \
  figures --project-root /path/to/nvidia_project --mode compose --ids 6
```

Recompute the 90-pose proxy metrics from preserved renders:

```bash
python experiment_configs/evaluate_matched_orbit_specularity.py \
  --stage evaluate \
  --output-dir results/photo_scene6/full96_reestimated_pose/matched_orbit_specularity_3x30 \
  --scene-name photo_scene6
```

Use a fresh output directory for a non-destructive rerun. A full render rerun
also requires the two 30k-step checkpoints and CUDA.

## Tables 1--5

Generate every table from the current live configuration and result files:

```bash
python release_packages/2026-08-15_paper_reproducibility/run_reproduction.py \
  tables --project-root /path/to/nvidia_project
```

The command writes reusable LaTeX, Markdown, and JSON. It does not edit the
paper source.

### Table 3 metric reruns

Panel (a), one scene at a time:

```bash
python experiment_configs/evaluate_v6_specular_reduction.py --help
```

Panel (b), all six scenes:

```bash
python experiment_configs/evaluate_2d_structure_fidelity.py \
  --scenes photo_scene photo_scene2 photo_scene3 photo_scene4 photo_scene5 photo_scene6 \
  --workers 2
```

### Table 4 metric reruns

The paired datasets are prepared with:

```bash
python experiment_configs/prepare_conventional_reconstruction_pair.py \
  --scene photo_scene6 --data-factor 4
```

Validation rendering and metrics use:

```bash
python experiment_configs/evaluate_conventional_reconstruction_pair.py --help
python experiment_configs/evaluate_specular_recon.py image-metrics --help
python experiment_configs/evaluate_specular_recon.py view-dependence --help
```

The exact paper metrics can be recalculated from the preserved checkpoints and
renders. Exact training replay is not claimed because the original seeds were
not archived.

### Table 5 full and evaluation reruns

S2--S5 end-to-end experiments:

```bash
python experiment_configs/run_cross_scene_end_to_end_3dgs.py \
  --scene photo_scene2 --stage all --seed 20260811
```

Repeat for `photo_scene3`, `photo_scene4`, and `photo_scene5`. S2 and S5 then
use the jointly supported midpoint evaluator:

```bash
python experiment_configs/evaluate_common_support_trajectory.py --help
```

S3, S4, and S6 use the supported three-height ellipse evaluator:

```bash
python experiment_configs/evaluate_matched_orbit_specularity.py --help
```

Always run `--help` or a dry planning stage first. Full COLMAP and 3DGS reruns
are expensive, write large artifacts, and require the correct CUDA environment.

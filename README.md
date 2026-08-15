# Paper Reproducibility Package

This package documents and reproduces the six figures and five tables in the
current paper source at
`paper/2D-Deglare-Reconstruction-Framework-Extended/main.tex`.

The package separates three claims that are often conflated:

1. **Archived-output verification** checks the exact files and displayed
   numbers used by the paper.
2. **Deterministic post-processing** rebuilds crops, montages, plots, and
   tables from preserved images or machine-readable metrics.
3. **Full experiment reruns** repeat detection, COLMAP, or 3DGS training from
   earlier-stage inputs. These runs require the complete repository, datasets,
   model weights, and the documented software environments. GPU training is
   not claimed to be bitwise deterministic.

The current manuscript numbers its figures 1--6. Some preserved source files
retain their historical names `Fig07`, `Fig08`, and `Fig09`; the manifest in
this package records the unambiguous mapping.

## Quick start

Use Python 3.8 or newer for the lightweight verification and table-export
tools. Run the following commands from this package directory:

```bash
python run_reproduction.py list
python run_reproduction.py verify --project-root /path/to/nvidia_project
python run_reproduction.py tables --project-root /path/to/nvidia_project
```

On the original Windows workstation, the project root is
`D:\nv\nvidia_project`:

```powershell
python run_reproduction.py verify --project-root D:\nv\nvidia_project
python run_reproduction.py tables --project-root D:\nv\nvidia_project
python run_reproduction.py figures --project-root D:\nv\nvidia_project --mode compose --ids 1 2 4 5 6
```

`--mode compose` rebuilds presentation figures from preserved project inputs
without modifying those inputs. Figure 3 has a separate audited metric rerun:

```bash
python run_reproduction.py figures --project-root /path/to/nvidia_project --mode recompute --ids 3
```

The recomputation reads all 96 source/output images and can take substantially
longer than montage composition.

## Outputs

- `generated_tables/`: LaTeX, Markdown, and JSON representations of Tables
  1--5.
- `reference/paper_expected_values.json`: frozen displayed table values used
  for verification.
- `artifact_manifest.json`: machine-readable provenance, commands, expected
  files, and limitations for every figure and table.
- `REPRODUCIBILITY_MATRIX.md`: concise human-readable reproducibility status.
- `source_snapshot/`: reviewable copies of the project scripts that generated
  or evaluated the reported artifacts. The authoritative live copies remain
  at their documented repository paths.

## Environment boundaries

Lightweight verification, table export, and most montage composition are
portable to Windows. The historical evaluation and training scripts were
developed under WSL and several archived scripts still contain the original
`/home/dministrator/nvidia_project` path. Full CUDA/COLMAP/3DGS reruns should
therefore use the documented WSL Conda environments unless those paths are
ported first.

Use the `stabledelight` environment for diffusion inference and the
`nerfstudio` environment for COLMAP, metric evaluation, and vanilla 3DGS.
Do not mix these environments.

## Important limitations

- The proxy reflection evaluator belongs to the same method family as the
  proposed preprocessing. It is not physical reflectance ground truth.
- Table 4 can be recalculated exactly from the preserved checkpoints/renders,
  but the original training seeds were not archived. A fresh training run is
  therefore not expected to reproduce the exact displayed decimals.
- Figure 5 can be reconstructed exactly from its archived crop images. The
  source screenshots do not have archived camera matrices or deterministic
  crop manifests, so the figure is an appearance-only case study.
- Tables are generated from JSON/CSV sources, but the manuscript still embeds
  manually formatted LaTeX. The verifier checks the generated displayed values
  against the frozen paper values.

No paper PNG, raw dataset, model weight, checkpoint, or third-party source
archive is duplicated here. The manifest retains the expected SHA-256 value
for each figure so a local reconstruction can still be checked. See
`DATA_LAYOUT.md` for the expected repository layout.

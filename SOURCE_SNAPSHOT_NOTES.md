# Source Snapshot Notes

`source_snapshot/` contains the project-authored files directly involved in
the figures, tables, detector, fusion profile, evaluation, and 3DGS protocol.
The snapshot is included for review and archival convenience; the full
repository remains the execution source of truth because datasets, shared
modules, StableDelight weights, and the locked gsplat checkout are not copied
into this package.

Two report-writing blocks were translated from Chinese to English in the
snapshot so that all package comments and documentation are English:

- `experiment_configs/full96_novel_view_comparison.py`
- `experiment_configs/evaluate_matched_orbit_specularity.py`

Only generated README prose and filenames were changed in those snapshot
copies. Metric definitions, numerical calculations, render logic, parameters,
and experiment paths were not changed. `SOURCE_SHA256SUMS.txt` identifies the
exact packaged copies.

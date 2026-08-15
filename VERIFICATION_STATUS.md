# Verification Status

Audit date: 2026-08-15

The package-level verifier completed successfully on the restored Windows
workspace:

- 44 checks passed.
- 0 checks failed.
- All six live paper figures and their preserved source outputs match the
  frozen SHA-256 values stored in the manifest.
- The displayed values of all five generated tables match the frozen paper
  expectations.
- All 22 files in the reviewable source snapshot match
  `SOURCE_SHA256SUMS.txt`.

Deterministic composition was smoke-tested on Windows for Figures 1, 2, 4,
and 5. The composition completed successfully, but the resulting annotated
PNGs were not byte- or pixel-identical because the restored Windows font and
FreeType rasterization stack differs from the historical environment. This
does not affect verification against the frozen hashes. It is one reason the
package distinguishes R1 archived-artifact verification from R2
post-processing.

Figure 3's full 96-image metric recomputation was not repeated during this
lightweight packaging audit because it is substantially more expensive. Its
audited evaluator, frozen thresholds, expected outputs, and command are
included. Figure 6's plotting path was syntax-checked but not executed in the
minimal local Python runtime because Matplotlib was unavailable; its frozen
hash and machine-readable inputs remain available for verification.

Paper PNG files are intentionally excluded from the public package. This
status records the pre-release audit performed against the restored local
workspace; it does not imply that the image assets are redistributed.

Machine-specific JSON reports produced by `run_reproduction.py verify` are
intentionally ignored by Git. Contributors can generate a fresh local report
with:

```bash
python run_reproduction.py verify --project-root /path/to/nvidia_project
```

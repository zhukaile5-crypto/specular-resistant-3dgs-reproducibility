#!/usr/bin/env python3
"""Write a compact, auditable summary of the August 2026 cross-scene checks.

This is deliberately a reporting utility, not a new evaluator.  It only reads
the frozen per-scene JSON files emitted by the 2-D structure and matched-orbit
evaluators, so that the summary never silently recomputes or changes a metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_2D_ROOT = PROJECT_ROOT / "results" / "2d_structure_fidelity_20260811"
DEFAULT_3D_ROOT = PROJECT_ROOT / "results" / "end_to_end_3dgs_cross_scene_20260811"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    header_row = list(headers)
    body = [list(row) for row in rows]
    separator = ["---"] * len(header_row)
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [header_row, separator, *body]
    )


def aggregate_supported_views(path: Path) -> Dict[str, float]:
    """Reaggregate additive metrics after excluding out-of-support orbit poses."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["within_support"] == "True"]
    if not rows:
        raise RuntimeError("No supported views in {}".format(path))

    def total(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return sum(value for value in values if math.isfinite(value))

    pixels = total("pixels")
    raw_score = sum(float(row["raw_mean_score"]) * float(row["pixels"]) for row in rows)
    deglared_score = sum(
        float(row["deglared_mean_score"]) * float(row["pixels"]) for row in rows
    )
    raw_any = sum(
        float(row["raw_any_percent"]) * float(row["pixels"]) / 100.0 for row in rows
    )
    deglared_any = sum(
        float(row["deglared_any_percent"]) * float(row["pixels"]) / 100.0 for row in rows
    )
    raw_strong = sum(
        float(row["raw_strong_percent"]) * float(row["pixels"]) / 100.0
        for row in rows
    )
    deglared_strong = sum(
        float(row["deglared_strong_percent"]) * float(row["pixels"]) / 100.0
        for row in rows
    )

    def reduction(before: float, after: float) -> float:
        return 100.0 * (before - after) / before if before > 0.0 else float("nan")

    source_pixels = total("source_reflection_pixels")
    source_score = total("source_reflection_score")
    source_strong = total("source_strong_pixels")
    return {
        "view_count": len(rows),
        "pixel_count": int(pixels),
        "raw_mean_score": raw_score / pixels,
        "deglared_mean_score": deglared_score / pixels,
        "source_region_attenuation_percent": reduction(
            source_score, total("residual_score_on_source_region")
        ),
        "broad_improvement_coverage_percent": 100.0
        * total("source_improved_pixels")
        / source_pixels,
        "strict_significant_coverage_percent": 100.0
        * total("source_strict_significant_pixels")
        / source_pixels,
        "strong_suppression_percent": 100.0
        * total("source_strong_suppressed_pixels")
        / source_strong,
        "mean_score_reduction_percent": reduction(raw_score, deglared_score),
        "any_area_reduction_percent": reduction(raw_any, deglared_any),
        "strong_area_reduction_percent": reduction(raw_strong, deglared_strong),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-d-root", type=Path, default=DEFAULT_2D_ROOT)
    parser.add_argument("--three-d-root", type=Path, default=DEFAULT_3D_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_3D_ROOT)
    args = parser.parse_args()

    two_d_scenes = [
        "photo_scene",
        "photo_scene2",
        "photo_scene3",
        "photo_scene4",
        "photo_scene5",
        "photo_scene6",
    ]
    three_d_scenes = ["photo_scene2", "photo_scene3", "photo_scene4", "photo_scene5"]

    two_d: Dict[str, Dict[str, Any]] = {}
    for scene in two_d_scenes:
        summary = read_json(args.two_d_root / scene / "summary.json")
        stats = summary["per_image_statistics"]
        two_d[scene] = {
            "frames": summary["frames"],
            "protected_percent_median": stats["protected_percent"]["median"],
            "fusion_delta_e_median": stats["fusion_cie76_delta_e_mean"]["median"],
            "direct_delta_e_median": stats["direct_cie76_delta_e_mean"]["median"],
            "fusion_highpass_corr_median": stats["fusion_highpass_correlation"]["median"],
            "direct_highpass_corr_median": stats["direct_highpass_correlation"]["median"],
            "fusion_edge_energy_median": stats["fusion_edge_energy_ratio"]["median"],
            "direct_edge_energy_median": stats["direct_edge_energy_ratio"]["median"],
            "fusion_to_direct_delta_e_ratio_median": stats[
                "fusion_to_direct_delta_e_ratio"
            ]["median"],
        }

    three_d: Dict[str, Dict[str, Any]] = {}
    for scene in three_d_scenes:
        root = args.three_d_root / scene
        summary = read_json(root / "matched_orbit_3x30" / "summary.json")
        support_only = aggregate_supported_views(
            root / "matched_orbit_3x30" / "per_view_metrics.csv"
        )
        raw_count = len(read_json(PROJECT_ROOT / "data" / "custom" / scene / "transforms.json")["frames"])
        deglared_count = len(
            read_json(
                PROJECT_ROOT
                / "data"
                / "custom"
                / f"{scene}_end2end_deglared_colmap_20260811"
                / "transforms.json"
            )["frames"]
        )
        overall = summary["overall"]
        three_d[scene] = {
            "raw_registered_frames": raw_count,
            "deglared_registered_frames": deglared_count,
            "within_support_views": summary["trajectory_support"]["within_support_count"],
            "view_count": overall["view_count"],
            "support_only": support_only,
            **overall,
        }

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "cross_scene_validation_summary.json").write_text(
        json.dumps(
            {
                "schema": "cross_scene_validation_summary_v1",
                "two_d_structure_fidelity": two_d,
                "three_d_end_to_end_matched_orbit": three_d,
                "notes": [
                    "2-D values are per-image medians on masks selected only from the original image; they quantify collateral change, not reflection-free accuracy.",
                    "3-D values compare raw-image COLMAP+3DGS against deglared-image reestimated-COLMAP+3DGS. Both image appearance and poses intentionally change.",
                    "The 90 views per scene are 3 heights x 30 directions of one matched orbit, not 90 independent scenes or held-out test samples.",
                    "All 3-D metrics use the frozen V3 single-view score as a proxy detector; they are not physical BRDF, reflectance, or geometry ground truth.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    all_two_d = [value for value in two_d.values()]
    all_three_d = [value for value in three_d.values()]
    two_d_rows = [
        [
            scene,
            str(value["frames"]),
            fmt(value["protected_percent_median"]),
            fmt(value["fusion_delta_e_median"]),
            fmt(value["direct_delta_e_median"]),
            fmt(value["fusion_highpass_corr_median"], 3),
            fmt(value["direct_highpass_corr_median"], 3),
            fmt(value["fusion_edge_energy_median"], 3),
            fmt(value["direct_edge_energy_median"], 3),
            fmt(value["fusion_to_direct_delta_e_ratio_median"], 3),
        ]
        for scene, value in two_d.items()
    ]
    three_d_rows = [
        [
            scene,
            f"{value['raw_registered_frames']}/{value['deglared_registered_frames']}",
            f"{value['within_support_views']}/{value['view_count']}",
            fmt(value["support_only"]["source_region_attenuation_percent"]),
            fmt(value["support_only"]["broad_improvement_coverage_percent"]),
            fmt(value["support_only"]["strict_significant_coverage_percent"]),
            fmt(value["support_only"]["strong_suppression_percent"]),
            fmt(value["support_only"]["mean_score_reduction_percent"]),
            fmt(value["support_only"]["any_area_reduction_percent"]),
            fmt(value["support_only"]["strong_area_reduction_percent"]),
        ]
        for scene, value in three_d.items()
    ]

    lines = [
        "# Cross-Scene Validation: Core Numerical Results",
        "",
        "This report contains numerical validation only; no manuscript files or presentation figures were changed.",
        "",
        "## Protocol",
        "",
        "- **2-D structural-fidelity check:** `photo_scene` through `photo_scene6` (561 source frames). The protected mask is selected from the original image only: low frozen-V3 specular score and high stable-structure penalty, with a protected-edge subset from the upper gradient quartile. It estimates collateral change in stable, non-specular-looking structure; it does **not** establish reflection-free restoration accuracy.",
        "- **3-D end-to-end check:** `photo_scene2` through `photo_scene5`. Each comparison is raw images → original COLMAP poses → 30k-step vanilla 3DGS versus final deglared images → newly estimated COLMAP poses → the same 30k-step vanilla 3DGS (same seed). All registered images are used in training; there is no held-out set or geometry ground truth.",
        "- **3-D readout:** both models are rendered at the same 3 heights × 30 orbit directions (90 paired poses). The frozen V3 score is a proxy detector. Consequently, views in one orbit are correlated and the scores should not be read as physical reflectance or independent-sample statistics.",
        "",
        "## 2-D Structural Fidelity",
        "",
        table(
            [
                "Scene",
                "Frames",
                "Protected area (%)",
                "Final ΔE76",
                "Direct ΔE76",
                "Final HF corr.",
                "Direct HF corr.",
                "Final edge energy",
                "Direct edge energy",
                "Final/direct ΔE",
            ],
            two_d_rows,
        ),
        "",
        "Each entry is the median over frames. `Final` is the structure-aware fused output; `Direct` is the unblended StableDelight correction candidate. High-pass correlation and edge-energy ratio are relative to the original (1.0 indicates unchanged local high-frequency energy). Across all six scenes, the unweighted mean of the per-image median final/direct ΔE ratio is "
        + fmt(mean([value["fusion_to_direct_delta_e_ratio_median"] for value in all_two_d]), 3)
        + "; this is descriptive, not a confidence interval.",
        "",
        "## End-to-End 3DGS Specular-Proxy Results",
        "",
        table(
            [
                "Scene",
                "Registered raw/deglared",
                "Orbit support",
                "Source evidence attenuation (%)",
                "Broad improvement coverage (%)",
                "Strict coverage (%)",
                "Strong suppression (%)",
                "Global score ↓ (%)",
                "Any-area ↓ (%)",
                "Strong-area ↓ (%)",
            ],
            three_d_rows,
        ),
        "",
        "Definitions: **source evidence attenuation** is the area-weighted reduction in frozen score inside pixels classified as weak-or-strong source reflections in the raw 3DGS render. **Broad coverage** is the share of those source pixels with any score decrease. **Strict coverage** additionally requires an absolute decrease ≥ 0.03 and a relative decrease ≥ 30%. **Strong suppression** is the share of source strong-reflection pixels that are no longer strong after the end-to-end pipeline. The table is reaggregated over only the trajectory poses satisfying the predefined camera-support bounds; a smaller global change can coexist with substantial source-region attenuation because the source regions are small.",
        "",
        "Across scene-level values (unweighted, descriptive mean): source evidence attenuation = "
        + fmt(mean([value["support_only"]["source_region_attenuation_percent"] for value in all_three_d]))
        + "%, broad improvement coverage = "
        + fmt(mean([value["support_only"]["broad_improvement_coverage_percent"] for value in all_three_d]))
        + "%, strict coverage = "
        + fmt(mean([value["support_only"]["strict_significant_coverage_percent"] for value in all_three_d]))
        + "%, and strong suppression = "
        + fmt(mean([value["support_only"]["strong_suppression_percent"] for value in all_three_d]))
        + "%. The four scenes are intentionally shown separately because their baseline reconstruction quality and proxy-score distributions differ.",
        "",
        "## Interpretation Boundaries",
        "",
        "- `photo_scene` is retained as a low-specularity 2-D control only; it was not used for the 3-DGS benefit comparison.",
        "- `photo_scene2` has 108 raw and 102 reestimated-pose registered frames. Its result is a valid complete-pipeline outcome, but it is not a same-frame-count controlled comparison. Scenes 3–5 retain equal counts (57/57, 71/71, and 94/94).",
        "- These are first cross-scene numbers using a frozen proxy evaluator. They support a case that the system often suppresses source-region specular evidence while keeping selected stable 2-D structure close to the original. They do not by themselves prove general geometry accuracy or causally isolate the contributions of deglared appearance and reestimated poses.",
        "",
        "## Source Files",
        "",
        "- `../2d_structure_fidelity_20260811/summary.json` and per-scene `summary.json`",
        "- `photo_scene{2,3,4,5}/matched_orbit_3x30/summary.json` and `per_view_metrics.csv`",
        "- `cross_scene_validation_summary.json` (machine-readable copy of the values above)",
        "",
    ]
    (output / "CROSS_SCENE_VALIDATION_CORE_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

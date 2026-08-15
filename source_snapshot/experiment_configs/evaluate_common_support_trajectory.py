#!/usr/bin/env python3
"""Render and score a geometry-only, jointly supported 3DGS trajectory.

The original cross-scene experiment fitted a regular three-height ellipse to
the raw COLMAP cameras.  That ellipse leaves the captured camera manifold in
some scenes.  This script instead constructs novel midpoint poses between
consecutive cameras registered by *both* the raw and specular-suppressed
branches.  Candidate selection uses camera geometry only; no rendered image or
reflection score is inspected.

The two COLMAP coordinate systems are aligned by Sim(3).  To avoid privileging
either reconstruction, corresponding raw poses and inverse-aligned suppressed
poses are first averaged in SE(3).  Consecutive symmetric anchors are then
interpolated at t=0.5.  Candidates must pass conservative support limits in
both branches before 90 samples are selected uniformly along the capture path.

Existing checkpoints are reused.  The script never trains a model and writes
to a new ``matched_common_support_trajectory_90`` directory by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
GSPLAT_ROOT = PROJECT_ROOT / "third_party/gsplat"
GSPLAT_EXAMPLES = GSPLAT_ROOT / "examples"
for path in (PROJECT_ROOT, GSPLAT_ROOT, GSPLAT_EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets.colmap import Parser  # noqa: E402
from experiment_configs.evaluate_matched_orbit_specularity import (  # noqa: E402
    STRONG_THRESHOLD,
    WEAK_THRESHOLD,
    _aggregate,
    _atomic_json,
    _frame_number,
    _median_camera_spacing,
    _pose_index,
    _relative_reduction,
    _score_pair,
    _similarity_between_camera_centers,
    _transform_pose,
    render_all,
)


DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / "results/end_to_end_3dgs_cross_scene_20260811"
)
DEFAULT_TRAJECTORY_NAME = "matched_common_support_trajectory_90"
PLANNING_DISTANCE_LIMIT = 2.25
PLANNING_ORIENTATION_LIMIT = 27.5
AUDIT_DISTANCE_LIMIT = 2.5
AUDIT_ORIENTATION_LIMIT = 30.0


def _full_pose(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3, :4] = np.asarray(pose, dtype=np.float32)[:3, :4]
    return result


def _interpolate_pose(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate translation linearly and rotation geodesically in SO(3)."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Interpolation fraction must lie in [0, 1]")
    first = _full_pose(first)
    second = _full_pose(second)
    result = np.eye(4, dtype=np.float32)
    result[:3, 3] = (
        (1.0 - fraction) * first[:3, 3] + fraction * second[:3, 3]
    )
    rotations = Rotation.from_matrix(
        np.stack([first[:3, :3], second[:3, :3]], axis=0)
    )
    result[:3, :3] = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
    return result


def _inverse_similarity_pose(
    pose: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Map a camera-to-world pose from target coordinates back to source."""
    pose = _full_pose(pose)
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rotation.T @ pose[:3, :3]
    result[:3, 3] = rotation.T @ ((pose[:3, 3] - translation) / scale)
    return result


def _support_measurement(
    pose: np.ndarray,
    training_poses: np.ndarray,
    training_frames: Sequence[int],
    median_spacing: float,
) -> Dict[str, Any]:
    centers = training_poses[:, :3, 3]
    distances = np.linalg.norm(centers - pose[:3, 3][None, :], axis=1)
    nearest = int(np.argmin(distances))
    cosine = float(
        np.clip(
            np.dot(pose[:3, 2], training_poses[nearest, :3, 2]),
            -1.0,
            1.0,
        )
    )
    orientation_degrees = float(math.degrees(math.acos(cosine)))
    distance_ratio = float(distances[nearest] / median_spacing)
    return {
        "nearest_training_frame": int(training_frames[nearest]),
        "nearest_center_distance": float(distances[nearest]),
        "distance_in_median_spacings": distance_ratio,
        "nearest_orientation_difference_degrees": orientation_degrees,
        "within_planning_support": bool(
            distance_ratio <= PLANNING_DISTANCE_LIMIT
            and orientation_degrees <= PLANNING_ORIENTATION_LIMIT
        ),
        "within_audit_support": bool(
            distance_ratio <= AUDIT_DISTANCE_LIMIT
            and orientation_degrees <= AUDIT_ORIENTATION_LIMIT
        ),
    }


def _uniform_indices(count: int, requested: int) -> List[int]:
    if requested < 1:
        raise ValueError("Requested view count must be positive")
    if count < requested:
        raise RuntimeError(
            "Only {} geometry-qualified candidates are available for {} views".format(
                count, requested
            )
        )
    indices = np.rint(np.linspace(0, count - 1, requested)).astype(np.int64)
    if len(set(indices.tolist())) != requested:
        raise RuntimeError("Uniform trajectory sampling produced duplicate candidates")
    return [int(index) for index in indices]


def _orientation_residual_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = second @ first.T
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def _azimuth_coverage_degrees(poses: Sequence[np.ndarray]) -> Dict[str, float]:
    centers = np.stack([pose[:3, 3] for pose in poses])
    origin = centers.mean(axis=0)
    angles = np.mod(
        np.degrees(np.arctan2(centers[:, 1] - origin[1], centers[:, 0] - origin[0])),
        360.0,
    )
    ordered = np.sort(angles)
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))
    maximum_gap = float(gaps.max())
    return {
        "approximate_azimuth_coverage": float(360.0 - maximum_gap),
        "maximum_uncovered_azimuth_gap": maximum_gap,
    }


def build_plan(
    raw_data: Path,
    deglared_data: Path,
    output_dir: Path,
    requested_views: int = 90,
) -> Dict[str, Any]:
    """Build and freeze a jointly supported trajectory without using CUDA."""
    manifest_path = output_dir / "trajectory_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing trajectory manifest: {}".format(
                manifest_path
            )
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to plan inside a non-empty output directory: {}".format(
                output_dir
            )
        )

    raw_parser = Parser(str(raw_data), factor=4, normalize=True, test_every=8)
    deglared_parser = Parser(
        str(deglared_data), factor=4, normalize=True, test_every=8
    )
    raw_indices = _pose_index(raw_parser)
    deglared_indices = _pose_index(deglared_parser)
    common_frames = sorted(set(raw_indices) & set(deglared_indices))
    if len(common_frames) < requested_views + 1:
        raise RuntimeError(
            "At least {} common cameras are required for {} midpoint views; found {}".format(
                requested_views + 1, requested_views, len(common_frames)
            )
        )

    raw_all = np.asarray(raw_parser.camtoworlds, dtype=np.float32)
    deglared_all = np.asarray(deglared_parser.camtoworlds, dtype=np.float32)
    raw_common = np.stack(
        [_full_pose(raw_all[raw_indices[frame]]) for frame in common_frames]
    )
    deglared_common = np.stack(
        [_full_pose(deglared_all[deglared_indices[frame]]) for frame in common_frames]
    )
    scale, rotation, translation = _similarity_between_camera_centers(
        raw_common[:, :3, 3], deglared_common[:, :3, 3]
    )
    mapped_raw_centers = (
        scale * (raw_common[:, :3, 3] @ rotation.T) + translation
    )
    alignment_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (mapped_raw_centers - deglared_common[:, :3, 3]) ** 2,
                    axis=1,
                )
            )
        )
    )
    orientation_residuals = [
        _orientation_residual_degrees(
            rotation @ raw_pose[:3, :3], deglared_pose[:3, :3]
        )
        for raw_pose, deglared_pose in zip(raw_common, deglared_common)
    ]

    raw_spacing = _median_camera_spacing(raw_common[:, :3, 3])
    deglared_spacing = _median_camera_spacing(deglared_common[:, :3, 3])
    deglared_in_raw = np.stack(
        [
            _inverse_similarity_pose(pose, scale, rotation, translation)
            for pose in deglared_common
        ]
    )
    symmetric_anchors = np.stack(
        [
            _interpolate_pose(raw_pose, mapped_pose, 0.5)
            for raw_pose, mapped_pose in zip(raw_common, deglared_in_raw)
        ]
    )

    candidates: List[Dict[str, Any]] = []
    for segment_index in range(len(common_frames) - 1):
        raw_pose = _interpolate_pose(
            symmetric_anchors[segment_index],
            symmetric_anchors[segment_index + 1],
            0.5,
        )
        deglared_pose = _transform_pose(raw_pose, scale, rotation, translation)
        raw_support = _support_measurement(
            raw_pose, raw_common, common_frames, raw_spacing
        )
        deglared_support = _support_measurement(
            deglared_pose, deglared_common, common_frames, deglared_spacing
        )
        candidates.append(
            {
                "candidate_index": segment_index + 1,
                "source_frame_start": int(common_frames[segment_index]),
                "source_frame_end": int(common_frames[segment_index + 1]),
                "source_frame_gap": int(
                    common_frames[segment_index + 1] - common_frames[segment_index]
                ),
                "raw_camera_to_world": raw_pose.tolist(),
                "deglared_camera_to_world": deglared_pose.tolist(),
                "raw_support": raw_support,
                "deglared_support": deglared_support,
                "within_planning_support": bool(
                    raw_support["within_planning_support"]
                    and deglared_support["within_planning_support"]
                ),
                "within_support": bool(
                    raw_support["within_audit_support"]
                    and deglared_support["within_audit_support"]
                ),
            }
        )

    eligible = [candidate for candidate in candidates if candidate["within_planning_support"]]
    selected = [eligible[index] for index in _uniform_indices(len(eligible), requested_views)]
    views: List[Dict[str, Any]] = []
    for sequence_index, candidate in enumerate(selected, start=1):
        view = dict(candidate)
        view["view_id"] = "p{:03d}".format(sequence_index)
        view["sequence_index"] = sequence_index
        view["path_fraction"] = (
            float(sequence_index - 1) / float(requested_views - 1)
            if requested_views > 1
            else 0.0
        )
        views.append(view)

    if len(views) != requested_views or not all(view["within_support"] for view in views):
        raise RuntimeError("Trajectory audit failed to produce {}/{} supported views".format(
            sum(view["within_support"] for view in views), requested_views
        ))

    raw_distance_max = max(
        view["raw_support"]["distance_in_median_spacings"] for view in views
    )
    deglared_distance_max = max(
        view["deglared_support"]["distance_in_median_spacings"] for view in views
    )
    raw_orientation_max = max(
        view["raw_support"]["nearest_orientation_difference_degrees"] for view in views
    )
    deglared_orientation_max = max(
        view["deglared_support"]["nearest_orientation_difference_degrees"]
        for view in views
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "schema": "matched_common_support_trajectory_v1",
        "trajectory_type": "symmetric consecutive-camera midpoint trajectory",
        "selection_uses_rendered_images_or_scores": False,
        "raw_data": str(raw_data),
        "deglared_data": str(deglared_data),
        "data_factor": 4,
        "common_camera_count": len(common_frames),
        "common_frames": common_frames,
        "candidate_count": len(candidates),
        "planning_qualified_candidate_count": len(eligible),
        "view_count": len(views),
        "support": {
            "support_camera_set": "common registered frames only",
            "planning_distance_limit_in_median_spacings": PLANNING_DISTANCE_LIMIT,
            "planning_orientation_limit_degrees": PLANNING_ORIENTATION_LIMIT,
            "audit_distance_limit_in_median_spacings": AUDIT_DISTANCE_LIMIT,
            "audit_orientation_limit_degrees": AUDIT_ORIENTATION_LIMIT,
            "raw_median_training_camera_spacing": raw_spacing,
            "deglared_median_training_camera_spacing": deglared_spacing,
            "within_support_count": len(views),
            "within_support_fraction": 1.0,
            "maximum_raw_distance_in_median_spacings": raw_distance_max,
            "maximum_deglared_distance_in_median_spacings": deglared_distance_max,
            "maximum_raw_orientation_difference_degrees": raw_orientation_max,
            "maximum_deglared_orientation_difference_degrees": deglared_orientation_max,
            "unique_nearest_raw_training_frames": len(
                {view["raw_support"]["nearest_training_frame"] for view in views}
            ),
            "unique_nearest_deglared_training_frames": len(
                {
                    view["deglared_support"]["nearest_training_frame"]
                    for view in views
                }
            ),
            **_azimuth_coverage_degrees(
                [np.asarray(view["raw_camera_to_world"], dtype=np.float32) for view in views]
            ),
        },
        "similarity_raw_to_deglared": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "camera_center_alignment_rmse": alignment_rmse,
            "camera_center_alignment_rmse_in_raw_spacings": alignment_rmse
            / raw_spacing,
            "orientation_residual_median_degrees": float(
                np.median(orientation_residuals)
            ),
            "orientation_residual_maximum_degrees": float(
                np.max(orientation_residuals)
            ),
        },
        "views": views,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _write_evaluation_readme(
    output_dir: Path, scene: str, manifest: Dict[str, Any], summary: Dict[str, Any]
) -> None:
    support = manifest["support"]
    overall = summary["overall"]
    lines = [
        "# {} matched common-support trajectory".format(scene),
        "",
        "This experiment reuses the completed raw-image and specular-suppressed 3DGS checkpoints. No COLMAP or 3DGS training was rerun.",
        "",
        "The 90 novel poses are SE(3) midpoints between consecutive symmetric camera anchors shared by both COLMAP branches. Candidate planning used camera geometry only and was frozen before rendering or reflection scoring.",
        "",
        "## Geometry audit",
        "",
        "- Common registered cameras: {}".format(manifest["common_camera_count"]),
        "- Planning-qualified midpoint candidates: {}/{}".format(
            manifest["planning_qualified_candidate_count"], manifest["candidate_count"]
        ),
        "- Jointly supported selected poses: {}/{}".format(
            support["within_support_count"], manifest["view_count"]
        ),
        "- Maximum raw/deglared distance: {:.3f}/{:.3f} median camera spacings".format(
            support["maximum_raw_distance_in_median_spacings"],
            support["maximum_deglared_distance_in_median_spacings"],
        ),
        "- Maximum raw/deglared orientation difference: {:.2f}/{:.2f} degrees".format(
            support["maximum_raw_orientation_difference_degrees"],
            support["maximum_deglared_orientation_difference_degrees"],
        ),
        "- Approximate azimuth coverage: {:.1f} degrees".format(
            support["approximate_azimuth_coverage"]
        ),
        "",
        "## Frozen proxy results",
        "",
        "- Whole-image mean-score reduction: {:.2f}%".format(
            overall["mean_score_reduction_percent"]
        ),
        "- Any-response area reduction: {:.2f}%".format(
            overall["any_area_reduction_percent"]
        ),
        "- Strong-response area reduction: {:.2f}%".format(
            overall["strong_area_reduction_percent"]
        ),
        "- Raw-mask conditional attenuation: {:.2f}%".format(
            overall["source_region_attenuation_percent"]
        ),
        "- Broad conditional coverage: {:.2f}%".format(
            overall["broad_improvement_coverage_percent"]
        ),
        "- Strict conditional coverage: {:.2f}%".format(
            overall["strict_significant_coverage_percent"]
        ),
        "- Strong-state suppression: {:.2f}%".format(
            overall["strong_suppression_percent"]
        ),
        "",
        "All reflection quantities are responses of the frozen same-family V3 single-view proxy. The 90 poses are correlated trajectory samples, not independent scenes. The experiment removes the previous trajectory-extrapolation confound but does not provide reflection-free or geometry ground truth.",
    ]
    destination = output_dir / "README.md"
    partial = output_dir / "README.partial.md"
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(partial, destination)


def evaluate(output_dir: Path, score_workers: int, scene: str) -> Dict[str, Any]:
    manifest_path = output_dir / "trajectory_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing frozen trajectory: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest["views"]
    if len(views) != 90 or not all(view["within_support"] for view in views):
        raise RuntimeError("Evaluation requires exactly 90 jointly supported poses")
    tasks: List[Tuple[str, str, str]] = [
        (
            view["view_id"],
            str(output_dir / "renders/raw_image_model" / (view["view_id"] + ".png")),
            str(
                output_dir
                / "renders/deglared_image_model"
                / (view["view_id"] + ".png")
            ),
        )
        for view in views
    ]
    for _, raw_path, deglared_path in tasks:
        if not Path(raw_path).is_file() or not Path(deglared_path).is_file():
            raise FileNotFoundError(
                "Missing paired render: {} / {}".format(raw_path, deglared_path)
            )
    if score_workers > 1:
        with ProcessPoolExecutor(max_workers=score_workers) as executor:
            scored = list(executor.map(_score_pair, tasks))
    else:
        scored = [_score_pair(task) for task in tasks]
    view_lookup = {view["view_id"]: view for view in views}
    rows: List[Dict[str, Any]] = []
    for score in scored:
        view = view_lookup[score["view_id"]]
        row = {
            "view_id": score["view_id"],
            "sequence_index": view["sequence_index"],
            "path_fraction": view["path_fraction"],
            "source_frame_start": view["source_frame_start"],
            "source_frame_end": view["source_frame_end"],
            "within_support": view["within_support"],
            **score,
        }
        row["mean_score_reduction_percent"] = _relative_reduction(
            row["raw_mean_score"], row["deglared_mean_score"]
        )
        row["any_area_reduction_percent"] = _relative_reduction(
            row["raw_any_percent"], row["deglared_any_percent"]
        )
        row["strong_area_reduction_percent"] = _relative_reduction(
            row["raw_strong_percent"], row["deglared_strong_percent"]
        )
        rows.append(row)
    rows.sort(key=lambda row: int(row["sequence_index"]))
    thirds = np.array_split(np.arange(len(rows)), 3)
    summary: Dict[str, Any] = {
        "schema": "matched_common_support_proxy_metrics_v1",
        "scene": scene,
        "evaluator": "frozen V3 single-view method-family proxy",
        "weak_threshold": WEAK_THRESHOLD,
        "strong_threshold": STRONG_THRESHOLD,
        "trajectory_support": manifest["support"],
        "overall": _aggregate(rows),
        "by_path_third": {
            str(index + 1): _aggregate([rows[int(row_index)] for row_index in indices])
            for index, indices in enumerate(thirds)
        },
        "uncertainty_note": (
            "Descriptive aggregation only. No direction bootstrap is reported because "
            "the 90 samples follow one correlated acquisition-path interpolation."
        ),
    }
    fieldnames = list(rows[0].keys())
    partial_csv = output_dir / "per_view_metrics.partial.csv"
    with partial_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial_csv, output_dir / "per_view_metrics.csv")
    _atomic_json(output_dir / "summary.json", summary)
    _write_evaluation_readme(output_dir, scene, manifest, summary)
    return summary


def _default_paths(
    scene: str, results_root: Path
) -> Tuple[Path, Path, Path, Path, Path]:
    scene_root = results_root / scene
    return (
        scene_root / "input_adapters/raw_image_colmap",
        scene_root / "input_adapters/deglared_reestimated_colmap",
        scene_root / "raw_image_colmap_3dgs/ckpts/ckpt_29999_rank0.pt",
        scene_root
        / "deglared_reestimated_colmap_3dgs/ckpts/ckpt_29999_rank0.pt",
        scene_root / DEFAULT_TRAJECTORY_NAME,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--stage", choices=("plan", "render", "evaluate", "all"), default="all"
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--raw-data", type=Path)
    parser.add_argument("--deglared-data", type=Path)
    parser.add_argument("--raw-checkpoint", type=Path)
    parser.add_argument("--deglared-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--views", type=int, default=90)
    parser.add_argument("--score-workers", type=int, default=3)
    parser.add_argument("--overwrite-renders", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    defaults = _default_paths(args.scene, args.results_root)
    raw_data = args.raw_data or defaults[0]
    deglared_data = args.deglared_data or defaults[1]
    raw_checkpoint = args.raw_checkpoint or defaults[2]
    deglared_checkpoint = args.deglared_checkpoint or defaults[3]
    output_dir = args.output_dir or defaults[4]
    for path, label in (
        (raw_data, "raw adapter"),
        (deglared_data, "suppressed adapter"),
        (raw_checkpoint, "raw checkpoint"),
        (deglared_checkpoint, "suppressed checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError("Missing {}: {}".format(label, path))

    if args.stage in ("plan", "all"):
        manifest = build_plan(raw_data, deglared_data, output_dir, args.views)
        print(json.dumps(manifest["support"], indent=2), flush=True)
    if args.stage in ("render", "all"):
        render_all(
            raw_data,
            deglared_data,
            raw_checkpoint,
            deglared_checkpoint,
            output_dir,
            args.overwrite_renders,
        )
    if args.stage in ("evaluate", "all"):
        summary = evaluate(output_dir, args.score_workers, args.scene)
        print(json.dumps(summary["overall"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

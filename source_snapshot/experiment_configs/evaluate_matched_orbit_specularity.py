#!/usr/bin/env python3
"""Render and evaluate matched 3DGS orbit views for photo_scene6.

The raw-image and deglared-image reconstructions live in independently
normalised COLMAP coordinate systems.  This tool first fits a Sim(3) transform
from all corresponding camera centres.  It then creates one three-height orbit
in the raw coordinate system and transforms the exact physical poses into the
deglared coordinate system before rendering.

Reflection scores are recomputed from each rendered RGB image with the frozen
V3 single-view detector.  No training-image score map, blend matte, or
cross-view map is reused for the synthetic views.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
GSPLAT_ROOT = PROJECT_ROOT / "third_party" / "gsplat"
GSPLAT_EXAMPLES = GSPLAT_ROOT / "examples"
STABLEDELIGHT_ROOT = PROJECT_ROOT / "StableDelight"
for path in (GSPLAT_ROOT, GSPLAT_EXAMPLES, STABLEDELIGHT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets.colmap import Parser  # noqa: E402
from datasets.traj import generate_ellipse_path_z  # noqa: E402
from stabledelight.utils.specular_detector_v3 import compute_specular_score  # noqa: E402


RAW_DATA = PROJECT_ROOT / "data/custom/photo_scene6_conventional_raw_colmap"
DEGLARED_DATA = PROJECT_ROOT / "data/custom/photo_scene6_v6_v7_colmap_reestimated_sequential"
RAW_CHECKPOINT = (
    PROJECT_ROOT / "results/photo_scene6/full96_novel_view/raw/ckpts/ckpt_29999_rank0.pt"
)
DEGLARED_CHECKPOINT = (
    PROJECT_ROOT
    / "results/photo_scene6/full96_reestimated_pose/v6_v7_deglared_reestimated_pose"
    / "ckpts/ckpt_29999_rank0.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results/photo_scene6/full96_reestimated_pose/matched_orbit_specularity_3x30"
)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
HEIGHT_PERCENTILES = (20.0, 50.0, 80.0)
WEAK_THRESHOLD = 0.075
STRONG_THRESHOLD = 0.16


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    partial = path.with_name(path.stem + ".partial.json")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, path)


def _frame_number(name: str) -> int:
    groups = re.findall(r"\d+", Path(name).stem)
    if not groups:
        raise ValueError("No numeric frame identifier in {}".format(name))
    return int(groups[-1])


def _pose_index(parser: Parser) -> Dict[int, int]:
    return {
        _frame_number(name): index for index, name in enumerate(parser.image_names)
    }


def _similarity_between_camera_centers(
    source_centers: np.ndarray, target_centers: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Fit target = scale * rotation * source + translation (Umeyama)."""
    source_mean = source_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_zero = source_centers - source_mean
    target_zero = target_centers - target_mean
    covariance = target_zero.T @ source_zero / len(source_centers)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    source_variance = np.mean(np.sum(source_zero ** 2, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _transform_pose(
    pose: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    transformed = np.eye(4, dtype=np.float32)
    transformed[:3, :3] = rotation @ pose[:3, :3]
    transformed[:3, 3] = scale * rotation @ pose[:3, 3] + translation
    return transformed


def _full_pose(pose_3x4: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3] = pose_3x4.astype(np.float32)
    return pose


def _median_camera_spacing(centers: np.ndarray) -> float:
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    distances += np.eye(len(centers)) * 1e9
    return float(np.median(np.min(distances, axis=1)))


def build_plan(
    raw_data: Path,
    deglared_data: Path,
    output_dir: Path,
    directions: int,
    height_percentiles: Sequence[float],
) -> Dict[str, Any]:
    """Create the aligned trajectory manifest without initialising CUDA."""
    if directions < 3:
        raise ValueError("At least three directions are required")
    if len(height_percentiles) != 3:
        raise ValueError("Exactly three height percentiles are required")
    raw_parser = Parser(str(raw_data), factor=4, normalize=True, test_every=8)
    deglared_parser = Parser(str(deglared_data), factor=4, normalize=True, test_every=8)
    raw_indices = _pose_index(raw_parser)
    deglared_indices = _pose_index(deglared_parser)
    common_frames = sorted(set(raw_indices) & set(deglared_indices))
    if len(common_frames) < 3:
        raise RuntimeError("Too few corresponding cameras for Sim(3) alignment")
    raw_centers = np.stack(
        [raw_parser.camtoworlds[raw_indices[frame], :3, 3] for frame in common_frames]
    )
    deglared_centers = np.stack(
        [deglared_parser.camtoworlds[deglared_indices[frame], :3, 3] for frame in common_frames]
    )
    scale, rotation, translation = _similarity_between_camera_centers(
        raw_centers, deglared_centers
    )
    aligned_centers = scale * (raw_centers @ rotation.T) + translation
    alignment_rmse = float(
        np.sqrt(np.mean(np.sum((aligned_centers - deglared_centers) ** 2, axis=1)))
    )

    raw_poses = np.asarray(raw_parser.camtoworlds)
    all_raw_centers = raw_poses[:, :3, 3]
    median_spacing = _median_camera_spacing(all_raw_centers)
    heights = np.percentile(all_raw_centers[:, 2], height_percentiles)
    views: List[Dict[str, Any]] = []
    for height_index, (percentile, height) in enumerate(
        zip(height_percentiles, heights), start=1
    ):
        orbit = generate_ellipse_path_z(
            raw_poses, n_frames=directions, height=float(height)
        )
        for direction_index, pose_3x4 in enumerate(orbit):
            raw_pose = _full_pose(pose_3x4)
            distances = np.linalg.norm(
                all_raw_centers - raw_pose[:3, 3][None, :], axis=1
            )
            nearest_index = int(np.argmin(distances))
            orientation_cosine = float(
                np.clip(
                    np.dot(
                        raw_pose[:3, 2], raw_poses[nearest_index, :3, 2]
                    ),
                    -1.0,
                    1.0,
                )
            )
            orientation_degrees = float(math.degrees(math.acos(orientation_cosine)))
            distance_ratio = float(distances[nearest_index] / median_spacing)
            view_id = "h{:02d}_d{:02d}".format(height_index, direction_index + 1)
            views.append(
                {
                    "view_id": view_id,
                    "height_index": height_index,
                    "height_percentile": float(percentile),
                    "height_coordinate": float(height),
                    "direction_index": direction_index + 1,
                    "azimuth_degrees": 360.0 * direction_index / directions,
                    "raw_camera_to_world": raw_pose.tolist(),
                    "deglared_camera_to_world": _transform_pose(
                        raw_pose, scale, rotation, translation
                    ).tolist(),
                    "nearest_training_frame": _frame_number(
                        raw_parser.image_names[nearest_index]
                    ),
                    "nearest_center_distance": float(distances[nearest_index]),
                    "distance_in_median_spacings": distance_ratio,
                    "nearest_orientation_difference_degrees": orientation_degrees,
                    "within_support": bool(
                        distance_ratio <= 2.5 and orientation_degrees <= 30.0
                    ),
                }
            )
    support_count = sum(view["within_support"] for view in views)
    manifest = {
        "schema": "matched_orbit_specularity_v1",
        "raw_data": str(raw_data),
        "deglared_data": str(deglared_data),
        "data_factor": 4,
        "common_camera_count": len(common_frames),
        "directions_per_height": directions,
        "height_percentiles": [float(value) for value in height_percentiles],
        "view_count": len(views),
        "support": {
            "median_training_camera_spacing": median_spacing,
            "distance_limit_in_median_spacings": 2.5,
            "orientation_limit_degrees": 30.0,
            "within_support_count": support_count,
            "within_support_fraction": support_count / float(len(views)),
            "maximum_distance_in_median_spacings": max(
                view["distance_in_median_spacings"] for view in views
            ),
            "maximum_orientation_difference_degrees": max(
                view["nearest_orientation_difference_degrees"] for view in views
            ),
        },
        "similarity_raw_to_deglared": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "camera_center_alignment_rmse": alignment_rmse,
        },
        "views": views,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "trajectory_manifest.json", manifest)
    return manifest


def _runner_config(data_dir: Path, result_dir: Path) -> Any:
    from simple_trainer import Config

    return Config(
        disable_viewer=True,
        data_dir=str(data_dir),
        data_factor=4,
        result_dir=str(result_dir),
        test_every=8,
        max_steps=30_000,
        save_steps=[30_000],
        eval_steps=[-1],
        init_type="sfm",
        pose_opt=False,
    )


def _load_checkpoint(runner: Any, checkpoint_path: Path) -> int:
    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location=runner.device, weights_only=True
    )
    for key in runner.splats.keys():
        runner.splats[key].data = checkpoint["splats"][key]
    return int(checkpoint["step"])


def _render_one(
    runner: Any, pose: np.ndarray, intrinsics: Any, width: int, height: int
) -> np.ndarray:
    import torch

    c2w = torch.from_numpy(pose).float().to(runner.device).unsqueeze(0)
    with torch.no_grad():
        colors, _, _ = runner.rasterize_splats(
            camtoworlds=c2w,
            Ks=intrinsics,
            width=width,
            height=height,
            sh_degree=runner.cfg.sh_degree,
            near_plane=runner.cfg.near_plane,
            far_plane=runner.cfg.far_plane,
            masks=None,
        )
    return (
        torch.clamp(colors[0, ..., :3], 0.0, 1.0).mul(255.0).byte().cpu().numpy()
    )


def _render_model(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    views: Sequence[Dict[str, Any]],
    pose_key: str,
    overwrite: bool,
) -> Dict[str, Any]:
    import torch
    from simple_trainer import Runner

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / (view["view_id"] + ".png") for view in views]
    if not overwrite and any(path.exists() for path in existing):
        raise FileExistsError(
            "Rendered files already exist under {}; pass --overwrite explicitly".format(
                output_dir
            )
        )
    with tempfile.TemporaryDirectory(prefix=".orbit_runtime_", dir=str(output_dir.parent)) as runtime:
        runner = Runner(0, 0, 1, _runner_config(data_dir, Path(runtime)))
        step = _load_checkpoint(runner, checkpoint_path)
        camera_id = runner.parser.camera_ids[0]
        intrinsics = (
            torch.from_numpy(runner.parser.Ks_dict[camera_id])
            .float()
            .to(runner.device)
            .unsqueeze(0)
        )
        width, height = runner.parser.imsize_dict[camera_id]
        for index, view in enumerate(views, start=1):
            print(
                "[render {}/{}] {} -> {}".format(
                    index, len(views), view["view_id"], output_dir.name
                ),
                flush=True,
            )
            image = _render_one(
                runner,
                np.asarray(view[pose_key], dtype=np.float32),
                intrinsics,
                int(width),
                int(height),
            )
            destination = output_dir / (view["view_id"] + ".png")
            partial = destination.with_name(destination.stem + ".partial.png")
            imageio.imwrite(partial, image)
            os.replace(partial, destination)
        if hasattr(runner, "writer"):
            runner.writer.close()
        del runner
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "width": int(width),
        "height": int(height),
        "render_count": len(views),
    }


def render_all(
    raw_data: Path,
    deglared_data: Path,
    raw_checkpoint: Path,
    deglared_checkpoint: Path,
    output_dir: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    manifest_path = output_dir / "trajectory_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run the plan stage first: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest["views"]
    render_manifest = {
        "raw": _render_model(
            raw_data,
            raw_checkpoint,
            output_dir / "renders/raw_image_model",
            views,
            "raw_camera_to_world",
            overwrite,
        ),
        "deglared": _render_model(
            deglared_data,
            deglared_checkpoint,
            output_dir / "renders/deglared_image_model",
            views,
            "deglared_camera_to_world",
            overwrite,
        ),
    }
    _atomic_json(output_dir / "render_manifest.json", render_manifest)
    return render_manifest


def _score_pair(task: Tuple[str, str, str]) -> Dict[str, Any]:
    view_id, raw_path_text, deglared_path_text = task
    raw_image = np.asarray(Image.open(raw_path_text).convert("RGB"))
    deglared_image = np.asarray(Image.open(deglared_path_text).convert("RGB"))
    raw_score = compute_specular_score(raw_image, delighted_image=None)
    deglared_score = compute_specular_score(deglared_image, delighted_image=None)
    raw_weak = raw_score >= WEAK_THRESHOLD
    raw_strong = raw_score >= STRONG_THRESHOLD
    deglared_weak = deglared_score >= WEAK_THRESHOLD
    deglared_strong = deglared_score >= STRONG_THRESHOLD
    score_drop = raw_score - deglared_score
    broad_improvement = raw_weak & (score_drop > 0.0)
    strict_significant_improvement = raw_weak & (score_drop >= 0.03) & (
        score_drop >= 0.30 * raw_score
    )
    strong_suppression = raw_strong & ~deglared_strong
    source_score = float(raw_score[raw_weak].sum())
    residual_score = float(deglared_score[raw_weak].sum())
    pixels = float(raw_score.size)
    return {
        "view_id": view_id,
        "pixels": int(pixels),
        "raw_mean_score": float(raw_score.mean()),
        "deglared_mean_score": float(deglared_score.mean()),
        "raw_p95_score": float(np.percentile(raw_score, 95.0)),
        "deglared_p95_score": float(np.percentile(deglared_score, 95.0)),
        "raw_any_percent": 100.0 * float(raw_weak.sum()) / pixels,
        "deglared_any_percent": 100.0 * float(deglared_weak.sum()) / pixels,
        "raw_strong_percent": 100.0 * float(raw_strong.sum()) / pixels,
        "deglared_strong_percent": 100.0 * float(deglared_strong.sum()) / pixels,
        "source_reflection_pixels": int(raw_weak.sum()),
        "source_improved_pixels": int(broad_improvement.sum()),
        "source_strict_significant_pixels": int(
            strict_significant_improvement.sum()
        ),
        "source_strong_pixels": int(raw_strong.sum()),
        "source_strong_suppressed_pixels": int(strong_suppression.sum()),
        "source_reflection_score": source_score,
        "residual_score_on_source_region": residual_score,
        "source_region_attenuation_percent": (
            100.0 * (source_score - residual_score) / source_score
            if source_score > 0.0
            else float("nan")
        ),
        "raw_mean_luminance": float(
            np.mean(raw_image[..., 0] * 0.2126 + raw_image[..., 1] * 0.7152 + raw_image[..., 2] * 0.0722)
            / 255.0
        ),
        "deglared_mean_luminance": float(
            np.mean(
                deglared_image[..., 0] * 0.2126
                + deglared_image[..., 1] * 0.7152
                + deglared_image[..., 2] * 0.0722
            )
            / 255.0
        ),
    }


def _relative_reduction(before: float, after: float) -> float:
    return 100.0 * (before - after) / before if before > 0.0 else float("nan")


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pixels = float(sum(row["pixels"] for row in rows))
    source_score = sum(row["source_reflection_score"] for row in rows)
    residual_score = sum(row["residual_score_on_source_region"] for row in rows)
    source_pixels = sum(row["source_reflection_pixels"] for row in rows)
    source_improved = sum(row["source_improved_pixels"] for row in rows)
    source_strict = sum(
        row["source_strict_significant_pixels"] for row in rows
    )
    source_strong = sum(row["source_strong_pixels"] for row in rows)
    source_strong_suppressed = sum(
        row["source_strong_suppressed_pixels"] for row in rows
    )
    result: Dict[str, Any] = {
        "view_count": len(rows),
        "pixel_count": int(pixels),
        "raw_mean_score": sum(row["raw_mean_score"] * row["pixels"] for row in rows) / pixels,
        "deglared_mean_score": sum(
            row["deglared_mean_score"] * row["pixels"] for row in rows
        )
        / pixels,
        "raw_any_percent": sum(row["raw_any_percent"] * row["pixels"] for row in rows) / pixels,
        "deglared_any_percent": sum(
            row["deglared_any_percent"] * row["pixels"] for row in rows
        )
        / pixels,
        "raw_strong_percent": sum(
            row["raw_strong_percent"] * row["pixels"] for row in rows
        )
        / pixels,
        "deglared_strong_percent": sum(
            row["deglared_strong_percent"] * row["pixels"] for row in rows
        )
        / pixels,
        "source_region_attenuation_percent": (
            100.0 * (source_score - residual_score) / source_score
            if source_score > 0.0
            else float("nan")
        ),
        "broad_improvement_coverage_percent": (
            100.0 * source_improved / source_pixels
            if source_pixels > 0
            else float("nan")
        ),
        "strict_significant_coverage_percent": (
            100.0 * source_strict / source_pixels
            if source_pixels > 0
            else float("nan")
        ),
        "strong_suppression_percent": (
            100.0 * source_strong_suppressed / source_strong
            if source_strong > 0
            else float("nan")
        ),
        "views_with_lower_mean_score_percent": 100.0
        * sum(row["deglared_mean_score"] < row["raw_mean_score"] for row in rows)
        / len(rows),
        "views_with_lower_any_area_percent": 100.0
        * sum(row["deglared_any_percent"] < row["raw_any_percent"] for row in rows)
        / len(rows),
        "views_with_lower_strong_area_percent": 100.0
        * sum(row["deglared_strong_percent"] < row["raw_strong_percent"] for row in rows)
        / len(rows),
    }
    result["mean_score_reduction_percent"] = _relative_reduction(
        result["raw_mean_score"], result["deglared_mean_score"]
    )
    result["any_area_reduction_percent"] = _relative_reduction(
        result["raw_any_percent"], result["deglared_any_percent"]
    )
    result["strong_area_reduction_percent"] = _relative_reduction(
        result["raw_strong_percent"], result["deglared_strong_percent"]
    )
    return result


def _direction_cluster_bootstrap(
    rows: Sequence[Dict[str, Any]], samples: int = 10_000, seed: int = 20260808
) -> Dict[str, List[float]]:
    """Bootstrap paired reductions while keeping the three heights together."""
    directions = sorted(set(int(row["direction_index"]) for row in rows))
    group_columns: List[List[float]] = []
    for direction in directions:
        selected = [row for row in rows if int(row["direction_index"]) == direction]
        group_columns.append(
            [
                sum(row["raw_mean_score"] * row["pixels"] for row in selected),
                sum(row["deglared_mean_score"] * row["pixels"] for row in selected),
                sum(row["raw_any_percent"] * row["pixels"] for row in selected),
                sum(row["deglared_any_percent"] * row["pixels"] for row in selected),
                sum(row["raw_strong_percent"] * row["pixels"] for row in selected),
                sum(row["deglared_strong_percent"] * row["pixels"] for row in selected),
                sum(row["source_reflection_score"] for row in selected),
                sum(row["residual_score_on_source_region"] for row in selected),
                sum(row["source_reflection_pixels"] for row in selected),
                sum(row["source_improved_pixels"] for row in selected),
                sum(row["source_strict_significant_pixels"] for row in selected),
                sum(row["source_strong_pixels"] for row in selected),
                sum(row["source_strong_suppressed_pixels"] for row in selected),
            ]
        )
    groups = np.asarray(group_columns, dtype=np.float64)
    rng = np.random.RandomState(seed)
    selections = rng.randint(0, len(groups), size=(samples, len(groups)))
    totals = groups[selections].sum(axis=1)

    def interval(before_column: int, after_column: int) -> List[float]:
        before = totals[:, before_column]
        values = 100.0 * (before - totals[:, after_column]) / np.maximum(before, 1e-12)
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    def coverage_interval(numerator_column: int, denominator_column: int) -> List[float]:
        denominator = totals[:, denominator_column]
        values = 100.0 * totals[:, numerator_column] / np.maximum(
            denominator, 1e-12
        )
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    return {
        "mean_score_reduction_percent": interval(0, 1),
        "any_area_reduction_percent": interval(2, 3),
        "strong_area_reduction_percent": interval(4, 5),
        "source_region_attenuation_percent": interval(6, 7),
        "broad_improvement_coverage_percent": coverage_interval(9, 8),
        "strict_significant_coverage_percent": coverage_interval(10, 8),
        "strong_suppression_percent": coverage_interval(12, 11),
    }


def _save_contact_sheet(
    views: Sequence[Dict[str, Any]], output_dir: Path, height_index: int
) -> None:
    selected = [view for view in views if view["height_index"] == height_index]
    groups = [selected[index : index + 10] for index in range(0, len(selected), 10)]
    first = Image.open(
        output_dir / "renders/raw_image_model" / (selected[0]["view_id"] + ".png")
    ).convert("RGB")
    tile_width = 280
    tile_height = int(round(first.height * tile_width / first.width))
    label_height = 34
    canvas = Image.new(
        "RGB", (tile_width * 10, (tile_height + label_height) * 2 * len(groups)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 19)
    for group_index, group in enumerate(groups):
        for row_index, (folder, label) in enumerate(
            (("raw_image_model", "Raw-trained"), ("deglared_image_model", "Deglared-trained"))
        ):
            y = (group_index * 2 + row_index) * (tile_height + label_height)
            for column, view in enumerate(group):
                x = column * tile_width
                image = Image.open(
                    output_dir / "renders" / folder / (view["view_id"] + ".png")
                ).convert("RGB")
                image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (tile_width, tile_height), "black")
                tile.paste(image, ((tile_width - image.width) // 2, 0))
                canvas.paste(tile, (x, y + label_height))
                draw.rectangle((x, y, x + tile_width, y + label_height), fill=(32, 32, 32))
                draw.text(
                    (x + 7, y + 6),
                    "{} | {:03.0f} deg".format(label, view["azimuth_degrees"]),
                    font=font,
                    fill="white",
                )
    figures = output_dir / "paper_figures"
    figures.mkdir(parents=True, exist_ok=True)
    canvas.save(figures / "height_{:02d}_paired_contact_sheet.jpg".format(height_index), quality=94)


def _save_academic_figure(
    rows: Sequence[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "evidence": "#3E6E8C",
        "strict": "#5E8C61",
        "strong": "#B26055",
        "broad": "#C3933B",
        "global": "#777777",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    overall = summary["overall"]
    confidence = summary["direction_cluster_bootstrap_95_ci"]
    primary_keys = (
        "source_region_attenuation_percent",
        "strict_significant_coverage_percent",
        "strong_suppression_percent",
        "broad_improvement_coverage_percent",
    )
    primary_labels = (
        "Source evidence\nattenuation",
        "Strict coverage\n(≥30% & ≥0.03)",
        "Strong-state\nsuppression",
        "Broad coverage\n(any decrease)",
    )
    primary_values = [overall[key] for key in primary_keys]
    primary_intervals = [confidence[key] for key in primary_keys]
    lower_error = [
        value - interval[0]
        for value, interval in zip(primary_values, primary_intervals)
    ]
    upper_error = [
        interval[1] - value
        for value, interval in zip(primary_values, primary_intervals)
    ]
    x = np.arange(len(primary_values))
    axes[0, 0].bar(
        x,
        primary_values,
        yerr=np.asarray([lower_error, upper_error]),
        capsize=5,
        color=[
            colors["evidence"],
            colors["strict"],
            colors["strong"],
            colors["broad"],
        ],
        edgecolor="white",
        linewidth=0.8,
    )
    for position, value in zip(x, primary_values):
        axes[0, 0].text(
            position,
            value + 3.0,
            "{:.1f}%".format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0, 0].set_xticks(x, primary_labels)
    axes[0, 0].set_ylim(0.0, 86.0)
    axes[0, 0].set_xlabel("Source-region metric")
    axes[0, 0].set_ylabel("Suppression metric (%)")
    axes[0, 0].set_title("(a) Source-region reflection suppression")
    axes[0, 0].grid(axis="y", alpha=0.25)

    height_keys = sorted(summary["by_height"], key=int)
    evidence_by_height = [
        summary["by_height"][key]["source_region_attenuation_percent"]
        for key in height_keys
    ]
    strict_by_height = [
        summary["by_height"][key]["strict_significant_coverage_percent"]
        for key in height_keys
    ]
    strong_by_height = [
        summary["by_height"][key]["strong_suppression_percent"]
        for key in height_keys
    ]
    hx = np.arange(3)
    axes[0, 1].bar(
        hx - 0.25,
        evidence_by_height,
        0.25,
        label="Evidence attenuation",
        color=colors["evidence"],
    )
    axes[0, 1].bar(
        hx,
        strict_by_height,
        0.25,
        label="Strict coverage",
        color=colors["strict"],
    )
    axes[0, 1].bar(
        hx + 0.25,
        strong_by_height,
        0.25,
        label="Strong suppression",
        color=colors["strong"],
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(hx, ["Low", "Middle", "High"])
    axes[0, 1].set_xlabel("Camera-height level")
    axes[0, 1].set_ylabel("Suppression metric (%)")
    axes[0, 1].set_title("(b) Source-region consistency across heights")
    axes[0, 1].legend(frameon=False, ncol=3, fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.25)

    direction_rows = []
    for direction_index in range(1, 31):
        selected = [row for row in rows if row["direction_index"] == direction_index]
        aggregated = _aggregate(selected)
        aggregated["azimuth_degrees"] = selected[0]["azimuth_degrees"]
        direction_rows.append(aggregated)
    azimuths = [row["azimuth_degrees"] for row in direction_rows]
    direction_evidence = [
        row["source_region_attenuation_percent"] for row in direction_rows
    ]
    axes[1, 0].plot(
        azimuths,
        direction_evidence,
        color=colors["evidence"],
        linewidth=2.0,
        marker="o",
        markersize=3.2,
        label="Height-aggregated direction",
    )
    axes[1, 0].axhline(
        overall["source_region_attenuation_percent"],
        color=colors["global"],
        linestyle="--",
        linewidth=1.2,
        label="Overall {:.1f}%".format(
            overall["source_region_attenuation_percent"]
        ),
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("Orbit azimuth (degrees)")
    axes[1, 0].set_ylabel("Evidence attenuation (%)")
    axes[1, 0].set_title("(c) Source-region attenuation by direction")
    axes[1, 0].set_xlim(0.0, 348.0)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False)

    conservative_keys = (
        "mean_score_reduction_percent",
        "any_area_reduction_percent",
        "strong_area_reduction_percent",
    )
    conservative_labels = (
        "Whole-image\nmean score",
        "Net detected\narea",
        "Net strong\narea",
    )
    conservative_values = [overall[key] for key in conservative_keys]
    conservative_intervals = [confidence[key] for key in conservative_keys]
    conservative_lower = [
        value - interval[0]
        for value, interval in zip(conservative_values, conservative_intervals)
    ]
    conservative_upper = [
        interval[1] - value
        for value, interval in zip(conservative_values, conservative_intervals)
    ]
    cx = np.arange(3)
    axes[1, 1].bar(
        cx,
        conservative_values,
        yerr=np.asarray([conservative_lower, conservative_upper]),
        capsize=5,
        color=["#7B8794", "#8E9A82", "#9B7E70"],
        edgecolor="white",
        linewidth=0.8,
    )
    for position, value in zip(cx, conservative_values):
        axes[1, 1].text(
            position,
            value + 2.0,
            "{:.1f}%".format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1, 1].set_xticks(cx, conservative_labels)
    axes[1, 1].set_ylim(0.0, 36.0)
    axes[1, 1].set_xlabel("Whole-image metric")
    axes[1, 1].set_ylabel("Net relative reduction (%)")
    axes[1, 1].set_title("(d) Conservative whole-image net effects")
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Reflection suppression in matched 3DGS novel views\n"
        "Frozen structure-aware evaluator; 3 heights x 30 directions; "
        "error bars: direction-cluster bootstrap 95% CI",
        fontsize=14,
    )
    figures = output_dir / "paper_figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "matched_orbit_reflection_evaluation.png", dpi=240)
    figure.savefig(figures / "matched_orbit_reflection_evaluation.pdf")
    plt.close(figure)


def redraw_academic_figure(output_dir: Path) -> None:
    """Rebuild only the paper figure from the frozen CSV/JSON evaluation."""
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "per_view_metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            "Existing summary.json and per_view_metrics.csv are required for figure-only output"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for source_row in csv.DictReader(handle):
            row: Dict[str, Any] = {}
            for key, value in source_row.items():
                if key == "view_id":
                    row[key] = value
                elif key == "within_support":
                    row[key] = value.lower() == "true"
                else:
                    row[key] = float(value)
            rows.append(row)
    if len(rows) != 90:
        raise RuntimeError("Expected the complete 90-view evaluation table")
    _save_academic_figure(rows, summary, output_dir)


def _save_readme(
    output_dir: Path, summary: Dict[str, Any], scene_name: str
) -> None:
    overall = summary["overall"]
    lines = [
        "# {} aligned-orbit reflection evaluation".format(scene_name),
        "",
        "The raw-trained and deglared-trained models use the same physical observation poses: "
        "three in-distribution camera heights and 30 azimuths per height, for 90 paired renders. "
        "The two COLMAP coordinate systems are aligned by fitting Sim(3) to 96 corresponding camera centers.",
        "",
        "Synthetic renders are rescored with the frozen V3 single-view detector. No training-image "
        "cross-view map, score map, or fusion weight is reused. The result therefore measures proxy "
        "reflection response in reconstructed renders, not physical optical intensity.",
        "",
        "## Summary",
        "",
        "- Relative mean-score change: {:.2f}%".format(overall["mean_score_reduction_percent"]),
        "- Relative any-response area change: {:.2f}%".format(overall["any_area_reduction_percent"]),
        "- Relative strong-response area change: {:.2f}%".format(overall["strong_area_reduction_percent"]),
        "- Source-region evidence attenuation: {:.2f}%".format(overall["source_region_attenuation_percent"]),
        "- Strict improvement coverage (relative drop ≥30% and absolute drop ≥0.03): {:.2f}%".format(
            overall["strict_significant_coverage_percent"]
        ),
        "- Strong-response suppression (strong becomes weak or normal): {:.2f}%".format(
            overall["strong_suppression_percent"]
        ),
        "- Broad improvement coverage (score drop > 0): {:.2f}%".format(
            overall["broad_improvement_coverage_percent"]
        ),
        "- Views with a lower mean score among the 90 poses: {:.2f}%".format(overall["views_with_lower_mean_score_percent"]),
        "- Direction-cluster bootstrap 95% interval for mean-score reduction: [{:.2f}%, {:.2f}%]".format(
            *summary["direction_cluster_bootstrap_95_ci"]["mean_score_reduction_percent"]
        ),
        "",
        "Relative change is computed as `1 - deglared / raw`; positive values indicate a lower "
        "response for the deglared model, and negative values indicate an increase. Complete "
        "per-view values are in `per_view_metrics.csv`; alignment and trajectory audits are in "
        "`trajectory_manifest.json`.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    output_dir: Path,
    score_workers: int,
    *,
    save_figures: bool = True,
    save_contact_sheets: bool = True,
    scene_name: str = "scene",
) -> Dict[str, Any]:
    manifest = json.loads(
        (output_dir / "trajectory_manifest.json").read_text(encoding="utf-8")
    )
    views = manifest["views"]
    tasks = [
        (
            view["view_id"],
            str(output_dir / "renders/raw_image_model" / (view["view_id"] + ".png")),
            str(output_dir / "renders/deglared_image_model" / (view["view_id"] + ".png")),
        )
        for view in views
    ]
    for _, raw_path, deglared_path in tasks:
        if not Path(raw_path).is_file() or not Path(deglared_path).is_file():
            raise FileNotFoundError("Missing paired render: {} / {}".format(raw_path, deglared_path))
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
            "height_index": view["height_index"],
            "height_percentile": view["height_percentile"],
            "direction_index": view["direction_index"],
            "azimuth_degrees": view["azimuth_degrees"],
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
    summary = {
        "schema": "matched_orbit_specularity_metrics_v1",
        "evaluator": "frozen V3 single-view detector",
        "weak_threshold": WEAK_THRESHOLD,
        "strong_threshold": STRONG_THRESHOLD,
        "trajectory_support": manifest["support"],
        "overall": _aggregate(rows),
        "direction_cluster_bootstrap_95_ci": _direction_cluster_bootstrap(rows),
        "by_height": {
            str(height_index): _aggregate(
                [row for row in rows if row["height_index"] == height_index]
            )
            for height_index in (1, 2, 3)
        },
    }
    fieldnames = list(rows[0].keys())
    partial_csv = output_dir / "per_view_metrics.partial.csv"
    with partial_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial_csv, output_dir / "per_view_metrics.csv")
    _atomic_json(output_dir / "summary.json", summary)
    if save_figures:
        _save_academic_figure(rows, summary, output_dir)
    if save_contact_sheets:
        for height_index in (1, 2, 3):
            _save_contact_sheet(views, output_dir, height_index)
    _save_readme(output_dir, summary, scene_name)
    return summary


def _parse_height_percentiles(values: Sequence[float]) -> Tuple[float, float, float]:
    if len(values) != 3 or any(value < 0.0 or value > 100.0 for value in values):
        raise argparse.ArgumentTypeError("Require three percentiles in [0, 100]")
    return float(values[0]), float(values[1]), float(values[2])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("plan", "render", "evaluate", "figure", "all"), default="all"
    )
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA)
    parser.add_argument("--deglared-data", type=Path, default=DEGLARED_DATA)
    parser.add_argument("--raw-checkpoint", type=Path, default=RAW_CHECKPOINT)
    parser.add_argument("--deglared-checkpoint", type=Path, default=DEGLARED_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--directions", type=int, default=30)
    parser.add_argument(
        "--height-percentiles", type=float, nargs=3, default=HEIGHT_PERCENTILES
    )
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write only numerical metrics, manifests, and paired render images.",
    )
    parser.add_argument(
        "--no-contact-sheets",
        action="store_true",
        help="Do not assemble rendered images into contact sheets.",
    )
    parser.add_argument(
        "--scene-name",
        default="scene",
        help="Scene label recorded in the numerical evaluation README.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    height_percentiles = _parse_height_percentiles(args.height_percentiles)
    if args.stage in ("plan", "all"):
        plan = build_plan(
            args.raw_data,
            args.deglared_data,
            args.output_dir,
            args.directions,
            height_percentiles,
        )
        print(json.dumps(plan["support"], indent=2))
    if args.stage in ("render", "all"):
        render_all(
            args.raw_data,
            args.deglared_data,
            args.raw_checkpoint,
            args.deglared_checkpoint,
            args.output_dir,
            args.overwrite,
        )
    if args.stage in ("evaluate", "all"):
        summary = evaluate(
            args.output_dir,
            args.score_workers,
            save_figures=not args.no_figures,
            save_contact_sheets=not args.no_contact_sheets,
            scene_name=args.scene_name,
        )
        print(json.dumps(summary["overall"], indent=2))
    if args.stage == "figure":
        redraw_academic_figure(args.output_dir)
        print(args.output_dir / "paper_figures/matched_orbit_reflection_evaluation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

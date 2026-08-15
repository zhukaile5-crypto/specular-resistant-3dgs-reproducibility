#!/usr/bin/env python3
"""Generate V3 per-observation cross-view specular evidence maps.

Unlike V2's per-point variation, V3 assigns a separate positive photometric
residual to every point observation.  A point that becomes bright in one view
therefore raises only that frame's map.  A companion confidence map records
where well-tracked, photometrically stable structure can safely downweight a
single-frame edge response.

Run in the ``nerfstudio`` environment::

    python experiment_configs/cross_view_specular_v3.py \
        --scene photo_scene6 \
        --colmap-dir data/custom/photo_scene6 \
        --output-dir data/custom/photo_scene6_crossview_maps_v3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
DATA_CUSTOM = PROJECT_ROOT / "data" / "custom"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_configs.cross_view_specular import (
    _dense_from_sparse,
    _find_frame_image,
    _load_rgb,
    _lum,
    _patch_mean,
    _robust_affine_normalize,
    _sample_at,
    load_colmap_model,
)

CROSSVIEW_ALGORITHM_VERSION = "observation_residual_confidence_v3"


def _point_index(manager) -> np.ndarray:
    max_id = int(manager.point3D_ids.max())
    index = np.full(max_id + 1, -1, dtype=np.int64)
    index[manager.point3D_ids.astype(np.int64)] = np.arange(
        len(manager.points3D)
    )
    return index


def sample_track_observations(
    manager,
    frames,
    colmap_dir: Path,
    *,
    patch_radius: int = 3,
    broad_patch_radius: int = 9,
) -> np.ndarray:
    """Sample multi-scale patch luminance into a frame-by-point matrix."""
    n_points = len(manager.points3D)
    point_index = _point_index(manager)
    observations = np.full(
        (len(frames), n_points), np.nan, dtype=np.float64
    )

    broad_patch_radius = max(int(broad_patch_radius), int(patch_radius))
    for frame_index, frame in enumerate(frames):
        valid_observation = frame.point3D_ids.astype(np.int64) >= 0
        if not valid_observation.any():
            continue
        point_ids = point_index[
            frame.point3D_ids[valid_observation].astype(np.int64)
        ]
        coordinates = frame.points2D[valid_observation]
        try:
            image = _load_rgb(_find_frame_image(colmap_dir, frame.name))
        except FileNotFoundError:
            continue
        gray = _lum(image)
        fine = _patch_mean(gray, patch_radius)
        broad = _patch_mean(gray, broad_patch_radius)
        camera = manager.cameras[frame.camera_id]
        scale = np.array(
            [gray.shape[1] / camera.width, gray.shape[0] / camera.height]
        )
        scaled = coordinates * scale
        # Broad sampling suppresses ribbed texture and sub-pixel edge jitter;
        # retaining a fine contribution preserves compact highlights.
        values = 0.35 * _sample_at(fine, scaled) + 0.65 * _sample_at(
            broad, scaled
        )
        observations[frame_index, point_ids] = values
    return observations


def compute_positive_observation_residuals(
    aligned_observations: np.ndarray,
    *,
    min_views: int = 4,
    noise_floor: float = 0.01,
    mad_scale: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-observation positive residual and per-point stability.

    Returns ``(residual, stability, n_views, usable)``.  Missing observations
    remain NaN, while finite non-specular observations receive residual zero.
    """
    observations = aligned_observations.astype(np.float64, copy=False)
    finite = np.isfinite(observations)
    n_views = finite.sum(axis=0)
    usable = n_views >= int(min_views)

    with np.errstate(invalid="ignore"):
        median = np.nanmedian(observations, axis=0)
        mad = np.nanmedian(np.abs(observations - median[None, :]), axis=0)
        upper = np.nanpercentile(observations, 90, axis=0)
    robust_sigma = 1.4826 * mad
    threshold = np.maximum(
        float(noise_floor), float(noise_floor) + mad_scale * robust_sigma
    )

    residual = observations - median[None, :] - threshold[None, :]
    residual = np.maximum(residual, 0.0)
    residual[~finite] = np.nan
    residual[:, ~usable] = np.nan

    positive_variation = np.maximum(upper - median, 0.0)
    stability_scale = np.maximum(3.0 * robust_sigma + noise_floor, 0.02)
    stability = np.exp(-positive_variation / stability_scale)
    view_confidence = np.clip(
        (n_views - int(min_views) + 1) / 4.0, 0.0, 1.0
    )
    stability = np.clip(stability * view_confidence, 0.0, 1.0)
    stability[~usable] = 0.0
    return residual, stability, n_views, usable


def generate_dense_observation_maps(
    manager,
    frame,
    frame_index: int,
    residual: np.ndarray,
    stability: np.ndarray,
    n_views: np.ndarray,
    usable: np.ndarray,
    colmap_dir: Path,
    *,
    blur_sigma: float = 6.0,
    norm_scale: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project V3 residual and stability values into one registered frame."""
    image = _load_rgb(_find_frame_image(colmap_dir, frame.name))
    empty = np.zeros(image.shape[:2], dtype=np.float32)
    observed = frame.point3D_ids.astype(np.int64) >= 0
    if not observed.any():
        return empty, empty.copy()

    point_index = _point_index(manager)
    point_ids = point_index[frame.point3D_ids[observed].astype(np.int64)]
    values = residual[frame_index, point_ids]
    keep = usable[point_ids] & np.isfinite(values)
    if not keep.any():
        return empty, empty.copy()

    point_ids = point_ids[keep]
    values = values[keep]
    coordinates = frame.points2D[observed][keep]
    camera = manager.cameras[frame.camera_id]
    scale = np.array(
        [image.shape[1] / camera.width, image.shape[0] / camera.height]
    )
    coordinates = coordinates * scale

    weights = np.sqrt(np.minimum(n_views[point_ids], 10.0)).astype(np.float64)
    gray = _lum(image)
    grad_y, grad_x = np.gradient(_patch_mean(gray, 1))
    gradient = np.hypot(grad_x, grad_y)
    gradient_scale = float(np.percentile(gradient, 95))
    if gradient_scale > 1e-8:
        edge = np.clip(
            _sample_at(gradient, coordinates) / gradient_scale, 0.0, 1.0
        )
        # High-frequency tracks are exactly where sub-pixel reprojection
        # jitter resembles specularity; keep them only as weak evidence.
        weights *= 0.1 + 0.9 * (1.0 - edge) ** 2

    residual_map = _dense_from_sparse(
        values,
        coordinates,
        image.shape[:2],
        blur_sigma=blur_sigma,
        weights=weights,
        norm_scale=norm_scale,
    )
    confidence_map = _dense_from_sparse(
        stability[point_ids],
        coordinates,
        image.shape[:2],
        blur_sigma=max(1.0, blur_sigma * 0.65),
        weights=weights,
        norm_scale=1.0,
    )
    return residual_map, confidence_map


def _atomic_save_u16(array: np.ndarray, destination: Path) -> None:
    partial = destination.with_name(destination.stem + ".partial.png")
    encoded = np.ascontiguousarray(
        (np.clip(array, 0.0, 1.0) * 65535).astype(np.uint16)
    )
    Image.fromarray(encoded, mode="I;16").save(partial)
    os.replace(partial, destination)


def generate_cross_view_maps(
    scene: str,
    colmap_dir: Path,
    output_dir: Path,
    *,
    min_views: int = 4,
    blur_sigma: float = 6.0,
    norm_percentile: float = 95.0,
    limit: int = 0,
    overwrite: bool = False,
    exposure_normalize: bool = True,
    patch_radius: int = 3,
    broad_patch_radius: int = 9,
    max_reprojection_error: float = 2.5,
) -> Path:
    """Generate V3 residual maps and companion confidence maps."""
    manifest_path = output_dir / "_crossview_manifest.json"
    existing_maps = output_dir.is_dir() and any(output_dir.glob("*.png"))
    if existing_maps and not overwrite:
        existing_version = None
        if manifest_path.is_file():
            try:
                existing_version = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("algorithm_version")
            except (OSError, ValueError, TypeError):
                existing_version = None
        if existing_version != CROSSVIEW_ALGORITHM_VERSION:
            raise RuntimeError(
                f"Legacy or incompatible maps found in {output_dir}; "
                "use a new V3 directory or pass --overwrite."
            )

    manager, frames = load_colmap_model(colmap_dir)
    print(
        f"Loaded COLMAP model: {len(manager.points3D)} points, "
        f"{len(frames)} frames"
    )
    observations = sample_track_observations(
        manager,
        frames,
        colmap_dir,
        patch_radius=patch_radius,
        broad_patch_radius=broad_patch_radius,
    )
    if exposure_normalize:
        observations, gains, offsets = _robust_affine_normalize(observations)
        valid_frames = np.sum(np.isfinite(observations), axis=1) >= 20
        if valid_frames.any():
            print(
                "Photometric affine normalisation: "
                f"gain p5/p50/p95="
                f"{np.percentile(gains[valid_frames], [5, 50, 95]).round(3).tolist()} "
                f"offset p5/p50/p95="
                f"{np.percentile(offsets[valid_frames], [5, 50, 95]).round(3).tolist()}"
            )

    residual, stability, n_views, usable = compute_positive_observation_residuals(
        observations, min_views=min_views
    )
    if max_reprojection_error > 0 and hasattr(manager, "point3D_errors"):
        usable &= (
            np.asarray(manager.point3D_errors) <= max_reprojection_error
        )
        residual[:, ~usable] = np.nan
        stability[~usable] = 0.0
    if not usable.any():
        raise RuntimeError("No reliable COLMAP tracks remain for V3")

    finite_residual = residual[:, usable]
    positive = finite_residual[
        np.isfinite(finite_residual) & (finite_residual > 0)
    ]
    if positive.size == 0:
        raise RuntimeError("No positive per-observation residuals were found")
    norm_scale = float(np.percentile(positive, norm_percentile))
    print(
        f"V3 observations: {int(usable.sum())} reliable points; "
        f"positive residual p50/p90/p99="
        f"{np.percentile(positive, [50, 90, 99]).round(4).tolist()}; "
        f"norm p{norm_percentile:g}={norm_scale:.4f}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir = output_dir / "confidence"
    confidence_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for frame_index, frame in enumerate(frames):
        if limit > 0 and frame_index >= limit:
            break
        stem = Path(frame.name).stem
        output_path = output_dir / f"{stem}.png"
        confidence_path = confidence_dir / f"{stem}.png"
        if output_path.exists() and confidence_path.exists() and not overwrite:
            continue
        residual_map, confidence_map = generate_dense_observation_maps(
            manager,
            frame,
            frame_index,
            residual,
            stability,
            n_views,
            usable,
            colmap_dir,
            blur_sigma=blur_sigma,
            norm_scale=norm_scale,
        )
        _atomic_save_u16(residual_map, output_path)
        _atomic_save_u16(confidence_map, confidence_path)
        processed += 1
        if processed <= 3 or processed % 20 == 0:
            print(f"  [{frame_index + 1}/{len(frames)}] {frame.name}")

    complete = limit <= 0 or limit >= len(frames)
    manifest = {
        "algorithm_version": CROSSVIEW_ALGORITHM_VERSION,
        "scene": scene,
        "colmap_dir": str(colmap_dir),
        "output_dir": str(output_dir),
        "confidence_dir": str(confidence_dir),
        "registered_frames": len(frames),
        "processed_this_run": processed,
        "limit": limit,
        "complete": complete,
        "total_points": len(manager.points3D),
        "usable_points": int(usable.sum()),
        "min_views": min_views,
        "patch_radius": patch_radius,
        "broad_patch_radius": broad_patch_radius,
        "max_reprojection_error": max_reprojection_error,
        "exposure_normalize": exposure_normalize,
        "blur_sigma": blur_sigma,
        "norm_percentile": norm_percentile,
        "norm_scale": norm_scale,
    }
    partial_manifest = output_dir / "_crossview_manifest.partial.json"
    partial_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(partial_manifest, manifest_path)
    print(f"Saved {processed} V3 maps to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--colmap-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-views", type=int, default=4)
    parser.add_argument("--blur-sigma", type=float, default=6.0)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--broad-patch-radius", type=int, default=9)
    parser.add_argument("--max-reprojection-error", type=float, default=2.5)
    parser.add_argument("--norm-percentile", type=float, default=95.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-exposure-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    colmap_dir = args.colmap_dir or DATA_CUSTOM / f"{args.scene}_colmap"
    output_dir = (
        args.output_dir
        or DATA_CUSTOM / f"{args.scene}_crossview_maps_v3"
    )
    try:
        generate_cross_view_maps(
            args.scene,
            colmap_dir,
            output_dir,
            min_views=args.min_views,
            blur_sigma=args.blur_sigma,
            norm_percentile=args.norm_percentile,
            limit=args.limit,
            overwrite=args.overwrite,
            exposure_normalize=not args.no_exposure_normalize,
            patch_radius=args.patch_radius,
            broad_patch_radius=args.broad_patch_radius,
            max_reprojection_error=args.max_reprojection_error,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

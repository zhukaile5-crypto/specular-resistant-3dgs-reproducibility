#!/usr/bin/env python3
"""Specular-aware 3DGS reconstruction pipeline (full images).

Implements the ``nerfstudio``-env stages of the specular-reflection
reconstruction pipeline.  With cross-view detection enabled the flow is::

    raw_images/<scene>/                         original captures
        -> COLMAP on original images
           -> data/custom/<scene>_colmap/
        -> generate cross-view variance maps
           -> data/custom/<scene>_crossview_maps/
        -> blend original + delighted (single-frame + cross-view score)
           -> data/custom/<scene>_blended_<tag>/
           -> data/custom/<scene>_scoremaps/
        -> build training COLMAP dir (original poses + blended images)
           -> data/custom/<scene>_train_colmap_<tag>/
        -> vanilla 3DGS training
           -> results/<scene>/specular_aware_<tag>/

Run individual stages with ``--mode {prepare,colmap,crossview,train-colmap,train}``
or the full chain with ``--mode pipeline``.

Examples::

    # Stage 1: COLMAP on original images (for poses + cross-view tracks)
    python experiment_configs/gsplat_specular_aware_trainer.py \
        --scene photo_scene6 --mode colmap

    # Stage 2: generate dense cross-view variance maps
    python experiment_configs/gsplat_specular_aware_trainer.py \
        --scene photo_scene6 --mode crossview

    # Stage 3: blend original + delighted images (uses cross-view maps)
    python experiment_configs/gsplat_specular_aware_trainer.py \
        --scene photo_scene6 --mode prepare

    # Stage 4: build training COLMAP dir (original poses + blended images)
    python experiment_configs/gsplat_specular_aware_trainer.py \
        --scene photo_scene6 --mode train-colmap

    # Stage 5: train vanilla 3DGS
    python experiment_configs/gsplat_specular_aware_trainer.py \
        --scene photo_scene6 --mode train

    # Or run the whole chain
    python experiment_configs/gsplat_specular_aware_trainer.py \
        --scene photo_scene6 --mode pipeline
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
GSPLAT_EXAMPLES = PROJECT_ROOT / "third_party" / "gsplat" / "examples"
STABLEDELIGHT_ROOT = PROJECT_ROOT / "StableDelight"
RAW_IMAGES = PROJECT_ROOT / "raw_images"
DATA_CUSTOM = PROJECT_ROOT / "data" / "custom"
RESULTS = PROJECT_ROOT / "results"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

# Make locked gsplat source importable
if str(GSPLAT_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(GSPLAT_EXAMPLES))
if str(GSPLAT_EXAMPLES.parent) not in sys.path:
    sys.path.insert(0, str(GSPLAT_EXAMPLES.parent))

# Make StableDelight utilities importable in the nerfstudio env (needed for
# the specular detector during blending).
if str(STABLEDELIGHT_ROOT) not in sys.path:
    sys.path.insert(0, str(STABLEDELIGHT_ROOT))

# Make the project root importable (for experiment_configs.blend_strategies)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_configs.blend_strategies import (
    ADAPTIVE_FEATHER_DEFAULT_PARAMETERS,
    EVIDENCE_MATTE_DEFAULT_PARAMETERS,
    WEAK_HYSTERESIS_FEATHER_DEFAULT_PARAMETERS,
    WEAK_HYSTERESIS_OUTER_FEATHER_DEFAULT_PARAMETERS,
    available_strategies,
    get_strategy,
)
from experiment_configs.version_profiles import get_version_profile


# Resolve all flagship defaults from the V1--V6 registry so entry points and
# documentation exporters cannot silently drift apart.
_V4_RELEASE = get_version_profile("v4")
_V5_CANDIDATE = get_version_profile("v5")
_V51_CANDIDATE = get_version_profile("v5.1")
_FLAGSHIP = get_version_profile("v6")
FLAGSHIP_RELEASE = _FLAGSHIP.release
FLAGSHIP_PROFILE = FLAGSHIP_RELEASE
FLAGSHIP_DETECTOR_VERSION = str(_FLAGSHIP.detector_version)
FLAGSHIP_BLEND_MODE = _FLAGSHIP.blend_mode
FLAGSHIP_SPECULAR_THRESHOLD = _FLAGSHIP.threshold
FLAGSHIP_DATA_FACTOR = _FLAGSHIP.data_factor
DEFAULT_PREPARE_WORKERS = 4
DEFAULT_ENCODE_WORKERS_PER_PROCESS = 2
DEFAULT_DETECTOR_BACKEND = "cpu"
_CUDA_WORKER_STATE = threading.local()


# ---------------------------------------------------------------------------
# Scene defaults
# ---------------------------------------------------------------------------

def pipeline_profile(
    detector_version: str,
    blend_mode: str,
    specular_threshold: float,
    data_factor: int,
) -> str:
    """Return a named V4/V5/V5.1/V6 profile or ``custom`` for an ablation."""
    if (
        detector_version == FLAGSHIP_DETECTOR_VERSION
        and blend_mode == FLAGSHIP_BLEND_MODE
        and specular_threshold == FLAGSHIP_SPECULAR_THRESHOLD
        and data_factor == FLAGSHIP_DATA_FACTOR
    ):
        return FLAGSHIP_PROFILE
    if (
        detector_version == _V4_RELEASE.detector_version
        and blend_mode == _V4_RELEASE.blend_mode
        and specular_threshold == _V4_RELEASE.threshold
        and data_factor == _V4_RELEASE.data_factor
    ):
        return _V4_RELEASE.release
    if (
        detector_version == _V5_CANDIDATE.detector_version
        and blend_mode == _V5_CANDIDATE.blend_mode
        and specular_threshold == _V5_CANDIDATE.threshold
        and data_factor == _V5_CANDIDATE.data_factor
    ):
        return _V5_CANDIDATE.release
    if (
        detector_version == _V51_CANDIDATE.detector_version
        and blend_mode == _V51_CANDIDATE.blend_mode
        and specular_threshold == _V51_CANDIDATE.threshold
        and data_factor == _V51_CANDIDATE.data_factor
    ):
        return _V51_CANDIDATE.release
    return "custom"


def experiment_tag(blend_mode: str, specular_threshold: float) -> str:
    """Short tag identifying a blending experiment, e.g. ``soft_t0.3``.

    Used to give every blending configuration its own output directories so
    that ablation runs never overwrite each other.
    """
    if blend_mode in (
        "soft",
        "hard",
        "adaptive_feather",
        "weak_hysteresis_feather",
        "weak_hysteresis_outer_feather",
        "evidence_matte",
    ):
        return f"{blend_mode}_t{specular_threshold:g}"
    return blend_mode


def default_paths(
    scene: str,
    *,
    blend_mode: str = FLAGSHIP_BLEND_MODE,
    specular_threshold: float = FLAGSHIP_SPECULAR_THRESHOLD,
    detector_version: str = FLAGSHIP_DETECTOR_VERSION,
) -> Dict[str, Path]:
    """Return canonical paths for a scene and blending configuration."""
    if detector_version not in ("v2", "v3"):
        raise ValueError("detector_version must be 'v2' or 'v3'")
    tag = experiment_tag(blend_mode, specular_threshold)
    # Unversioned detector artifacts belong to the historical V1 release.
    # Keep every runnable detector isolated so V2 cannot overwrite V1.
    suffix = f"_{detector_version}"
    return {
        "raw_dir": RAW_IMAGES / scene,
        "delighted_dir": DATA_CUSTOM / f"{scene}_delighted",
        "scoremap_dir": DATA_CUSTOM / f"{scene}_scoremaps{suffix}",
        "blended_dir": DATA_CUSTOM / f"{scene}_blended_{tag}{suffix}",
        "orig_colmap_dir": DATA_CUSTOM / f"{scene}_colmap",
        "crossview_dir": DATA_CUSTOM / f"{scene}_crossview_maps{suffix}",
        "train_colmap_dir": DATA_CUSTOM / f"{scene}_train_colmap_{tag}{suffix}",
        "result_dir": RESULTS / scene / f"specular_aware_{tag}{suffix}",
    }


def _find_delighted_counterpart(
    original_path: Path, delighted_dir: Path
) -> Optional[Path]:
    """Find a delighted image matching the original stem."""
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = delighted_dir / original_path.with_suffix(ext).name
        if candidate.exists():
            return candidate
    return None


def _load_cross_view_map(crossview_dir: Path, image_name: str) -> Optional[np.ndarray]:
    """Load a dense cross-view variance map for an image, if present.

    Handles naming mismatch between raw images (``0012.jpg``) and COLMAP
    frames (``frame_00012.jpg``).
    """
    stem = Path(image_name).stem
    candidates = [stem]
    if stem.isdigit():
        candidates.append(f"frame_{int(stem):05d}")
    elif stem.startswith("frame_"):
        candidates.append(stem.replace("frame_", ""))

    for name in candidates:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = crossview_dir / f"{name}{ext}"
            if candidate.is_file():
                arr = np.asarray(Image.open(candidate))
                if arr.dtype == np.uint16:
                    return arr.astype(np.float32) / 65535.0
                return arr.astype(np.float32) / 255.0
    return None


def _save_float_map_atomic(path: Path, values: np.ndarray) -> None:
    """Save a [0, 1] float map as 16-bit PNG without exposing partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial{path.suffix}")
    encoded = (np.clip(values, 0.0, 1.0) * 65535).astype(np.uint16)
    Image.fromarray(encoded, mode="I;16").save(partial)
    os.replace(partial, path)


def _save_blended_jpeg(
    path: Path,
    values: np.ndarray,
    *,
    quality: int,
    disable_subsampling: bool,
) -> None:
    """Encode one blended RGB array with the established V6 JPEG settings."""
    save_kwargs: Dict[str, Any] = {"quality": quality}
    if disable_subsampling:
        save_kwargs["subsampling"] = 0
    Image.fromarray(values, mode="RGB").save(path, **save_kwargs)


def _prepare_blended_image_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Process one image without changing the established V6 math.

    The task is deliberately self-contained so it can run either in the
    caller (``workers=1``) or in a separate process.  Process-level
    parallelism changes only which images are evaluated concurrently; all
    per-image operations, dtypes, interpolation modes, and encoders remain
    identical to the serial implementation.
    """
    detector_backend = str(task.get("detector_backend", "cpu"))
    from stabledelight.utils.specular_detector import set_box_blur_backend

    set_box_blur_backend(detector_backend)
    if detector_backend == "cuda-exact":
        import torch

        if not hasattr(_CUDA_WORKER_STATE, "stream"):
            _CUDA_WORKER_STATE.stream = torch.cuda.Stream()
        torch.cuda.set_stream(_CUDA_WORKER_STATE.stream)

    detector_version = str(task["detector_version"])
    if detector_version == "v3":
        from stabledelight.utils.specular_detector_v3 import (
            compute_specular_score,
        )
    elif detector_version == "v2":
        from stabledelight.utils.specular_detector import (
            compute_specular_score,
        )
    else:
        raise ValueError("detector_version must be 'v2' or 'v3'")

    name = str(task["name"])
    orig_path = Path(task["orig_path"])
    deli_path = Path(task["deli_path"])
    out_path = Path(task["out_path"])
    scoremap_path = Path(task["scoremap_path"])
    crossview_dir = Path(task["crossview_dir"])
    use_crossview = bool(task["use_crossview"])
    save_scoremaps = bool(task["save_scoremaps"])
    save_blend_weights = bool(task["save_blend_weights"])

    strategy = get_strategy(str(task["blend_mode"]))
    needs_scoremap = getattr(strategy, "needs_scoremap", True)
    provides_diagnostics = getattr(strategy, "provides_diagnostics", False)
    needs_v3_context = getattr(strategy, "needs_v3_context", False)
    needs_detector_cues = getattr(strategy, "needs_detector_cues", False)

    original_u8 = np.asarray(Image.open(orig_path).convert("RGB"))
    original = original_u8.astype(np.float32) / 255.0
    delighted = (
        np.asarray(Image.open(deli_path).convert("RGB")).astype(np.float32) / 255.0
    )

    if original.shape[:2] != delighted.shape[:2]:
        delighted = (
            np.asarray(
                Image.fromarray((delighted * 255).astype(np.uint8)).resize(
                    (original.shape[1], original.shape[0]), Image.LANCZOS
                )
            ).astype(np.float32)
            / 255.0
        )

    score: Optional[np.ndarray] = None
    detector_cues: Optional[Dict[str, np.ndarray]] = None
    cross_view_map = None
    cross_view_confidence = None
    score_stats: Optional[Dict[str, float]] = None
    score_mean = 1.0
    if needs_scoremap:
        if use_crossview:
            cross_view_map = _load_cross_view_map(crossview_dir, name)
            if (
                cross_view_map is not None
                and cross_view_map.shape[:2] != original.shape[:2]
            ):
                pil = Image.fromarray(
                    cross_view_map.astype(np.float32, copy=False), mode="F"
                )
                pil = pil.resize(
                    (original.shape[1], original.shape[0]), Image.BILINEAR
                )
                cross_view_map = np.asarray(pil).astype(
                    np.float32, copy=False
                )

            if detector_version == "v3":
                cross_view_confidence = _load_cross_view_map(
                    crossview_dir / "confidence", name
                )
                if cross_view_map is None or cross_view_confidence is None:
                    raise RuntimeError(f"Incomplete V3 cross-view evidence for {name}")
                if cross_view_confidence.shape[:2] != original.shape[:2]:
                    confidence_pil = Image.fromarray(
                        cross_view_confidence.astype(
                            np.float32, copy=False
                        ),
                        mode="F",
                    )
                    confidence_pil = confidence_pil.resize(
                        (original.shape[1], original.shape[0]), Image.BILINEAR
                    )
                    cross_view_confidence = np.asarray(confidence_pil).astype(
                        np.float32, copy=False
                    )

        score_kwargs = {
            "cross_view_map": cross_view_map,
            "cross_view_weight": float(task["cross_view_weight"]),
        }
        if detector_version == "v3":
            score_kwargs.update(
                {
                    "cross_view_confidence": cross_view_confidence,
                    "delighted_image": delighted,
                }
            )
        if needs_detector_cues:
            score, detector_cues = compute_specular_score(
                original_u8,
                return_cues=True,
                **score_kwargs,
            )
        else:
            score = compute_specular_score(
                original_u8, **score_kwargs
            )
        score_mean = float(score.mean())
        p50, p90, p99 = np.percentile(score, [50, 90, 99])
        score_stats = {
            "mean": round(score_mean, 4),
            "p50": round(float(p50), 4),
            "p90": round(float(p90), 4),
            "p99": round(float(p99), 4),
        }
        if cross_view_map is not None:
            score_stats["cross_view_mean"] = round(float(cross_view_map.mean()), 4)
        if cross_view_confidence is not None:
            score_stats["cross_view_confidence_mean"] = round(
                float(cross_view_confidence.mean()), 4
            )
        if save_scoremaps:
            score_u16 = (np.clip(score, 0, 1) * 65535).astype(np.uint16)
            Image.fromarray(score_u16, mode="I;16").save(scoremap_path)

    strategy_kwargs: Dict[str, Any] = {
        "threshold": float(task["specular_threshold"])
    }
    if needs_v3_context:
        strategy_kwargs.update(
            {
                "cross_view_map": cross_view_map,
                "cross_view_confidence": cross_view_confidence,
                "data_factor": int(task["data_factor"]),
                "return_diagnostics": save_blend_weights,
            }
        )
    if needs_detector_cues:
        strategy_kwargs["detector_cues"] = detector_cues
    strategy_kwargs.update(task["blend_parameters"])
    strategy_result = strategy(original, delighted, score, **strategy_kwargs)
    diagnostic_jobs: List[Tuple[Path, np.ndarray]] = []
    if provides_diagnostics and save_blend_weights:
        blended, blend_diagnostics = strategy_result
        stem = Path(name).stem
        diagnostic_dirs = {
            key: Path(value)
            for key, value in task["diagnostic_dirs"].items()
        }
        for key, values in blend_diagnostics.items():
            if key in diagnostic_dirs:
                diagnostic_jobs.append(
                    (diagnostic_dirs[key] / f"{stem}.png", values)
                )
    else:
        blended = strategy_result

    blended_uint8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    encode_workers = max(int(task["encode_workers_per_process"]), 1)
    if encode_workers == 1 or not diagnostic_jobs:
        for path, values in diagnostic_jobs:
            _save_float_map_atomic(path, values)
        _save_blended_jpeg(
            out_path,
            blended_uint8,
            quality=int(task["quality"]),
            disable_subsampling=provides_diagnostics,
        )
    else:
        with ThreadPoolExecutor(max_workers=encode_workers) as encode_pool:
            futures = [
                encode_pool.submit(_save_float_map_atomic, path, values)
                for path, values in diagnostic_jobs
            ]
            futures.append(
                encode_pool.submit(
                    _save_blended_jpeg,
                    out_path,
                    blended_uint8,
                    quality=int(task["quality"]),
                    disable_subsampling=provides_diagnostics,
                )
            )
            for future in futures:
                future.result()

    entry: Dict[str, Any] = {
        "original": str(orig_path),
        "delighted": str(deli_path),
        "output": str(out_path),
        "mean_score": round(score_mean, 4),
    }
    if score_stats is not None:
        entry["score_stats"] = score_stats
        if save_scoremaps:
            entry["scoremap"] = str(scoremap_path)
    return {
        "name": name,
        "score_mean": score_mean,
        "entry": entry,
    }


# ---------------------------------------------------------------------------
# Mode: colmap — run COLMAP on original images
# ---------------------------------------------------------------------------

def run_colmap(
    scene: str,
    *,
    raw_dir: Optional[Path] = None,
    colmap_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Run ``ns-process-data images`` on original images.

    The resulting poses and 3D tracks are used for cross-view specular
    detection and (later) for 3DGS training.

    Args:
        scene: Scene name.
        raw_dir: Directory containing original images. Defaults to
            ``raw_images/<scene>``.
        colmap_dir: COLMAP output directory. Defaults to
            ``data/custom/<scene>_colmap``.
        overwrite: If True, remove an existing ``colmap_dir`` before processing.

    Returns:
        Path to the COLMAP output directory.
    """
    paths = default_paths(scene)
    if raw_dir is None:
        raw_dir = paths["raw_dir"]
    if colmap_dir is None:
        colmap_dir = paths["orig_colmap_dir"]

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw image directory not found: {raw_dir}")

    if colmap_dir.exists():
        if overwrite:
            print(f"Removing existing COLMAP output: {colmap_dir}")
            shutil.rmtree(colmap_dir)
        else:
            print(f"COLMAP output already exists: {colmap_dir} -- skipping")
            return colmap_dir

    ns_process_data = shutil.which("ns-process-data")
    if ns_process_data is None:
        raise RuntimeError(
            "ns-process-data not found in PATH. "
            "Activate the nerfstudio conda environment first."
        )

    print(f"Running COLMAP on original images for scene: {scene}")
    print(f"  Input:  {raw_dir}")
    print(f"  Output: {colmap_dir}")

    cmd = [
        ns_process_data,
        "images",
        "--data", str(raw_dir),
        "--output-dir", str(colmap_dir),
    ]
    subprocess.run(cmd, check=True)

    if not (colmap_dir / "transforms.json").exists():
        raise RuntimeError(
            f"COLMAP processing failed: {colmap_dir / 'transforms.json'} not found."
        )

    print(f"COLMAP finished: {colmap_dir}")
    return colmap_dir


# ---------------------------------------------------------------------------
# Mode: crossview — generate dense cross-view variance maps
# ---------------------------------------------------------------------------

def run_crossview(
    scene: str,
    *,
    colmap_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_views: int = 4,
    blur_sigma: float = 6.0,
    patch_radius: int = 3,
    broad_patch_radius: int = 9,
    max_reprojection_error: float = 2.5,
    variation_metric: str = "upper",
    exposure_normalize: bool = True,
    limit: int = 0,
    overwrite: bool = False,
    detector_version: str = FLAGSHIP_DETECTOR_VERSION,
) -> Path:
    """Generate dense cross-view specular variance maps from the COLMAP model.

    Thin wrapper around ``experiment_configs.cross_view_specular``.
    """
    if detector_version == "v3":
        from experiment_configs.cross_view_specular_v3 import generate_cross_view_maps
    elif detector_version == "v2":
        from experiment_configs.cross_view_specular import generate_cross_view_maps
    else:
        raise ValueError("detector_version must be 'v2' or 'v3'")

    paths = default_paths(scene, detector_version=detector_version)
    if colmap_dir is None:
        colmap_dir = paths["orig_colmap_dir"]
    if output_dir is None:
        output_dir = paths["crossview_dir"]

    kwargs = {
        "min_views": min_views,
        "blur_sigma": blur_sigma,
        "patch_radius": patch_radius,
        "max_reprojection_error": max_reprojection_error,
        "exposure_normalize": exposure_normalize,
        "limit": limit,
        "overwrite": overwrite,
    }
    if detector_version == "v3":
        kwargs["broad_patch_radius"] = broad_patch_radius
    else:
        kwargs["variation_metric"] = variation_metric
    return generate_cross_view_maps(scene, colmap_dir, output_dir, **kwargs)


# ---------------------------------------------------------------------------
# Mode: prepare — blend original + delighted images
# ---------------------------------------------------------------------------

def prepare_blended_images(
    scene: str,
    *,
    delighted_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    scoremap_dir: Optional[Path] = None,
    crossview_dir: Optional[Path] = None,
    blend_mode: str = FLAGSHIP_BLEND_MODE,
    specular_threshold: float = FLAGSHIP_SPECULAR_THRESHOLD,
    cross_view_weight: float = 1.0,
    save_scoremaps: bool = True,
    save_blend_weights: bool = True,
    data_factor: int = FLAGSHIP_DATA_FACTOR,
    quality: int = 100,
    overwrite: bool = False,
    limit: int = 0,
    detector_version: str = FLAGSHIP_DETECTOR_VERSION,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    workers: int = DEFAULT_PREPARE_WORKERS,
    encode_workers_per_process: int = DEFAULT_ENCODE_WORKERS_PER_PROCESS,
) -> Path:
    """Generate blended training images for ``scene``.

    For every raw image in ``raw_images/<scene>/`` that has a delighted
    counterpart, compute an edge-aware specular probability map on the
    original (unless the strategy does not need one) and blend with the
    registered strategy from ``experiment_configs.blend_strategies``::

        blended = strategy(original, delighted, prob, threshold=...)

    If dense cross-view variance maps are available (from
    ``data/custom/<scene>_crossview_maps/``), they are fused with the
    single-frame cues so that broad smooth reflections are also detected.

    The blended images are written as JPEGs to
    ``data/custom/<scene>_blended_<tag>/`` and the threshold-independent
    specular score maps as 16-bit PNGs to ``data/custom/<scene>_scoremaps/``
    for later analysis / adaptive threshold strategies.

    Args:
        scene: Scene name.
        delighted_dir: Directory containing delighted PNGs. Defaults to
            ``data/custom/<scene>_delighted``.
        output_dir: Where to write blended JPEGs. Defaults to
            ``data/custom/<scene>_blended_<tag>``.
        scoremap_dir: Where to write specular score maps. Defaults
            to ``data/custom/<scene>_scoremaps``.
        crossview_dir: Directory with dense cross-view variance maps.
            Defaults to ``data/custom/<scene>_crossview_maps``.  Pass
            ``None`` or a non-existent path to disable cross-view fusion.
        blend_mode: Name of a registered blending strategy
            (``soft`` / ``hard`` / ``delight_only`` / ...).
        specular_threshold: Passed to the strategy as its score-mapping
            threshold (sigmoid bias for ``soft``, cutoff for ``hard``).
        cross_view_weight: Weight of the cross-view cue when fusing with
            single-frame cues.
        save_scoremaps: Save per-image score maps to ``scoremap_dir``.
        save_blend_weights: Save adaptive strategy base/detail/chroma and
            diagnostic weight maps beneath ``<output_dir>/_weights``.
        data_factor: Final gsplat downsample factor used when measuring
            component size for scale-adaptive blending.
        quality: JPEG output quality.
        overwrite: Re-generate existing outputs.
        limit: If > 0, process at most this many images (smoke tests).
        detector_backend: ``cpu`` for the SciPy reference or ``cuda-exact``
            for the operation-order-compatible CUDA box-filter backend.
        workers: Number of image-level workers. ``1`` preserves serial
            execution. CPU uses processes; ``cuda-exact`` uses CUDA streams
            hosted by threads so independent images can overlap safely.
        encode_workers_per_process: Threads used by each image process to
            encode independent diagnostic PNGs and the final JPEG.

    Returns:
        Path to the output directory.
    """
    if detector_version == "v3":
        from stabledelight.utils.specular_detector_v3 import (
            DETECTOR_VERSION,
        )
        from experiment_configs.cross_view_specular_v3 import (
            CROSSVIEW_ALGORITHM_VERSION,
        )
    elif detector_version == "v2":
        from stabledelight.utils.specular_detector import (
            DETECTOR_VERSION,
        )
        from experiment_configs.cross_view_specular import (
            CROSSVIEW_ALGORITHM_VERSION,
        )
    else:
        raise ValueError("detector_version must be 'v2' or 'v3'")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if encode_workers_per_process < 1:
        raise ValueError("encode_workers_per_process must be at least 1")
    if detector_backend not in ("cpu", "cuda-exact"):
        raise ValueError("detector_backend must be 'cpu' or 'cuda-exact'")

    strategy = get_strategy(blend_mode)
    needs_scoremap = getattr(strategy, "needs_scoremap", True)
    provides_diagnostics = getattr(strategy, "provides_diagnostics", False)
    needs_v3_context = getattr(strategy, "needs_v3_context", False)
    needs_detector_cues = getattr(strategy, "needs_detector_cues", False)
    strategy_version = getattr(strategy, "strategy_version", None)
    if (needs_v3_context or needs_detector_cues) and detector_version != "v3":
        raise ValueError(
            f"Blend mode '{blend_mode}' requires --detector-version v3"
        )

    paths = default_paths(
        scene,
        blend_mode=blend_mode,
        specular_threshold=specular_threshold,
        detector_version=detector_version,
    )
    raw_dir = paths["raw_dir"]
    if delighted_dir is None:
        delighted_dir = paths["delighted_dir"]
    if output_dir is None:
        output_dir = paths["blended_dir"]
    if scoremap_dir is None:
        scoremap_dir = paths["scoremap_dir"]
    if crossview_dir is None:
        crossview_dir = paths["crossview_dir"]

    blend_parameters: Dict[str, float] = {}
    if blend_mode == "adaptive_feather":
        blend_parameters = dict(ADAPTIVE_FEATHER_DEFAULT_PARAMETERS)
    elif blend_mode == "weak_hysteresis_feather":
        blend_parameters = dict(WEAK_HYSTERESIS_FEATHER_DEFAULT_PARAMETERS)
    elif blend_mode == "weak_hysteresis_outer_feather":
        blend_parameters = dict(
            WEAK_HYSTERESIS_OUTER_FEATHER_DEFAULT_PARAMETERS
        )
    elif blend_mode == "evidence_matte":
        blend_parameters = dict(EVIDENCE_MATTE_DEFAULT_PARAMETERS)

    # The canonical output path intentionally remains stable across flagship
    # parameter tuning. Refuse to silently mix an older parameter set with
    # the current release when existing images would otherwise be skipped.
    existing_manifest_path = output_dir / "_blend_manifest.json"
    existing_images = output_dir.is_dir() and any(output_dir.glob("*.jpg"))
    if existing_images and not overwrite and blend_parameters:
        existing_parameters = None
        if existing_manifest_path.is_file():
            try:
                with open(existing_manifest_path, "r", encoding="utf-8") as handle:
                    existing_parameters = json.load(handle).get("blend_parameters")
            except (OSError, ValueError, TypeError):
                existing_parameters = None
        if existing_parameters != blend_parameters:
            raise RuntimeError(
                f"Existing blended images in {output_dir} use legacy, unknown, "
                "or incompatible strategy parameters. Existing files "
                "were preserved. Re-run with --overwrite to generate the "
                f"current parameters {blend_parameters}, or select a "
                "different --blended-dir."
            )

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw image directory not found: {raw_dir}")
    if not delighted_dir.is_dir():
        raise FileNotFoundError(f"Delighted directory not found: {delighted_dir}")

    use_crossview = crossview_dir is not None and crossview_dir.is_dir()
    if use_crossview:
        manifest_path = crossview_dir / "_crossview_manifest.json"
        crossview_version = None
        crossview_complete = False
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    crossview_manifest = json.load(handle)
                crossview_version = crossview_manifest.get("algorithm_version")
                crossview_complete = bool(crossview_manifest.get("complete"))
            except (OSError, ValueError, TypeError):
                crossview_version = None
        if (
            crossview_version != CROSSVIEW_ALGORITHM_VERSION
            or not crossview_complete
        ):
            raise RuntimeError(
                f"Cross-view maps in {crossview_dir} are legacy, incomplete, or "
                "incompatible. Regenerate the full set with --mode crossview "
                "--overwrite before prepare."
            )
        print(f"  Cross-view maps: {crossview_dir} (weight={cross_view_weight})")
    else:
        print("  Cross-view maps: not available, using single-frame detector only")

    output_dir.mkdir(parents=True, exist_ok=True)
    if needs_scoremap and save_scoremaps:
        scoremap_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dirs: Dict[str, Path] = {}
    if provides_diagnostics and save_blend_weights:
        weight_root = output_dir / "_weights"
        for key in (
            "base_weight",
            "detail_weight",
            "chroma_weight",
            "size_scale",
            "size_gain",
            "view_override",
            "hysteresis_support",
            "feather_support",
            "edge_protection",
            "transition_risk",
            "transition_guard",
            "outer_support",
            "reflection_matte",
            "matte_likelihood",
            "matte_uncertainty",
            "foreground_seed",
            "background_seed",
        ):
            diagnostic_dirs[key] = weight_root / key
            diagnostic_dirs[key].mkdir(parents=True, exist_ok=True)

    raw_names = sorted(
        p.name
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if limit > 0:
        raw_names = raw_names[:limit]

    print(f"Preparing blended images for scene: {scene}")
    print(f"  Raw images: {raw_dir} ({len(raw_names)} found)")
    print(f"  Delighted source: {delighted_dir}")
    print(f"  Output: {output_dir}")
    print(
        "  Pipeline profile: "
        f"{pipeline_profile(detector_version, blend_mode, specular_threshold, data_factor)}"
    )
    print(f"  Detector: {detector_version} ({DETECTOR_VERSION})")
    print(f"  Detector backend: {detector_backend}")
    print(f"  Blend mode: {blend_mode}")
    if provides_diagnostics:
        print(f"  Adaptive strategy: {strategy_version}")
        print(f"  Component scale: data_factor={data_factor}")
        if blend_parameters:
            print(f"  Blend parameters: {blend_parameters}")
        if save_blend_weights:
            print(f"  Blend weight maps: {output_dir / '_weights'}")
    if needs_scoremap:
        print(f"  Specular threshold: {specular_threshold}")
        if save_scoremaps:
            print(f"  Score maps: {scoremap_dir}")

    if workers == 1:
        execution_label = "serial"
    elif detector_backend == "cuda-exact":
        execution_label = f"{workers} CUDA image threads"
    else:
        execution_label = f"{workers} CPU image processes"
    print(f"  Prepare execution: {execution_label}")
    print(f"  Encoders per image worker: {encode_workers_per_process}")
    stats = {"processed": 0, "skipped": 0, "existing": 0, "mean_score": 0.0}
    score_means: List[float] = []
    manifest_entries: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []

    for name in raw_names:
        orig_path = raw_dir / name
        out_path = output_dir / Path(name).with_suffix(".jpg").name
        scoremap_path = scoremap_dir / Path(name).with_suffix(".png").name

        if out_path.exists() and not overwrite:
            stats["existing"] += 1
            continue

        deli_path = _find_delighted_counterpart(orig_path, delighted_dir)
        if deli_path is None:
            print(f"  WARN: No delighted counterpart for {name} -- skipping")
            stats["skipped"] += 1
            continue

        tasks.append(
            {
                "name": name,
                "orig_path": orig_path,
                "deli_path": deli_path,
                "out_path": out_path,
                "scoremap_path": scoremap_path,
                "crossview_dir": crossview_dir,
                "use_crossview": use_crossview,
                "cross_view_weight": cross_view_weight,
                "detector_version": detector_version,
                "detector_backend": detector_backend,
                "blend_mode": blend_mode,
                "specular_threshold": specular_threshold,
                "data_factor": data_factor,
                "quality": quality,
                "encode_workers_per_process": encode_workers_per_process,
                "save_scoremaps": save_scoremaps,
                "save_blend_weights": save_blend_weights,
                "diagnostic_dirs": diagnostic_dirs,
                "blend_parameters": blend_parameters,
            }
        )

    prepare_started = time.perf_counter()
    pool: Optional[Any] = None
    if not tasks:
        results = iter(())
    elif workers == 1:
        results = map(_prepare_blended_image_task, tasks)
    elif detector_backend == "cuda-exact":
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(_prepare_blended_image_task, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_prepare_blended_image_task, tasks, chunksize=1)

    try:
        for result in results:
            score_mean = float(result["score_mean"])
            score_means.append(score_mean)
            stats["processed"] += 1
            manifest_entries.append(result["entry"])
            if stats["processed"] <= 3:
                print(f"  {result['name']}: mean score = {score_mean:.3f}")
    finally:
        if pool is not None:
            pool.shutdown()

    if score_means:
        stats["mean_score"] = float(np.mean(score_means))
    prepare_elapsed_seconds = time.perf_counter() - prepare_started

    manifest = {
        "scene": scene,
        "pipeline_profile": pipeline_profile(
            detector_version,
            blend_mode,
            specular_threshold,
            data_factor,
        ),
        "detector_profile": detector_version,
        "detector_version": DETECTOR_VERSION,
        "detector_backend": detector_backend,
        "blend_mode": blend_mode,
        "specular_threshold": specular_threshold,
        "cross_view_weight": cross_view_weight,
        "data_factor": data_factor,
        "prepare_workers": workers,
        "encode_workers_per_process": encode_workers_per_process,
        "prepare_elapsed_seconds": prepare_elapsed_seconds,
        "crossview_dir": str(crossview_dir) if use_crossview else None,
        "delighted_source": str(delighted_dir),
        "output_dir": str(output_dir),
        "stats": stats,
        "images": manifest_entries,
    }
    if manifest["pipeline_profile"] in (
        FLAGSHIP_PROFILE,
        _V4_RELEASE.release,
        _V5_CANDIDATE.release,
        _V51_CANDIDATE.release,
    ):
        manifest["release"] = manifest["pipeline_profile"]
    if needs_scoremap and save_scoremaps:
        manifest["scoremap_dir"] = str(scoremap_dir)
    if provides_diagnostics:
        manifest["blend_strategy_version"] = strategy_version
        manifest["blend_parameters"] = blend_parameters
        manifest["jpeg_subsampling"] = 0
        manifest["blend_weight_dir"] = (
            str(output_dir / "_weights") if save_blend_weights else None
        )
    with open(output_dir / "_blend_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\nBlended images saved to: {output_dir}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Existing (skipped): {stats['existing']}")
    print(f"  Skipped (missing counterpart): {stats['skipped']}")
    print(f"  Mean score: {stats['mean_score']:.3f}")
    print(f"  Prepare elapsed: {prepare_elapsed_seconds:.2f}s")
    print(f"  Manifest: {output_dir / '_blend_manifest.json'}")

    return output_dir


# ---------------------------------------------------------------------------
# Mode: train-colmap — build training COLMAP dir (original poses + blended images)
# ---------------------------------------------------------------------------

def prepare_train_colmap(
    scene: str,
    *,
    orig_colmap_dir: Optional[Path] = None,
    blended_dir: Optional[Path] = None,
    train_colmap_dir: Optional[Path] = None,
    overwrite: bool = False,
    downsample_factors: Tuple[int, ...] = (4,),
    detector_version: str = FLAGSHIP_DETECTOR_VERSION,
) -> Path:
    """Build a training-ready COLMAP directory from original poses and blended images.

    The gsplat parser expects ``data_dir/sparse/0/`` (COLMAP model) and
    ``data_dir/images/`` (training images).  This function copies the sparse
    model from the original COLMAP run and replaces the images with the
    blended ones, so 3DGS trains on the anti-specular inputs while keeping
    the original camera poses.
    """
    paths = default_paths(scene, detector_version=detector_version)
    if orig_colmap_dir is None:
        orig_colmap_dir = paths["orig_colmap_dir"]
    if blended_dir is None:
        blended_dir = paths["blended_dir"]
    if train_colmap_dir is None:
        train_colmap_dir = paths["train_colmap_dir"]

    if not orig_colmap_dir.is_dir():
        raise FileNotFoundError(
            f"Original COLMAP output not found: {orig_colmap_dir}. "
            "Run --mode colmap first."
        )
    if not blended_dir.is_dir():
        raise FileNotFoundError(
            f"Blended image directory not found: {blended_dir}. "
            "Run --mode prepare first."
        )

    if train_colmap_dir.exists():
        if overwrite:
            print(f"Removing existing train COLMAP dir: {train_colmap_dir}")
            shutil.rmtree(train_colmap_dir)
        else:
            print(f"Train COLMAP dir already exists: {train_colmap_dir} -- skipping")
            return train_colmap_dir

    train_colmap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy sparse model
    src_sparse = orig_colmap_dir / "sparse"
    dst_sparse = train_colmap_dir / "sparse"
    if src_sparse.is_dir():
        shutil.copytree(src_sparse, dst_sparse)
    else:
        raise FileNotFoundError(f"Sparse model not found: {src_sparse}")

    # 2. Copy transforms.json if present
    src_transforms = orig_colmap_dir / "transforms.json"
    if src_transforms.is_file():
        shutil.copy2(src_transforms, train_colmap_dir / "transforms.json")

    # 3. Copy blended images into images/, RENAMED to the COLMAP frame names.
    #    gsplat's Parser looks up images by the exact names stored in the
    #    sparse model (e.g. ``frame_00001.jpg``); our blended images keep the
    #    raw names (``0001.jpg``), so we must map raw stem -> frame name via
    #    the digit part of the stem.
    images_dir = train_colmap_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Frame names from the original COLMAP run (images/ dir mirrors the
    # sparse model names).
    orig_images_dir = orig_colmap_dir / "images"
    if orig_images_dir.is_dir():
        frame_names = sorted(
            p.name for p in orig_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
    else:
        frame_names = []
    digits_to_frame = {}
    for fname in frame_names:
        digits = "".join(ch for ch in Path(fname).stem if ch.isdigit())
        if digits:
            digits_to_frame[digits.lstrip("0") or "0"] = fname

    blended_files = sorted(
        p for p in blended_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    renamed, kept, missing = 0, 0, []
    for src in blended_files:
        digits = "".join(ch for ch in src.stem if ch.isdigit())
        key = digits.lstrip("0") or "0"
        frame_name = digits_to_frame.get(key)
        if frame_name is not None and frame_name != src.name:
            dst = images_dir / frame_name
            renamed += 1
        else:
            if frame_name is None:
                missing.append(src.name)
            dst = images_dir / src.name
            kept += 1
        shutil.copy2(src, dst)
    if renamed:
        print(f"  Renamed {renamed} blended images to COLMAP frame names")
    if missing:
        print(f"  WARNING: {len(missing)} images had no COLMAP frame match "
              f"(kept original names): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # 4. Pre-create the downsampled image dirs that gsplat's Parser expects
    #    (factor>1 -> ``images_<factor>`` must exist; the Parser does NOT
    #    downsample on the fly, it raises if the dir is missing).
    for factor in downsample_factors:
        if factor <= 1:
            continue
        ds_dir = train_colmap_dir / f"images_{factor}"
        ds_dir.mkdir(exist_ok=True)
        for src in sorted(images_dir.iterdir()):
            if not (src.is_file() and src.suffix.lower() in SUPPORTED_EXTS):
                continue
            dst = ds_dir / src.name
            if dst.exists():
                continue
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                im.resize((w // factor, h // factor), Image.LANCZOS).save(dst)
        print(f"  Created {ds_dir.name}/ ({len(list(ds_dir.iterdir()))} images, factor {factor})")

    print(f"Train COLMAP dir built: {train_colmap_dir}")
    print(f"  Sparse model: {dst_sparse}")
    print(f"  Images: {len(blended_files)} blended images copied")
    return train_colmap_dir


# ---------------------------------------------------------------------------
# Mode: train — vanilla 3DGS on COLMAP-processed blended images
# ---------------------------------------------------------------------------

def run_specular_aware_training(
    scene: str,
    *,
    colmap_dir: Optional[Path] = None,
    result_dir: Optional[Path] = None,
    data_factor: int = FLAGSHIP_DATA_FACTOR,
    test_every: int = 8,
    max_steps: int = 30_000,
    save_steps: Optional[List[int]] = None,
    eval_steps: Optional[List[int]] = None,
    init_type: str = "sfm",
    pose_opt: bool = False,
    disable_viewer: bool = True,
    detector_version: str = FLAGSHIP_DETECTOR_VERSION,
) -> None:
    """Launch vanilla 3DGS training on the COLMAP-processed blended scene.

    Must be run in a WSL terminal with GPU access, NOT in the Agent sandbox.

    Args:
        scene: Scene name.
        colmap_dir: COLMAP output directory. Defaults to
            ``data/custom/<scene>_train_colmap_<tag>``.
        result_dir: Where to write checkpoints and renders. Defaults to
            ``results/<scene>/specular_aware_<tag>``.
        data_factor: gsplat image downsampling factor.
        test_every: Every N-th image (by sorted filename) is used for validation.
        max_steps: Training iterations.
        save_steps: Checkpoints to save.
        eval_steps: Steps at which to run evaluation; ``[-1]`` disables eval.
        init_type: ``"sfm"`` (COLMAP points) or ``"random"``.
        pose_opt: Whether to optimize camera poses (default False).
        disable_viewer: Disable the live viewer (default True).
    """
    paths = default_paths(scene, detector_version=detector_version)
    if colmap_dir is None:
        colmap_dir = paths["train_colmap_dir"]
    if result_dir is None:
        result_dir = paths["result_dir"]
    if save_steps is None:
        save_steps = [1_000, 3_000, 5_000, 10_000, 20_000, 30_000]
    if eval_steps is None:
        eval_steps = [-1]

    if not (colmap_dir / "sparse" / "0").exists() and not (colmap_dir / "sparse").exists():
        raise FileNotFoundError(
            f"COLMAP sparse model not found under {colmap_dir}. "
            "Run --mode train-colmap first."
        )

    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training vanilla 3DGS on blended data")
    print(f"  Scene: {scene}")
    print(f"  Data dir: {colmap_dir}")
    print(f"  Result dir: {result_dir}")
    print(f"  Data factor: {data_factor}")
    print(f"  Test every: {test_every}")
    print(f"  Max steps: {max_steps}")

    from gsplat.distributed import cli
    from simple_trainer import Config, Runner

    config = Config(
        disable_viewer=disable_viewer,
        data_dir=str(colmap_dir),
        data_factor=data_factor,
        result_dir=str(result_dir),
        test_every=test_every,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        init_type=init_type,
        pose_opt=pose_opt,
    )

    def train_fn(local_rank: int, world_rank: int, world_size: int, cfg_obj: object):
        runner = Runner(local_rank, world_rank, world_size, cfg_obj)
        print(f"SPECULAR_AWARE_TRAIN_COUNT={len(runner.trainset)}")
        print(f"SPECULAR_AWARE_VAL_COUNT={len(runner.valset)}")
        print(f"SPECULAR_AWARE_POSE_OPT={pose_opt}")
        print(f"SPECULAR_AWARE_INIT_TYPE={init_type}")
        runner.train()

    cli(train_fn, config, verbose=True)


# ---------------------------------------------------------------------------
# Mode: pipeline — run colmap, crossview, prepare, train-colmap, train
# ---------------------------------------------------------------------------

def run_pipeline(
    scene: str,
    *,
    blend_mode: str = FLAGSHIP_BLEND_MODE,
    specular_threshold: float = FLAGSHIP_SPECULAR_THRESHOLD,
    cross_view_weight: float = 1.0,
    crossview_min_views: int = 4,
    crossview_blur_sigma: float = 6.0,
    photometric_patch_radius: int = 3,
    broad_patch_radius: int = 9,
    max_reprojection_error: float = 2.5,
    variation_metric: str = "upper",
    exposure_normalize: bool = True,
    skip_colmap: bool = False,
    skip_crossview: bool = False,
    skip_prepare: bool = False,
    skip_train_colmap: bool = False,
    skip_train: bool = False,
    colmap_overwrite: bool = False,
    prepare_workers: int = DEFAULT_PREPARE_WORKERS,
    encode_workers_per_process: int = DEFAULT_ENCODE_WORKERS_PER_PROCESS,
    detector_backend: str = DEFAULT_DETECTOR_BACKEND,
    train_kwargs: Optional[Dict[str, Any]] = None,
    detector_version: str = FLAGSHIP_DETECTOR_VERSION,
) -> int:
    """Run the full nerfstudio-side pipeline with cross-view detection."""
    if train_kwargs is None:
        train_kwargs = {}

    paths = default_paths(
        scene,
        blend_mode=blend_mode,
        specular_threshold=specular_threshold,
        detector_version=detector_version,
    )

    step = 1
    if skip_colmap:
        print("Skipping COLMAP on original images (--skip-colmap).")
    else:
        print("=" * 60)
        print(f"STEP {step}/5: COLMAP on original images")
        print("=" * 60)
        run_colmap(
            scene,
            raw_dir=paths["raw_dir"],
            colmap_dir=paths["orig_colmap_dir"],
            overwrite=colmap_overwrite,
        )
        step += 1

    if skip_crossview:
        print("\nSkipping cross-view map generation (--skip-crossview).")
    else:
        print("\n" + "=" * 60)
        print(f"STEP {step}/5: Generate cross-view variance maps")
        print("=" * 60)
        run_crossview(
            scene,
            colmap_dir=paths["orig_colmap_dir"],
            output_dir=paths["crossview_dir"],
            min_views=crossview_min_views,
            blur_sigma=crossview_blur_sigma,
            patch_radius=photometric_patch_radius,
            broad_patch_radius=broad_patch_radius,
            max_reprojection_error=max_reprojection_error,
            variation_metric=variation_metric,
            exposure_normalize=exposure_normalize,
            overwrite=colmap_overwrite,
            detector_version=detector_version,
        )
        step += 1

    if skip_prepare:
        print("\nSkipping blended image preparation (--skip-prepare).")
    else:
        print("\n" + "=" * 60)
        print(f"STEP {step}/5: Prepare blended images")
        print("=" * 60)
        prepare_blended_images(
            scene,
            blend_mode=blend_mode,
            specular_threshold=specular_threshold,
            cross_view_weight=cross_view_weight,
            data_factor=int(train_kwargs.get("data_factor", FLAGSHIP_DATA_FACTOR)),
            overwrite=colmap_overwrite,
            detector_version=detector_version,
            detector_backend=detector_backend,
            workers=prepare_workers,
            encode_workers_per_process=encode_workers_per_process,
        )
        step += 1

    if skip_train_colmap:
        print("\nSkipping train COLMAP dir build (--skip-train-colmap).")
    else:
        print("\n" + "=" * 60)
        print(f"STEP {step}/5: Build training COLMAP directory")
        print("=" * 60)
        prepare_train_colmap(
            scene,
            orig_colmap_dir=paths["orig_colmap_dir"],
            blended_dir=paths["blended_dir"],
            train_colmap_dir=paths["train_colmap_dir"],
            overwrite=colmap_overwrite,
            detector_version=detector_version,
        )
        step += 1

    if skip_train:
        print("\nSkipping training (--skip-train).")
    else:
        print("\n" + "=" * 60)
        print(f"STEP {step}/5: Train vanilla 3DGS")
        print("=" * 60)
        train_kwargs.setdefault("colmap_dir", paths["train_colmap_dir"])
        train_kwargs.setdefault("result_dir", paths["result_dir"])
        train_kwargs.setdefault("detector_version", detector_version)
        run_specular_aware_training(scene, **train_kwargs)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Specular-aware 3DGS reconstruction pipeline (full images)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument(
        "--detector-version",
        choices=("v2", "v3"),
        default=FLAGSHIP_DETECTOR_VERSION,
        help="Detector profile; V3 underlies the V6 flagship and V2 remains reproducible",
    )
    parser.add_argument(
        "--mode",
        choices=("colmap", "crossview", "prepare", "train-colmap", "train", "pipeline", "dry-run"),
        default="pipeline",
        help="colmap=COLMAP on originals | crossview=variance maps | prepare=blend | "
             "train-colmap=build train dir | train=3DGS | pipeline=all",
    )
    parser.add_argument(
        "--delighted-dir", type=Path, help="Override delighted image directory"
    )
    parser.add_argument(
        "--blended-dir", type=Path, help="Override blended image directory"
    )
    parser.add_argument(
        "--scoremap-dir", type=Path, help="Override specular score-map directory"
    )
    parser.add_argument(
        "--colmap-dir", type=Path, help="Override COLMAP output directory"
    )
    parser.add_argument(
        "--result-dir", type=Path, help="Override training result directory"
    )
    parser.add_argument(
        "--blend-mode",
        choices=available_strategies(),
        default=FLAGSHIP_BLEND_MODE,
        help=(
            "Blending strategy; evidence_matte is the V6 flagship default; "
            "adaptive_feather preserves V4; weak_hysteresis_feather and "
            "weak_hysteresis_outer_feather preserve V5/V5.1"
        ),
    )
    parser.add_argument(
        "--specular-threshold",
        type=float,
        default=FLAGSHIP_SPECULAR_THRESHOLD,
        help="Soft threshold for specular probability",
    )
    parser.add_argument(
        "--cross-view-weight",
        type=float,
        default=1.0,
        help="Weight of cross-view variance cue in specular score fusion",
    )
    parser.add_argument(
        "--crossview-min-views", type=int, default=4,
        help="Minimum track observations for cross-view evidence",
    )
    parser.add_argument(
        "--crossview-blur-sigma", type=float, default=6.0,
        help="Coarse-grid interpolation sigma for cross-view maps",
    )
    parser.add_argument(
        "--photometric-patch-radius", type=int, default=3,
        help="Patch radius for seam-robust track luminance sampling",
    )
    parser.add_argument(
        "--broad-patch-radius", type=int, default=9,
        help="V3 broad patch radius for suppressing high-frequency jitter",
    )
    parser.add_argument(
        "--max-reprojection-error", type=float, default=2.5,
        help="Reject COLMAP points above this pixel error; <=0 disables",
    )
    parser.add_argument(
        "--variation-metric", choices=("upper", "std"), default="upper",
        help="Cross-view metric; std selects the legacy two-sided deviation",
    )
    parser.add_argument(
        "--no-exposure-normalize", action="store_true",
        help="Disable robust affine photometric alignment (ablation only)",
    )
    parser.add_argument(
        "--no-save-scoremaps",
        action="store_true",
        help="Do not save per-image specular score maps",
    )
    parser.add_argument(
        "--no-save-blend-weights",
        action="store_true",
        help="Do not save adaptive base/detail/chroma diagnostic maps",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N images in prepare/crossview mode (smoke tests)",
    )
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=DEFAULT_PREPARE_WORKERS,
        help=(
            "Image-level prepare workers; CPU uses processes and cuda-exact "
            "uses threads/streams; 4 is validated and 1 preserves serial execution"
        ),
    )
    parser.add_argument(
        "--detector-backend",
        choices=("cpu", "cuda-exact"),
        default=DEFAULT_DETECTOR_BACKEND,
        help=(
            "Box-filter backend used by V6 detection; cuda-exact preserves "
            "SciPy operation order and requires a normal WSL CUDA terminal"
        ),
    )
    parser.add_argument(
        "--encode-workers-per-process",
        type=int,
        default=DEFAULT_ENCODE_WORKERS_PER_PROCESS,
        help="Threads per image process for independent JPEG/diagnostic PNG encoding",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--skip-colmap", action="store_true", help="Skip COLMAP in pipeline mode"
    )
    parser.add_argument(
        "--skip-crossview", action="store_true", help="Skip cross-view maps in pipeline mode"
    )
    parser.add_argument(
        "--skip-prepare", action="store_true", help="Skip prepare in pipeline mode"
    )
    parser.add_argument(
        "--skip-train-colmap", action="store_true", help="Skip train-colmap in pipeline mode"
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training in pipeline mode"
    )
    parser.add_argument(
        "--data-factor",
        type=int,
        default=FLAGSHIP_DATA_FACTOR,
        help="gsplat image downsample factor",
    )
    parser.add_argument(
        "--test-every", type=int, default=8, help="Use every N-th image for validation"
    )
    parser.add_argument(
        "--max-steps", type=int, default=30_000, help="Training iterations"
    )
    parser.add_argument(
        "--pose-opt", action="store_true", help="Enable camera pose optimization"
    )

    args = parser.parse_args()

    try:
        if args.mode == "dry-run":
            paths = default_paths(
                args.scene,
                blend_mode=args.blend_mode,
                specular_threshold=args.specular_threshold,
                detector_version=args.detector_version,
            )
            print("Dry-run plan:")
            print(
                "  Profile:           "
                f"{pipeline_profile(args.detector_version, args.blend_mode, args.specular_threshold, args.data_factor)}"
            )
            print(f"  Detector:          {args.detector_version}")
            print(f"  Detector backend:  {args.detector_backend}")
            print(f"  Prepare workers:   {args.prepare_workers}")
            print(f"  Encode workers:    {args.encode_workers_per_process} per image worker")
            if args.blend_mode == "adaptive_feather":
                print(
                    "  Blend parameters:  "
                    f"{dict(ADAPTIVE_FEATHER_DEFAULT_PARAMETERS)}"
                )
            elif args.blend_mode == "weak_hysteresis_feather":
                print(
                    "  Blend parameters:  "
                    f"{dict(WEAK_HYSTERESIS_FEATHER_DEFAULT_PARAMETERS)}"
                )
            elif args.blend_mode == "weak_hysteresis_outer_feather":
                print(
                    "  Blend parameters:  "
                    f"{dict(WEAK_HYSTERESIS_OUTER_FEATHER_DEFAULT_PARAMETERS)}"
                )
            elif args.blend_mode == "evidence_matte":
                print(
                    "  Blend parameters:  "
                    f"{dict(EVIDENCE_MATTE_DEFAULT_PARAMETERS)}"
                )
            print(f"  Raw images:        {paths['raw_dir']}")
            print(f"  Delighted:         {paths['delighted_dir']}")
            print(f"  Blended output:    {paths['blended_dir']}")
            print(f"  Score maps:        {args.scoremap_dir or paths['scoremap_dir']}")
            print(f"  Original COLMAP:   {paths['orig_colmap_dir']}")
            print(f"  Cross-view maps:   {paths['crossview_dir']}")
            cross_config = (
                f"min_views={args.crossview_min_views}, "
                f"patch_radius={args.photometric_patch_radius}, "
                f"blur_sigma={args.crossview_blur_sigma}"
            )
            if args.detector_version == "v3":
                cross_config += f", broad_patch_radius={args.broad_patch_radius}"
            else:
                cross_config += f", metric={args.variation_metric}"
            print(f"  Cross-view config: {cross_config}")
            print(f"  Train COLMAP dir:  {paths['train_colmap_dir']}")
            print(f"  Training output:   {paths['result_dir']}")
            return 0

        if args.mode == "colmap":
            run_colmap(
                args.scene,
                raw_dir=None,
                colmap_dir=args.colmap_dir,
                overwrite=args.overwrite,
            )
            return 0

        if args.mode == "crossview":
            run_crossview(
                args.scene,
                colmap_dir=args.colmap_dir,
                min_views=args.crossview_min_views,
                blur_sigma=args.crossview_blur_sigma,
                patch_radius=args.photometric_patch_radius,
                broad_patch_radius=args.broad_patch_radius,
                max_reprojection_error=args.max_reprojection_error,
                variation_metric=args.variation_metric,
                exposure_normalize=not args.no_exposure_normalize,
                limit=args.limit,
                overwrite=args.overwrite,
                detector_version=args.detector_version,
            )
            return 0

        if args.mode == "prepare":
            prepare_blended_images(
                args.scene,
                delighted_dir=args.delighted_dir,
                output_dir=args.blended_dir,
                scoremap_dir=args.scoremap_dir,
                blend_mode=args.blend_mode,
                specular_threshold=args.specular_threshold,
                cross_view_weight=args.cross_view_weight,
                save_scoremaps=not args.no_save_scoremaps,
                save_blend_weights=not args.no_save_blend_weights,
                data_factor=args.data_factor,
                overwrite=args.overwrite,
                limit=args.limit,
                detector_version=args.detector_version,
                detector_backend=args.detector_backend,
                workers=args.prepare_workers,
                encode_workers_per_process=args.encode_workers_per_process,
            )
            return 0

        if args.mode == "train-colmap":
            prepare_train_colmap(
                args.scene,
                orig_colmap_dir=args.colmap_dir,
                blended_dir=args.blended_dir,
                overwrite=args.overwrite,
                detector_version=args.detector_version,
            )
            return 0

        if args.mode == "train":
            exp_paths = default_paths(
                args.scene,
                blend_mode=args.blend_mode,
                specular_threshold=args.specular_threshold,
                detector_version=args.detector_version,
            )
            run_specular_aware_training(
                args.scene,
                colmap_dir=args.colmap_dir or exp_paths["train_colmap_dir"],
                result_dir=args.result_dir or exp_paths["result_dir"],
                data_factor=args.data_factor,
                test_every=args.test_every,
                max_steps=args.max_steps,
                pose_opt=args.pose_opt,
                detector_version=args.detector_version,
            )
            return 0

        if args.mode == "pipeline":
            train_kwargs = {
                "data_factor": args.data_factor,
                "test_every": args.test_every,
                "max_steps": args.max_steps,
                "pose_opt": args.pose_opt,
            }
            if args.result_dir:
                train_kwargs["result_dir"] = args.result_dir

            return run_pipeline(
                args.scene,
                blend_mode=args.blend_mode,
                specular_threshold=args.specular_threshold,
                cross_view_weight=args.cross_view_weight,
                crossview_min_views=args.crossview_min_views,
                crossview_blur_sigma=args.crossview_blur_sigma,
                photometric_patch_radius=args.photometric_patch_radius,
                broad_patch_radius=args.broad_patch_radius,
                max_reprojection_error=args.max_reprojection_error,
                variation_metric=args.variation_metric,
                exposure_normalize=not args.no_exposure_normalize,
                skip_colmap=args.skip_colmap,
                skip_crossview=args.skip_crossview,
                skip_prepare=args.skip_prepare,
                skip_train_colmap=args.skip_train_colmap,
                skip_train=args.skip_train,
                colmap_overwrite=args.overwrite,
                prepare_workers=args.prepare_workers,
                encode_workers_per_process=args.encode_workers_per_process,
                detector_backend=args.detector_backend,
                train_kwargs=train_kwargs,
                detector_version=args.detector_version,
            )

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

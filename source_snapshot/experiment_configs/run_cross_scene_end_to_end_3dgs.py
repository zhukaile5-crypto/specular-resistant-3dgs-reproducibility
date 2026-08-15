#!/usr/bin/env python3
"""Run a traceable raw-versus-deglared 3DGS comparison for one scene.

The experiment deliberately compares complete input pipelines:

* original captures -> their original-image COLMAP reconstruction -> vanilla 3DGS;
* final deglared captures -> fresh deglared-image COLMAP -> vanilla 3DGS.

Both branches use all available captures for training.  The final measurement
renders a matched 3-height x 30-direction orbit and re-scores the renders with
the frozen single-view proxy detector.  It is an appearance-proxy comparison,
not a held-out or geometry-ground-truth benchmark.

The script writes numerical manifests and paired render files only.  It does
not create paper figures or contact sheets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results/end_to_end_3dgs_cross_scene_20260811"
DEFAULT_FUSION_SUFFIX = "_blended_evidence_matte_t0.3_v3_cross_scene_frozen_20260811"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.json")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, path)


def _require_dataset(path: Path, label: str) -> None:
    required = (path / "transforms.json", path / "images", path / "images_4")
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError("{} is not a usable nerfstudio/COLMAP dataset: {}".format(label, missing))


def _run_cpu_colmap(
    input_dir: Path,
    output_dir: Path,
    matching_method: str,
    workers: int,
) -> None:
    """Run fresh memory-bounded CPU COLMAP without overwriting an existing run."""
    if output_dir.exists():
        try:
            _require_dataset(output_dir, "Existing re-estimated dataset")
        except FileNotFoundError as exc:
            raise FileExistsError(
                "Refusing to reuse or overwrite an incomplete COLMAP directory: {}. "
                "Inspect and remove only that failed generated directory before retrying."
                .format(output_dir)
            ) from exc
        print("Reusing existing re-estimated COLMAP dataset: {}".format(output_dir), flush=True)
        return
    if workers < 1:
        raise ValueError("--colmap-workers must be positive")
    from nerfstudio.process_data import colmap_utils
    from nerfstudio.process_data.process_data_utils import copy_images

    colmap = shutil.which("colmap")
    if colmap is None:
        raise RuntimeError("COLMAP executable not found; activate the nerfstudio environment.")
    output_dir.mkdir(parents=True)
    copied = copy_images(
        data=input_dir,
        image_dir=output_dir / "images",
        image_prefix="frame_",
        num_downscales=3,
        same_dimensions=True,
        verbose=False,
    )
    if not copied:
        raise RuntimeError("No input images were copied from {}".format(input_dir))
    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir()
    database = colmap_dir / "database.db"
    command_prefix = [str(colmap)]
    commands = [
        command_prefix
        + [
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(output_dir / "images"),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "OPENCV",
            "--SiftExtraction.use_gpu",
            "0",
            "--SiftExtraction.num_threads",
            str(workers),
        ],
        command_prefix
        + [
            "{}_matcher".format(matching_method),
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            "0",
            "--SiftMatching.num_threads",
            str(workers),
        ],
    ]
    if matching_method == "vocab_tree":
        commands[1].extend(
            ["--VocabTreeMatching.vocab_tree_path", str(colmap_utils.get_vocab_tree())]
        )
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir()
    commands.append(
        command_prefix
        + [
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(output_dir / "images"),
            "--output_path",
            str(sparse_dir),
            "--Mapper.num_threads",
            str(workers),
            "--Mapper.ba_global_function_tolerance",
            "1e-6",
        ]
    )
    commands.append(
        command_prefix
        + [
            "bundle_adjuster",
            "--input_path",
            str(sparse_dir / "0"),
            "--output_path",
            str(sparse_dir / "0"),
            "--BundleAdjustment.refine_principal_point",
            "1",
        ]
    )
    for command in commands:
        print("Running memory-bounded CPU COLMAP:\n  {}".format(" ".join(command)), flush=True)
        subprocess.run(command, check=True)
    colmap_utils.colmap_to_json(sparse_dir / "0", output_dir)
    _require_dataset(output_dir, "Fresh re-estimated dataset")


def _checkpoint_path(result_dir: Path, max_steps: int) -> Path:
    checkpoint = result_dir / "ckpts/ckpt_{:05d}_rank0.pt".format(max_steps - 1)
    if not checkpoint.is_file():
        raise FileNotFoundError("Expected completed 3DGS checkpoint: {}".format(checkpoint))
    return checkpoint


def _registered_frame_count(dataset_dir: Path) -> int:
    with (dataset_dir / "transforms.json").open(encoding="utf-8") as handle:
        return len(json.load(handle).get("frames", []))


def _ensure_gsplat_adapter(source_dir: Path, adapter_dir: Path) -> Path:
    """Expose a nerfstudio layout through gsplat's root-level ``sparse`` convention.

    The adapters contain only relative symlinks, preserving generated source
    datasets unchanged.  This also makes the exact training inputs explicit in
    the experiment output directory.
    """
    sparse_source = source_dir / "sparse"
    if not sparse_source.is_dir():
        sparse_source = source_dir / "colmap/sparse"
    required = {
        "images": source_dir / "images",
        "images_4": source_dir / "images_4",
        "sparse": sparse_source,
        "transforms.json": source_dir / "transforms.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot build gsplat adapter; missing {}".format(missing))
    if adapter_dir.exists():
        if all((adapter_dir / name).exists() for name in required):
            return adapter_dir
        raise FileExistsError("Refusing to overwrite incomplete gsplat adapter: {}".format(adapter_dir))
    adapter_dir.mkdir(parents=True)
    for name, target in required.items():
        relative_target = os.path.relpath(target, start=adapter_dir)
        (adapter_dir / name).symlink_to(relative_target, target_is_directory=target.is_dir())
    for scale in (2, 8):
        target = source_dir / "images_{}".format(scale)
        if target.is_dir():
            (adapter_dir / "images_{}".format(scale)).symlink_to(
                os.path.relpath(target, start=adapter_dir), target_is_directory=True
            )
    return adapter_dir


def _write_protocol(
    path: Path,
    *,
    scene: str,
    raw_input: Path,
    raw_data: Path,
    deglared_input: Path,
    deglared_data: Path,
    matching_method: str,
    max_steps: int,
    seed: int,
) -> None:
    _atomic_json(
        path,
        {
            "schema": "cross_scene_end_to_end_3dgs_v1",
            "scene": scene,
            "comparison": {
                "raw_branch": "original images -> existing original-image COLMAP reconstruction -> vanilla 3DGS",
                "deglared_branch": "final deglared images -> fresh COLMAP -> vanilla 3DGS",
                "shared_training_protocol": "all available registered images; same vanilla 3DGS configuration and random seed",
            },
            "paths": {
                "raw_input_images": str(raw_input),
                "raw_colmap_dataset": str(raw_data),
                "deglared_input_images": str(deglared_input),
                "deglared_reestimated_colmap_dataset": str(deglared_data),
            },
            "colmap": {
                "matching_method": matching_method,
                "gpu": False,
                "reason": "memory-bounded CPU pose re-estimation for the deglared branch; the raw branch reuses its pre-existing original-image reconstruction",
            },
            "training": {"max_steps": max_steps, "seed": seed, "all_images_used": True},
            "render_evaluation": {
                "trajectory": "3 camera heights x 30 azimuth directions",
                "detector": "frozen V3 single-view proxy detector",
                "important_limitations": [
                    "The 90 rendered views are paired trajectory samples, not 90 independent scenes.",
                    "All captures are used for 3DGS training; this is not a held-out image-quality test.",
                    "No reflection-free or geometry ground truth is available, so metrics are proxy evidence rather than physical reflectance or depth accuracy.",
                    "The two branches intentionally differ in both input images and recovered COLMAP poses; the result measures the complete-pipeline effect rather than isolating either factor.",
                    "Registered-image counts are recorded for both branches. If they differ, the result must be read as a complete-pipeline comparison rather than a fixed-input-count ablation.",
                ],
            },
        },
    )


def run(args: argparse.Namespace) -> None:
    scene = args.scene
    raw_input = args.raw_image_dir or PROJECT_ROOT / "raw_images" / scene
    raw_data = args.raw_data_dir or PROJECT_ROOT / "data/custom" / scene
    deglared_input = args.deglared_image_dir or (
        PROJECT_ROOT / "data/custom" / (scene + DEFAULT_FUSION_SUFFIX)
    )
    deglared_data = args.deglared_colmap_dir or (
        PROJECT_ROOT / "data/custom" / (scene + "_end2end_deglared_colmap_20260811")
    )
    scene_root = args.results_root / scene
    raw_result = scene_root / "raw_image_colmap_3dgs"
    deglared_result = scene_root / "deglared_reestimated_colmap_3dgs"
    orbit_dir = scene_root / "matched_orbit_3x30"
    raw_training_data = scene_root / "input_adapters/raw_image_colmap"
    deglared_training_data = scene_root / "input_adapters/deglared_reestimated_colmap"
    if not raw_input.is_dir():
        raise FileNotFoundError("Original input directory not found: {}".format(raw_input))
    _require_dataset(raw_data, "Original-image COLMAP dataset")
    if not deglared_input.is_dir():
        raise FileNotFoundError("Final deglared input directory not found: {}".format(deglared_input))
    _write_protocol(
        scene_root / "experiment_protocol.json",
        scene=scene,
        raw_input=raw_input,
        raw_data=raw_data,
        deglared_input=deglared_input,
        deglared_data=deglared_data,
        matching_method=args.matching_method,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    if args.stage in ("colmap", "all"):
        _run_cpu_colmap(
            deglared_input,
            deglared_data,
            args.matching_method,
            args.colmap_workers,
        )

    if args.stage in ("colmap", "train", "evaluate", "all"):
        _require_dataset(deglared_data, "Deglared-image COLMAP dataset")
        registered = {
            "raw_registered_frames": _registered_frame_count(raw_data),
            "deglared_registered_frames": _registered_frame_count(deglared_data),
        }
        registered["counts_match"] = (
            registered["raw_registered_frames"]
            == registered["deglared_registered_frames"]
        )
        _atomic_json(scene_root / "registered_frame_counts.json", registered)
        if not registered["counts_match"]:
            print(
                "WARNING: raw/deglared registered-frame counts differ: {}".format(
                    registered
                ),
                flush=True,
            )

    if args.stage in ("train", "all"):
        _require_dataset(deglared_data, "Deglared-image COLMAP dataset")
        _ensure_gsplat_adapter(raw_data, raw_training_data)
        _ensure_gsplat_adapter(deglared_data, deglared_training_data)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable in this terminal. COLMAP preparation can run on CPU, "
                "but vanilla 3DGS training must be launched after CUDA access is restored."
            )
        from experiment_configs.full96_novel_view_comparison import train_all_images

        if not raw_result.exists():
            print("Training original-image baseline: {}".format(scene), flush=True)
            train_all_images(raw_training_data, raw_result, args.max_steps, args.seed)
        else:
            _checkpoint_path(raw_result, args.max_steps)
            print("Reusing completed original-image model: {}".format(raw_result), flush=True)
        if not deglared_result.exists():
            print("Training deglared + re-estimated-pose model: {}".format(scene), flush=True)
            train_all_images(deglared_training_data, deglared_result, args.max_steps, args.seed)
        else:
            _checkpoint_path(deglared_result, args.max_steps)
            print("Reusing completed deglared model: {}".format(deglared_result), flush=True)

    if args.stage in ("evaluate", "all"):
        _require_dataset(deglared_data, "Deglared-image COLMAP dataset")
        _ensure_gsplat_adapter(raw_data, raw_training_data)
        _ensure_gsplat_adapter(deglared_data, deglared_training_data)
        raw_checkpoint = _checkpoint_path(raw_result, args.max_steps)
        deglared_checkpoint = _checkpoint_path(deglared_result, args.max_steps)
        from experiment_configs.evaluate_matched_orbit_specularity import (
            build_plan,
            evaluate,
            render_all,
        )

        build_plan(
            raw_training_data,
            deglared_training_data,
            orbit_dir,
            args.directions,
            args.height_percentiles,
        )
        render_all(
            raw_training_data,
            deglared_training_data,
            raw_checkpoint,
            deglared_checkpoint,
            orbit_dir,
            args.overwrite_renders,
        )
        summary = evaluate(
            orbit_dir,
            args.score_workers,
            save_figures=False,
            save_contact_sheets=False,
            scene_name=scene,
        )
        print(json.dumps(summary["overall"], indent=2), flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--stage", choices=("colmap", "train", "evaluate", "all"), default="all")
    parser.add_argument("--raw-image-dir", type=Path)
    parser.add_argument("--raw-data-dir", type=Path)
    parser.add_argument("--deglared-image-dir", type=Path)
    parser.add_argument("--deglared-colmap-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--matching-method", choices=("vocab_tree", "sequential"), default="vocab_tree")
    parser.add_argument(
        "--colmap-workers",
        type=int,
        default=2,
        help="Bound CPU SIFT/matching/mapper threads to avoid out-of-memory termination.",
    )
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--directions", type=int, default=30)
    parser.add_argument("--height-percentiles", type=float, nargs=3, default=(20.0, 50.0, 80.0))
    parser.add_argument("--score-workers", type=int, default=3)
    parser.add_argument("--overwrite-renders", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

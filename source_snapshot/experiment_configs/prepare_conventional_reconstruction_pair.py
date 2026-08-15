#!/usr/bin/env python3
"""Prepare matched raw and V6/V7-deglared COLMAP datasets for vanilla 3DGS.

The camera model, COLMAP poses, sparse points, frame split, and image scale are
identical in both outputs. Only the training image content differs. Existing
sources are read only; this script refuses to replace an existing pair unless
``--overwrite`` is passed explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
DATA_CUSTOM = PROJECT_ROOT / "data" / "custom"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _image_files(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _digits(name: str) -> str:
    value = "".join(character for character in Path(name).stem if character.isdigit())
    if not value:
        raise ValueError("Image name has no numeric frame identifier: {}".format(name))
    return str(int(value))


def _atomic_json(path: Path, payload: Dict[str, object]) -> None:
    partial = path.with_name(".{}.partial".format(path.name))
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(partial, path)


def _link_or_copy(source: Path, destination: Path) -> str:
    """Create a same-filesystem hard link, falling back to a metadata copy."""
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _reset_output(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            "Output already exists: {}. Use --overwrite only after reviewing it.".format(path)
        )
    shutil.rmtree(path)


def _source_v6_image(v6_dir: Path, frame_name: str) -> Path:
    numeric_id = int(_digits(frame_name))
    for extension in (".jpg", ".jpeg", ".png"):
        candidate = v6_dir / ("{:04d}".format(numeric_id) + extension)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No V6/V7 image for COLMAP frame {} in {}".format(frame_name, v6_dir)
    )


def _write_downsampled(source: Path, destination: Path, factor: int) -> None:
    with Image.open(source) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        reduced = image.resize((width // factor, height // factor), Image.LANCZOS)
        reduced.save(destination, quality=95, subsampling=0)


def _build_variant(
    *,
    name: str,
    output_dir: Path,
    sparse_source: Path,
    transforms_source: Path,
    frame_sources: Iterable[Tuple[str, Path]],
    raw_downsample_dir: Path,
    factor: int,
    overwrite: bool,
) -> Dict[str, object]:
    _reset_output(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(sparse_source, output_dir / "sparse")
    shutil.copy2(transforms_source, output_dir / "transforms.json")

    images_dir = output_dir / "images"
    images_dir.mkdir()
    downsample_dir = output_dir / "images_{}".format(factor)
    downsample_dir.mkdir()

    records = []
    for frame_name, source in frame_sources:
        destination = images_dir / frame_name
        image_method = _link_or_copy(source, destination)
        raw_downsample = raw_downsample_dir / frame_name
        downsample_destination = downsample_dir / frame_name
        if name == "raw" and raw_downsample.is_file():
            downsample_method = _link_or_copy(raw_downsample, downsample_destination)
        else:
            _write_downsampled(source, downsample_destination, factor)
            downsample_method = "resized"
        records.append(
            {
                "frame": frame_name,
                "source": str(source),
                "image_method": image_method,
                "downsample_method": downsample_method,
            }
        )

    _atomic_json(
        output_dir / "pair_manifest.json",
        {
            "variant": name,
            "sparse_source": str(sparse_source),
            "transforms_source": str(transforms_source),
            "data_factor": factor,
            "frame_count": len(records),
            "frames": records,
        },
    )
    return {"output": str(output_dir), "frame_count": len(records)}


def prepare_pair(scene: str, *, data_factor: int, overwrite: bool) -> Dict[str, object]:
    """Create the raw and V6/V7 paired training inputs for one scene."""
    scene_root = DATA_CUSTOM / scene
    sparse_source = scene_root / "colmap" / "sparse"
    transforms_source = scene_root / "transforms.json"
    raw_images = scene_root / "images"
    raw_downsample = scene_root / "images_{}".format(data_factor)
    v6_images = DATA_CUSTOM / "{}_blended_evidence_matte_t0.3_v3".format(scene)
    if not sparse_source.is_dir() or not transforms_source.is_file():
        raise FileNotFoundError("Missing original COLMAP model under {}".format(scene_root))
    if not raw_images.is_dir() or not raw_downsample.is_dir() or not v6_images.is_dir():
        raise FileNotFoundError("Missing images, downsampled images, or V6 output for {}".format(scene))

    frames = _image_files(raw_images)
    if not frames:
        raise RuntimeError("No original COLMAP frame images in {}".format(raw_images))
    raw_sources = [(frame.name, frame) for frame in frames]
    v6_sources = [(frame.name, _source_v6_image(v6_images, frame.name)) for frame in frames]
    if len(v6_sources) != len(raw_sources):
        raise RuntimeError("Raw and V6 frame counts differ")

    raw_output = DATA_CUSTOM / "{}_conventional_raw_colmap".format(scene)
    v6_output = DATA_CUSTOM / "{}_conventional_v6_colmap".format(scene)
    raw_result = _build_variant(
        name="raw",
        output_dir=raw_output,
        sparse_source=sparse_source,
        transforms_source=transforms_source,
        frame_sources=raw_sources,
        raw_downsample_dir=raw_downsample,
        factor=data_factor,
        overwrite=overwrite,
    )
    v6_result = _build_variant(
        name="v6_v7_deglared",
        output_dir=v6_output,
        sparse_source=sparse_source,
        transforms_source=transforms_source,
        frame_sources=v6_sources,
        raw_downsample_dir=raw_downsample,
        factor=data_factor,
        overwrite=overwrite,
    )
    pair_manifest = {
        "scene": scene,
        "comparison": "raw original images versus V6/V7 byte-identical deglared images",
        "shared_colmap_sparse_model": str(sparse_source),
        "shared_camera_poses": True,
        "shared_data_factor": data_factor,
        "raw": raw_result,
        "v6_v7_deglared": v6_result,
    }
    _atomic_json(DATA_CUSTOM / "{}_conventional_reconstruction_pair.json".format(scene), pair_manifest)
    return pair_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.data_factor <= 1:
        parser.error("--data-factor must be greater than one for the paired benchmark")
    print(json.dumps(prepare_pair(args.scene, data_factor=args.data_factor, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

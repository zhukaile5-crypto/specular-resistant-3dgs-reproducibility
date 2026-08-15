#!/usr/bin/env python3
"""Build paper-ready V1--V6 comparison sheets from preserved outputs.

The script is intentionally read-only with respect to model outputs: it crops
and tiles existing images only. V5 uses the preserved, validated candidate for
the selected representative frames because its full-dataset directory was
archived before this figure was requested.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
SCENE = "photo_scene6"
OUTPUT_DIR = PROJECT_ROOT / "results" / "version_evolution_figures"

GLOBAL_CROP = (600, 1950, 2500, 3600)
CUP_CROP = (1100, 1750, 2200, 2700)


def _sources(image_name: str) -> List[Tuple[str, Path]]:
    """Return the preserved image for each comparison panel."""
    return [
        ("Original", PROJECT_ROOT / "raw_images" / SCENE / image_name),
        (
            "StableDelight",
            PROJECT_ROOT / "data" / "custom" / f"{SCENE}_delighted" / image_name.replace(".jpg", ".png"),
        ),
        ("V1", PROJECT_ROOT / "data" / "custom" / f"{SCENE}_blended_soft_t0.3" / image_name),
        ("V2", PROJECT_ROOT / "data" / "custom" / f"{SCENE}_blended_soft_t0.3_v2" / image_name),
        ("V3", PROJECT_ROOT / "data" / "custom" / f"{SCENE}_blended_soft_t0.3_v3" / image_name),
        (
            "V4",
            PROJECT_ROOT / "data" / "custom" / f"{SCENE}_blended_adaptive_feather_t0.3_v3" / image_name,
        ),
        (
            "V5 (candidate)",
            PROJECT_ROOT / "results" / SCENE / "v5_candidate_validation" / Path(image_name).stem / "v5.jpg",
        ),
        (
            "V6 (Flagship)",
            PROJECT_ROOT / "data" / "custom" / f"{SCENE}_blended_evidence_matte_t0.3_v3" / image_name,
        ),
    ]


def _final_method_sources(image_name: str) -> List[Tuple[str, Path]]:
    """Return the original, direct delighting, and final proposed output."""
    sources = _sources(image_name)
    return [
        sources[0],
        sources[1],
        (
            "Proposed method",
            PROJECT_ROOT
            / "data"
            / "custom"
            / f"{SCENE}_blended_evidence_matte_t0.3_v3"
            / image_name,
        ),
    ]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(
    source: Path,
    label: str,
    crop: Tuple[int, int, int, int],
    panel_width: int,
    brightness: float,
) -> Image.Image:
    if not source.is_file():
        raise FileNotFoundError("Missing comparison source: {}".format(source))
    with Image.open(source) as handle:
        image = handle.convert("RGB").crop(crop)
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    height = max(int(round(image.height * panel_width / image.width)), 1)
    panel = image.resize((panel_width, height), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(panel, "RGBA")
    font = _font(max(panel_width // 27, 22))
    left, top = max(panel_width // 42, 12), max(height // 40, 12)
    box = draw.textbbox((left, top), label, font=font)
    pad = max(panel_width // 90, 7)
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=pad,
        fill=(0, 0, 0, 180),
    )
    draw.text((left, top), label, font=font, fill=(255, 255, 255, 255))
    return panel


def _tile(panels: Sequence[Image.Image], columns: int = 4, gap: int = 10) -> Image.Image:
    if not panels:
        raise ValueError("No panels supplied")
    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            gap + columns * (panel_width + gap),
            gap + rows * (panel_height + gap),
        ),
        "#151515",
    )
    for index, panel in enumerate(panels):
        x = gap + (index % columns) * (panel_width + gap)
        y = gap + (index // columns) * (panel_height + gap)
        canvas.paste(panel, (x, y))
    return canvas


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_png_atomic(image: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(".{}.partial.png".format(output.stem))
    image.save(partial, format="PNG", optimize=True)
    os.replace(partial, output)


def _build(
    *,
    image_name: str,
    crop: Tuple[int, int, int, int],
    panel_width: int,
    output_name: str,
    description: str,
    brightness: float = 1.0,
    sources: Optional[Sequence[Tuple[str, Path]]] = None,
) -> Dict[str, object]:
    panels = []
    source_rows = []
    for label, source in _sources(image_name) if sources is None else sources:
        panels.append(_panel(source, label, crop, panel_width, brightness))
        source_rows.append({"label": label, "path": str(source), "sha256": _sha256(source)})
    output = OUTPUT_DIR / output_name
    _save_png_atomic(_tile(panels), output)
    return {
        "description": description,
        "image": image_name,
        "crop_xyxy": list(crop),
        "display_brightness_factor": brightness,
        "output": str(output),
        "sources": source_rows,
    }


def main() -> int:
    records = [
        _build(
            image_name="0051.jpg",
            crop=GLOBAL_CROP,
            panel_width=700,
            output_name="photo_scene6_0051_v1_v6_global.png",
            description="Central chair/table reflection from photo_scene6 frame 0051.",
        ),
        _build(
            image_name="0051.jpg",
            crop=GLOBAL_CROP,
            panel_width=700,
            output_name="photo_scene6_0051_v1_v6_global_bright.png",
            description=(
                "Display-only brightened version of the central chair/table reflection; "
                "the same 1.10 brightness factor is applied to every panel."
            ),
            brightness=1.10,
        ),
        _build(
            image_name="0012.jpg",
            crop=CUP_CROP,
            panel_width=700,
            output_name="photo_scene6_0012_v1_v6_cup_detail.png",
            description="More frontal cup, logo, and fine wood-texture detail from frame 0012.",
        ),
        _build(
            image_name="0051.jpg",
            crop=GLOBAL_CROP,
            panel_width=930,
            output_name="photo_scene6_0051_final_method_global.png",
            description=(
                "Final-method comparison of the central chair/table reflection; "
                "the same 1.10 display brightness factor is applied to every panel."
            ),
            brightness=1.10,
            sources=_final_method_sources("0051.jpg"),
        ),
        _build(
            image_name="0012.jpg",
            crop=CUP_CROP,
            panel_width=930,
            output_name="photo_scene6_0012_final_method_cup_detail.png",
            description=(
                "Final-method comparison of the cup, emblem, wood texture, and "
                "narrow chair reflection."
            ),
            sources=_final_method_sources("0012.jpg"),
        ),
    ]
    manifest = OUTPUT_DIR / "v1_v6_visual_comparison_manifest.json"
    partial = manifest.with_name(".{}.partial".format(manifest.name))
    with partial.open("w", encoding="utf-8") as handle:
        json.dump({"scene": SCENE, "records": records}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(partial, manifest)
    for record in records:
        print(record["output"])
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

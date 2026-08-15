#!/usr/bin/env python3
"""Archive user-selected A/C artifact crops and build a lossless two-row sheet."""

import json
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
SOURCE_DIR = Path("/mnt/c/Users/Administrator/Desktop/8_images_AC")
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/photo_scene6/full96_reestimated_pose/qualitative_artifact_crops/a_vs_c"
)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

GROUPS: Sequence[Tuple[str, str, str]] = (
    ("A", "raw_original_pose", "A: raw images + original COLMAP pose"),
    ("C", "v6_v7_reestimated_pose", "C: V6/V7 deglared images + re-estimated COLMAP pose"),
)


def copy_sources() -> List[Tuple[str, str, List[Path]]]:
    """Copy the eight source crops with stable, experiment-meaningful names."""
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(SOURCE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for group_number, (group, tag, label) in enumerate(GROUPS, start=1):
        paths = []
        for crop_index in range(1, 5):
            source = SOURCE_DIR / "{}.{}.png".format(group_number, crop_index)
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = OUTPUT_DIR / "photo_scene6_{}_artifact_crop_{:02d}.png".format(
                tag, crop_index
            )
            if not destination.exists():
                shutil.copy2(source, destination)
            paths.append(destination)
        copied.append((group, label, paths))
    return copied


def build_montage(copied: List[Tuple[str, str, List[Path]]]) -> Path:
    """Arrange A/C crops in matching two-row, four-column order without resampling."""
    images = [[Image.open(path).convert("RGB") for path in paths] for _, _, paths in copied]
    width = max(image.width for row in images for image in row)
    height = max(image.height for row in images for image in row)
    title_height, row_label_width = 56, 260
    canvas = Image.new(
        "RGB", (row_label_width + 4 * width, title_height + 2 * height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT_PATH, 27)
    row_font = ImageFont.truetype(FONT_PATH, 20)
    draw.rectangle((0, 0, canvas.width, title_height), fill=(30, 30, 30))
    draw.text(
        (18, 15),
        "Artifact-focused comparison | matched A/C crops selected by user",
        font=title_font,
        fill="white",
    )
    row_colors = ((235, 240, 245), (239, 247, 238))
    for row, ((group, label, _), row_images) in enumerate(zip(copied, images)):
        y = title_height + row * height
        draw.rectangle((0, y, row_label_width, y + height), fill=row_colors[row])
        label_lines = [group + " model"] + label.split(" + ")
        for index, line in enumerate(label_lines):
            draw.text(
                (14, y + height // 2 - 35 + 26 * index),
                line,
                font=row_font,
                fill=(25, 25, 25),
            )
        for column, image in enumerate(row_images):
            x = row_label_width + column * width + (width - image.width) // 2
            image_y = y + (height - image.height) // 2
            canvas.paste(image, (x, image_y))
    output = OUTPUT_DIR / "photo_scene6_A_vs_C_floating_specular_artifact_crops_2x4.png"
    canvas.save(output)
    return output


def main() -> None:
    copied = copy_sources()
    montage = build_montage(copied)
    manifest = {
        "source_directory": str(SOURCE_DIR),
        "group_assumption": "1.1-1.4 are A; 2.1-2.4 are C",
        "groups": [
            {
                "group": group,
                "description": label,
                "copied_files": [str(path) for path in paths],
            }
            for group, label, paths in copied
        ],
        "montage": str(montage),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(montage)


if __name__ == "__main__":
    main()

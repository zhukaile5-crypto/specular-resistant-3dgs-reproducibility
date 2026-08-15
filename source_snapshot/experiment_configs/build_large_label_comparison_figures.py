#!/usr/bin/env python3
"""Create non-destructive large-label versions of the A/C comparison figures."""

from pathlib import Path
from math import cos, pi, sin
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PRESENTATION_DIR = PROJECT_ROOT / "results/photo_scene6/full96_reestimated_pose/presentation_figures"


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def _load(paths: Sequence[Path]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def build_novel_view_sheet() -> Path:
    """Recompose the existing A/C three-view sheet with prominent tile labels."""
    root = (
        PROJECT_ROOT
        / "results/photo_scene6/full96_reestimated_pose/a_vs_c_novel_view_comparison_0051_0055_0056"
    )
    anchors = (51, 55, 56)
    top_images = _load([root / "near_{:04d}_a_raw_original_pose.png".format(anchor) for anchor in anchors])
    bottom_images = _load(
        [root / "near_{:04d}_c_v6_v7_reestimated_pose.png".format(anchor) for anchor in anchors]
    )
    width, height = top_images[0].size
    if any(image.size != (width, height) for image in top_images + bottom_images):
        raise RuntimeError("Novel-view images must share a common resolution")
    bar_height = 132
    canvas = Image.new("RGB", (3 * width, 2 * (height + bar_height)), "black")
    draw = ImageDraw.Draw(canvas)
    rows = (
        ("Raw-image-trained 3DGS", top_images, (225, 236, 248)),
        ("Deglared-image-trained 3DGS", bottom_images, (228, 245, 231)),
    )
    for row_index, (label, images, color) in enumerate(rows):
        for column, (anchor, image) in enumerate(zip(anchors, images)):
            x = column * width
            y = row_index * (height + bar_height)
            draw.rectangle((x, y, x + width, y + bar_height), fill=color)
            draw.text(
                (x + 18, y + 18),
                label,
                font=_font(48),
                fill=(20, 20, 20),
            )
            draw.text(
                (x + 20, y + 78),
                "Pose {}".format(column + 1),
                font=_font(34),
                fill=(20, 20, 20),
            )
            canvas.paste(image, (x, y + bar_height))
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    output = PRESENTATION_DIR / "raw_vs_deglared_matched_novel_views_2x3.png"
    canvas.save(output)
    return output


def build_novel_view_sheet_1x6() -> Path:
    """Place each raw/deglared pose pair side by side in a single compact row."""
    root = (
        PROJECT_ROOT
        / "results/photo_scene6/full96_reestimated_pose/a_vs_c_novel_view_comparison_0051_0055_0056"
    )
    anchors = (51, 55, 56)
    panels: List[Tuple[str, int, Image.Image, Tuple[int, int, int]]] = []
    for pose_index, anchor in enumerate(anchors, start=1):
        panels.append(
            (
                "Raw-image-trained",
                pose_index,
                Image.open(root / "near_{:04d}_a_raw_original_pose.png".format(anchor)).convert("RGB"),
                (225, 236, 248),
            )
        )
        panels.append(
            (
                "Deglared-image-trained",
                pose_index,
                Image.open(root / "near_{:04d}_c_v6_v7_reestimated_pose.png".format(anchor)).convert("RGB"),
                (228, 245, 231),
            )
        )

    width, height = panels[0][2].size
    if any(image.size != (width, height) for _, _, image, _ in panels):
        raise RuntimeError("Novel-view images must share a common resolution")

    bar_height = 116
    canvas = Image.new("RGB", (len(panels) * width, height + bar_height), "black")
    draw = ImageDraw.Draw(canvas)
    for column, (label, pose_index, image, color) in enumerate(panels):
        x = column * width
        draw.rectangle((x, 0, x + width, bar_height), fill=color)
        draw.text((x + 14, 14), label, font=_font(34), fill=(20, 20, 20))
        draw.text((x + 16, 68), "Pose {}".format(pose_index), font=_font(27), fill=(20, 20, 20))
        canvas.paste(image, (x, bar_height))

    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    output = PRESENTATION_DIR / "raw_vs_deglared_matched_novel_views_1x6.png"
    canvas.save(output)
    return output


def build_novel_view_sheet_1x6_reflection_marked() -> Path:
    """Build the final sheet with matched reflection-region annotations."""
    root = (
        PROJECT_ROOT
        / "results/photo_scene6/full96_reestimated_pose/a_vs_c_novel_view_comparison_0051_0055_0056"
    )
    anchors = (51, 55, 56)
    # Coordinates are expressed in each native 767x1023 render. The same
    # annotation is applied to both members of a matched raw/deglared pair.
    # Pose 1 and Pose 3 use tilted ellipses whose lower arcs extend beyond the
    # panel, producing open boundary-clipped marks for edge-touching glare.
    annotation_specs = (
        ("rotated", (170, 977, 150, 86, 62)),
        ("axis_aligned", (214, 790, 456, 994)),
        ("rotated", (273, 1010, 118, 68, 68)),
    )
    panels: List[Tuple[str, int, Image.Image, Tuple[int, int, int]]] = []
    for pose_index, (anchor, annotation_spec) in enumerate(
        zip(anchors, annotation_specs), start=1
    ):
        pair = (
            (
                "Raw-image-trained",
                root / "near_{:04d}_a_raw_original_pose.png".format(anchor),
                (225, 236, 248),
            ),
            (
                "Deglared-image-trained",
                root / "near_{:04d}_c_v6_v7_reestimated_pose.png".format(anchor),
                (228, 245, 231),
            ),
        )
        for label, path, color in pair:
            image = Image.open(path).convert("RGB")
            annotation = ImageDraw.Draw(image)
            annotation_kind, values = annotation_spec
            if annotation_kind == "axis_aligned":
                annotation.ellipse(values, outline=(0, 214, 196), width=10)
            else:
                center_x, center_y, radius_a, radius_b, angle_degrees = values
                angle = angle_degrees * pi / 180.0
                points = []
                for sample in range(361):
                    parameter = sample * 2.0 * pi / 360.0
                    local_x = radius_a * cos(parameter)
                    local_y = radius_b * sin(parameter)
                    points.append(
                        (
                            round(center_x + local_x * cos(angle) - local_y * sin(angle)),
                            round(center_y + local_x * sin(angle) + local_y * cos(angle)),
                        )
                    )
                annotation.line(points, fill=(0, 214, 196), width=10, joint="curve")
            panels.append((label, pose_index, image, color))

    width, height = panels[0][2].size
    if any(image.size != (width, height) for _, _, image, _ in panels):
        raise RuntimeError("Novel-view images must share a common resolution")

    bar_height = 116
    canvas = Image.new("RGB", (len(panels) * width, height + bar_height), "black")
    draw = ImageDraw.Draw(canvas)
    for column, (label, pose_index, image, color) in enumerate(panels):
        x = column * width
        draw.rectangle((x, 0, x + width, bar_height), fill=color)
        draw.text((x + 14, 14), label, font=_font(34), fill=(20, 20, 20))
        draw.text((x + 16, 68), "Pose {}".format(pose_index), font=_font(27), fill=(20, 20, 20))
        canvas.paste(image, (x, bar_height))

    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    output = (
        PRESENTATION_DIR
        / "raw_vs_deglared_matched_novel_views_1x6_reflection_marked.png"
    )
    canvas.save(output)
    return output


def build_artifact_sheet() -> Path:
    """Recompose user-selected crops with prominent title and A/C row labels."""
    root = (
        PROJECT_ROOT
        / "results/photo_scene6/full96_reestimated_pose/qualitative_artifact_crops/a_vs_c"
    )
    top_images = _load(
        [root / "photo_scene6_raw_original_pose_artifact_crop_{:02d}.png".format(index) for index in range(1, 5)]
    )
    bottom_images = _load(
        [root / "photo_scene6_v6_v7_reestimated_pose_artifact_crop_{:02d}.png".format(index) for index in range(1, 5)]
    )
    cell_width, cell_height = 1440, 1136
    top_images = [image.resize((cell_width, cell_height), Image.Resampling.LANCZOS) for image in top_images]
    bottom_images = [image.resize((cell_width, cell_height), Image.Resampling.LANCZOS) for image in bottom_images]
    title_height, label_width = 140, 720
    canvas = Image.new(
        "RGB", (label_width + 4 * cell_width, title_height + 2 * cell_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, title_height), fill=(30, 30, 30))
    draw.text(
        (24, 24),
        "Floating-specular artifact comparison | user-selected matched crops",
        font=_font(64),
        fill="white",
    )
    rows: Sequence[Tuple[Sequence[str], Sequence[Image.Image], Tuple[int, int, int]]] = (
        (("Trained with", "raw images", "Original COLMAP pose"), top_images, (225, 236, 248)),
        (("Trained with", "our deglared images", "Re-estimated COLMAP pose"), bottom_images, (228, 245, 231)),
    )
    for row_index, (lines, images, color) in enumerate(rows):
        y = title_height + row_index * cell_height
        draw.rectangle((0, y, label_width, y + cell_height), fill=color)
        total_height = len(lines) * 76
        for line_index, line in enumerate(lines):
            draw.text(
                (34, y + cell_height // 2 - total_height // 2 + line_index * 76),
                line,
                font=_font(64 if line_index == 0 else 50),
                fill=(20, 20, 20),
            )
        for column, image in enumerate(images):
            x = label_width + column * cell_width
            canvas.paste(image, (x, y))
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    output = PRESENTATION_DIR / "raw_vs_deglared_floating_specular_artifact_crops_2x4.png"
    canvas.save(output)
    return output


def build_artifact_sheet_marked() -> Path:
    """Build the final artifact sheet with matched attention boxes and pose labels."""
    source = PRESENTATION_DIR / "raw_vs_deglared_floating_specular_artifact_crops_2x4.png"
    image = Image.open(source).convert("RGBA")
    title_height, label_width = 140, 720
    cell_width, cell_height = 1440, 1136
    expected_size = (label_width + 4 * cell_width, title_height + 2 * cell_height)
    if image.size != expected_size:
        raise RuntimeError(
            "Unexpected artifact-sheet size: {} != {}".format(image.size, expected_size)
        )

    annotation = ImageDraw.Draw(image)
    attention_box = (164, 755, 1439, 1080)
    annotation_color = (0, 214, 196, 255)
    for row in range(2):
        for column in range(4):
            cell_x = label_width + column * cell_width
            cell_y = title_height + row * cell_height
            box = (
                cell_x + attention_box[0],
                cell_y + attention_box[1],
                cell_x + attention_box[2],
                cell_y + attention_box[3],
            )
            annotation.rectangle(box, outline=annotation_color, width=14)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    pose_font = ImageFont.truetype(FONT_BOLD_PATH, 70)
    for row in range(2):
        for column in range(4):
            cell_x = label_width + column * cell_width
            cell_y = title_height + row * cell_height
            label_box = (cell_x + 24, cell_y + 24, cell_x + 330, cell_y + 120)
            overlay_draw.rounded_rectangle(label_box, radius=15, fill=(20, 20, 20, 190))
            overlay_draw.text(
                (cell_x + 46, cell_y + 29),
                "Pose {}".format(column + 1),
                font=pose_font,
                fill=(255, 255, 255, 255),
            )
    image = Image.alpha_composite(image, overlay).convert("RGB")

    output = (
        PRESENTATION_DIR
        / "raw_vs_deglared_floating_specular_artifact_crops_2x4_marked.png"
    )
    image.save(output)
    return output


def main() -> None:
    print(build_novel_view_sheet())
    print(build_novel_view_sheet_1x6())
    print(build_novel_view_sheet_1x6_reflection_marked())
    print(build_artifact_sheet())
    print(build_artifact_sheet_marked())


if __name__ == "__main__":
    main()

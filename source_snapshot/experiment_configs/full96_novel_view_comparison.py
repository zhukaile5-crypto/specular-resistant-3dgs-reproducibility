#!/usr/bin/env python3
"""Train all 96 photo_scene6 images and render matched local novel views.

The existing 84/12 experiment remains the quantitative held-out validation.
This tool creates a complementary qualitative demonstration: both image-input
variants are trained on all 96 captures, then rendered at the same four
midpoint camera poses near frames 0012, 0051, 0056, and 0074.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
GSPLAT_ROOT = PROJECT_ROOT / "third_party" / "gsplat"
GSPLAT_EXAMPLES = GSPLAT_ROOT / "examples"
for path in (GSPLAT_ROOT, GSPLAT_EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from simple_trainer import Config, Runner  # noqa: E402


ANCHOR_FRAMES = (12, 51, 56, 74)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _frame_number(name: str) -> int:
    digits = re.findall(r"\d+", Path(name).stem)
    if not digits:
        raise ValueError(f"No numeric frame identifier in {name}")
    return int(digits[-1])


def _config(data_dir: Path, result_dir: Path, max_steps: int) -> Config:
    return Config(
        disable_viewer=True,
        data_dir=str(data_dir),
        data_factor=4,
        result_dir=str(result_dir),
        # Runner requires a positive value here to instantiate its datasets.
        # The train set is explicitly expanded to all parser indices below.
        test_every=8,
        max_steps=max_steps,
        save_steps=[1_000, 3_000, 5_000, 10_000, 20_000, max_steps],
        eval_steps=[-1],
        init_type="sfm",
        pose_opt=False,
    )


def train_all_images(data_dir: Path, result_dir: Path, max_steps: int, seed: int) -> None:
    """Train a vanilla 3DGS model using every parser image as a train view."""
    if result_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {result_dir}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runner = Runner(0, 0, 1, _config(data_dir, result_dir, max_steps))
    runner.trainset.indices = np.arange(len(runner.parser.camtoworlds), dtype=np.int64)
    print(f"FULL96_TRAIN_COUNT={len(runner.trainset)}")
    print("FULL96_HELDOUT_EVAL_DISABLED=True")
    runner.train()


def _load_checkpoint(runner: Runner, checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=runner.device, weights_only=True)
    for key in runner.splats.keys():
        runner.splats[key].data = checkpoint["splats"][key]
    return int(checkpoint["step"])


def _midpoint_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Interpolate a camera-to-world pose at the geodesic rotation midpoint."""
    rotations = Rotation.from_matrix(np.stack([first[:3, :3], second[:3, :3]]))
    rotation = Slerp([0.0, 1.0], rotations)([0.5]).as_matrix()[0]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = ((first[:3, 3] + second[:3, 3]) * 0.5).astype(np.float32)
    return pose


def _frame_indices(runner: Runner) -> Dict[int, int]:
    return {
        _frame_number(Path(path).name): index
        for index, path in enumerate(runner.parser.image_paths)
    }


def _dataset_sample_for_parser_index(runner: Runner, parser_index: int) -> Dict[str, torch.Tensor]:
    matches = np.flatnonzero(runner.trainset.indices == parser_index)
    if len(matches) != 1:
        raise RuntimeError(f"Could not identify parser index {parser_index} in the train dataset")
    return runner.trainset[int(matches[0])]


def _render_one(
    runner: Runner, pose: np.ndarray, intrinsics: torch.Tensor, size: Tuple[int, int]
) -> np.ndarray:
    """Render a novel RGB view at the supplied c2w pose and intrinsics."""
    height, width = size
    c2w = torch.from_numpy(pose).float().to(runner.device).unsqueeze(0)
    intrinsics = intrinsics.to(runner.device).unsqueeze(0)
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
    return (torch.clamp(colors[0], 0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)


def _four_view_montage(
    images: Iterable[np.ndarray], title: str, output_path: Path, anchors: Sequence[int]
) -> None:
    """Save four matched novel views as one labelled 2x2 image."""
    images = list(images)
    height, width = images[0].shape[:2]
    label_height = 42
    canvas = Image.new("RGB", (width * 2, (height + label_height) * 2), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 21)
    for index, (frame, image) in enumerate(zip(anchors, images)):
        x = (index % 2) * width
        y = (index // 2) * (height + label_height)
        draw.rectangle((x, y, x + width, y + label_height), fill=(28, 28, 28))
        draw.text((x + 12, y + 10), f"{title} | novel view near {frame:04d}", font=font, fill="white")
        canvas.paste(Image.fromarray(image), (x, y + label_height))
    canvas.save(output_path)


def render_novel_views(
    raw_checkpoint: Path,
    deglared_checkpoint: Path,
    raw_data_dir: Path,
    deglared_data_dir: Path,
    output_dir: Path,
    anchors: Sequence[int],
) -> None:
    """Render both full-96 models at identical four local midpoint poses."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_runner = Runner(0, 0, 1, _config(raw_data_dir, output_dir / "_raw_runtime", 30_000))
    raw_step = _load_checkpoint(raw_runner, raw_checkpoint)
    deglared_runner = Runner(
        0, 0, 1, _config(deglared_data_dir, output_dir / "_deglared_runtime", 30_000)
    )
    deglared_step = _load_checkpoint(deglared_runner, deglared_checkpoint)

    raw_indices = _frame_indices(raw_runner)
    deglared_indices = _frame_indices(deglared_runner)
    raw_images: List[np.ndarray] = []
    deglared_images: List[np.ndarray] = []
    pose_manifest = []
    for anchor in anchors:
        endpoint = anchor + 1
        first_index = raw_indices[anchor]
        second_index = raw_indices[endpoint]
        pose = _midpoint_pose(
            np.asarray(raw_runner.parser.camtoworlds[first_index]),
            np.asarray(raw_runner.parser.camtoworlds[second_index]),
        )
        raw_sample = _dataset_sample_for_parser_index(raw_runner, first_index)
        deglared_sample = _dataset_sample_for_parser_index(
            deglared_runner, deglared_indices[anchor]
        )
        size = tuple(int(value) for value in raw_sample["image"].shape[:2])
        raw_image = _render_one(raw_runner, pose, raw_sample["K"], size)
        deglared_image = _render_one(
            deglared_runner, pose, deglared_sample["K"], size
        )
        imageio.imwrite(output_dir / f"near_{anchor:04d}_raw_3dgs.png", raw_image)
        imageio.imwrite(
            output_dir / f"near_{anchor:04d}_v6_v7_deglared_3dgs.png", deglared_image
        )
        raw_images.append(raw_image)
        deglared_images.append(deglared_image)
        pose_manifest.append(
            {
                "anchor_frame": anchor,
                "midpoint_endpoints": [anchor, endpoint],
                "camera_to_world": pose.tolist(),
            }
        )

    if len(anchors) == 4:
        _four_view_montage(
            raw_images,
            "Raw-image 3DGS",
            output_dir / "raw_3dgs_novel_views_2x2.png",
            anchors,
        )
        _four_view_montage(
            deglared_images,
            "V6/V7 deglared-image 3DGS",
            output_dir / "v6_v7_deglared_3dgs_novel_views_2x2.png",
            anchors,
        )
    (output_dir / "novel_view_manifest.json").write_text(
        json.dumps(
            {
                "raw_checkpoint": str(raw_checkpoint),
                "deglared_checkpoint": str(deglared_checkpoint),
                "raw_step": raw_step,
                "deglared_step": deglared_step,
                "camera_poses": pose_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Local novel-view visualization after full 96-image training\n\n"
        "Both vanilla 3DGS models use all 96 images for training. Each displayed "
        "view is the midpoint of two neighboring captured cameras "
        "(0012–0013, 0051–0052, 0056–0057, or 0074–0075), so it is not a training camera.\n\n"
        "- `raw_3dgs_novel_views_2x2.png`: model trained on raw images.\n"
        "- `v6_v7_deglared_3dgs_novel_views_2x2.png`: model trained on V6/V7 deglared inputs.\n\n"
        "This is a qualitative novel-view display. The 84/12 held-out experiment "
        "remains the quantitative control and cannot be replaced by the full-view result.\n",
        encoding="utf-8",
    )


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


def _transform_pose_between_worlds(
    pose: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Express a camera-to-world pose in a similarity-aligned world system."""
    transformed = np.eye(4, dtype=np.float32)
    transformed[:3, :3] = rotation @ pose[:3, :3]
    transformed[:3, 3] = scale * rotation @ pose[:3, 3] + translation
    return transformed


def _pose_ablation_sheet(
    fixed_images: Sequence[np.ndarray],
    reestimated_images: Sequence[np.ndarray],
    anchors: Sequence[int],
    output_path: Path,
    top_label: str,
    bottom_label: str,
) -> None:
    """Save a two-row pose ablation sheet in the established labelled-tile style."""
    height, width = fixed_images[0].shape[:2]
    label_height = 42
    canvas = Image.new(
        "RGB",
        (width * len(anchors), (height + label_height) * 2),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    label_font = ImageFont.truetype(FONT_PATH, 20)
    rows = (
        (top_label, fixed_images),
        (bottom_label, reestimated_images),
    )
    for row, (label, images) in enumerate(rows):
        normalized_label = label.replace("\\n", " ").replace("\n", " ")
        for column, (anchor, image) in enumerate(zip(anchors, images)):
            x = column * width
            y = row * (height + label_height)
            draw.rectangle((x, y, x + width, y + label_height), fill=(28, 28, 28))
            draw.text(
                (x + 12, y + 10),
                f"{normalized_label} | midpoint of {anchor:04d}-{anchor + 1:04d}",
                font=label_font,
                fill="white",
            )
            canvas.paste(Image.fromarray(image), (x, y + label_height))
    canvas.save(output_path)


def render_reestimated_pose_comparison(
    fixed_checkpoint: Path,
    reestimated_checkpoint: Path,
    fixed_data_dir: Path,
    reestimated_data_dir: Path,
    output_dir: Path,
    anchors: Sequence[int],
    top_label: str,
    bottom_label: str,
    top_tag: str,
    bottom_tag: str,
    sheet_name: str,
) -> None:
    """Render B/C at matched physical poses after camera-center alignment."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    fixed_runner = Runner(0, 0, 1, _config(fixed_data_dir, output_dir / "_b_runtime", 30_000))
    fixed_step = _load_checkpoint(fixed_runner, fixed_checkpoint)
    reestimated_runner = Runner(
        0, 0, 1, _config(reestimated_data_dir, output_dir / "_c_runtime", 30_000)
    )
    reestimated_step = _load_checkpoint(reestimated_runner, reestimated_checkpoint)
    fixed_indices = _frame_indices(fixed_runner)
    reestimated_indices = _frame_indices(reestimated_runner)
    common_frames = sorted(set(fixed_indices) & set(reestimated_indices))
    fixed_centers = np.stack(
        [np.asarray(fixed_runner.parser.camtoworlds[fixed_indices[frame]])[:3, 3] for frame in common_frames]
    )
    reestimated_centers = np.stack(
        [np.asarray(reestimated_runner.parser.camtoworlds[reestimated_indices[frame]])[:3, 3] for frame in common_frames]
    )
    scale, rotation, translation = _similarity_between_camera_centers(
        fixed_centers, reestimated_centers
    )
    aligned_centers = scale * (fixed_centers @ rotation.T) + translation
    alignment_rmse = float(np.sqrt(np.mean(np.sum((aligned_centers - reestimated_centers) ** 2, axis=1))))

    fixed_images: List[np.ndarray] = []
    reestimated_images: List[np.ndarray] = []
    pose_manifest = []
    for anchor in anchors:
        fixed_index = fixed_indices[anchor]
        fixed_next_index = fixed_indices[anchor + 1]
        fixed_pose = _midpoint_pose(
            np.asarray(fixed_runner.parser.camtoworlds[fixed_index]),
            np.asarray(fixed_runner.parser.camtoworlds[fixed_next_index]),
        )
        reestimated_pose = _transform_pose_between_worlds(
            fixed_pose, scale, rotation, translation
        )
        fixed_sample = _dataset_sample_for_parser_index(fixed_runner, fixed_index)
        reestimated_sample = _dataset_sample_for_parser_index(
            reestimated_runner, reestimated_indices[anchor]
        )
        fixed_size = tuple(int(value) for value in fixed_sample["image"].shape[:2])
        reestimated_size = tuple(int(value) for value in reestimated_sample["image"].shape[:2])
        if fixed_size != reestimated_size:
            raise RuntimeError(f"Mismatched render sizes: {fixed_size} versus {reestimated_size}")
        fixed_image = _render_one(fixed_runner, fixed_pose, fixed_sample["K"], fixed_size)
        reestimated_image = _render_one(
            reestimated_runner, reestimated_pose, reestimated_sample["K"], reestimated_size
        )
        imageio.imwrite(output_dir / f"near_{anchor:04d}_{top_tag}.png", fixed_image)
        imageio.imwrite(
            output_dir / f"near_{anchor:04d}_{bottom_tag}.png", reestimated_image
        )
        fixed_images.append(fixed_image)
        reestimated_images.append(reestimated_image)
        pose_manifest.append(
            {
                "anchor_frame": anchor,
                "midpoint_endpoints": [anchor, anchor + 1],
                "b_camera_to_world": fixed_pose.tolist(),
                "c_camera_to_world": reestimated_pose.tolist(),
            }
        )
    sheet_path = output_dir / sheet_name
    _pose_ablation_sheet(
        fixed_images, reestimated_images, anchors, sheet_path, top_label, bottom_label
    )
    (output_dir / "pose_alignment_manifest.json").write_text(
        json.dumps(
            {
                "b_checkpoint": str(fixed_checkpoint),
                "c_checkpoint": str(reestimated_checkpoint),
                "b_step": fixed_step,
                "c_step": reestimated_step,
                "common_camera_count": len(common_frames),
                "similarity_b_to_c": {
                    "scale": scale,
                    "rotation": rotation.tolist(),
                    "translation": translation.tolist(),
                    "camera_center_alignment_rmse": alignment_rmse,
                },
                "camera_poses": pose_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train-all")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--result-dir", type=Path, required=True)
    train.add_argument("--max-steps", type=int, default=30_000)
    train.add_argument("--seed", type=int, default=20260807)
    render = subparsers.add_parser("render-novel")
    render.add_argument("--raw-checkpoint", type=Path, required=True)
    render.add_argument("--deglared-checkpoint", type=Path, required=True)
    render.add_argument("--raw-data-dir", type=Path, required=True)
    render.add_argument("--deglared-data-dir", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--anchors", type=int, nargs="+", default=list(ANCHOR_FRAMES))
    compare = subparsers.add_parser("compare-reestimated-pose")
    compare.add_argument("--b-checkpoint", type=Path, required=True)
    compare.add_argument("--c-checkpoint", type=Path, required=True)
    compare.add_argument("--b-data-dir", type=Path, required=True)
    compare.add_argument("--c-data-dir", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--anchors", type=int, nargs="+", default=[51, 55, 56])
    compare.add_argument("--top-label", default="B: V6/V7 image\n+original COLMAP pose")
    compare.add_argument("--bottom-label", default="C: V6/V7 image\n+re-estimated pose")
    compare.add_argument("--top-tag", default="b_original_pose")
    compare.add_argument("--bottom-tag", default="c_reestimated_pose")
    compare.add_argument("--sheet-name", default="b_vs_c_reestimated_pose_matched_novel_views.png")
    args = parser.parse_args(argv)
    if args.command == "train-all":
        train_all_images(args.data_dir, args.result_dir, args.max_steps, args.seed)
    elif args.command == "render-novel":
        render_novel_views(
            args.raw_checkpoint,
            args.deglared_checkpoint,
            args.raw_data_dir,
            args.deglared_data_dir,
            args.output_dir,
            args.anchors,
        )
    else:
        render_reestimated_pose_comparison(
            args.b_checkpoint,
            args.c_checkpoint,
            args.b_data_dir,
            args.c_data_dir,
            args.output_dir,
            args.anchors,
            args.top_label,
            args.bottom_label,
            args.top_tag,
            args.bottom_tag,
            args.sheet_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

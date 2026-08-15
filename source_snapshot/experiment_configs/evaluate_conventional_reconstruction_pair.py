#!/usr/bin/env python3
"""Render held-out views for a completed matched vanilla-3DGS pair.

This small runner deliberately calls ``Runner.eval`` directly, rather than
the example CLI's checkpoint mode.  It therefore renders the validation
views only and does not spend time producing an interpolated trajectory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import imageio
import numpy as np
import torch


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
GSPLAT_ROOT = PROJECT_ROOT / "third_party" / "gsplat"
GSPLAT_EXAMPLES = PROJECT_ROOT / "third_party" / "gsplat" / "examples"

# Evaluation must use the locked gsplat checkout that produced the checkpoints.
# Otherwise Python resolves the unrelated site-package installation and requests
# another, incompatible CUDA JIT build.
if str(GSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(GSPLAT_ROOT))
if str(GSPLAT_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(GSPLAT_EXAMPLES))

from simple_trainer import Config, Runner  # noqa: E402


def render_validation(
    data_dir: Path,
    result_dir: Path,
    checkpoint_path: Path,
    data_factor: int,
    test_every: int,
) -> None:
    """Load one checkpoint and save its validation canvases and metrics."""
    config = Config(
        disable_viewer=True,
        data_dir=str(data_dir),
        data_factor=data_factor,
        result_dir=str(result_dir),
        test_every=test_every,
        init_type="sfm",
        pose_opt=False,
    )
    runner = Runner(0, 0, 1, config)
    checkpoint = torch.load(
        checkpoint_path, map_location=runner.device, weights_only=True
    )
    for key in runner.splats.keys():
        runner.splats[key].data = checkpoint["splats"][key]

    step = int(checkpoint["step"])
    print(f"Rendering validation views at checkpoint step {step}.")
    _render_validation_without_workers(runner, step)


@torch.no_grad()
def _render_validation_without_workers(runner: Runner, step: int) -> None:
    """Render validation views without a post-CUDA DataLoader worker.

    ``simple_trainer.Runner.eval`` starts a worker after CUDA initialization.
    That worker can hang under this WSL setup.  The loop below is otherwise
    the same evaluation logic, including canvas and metric generation.
    """
    cfg = runner.cfg
    device = runner.device
    metrics = defaultdict(list)
    ellipse_time = 0.0

    for index in range(len(runner.valset)):
        data = runner.valset[index]
        camtoworlds = data["camtoworld"].to(device).unsqueeze(0)
        intrinsics = data["K"].to(device).unsqueeze(0)
        pixels = data["image"].to(device).unsqueeze(0) / 255.0
        masks = data.get("mask")
        if masks is not None:
            masks = masks.to(device).unsqueeze(0)
        height, width = pixels.shape[1:3]

        torch.cuda.synchronize()
        start = time.time()
        colors, _, _ = runner.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=intrinsics,
            width=width,
            height=height,
            sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            masks=masks,
        )
        torch.cuda.synchronize()
        ellipse_time += time.time() - start

        colors = torch.clamp(colors, 0.0, 1.0)
        canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
        imageio.imwrite(
            f"{runner.render_dir}/val_step{step}_{index:04d}.png",
            (canvas * 255).astype(np.uint8),
        )
        pixels_chw = pixels.permute(0, 3, 1, 2)
        colors_chw = colors.permute(0, 3, 1, 2)
        metrics["psnr"].append(runner.psnr(colors_chw, pixels_chw))
        metrics["ssim"].append(runner.ssim(colors_chw, pixels_chw))
        metrics["lpips"].append(runner.lpips(colors_chw, pixels_chw))
        print(f"Rendered validation view {index + 1}/{len(runner.valset)}.")

    stats = {key: torch.stack(values).mean().item() for key, values in metrics.items()}
    stats.update(
        {
            "ellipse_time": ellipse_time / len(runner.valset),
            "num_GS": len(runner.splats["means"]),
        }
    )
    with open(f"{runner.stats_dir}/val_step{step:04d}.json", "w") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-factor", type=int, default=4)
    parser.add_argument("--test-every", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    for path in (args.data_dir, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    render_validation(
        args.data_dir,
        args.result_dir,
        args.checkpoint,
        args.data_factor,
        args.test_every,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

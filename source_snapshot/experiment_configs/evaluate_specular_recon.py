#!/usr/bin/env python3
"""Evaluate specular-aware reconstructions (metrics 1/2/4).

Standard PSNR/SSIM against the *original* ground truth cannot tell whether
a reconstruction is good: the GT itself contains specular reflections, so a
model that bakes reflections in scores *higher*.  This script implements
the three evaluation signals designed for this project:

1. **Render-level specular audit** — run the same specular detector on
   the renders and on GT: does the rendered image still contain
   highlight-like pixels, especially inside GT specular regions?
2. **Dual-GT PSNR** — every view (including held-out) has both an
   original and a delighted version, so PSNR can be measured against
   both references, full-frame and masked by GT specular regions.
3. **3D-anchored view-dependence** — true reflections are view-dependent.
   Project the COLMAP sparse points into every observing view and
   measure per-point luminance variance across views, in GT images and
   in renders.  ``R_vd = mean(std_render) / mean(std_gt)`` on the
   specular point set: ~1 = view-dependent appearance preserved
   (reflections baked into SH), ~0 = stable across views.

Modes::

    # Metrics 1+2 for one experiment (renders must exist)
    python experiment_configs/evaluate_specular_recon.py image-metrics \
        --scene photo_scene6 \
        --result-dir results/photo_scene6/specular_aware_adaptive_feather_t0.3_v3

    # Metric 4, GT-side only (works without any renders)
    python experiment_configs/evaluate_specular_recon.py view-dependence \
        --scene photo_scene6 --colmap-dir data/custom/photo_scene6

    # Metric 4 including render-side (needs renders + canvases)
    python experiment_configs/evaluate_specular_recon.py view-dependence \
        --scene photo_scene6 \
        --colmap-dir data/custom/photo_scene6_train_colmap_adaptive_feather_t0.3_v3 \
        --result-dir results/photo_scene6/specular_aware_adaptive_feather_t0.3_v3

    # Aggregate several experiments into a markdown table
    python experiment_configs/evaluate_specular_recon.py compare \
        results/photo_scene6/specular_aware_*/spec_eval_step*.json

Run inside the ``nerfstudio`` conda environment.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
RAW_IMAGES = PROJECT_ROOT / "raw_images"
DATA_CUSTOM = PROJECT_ROOT / "data" / "custom"
RESULTS = PROJECT_ROOT / "results"

if str(PROJECT_ROOT / "StableDelight") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "StableDelight"))

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _lum(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luminance of an (..., 3) array in [0, 1]."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _psnr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """PSNR between two float arrays in [0, 1], optionally over a 2D mask."""
    if mask is not None:
        if mask.sum() < 10:
            return float("nan")
        diff = (a[mask] - b[mask]).ravel()
    else:
        diff = (a - b).ravel()
    mse = float(np.mean(diff**2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 10:
        return float("nan")
    return float(np.abs(a[mask] - b[mask]).mean())


def _hifrac(img: np.ndarray) -> float:
    """Fraction of pixels with max(R,G,B) > 200/255 (matches reflection_metrics)."""
    return float((img.max(axis=-1) > 200.0 / 255.0).mean())


# ---------------------------------------------------------------------------
# High-frequency / texture-masked fidelity (emphasises fine-detail loss)
# ---------------------------------------------------------------------------

def _box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Fast separable box mean of a 2D array via cumsum (edge padding)."""
    if k <= 1:
        return arr
    pad = k // 2
    padded = np.pad(arr.astype(np.float64), pad, mode="edge")
    c = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    h, w = arr.shape
    return (c[k : k + h, k : k + w] - c[0:h, k : k + w]
            - c[k : k + h, 0:w] + c[0:h, 0:w]) / (k * k)


def _local_std(lum: np.ndarray, k: int = 9) -> np.ndarray:
    """Local luminance std in a k×k window (texture energy measure)."""
    mean = _box_mean(lum, k)
    var = np.maximum(_box_mean(lum**2, k) - mean**2, 0.0)
    return np.sqrt(var)


def _highpass(img: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """High-pass residual ``img − gaussian_blur(img)``, per RGB channel."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    blur = (
        np.asarray(Image.fromarray(u8).filter(ImageFilter.GaussianBlur(radius)))
        .astype(np.float32)
        / 255.0
    )
    return img - blur


def _texture_mask(
    gt_orig: np.ndarray, diffuse: np.ndarray, tex_frac: float
) -> np.ndarray:
    """Diffuse pixels with the highest local luminance std (top ``tex_frac``).

    Fine-detail loss happens where there is detail; smooth walls/floors
    dominate the diffuse area but contribute nothing to texture loss, so
    fidelity metrics are restricted to the most textured diffuse pixels.
    """
    std = _local_std(_lum(gt_orig))
    if diffuse.any():
        thr = float(np.quantile(std[diffuse], 1.0 - tex_frac))
    else:
        thr = float("inf")
    return diffuse & (std >= thr)


# ---------------------------------------------------------------------------
# Optional perceptual metric (torchmetrics LPIPS, same as gsplat trainer)
# ---------------------------------------------------------------------------

_LPIPS_MODEL = None


def _lpips_dist(a: np.ndarray, b: np.ndarray) -> float:
    """LPIPS(VGG) distance between two (H, W, 3) float32 images in [0, 1]."""
    global _LPIPS_MODEL
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False
        )

    def _t(x: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(x.transpose(2, 0, 1)[None]).float() * 2.0 - 1.0

    with torch.no_grad():
        return float(_LPIPS_MODEL(_t(a), _t(b)))


def _stem_key(name: str) -> int:
    """Numeric key of a filename stem (``frame_00012`` / ``0012`` -> 12)."""
    digits = re.findall(r"\d+", Path(name).stem)
    return int(digits[-1]) if digits else 0


def _load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image as float32 (H, W, 3) in [0, 1]; optional resize to (W, H)."""
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Renders produced by gsplat simple_trainer eval
# ---------------------------------------------------------------------------

def find_val_canvases(result_dir: Path, step: Optional[int] = None) -> List[Path]:
    """Return val canvas PNGs (``val_step{step}_{i:04d}.png``), sorted by index."""
    render_dir = result_dir / "renders"
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Render directory not found: {render_dir}")
    if step is None:
        steps = [
            int(m.group(1))
            for p in render_dir.glob("val_step*_*.png")
            if (m := re.match(r"val_step(\d+)_", p.name))
        ]
        if not steps:
            raise FileNotFoundError(f"No val_step*.png found in {render_dir}")
        step = max(steps)
    canvases = sorted(
        render_dir.glob(f"val_step{step}_*.png"), key=lambda p: _stem_key(p.name)
    )
    if not canvases:
        raise FileNotFoundError(f"No canvases for step {step} in {render_dir}")
    return canvases


def split_canvas(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Split an eval canvas into (GT half, render half), float32 in [0, 1]."""
    canvas = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    h, w2 = canvas.shape[:2]
    w = w2 // 2
    return canvas[:, :w], canvas[:, w : 2 * w]


# ---------------------------------------------------------------------------
# Mode: image-metrics (signals 1 + 2)
# ---------------------------------------------------------------------------

def image_metrics(
    scene: str,
    result_dir: Path,
    *,
    step: Optional[int] = None,
    test_every: int = 8,
    eval_threshold: float = 0.5,
    tex_frac: float = 0.3,
    use_lpips: bool = False,
    raw_dir: Optional[Path] = None,
    delighted_dir: Optional[Path] = None,
    limit: int = 0,
) -> Dict[str, Any]:
    """Compute render-level specular audit (1) and dual-GT PSNR (2).

    Val views are ``sorted(raw stems)[0::test_every]`` — the same split the
    gsplat parser uses.  GT references are the raw original and the
    delighted image, both resized to the canvas resolution.  (The canvas GT
    half is the undistorted *training* image, so comparisons against raw
    originals carry a small uniform resampling bias, identical across
    experiments.)

    Fidelity metrics emphasise fine-detail loss: ``psnr_hf_textured`` is
    PSNR of the high-pass residual restricted to the most textured diffuse
    pixels (top ``tex_frac`` by local luminance std).
    """
    from stabledelight.utils.specular_detector import compute_specular_score

    if raw_dir is None:
        raw_dir = RAW_IMAGES / scene
    if delighted_dir is None:
        delighted_dir = DATA_CUSTOM / f"{scene}_delighted"

    raw_by_key = {
        _stem_key(p.name): p
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    }
    deli_by_key = {
        _stem_key(p.name): p
        for p in delighted_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    }
    val_keys = sorted(raw_by_key)[0::test_every]

    canvases = find_val_canvases(result_dir, step)
    step_used = int(re.match(r"val_step(\d+)_", canvases[0].name).group(1))
    if len(canvases) != len(val_keys):
        print(
            f"  WARN: {len(canvases)} canvases vs {len(val_keys)} expected val views"
        )
    if limit > 0:
        canvases = canvases[:limit]

    print(f"Image metrics for: {result_dir}")
    print(f"  Step: {step_used}  ({len(canvases)} val views, eval threshold {eval_threshold})")

    per_view: List[Dict[str, Any]] = []
    for i, canvas_path in enumerate(canvases):
        key = val_keys[i]
        gt_half, render = split_canvas(canvas_path)
        h, w = render.shape[:2]

        gt_orig = _load_rgb(raw_by_key[key], size=(w, h))
        gt_deli = _load_rgb(deli_by_key[key], size=(w, h))

        s_orig = compute_specular_score((gt_orig * 255).astype(np.uint8))
        s_render = compute_specular_score((render * 255).astype(np.uint8))
        spec = s_orig >= eval_threshold
        diffuse = ~spec
        tex = _texture_mask(gt_orig, diffuse, tex_frac)

        entry: Dict[str, Any] = {
            "view": raw_by_key[key].name,
            # (1) specular audit
            "render_score_mean": float(s_render.mean()),
            "gt_score_mean": float(s_orig.mean()),
            "render_spec_score": float(s_render[spec].mean()) if spec.any() else float("nan"),
            "render_hifrac": _hifrac(render),
            "gt_hifrac": _hifrac(gt_orig),
            "spec_fraction_gt": float(spec.mean()),
            # (2) dual-GT PSNR
            "psnr_orig": _psnr(render, gt_orig),
            "psnr_deli": _psnr(render, gt_deli),
            "psnr_diffuse_orig": _psnr(render, gt_orig, diffuse),
            "psnr_spec_deli": _psnr(render, gt_deli, spec),
            "mae_spec_orig": _mae(render, gt_orig, spec),
            "mae_spec_deli": _mae(render, gt_deli, spec),
            # fine-detail fidelity (texture-masked, high-frequency)
            "tex_fraction": float(tex.mean()),
            "psnr_textured_orig": _psnr(render, gt_orig, tex),
            "psnr_hf_textured": _psnr(_highpass(render), _highpass(gt_orig), tex),
            # data-level delight collateral damage (same across experiments)
            "delight_damage_diffuse": _mae(gt_deli, gt_orig, diffuse),
            "delight_damage_hf_textured": _mae(
                _highpass(gt_deli), _highpass(gt_orig), tex
            ),
        }
        if use_lpips:
            entry["lpips_orig"] = _lpips_dist(render, gt_orig)
            entry["lpips_deli"] = _lpips_dist(render, gt_deli)
        per_view.append(entry)
        if i < 3:
            print(
                f"  {entry['view']}: psnr_orig={entry['psnr_orig']:.2f} "
                f"psnr_deli={entry['psnr_deli']:.2f} "
                f"render_score={entry['render_score_mean']:.3f}"
            )

    agg_keys = [k for k in per_view[0] if k != "view"]
    aggregate = {
        k: float(np.nanmean([v[k] for v in per_view])) for k in agg_keys
    }
    result: Dict[str, Any] = {
        "scene": scene,
        "result_dir": str(result_dir),
        "step": step_used,
        "test_every": test_every,
        "eval_threshold": eval_threshold,
        "tex_frac": tex_frac,
        "use_lpips": use_lpips,
        "n_val_views": len(per_view),
        "aggregate": aggregate,
        "per_view": per_view,
    }

    out_path = result_dir / f"spec_eval_step{step_used}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print("\nAggregate:")
    for k in agg_keys:
        print(f"  {k}: {aggregate[k]:.4f}")
    print(f"Saved: {out_path}")
    return result


# ---------------------------------------------------------------------------
# Mode: view-dependence (signal 4)
# ---------------------------------------------------------------------------

def _load_colmap_model(colmap_dir: Path):
    """Load the sparse model with ``pycolmap.SceneManager`` (gsplat's reader).

    Returns (manager, frames) where frames are the model images sorted by
    name — the same ordering the gsplat parser uses for its train/val split.
    """
    from pycolmap import SceneManager

    candidates = [
        colmap_dir / "sparse" / "0",
        colmap_dir / "colmap" / "sparse" / "0",
    ]
    model_dir = next((c for c in candidates if (c / "images.bin").exists()
                      or (c / "images.txt").exists()), None)
    if model_dir is None:
        raise FileNotFoundError(
            f"No COLMAP sparse model found under {colmap_dir} "
            f"(tried {', '.join(str(c) for c in candidates)})"
        )
    manager = SceneManager(str(model_dir))
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    frames = sorted(manager.images.values(), key=lambda im: im.name)
    return manager, frames


def _find_frame_image(colmap_dir: Path, name: str) -> Path:
    """Locate the image file belonging to a model image name."""
    for cand in (colmap_dir / "images" / name, colmap_dir / "images" / Path(name).name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Image for frame '{name}' not found under {colmap_dir}/images")


def _sample_at(gray: np.ndarray, xys: np.ndarray) -> np.ndarray:
    """Nearest-neighbour luminance samples at (x, y) pixel coords."""
    xi = np.clip(xys[:, 0].astype(int), 0, gray.shape[1] - 1)
    yi = np.clip(xys[:, 1].astype(int), 0, gray.shape[0] - 1)
    return gray[yi, xi]


def view_dependence(
    scene: str,
    colmap_dir: Path,
    *,
    result_dir: Optional[Path] = None,
    step: Optional[int] = None,
    test_every: int = 8,
    min_views: int = 4,
    top_frac: float = 0.1,
) -> Dict[str, Any]:
    """Metric 4: per-point luminance variance across observing views.

    Uses the COLMAP track directly: for every 3D point, the model stores
    the exact 2D pixel where it was observed in each image
    (``points2D[point3D_ids >= 0]``).  Luminance is sampled at those
    positions across views — no reprojection convention issues.

    GT-side (always): samples from ``colmap_dir/images/`` (the exact
    images COLMAP/the trainer saw).  Render-side (when ``result_dir`` is
    given): samples from the val canvases, separately for the GT half and
    the render half, restricted to the specular point set (top
    ``top_frac`` by GT view-variance).
    """
    manager, frames = _load_colmap_model(colmap_dir)
    n_points = len(manager.points3D)
    print(f"View-dependence for: {colmap_dir}")
    print(f"  Points: {n_points}  Frames: {len(frames)}")

    # COLMAP point3D IDs are not contiguous row indices; build a lookup.
    max_id = int(manager.point3D_ids.max())
    id_to_idx = np.full(max_id + 1, -1, dtype=np.int64)
    id_to_idx[manager.point3D_ids.astype(np.int64)] = np.arange(n_points)

    # ---- GT side: sample each frame's image at the track positions ----
    lum = np.full((len(frames), n_points), np.nan, dtype=np.float64)
    tracks: List[Tuple[np.ndarray, np.ndarray]] = []  # (row_idx, xys) per frame
    for fi, im in enumerate(frames):
        # point3D_ids is uint64; invalid entries are 2**64-1, so cast to
        # int64 BEFORE comparing with 0 (uint64 >= 0 is always true!).
        obs = im.point3D_ids.astype(np.int64) >= 0
        pids = id_to_idx[im.point3D_ids[obs].astype(np.int64)]
        xys = im.points2D[obs]
        tracks.append((pids, xys))
        img_path = _find_frame_image(colmap_dir, im.name)
        gray = _lum(_load_rgb(img_path))
        cam = manager.cameras[im.camera_id]
        scaled = xys * np.array([gray.shape[1] / cam.width, gray.shape[0] / cam.height])
        lum[fi, pids] = _sample_at(gray, scaled)
        if fi % 20 == 0:
            print(f"  sampled frame {fi + 1}/{len(frames)}")

    n_views = np.sum(~np.isnan(lum), axis=0)
    usable = n_views >= min_views
    std_gt = np.nanstd(lum[:, usable], axis=0)
    print(
        f"  GT-side: {usable.sum()} points visible in >= {min_views} views; "
        f"std_gt p50/p90/p99 = "
        f"{np.percentile(std_gt, 50):.4f}/{np.percentile(std_gt, 90):.4f}/"
        f"{np.percentile(std_gt, 99):.4f}"
    )

    result: Dict[str, Any] = {
        "scene": scene,
        "colmap_dir": str(colmap_dir),
        "n_points": int(n_points),
        "min_views": min_views,
        "top_frac": top_frac,
        "gt": {
            "n_points_used": int(usable.sum()),
            "std_p50": float(np.percentile(std_gt, 50)),
            "std_p90": float(np.percentile(std_gt, 90)),
            "std_p99": float(np.percentile(std_gt, 99)),
        },
    }

    # specular point set: top top_frac by GT view-variance
    spec_cut = float(np.quantile(std_gt, 1.0 - top_frac))
    spec_idx = np.flatnonzero(usable)[std_gt >= spec_cut]
    result["gt"]["spec_cut"] = spec_cut
    result["gt"]["n_specular_points"] = int(len(spec_idx))
    print(f"  Specular point set: {len(spec_idx)} points (std_gt >= {spec_cut:.4f})")

    # ---- render side: val canvases ----
    if result_dir is not None:
        canvases = find_val_canvases(result_dir, step)
        step_used = int(re.match(r"val_step(\d+)_", canvases[0].name).group(1))
        val_frames = frames[0::test_every]
        val_tracks = tracks[0::test_every]
        n = min(len(canvases), len(val_frames))
        pair_std = []
        spec_set = set(spec_idx.tolist())
        per_frame_samples = []
        for i in range(n):
            im = val_frames[i]
            pids, xys = val_tracks[i]
            cam = manager.cameras[im.camera_id]
            gt_half, render = split_canvas(canvases[i])
            h, w = render.shape[:2]
            scaled = xys * np.array([w / cam.width, h / cam.height])
            mask = np.array([p in spec_set for p in pids])
            if mask.any():
                per_frame_samples.append(
                    (pids[mask],
                     _sample_at(_lum(gt_half), scaled[mask]),
                     _sample_at(_lum(render), scaled[mask]))
                )
        # accumulate per-point (std_canvas_gt, std_render) over val views
        from collections import defaultdict
        acc: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        for pids_m, gt_s, re_s in per_frame_samples:
            for pid, g, r in zip(pids_m.tolist(), gt_s.tolist(), re_s.tolist()):
                acc[pid].append((g, r))
        for pid, samples in acc.items():
            if len(samples) >= 2:
                arr = np.asarray(samples)
                pair_std.append((float(arr[:, 0].std()), float(arr[:, 1].std())))
        if pair_std:
            arr = np.asarray(pair_std)
            r_vd = float(arr[:, 1].mean() / max(arr[:, 0].mean(), 1e-8))
            result["render"] = {
                "result_dir": str(result_dir),
                "step": step_used,
                "n_specular_points_with_2plus_val_views": int(len(arr)),
                "mean_std_canvas_gt": float(arr[:, 0].mean()),
                "mean_std_render": float(arr[:, 1].mean()),
                "R_vd": r_vd,
            }
            print(
                f"  Render-side: {len(arr)} specular points with >=2 val views; "
                f"R_vd = {r_vd:.3f} "
                f"(render std {arr[:, 1].mean():.4f} vs canvas-GT std {arr[:, 0].mean():.4f})"
            )
        else:
            print("  Render-side: no specular points with >=2 val views")

        out_path = result_dir / f"view_dependence_step{step_used}.json"
    else:
        out_path = colmap_dir / "view_dependence_gt.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")
    return result


# ---------------------------------------------------------------------------
# Mode: compare
# ---------------------------------------------------------------------------

def compare(json_paths: List[Path]) -> str:
    """Aggregate spec_eval JSONs into a markdown table."""
    cols = [
        ("psnr_orig", "PSNR orig", "{:.2f}"),
        ("psnr_deli", "PSNR deli", "{:.2f}"),
        ("psnr_diffuse_orig", "PSNR diff·orig", "{:.2f}"),
        ("psnr_spec_deli", "PSNR spec·deli", "{:.2f}"),
        ("psnr_hf_textured", "PSNR hf·tex", "{:.2f}"),
        ("lpips_orig", "LPIPS orig", "{:.3f}"),
        ("lpips_deli", "LPIPS deli", "{:.3f}"),
        ("render_spec_score", "render spec score", "{:.4f}"),
        ("render_hifrac", "render hi-frac", "{:.4f}"),
    ]

    def _fmt(agg: Dict[str, Any], key: str, spec: str) -> str:
        v = agg.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return spec.format(v)

    rows = []
    for p in json_paths:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        agg = data["aggregate"]
        name = Path(data["result_dir"]).name
        rows.append(
            [name]
            + [_fmt(agg, k, spec) for k, _, spec in cols]
            + [str(data["n_val_views"])]
        )
    rows.sort(key=lambda r: r[0])
    header = "| experiment | " + " | ".join(label for _, label, _ in cols) + " | n |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [header, sep] + ["| " + " | ".join(r) + " |" for r in rows]
    table = "\n".join(lines)
    print(table)
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--scene", required=True)
        p.add_argument("--step", type=int, default=None)
        p.add_argument("--test-every", type=int, default=8)

    p_img = sub.add_parser("image-metrics", help="Signals 1+2 from val renders")
    _common(p_img)
    p_img.add_argument("--result-dir", type=Path, required=True)
    p_img.add_argument("--raw-dir", type=Path, default=None)
    p_img.add_argument("--delighted-dir", type=Path, default=None)
    p_img.add_argument("--eval-threshold", type=float, default=0.5)
    p_img.add_argument(
        "--tex-frac",
        type=float,
        default=0.3,
        help="Top fraction of textured diffuse pixels used for detail metrics",
    )
    p_img.add_argument(
        "--lpips",
        action="store_true",
        help="Also compute LPIPS(VGG) against original and delighted GT (needs torch)",
    )
    p_img.add_argument("--limit", type=int, default=0)

    p_vd = sub.add_parser("view-dependence", help="Signal 4 via COLMAP points")
    _common(p_vd)
    p_vd.add_argument("--colmap-dir", type=Path, required=True)
    p_vd.add_argument("--result-dir", type=Path, default=None)
    p_vd.add_argument("--min-views", type=int, default=4)
    p_vd.add_argument("--top-frac", type=float, default=0.1)

    p_cmp = sub.add_parser("compare", help="Aggregate spec_eval JSONs")
    p_cmp.add_argument("jsons", nargs="+", help="spec_eval_step*.json paths or globs")

    args = parser.parse_args()

    try:
        if args.command == "image-metrics":
            image_metrics(
                args.scene,
                args.result_dir,
                step=args.step,
                test_every=args.test_every,
                eval_threshold=args.eval_threshold,
                tex_frac=args.tex_frac,
                use_lpips=args.lpips,
                raw_dir=args.raw_dir,
                delighted_dir=args.delighted_dir,
                limit=args.limit,
            )
        elif args.command == "view-dependence":
            view_dependence(
                args.scene,
                args.colmap_dir,
                result_dir=args.result_dir,
                step=args.step,
                test_every=args.test_every,
                min_views=args.min_views,
                top_frac=args.top_frac,
            )
        elif args.command == "compare":
            paths: List[Path] = []
            for item in args.jsons:
                expanded = sorted(glob.glob(item))
                paths.extend(Path(e) for e in expanded) if expanded else paths.append(Path(item))
            compare(paths)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Quantify 2D structural fidelity outside detected specular regions.

The original capture is not reflection-free ground truth, so a global
original/output difference would count intended highlight suppression as
damage.  This evaluator instead freezes a *protected-structure* mask from the
original image only:

    frozen V3 single-view score < 0.075
    and V3 stable-structure penalty >= 0.65.

No final output, blend matte, cross-view map, or StableDelight residual is
used to select that mask.  On the selected pixels, the script compares direct
StableDelight and the final selectively fused output with the original image.
It reports CIE76 colour change, altered-pixel fractions, high-pass agreement,
and gradient preservation.  These are collateral-change/fidelity measures,
not reflection-removal accuracy or reflection-free reconstruction metrics.

Example (run in the nerfstudio environment)::

    python experiment_configs/evaluate_2d_structure_fidelity.py \
        --scenes photo_scene photo_scene2 photo_scene3 photo_scene4 \
                 photo_scene5 photo_scene6 --workers 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
RAW_IMAGES = PROJECT_ROOT / "raw_images"
DATA_CUSTOM = PROJECT_ROOT / "data" / "custom"
RESULTS = PROJECT_ROOT / "results"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EVALUATOR_VERSION = "frozen_original_structure_fidelity_v1"

if str(PROJECT_ROOT / "StableDelight") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "StableDelight"))

from stabledelight.utils.specular_detector_v3 import compute_specular_score


def _image_paths(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _numeric_key(path: Path) -> int:
    groups = re.findall(r"\d+", path.stem)
    if not groups:
        raise ValueError("No numeric frame identifier in {}".format(path))
    return int(groups[-1])


def _index_images(directory: Path) -> Dict[int, Path]:
    paths = _image_paths(directory)
    index = {_numeric_key(path): path for path in paths}
    if not index:
        raise FileNotFoundError("No supported image files in {}".format(directory))
    return index


def _load_rgb(path: Path, expected_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if expected_size is not None and image.size != expected_size:
        image = image.resize(expected_size, Image.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _resolve_fusion_dir(scene: str, explicit_root: Optional[Path]) -> Path:
    if explicit_root is not None:
        candidate = explicit_root / scene
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError("No fusion directory for {} under {}".format(scene, explicit_root))
    frozen = DATA_CUSTOM / "{}_blended_evidence_matte_t0.3_v3_cross_scene_frozen_20260811".format(scene)
    canonical = DATA_CUSTOM / "{}_blended_evidence_matte_t0.3_v3".format(scene)
    for candidate in (frozen, canonical):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("No frozen or canonical fusion output found for {}".format(scene))


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2LAB)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2],
        dtype=np.float32,
    )


def _highpass(luminance: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(luminance, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return luminance - blur


def _gradient(luminance: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return grad_x, grad_y, magnitude


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(values[mask]))


def _masked_median(values: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.median(values[mask]))


def _masked_correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 10:
        return float("nan")
    first_values = first[mask].astype(np.float64)
    second_values = second[mask].astype(np.float64)
    first_values -= first_values.mean()
    second_values -= second_values.mean()
    denominator = np.sqrt(np.sum(first_values * first_values) * np.sum(second_values * second_values))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(first_values * second_values) / denominator)


def _variant_metrics(
    original_lab: np.ndarray,
    original_hp: np.ndarray,
    original_grad_x: np.ndarray,
    original_grad_y: np.ndarray,
    original_grad_mag: np.ndarray,
    protected_mask: np.ndarray,
    edge_mask: np.ndarray,
    variant: np.ndarray,
) -> Dict[str, float]:
    """Measure collateral colour/detail change on an original-only mask."""
    variant_lab = _rgb_to_lab(variant)
    delta_e = np.sqrt(np.sum((variant_lab - original_lab) ** 2, axis=2))
    variant_lum = _luminance(variant)
    variant_hp = _highpass(variant_lum)
    variant_grad_x, variant_grad_y, variant_grad_mag = _gradient(variant_lum)

    protected_count = int(protected_mask.sum())
    edge_count = int(edge_mask.sum())
    if protected_count == 0 or edge_count == 0:
        raise RuntimeError("Protected structure mask is empty")
    highpass_difference = variant_hp - original_hp
    highpass_rmse = float(np.sqrt(np.mean(highpass_difference[protected_mask] ** 2)))
    highpass_psnr = 99.0 if highpass_rmse <= 1e-12 else float(20.0 * np.log10(1.0 / highpass_rmse))
    orientation_numerator = (
        original_grad_x[edge_mask] * variant_grad_x[edge_mask]
        + original_grad_y[edge_mask] * variant_grad_y[edge_mask]
    )
    orientation_denominator = (
        original_grad_mag[edge_mask] * variant_grad_mag[edge_mask] + 1e-6
    )
    original_edge_energy = float(original_grad_mag[edge_mask].sum())

    return {
        "cie76_delta_e_mean": _masked_mean(delta_e, protected_mask),
        "cie76_delta_e_median": _masked_median(delta_e, protected_mask),
        "cie76_delta_e_gt_2_percent": 100.0 * float((delta_e[protected_mask] > 2.0).mean()),
        "cie76_delta_e_gt_5_percent": 100.0 * float((delta_e[protected_mask] > 5.0).mean()),
        "highpass_rmse": highpass_rmse,
        "highpass_psnr_db": highpass_psnr,
        "highpass_correlation": _masked_correlation(original_hp, variant_hp, protected_mask),
        "edge_energy_ratio": float(variant_grad_mag[edge_mask].sum() / max(original_edge_energy, 1e-8)),
        "edge_orientation_agreement": float(np.mean(orientation_numerator / orientation_denominator)),
        "edge_magnitude_mae": _masked_mean(np.abs(variant_grad_mag - original_grad_mag), edge_mask),
    }


def _evaluate_frame(task: Dict[str, Any]) -> Dict[str, Any]:
    raw_path = Path(task["raw_path"])
    candidate_path = Path(task["candidate_path"])
    fusion_path = Path(task["fusion_path"])
    low_reflection_threshold = float(task["low_reflection_threshold"])
    structure_threshold = float(task["structure_threshold"])
    edge_percentile = float(task["edge_percentile"])

    original = _load_rgb(raw_path)
    candidate = _load_rgb(candidate_path, expected_size=(original.shape[1], original.shape[0]))
    fusion = _load_rgb(fusion_path, expected_size=(original.shape[1], original.shape[0]))
    score, cues = compute_specular_score(
        (original * 255.0).astype(np.uint8),
        delighted_image=None,
        return_cues=True,
    )
    structure = np.asarray(cues["structure_penalty"], dtype=np.float32)
    protected = (score < low_reflection_threshold) & (structure >= structure_threshold)
    del score, cues, structure

    original_lum = _luminance(original)
    original_grad_x, original_grad_y, original_grad_mag = _gradient(original_lum)
    if int(protected.sum()) < 100:
        raise RuntimeError("Too few protected pixels for {}".format(raw_path.name))
    edge_cutoff = float(np.quantile(original_grad_mag[protected], edge_percentile))
    edge_mask = protected & (original_grad_mag >= edge_cutoff)
    if int(edge_mask.sum()) < 100:
        raise RuntimeError("Too few protected edge pixels for {}".format(raw_path.name))
    original_lab = _rgb_to_lab(original)
    original_hp = _highpass(original_lum)

    direct_metrics = _variant_metrics(
        original_lab,
        original_hp,
        original_grad_x,
        original_grad_y,
        original_grad_mag,
        protected,
        edge_mask,
        candidate,
    )
    fusion_metrics = _variant_metrics(
        original_lab,
        original_hp,
        original_grad_x,
        original_grad_y,
        original_grad_mag,
        protected,
        edge_mask,
        fusion,
    )
    row: Dict[str, Any] = {
        "scene": task["scene"],
        "frame": raw_path.name,
        "pixels": int(protected.size),
        "protected_pixels": int(protected.sum()),
        "protected_percent": 100.0 * float(protected.mean()),
        "protected_edge_pixels": int(edge_mask.sum()),
        "protected_edge_percent": 100.0 * float(edge_mask.mean()),
        "edge_percentile_cutoff": edge_cutoff,
    }
    row.update({"direct_{}".format(key): value for key, value in direct_metrics.items()})
    row.update({"fusion_{}".format(key): value for key, value in fusion_metrics.items()})
    row["fusion_to_direct_delta_e_ratio"] = (
        fusion_metrics["cie76_delta_e_mean"] / max(direct_metrics["cie76_delta_e_mean"], 1e-8)
    )
    row["fusion_to_direct_highpass_rmse_ratio"] = (
        fusion_metrics["highpass_rmse"] / max(direct_metrics["highpass_rmse"], 1e-8)
    )
    return row


def _statistic(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    return {
        "frames": len(rows),
        "per_image_statistics": {
            key: _statistic(float(row[key]) for row in rows)
            for key in numeric
            if key not in ("pixels", "protected_pixels", "protected_edge_pixels")
        },
    }


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    partial = path.with_name(path.stem + ".partial.json")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    partial = path.with_name(path.stem + ".partial.csv")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _format_stat(stat: Dict[str, float]) -> str:
    return "{:.3f} [{:.3f}, {:.3f}]".format(stat["median"], stat["q25"], stat["q75"])


def _write_markdown(path: Path, result: Dict[str, Any]) -> None:
    scenes = result["scenes"]
    lines = [
        "# 2D Protected-Structure Fidelity Evaluation",
        "",
        "This report measures collateral changes on original-only, low-reflection stable-structure masks. It is not a reflection-free accuracy benchmark.",
        "",
        "## Frozen mask",
        "",
        "- Frozen V3 single-view score < {low_reflection_threshold:.3f}.".format(**result["config"]),
        "- V3 stable-structure penalty >= {structure_threshold:.2f}.".format(**result["config"]),
        "- Protected-edge subset: top {edge_percentile:.0%} original gradient magnitude within the protected mask.".format(**result["config"]),
        "- Cross-view maps, StableDelight residuals, blend mattes, and final outputs are excluded from mask selection.",
        "",
        "## Per-scene per-image median [IQR]",
        "",
        "| Scene | Frames | Protected pixels | Final $\\Delta E_{76}$ | Direct $\\Delta E_{76}$ | Final HF corr. | Direct HF corr. | Final edge energy | Direct edge energy | Final/direct $\\Delta E$ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, summary in scenes.items():
        stats = summary["per_image_statistics"]
        lines.append(
            "| {} | {} | {}% | {} | {} | {} | {} | {} | {} | {} |".format(
                scene,
                summary["frames"],
                _format_stat(stats["protected_percent"]),
                _format_stat(stats["fusion_cie76_delta_e_mean"]),
                _format_stat(stats["direct_cie76_delta_e_mean"]),
                _format_stat(stats["fusion_highpass_correlation"]),
                _format_stat(stats["direct_highpass_correlation"]),
                _format_stat(stats["fusion_edge_energy_ratio"]),
                _format_stat(stats["direct_edge_energy_ratio"]),
                _format_stat(stats["fusion_to_direct_delta_e_ratio"]),
            )
        )
    overall = result["overall"]["per_image_statistics"]
    lines.extend([
        "",
        "## Aggregate per-image distribution",
        "",
        "| Measure | Median [IQR] across all frames |",
        "|---|---:|",
        "| Final CIE76 colour change | {} |".format(_format_stat(overall["fusion_cie76_delta_e_mean"])),
        "| Direct StableDelight CIE76 colour change | {} |".format(_format_stat(overall["direct_cie76_delta_e_mean"])),
        "| Final high-pass correlation | {} |".format(_format_stat(overall["fusion_highpass_correlation"])),
        "| Direct StableDelight high-pass correlation | {} |".format(_format_stat(overall["direct_highpass_correlation"])),
        "| Final edge-energy ratio | {} |".format(_format_stat(overall["fusion_edge_energy_ratio"])),
        "| Direct StableDelight edge-energy ratio | {} |".format(_format_stat(overall["direct_edge_energy_ratio"])),
        "| Final/direct colour-change ratio | {} |".format(_format_stat(overall["fusion_to_direct_delta_e_ratio"])),
        "",
        "A value near 1.0 is preferable for high-pass correlation and edge-energy ratio. Lower is preferable for colour change and final/direct ratios.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_scene(
    scene: str,
    fusion_root: Optional[Path],
    *,
    low_reflection_threshold: float,
    structure_threshold: float,
    edge_percentile: float,
    workers: int,
    limit: int,
) -> Tuple[List[Dict[str, Any]], Path, Path, Path]:
    raw_dir = RAW_IMAGES / scene
    candidate_dir = DATA_CUSTOM / "{}_delighted".format(scene)
    fusion_dir = _resolve_fusion_dir(scene, fusion_root)
    raw_index = _index_images(raw_dir)
    candidate_index = _index_images(candidate_dir)
    fusion_index = _index_images(fusion_dir)
    keys = sorted(set(raw_index) & set(candidate_index) & set(fusion_index))
    if limit > 0:
        keys = keys[:limit]
    if not keys:
        raise RuntimeError("No matched raw/candidate/fusion frames for {}".format(scene))
    tasks = [
        {
            "scene": scene,
            "raw_path": str(raw_index[key]),
            "candidate_path": str(candidate_index[key]),
            "fusion_path": str(fusion_index[key]),
            "low_reflection_threshold": low_reflection_threshold,
            "structure_threshold": structure_threshold,
            "edge_percentile": edge_percentile,
        }
        for key in keys
    ]
    print("{}: {} matched frames".format(scene, len(tasks)), flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_evaluate_frame, tasks, chunksize=1))
    else:
        rows = [_evaluate_frame(task) for task in tasks]
    return rows, raw_dir, candidate_dir, fusion_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["photo_scene", "photo_scene2", "photo_scene3", "photo_scene4", "photo_scene5", "photo_scene6"],
    )
    parser.add_argument(
        "--fusion-root",
        type=Path,
        help="Optional root containing one final-fusion directory per scene",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS / "2d_structure_fidelity_20260811",
    )
    parser.add_argument("--low-reflection-threshold", type=float, default=0.075)
    parser.add_argument("--structure-threshold", type=float, default=0.65)
    parser.add_argument("--edge-percentile", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0.0 <= args.low_reflection_threshold <= 1.0:
        parser.error("--low-reflection-threshold must be in [0, 1]")
    if not 0.0 <= args.structure_threshold <= 1.0:
        parser.error("--structure-threshold must be in [0, 1]")
    if not 0.0 < args.edge_percentile < 1.0:
        parser.error("--edge-percentile must be in (0, 1)")
    if args.workers < 1:
        parser.error("--workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_summaries: Dict[str, Any] = {}
    all_rows: List[Dict[str, Any]] = []
    for scene in args.scenes:
        rows, raw_dir, candidate_dir, fusion_dir = evaluate_scene(
            scene,
            args.fusion_root,
            low_reflection_threshold=args.low_reflection_threshold,
            structure_threshold=args.structure_threshold,
            edge_percentile=args.edge_percentile,
            workers=args.workers,
            limit=args.limit,
        )
        scene_dir = args.output_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(scene_dir / "per_image.csv", rows)
        scene_summaries[scene] = {
            **_summarise(rows),
            "raw_dir": str(raw_dir),
            "candidate_dir": str(candidate_dir),
            "fusion_dir": str(fusion_dir),
        }
        _atomic_json(scene_dir / "summary.json", scene_summaries[scene])
        all_rows.extend(rows)

    result = {
        "schema": EVALUATOR_VERSION,
        "config": {
            "low_reflection_threshold": args.low_reflection_threshold,
            "structure_threshold": args.structure_threshold,
            "edge_percentile": args.edge_percentile,
            "mask_selection": "frozen original V3 single-view score and structure cue only",
        },
        "scenes": scene_summaries,
        "overall": _summarise(all_rows),
    }
    _write_csv(args.output_dir / "all_frames.csv", all_rows)
    _atomic_json(args.output_dir / "summary.json", result)
    _write_markdown(args.output_dir / "README.md", result)
    print("Saved: {}".format(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Quantify V6/V7 specular reduction with a frozen V3 evaluator.

The evaluator deliberately recomputes V3 reflection scores for the input and
the V6/V7 output. It can use separately generated cross-view maps, or its
strictly symmetric structure-aware single-view mode. It never reuses the V6
blend matte or the score map used to create an output image. Thus the reported
values are detector-based evidence, rather than a ground-truth error metric.

Class thresholds are fixed from the V6 evidence-matte profile:

* weak reflection: score in [0.075, 0.16)
* strong reflection: score >= 0.16

For each corresponding pixel, a class has ordinal severity normal=0,
weak=1, strong=2.  A strong-to-weak transition is therefore reported as a
successful strong-reflection suppression and contributes one half of that
pixel's maximum severity to the total reduction.  Net reduction is relative:
``1 - residual / input``.  For example, 30%% to 10%% gives 66.7%%.

The V6 and V7 final JPEGs are byte-identical under the flagship profile, so
one report applies to both implementations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


PROJECT_ROOT = Path("/home/dministrator/nvidia_project")
DATA_CUSTOM = PROJECT_ROOT / "data" / "custom"
RESULTS = PROJECT_ROOT / "results"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Cumulative suppression levels.  The two thresholds are conjunctive: a
# pixel must pass both the relative and absolute score-drop requirements.
# The detectable tier intentionally has a non-zero absolute floor so JPEG
# round-off and tiny detector fluctuations are not presented as improvement.
DEFAULT_SUPPRESSION_TIERS = (
    ("detectable", "Detectable", 0.00, 0.005),
    ("mild", "Mild", 0.10, 0.010),
    ("meaningful", "Meaningful", 0.20, 0.020),
    ("strong", "Strong", 0.30, 0.030),
)

REGION_CONSENSUS_LEVELS = (
    ("simple_majority", "Simple majority", 0.50),
    ("robust_majority", "Robust majority", 0.55),
)

# Per-image CSV schema v1 used ``strong_suppression_percent`` twice: first for
# the strong -> weak/normal state-transition rate, then for the Strong
# suppression-tier coverage.  The latter silently replaced the former in the
# row dictionary.  Schema v2 gives the two quantities unambiguous names while
# leaving the aggregate JSON metric names unchanged.
PER_IMAGE_SCHEMA_VERSION = 2
STRONG_STATE_COLUMN = "strong_state_suppression_percent"
STRONG_TIER_COLUMN = "strong_tier_coverage_percent"

if str(PROJECT_ROOT / "StableDelight") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "StableDelight"))

from stabledelight.utils.specular_detector_v3 import compute_specular_score


def _numeric_key(path: Path) -> int:
    """Extract the final numeric group from a frame filename."""
    groups = re.findall(r"\d+", path.stem)
    if not groups:
        raise ValueError("Cannot derive numeric key from {}".format(path.name))
    return int(groups[-1])


def _image_index(image_dir: Path) -> Dict[int, Path]:
    valid = {".jpg", ".jpeg", ".png"}
    items = [path for path in image_dir.iterdir() if path.suffix.lower() in valid]
    index = {_numeric_key(path): path for path in items}
    if not index:
        raise FileNotFoundError("No image files under {}".format(image_dir))
    return index


def _map_path(map_dir: Path, key: int) -> Path:
    return map_dir / "frame_{:05d}.png".format(key)


def _load_map(path: Path) -> np.ndarray:
    """Load a u16 V3 map as float32 in [0, 1]."""
    array = np.asarray(Image.open(path))
    if array.dtype == np.uint16:
        scale = 65535.0
    elif array.dtype == np.uint8:
        scale = 255.0
    else:
        scale = float(np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else 1.0
    return np.clip(array.astype(np.float32) / scale, 0.0, 1.0)


def _score(image_path: Path, map_dir: Optional[Path]) -> np.ndarray:
    """Score one image, optionally with independently generated V3 maps."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    if map_dir is None:
        return compute_specular_score(image, delighted_image=None)
    key = _numeric_key(image_path)
    cross_path = _map_path(map_dir, key)
    confidence_path = map_dir / "confidence" / cross_path.name
    if not cross_path.is_file() or not confidence_path.is_file():
        raise FileNotFoundError("Missing V3 maps for {} in {}".format(image_path.name, map_dir))
    cross = _load_map(cross_path)
    confidence = _load_map(confidence_path)
    if cross.shape != image.shape[:2] or confidence.shape != image.shape[:2]:
        raise ValueError("Map/image size mismatch for {}".format(image_path))
    return compute_specular_score(
        image,
        cross_view_map=cross,
        cross_view_confidence=confidence,
        # No StableDelight residual is supplied to either side.  This keeps
        # the evaluator symmetric and independent of the blend input.
        delighted_image=None,
    )


def _states(score: np.ndarray, weak_threshold: float, strong_threshold: float) -> np.ndarray:
    """Encode normal/weak/strong as 0/1/2."""
    state = np.zeros(score.shape, dtype=np.uint8)
    state[score >= weak_threshold] = 1
    state[score >= strong_threshold] = 2
    return state


def _transition_counts(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return a 3x3 row=before, column=after transition matrix."""
    encoded = before.astype(np.int64) * 3 + after.astype(np.int64)
    return np.bincount(encoded.ravel(), minlength=9).reshape(3, 3)


def _percentage(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator > 0 else float("nan")


def _build_per_image_row(
    key: int,
    raw_score: np.ndarray,
    output_score: np.ndarray,
    raw_state: np.ndarray,
    output_state: np.ndarray,
    source_mask: np.ndarray,
    meaningful: np.ndarray,
    tier_masks: Dict[str, np.ndarray],
    region_consensus_masks: Dict[str, np.ndarray],
    frame_transitions: np.ndarray,
) -> Dict[str, Any]:
    """Build one schema-v2 CSV row without ambiguous Strong metric names."""
    pixels = float(raw_state.size)
    source_score_before = float(raw_score[source_mask].sum())
    source_score_after = float(output_score[source_mask].sum())
    row: Dict[str, Any] = {
        "frame": "frame_{:05d}".format(key),
        "pixels": int(pixels),
        "raw_weak_percent": _percentage((raw_state == 1).sum(), pixels),
        "raw_strong_percent": _percentage((raw_state == 2).sum(), pixels),
        "output_weak_percent": _percentage((output_state == 1).sum(), pixels),
        "output_strong_percent": _percentage((output_state == 2).sum(), pixels),
        STRONG_STATE_COLUMN: _percentage(
            frame_transitions[2, 0] + frame_transitions[2, 1],
            frame_transitions[2].sum(),
        ),
        "weak_clearance_percent": _percentage(
            frame_transitions[1, 0], frame_transitions[1].sum()
        ),
        "source_evidence_attenuation_percent": _percentage(
            source_score_before - source_score_after,
            source_score_before,
        ),
        "meaningful_improvement_percent": _percentage(
            meaningful.sum(), source_mask.sum()
        ),
    }
    for tier_name, tier_mask in tier_masks.items():
        column = "{}_suppression_percent".format(tier_name)
        if tier_name == "strong":
            column = STRONG_TIER_COLUMN
        row[column] = _percentage(tier_mask.sum(), source_mask.sum())
    for level_name, region_mask in region_consensus_masks.items():
        row["{}_region_consensus_percent".format(level_name)] = _percentage(
            region_mask.sum(), source_mask.sum()
        )
    return row


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    partial = path.with_name(path.stem + ".partial.json")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(partial, path)


def _save_markdown(path: Path, summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    area = metrics["global_area_percent"]
    tiers = metrics["suppression_coverage_percent"]
    lines = [
        "# V6/V7 Reflection-Reduction Evaluation",
        "",
        "This detector-based evaluation recomputes a frozen V3 score on both input and output images. It does not reuse the V6 blend matte. V6 and V7 final JPEGs are byte-identical under the flagship profile, so this report applies to both.",
        "",
        "## Thresholds and Interpretation",
        "",
        "- Weak reflection: score >= {weak_threshold:.3f} and < {strong_threshold:.3f}.".format(**summary),
        "- Strong reflection: score >= {strong_threshold:.3f}.".format(**summary),
        "- Severity weights: normal = 0, weak = 1, strong = 2. A strong-to-weak transition is a successful strong suppression and provides 50% of that pixel's maximum severity reduction.",
        "- All reductions are relative: `1 - after / before`; a 30% to 10% reduction is therefore 66.7%, not 20 percentage points.",
        "- A pixel has a meaningful improvement when its reflection score drops by at least {min_relative_drop:.0%} and at least {min_absolute_drop:.3f}.".format(**summary),
        "- Suppression levels are cumulative. Passing a stricter level also passes every lower level.",
        "",
        "## Aggregate Results",
        "",
        "| Measure | Input | V6/V7 output | Relative change |",
        "|---|---:|---:|---:|",
        "| Weak-reflection area | {weak_before:.3f}% | {weak_after:.3f}% | {weak_area_reduction:.2f}% |".format(**area),
        "| Strong-reflection area | {strong_before:.3f}% | {strong_after:.3f}% | {strong_area_reduction:.2f}% |".format(**area),
        "| Any-reflection area | {any_before:.3f}% | {any_after:.3f}% | {any_area_reduction:.2f}% |".format(**area),
        "",
        "## Suppression Coverage",
        "",
        "Coverage answers how much of the originally detected reflection region was weakened by at least a stated amount. The zero-threshold diagnostic is shown separately and is not a headline metric because it is sensitive to tiny numerical changes.",
        "",
        "| Level | Criterion | All reflection | Original weak | Original strong |",
        "|---|---|---:|---:|---:|",
        "| Any numerical decrease (diagnostic) | score drop > 0 | {any_positive[all]:.2f}% | {any_positive[weak]:.2f}% | {any_positive[strong]:.2f}% |".format(**tiers),
    ]
    for tier in summary["suppression_tiers"]:
        values = tiers[tier["name"]]
        criterion = "drop >= {:.1%} and >= {:.3f}".format(
            tier["min_relative_drop"], tier["min_absolute_drop"]
        )
        lines.append(
            "| {} | {} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
                tier["label"], criterion, values["all"], values["weak"], values["strong"]
            )
        )
    region = metrics["region_consistent_coverage_percent"]
    lines.extend([
        "",
        "## Region-Consistent Suppression",
        "",
        "A connected input-reflection component is suppressed when its mean score decreases and a required majority of its pixels also decrease. This metric tolerates small boundary shifts while rejecting regions whose changes are spatially inconsistent.",
        "",
        "| Consensus rule | All reflection | Original weak | Original strong |",
        "|---|---:|---:|---:|",
    ])
    for level in summary["region_consensus_levels"]:
        values = region[level["name"]]
        lines.append(
            "| {} (>{:.0f}% pixels decrease) | {:.2f}% | {:.2f}% | {:.2f}% |".format(
                level["label"],
                100.0 * level["minimum_decreasing_fraction"],
                values["all"],
                values["weak"],
                values["strong"],
            )
        )
    lines.extend([
        "",
        "| Transition-aware effectiveness | Value |",
        "|---|---:|",
        "| Source-region reflection-evidence attenuation (continuous score) | {source_evidence_attenuation:.2f}% |".format(**metrics),
        "| Meaningfully improved original reflection pixels | {meaningful_improvement_rate:.2f}% |".format(**metrics),
        "| Meaningfully improved original strong-reflection pixels | {strong_meaningful_improvement_rate:.2f}% |".format(**metrics),
        "| Meaningfully improved original weak-reflection pixels | {weak_meaningful_improvement_rate:.2f}% |".format(**metrics),
        "| Strong-reflection suppression (strong → weak or normal) | {strong_suppression_rate:.2f}% |".format(**metrics),
        "| Strong-reflection full clearance (strong → normal) | {strong_clearance_rate:.2f}% |".format(**metrics),
        "| Weak-reflection clearance (weak → normal) | {weak_clearance_rate:.2f}% |".format(**metrics),
        "| Net severity-weighted reflection reduction | {net_severity_reduction:.2f}% |".format(**metrics),
        "| Source-region severity reduction (does not penalise new responses) | {source_region_severity_reduction:.2f}% |".format(**metrics),
        "| New reflection severity burden, relative to input severity | {new_severity_burden:.2f}% |".format(**metrics),
        "",
        "The continuous attenuation and meaningful-improvement rates measure change only on regions detected as reflective in the input. The net severity result additionally includes new detector responses in the final image.",
        "",
        "Detailed counts are in `per_image.csv` and `summary.json`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def _save_figure(path: Path, summary: Dict[str, Any]) -> None:
    """Save a compact publication-ready overview using PIL only."""
    metrics = summary["metrics"]
    area = metrics["global_area_percent"]
    width, height = 1800, 1040
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 130), fill=(30, 30, 30))
    draw.text((48, 28), "V6/V7 detector-based reflection reduction", font=_font(56), fill="white")
    draw.text(
        (50, 94),
        "Frozen V3 evaluator; relative reduction = 1 - output / input",
        font=_font(28),
        fill=(225, 225, 225),
    )

    groups = (
        ("Weak", area["weak_before"], area["weak_after"]),
        ("Strong", area["strong_before"], area["strong_after"]),
        ("Any reflection", area["any_before"], area["any_after"]),
    )
    maximum = max(max(before, after) for _, before, after in groups)
    maximum = max(maximum, 0.1)
    chart_top, chart_bottom = 220, 620
    draw.line((120, chart_bottom, width - 80, chart_bottom), fill=(80, 80, 80), width=3)
    for index, (label, before, after) in enumerate(groups):
        center = 360 + index * 540
        for offset, value, color, caption in (
            (-105, before, (120, 160, 205), "Input"),
            (35, after, (115, 185, 135), "V6/V7"),
        ):
            bar_height = int((value / maximum) * (chart_bottom - chart_top))
            x0 = center + offset
            x1 = x0 + 105
            y0 = chart_bottom - bar_height
            draw.rectangle((x0, y0, x1, chart_bottom), fill=color)
            draw.text((x0 - 6, y0 - 42), "{:.2f}%".format(value), font=_font(28), fill=(25, 25, 25))
            draw.text((x0 - 2, chart_bottom + 16), caption, font=_font(24), fill=(50, 50, 50))
        bbox = draw.textbbox((0, 0), label, font=_font(36))
        draw.text((center - (bbox[2] - bbox[0]) // 2, chart_bottom + 58), label, font=_font(36), fill=(25, 25, 25))

    draw.rectangle((40, 760, width - 40, 980), fill=(238, 247, 240), outline=(145, 190, 150), width=3)
    headline = "Meaningfully improved reflection pixels: {meaningful_improvement_rate:.2f}%".format(**metrics)
    draw.text((80, 792), headline, font=_font(48), fill=(25, 90, 45))
    details = (
        "Evidence attenuation: {source_evidence_attenuation:.2f}%   |   "
        "Strong suppression: {strong_suppression_rate:.2f}%   |   "
        "Net severity reduction: {net_severity_reduction:.2f}%"
    ).format(**metrics)
    draw.text((80, 864), details, font=_font(32), fill=(30, 55, 35))
    canvas.save(path)


def _save_coverage_figure(path: Path, summary: Dict[str, Any]) -> None:
    """Save cumulative suppression coverage for all/weak/strong regions."""
    coverage = summary["metrics"]["suppression_coverage_percent"]
    tiers = summary["suppression_tiers"]
    width, height = 1900, 1120
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 150), fill=(30, 30, 30))
    draw.text((48, 24), "V6/V7 reflection-suppression coverage", font=_font(58), fill="white")
    draw.text(
        (50, 101),
        "Fraction of originally detected reflection pixels exceeding each cumulative reduction level",
        font=_font(27),
        fill=(225, 225, 225),
    )

    chart_left, chart_right = 130, width - 70
    chart_top, chart_bottom = 235, 825
    for percent in range(0, 101, 20):
        y = chart_bottom - int((percent / 100.0) * (chart_bottom - chart_top))
        draw.line((chart_left, y, chart_right, y), fill=(225, 225, 225), width=2)
        draw.text((45, y - 18), "{}%".format(percent), font=_font(27), fill=(65, 65, 65))

    colors = {
        "all": (92, 137, 190),
        "weak": (108, 180, 125),
        "strong": (224, 145, 85),
    }
    series = (("all", "All reflection"), ("weak", "Original weak"), ("strong", "Original strong"))
    group_width = (chart_right - chart_left) // len(tiers)
    bar_width, gap = 90, 18
    for tier_index, tier in enumerate(tiers):
        center = chart_left + group_width * tier_index + group_width // 2
        values = coverage[tier["name"]]
        start_x = center - (3 * bar_width + 2 * gap) // 2
        for series_index, (key, _) in enumerate(series):
            value = values[key]
            x0 = start_x + series_index * (bar_width + gap)
            x1 = x0 + bar_width
            y0 = chart_bottom - int((value / 100.0) * (chart_bottom - chart_top))
            draw.rectangle((x0, y0, x1, chart_bottom), fill=colors[key])
            draw.text((x0 - 4, y0 - 36), "{:.1f}".format(value), font=_font(24), fill=(30, 30, 30))
        criterion = ">={:.0f}% + {:.3f}".format(
            100.0 * tier["min_relative_drop"], tier["min_absolute_drop"]
        )
        label_box = draw.textbbox((0, 0), tier["label"], font=_font(34))
        draw.text(
            (center - (label_box[2] - label_box[0]) // 2, chart_bottom + 32),
            tier["label"],
            font=_font(34),
            fill=(25, 25, 25),
        )
        criterion_box = draw.textbbox((0, 0), criterion, font=_font(23))
        draw.text(
            (center - (criterion_box[2] - criterion_box[0]) // 2, chart_bottom + 78),
            criterion,
            font=_font(23),
            fill=(75, 75, 75),
        )

    legend_y = 1010
    legend_x = 400
    for index, (key, label) in enumerate(series):
        x = legend_x + index * 410
        draw.rectangle((x, legend_y, x + 42, legend_y + 30), fill=colors[key])
        draw.text((x + 58, legend_y - 4), label, font=_font(28), fill=(35, 35, 35))
    canvas.save(path)


def _save_region_figure(path: Path, summary: Dict[str, Any]) -> None:
    """Save area-weighted connected-region suppression coverage."""
    coverage = summary["metrics"]["region_consistent_coverage_percent"]
    levels = summary["region_consensus_levels"]
    width, height = 1500, 940
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 145), fill=(30, 30, 30))
    draw.text((46, 24), "Region-consistent reflection suppression", font=_font(54), fill="white")
    draw.text(
        (48, 96),
        "Area-weighted coverage of connected input-reflection regions",
        font=_font(27),
        fill=(225, 225, 225),
    )
    chart_left, chart_right = 120, width - 60
    chart_top, chart_bottom = 220, 680
    for percent in range(0, 101, 20):
        y = chart_bottom - int((percent / 100.0) * (chart_bottom - chart_top))
        draw.line((chart_left, y, chart_right, y), fill=(225, 225, 225), width=2)
        draw.text((38, y - 16), "{}%".format(percent), font=_font(25), fill=(65, 65, 65))
    colors = {"all": (92, 137, 190), "weak": (108, 180, 125), "strong": (224, 145, 85)}
    series = (("all", "All reflection"), ("weak", "Original weak"), ("strong", "Original strong"))
    group_width = (chart_right - chart_left) // len(levels)
    bar_width, gap = 100, 24
    for level_index, level in enumerate(levels):
        center = chart_left + group_width * level_index + group_width // 2
        values = coverage[level["name"]]
        start_x = center - (3 * bar_width + 2 * gap) // 2
        for series_index, (key, _) in enumerate(series):
            value = values[key]
            x0 = start_x + series_index * (bar_width + gap)
            x1 = x0 + bar_width
            y0 = chart_bottom - int((value / 100.0) * (chart_bottom - chart_top))
            draw.rectangle((x0, y0, x1, chart_bottom), fill=colors[key])
            draw.text((x0 - 2, y0 - 36), "{:.1f}".format(value), font=_font(25), fill=(30, 30, 30))
        label = "{} (> {:.0f}%)".format(
            level["label"], 100.0 * level["minimum_decreasing_fraction"]
        )
        box = draw.textbbox((0, 0), label, font=_font(33))
        draw.text(
            (center - (box[2] - box[0]) // 2, chart_bottom + 35),
            label,
            font=_font(33),
            fill=(25, 25, 25),
        )
    legend_y = 845
    legend_x = 235
    for index, (key, label) in enumerate(series):
        x = legend_x + index * 365
        draw.rectangle((x, legend_y, x + 40, legend_y + 28), fill=colors[key])
        draw.text((x + 54, legend_y - 5), label, font=_font(26), fill=(35, 35, 35))
    canvas.save(path)


def _summarise(transitions: np.ndarray, continuous: Dict[str, Any]) -> Dict[str, Any]:
    before = transitions.sum(axis=1).astype(np.float64)
    after = transitions.sum(axis=0).astype(np.float64)
    pixels = float(before.sum())
    weak_before, strong_before = before[1], before[2]
    weak_after, strong_after = after[1], after[2]
    any_before = weak_before + strong_before
    any_after = weak_after + strong_after
    severity_before = before[1] + 2.0 * before[2]
    severity_after = after[1] + 2.0 * after[2]
    source_loss = sum(
        max(float(source - target), 0.0) * transitions[source, target]
        for source in range(3)
        for target in range(3)
    )
    new_burden = sum(
        max(float(target - source), 0.0) * transitions[source, target]
        for source in range(3)
        for target in range(3)
    )
    suppression_coverage = {
        name: {
            "all": _percentage(counts["all"], continuous["source_pixels"]),
            "weak": _percentage(counts["weak"], continuous["weak_pixels"]),
            "strong": _percentage(counts["strong"], continuous["strong_pixels"]),
        }
        for name, counts in continuous["tier_counts"].items()
    }
    region_consistent_coverage = {
        name: {
            "all": _percentage(counts["all"], continuous["source_pixels"]),
            "weak": _percentage(counts["weak"], continuous["weak_pixels"]),
            "strong": _percentage(counts["strong"], continuous["strong_pixels"]),
        }
        for name, counts in continuous["region_consensus_counts"].items()
    }
    return {
        "pixels": int(pixels),
        "transition_counts": transitions.astype(int).tolist(),
        "global_area_percent": {
            "weak_before": _percentage(weak_before, pixels),
            "weak_after": _percentage(weak_after, pixels),
            "weak_area_reduction": _percentage(weak_before - weak_after, weak_before),
            "strong_before": _percentage(strong_before, pixels),
            "strong_after": _percentage(strong_after, pixels),
            "strong_area_reduction": _percentage(strong_before - strong_after, strong_before),
            "any_before": _percentage(any_before, pixels),
            "any_after": _percentage(any_after, pixels),
            "any_area_reduction": _percentage(any_before - any_after, any_before),
        },
        "strong_suppression_rate": _percentage(transitions[2, 0] + transitions[2, 1], strong_before),
        "strong_clearance_rate": _percentage(transitions[2, 0], strong_before),
        "weak_clearance_rate": _percentage(transitions[1, 0], weak_before),
        "net_severity_reduction": _percentage(severity_before - severity_after, severity_before),
        "source_region_severity_reduction": _percentage(source_loss, severity_before),
        "new_severity_burden": _percentage(new_burden, severity_before),
        "suppression_coverage_percent": suppression_coverage,
        "region_consistent_coverage_percent": region_consistent_coverage,
        "source_evidence_attenuation": _percentage(
            continuous["source_score_before"] - continuous["source_score_after"],
            continuous["source_score_before"],
        ),
        "meaningful_improvement_rate": _percentage(
            continuous["meaningful_all"], continuous["source_pixels"]
        ),
        "strong_meaningful_improvement_rate": _percentage(
            continuous["meaningful_strong"], continuous["strong_pixels"]
        ),
        "weak_meaningful_improvement_rate": _percentage(
            continuous["meaningful_weak"], continuous["weak_pixels"]
        ),
    }


def evaluate(
    scene: str,
    raw_image_dir: Path,
    output_image_dir: Path,
    raw_crossview_dir: Optional[Path],
    output_crossview_dir: Optional[Path],
    output_dir: Path,
    weak_threshold: float,
    strong_threshold: float,
    min_relative_drop: float,
    min_absolute_drop: float,
    limit: int,
) -> Dict[str, Any]:
    """Evaluate corresponding raw and V6/V7 images, then write artifacts."""
    if not 0.0 <= weak_threshold < strong_threshold <= 1.0:
        raise ValueError("Require 0 <= weak threshold < strong threshold <= 1")
    if not 0.0 < min_relative_drop <= 1.0 or not 0.0 < min_absolute_drop <= 1.0:
        raise ValueError("Meaningful-improvement thresholds must be in (0, 1]")
    suppression_tiers = tuple(
        (
            name,
            label,
            min_relative_drop if name == "meaningful" else relative,
            min_absolute_drop if name == "meaningful" else absolute,
        )
        for name, label, relative, absolute in DEFAULT_SUPPRESSION_TIERS
    )
    raw_images = _image_index(raw_image_dir)
    output_images = _image_index(output_image_dir)
    keys = sorted(set(raw_images) & set(output_images))
    if limit > 0:
        keys = keys[:limit]
    if not keys:
        raise RuntimeError("No corresponding raw/output images found")

    output_dir.mkdir(parents=True, exist_ok=True)
    transitions = np.zeros((3, 3), dtype=np.int64)
    continuous = {
        "source_pixels": 0.0,
        "weak_pixels": 0.0,
        "strong_pixels": 0.0,
        "source_score_before": 0.0,
        "source_score_after": 0.0,
        "meaningful_all": 0.0,
        "meaningful_weak": 0.0,
        "meaningful_strong": 0.0,
        "tier_counts": {
            "any_positive": {"all": 0.0, "weak": 0.0, "strong": 0.0},
            **{
                name: {"all": 0.0, "weak": 0.0, "strong": 0.0}
                for name, _, _, _ in suppression_tiers
            },
        },
        "region_consensus_counts": {
            name: {"all": 0.0, "weak": 0.0, "strong": 0.0}
            for name, _, _ in REGION_CONSENSUS_LEVELS
        },
    }
    rows: List[Dict[str, Any]] = []
    for index, key in enumerate(keys, start=1):
        print("[{}/{}] frame_{:05d}".format(index, len(keys), key), flush=True)
        raw_score = _score(raw_images[key], raw_crossview_dir)
        output_score = _score(output_images[key], output_crossview_dir)
        raw_state = _states(raw_score, weak_threshold, strong_threshold)
        output_state = _states(output_score, weak_threshold, strong_threshold)
        source_mask = raw_state > 0
        weak_mask = raw_state == 1
        strong_mask = raw_state == 2
        score_drop = raw_score - output_score
        meaningful = source_mask & (score_drop >= min_absolute_drop) & (
            score_drop >= min_relative_drop * raw_score
        )
        tier_masks = {
            "any_positive": source_mask & (score_drop > 0.0),
        }
        for tier_name, _, tier_relative, tier_absolute in suppression_tiers:
            tier_masks[tier_name] = source_mask & (score_drop >= tier_absolute) & (
                score_drop >= tier_relative * raw_score
            )
        component_labels, component_count = ndimage.label(
            source_mask, structure=np.ones((3, 3), dtype=np.uint8)
        )
        component_sizes = np.bincount(
            component_labels.ravel(), minlength=component_count + 1
        ).astype(np.float64)
        component_drop_sum = np.bincount(
            component_labels.ravel(), weights=score_drop.ravel(), minlength=component_count + 1
        )
        component_decreasing = np.bincount(
            component_labels.ravel(),
            weights=(score_drop > 0.0).ravel(),
            minlength=component_count + 1,
        )
        component_mean_drop = component_drop_sum / np.maximum(component_sizes, 1.0)
        component_decreasing_fraction = component_decreasing / np.maximum(component_sizes, 1.0)
        region_consensus_masks = {}
        for level_name, _, minimum_fraction in REGION_CONSENSUS_LEVELS:
            accepted_components = (component_mean_drop > 0.0) & (
                component_decreasing_fraction > minimum_fraction
            )
            accepted_components[0] = False
            region_consensus_masks[level_name] = accepted_components[component_labels]
        frame_transitions = _transition_counts(raw_state, output_state)
        transitions += frame_transitions
        continuous["source_pixels"] += float(source_mask.sum())
        continuous["weak_pixels"] += float(weak_mask.sum())
        continuous["strong_pixels"] += float(strong_mask.sum())
        continuous["source_score_before"] += float(raw_score[source_mask].sum())
        continuous["source_score_after"] += float(output_score[source_mask].sum())
        continuous["meaningful_all"] += float(meaningful.sum())
        continuous["meaningful_weak"] += float((meaningful & weak_mask).sum())
        continuous["meaningful_strong"] += float((meaningful & strong_mask).sum())
        for tier_name, tier_mask in tier_masks.items():
            continuous["tier_counts"][tier_name]["all"] += float(tier_mask.sum())
            continuous["tier_counts"][tier_name]["weak"] += float((tier_mask & weak_mask).sum())
            continuous["tier_counts"][tier_name]["strong"] += float((tier_mask & strong_mask).sum())
        for level_name, region_mask in region_consensus_masks.items():
            continuous["region_consensus_counts"][level_name]["all"] += float(region_mask.sum())
            continuous["region_consensus_counts"][level_name]["weak"] += float((region_mask & weak_mask).sum())
            continuous["region_consensus_counts"][level_name]["strong"] += float((region_mask & strong_mask).sum())
        rows.append(
            _build_per_image_row(
                key,
                raw_score,
                output_score,
                raw_state,
                output_state,
                source_mask,
                meaningful,
                tier_masks,
                region_consensus_masks,
                frame_transitions,
            )
        )

    summary = {
        "metric_name": (
            "frozen_v3_multiview_specular_reduction"
            if raw_crossview_dir is not None
            else "frozen_v3_singleview_specular_reduction"
        ),
        "scene": scene,
        "implementation": "V6/V7 flagship final JPEGs (byte-identical)",
        "raw_image_dir": str(raw_image_dir),
        "output_image_dir": str(output_image_dir),
        "raw_crossview_dir": str(raw_crossview_dir) if raw_crossview_dir is not None else None,
        "output_crossview_dir": str(output_crossview_dir) if output_crossview_dir is not None else None,
        "images_evaluated": len(rows),
        "weak_threshold": weak_threshold,
        "strong_threshold": strong_threshold,
        "min_relative_drop": min_relative_drop,
        "min_absolute_drop": min_absolute_drop,
        "suppression_tiers": [
            {
                "name": name,
                "label": label,
                "min_relative_drop": relative,
                "min_absolute_drop": absolute,
            }
            for name, label, relative, absolute in suppression_tiers
        ],
        "region_consensus_levels": [
            {
                "name": name,
                "label": label,
                "minimum_decreasing_fraction": fraction,
                "requires_positive_component_mean_drop": True,
            }
            for name, label, fraction in REGION_CONSENSUS_LEVELS
        ],
        "per_image_schema": {
            "version": PER_IMAGE_SCHEMA_VERSION,
            "strong_state_column": STRONG_STATE_COLUMN,
            "strong_tier_column": STRONG_TIER_COLUMN,
            "legacy_v1_note": (
                "In schema v1, strong_suppression_percent contained Strong-tier "
                "coverage because it overwrote the state-transition field."
            ),
        },
        "state_encoding": {"normal": 0, "weak": 1, "strong": 2},
        "metrics": _summarise(transitions, continuous),
    }
    _atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _save_markdown(output_dir / "README.md", summary)
    _save_figure(output_dir / "specular_reduction_summary.png", summary)
    _save_coverage_figure(output_dir / "specular_suppression_coverage.png", summary)
    _save_region_figure(output_dir / "region_consistent_suppression.png", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="photo_scene6")
    parser.add_argument("--raw-colmap-dir", type=Path, default=DATA_CUSTOM / "photo_scene6")
    parser.add_argument(
        "--output-colmap-dir",
        type=Path,
        default=DATA_CUSTOM / "photo_scene6_conventional_v6_colmap",
    )
    parser.add_argument(
        "--raw-image-dir",
        type=Path,
        help="Optional direct raw-image directory; overrides --raw-colmap-dir/images",
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        help="Optional direct output-image directory; overrides --output-colmap-dir/images",
    )
    parser.add_argument(
        "--raw-crossview-dir", type=Path, default=DATA_CUSTOM / "photo_scene6_crossview_maps_v3"
    )
    parser.add_argument("--output-crossview-dir", type=Path)
    parser.add_argument(
        "--single-view",
        action="store_true",
        help="Use the symmetric structure-aware V3 image score without cross-view maps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS / "photo_scene6" / "v6_v7_specular_reduction",
    )
    parser.add_argument("--weak-threshold", type=float, default=0.075)
    parser.add_argument("--strong-threshold", type=float, default=0.16)
    parser.add_argument("--min-relative-drop", type=float, default=0.20)
    parser.add_argument("--min-absolute-drop", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N frames for a smoke test")
    args = parser.parse_args()
    try:
        if not args.single_view and args.output_crossview_dir is None:
            raise ValueError("--output-crossview-dir is required unless --single-view is used")
        summary = evaluate(
            args.scene,
            args.raw_image_dir or args.raw_colmap_dir / "images",
            args.output_image_dir or args.output_colmap_dir / "images",
            None if args.single_view else args.raw_crossview_dir,
            None if args.single_view else args.output_crossview_dir,
            args.output_dir,
            args.weak_threshold,
            args.strong_threshold,
            args.min_relative_drop,
            args.min_absolute_drop,
            args.limit,
        )
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    metrics = summary["metrics"]
    print("Meaningfully improved reflection pixels: {:.2f}%".format(metrics["meaningful_improvement_rate"]))
    print("Source-region evidence attenuation: {:.2f}%".format(metrics["source_evidence_attenuation"]))
    print("Net severity-weighted reduction: {:.2f}%".format(metrics["net_severity_reduction"]))
    print("Strong suppression: {:.2f}%".format(metrics["strong_suppression_rate"]))
    print("Weak clearance: {:.2f}%".format(metrics["weak_clearance_rate"]))
    print("Saved: {}".format(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

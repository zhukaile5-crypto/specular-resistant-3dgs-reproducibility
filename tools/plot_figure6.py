#!/usr/bin/env python3
"""Redraw paper Figure 6 from the archived S6 summary and per-view CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _relative_reduction(before: float, after: float) -> float:
    return 100.0 * (before - after) / before if before > 0.0 else float("nan")


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pixels = float(sum(row["pixels"] for row in rows))
    source_score = sum(row["source_reflection_score"] for row in rows)
    residual_score = sum(row["residual_score_on_source_region"] for row in rows)
    source_pixels = sum(row["source_reflection_pixels"] for row in rows)
    source_improved = sum(row["source_improved_pixels"] for row in rows)
    source_strict = sum(row["source_strict_significant_pixels"] for row in rows)
    source_strong = sum(row["source_strong_pixels"] for row in rows)
    source_strong_suppressed = sum(row["source_strong_suppressed_pixels"] for row in rows)
    result = {
        "raw_mean_score": sum(row["raw_mean_score"] * row["pixels"] for row in rows) / pixels,
        "deglared_mean_score": sum(row["deglared_mean_score"] * row["pixels"] for row in rows) / pixels,
        "raw_any_percent": sum(row["raw_any_percent"] * row["pixels"] for row in rows) / pixels,
        "deglared_any_percent": sum(row["deglared_any_percent"] * row["pixels"] for row in rows) / pixels,
        "raw_strong_percent": sum(row["raw_strong_percent"] * row["pixels"] for row in rows) / pixels,
        "deglared_strong_percent": sum(row["deglared_strong_percent"] * row["pixels"] for row in rows) / pixels,
        "source_region_attenuation_percent": 100.0 * (source_score - residual_score) / source_score,
        "broad_improvement_coverage_percent": 100.0 * source_improved / source_pixels,
        "strict_significant_coverage_percent": 100.0 * source_strict / source_pixels,
        "strong_suppression_percent": 100.0 * source_strong_suppressed / source_strong,
    }
    result["mean_score_reduction_percent"] = _relative_reduction(result["raw_mean_score"], result["deglared_mean_score"])
    result["any_area_reduction_percent"] = _relative_reduction(result["raw_any_percent"], result["deglared_any_percent"])
    result["strong_area_reduction_percent"] = _relative_reduction(result["raw_strong_percent"], result["deglared_strong_percent"])
    return result


def read_rows(path: Path) -> List[Dict[str, Any]]:
    """Read the additive per-view metrics required by the plot."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for source in csv.DictReader(handle):
            row: Dict[str, Any] = {"view_id": source["view_id"]}
            for key, value in source.items():
                if key == "view_id":
                    continue
                if key == "within_support":
                    row[key] = value == "True"
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def draw(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any], output: Path) -> None:
    """Render the four-panel academic chart used in the paper."""
    colors = {
        "evidence": "#3E6E8C",
        "strict": "#5E8C61",
        "strong": "#B26055",
        "broad": "#C3933B",
        "global": "#777777",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    overall = summary["overall"]
    confidence = summary["direction_cluster_bootstrap_95_ci"]
    primary_keys = (
        "source_region_attenuation_percent",
        "strict_significant_coverage_percent",
        "strong_suppression_percent",
        "broad_improvement_coverage_percent",
    )
    primary_labels = (
        "Source evidence\nattenuation",
        "Strict coverage\n(≥30% & ≥0.03)",
        "Strong-state\nsuppression",
        "Broad coverage\n(any decrease)",
    )
    primary_values = [overall[key] for key in primary_keys]
    primary_intervals = [confidence[key] for key in primary_keys]
    lower_error = [value - interval[0] for value, interval in zip(primary_values, primary_intervals)]
    upper_error = [interval[1] - value for value, interval in zip(primary_values, primary_intervals)]
    x = np.arange(len(primary_values))
    axes[0, 0].bar(
        x,
        primary_values,
        yerr=np.asarray([lower_error, upper_error]),
        capsize=5,
        color=[colors["evidence"], colors["strict"], colors["strong"], colors["broad"]],
        edgecolor="white",
        linewidth=0.8,
    )
    for position, value in zip(x, primary_values):
        axes[0, 0].text(position, value + 3.0, "{:.1f}%".format(value), ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_xticks(x, primary_labels)
    axes[0, 0].set_ylim(0.0, 86.0)
    axes[0, 0].set_xlabel("Source-region metric")
    axes[0, 0].set_ylabel("Suppression metric (%)")
    axes[0, 0].set_title("(a) Source-region reflection suppression")
    axes[0, 0].grid(axis="y", alpha=0.25)

    height_keys = sorted(summary["by_height"], key=int)
    evidence_by_height = [summary["by_height"][key]["source_region_attenuation_percent"] for key in height_keys]
    strict_by_height = [summary["by_height"][key]["strict_significant_coverage_percent"] for key in height_keys]
    strong_by_height = [summary["by_height"][key]["strong_suppression_percent"] for key in height_keys]
    hx = np.arange(3)
    axes[0, 1].bar(hx - 0.25, evidence_by_height, 0.25, label="Evidence attenuation", color=colors["evidence"])
    axes[0, 1].bar(hx, strict_by_height, 0.25, label="Strict coverage", color=colors["strict"])
    axes[0, 1].bar(hx + 0.25, strong_by_height, 0.25, label="Strong suppression", color=colors["strong"])
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(hx, ["Low", "Middle", "High"])
    axes[0, 1].set_xlabel("Camera-height level")
    axes[0, 1].set_ylabel("Suppression metric (%)")
    axes[0, 1].set_title("(b) Source-region consistency across heights")
    axes[0, 1].legend(frameon=False, ncol=3, fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.25)

    direction_rows = []
    for direction_index in range(1, 31):
        selected = [row for row in rows if int(row["direction_index"]) == direction_index]
        aggregated = _aggregate(selected)
        aggregated["azimuth_degrees"] = selected[0]["azimuth_degrees"]
        direction_rows.append(aggregated)
    azimuths = [row["azimuth_degrees"] for row in direction_rows]
    direction_evidence = [row["source_region_attenuation_percent"] for row in direction_rows]
    axes[1, 0].plot(azimuths, direction_evidence, color=colors["evidence"], linewidth=2.0, marker="o", markersize=3.2, label="Height-aggregated direction")
    axes[1, 0].axhline(overall["source_region_attenuation_percent"], color=colors["global"], linestyle="--", linewidth=1.2, label="Overall {:.1f}%".format(overall["source_region_attenuation_percent"]))
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("Orbit azimuth (degrees)")
    axes[1, 0].set_ylabel("Evidence attenuation (%)")
    axes[1, 0].set_title("(c) Source-region attenuation by direction")
    axes[1, 0].set_xlim(0.0, 348.0)
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False)

    conservative_keys = ("mean_score_reduction_percent", "any_area_reduction_percent", "strong_area_reduction_percent")
    conservative_labels = ("Whole-image\nmean score", "Net detected\narea", "Net strong\narea")
    conservative_values = [overall[key] for key in conservative_keys]
    conservative_intervals = [confidence[key] for key in conservative_keys]
    conservative_lower = [value - interval[0] for value, interval in zip(conservative_values, conservative_intervals)]
    conservative_upper = [interval[1] - value for value, interval in zip(conservative_values, conservative_intervals)]
    cx = np.arange(3)
    axes[1, 1].bar(cx, conservative_values, yerr=np.asarray([conservative_lower, conservative_upper]), capsize=5, color=["#7B8794", "#8E9A82", "#9B7E70"], edgecolor="white", linewidth=0.8)
    for position, value in zip(cx, conservative_values):
        axes[1, 1].text(position, value + 2.0, "{:.1f}%".format(value), ha="center", va="bottom", fontsize=9)
    axes[1, 1].set_xticks(cx, conservative_labels)
    axes[1, 1].set_ylim(0.0, 36.0)
    axes[1, 1].set_xlabel("Whole-image metric")
    axes[1, 1].set_ylabel("Net relative reduction (%)")
    axes[1, 1].set_title("(d) Conservative whole-image net effects")
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Reflection suppression in matched 3DGS novel views\n"
        "Frozen structure-aware evaluator; 3 heights x 30 directions; "
        "error bars: direction-cluster bootstrap 95% CI",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv or None)
    with args.summary.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    draw(read_rows(args.per_view), summary, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

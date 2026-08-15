#!/usr/bin/env python3
"""Reproduce and verify the frozen 2026-08-08 data behind paper Fig. 3.

This is an audited reproduction entry point, not a claim that the original
outer evaluation script was recovered byte-for-byte.  It freezes the exact
single-view profile, protects the historical result directory from writes,
checks the audited source/reference hashes, and compares a fresh run with the
preserved summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import scipy
from PIL import __version__ as PILLOW_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "StableDelight"))

from experiment_configs import evaluate_v6_specular_reduction as evaluator


REFERENCE_DIR = PROJECT_ROOT / "results/photo_scene6/v6_v7_specular_reduction"
REFERENCE_SUMMARY = REFERENCE_DIR / "summary.json"
REFERENCE_PER_IMAGE = REFERENCE_DIR / "per_image.csv"
REFERENCE_FIGURE = (
    PROJECT_ROOT
    / "results/final_paper_figures/Fig03_2D_reflection_suppression_metrics.png"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results/photo_scene6/fig3_specular_reduction_reproduction"
)
RAW_IMAGE_DIR = PROJECT_ROOT / "data/custom/photo_scene6/images"
OUTPUT_IMAGE_DIR = (
    PROJECT_ROOT / "data/custom/photo_scene6_conventional_v6_colmap/images"
)

FROZEN_PROFILE = {
    "scene": "photo_scene6",
    "images_evaluated": 96,
    "weak_threshold": 0.075,
    "strong_threshold": 0.16,
    "min_relative_drop": 0.20,
    "min_absolute_drop": 0.02,
    "suppression_tiers": (
        ("detectable", "Detectable", 0.00, 0.005),
        ("mild", "Mild", 0.10, 0.010),
        ("meaningful", "Meaningful", 0.20, 0.020),
        ("strong", "Strong", 0.30, 0.030),
    ),
    "region_consensus_levels": (
        ("simple_majority", "Simple majority", 0.50),
        ("robust_majority", "Robust majority", 0.55),
    ),
}

# These hashes identify the audited implementation, including the schema-v2
# per-image naming fix.  The detector hashes also match the 2026-08-09 release.
EXPECTED_SOURCE_HASHES = {
    "experiment_configs/evaluate_v6_specular_reduction.py": (
        "4A3D478895B60D062DEC8232DA4722FCD19707B6CD06943AE7FBE77ACDEC6F4D"
    ),
    "StableDelight/stabledelight/utils/specular_detector.py": (
        "A474E1E1D1E794FB77F560485C06451F4B3EADA4FAD6FDAEB86605F8B504684F"
    ),
    "StableDelight/stabledelight/utils/specular_detector_v3.py": (
        "14954B30DBC200ADDC85FC85612CE29722F8F215392643D93E6FDA0F8CA17A61"
    ),
}

EXPECTED_REFERENCE_HASHES = {
    "summary.json": "4A82513741DFEE49440906087340620FBB1798D0740A6F7C23FC1DA781D023E1",
    "per_image.csv": "61A8F4F818371CE2DC9A7F7ACFC340762094FC0894AC0538C8ED6739373DD77D",
    "Fig03_2D_reflection_suppression_metrics.png": (
        "88E3A897C84013BA3298ED3FB0D2B74D7EF699A8AF497BD5F9D848B7E332F08E"
    ),
}

EXPECTED_INPUT_SETS = {
    "raw": {
        "count": 96,
        "digest": "39E466E6CE194C5C168EC9519E3DA94CB748CCEAA5B8218C6FF9A98E8816072A",
    },
    "processed": {
        "count": 96,
        "digest": "D0EBAB727CBB663E215EBC9EDDA0C4699D728743EC829E47B33A6E971F1519D7",
    },
}

REFERENCE_COMPARISON_KEYS = (
    "metric_name",
    "scene",
    "implementation",
    "images_evaluated",
    "weak_threshold",
    "strong_threshold",
    "min_relative_drop",
    "min_absolute_drop",
    "suppression_tiers",
    "region_consensus_levels",
    "state_encoding",
    "metrics",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _verify_hashes(paths: Dict[str, Path], expected: Dict[str, str]) -> Dict[str, str]:
    actual: Dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = _sha256(path)
        if actual[name] != expected[name]:
            raise RuntimeError(
                "Hash mismatch for {}: expected {}, got {}".format(
                    path, expected[name], actual[name]
                )
            )
    return actual


def _profile_guard() -> None:
    if tuple(evaluator.DEFAULT_SUPPRESSION_TIERS) != FROZEN_PROFILE["suppression_tiers"]:
        raise RuntimeError("Suppression-tier constants drifted from the Fig. 3 profile")
    if tuple(evaluator.REGION_CONSENSUS_LEVELS) != FROZEN_PROFILE["region_consensus_levels"]:
        raise RuntimeError("Region-consensus constants drifted from the Fig. 3 profile")


def _compare_values(
    actual: Any,
    reference: Any,
    location: str,
    differences: List[str],
) -> None:
    if isinstance(reference, dict):
        if not isinstance(actual, dict):
            differences.append("{}: expected a mapping".format(location))
            return
        if set(actual) != set(reference):
            differences.append(
                "{}: key mismatch actual={} reference={}".format(
                    location, sorted(actual), sorted(reference)
                )
            )
            return
        for key in sorted(reference):
            _compare_values(actual[key], reference[key], location + "." + key, differences)
        return
    if isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            differences.append("{}: sequence shape mismatch".format(location))
            return
        for index, (actual_item, reference_item) in enumerate(zip(actual, reference)):
            _compare_values(
                actual_item,
                reference_item,
                "{}[{}]".format(location, index),
                differences,
            )
        return
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        try:
            actual_number = float(actual)
            reference_number = float(reference)
        except (TypeError, ValueError):
            differences.append("{}: non-numeric value {!r}".format(location, actual))
            return
        if not math.isclose(actual_number, reference_number, rel_tol=1e-12, abs_tol=1e-9):
            differences.append(
                "{}: actual={!r}, reference={!r}".format(location, actual, reference)
            )
        return
    if actual != reference:
        differences.append("{}: actual={!r}, reference={!r}".format(location, actual, reference))


def _compare_summary(actual: Dict[str, Any], reference: Dict[str, Any]) -> List[str]:
    differences: List[str] = []
    for key in REFERENCE_COMPARISON_KEYS:
        if key not in actual or key not in reference:
            differences.append("Missing comparison key: {}".format(key))
            continue
        _compare_values(actual[key], reference[key], key, differences)
    return differences


def _image_hashes(directory: Path) -> Dict[str, str]:
    valid = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in valid
    )
    return {path.name: _sha256(path) for path in paths}


def _image_set_manifest(directory: Path) -> Dict[str, Any]:
    hashes = _image_hashes(directory)
    digest = hashlib.sha256()
    for name in sorted(hashes):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashes[name].encode("ascii"))
        digest.update(b"\n")
    return {
        "count": len(hashes),
        "digest": digest.hexdigest().upper(),
        "files": hashes,
    }


def _verify_input_sets(input_sets: Dict[str, Dict[str, Any]]) -> None:
    for name, expected in EXPECTED_INPUT_SETS.items():
        actual = input_sets[name]
        if actual["count"] != expected["count"] or actual["digest"] != expected["digest"]:
            raise RuntimeError(
                "Frozen {} image set drifted: expected {}, got {}".format(
                    name, expected, {"count": actual["count"], "digest": actual["digest"]}
                )
            )


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    partial = path.with_name(path.stem + ".partial.json")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(partial), str(path))


def _configure_font() -> str:
    candidates: Sequence[Path] = (
        Path(evaluator.FONT_PATH),
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf",
    )
    for candidate in candidates:
        if candidate.is_file():
            evaluator.FONT_PATH = str(candidate)
            return str(candidate)
    raise FileNotFoundError("No supported figure font found: {}".format(candidates))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir == REFERENCE_DIR.resolve():
        raise RuntimeError("The frozen historical result directory is read-only")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            "Reproduction output directory is not empty; choose a new --output-dir: {}".format(
                output_dir
            )
        )
    if not RAW_IMAGE_DIR.is_dir() or not OUTPUT_IMAGE_DIR.is_dir():
        raise FileNotFoundError("The frozen raw/output image directories are incomplete")

    _profile_guard()
    source_hashes = _verify_hashes(
        {name: PROJECT_ROOT / name for name in EXPECTED_SOURCE_HASHES},
        EXPECTED_SOURCE_HASHES,
    )
    reference_hashes = _verify_hashes(
        {
            "summary.json": REFERENCE_SUMMARY,
            "per_image.csv": REFERENCE_PER_IMAGE,
            "Fig03_2D_reflection_suppression_metrics.png": REFERENCE_FIGURE,
        },
        EXPECTED_REFERENCE_HASHES,
    )
    input_sets = {
        "raw": _image_set_manifest(RAW_IMAGE_DIR),
        "processed": _image_set_manifest(OUTPUT_IMAGE_DIR),
    }
    _verify_input_sets(input_sets)
    font_path = _configure_font()

    actual = evaluator.evaluate(
        FROZEN_PROFILE["scene"],
        RAW_IMAGE_DIR,
        OUTPUT_IMAGE_DIR,
        None,
        None,
        output_dir,
        FROZEN_PROFILE["weak_threshold"],
        FROZEN_PROFILE["strong_threshold"],
        FROZEN_PROFILE["min_relative_drop"],
        FROZEN_PROFILE["min_absolute_drop"],
        0,
    )
    reference = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))
    differences = _compare_summary(actual, reference)

    manifest = {
        "purpose": "Audited reproduction of the frozen data behind paper Fig. 3",
        "historical_note": (
            "The 2026-08-08 outer evaluator was not recovered byte-for-byte; "
            "this entry point freezes the audited implementation and verifies "
            "its outputs against the preserved historical summary."
        ),
        "reference_summary": str(REFERENCE_SUMMARY),
        "matches_reference": not differences,
        "differences": differences,
        "profile": FROZEN_PROFILE,
        "source_sha256": source_hashes,
        "reference_artifact_sha256": reference_hashes,
        "input_image_sets": input_sets,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": PILLOW_VERSION,
            "font": font_path,
        },
        "command_argv": sys.argv,
        "output_dir": str(output_dir),
    }
    _atomic_json(output_dir / "fig3_reproduction_manifest.json", manifest)

    if differences:
        print("Fig. 3 reproduction differs from the frozen reference:", file=sys.stderr)
        for difference in differences[:50]:
            print("- " + difference, file=sys.stderr)
        return 1
    print("Fig. 3 metrics match the frozen reference summary.")
    print(output_dir / "fig3_reproduction_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

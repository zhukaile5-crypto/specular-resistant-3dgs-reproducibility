#!/usr/bin/env python3
"""Rebuild the six current paper figures without changing source data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image

ASSETS = {
    1: "Fig01_photo_scene6_global_final_method_comparison.png",
    2: "Fig02_photo_scene6_local_final_method_comparison.png",
    3: "Fig03_2D_reflection_suppression_metrics.png",
    4: "Fig07_raw_vs_deglared_3DGS_matched_novel_views_reflection_marked.png",
    5: "Fig08_floating_specular_artifact_suppression_marked.png",
    6: "Fig09_matched_orbit_reflection_evaluation.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(".{}.partial".format(path.name))
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(partial), str(path))


def load_module(path: Path, name: str) -> Any:
    """Load one repository script after the caller has configured sys.path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_fonts() -> Tuple[str, str]:
    """Find regular and bold fonts, preferring the original DejaVu family."""
    candidates = []
    try:
        from matplotlib import font_manager

        regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
        bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
        candidates.append((regular, bold))
    except Exception:
        pass
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates.extend(
        [
            (str(windows / "arial.ttf"), str(windows / "arialbd.ttf")),
            ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
        ]
    )
    for regular, bold in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            return regular, bold
        try:
            from PIL import ImageFont

            ImageFont.truetype(regular, 12)
            ImageFont.truetype(bold, 12)
            return regular, bold
        except OSError:
            continue
    raise FileNotFoundError("No usable regular/bold font pair was found")


def pixel_match(left: Path, right: Path) -> bool:
    with Image.open(left) as a, Image.open(right) as b:
        return a.size == b.size and a.mode == b.mode and a.tobytes() == b.tobytes()


def _configure_imports(project_root: Path) -> None:
    for path in (
        project_root,
        project_root / "StableDelight",
        project_root / "third_party/gsplat",
        project_root / "third_party/gsplat/examples",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def compose_figures_1_2(project_root: Path, staging: Path) -> Dict[int, Path]:
    module = load_module(
        project_root / "experiment_configs/build_v1_v6_visual_comparison.py",
        "paper_repro_build_figures_1_2",
    )
    module.PROJECT_ROOT = project_root
    module.OUTPUT_DIR = staging / "figure_01_02_sources"
    module.main()
    return {
        1: module.OUTPUT_DIR / "photo_scene6_0051_final_method_global.png",
        2: module.OUTPUT_DIR / "photo_scene6_0012_final_method_cup_detail.png",
    }


def compose_figure_4(project_root: Path, staging: Path) -> Path:
    module = load_module(
        project_root / "experiment_configs/build_large_label_comparison_figures.py",
        "paper_repro_build_figure_4",
    )
    regular, bold = find_fonts()
    module.PROJECT_ROOT = project_root
    module.FONT_PATH = regular
    module.FONT_BOLD_PATH = bold
    module.PRESENTATION_DIR = staging / "figure_04_sources"
    return module.build_novel_view_sheet_1x6_reflection_marked()


def compose_figure_5(project_root: Path, staging: Path) -> Path:
    module = load_module(
        project_root / "experiment_configs/build_large_label_comparison_figures.py",
        "paper_repro_build_figure_5",
    )
    regular, bold = find_fonts()
    module.PROJECT_ROOT = project_root
    module.FONT_PATH = regular
    module.FONT_BOLD_PATH = bold
    module.PRESENTATION_DIR = staging / "figure_05_sources"
    module.build_artifact_sheet()
    return module.build_artifact_sheet_marked()


def compose_figure_6(project_root: Path, staging: Path) -> Path:
    # Matplotlib is optional unless Figure 6 is being redrawn.
    from tools.plot_figure6 import draw as draw_figure_6
    from tools.plot_figure6 import read_rows as read_figure_6_rows

    source = project_root / "results/photo_scene6/full96_reestimated_pose/matched_orbit_specularity_3x30"
    summary_path = source / "summary.json"
    per_view_path = source / "per_view_metrics.csv"
    if not summary_path.is_file() or not per_view_path.is_file():
        raise FileNotFoundError("Figure 6 requires {} and {}".format(summary_path, per_view_path))
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    output = staging / "figure_06_sources/matched_orbit_reflection_evaluation.png"
    draw_figure_6(read_figure_6_rows(per_view_path), summary, output)
    return output


def recompute_figure_3(project_root: Path, staging: Path) -> Path:
    """Run the audited 96-image evaluator into a fresh staging directory."""
    run_dir = staging / "figure_03_metric_run"
    command = [
        sys.executable,
        str(project_root / "experiment_configs/reproduce_fig3_specular_reduction.py"),
        "--output-dir",
        str(run_dir),
    ]
    subprocess.run(command, cwd=str(project_root), check=True)
    output = run_dir / "paper_figures/v6_v7_reflection_suppression_academic.png"
    if not output.is_file():
        raise FileNotFoundError("The Figure 3 evaluator did not produce {}".format(output))
    return output


def parse_ids(values: Iterable[str]) -> List[int]:
    values = list(values)
    if values == ["all"]:
        return list(range(1, 7))
    result = sorted(set(int(value) for value in values))
    if not result or result[0] < 1 or result[-1] > 6:
        raise ValueError("Figure IDs must be 1--6 or the single token 'all'")
    return result


def reproduce(
    package_root: Path,
    project_root: Path,
    output_dir: Path,
    ids: Sequence[int],
    mode: str,
) -> Dict[str, Any]:
    """Reproduce the selected figures and return a verification report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / "intermediate"
    staging.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    expected = {int(item["id"]): item["reference_sha256"] for item in manifest["figures"]}
    generated: Dict[int, Path] = {}

    project_root = project_root.resolve()
    _configure_imports(project_root)
    unsupported = set(ids)
    if mode == "compose":
        if 1 in ids or 2 in ids:
            generated.update({key: value for key, value in compose_figures_1_2(project_root, staging).items() if key in ids})
            unsupported.difference_update((1, 2))
        if 4 in ids:
            generated[4] = compose_figure_4(project_root, staging)
            unsupported.remove(4)
        if 5 in ids:
            generated[5] = compose_figure_5(project_root, staging)
            unsupported.remove(5)
        if 6 in ids:
            generated[6] = compose_figure_6(project_root, staging)
            unsupported.remove(6)
    elif mode == "recompute":
        if 3 in ids:
            generated[3] = recompute_figure_3(project_root, staging)
            unsupported.remove(3)
    if unsupported:
        raise ValueError(
            "Mode '{}' does not support figures {}. Use compose for 1,2,4,5,6 "
            "or recompute for 3.".format(mode, sorted(unsupported))
        )

    records = []
    for figure_id in ids:
        source = generated[figure_id]
        destination = output_dir / ASSETS[figure_id]
        if source.resolve() != destination.resolve():
            shutil.copy2(str(source), str(destination))
        reference = project_root / "results/final_paper_figures" / ASSETS[figure_id]
        digest = sha256(destination)
        records.append(
            {
                "figure": figure_id,
                "mode": mode,
                "output": str(destination),
                "sha256": digest,
                "expected_sha256": expected[figure_id],
                "byte_identical": digest.lower() == expected[figure_id].lower(),
                "pixel_identical": pixel_match(destination, reference) if reference.is_file() else None,
            }
        )
    report = {
        "schema": "paper_figure_reproduction_report_v1",
        "project_root": str(project_root),
        "mode": mode,
        "records": records,
    }
    write_json_atomic(output_dir / "figure_reproduction_report.json", report)
    return report


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("compose", "recompute"), required=True)
    parser.add_argument("--ids", nargs="+", default=["all"])
    args = parser.parse_args(argv or None)
    report = reproduce(
        args.package_root.resolve(),
        args.project_root.resolve(),
        args.output_dir.resolve(),
        parse_ids(args.ids),
        args.mode,
    )
    for record in report["records"]:
        print(
            "Figure {figure}: byte_identical={byte_identical}, "
            "pixel_identical={pixel_identical} -> {output}".format(**record)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

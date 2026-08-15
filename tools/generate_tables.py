#!/usr/bin/env python3
"""Generate all five paper tables from preserved machine-readable results."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SCENES = (
    ("photo_scene", "S1"),
    ("photo_scene2", "S2"),
    ("photo_scene3", "S3"),
    ("photo_scene4", "S4"),
    ("photo_scene5", "S5"),
    ("photo_scene6", "S6 (dev.)"),
)


def read_json(path: Path) -> Dict[str, Any]:
    """Read one UTF-8 JSON object and report a useful missing-file error."""
    if not path.is_file():
        raise FileNotFoundError("Required reproduction input is missing: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text_atomic(path: Path, text: str) -> None:
    """Write text through a sibling partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(".{}.partial".format(path.name))
    partial.write_text(text, encoding="utf-8")
    os.replace(str(partial), str(path))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _fmt(value: float, digits: int) -> str:
    return ("{:.%df}" % digits).format(float(value))


def _v6_steepness(project_root: Path) -> float:
    """Read the V6 mapping steepness from the implementation signature."""
    source = project_root / "experiment_configs/blend_strategies.py"
    text = source.read_text(encoding="utf-8")
    matches = re.findall(r"steepness:\s*float\s*=\s*([0-9.]+)", text)
    if "8.0" not in matches:
        raise RuntimeError("The frozen steepness=8.0 implementation guard failed")
    return 8.0


def collect_table_1(project_root: Path) -> List[List[str]]:
    """Collect the selected frozen V6 operational parameters."""
    profile = read_json(project_root / "versions/registry.json")["v6"]
    blend = profile["blend_parameters"]
    steepness = _v6_steepness(project_root)
    return [
        ["threshold/steepness/data_factor", "{:.2f}/{:g}/{:d}".format(profile["threshold"], steepness, profile["data_factor"])],
        ["foreground_score/supported_score", "{:.2f}/{:.3f}".format(blend["foreground_score"], blend["supported_score"])],
        ["background_score/background_evidence", "{:.3f}/{:.3f}".format(blend["background_score"], blend["background_evidence"])],
        ["matte_score_low/matte_score_high", "{:.3f}/{:.2f}".format(blend["matte_score_low"], blend["matte_score_high"])],
        ["matte_evidence_low/matte_evidence_high", "{:.2f}/{:.2f}".format(blend["matte_evidence_low"], blend["matte_evidence_high"])],
        ["structure_barrier_strength", "{:.2f}".format(blend["structure_barrier_strength"])],
        ["matte_smoothness/matte_iterations", "{:.1f}/{:g}".format(blend["matte_smoothness"], blend["matte_iterations"])],
        ["matte_edge_sigma/matte_component_floor", "{:.2f}/{:.2f}".format(blend["matte_edge_sigma"], blend["matte_component_floor"])],
        ["weak_mapping_threshold/weak_boost_gain", "{:.2f}/{:.2f}".format(blend["weak_mapping_threshold"], blend["weak_boost_gain"])],
        ["small_width/large_width", "{:g}/{:g} px".format(blend["small_width"], blend["large_width"])],
        ["small_gain/large_gain", "{:.2f}/{:.2f}".format(blend["small_gain"], blend["large_gain"])],
        ["width_reference_short_side", "{:g} px".format(blend["width_reference_short_side"])],
    ]


def collect_table_2() -> List[List[str]]:
    """Return the descriptive component-to-implementation mapping."""
    return [
        ["Single-view evidence", "Signed multi-scale luminance and chromaticity cues", "Dense reflection score without treating dark contrast as reflection"],
        ["Cross-view evidence", "Affine exposure alignment, multi-scale track sampling, positive per-observation residuals, and geometric confidence", "View-specific reflection support and reliability map"],
        ["Structure model", "Paired signed edges, gradient energy, orientation coherence, and auxiliary-evidence rescue", "Protection of edges and coherent high-frequency material texture"],
        ["Evidence matte", "Foreground/background seeds, connectivity filtering, and structure-aware constrained diffusion", "Continuous reflection likelihood, uncertainty, and boundary-consistent support"],
        ["Adaptive fusion", "Native-reference scale proxy plus separated CIELAB base/detail/chroma weights", "Scale-aware suppression with tone/detail preservation"],
        ["Accelerated execution", "Batched CUDA feature construction, GPU residency, concurrent streams, and CPU operations retained for numerical equivalence", "Reproducible final imagery at lower processing time"],
    ]


def _suppression_summary(project_root: Path, scene: str) -> Dict[str, Any]:
    if scene == "photo_scene6":
        path = project_root / "results/photo_scene6/v6_v7_specular_reduction/summary.json"
    else:
        path = project_root / "results/cross_scene_frozen_metrics_20260811" / scene / "summary.json"
    return read_json(path)


def collect_table_3(project_root: Path) -> Dict[str, List[List[str]]]:
    """Collect both panels of the cross-scene 2D table."""
    fidelity = read_json(project_root / "results/2d_structure_fidelity_20260811/summary.json")
    panel_a: List[List[str]] = []
    panel_b: List[List[str]] = []
    for scene, label in SCENES:
        suppression = _suppression_summary(project_root, scene)
        metrics = suppression["metrics"]
        area = metrics["global_area_percent"]
        panel_a.append(
            [
                label,
                str(suppression["images_evaluated"]),
                _fmt(area["any_area_reduction"], 2),
                _fmt(area["strong_area_reduction"], 2),
                _fmt(metrics["source_evidence_attenuation"], 2),
                _fmt(metrics["meaningful_improvement_rate"], 2),
                _fmt(metrics["strong_suppression_rate"], 2),
                _fmt(metrics["region_consistent_coverage_percent"]["robust_majority"]["all"], 2),
            ]
        )
        scene_fidelity = fidelity["scenes"][scene]
        stats = scene_fidelity["per_image_statistics"]
        panel_b.append(
            [
                label,
                str(scene_fidelity["frames"]),
                _fmt(stats["protected_percent"]["median"], 2),
                _fmt(stats["direct_cie76_delta_e_mean"]["median"], 2),
                _fmt(stats["fusion_cie76_delta_e_mean"]["median"], 2),
                _fmt(stats["direct_highpass_correlation"]["median"], 3),
                _fmt(stats["fusion_highpass_correlation"]["median"], 3),
                _fmt(stats["direct_edge_energy_ratio"]["median"], 3),
                _fmt(stats["fusion_edge_energy_ratio"]["median"], 3),
            ]
        )
    overall = fidelity["overall"]["per_image_statistics"]
    panel_b.append(
        [
            "All",
            str(fidelity["overall"]["frames"]),
            _fmt(overall["protected_percent"]["median"], 2),
            _fmt(overall["direct_cie76_delta_e_mean"]["median"], 2),
            _fmt(overall["fusion_cie76_delta_e_mean"]["median"], 2),
            _fmt(overall["direct_highpass_correlation"]["median"], 3),
            _fmt(overall["fusion_highpass_correlation"]["median"], 3),
            _fmt(overall["direct_edge_energy_ratio"]["median"], 3),
            _fmt(overall["fusion_edge_energy_ratio"]["median"], 3),
        ]
    )
    return {"panel_a": panel_a, "panel_b": panel_b}


def collect_table_4(project_root: Path) -> List[List[str]]:
    """Collect the fixed-pose 84/12 held-out control metrics."""
    root = project_root / "results/photo_scene6/conventional_reconstruction_pair_retry2"
    raw = read_json(root / "raw/spec_eval_step29999.json")["aggregate"]
    processed = read_json(root / "v6_v7_deglared/spec_eval_step29999.json")["aggregate"]
    raw_view = read_json(root / "raw/view_dependence_step29999.json")["render"]
    processed_view = read_json(root / "v6_v7_deglared/view_dependence_step29999.json")["render"]
    return [
        ["Raw-target PSNR (dB)", _fmt(raw["psnr_orig"], 2), _fmt(processed["psnr_orig"], 2)],
        ["Suppressed-target PSNR (dB)", _fmt(raw["psnr_deli"], 2), _fmt(processed["psnr_deli"], 2)],
        ["Source-core proxy score", _fmt(raw["render_spec_score"], 4), _fmt(processed["render_spec_score"], 4)],
        ["Whole-image proxy score", _fmt(raw["render_score_mean"], 4), _fmt(processed["render_score_mean"], 4)],
        ["Texture high-frequency PSNR (dB)", _fmt(raw["psnr_hf_textured"], 2), _fmt(processed["psnr_hf_textured"], 2)],
        ["High-variance point luminance variance", _fmt(raw_view["mean_std_render"], 4), _fmt(processed_view["mean_std_render"], 4)],
    ]


def _table_5_paths(project_root: Path, scene_number: int) -> Dict[str, Path]:
    if scene_number in (2, 5):
        root = project_root / "results/end_to_end_3dgs_cross_scene_20260811" / "photo_scene{}".format(scene_number)
        evaluation = root / "matched_common_support_trajectory_90"
    elif scene_number in (3, 4):
        root = project_root / "results/end_to_end_3dgs_cross_scene_20260811" / "photo_scene{}".format(scene_number)
        evaluation = root / "matched_orbit_3x30"
    elif scene_number == 6:
        root = project_root / "results/photo_scene6/full96_reestimated_pose"
        evaluation = root / "matched_orbit_specularity_3x30"
    else:
        raise ValueError(scene_number)
    return {"root": root, "evaluation": evaluation}


def collect_table_5(project_root: Path) -> Dict[str, List[List[str]]]:
    """Collect the final mixed-trajectory cross-scene 3DGS table."""
    panel_a: List[List[str]] = []
    panel_b: List[List[str]] = []
    for scene_number in range(2, 7):
        paths = _table_5_paths(project_root, scene_number)
        summary = read_json(paths["evaluation"] / "summary.json")
        manifest = read_json(paths["evaluation"] / "trajectory_manifest.json")
        if scene_number == 6:
            raw_count = processed_count = int(manifest["common_camera_count"])
        else:
            counts = read_json(paths["root"] / "registered_frame_counts.json")
            raw_count = int(counts["raw_registered_frames"])
            processed_count = int(counts["deglared_registered_frames"])
        support_count = int(summary["trajectory_support"]["within_support_count"])
        view_count = int(summary["overall"]["view_count"])
        similarity = manifest["similarity_raw_to_deglared"]
        if "camera_center_alignment_rmse_in_raw_spacings" in similarity:
            normalized_rmse = 100.0 * similarity["camera_center_alignment_rmse_in_raw_spacings"]
        else:
            spacing = manifest["support"]["median_training_camera_spacing"]
            normalized_rmse = 100.0 * similarity["camera_center_alignment_rmse"] / spacing
        label = "S{}".format(scene_number)
        role = "Development" if scene_number == 6 else "Additional"
        panel_a.append(
            [
                label,
                role,
                "{}/{}".format(raw_count, processed_count),
                "{}/{}".format(support_count, view_count),
                _fmt(normalized_rmse, 2),
            ]
        )
        overall = summary["overall"]
        panel_b.append(
            [
                label,
                _fmt(overall["mean_score_reduction_percent"], 2),
                _fmt(overall["any_area_reduction_percent"], 2),
                _fmt(overall["strong_area_reduction_percent"], 2),
                _fmt(overall["source_region_attenuation_percent"], 2),
                _fmt(overall["broad_improvement_coverage_percent"], 2),
                _fmt(overall["strict_significant_coverage_percent"], 2),
                _fmt(overall["strong_suppression_percent"], 2),
            ]
        )
    return {"panel_a": panel_a, "panel_b": panel_b}


def collect_all(project_root: Path) -> Dict[str, Any]:
    """Collect display-ready values for all paper tables."""
    table_3 = collect_table_3(project_root)
    table_5 = collect_table_5(project_root)
    return {
        "schema": "generated_paper_tables_v1",
        "table_1": collect_table_1(project_root),
        "table_2": collect_table_2(),
        "table_3a": table_3["panel_a"],
        "table_3b": table_3["panel_b"],
        "table_4": collect_table_4(project_root),
        "table_5a": table_5["panel_a"],
        "table_5b": table_5["panel_b"],
    }


def _markdown(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")


def _latex_tabular(headers: Sequence[str], rows: Sequence[Sequence[str]], alignment: str) -> str:
    lines = [r"\begin{tabular}{%s}" % alignment, r"\toprule", " & ".join(_latex_escape(value) for value in headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(_latex_escape(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_outputs(data: Dict[str, Any], output_dir: Path) -> None:
    """Write JSON, Markdown, and LaTeX outputs."""
    write_json_atomic(output_dir / "tables.json", data)

    markdown_sections = [
        "# Generated Paper Tables",
        "",
        "## Table 1 -- Method parameters",
        "",
        _markdown(("Parameter", "Value"), data["table_1"]),
        "",
        "## Table 2 -- Method components",
        "",
        _markdown(("Component", "Mechanism", "Role and output"), data["table_2"]),
        "",
        "## Table 3(a) -- Cross-scene 2D proxy suppression",
        "",
        _markdown(("Scene", "Frames", "Any area", "Strong area", "Conditional attenuation", "Meaningful coverage", "Strong-state suppression", "Region-consistent coverage"), data["table_3a"]),
        "",
        "## Table 3(b) -- Protected-structure fidelity",
        "",
        _markdown(("Scene", "Frames", "Protected area", "Direct DeltaE", "Proposed DeltaE", "Direct HF corr.", "Proposed HF corr.", "Direct edge energy", "Proposed edge energy"), data["table_3b"]),
        "",
        "## Table 4 -- Held-out fixed-pose control",
        "",
        _markdown(("Metric", "Raw input", "Suppressed input"), data["table_4"]),
        "",
        "## Table 5(a) -- Reconstruction and trajectory diagnostics",
        "",
        _markdown(("Scene", "Role", "Registered raw/suppressed", "Common-support poses", "Centre RMSE/spacing"), data["table_5a"]),
        "",
        "## Table 5(b) -- Matched-trajectory proxy changes",
        "",
        _markdown(("Scene", "Whole score", "Any area", "Strong area", "Conditional attenuation", "Broad coverage", "Strict coverage", "Strong-state suppression"), data["table_5b"]),
        "",
    ]
    write_text_atomic(output_dir / "tables.md", "\n".join(markdown_sections))

    write_text_atomic(output_dir / "table_01_method_parameters.tex", _latex_tabular(("Parameter", "Value"), data["table_1"], "lr"))
    write_text_atomic(output_dir / "table_02_method_components.tex", _latex_tabular(("Component", "Mechanism", "Role and output"), data["table_2"], "lll"))
    write_text_atomic(output_dir / "table_03a_cross_scene_2d_proxy.tex", _latex_tabular(("Scene", "Frames", "Any area", "Strong area", "Conditional attenuation", "Meaningful coverage", "Strong-state suppression", "Region-consistent coverage"), data["table_3a"], "lrrrrrrr"))
    write_text_atomic(output_dir / "table_03b_structure_fidelity.tex", _latex_tabular(("Scene", "Frames", "Protected area", "Direct $\\Delta E$", "Proposed $\\Delta E$", "Direct HF", "Proposed HF", "Direct edge", "Proposed edge"), data["table_3b"], "lrrrrrrrr"))
    write_text_atomic(output_dir / "table_04_heldout_control.tex", _latex_tabular(("Metric", "Raw input", "Suppressed input"), data["table_4"], "lrr"))
    write_text_atomic(output_dir / "table_05a_reconstruction_diagnostics.tex", _latex_tabular(("Scene", "Role", "Registered raw/suppressed", "Common-support poses", "Centre RMSE/spacing (\\%)"), data["table_5a"], "llrrr"))
    write_text_atomic(output_dir / "table_05b_cross_scene_3d.tex", _latex_tabular(("Scene", "Whole score", "Any area", "Strong area", "Conditional attenuation", "Broad coverage", "Strict coverage", "Strong-state suppression"), data["table_5b"], "lrrrrrrr"))


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv or None)
    data = collect_all(args.project_root.resolve())
    render_outputs(data, args.output_dir.resolve())
    print("Generated Tables 1--5 in {}".format(args.output_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

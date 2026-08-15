#!/usr/bin/env python3
"""Canonical V1--V6 processing-system registry.

The registry separates release names from detector/cache names.  V3--V6
share the V3 detector evidence, while their blending policies differ.  V1 is
kept as an artifact-only historical release because its exact source snapshot
is no longer present in the repository.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from experiment_configs.blend_strategies import (
    ADAPTIVE_FEATHER_DEFAULT_PARAMETERS,
    ADAPTIVE_FEATHER_VERSION,
    EVIDENCE_MATTE_DEFAULT_PARAMETERS,
    EVIDENCE_MATTE_VERSION,
    WEAK_HYSTERESIS_FEATHER_DEFAULT_PARAMETERS,
    WEAK_HYSTERESIS_FEATHER_VERSION,
    WEAK_HYSTERESIS_OUTER_FEATHER_DEFAULT_PARAMETERS,
    WEAK_HYSTERESIS_OUTER_FEATHER_VERSION,
)

PROJECT_ROOT = Path("/home/dministrator/nvidia_project")


@dataclass(frozen=True)
class VersionProfile:
    """Immutable description of one named processing release."""

    release: str
    title: str
    status: str
    detector_version: Optional[str]
    detector_algorithm: str
    crossview_algorithm: str
    blend_mode: str
    threshold: float = 0.3
    data_factor: int = 4
    blend_strategy_version: Optional[str] = None
    blend_parameters: Dict[str, float] = field(default_factory=dict)
    source_available: bool = True
    notes: str = ""

    @property
    def runnable(self) -> bool:
        return self.source_available and self.detector_version is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "release": self.release,
            "title": self.title,
            "status": self.status,
            "runnable": self.runnable,
            "source_available": self.source_available,
            "detector_version": self.detector_version,
            "detector_algorithm": self.detector_algorithm,
            "crossview_algorithm": self.crossview_algorithm,
            "blend_mode": self.blend_mode,
            "threshold": self.threshold,
            "data_factor": self.data_factor,
            "blend_strategy_version": self.blend_strategy_version,
            "blend_parameters": dict(self.blend_parameters),
            "notes": self.notes,
        }


VERSION_PROFILES: Dict[str, VersionProfile] = {
    "v1": VersionProfile(
        release="v1",
        title="Legacy score-map + soft blend",
        status="artifact_only",
        detector_version=None,
        detector_algorithm="legacy_unversioned_v1",
        crossview_algorithm="legacy_unversioned_v1",
        blend_mode="soft",
        source_available=False,
        notes=(
            "The 96-image photo_scene6 outputs and manifests remain, but the "
            "exact pre-V2 detector source snapshot is not present."
        ),
    ),
    "v2": VersionProfile(
        release="v2",
        title="Robust signed multi-scale detector",
        status="reproducible_baseline",
        detector_version="v2",
        detector_algorithm="signed_multiscale_v2",
        crossview_algorithm="robust_affine_upper_v2",
        blend_mode="soft",
        notes="Exposure-normalised cross-view support and dark-seam suppression.",
    ),
    "v3": VersionProfile(
        release="v3",
        title="Structure-aware detector + soft blend",
        status="reproducible_baseline",
        detector_version="v3",
        detector_algorithm="structure_observation_v3",
        crossview_algorithm="observation_residual_confidence_v3",
        blend_mode="soft",
        notes="Adds structure suppression, per-observation residuals, and confidence maps.",
    ),
    "v4": VersionProfile(
        release="v4",
        title="Scale-adaptive perceptual feathering",
        status="previous_flagship",
        detector_version="v3",
        detector_algorithm="structure_observation_v3",
        crossview_algorithm="observation_residual_confidence_v3",
        blend_mode="adaptive_feather",
        blend_strategy_version=ADAPTIVE_FEATHER_VERSION,
        blend_parameters=dict(ADAPTIVE_FEATHER_DEFAULT_PARAMETERS),
        notes="Preserved previous flagship; width 6/12 and gain 0.35/1.15.",
    ),
    "v5": VersionProfile(
        release="v5",
        title="Weak-reflection hysteresis + continuous feathering",
        status="experimental_candidate",
        detector_version="v3",
        detector_algorithm="structure_observation_v3",
        crossview_algorithm="observation_residual_confidence_v3",
        blend_mode="weak_hysteresis_feather",
        blend_strategy_version=WEAK_HYSTERESIS_FEATHER_VERSION,
        blend_parameters=dict(WEAK_HYSTERESIS_FEATHER_DEFAULT_PARAMETERS),
        notes=(
            "Experimental successor to V4; evidence-gated weak-highlight "
            "recovery and continuous component-boundary feathering."
        ),
    ),
    "v5.1": VersionProfile(
        release="v5.1",
        title="Evidence-constrained outer feathering",
        status="experimental_candidate",
        detector_version="v3",
        detector_algorithm="structure_observation_v3",
        crossview_algorithm="observation_residual_confidence_v3",
        blend_mode="weak_hysteresis_outer_feather",
        blend_strategy_version=WEAK_HYSTERESIS_OUTER_FEATHER_VERSION,
        blend_parameters=dict(
            WEAK_HYSTERESIS_OUTER_FEATHER_DEFAULT_PARAMETERS
        ),
        notes=(
            "Preserves V5 weak-highlight recovery while moving the blend "
            "transition into an evidence-constrained outer shell."
        ),
    ),
    "v6": VersionProfile(
        release="v6",
        title="Evidence-domain reflection matte",
        status="flagship",
        detector_version="v3",
        detector_algorithm="structure_observation_v3",
        crossview_algorithm="observation_residual_confidence_v3",
        blend_mode="evidence_matte",
        blend_strategy_version=EVIDENCE_MATTE_VERSION,
        blend_parameters=dict(EVIDENCE_MATTE_DEFAULT_PARAMETERS),
        notes=(
            "Current flagship. Infers a continuous reflection matte from "
            "foreground/background evidence and structure-aware diffusion; "
            "native-reference width 2/10 and gain 0.45/1.10 affect only "
            "post-matte strength; contains no boundary shift."
        ),
    ),
}


def get_version_profile(release: str) -> VersionProfile:
    """Return a registered release, accepting case-insensitive names."""
    key = release.lower()
    try:
        return VERSION_PROFILES[key]
    except KeyError:
        raise ValueError(
            f"Unknown release '{release}'; choose from {sorted(VERSION_PROFILES)}"
        ) from None


def version_paths(scene: str, release: str) -> Dict[str, Path]:
    """Return stable artifact paths for one scene/release.

    V1 owns the historical unversioned names.  V2 is explicitly isolated by
    ``_v2``; V3--V6 share V3 detector evidence but have distinct blends.
    """
    profile = get_version_profile(release)
    data = PROJECT_ROOT / "data" / "custom"
    results = PROJECT_ROOT / "results"
    common = {
        "raw_dir": PROJECT_ROOT / "raw_images" / scene,
        "delighted_dir": data / f"{scene}_delighted",
        "orig_colmap_dir": data / f"{scene}_colmap",
    }
    if profile.release == "v1":
        suffix = ""
    elif profile.release == "v2":
        suffix = "_v2"
    else:
        suffix = "_v3"

    tag = (
        f"{profile.blend_mode}_t{profile.threshold:g}"
        if profile.blend_mode
        in (
            "soft",
            "hard",
            "adaptive_feather",
            "weak_hysteresis_feather",
            "weak_hysteresis_outer_feather",
            "evidence_matte",
        )
        else profile.blend_mode
    )
    return {
        **common,
        "crossview_dir": data / f"{scene}_crossview_maps{suffix}",
        "scoremap_dir": data / f"{scene}_scoremaps{suffix}",
        "blended_dir": data / f"{scene}_blended_{tag}{suffix}",
        "train_colmap_dir": data / f"{scene}_train_colmap_{tag}{suffix}",
        "result_dir": results / scene / f"specular_aware_{tag}{suffix}",
    }

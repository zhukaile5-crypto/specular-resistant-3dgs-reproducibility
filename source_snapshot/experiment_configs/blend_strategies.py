#!/usr/bin/env python3
"""Pluggable blending strategies for specular-aware reconstruction.

Each strategy receives the original image, the delighted image and a
per-pixel **specular score map** (threshold-independent, see
``stabledelight.utils.specular_detector.compute_specular_score``), maps
the score to a blend weight however it likes, and returns the blended
training image::

    blended = strategy(original, delighted, score_map, **params)

Conventions:

- ``original`` / ``delighted``: float32 (H, W, 3) arrays in [0, 1].
- ``score_map``: float32 (H, W) array in [0, 1] (higher = more likely
  specular), or ``None`` for strategies registered with
  ``needs_scoremap=False``.
- Return value: float32 (H, W, 3) array in [0, 1].

The score → weight mapping is where adaptive research happens: fixed
global thresholds, per-image / per-region adaptive thresholds, or
learned policies are all just new registered strategies.  To add one,
simply register a function here — the data preparation and training
pipelines pick it up automatically via ``--blend-mode`` and remain
otherwise unchanged.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

BlendFn = Callable[..., Any]

ADAPTIVE_FEATHER_VERSION = "size_view_perceptual_v1"

# Canonical V4 parameters.  Keep these in one place so the production
# pipeline, manifests, tests, and ablation tooling cannot silently disagree.
ADAPTIVE_FEATHER_SMALL_WIDTH = 6.0
ADAPTIVE_FEATHER_LARGE_WIDTH = 12.0
ADAPTIVE_FEATHER_SMALL_GAIN = 0.35
ADAPTIVE_FEATHER_LARGE_GAIN = 1.15
ADAPTIVE_FEATHER_DEFAULT_PARAMETERS = {
    "small_width": ADAPTIVE_FEATHER_SMALL_WIDTH,
    "large_width": ADAPTIVE_FEATHER_LARGE_WIDTH,
    "small_gain": ADAPTIVE_FEATHER_SMALL_GAIN,
    "large_gain": ADAPTIVE_FEATHER_LARGE_GAIN,
}

# Historical V5 parameters. V4 retains the constants above; V5 has its own
# strategy name, manifest and output paths.
WEAK_HYSTERESIS_FEATHER_VERSION = "weak_hysteresis_continuous_v1"
WEAK_HYSTERESIS_FEATHER_DEFAULT_PARAMETERS = {
    "weak_low": 0.10,
    "weak_mapping_threshold": 0.16,
    "weak_boost_gain": 0.35,
    "evidence_seed": 0.18,
    "small_width": 6.0,
    "large_width": 12.0,
    "small_gain": 0.35,
    "large_gain": 1.15,
    "small_feather": 0.5,
    "large_feather": 2.5,
    "color_feather_gain": 1.0,
    "boundary_bias": 1.0,
    "transition_guard_strength": 0.35,
    "base_edge_lock": 0.2,
}

# Experimental V5.1 parameters.  The detector and weak-region recovery are
# inherited from V5, but the transition is moved to an evidence-constrained
# outer shell instead of being biased into the detected reflection.
WEAK_HYSTERESIS_OUTER_FEATHER_VERSION = "evidence_outer_feather_v2"
WEAK_HYSTERESIS_OUTER_FEATHER_DEFAULT_PARAMETERS = {
    "weak_low": 0.10,
    "weak_mapping_threshold": 0.16,
    "weak_boost_gain": 0.35,
    "evidence_seed": 0.18,
    "small_width": 6.0,
    "large_width": 12.0,
    "small_gain": 0.35,
    "large_gain": 1.15,
    "small_feather": 0.5,
    "large_feather": 2.5,
    "color_feather_gain": 1.0,
    "boundary_bias": 1.0,
    "transition_guard_strength": 0.35,
    "base_edge_lock": 0.2,
    "outer_feather_mode": 1.0,
    "outer_low": 0.05,
    "outer_evidence": 0.10,
    "small_outer_radius": 0.75,
    "large_outer_radius": 3.0,
    "inner_feather_radius": 0.5,
    "max_profile_shift": 4.0,
}

# V6 flagship parameters. Unlike V5.1, these are evidence-domain
# thresholds and optimisation weights: no parameter represents a pixel shift
# of the final transition.
EVIDENCE_MATTE_VERSION = "evidence_matte_diffusion_v2"
EVIDENCE_MATTE_DEFAULT_PARAMETERS = {
    "matte_score_low": 0.015,
    "matte_score_high": 0.12,
    "matte_evidence_low": 0.03,
    "matte_evidence_high": 0.18,
    "foreground_score": 0.30,
    "supported_score": 0.075,
    "foreground_evidence": 0.18,
    "background_score": 0.012,
    "background_evidence": 0.025,
    "structure_barrier_strength": 0.85,
    "matte_smoothness": 1.5,
    "matte_iterations": 24.0,
    "matte_edge_sigma": 0.08,
    "matte_component_floor": 0.02,
    "weak_mapping_threshold": 0.16,
    "weak_boost_gain": 0.35,
    "small_width": 2.0,
    "large_width": 10.0,
    "small_gain": 0.45,
    "large_gain": 1.10,
    "width_reference_short_side": 3072.0,
    "base_edge_lock": 0.2,
}

_REGISTRY: Dict[str, BlendFn] = {}


def register(
    name: str,
    *,
    needs_scoremap: bool = True,
    provides_diagnostics: bool = False,
    needs_v3_context: bool = False,
    needs_detector_cues: bool = False,
    version: Optional[str] = None,
) -> Callable[[BlendFn], BlendFn]:
    """Register a blending strategy under ``name``.

    Args:
        name: CLI-visible strategy name (used with ``--blend-mode``).
        needs_scoremap: Whether the strategy requires a specular score
            map. Strategies with ``needs_scoremap=False`` receive
            ``score_map=None`` and skip detector computation.
        provides_diagnostics: Whether ``return_diagnostics=True`` returns
            ``(image, maps)`` for audit output.
        needs_v3_context: Whether the strategy requires V3 observation and
            confidence maps in addition to the final score.
        needs_detector_cues: Whether the strategy consumes the V3 detector's
            independent support and structure maps.
    """

    def deco(fn: BlendFn) -> BlendFn:
        fn.needs_scoremap = needs_scoremap  # type: ignore[attr-defined]
        fn.provides_diagnostics = provides_diagnostics  # type: ignore[attr-defined]
        fn.needs_v3_context = needs_v3_context  # type: ignore[attr-defined]
        fn.needs_detector_cues = needs_detector_cues  # type: ignore[attr-defined]
        fn.strategy_version = version  # type: ignore[attr-defined]
        _REGISTRY[name] = fn
        return fn

    return deco


def get_strategy(name: str) -> BlendFn:
    """Look up a registered strategy by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown blend strategy '{name}'. "
            f"Available: {sorted(_REGISTRY)}"
        ) from None


def available_strategies() -> List[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(_REGISTRY)


def _blend_with_weight(
    original: np.ndarray, delighted: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """``weight * delighted + (1 - weight) * original`` for (H, W) weight."""
    w = np.stack([weight] * 3, axis=-1)
    blended = w * delighted + (1.0 - w) * original
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def _require_scoremap(score_map: Optional[np.ndarray], name: str) -> np.ndarray:
    if score_map is None:
        raise ValueError(f"{name} blending requires a specular score map")
    return score_map


def _smoothstep(
    low: float, high: float, values: Union[np.ndarray, float]
) -> np.ndarray:
    """Cubic transition from zero at ``low`` to one at ``high``."""
    if high <= low:
        raise ValueError("smoothstep high must be greater than low")
    x = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0, 1)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32, copy=False)


def _resize_map(values: np.ndarray, shape: Tuple[int, int], interpolation: int) -> np.ndarray:
    """Resize a float map to ``(height, width)`` without changing its range."""
    height, width = shape
    if values.shape == (height, width):
        return values.astype(np.float32, copy=True)
    return cv2.resize(
        values.astype(np.float32), (width, height), interpolation=interpolation
    ).astype(np.float32)


def _raster_effective_width(mask: np.ndarray) -> float:
    """Return a pixel-support-aware ``4A/P`` width for a binary component.

    OpenCV contours run through pixel centres, so the uncorrected ``4A/P``
    reports a 2x2 square as one pixel wide and degenerates for one-pixel
    structures. Adding one occupied pixel recovers the intended support:
    compact 2x2 and 10x10 squares measure 2 and 10 pixels respectively, while
    a one-pixel line remains one pixel wide.
    """
    component = np.asarray(mask, dtype=np.uint8)
    if component.ndim != 2 or not np.any(component):
        return 1.0
    contours = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    if not contours:
        return 1.0
    contour_area = float(
        sum(cv2.contourArea(contour) for contour in contours)
    )
    perimeter = float(
        sum(cv2.arcLength(contour, True) for contour in contours)
    )
    if perimeter <= 1e-6:
        return 1.0
    return max(4.0 * contour_area / perimeter + 1.0, 1.0)


def _native_reference_component_width(
    local_geometry_train: np.ndarray,
    *,
    train_bbox: Tuple[int, int, int, int],
    train_shape: Tuple[int, int],
    score: np.ndarray,
    independent_evidence: np.ndarray,
    matte_score_low: float,
    matte_score_high: float,
    matte_evidence_low: float,
    matte_evidence_high: float,
    reference_short_side: float,
) -> float:
    """Measure one matte component in native, reference-normalised pixels.

    The solved training-grid component is projected to native resolution,
    where native score/evidence refines its uncertain block boundary. This
    returns only a component-width prior and cannot change matte topology.
    """
    x, y, box_width, box_height = train_bbox
    height, width = score.shape
    train_height, train_width = train_shape
    x0 = max(int(np.floor(x * width / max(train_width, 1))), 0)
    y0 = max(int(np.floor(y * height / max(train_height, 1))), 0)
    x1 = min(
        int(np.ceil((x + box_width) * width / max(train_width, 1))),
        width,
    )
    y1 = min(
        int(np.ceil((y + box_height) * height / max(train_height, 1))),
        height,
    )
    native_width = max(x1 - x0, 1)
    native_height = max(y1 - y0, 1)
    projected = cv2.resize(
        local_geometry_train.astype(np.uint8),
        (native_width, native_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    score_support = (
        _smoothstep(
            matte_score_low,
            matte_score_high,
            score[y0:y1, x0:x1],
        )
        >= 0.5
    )
    evidence_support = (
        _smoothstep(
            matte_evidence_low,
            matte_evidence_high,
            independent_evidence[y0:y1, x0:x1],
        )
        >= 0.5
    )
    native_support = score_support | evidence_support

    scale_x = native_width / max(box_width, 1)
    scale_y = native_height / max(box_height, 1)
    boundary_radius = max(int(np.ceil(max(scale_x, scale_y))), 1)
    interior = cv2.erode(
        projected.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * boundary_radius + 1, 2 * boundary_radius + 1),
        ),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    native_geometry = interior | (projected & native_support)
    if not np.any(native_geometry):
        native_geometry = projected

    width_pixels = _raster_effective_width(native_geometry)
    normalization = float(reference_short_side) / max(min(height, width), 1)
    return width_pixels * normalization


def _guided_filter(
    guide: np.ndarray,
    values: np.ndarray,
    *,
    radius: int,
    epsilon: float,
) -> np.ndarray:
    """Smooth ``values`` while respecting coarse edges in ``guide``."""
    radius = max(int(radius), 1)
    kernel = (2 * radius + 1, 2 * radius + 1)
    guide = guide.astype(np.float32)
    values = values.astype(np.float32)
    mean_guide = cv2.boxFilter(guide, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    mean_values = cv2.boxFilter(values, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    corr_guide = cv2.boxFilter(
        guide * guide, -1, kernel, borderType=cv2.BORDER_REPLICATE
    )
    corr_cross = cv2.boxFilter(
        guide * values, -1, kernel, borderType=cv2.BORDER_REPLICATE
    )
    variance = np.maximum(corr_guide - mean_guide * mean_guide, 0.0)
    covariance = corr_cross - mean_guide * mean_values
    a = covariance / (variance + max(float(epsilon), 1e-6))
    b = mean_values - a * mean_guide
    mean_a = cv2.boxFilter(a, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    mean_b = cv2.boxFilter(b, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return (mean_a * guide + mean_b).astype(np.float32)


def _zero_floor_sigmoid(
    score: np.ndarray, *, threshold: float, steepness: float
) -> np.ndarray:
    """Sigmoid score mapping whose value is exactly zero when score is zero."""
    threshold = float(threshold)
    steepness = max(float(steepness), 1e-3)
    mapped = 1.0 / (1.0 + np.exp(-steepness * (score - threshold)))
    floor = 1.0 / (1.0 + np.exp(steepness * threshold))
    return np.clip((mapped - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0).astype(
        np.float32
    )


def _component_regularize(
    score: np.ndarray,
    alpha: np.ndarray,
    view_evidence: np.ndarray,
    *,
    threshold: float,
    small_width: float,
    large_width: float,
    small_gain: float,
    large_gain: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply scale-dependent, component-consistent alpha regularisation.

    All inputs are already at the final training scale.  The effective width
    uses ``4 * contour_area / perimeter``: it behaves like a diameter for a
    compact blob and like the thickness for a long seam, avoiding the common
    error of classifying a long thin edge as a large reflection.
    """
    core = (score >= float(threshold)).astype(np.uint8)
    join_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    joined = cv2.morphologyEx(core, cv2.MORPH_CLOSE, join_kernel)
    joined = cv2.dilate(joined, join_kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        joined, connectivity=8
    )

    regularized = alpha.astype(np.float32, copy=True)
    size_scale = np.zeros_like(alpha, dtype=np.float32)
    size_gain = np.zeros_like(alpha, dtype=np.float32)
    pixel_override = _smoothstep(0.08, 0.35, view_evidence)
    view_override = pixel_override.copy()

    for component_id in range(1, count):
        component = labels == component_id
        area_pixels = int(stats[component_id, cv2.CC_STAT_AREA])
        if area_pixels < 2:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])

        local = component[y : y + height, x : x + width].astype(np.uint8)
        contour_result = cv2.findContours(
            local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contour_result[-2]
        contour_area = float(sum(cv2.contourArea(contour) for contour in contours))
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        effective_width = (
            4.0 * contour_area / perimeter if perimeter > 1e-6 else 1.0
        )
        scale = float(_smoothstep(small_width, large_width, effective_width))

        component_view = float(np.percentile(view_evidence[component], 85))
        override = float(_smoothstep(0.08, 0.35, component_view))
        gain = small_gain + (large_gain - small_gain) * scale
        gain += (large_gain - gain) * override

        # Compact small components such as logos and screw heads are treated
        # as one region.  Long thin seams keep their own narrow support.
        region = component.copy()
        aspect = max(width, height) / max(min(width, height), 1)
        if scale < 0.5 and aspect <= 3.0 and contours:
            filled_local = np.zeros_like(local)
            cv2.drawContours(filled_local, contours, -1, 1, thickness=cv2.FILLED)
            region[y : y + height, x : x + width] |= filled_local.astype(bool)

        component_alpha = float(np.percentile(alpha[component], 70))
        candidate = gain * (
            (1.0 - scale) * component_alpha + scale * alpha[region]
        )
        regularized[region] = np.clip(candidate, 0.0, 1.0)
        size_scale[region] = scale
        size_gain[region] = gain
        view_override[region] = np.maximum(view_override[region], override)

    return regularized, size_scale, size_gain, view_override


def compute_adaptive_feather_weights(
    original: np.ndarray,
    score_map: np.ndarray,
    *,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    steepness: float = 8.0,
    data_factor: int = 4,
    small_width: float = ADAPTIVE_FEATHER_SMALL_WIDTH,
    large_width: float = ADAPTIVE_FEATHER_LARGE_WIDTH,
    small_gain: float = ADAPTIVE_FEATHER_SMALL_GAIN,
    large_gain: float = ADAPTIVE_FEATHER_LARGE_GAIN,
    stable_detail_delight: float = 0.12,
    stable_chroma_delight: float = 0.55,
) -> Dict[str, np.ndarray]:
    """Build scale-aware base/detail/chroma weights for V3 blending.

    Component size is measured after ``data_factor`` downsampling, matching
    what gsplat will actually see.  Small stable components are attenuated
    and made internally coherent; view-dependent observation evidence can
    restore full delighting even for a tiny highlight.
    """
    score = np.clip(score_map.astype(np.float32), 0.0, 1.0)
    height, width = score.shape
    if original.shape[:2] != score.shape:
        raise ValueError("original and score_map dimensions must match")

    cross = (
        np.zeros_like(score)
        if cross_view_map is None
        else np.clip(cross_view_map.astype(np.float32), 0.0, 1.0)
    )
    confidence = (
        np.zeros_like(score)
        if cross_view_confidence is None
        else np.clip(cross_view_confidence.astype(np.float32), 0.0, 1.0)
    )
    if cross.shape != score.shape or confidence.shape != score.shape:
        raise ValueError("cross-view maps must match score_map dimensions")

    factor = max(int(data_factor), 1)
    train_shape = (max(height // factor, 1), max(width // factor, 1))
    score_train = _resize_map(score, train_shape, cv2.INTER_AREA)
    cross_train = _resize_map(cross, train_shape, cv2.INTER_AREA)
    confidence_train = _resize_map(confidence, train_shape, cv2.INTER_AREA)
    view_train = cross_train * (0.25 + 0.75 * confidence_train)
    alpha_train = _zero_floor_sigmoid(
        score_train, threshold=threshold, steepness=steepness
    )
    alpha_train, size_scale, size_gain, view_override = _component_regularize(
        score_train,
        alpha_train,
        view_train,
        threshold=threshold,
        small_width=small_width,
        large_width=large_width,
        small_gain=small_gain,
        large_gain=large_gain,
    )

    # One training-scale pixel of connection plus a short feather suppresses
    # checkerboard colour switching without erasing the whole small object.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_train = cv2.dilate(alpha_train, kernel, iterations=1)
    alpha_train = cv2.GaussianBlur(
        alpha_train, (0, 0), sigmaX=1.25, sigmaY=1.25,
        borderType=cv2.BORDER_REPLICATE,
    )

    base_weight = _resize_map(alpha_train, (height, width), cv2.INTER_LINEAR)
    view_override = _resize_map(view_override, (height, width), cv2.INTER_LINEAR)
    size_scale = _resize_map(size_scale, (height, width), cv2.INTER_NEAREST)
    size_gain = _resize_map(size_gain, (height, width), cv2.INTER_NEAREST)

    image_scale = min(height, width) / 3072.0
    guide_sigma = max(3.0 * image_scale, 0.75)
    guide_radius = max(int(round(14.0 * image_scale)), 3)
    luminance = (
        0.299 * original[..., 0]
        + 0.587 * original[..., 1]
        + 0.114 * original[..., 2]
    ).astype(np.float32)
    coarse_guide = cv2.GaussianBlur(
        luminance, (0, 0), sigmaX=guide_sigma, sigmaY=guide_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    base_weight = np.clip(
        _guided_filter(
            coarse_guide, base_weight, radius=guide_radius, epsilon=0.006
        ),
        0.0,
        1.0,
    ).astype(np.float32)

    view_override = np.clip(
        cv2.GaussianBlur(
            view_override, (0, 0), sigmaX=guide_sigma, sigmaY=guide_sigma,
            borderType=cv2.BORDER_REPLICATE,
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    detail_fraction = np.clip(
        stable_detail_delight
        + (1.0 - stable_detail_delight) * view_override,
        0.0,
        1.0,
    )
    chroma_fraction = np.clip(
        stable_chroma_delight
        + (1.0 - stable_chroma_delight) * view_override,
        0.0,
        1.0,
    )
    return {
        "base_weight": base_weight,
        "detail_weight": (base_weight * detail_fraction).astype(np.float32),
        "chroma_weight": (base_weight * chroma_fraction).astype(np.float32),
        "size_scale": np.clip(size_scale, 0.0, 1.0).astype(np.float32),
        "size_gain": np.clip(size_gain / max(large_gain, 1e-6), 0.0, 1.0).astype(
            np.float32
        ),
        "view_override": view_override,
    }


def compute_weak_hysteresis_feather_weights(
    original: np.ndarray,
    score_map: np.ndarray,
    *,
    delighted: Optional[np.ndarray] = None,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    detector_cues: Optional[Dict[str, np.ndarray]] = None,
    threshold: float = 0.3,
    steepness: float = 8.0,
    data_factor: int = 4,
    weak_low: float = 0.10,
    weak_mapping_threshold: float = 0.16,
    weak_boost_gain: float = 0.35,
    evidence_seed: float = 0.18,
    small_width: float = 6.0,
    large_width: float = 12.0,
    small_gain: float = 0.35,
    large_gain: float = 1.15,
    small_feather: float = 0.5,
    large_feather: float = 2.5,
    color_feather_gain: float = 1.0,
    boundary_bias: float = 1.0,
    transition_guard_strength: float = 0.35,
    base_edge_lock: float = 0.2,
    outer_feather_mode: float = 0.0,
    outer_low: float = 0.05,
    outer_evidence: float = 0.10,
    small_outer_radius: float = 0.75,
    large_outer_radius: float = 3.0,
    inner_feather_radius: float = 0.5,
    max_profile_shift: float = 3.0,
    stable_detail_delight: float = 0.12,
    stable_chroma_delight: float = 0.55,
) -> Dict[str, np.ndarray]:
    """Build V5 weights with evidence-gated weak-highlight recovery.

    ``threshold`` remains the strong seed threshold.  Pixels above
    ``weak_low`` are promoted only when their connected component contains a
    strong seed or reliable cross-view/delight-removal evidence.  Per-region
    targets are extended with normalized Gaussian convolution, so both gain
    and blend weight change continuously instead of jumping at the binary
    component boundary.
    """
    if not 0.0 <= weak_low < threshold:
        raise ValueError("weak_low must be in [0, threshold)")
    if not weak_low <= weak_mapping_threshold < threshold:
        raise ValueError(
            "weak_mapping_threshold must be in [weak_low, threshold)"
        )
    if large_width <= small_width:
        raise ValueError("large_width must be greater than small_width")
    if small_feather <= 0.0 or large_feather < small_feather:
        raise ValueError(
            "feather scales must satisfy 0 < small_feather <= large_feather"
        )
    if outer_feather_mode >= 0.5:
        if not 0.0 <= outer_low < weak_low:
            raise ValueError("outer_low must be in [0, weak_low)")
        if not 0.0 <= outer_evidence <= 1.0:
            raise ValueError("outer_evidence must be in [0, 1]")
        if small_outer_radius <= 0.0 or large_outer_radius < small_outer_radius:
            raise ValueError(
                "outer radii must satisfy 0 < small_outer_radius <= large_outer_radius"
            )
        if inner_feather_radius <= 0.0:
            raise ValueError("inner_feather_radius must be positive")
        if max_profile_shift < 0.0:
            raise ValueError("max_profile_shift must be non-negative")

    score = np.clip(score_map.astype(np.float32), 0.0, 1.0)
    height, width = score.shape
    if original.shape[:2] != score.shape:
        raise ValueError("original and score_map dimensions must match")
    if delighted is not None and delighted.shape != original.shape:
        raise ValueError("delighted and original dimensions must match")

    cross = (
        np.zeros_like(score)
        if cross_view_map is None
        else np.clip(cross_view_map.astype(np.float32), 0.0, 1.0)
    )
    confidence = (
        np.zeros_like(score)
        if cross_view_confidence is None
        else np.clip(cross_view_confidence.astype(np.float32), 0.0, 1.0)
    )
    if cross.shape != score.shape or confidence.shape != score.shape:
        raise ValueError("cross-view maps must match score_map dimensions")

    cues = detector_cues or {}

    def cue(name: str) -> np.ndarray:
        values = cues.get(name)
        if values is None:
            return np.zeros_like(score)
        values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        if values.shape != score.shape:
            raise ValueError(f"detector cue '{name}' must match score_map")
        return values

    view_evidence = cross * (0.25 + 0.75 * confidence)
    independent_evidence = np.maximum.reduce(
        [view_evidence, cue("cross_support"), cue("delight_support")]
    ).astype(np.float32)
    structure = cue("structure_penalty")

    factor = max(int(data_factor), 1)
    train_shape = (max(height // factor, 1), max(width // factor, 1))
    score_train = _resize_map(score, train_shape, cv2.INTER_AREA)
    view_train = _resize_map(view_evidence, train_shape, cv2.INTER_AREA)
    evidence_train = _resize_map(
        independent_evidence, train_shape, cv2.INTER_AREA
    )
    structure_train = _resize_map(structure, train_shape, cv2.INTER_AREA)
    if delighted is None:
        change_train = np.zeros(train_shape, dtype=np.float32)
    else:
        source_change = np.mean(
            np.abs(original.astype(np.float32) - delighted.astype(np.float32)),
            axis=2,
        )
        change_train = _resize_map(source_change, train_shape, cv2.INTER_AREA)

    base_alpha = _zero_floor_sigmoid(
        score_train, threshold=threshold, steepness=steepness
    )
    weak_alpha = _zero_floor_sigmoid(
        score_train,
        threshold=weak_mapping_threshold,
        steepness=steepness,
    )
    weak_mix = 1.0 - _smoothstep(weak_low, threshold, score_train)
    boosted_alpha = base_alpha + weak_mix * np.maximum(
        weak_alpha - base_alpha, 0.0
    )
    boosted_alpha = np.clip(
        boosted_alpha
        * (1.0 + max(float(weak_boost_gain), 0.0) * weak_mix),
        0.0,
        1.0,
    )

    candidate = score_train >= float(weak_low)
    supported_seed = (
        (score_train >= max(0.75 * weak_low, 0.01))
        & (evidence_train >= float(evidence_seed))
    )
    strong_seed = score_train >= float(threshold)
    candidate |= supported_seed
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    seed = strong_seed | supported_seed
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )

    retained_ids: List[int] = []
    retained = np.zeros_like(candidate, dtype=bool)
    for component_id in range(1, count):
        component = labels == component_id
        if np.any(seed[component]):
            retained_ids.append(component_id)
            retained[component] = True

    regularized = base_alpha.astype(np.float32, copy=True)
    feather_support = np.zeros_like(base_alpha, dtype=np.float32)
    size_scale = np.zeros_like(base_alpha, dtype=np.float32)
    size_gain = np.zeros_like(base_alpha, dtype=np.float32)
    # Independent delight residuals may seed/strengthen the broad luminance
    # correction, but must not automatically replace fine texture or chroma.
    # Only observation-specific cross-view evidence relaxes those conservative
    # perceptual weights.
    source_override = _smoothstep(0.08, 0.35, view_train)
    transition_risk = np.zeros_like(base_alpha, dtype=np.float32)
    outer_support = np.zeros_like(base_alpha, dtype=np.float32)

    for component_id in retained_ids:
        component = labels == component_id
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        local_component = component[
            y : y + box_height, x : x + box_width
        ].astype(np.uint8)
        contours = cv2.findContours(
            local_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        contour_area = float(
            sum(cv2.contourArea(contour) for contour in contours)
        )
        perimeter = float(
            sum(cv2.arcLength(contour, True) for contour in contours)
        )
        effective_width = (
            4.0 * contour_area / perimeter if perimeter > 1e-6 else 1.0
        )
        scale = float(_smoothstep(small_width, large_width, effective_width))
        component_evidence = float(np.percentile(evidence_train[component], 85))
        override = float(_smoothstep(0.08, 0.35, component_evidence))
        component_view = float(np.percentile(view_train[component], 85))
        perceptual_override = float(_smoothstep(0.08, 0.35, component_view))
        gain = small_gain + (large_gain - small_gain) * scale
        gain += (large_gain - gain) * override

        component_alpha = float(np.percentile(boosted_alpha[component], 70))
        target = np.zeros_like(base_alpha, dtype=np.float32)
        target[component] = np.clip(
            gain
            * (
                (1.0 - scale) * component_alpha
                + scale * boosted_alpha[component]
            ),
            0.0,
            1.0,
        )

        component_change = float(np.percentile(change_train[component], 85))
        color_risk = float(_smoothstep(0.03, 0.15, component_change))
        sigma = small_feather + (large_feather - small_feather) * scale
        sigma *= 1.0 + max(float(color_feather_gain), 0.0) * color_risk
        outer_radius = small_outer_radius + (
            large_outer_radius - small_outer_radius
        ) * scale
        outer_radius *= 1.0 + max(float(color_feather_gain), 0.0) * color_risk
        margin_extent = (
            outer_radius + 2.0
            if outer_feather_mode >= 0.5
            else 4.0 * sigma
        )
        margin = max(int(np.ceil(margin_extent)) + 1, 2)
        x0 = max(x - margin, 0)
        y0 = max(y - margin, 0)
        x1 = min(x + box_width + margin, train_shape[1])
        y1 = min(y + box_height + margin, train_shape[0])
        roi = np.s_[y0:y1, x0:x1]
        mask_local = component[roi].astype(np.float32)
        target_local = target[roi]
        membership = cv2.GaussianBlur(
            mask_local,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        numerator = cv2.GaussianBlur(
            target_local,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        membership = np.clip(membership, 0.0, 1.0)
        membership_power = (
            1.0
            + max(float(boundary_bias), 0.0) * color_risk * scale
        )
        inward_membership = np.power(membership, membership_power)
        inward_target = numerator * np.power(
            membership, max(membership_power - 1.0, 0.0)
        )
        # This is the preserved V5 starting profile.  V5.1 may shift only its
        # narrow boundary band below; the deeper component interior remains
        # unchanged.
        regularized[roi] = np.clip(
            (1.0 - mask_local) * regularized[roi] + inward_target,
            0.0,
            1.0,
        )
        feather_membership = inward_membership
        if outer_feather_mode >= 0.5:
            component_local = mask_local >= 0.5
            allowed_local = (
                (score_train[roi] >= float(outer_low))
                | (evidence_train[roi] >= float(outer_evidence))
            )
            allowed_local &= (
                (structure_train[roi] <= 0.65)
                | (evidence_train[roi] >= float(evidence_seed))
            )
            allowed_local |= component_local
            outside_distance = cv2.distanceTransform(
                (~component_local).astype(np.uint8), cv2.DIST_L2, 5
            )
            inside_distance = cv2.distanceTransform(
                component_local.astype(np.uint8), cv2.DIST_L2, 5
            )
            grown = component_local.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for _ in range(max(int(np.ceil(outer_radius)), 1)):
                proposed = cv2.dilate(grown.astype(np.uint8), kernel).astype(bool)
                grown |= (
                    proposed
                    & allowed_local
                    & (outside_distance <= outer_radius + 1e-6)
                )
            signed_distance = inside_distance - outside_distance
            grown_soft = cv2.GaussianBlur(
                grown.astype(np.float32),
                (0, 0),
                sigmaX=0.75,
                sigmaY=0.75,
                borderType=cv2.BORDER_REPLICATE,
            )
            grown_soft = np.clip(grown_soft, 0.0, 1.0)
            shift_steps = int(round(float(max_profile_shift) * scale))
            if shift_steps > 0:
                # Shift the validated V5 weight profile toward the exterior.
                # Dilation takes values from a few pixels deeper inside the
                # reflection; the following blur avoids a staircase profile.
                shifted_profile = regularized[roi].copy()
                for _ in range(shift_steps):
                    shifted_profile = cv2.dilate(shifted_profile, kernel)
                shifted_profile = cv2.GaussianBlur(
                    shifted_profile,
                    (0, 0),
                    sigmaX=0.75,
                    sigmaY=0.75,
                    borderType=cv2.BORDER_REPLICATE,
                )
                profile_membership = _smoothstep(
                    -outer_radius,
                    max(float(shift_steps), inner_feather_radius),
                    signed_distance,
                )
                inside_band = component_local & (
                    inside_distance <= shift_steps + 1.0
                )
                outside_band = grown & ~component_local
                transition_zone = inside_band | outside_band
                constraint = np.where(
                    component_local, 1.0, grown_soft
                ).astype(np.float32)
                shifted_target = (
                    shifted_profile * profile_membership * constraint
                )
                local_regularized = regularized[roi]
                local_regularized[transition_zone] = np.maximum(
                    local_regularized[transition_zone],
                    shifted_target[transition_zone],
                )
                regularized[roi] = np.clip(local_regularized, 0.0, 1.0)
            outer_membership = _smoothstep(
                -outer_radius, inner_feather_radius, signed_distance
            )
            outer_membership *= grown_soft
            outer_membership[component_local] = 0.0
            feather_membership = np.maximum(
                feather_membership, outer_membership
            )
            outer_support[roi] = np.maximum(
                outer_support[roi],
                (grown & ~component_local).astype(np.float32),
            )
        feather_support[roi] = np.maximum(
            feather_support[roi], feather_membership
        )
        size_scale[roi] = (
            (1.0 - feather_membership) * size_scale[roi]
            + feather_membership * scale
        )
        size_gain[roi] = (
            (1.0 - feather_membership) * size_gain[roi]
            + feather_membership * gain
        )
        source_override[roi] = np.maximum(
            source_override[roi], feather_membership * perceptual_override
        )
        transition_risk[roi] = np.maximum(
            transition_risk[roi], feather_membership * color_risk
        )

    raw_weight = _resize_map(regularized, (height, width), cv2.INTER_LINEAR)
    image_scale = min(height, width) / 3072.0
    guide_sigma = max(3.0 * image_scale, 0.75)
    guide_radius = max(int(round(14.0 * image_scale)), 3)
    luminance = (
        0.299 * original[..., 0]
        + 0.587 * original[..., 1]
        + 0.114 * original[..., 2]
    ).astype(np.float32)
    coarse_guide = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=guide_sigma,
        sigmaY=guide_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    guided_weight = np.clip(
        _guided_filter(
            coarse_guide, raw_weight, radius=guide_radius, epsilon=0.006
        ),
        0.0,
        1.0,
    )
    edge_protection = np.clip(
        structure_train * (1.0 - evidence_train), 0.0, 1.0
    )
    edge_protection = _resize_map(
        edge_protection, (height, width), cv2.INTER_LINEAR
    )
    edge_protection = cv2.GaussianBlur(
        edge_protection,
        (0, 0),
        sigmaX=guide_sigma,
        sigmaY=guide_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    guide_mix = np.clip(
        float(base_edge_lock)
        + (1.0 - float(base_edge_lock)) * edge_protection,
        0.0,
        1.0,
    )
    base_weight = np.clip(
        guide_mix * guided_weight + (1.0 - guide_mix) * raw_weight,
        0.0,
        1.0,
    ).astype(np.float32)

    transition_guard = np.zeros_like(base_weight, dtype=np.float32)
    if delighted is not None and transition_guard_strength > 0.0:
        delighted_luminance = (
            0.299 * delighted[..., 0]
            + 0.587 * delighted[..., 1]
            + 0.114 * delighted[..., 2]
        ).astype(np.float32)
        source_delta = cv2.GaussianBlur(
            delighted_luminance - luminance,
            (0, 0),
            sigmaX=guide_sigma,
            sigmaY=guide_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        delta_grad_x = cv2.Sobel(source_delta, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        delta_grad_y = cv2.Sobel(source_delta, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        delta_gradient = cv2.magnitude(delta_grad_x, delta_grad_y)
        transition_guard = _smoothstep(0.005, 0.03, delta_gradient)
        transition_guard = cv2.GaussianBlur(
            transition_guard,
            (0, 0),
            sigmaX=2.0 * guide_sigma,
            sigmaY=2.0 * guide_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        outer_guard_relief = np.clip(
            _resize_map(outer_support, (height, width), cv2.INTER_LINEAR),
            0.0,
            1.0,
        )
        base_weight *= np.clip(
            1.0
            - float(transition_guard_strength)
            * np.clip(transition_guard, 0.0, 1.0),
            0.0,
            1.0,
        )
        if outer_feather_mode >= 0.5:
            # The guard remains identical to V5 inside the retained component,
            # but no longer suppresses the new outer shell and recreate a rim.
            unguarded_shell = np.clip(
                guide_mix * guided_weight + (1.0 - guide_mix) * raw_weight,
                0.0,
                1.0,
            )
            base_weight = (
                (1.0 - outer_guard_relief) * base_weight
                + outer_guard_relief * unguarded_shell
            )
        base_weight = np.clip(
            base_weight,
            0.0,
            1.0,
        )

    source_override = _resize_map(
        source_override, (height, width), cv2.INTER_LINEAR
    )
    source_override = np.clip(
        cv2.GaussianBlur(
            source_override,
            (0, 0),
            sigmaX=guide_sigma,
            sigmaY=guide_sigma,
            borderType=cv2.BORDER_REPLICATE,
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    detail_fraction = np.clip(
        stable_detail_delight
        + (1.0 - stable_detail_delight) * source_override,
        0.0,
        1.0,
    )
    chroma_fraction = np.clip(
        stable_chroma_delight
        + (1.0 - stable_chroma_delight) * source_override,
        0.0,
        1.0,
    )
    return {
        "base_weight": base_weight,
        "detail_weight": (base_weight * detail_fraction).astype(np.float32),
        "chroma_weight": (base_weight * chroma_fraction).astype(np.float32),
        "size_scale": np.clip(
            _resize_map(size_scale, (height, width), cv2.INTER_LINEAR), 0.0, 1.0
        ).astype(np.float32),
        "size_gain": np.clip(
            _resize_map(size_gain, (height, width), cv2.INTER_LINEAR)
            / max(large_gain, 1e-6),
            0.0,
            1.0,
        ).astype(np.float32),
        "view_override": source_override,
        "hysteresis_support": _resize_map(
            retained.astype(np.float32), (height, width), cv2.INTER_NEAREST
        ),
        "feather_support": np.clip(
            _resize_map(feather_support, (height, width), cv2.INTER_LINEAR),
            0.0,
            1.0,
        ).astype(np.float32),
        "edge_protection": np.clip(edge_protection, 0.0, 1.0).astype(np.float32),
        "transition_risk": np.clip(
            _resize_map(transition_risk, (height, width), cv2.INTER_LINEAR),
            0.0,
            1.0,
        ).astype(np.float32),
        "transition_guard": np.clip(transition_guard, 0.0, 1.0).astype(
            np.float32
        ),
        "outer_support": np.clip(
            _resize_map(outer_support, (height, width), cv2.INTER_NEAREST),
            0.0,
            1.0,
        ).astype(np.float32),
    }


def _solve_evidence_matte(
    likelihood: np.ndarray,
    foreground_seed: np.ndarray,
    background_seed: np.ndarray,
    guide: np.ndarray,
    structure: np.ndarray,
    evidence: np.ndarray,
    *,
    smoothness: float,
    iterations: int,
    edge_sigma: float,
    structure_barrier_strength: float,
) -> np.ndarray:
    """Solve a soft reflection matte by edge-aware constrained diffusion.

    Foreground/background constraints come from photometric evidence rather
    than spatial offsets.  Pairwise weights allow propagation through smooth
    reflection falloff and stop it at supported physical structure.
    """
    if likelihood.ndim != 2:
        raise ValueError("matte likelihood must be a 2-D array")
    edge_sigma = max(float(edge_sigma), 1e-4)
    smoothness = max(float(smoothness), 0.0)
    iterations = max(int(iterations), 1)

    guide = cv2.GaussianBlur(
        guide.astype(np.float32, copy=False),
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
        borderType=cv2.BORDER_REPLICATE,
    )
    rescue = _smoothstep(0.08, 0.30, evidence)
    barrier = np.clip(structure * (1.0 - rescue), 0.0, 1.0)
    horizontal_barrier = np.maximum(barrier[:, :-1], barrier[:, 1:])
    vertical_barrier = np.maximum(barrier[:-1, :], barrier[1:, :])
    horizontal_gradient = np.abs(guide[:, :-1] - guide[:, 1:])
    vertical_gradient = np.abs(guide[:-1, :] - guide[1:, :])
    horizontal_weight = np.exp(
        -np.square(horizontal_gradient / edge_sigma)
    ) * (
        1.0
        - float(structure_barrier_strength) * horizontal_barrier
    )
    vertical_weight = np.exp(
        -np.square(vertical_gradient / edge_sigma)
    ) * (
        1.0
        - float(structure_barrier_strength) * vertical_barrier
    )
    horizontal_weight = np.clip(horizontal_weight, 0.01, 1.0).astype(
        np.float32
    )
    vertical_weight = np.clip(vertical_weight, 0.01, 1.0).astype(np.float32)

    foreground_confidence = np.maximum(
        _smoothstep(0.45, 0.95, likelihood),
        foreground_seed.astype(np.float32),
    )
    background_confidence = np.maximum(
        _smoothstep(0.45, 0.95, 1.0 - likelihood),
        background_seed.astype(np.float32),
    )
    data_weight = (
        0.35
        + 3.0 * np.maximum(foreground_confidence, background_confidence)
    ).astype(np.float32)
    matte = np.clip(likelihood, 0.0, 1.0).astype(np.float32, copy=True)
    matte[foreground_seed] = 1.0
    matte[background_seed] = 0.0

    for _ in range(iterations):
        neighbour_sum = np.zeros_like(matte)
        neighbour_weight = np.zeros_like(matte)
        neighbour_sum[:, :-1] += horizontal_weight * matte[:, 1:]
        neighbour_sum[:, 1:] += horizontal_weight * matte[:, :-1]
        neighbour_weight[:, :-1] += horizontal_weight
        neighbour_weight[:, 1:] += horizontal_weight
        neighbour_sum[:-1, :] += vertical_weight * matte[1:, :]
        neighbour_sum[1:, :] += vertical_weight * matte[:-1, :]
        neighbour_weight[:-1, :] += vertical_weight
        neighbour_weight[1:, :] += vertical_weight
        matte = (
            data_weight * likelihood + smoothness * neighbour_sum
        ) / np.maximum(
            data_weight + smoothness * neighbour_weight,
            1e-6,
        )
        matte[foreground_seed] = 1.0
        matte[background_seed] = 0.0
    return np.clip(matte, 0.0, 1.0).astype(np.float32, copy=False)


def compute_evidence_matte_weights(
    original: np.ndarray,
    score_map: np.ndarray,
    *,
    delighted: Optional[np.ndarray] = None,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    detector_cues: Optional[Dict[str, np.ndarray]] = None,
    threshold: float = 0.3,
    steepness: float = 8.0,
    data_factor: int = 4,
    matte_score_low: float = 0.015,
    matte_score_high: float = 0.12,
    matte_evidence_low: float = 0.03,
    matte_evidence_high: float = 0.18,
    foreground_score: float = 0.30,
    supported_score: float = 0.075,
    foreground_evidence: float = 0.18,
    background_score: float = 0.012,
    background_evidence: float = 0.025,
    structure_barrier_strength: float = 0.85,
    matte_smoothness: float = 1.5,
    matte_iterations: float = 24.0,
    matte_edge_sigma: float = 0.08,
    matte_component_floor: float = 0.02,
    weak_mapping_threshold: float = 0.16,
    weak_boost_gain: float = 0.35,
    small_width: float = 2.0,
    large_width: float = 10.0,
    small_gain: float = 0.45,
    large_gain: float = 1.10,
    width_reference_short_side: float = 3072.0,
    base_edge_lock: float = 0.2,
    stable_detail_delight: float = 0.12,
    stable_chroma_delight: float = 0.55,
    include_diagnostics: bool = True,
) -> Dict[str, np.ndarray]:
    """Build V6 weights from an inferred continuous reflection matte.

    The matte boundary is the solution of a foreground/background constrained
    evidence field.  Component scale affects only the removal strength after
    the matte has been solved; it never moves or dilates the matte boundary.

    Current V6 defaults measure native-resolution evidence and express
    ``small_width``/``large_width`` in pixels normalised to a 3072-pixel image
    short side. Set ``width_reference_short_side=0`` only to reproduce the
    initial V6 training-grid width semantics. Set ``include_diagnostics=False``
    to return only the three weights required for final image synthesis.
    """
    del delighted  # Reserved for future source-transition diagnostics.
    if not 0.0 <= matte_score_low < matte_score_high <= 1.0:
        raise ValueError("matte score thresholds must be ordered in [0, 1]")
    if not 0.0 <= matte_evidence_low < matte_evidence_high <= 1.0:
        raise ValueError("matte evidence thresholds must be ordered in [0, 1]")
    if not 0.0 <= background_score < supported_score <= foreground_score <= 1.0:
        raise ValueError("foreground/background score thresholds are not ordered")
    if not 0.0 <= background_evidence < foreground_evidence <= 1.0:
        raise ValueError("foreground/background evidence thresholds are not ordered")
    if not 0.0 <= matte_component_floor < 0.5:
        raise ValueError("matte_component_floor must be in [0, 0.5)")
    if large_width <= small_width:
        raise ValueError("large_width must be greater than small_width")
    if width_reference_short_side < 0:
        raise ValueError("width_reference_short_side must be non-negative")

    score = np.clip(np.asarray(score_map, dtype=np.float32), 0.0, 1.0)
    height, width = score.shape
    if original.shape[:2] != score.shape:
        raise ValueError("original and score_map dimensions must match")
    cross = (
        np.zeros_like(score)
        if cross_view_map is None
        else np.clip(np.asarray(cross_view_map, dtype=np.float32), 0.0, 1.0)
    )
    confidence = (
        np.zeros_like(score)
        if cross_view_confidence is None
        else np.clip(
            np.asarray(cross_view_confidence, dtype=np.float32), 0.0, 1.0
        )
    )
    if cross.shape != score.shape or confidence.shape != score.shape:
        raise ValueError("cross-view maps must match score_map dimensions")

    cues = detector_cues or {}

    def cue(name: str) -> np.ndarray:
        values = cues.get(name)
        if values is None:
            return np.zeros_like(score)
        values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        if values.shape != score.shape:
            raise ValueError(f"detector cue '{name}' must match score_map")
        return values

    view_evidence = cross * (0.25 + 0.75 * confidence)
    independent_evidence = np.maximum.reduce(
        [view_evidence, cue("cross_support"), cue("delight_support")]
    ).astype(np.float32, copy=False)
    structure = cue("structure_penalty")

    factor = max(int(data_factor), 1)
    train_shape = (max(height // factor, 1), max(width // factor, 1))
    score_train = _resize_map(score, train_shape, cv2.INTER_AREA)
    evidence_train = _resize_map(
        independent_evidence, train_shape, cv2.INTER_AREA
    )
    view_train = _resize_map(view_evidence, train_shape, cv2.INTER_AREA)
    structure_train = _resize_map(structure, train_shape, cv2.INTER_AREA)
    luminance = (
        0.299 * original[..., 0]
        + 0.587 * original[..., 1]
        + 0.114 * original[..., 2]
    ).astype(np.float32, copy=False)
    guide_train = _resize_map(luminance, train_shape, cv2.INTER_AREA)

    # Seed and unary decisions use a locally robust evidence field.  This
    # removes single training-pixel texture spikes without translating the
    # boundary: the matte still follows the local evidence falloff.
    score_context = cv2.GaussianBlur(
        score_train,
        (0, 0),
        sigmaX=0.8,
        sigmaY=0.8,
        borderType=cv2.BORDER_REPLICATE,
    )
    evidence_context = cv2.GaussianBlur(
        evidence_train,
        (0, 0),
        sigmaX=0.8,
        sigmaY=0.8,
        borderType=cv2.BORDER_REPLICATE,
    )
    structure_context = cv2.GaussianBlur(
        structure_train,
        (0, 0),
        sigmaX=0.6,
        sigmaY=0.6,
        borderType=cv2.BORDER_REPLICATE,
    )
    score_likelihood = _smoothstep(
        matte_score_low, matte_score_high, score_context
    )
    evidence_likelihood = _smoothstep(
        matte_evidence_low, matte_evidence_high, evidence_context
    )
    structure_rescue = _smoothstep(0.08, 0.30, evidence_context)
    structure_gate = np.clip(
        1.0
        - float(structure_barrier_strength)
        * structure_context
        * (1.0 - structure_rescue),
        0.0,
        1.0,
    )
    likelihood = np.maximum(
        score_likelihood, 0.90 * evidence_likelihood
    ) * structure_gate
    seed_candidate = (
        (score_train >= float(foreground_score))
        & (score_context >= float(supported_score))
    ) | (
        (score_context >= float(supported_score))
        & (evidence_context >= float(foreground_evidence))
    )
    coherent_seed = cv2.morphologyEx(
        seed_candidate.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    independently_supported_seed = (
        (score_train >= float(foreground_score))
        | (
            (score_train >= float(supported_score))
            & (evidence_train >= max(float(foreground_evidence), 0.30))
        )
    ) & (evidence_context >= float(foreground_evidence))
    foreground_seed = coherent_seed | independently_supported_seed
    background_seed = (
        (score_context <= float(background_score))
        & (evidence_context <= float(background_evidence))
    )
    background_seed |= (
        (structure_context >= 0.90)
        & (evidence_context < float(foreground_evidence))
        & ~foreground_seed
    )
    likelihood = np.clip(likelihood, 0.0, 1.0).astype(
        np.float32, copy=False
    )
    likelihood[foreground_seed] = 1.0
    likelihood[background_seed] = 0.0
    matte = _solve_evidence_matte(
        likelihood,
        foreground_seed,
        background_seed,
        guide_train,
        structure_context,
        evidence_context,
        smoothness=matte_smoothness,
        iterations=int(round(matte_iterations)),
        edge_sigma=matte_edge_sigma,
        structure_barrier_strength=structure_barrier_strength,
    )

    # Remove isolated ambiguous fields that are not connected to any reliable
    # reflection seed.  This is evidence connectivity, not spatial expansion.
    candidate = matte >= float(matte_component_floor)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    retained = np.zeros_like(candidate, dtype=bool)
    retained_ids: List[int] = []
    for component_id in range(1, count):
        component = labels == component_id
        if np.any(foreground_seed[component]):
            retained[component] = True
            retained_ids.append(component_id)
    matte *= retained.astype(np.float32)

    base_alpha = _zero_floor_sigmoid(
        score_train, threshold=threshold, steepness=steepness
    )
    weak_alpha = _zero_floor_sigmoid(
        score_train,
        threshold=weak_mapping_threshold,
        steepness=steepness,
    )
    weak_mix = 1.0 - _smoothstep(
        min(matte_score_high, weak_mapping_threshold),
        threshold,
        score_train,
    )
    boosted_alpha = np.clip(
        base_alpha
        + weak_mix * np.maximum(weak_alpha - base_alpha, 0.0),
        0.0,
        1.0,
    )
    boosted_alpha = np.clip(
        boosted_alpha
        * (1.0 + max(float(weak_boost_gain), 0.0) * weak_mix),
        0.0,
        1.0,
    )

    regularized = np.zeros_like(matte, dtype=np.float32)
    size_scale = (
        np.zeros_like(matte, dtype=np.float32)
        if include_diagnostics
        else None
    )
    size_gain = (
        np.zeros_like(matte, dtype=np.float32)
        if include_diagnostics
        else None
    )
    source_override = _smoothstep(0.08, 0.35, view_train)
    for component_id in retained_ids:
        component = labels == component_id
        confident = component & (matte >= 0.5)
        geometry = confident if int(confident.sum()) >= 2 else component
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        local_geometry = geometry[
            y : y + box_height, x : x + box_width
        ].astype(np.uint8)
        if width_reference_short_side > 0:
            effective_width = _native_reference_component_width(
                local_geometry,
                train_bbox=(x, y, box_width, box_height),
                train_shape=train_shape,
                score=score,
                independent_evidence=independent_evidence,
                matte_score_low=matte_score_low,
                matte_score_high=matte_score_high,
                matte_evidence_low=matte_evidence_low,
                matte_evidence_high=matte_evidence_high,
                reference_short_side=width_reference_short_side,
            )
        else:
            contours = cv2.findContours(
                local_geometry, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2]
            contour_area = float(
                sum(cv2.contourArea(contour) for contour in contours)
            )
            perimeter = float(
                sum(cv2.arcLength(contour, True) for contour in contours)
            )
            effective_width = (
                4.0 * contour_area / perimeter if perimeter > 1e-6 else 1.0
            )
        scale = float(_smoothstep(small_width, large_width, effective_width))
        component_evidence = float(
            np.percentile(evidence_train[component], 85)
        )
        evidence_override = float(
            _smoothstep(0.08, 0.35, component_evidence)
        )
        gain = small_gain + (large_gain - small_gain) * scale
        gain += (large_gain - gain) * evidence_override
        core = confident if int(confident.sum()) >= 2 else component
        component_alpha = float(np.percentile(boosted_alpha[core], 70))
        local_strength = gain * (
            (1.0 - scale) * component_alpha
            + scale
            * (
                0.65 * component_alpha
                + 0.35 * boosted_alpha[component]
            )
        )
        regularized[component] = np.clip(
            matte[component] * local_strength, 0.0, 1.0
        )
        if include_diagnostics:
            size_scale[component] = scale * matte[component]
            size_gain[component] = gain * matte[component]
        component_view = float(np.percentile(view_train[component], 85))
        perceptual_override = float(
            _smoothstep(0.08, 0.35, component_view)
        )
        source_override[component] = np.maximum(
            source_override[component],
            matte[component] * perceptual_override,
        )

    raw_weight = _resize_map(regularized, (height, width), cv2.INTER_LINEAR)
    image_scale = min(height, width) / 3072.0
    guide_sigma = max(3.0 * image_scale, 0.75)
    guide_radius = max(int(round(14.0 * image_scale)), 3)
    coarse_guide = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=guide_sigma,
        sigmaY=guide_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    guided_weight = np.clip(
        _guided_filter(
            coarse_guide, raw_weight, radius=guide_radius, epsilon=0.006
        ),
        0.0,
        1.0,
    )
    edge_protection = np.clip(
        structure_train * (1.0 - evidence_train), 0.0, 1.0
    )
    edge_protection = _resize_map(
        edge_protection, (height, width), cv2.INTER_LINEAR
    )
    edge_protection = cv2.GaussianBlur(
        edge_protection,
        (0, 0),
        sigmaX=guide_sigma,
        sigmaY=guide_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    guide_mix = np.clip(
        float(base_edge_lock)
        + (1.0 - float(base_edge_lock)) * edge_protection,
        0.0,
        1.0,
    )
    base_weight = np.clip(
        guide_mix * guided_weight + (1.0 - guide_mix) * raw_weight,
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    source_override = _resize_map(
        source_override, (height, width), cv2.INTER_LINEAR
    )
    source_override = np.clip(
        cv2.GaussianBlur(
            source_override,
            (0, 0),
            sigmaX=guide_sigma,
            sigmaY=guide_sigma,
            borderType=cv2.BORDER_REPLICATE,
        ),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    detail_fraction = np.clip(
        stable_detail_delight
        + (1.0 - stable_detail_delight) * source_override,
        0.0,
        1.0,
    )
    chroma_fraction = np.clip(
        stable_chroma_delight
        + (1.0 - stable_chroma_delight) * source_override,
        0.0,
        1.0,
    )
    weights = {
        "base_weight": base_weight,
        "detail_weight": (base_weight * detail_fraction).astype(
            np.float32, copy=False
        ),
        "chroma_weight": (base_weight * chroma_fraction).astype(
            np.float32, copy=False
        ),
    }
    if not include_diagnostics:
        return weights

    matte_full = np.clip(
        _resize_map(matte, (height, width), cv2.INTER_LINEAR), 0.0, 1.0
    ).astype(np.float32, copy=False)
    likelihood_full = np.clip(
        _resize_map(likelihood, (height, width), cv2.INTER_LINEAR), 0.0, 1.0
    ).astype(np.float32, copy=False)
    weights.update({
        "size_scale": np.clip(
            _resize_map(size_scale, (height, width), cv2.INTER_LINEAR),
            0.0,
            1.0,
        ).astype(np.float32, copy=False),
        "size_gain": np.clip(
            _resize_map(size_gain, (height, width), cv2.INTER_LINEAR)
            / max(large_gain, 1e-6),
            0.0,
            1.0,
        ).astype(np.float32, copy=False),
        "view_override": source_override,
        "edge_protection": np.clip(edge_protection, 0.0, 1.0).astype(
            np.float32, copy=False
        ),
        "reflection_matte": matte_full,
        "matte_likelihood": likelihood_full,
        "matte_uncertainty": (4.0 * matte_full * (1.0 - matte_full)).astype(
            np.float32, copy=False
        ),
        "foreground_seed": _resize_map(
            foreground_seed.astype(np.float32),
            (height, width),
            cv2.INTER_NEAREST,
        ),
        "background_seed": _resize_map(
            background_seed.astype(np.float32),
            (height, width),
            cv2.INTER_NEAREST,
        ),
    })
    return weights


def _native_shape_regularize(
    original: np.ndarray,
    score: np.ndarray,
    alpha: np.ndarray,
    view_evidence: np.ndarray,
    *,
    threshold: float,
    reference_short_side: float,
    small_width: float,
    large_width: float,
    small_gain: float,
    large_gain: float,
    elongated_structure_gain: float,
    aspect_low: float,
    aspect_high: float,
    appearance_score_low: float,
    appearance_score_high: float,
    appearance_contrast_low: float,
    appearance_contrast_high: float,
) -> Dict[str, np.ndarray]:
    """Regularise native-resolution components using size and 2-D shape.

    ``small_width`` and ``large_width`` are measured after normalising the
    source image to ``reference_short_side``.  A component is compact-small
    only when both rotated-box axes are small.  Long thin components require
    bright local contrast, a strong V3 score, or reliable cross-view evidence
    before receiving a high delighting gain; dark seams remain attenuated.
    """
    height, width = score.shape
    normalization = float(reference_short_side) / max(min(height, width), 1)
    luminance = (
        0.299 * original[..., 0]
        + 0.587 * original[..., 1]
        + 0.114 * original[..., 2]
    ).astype(np.float32)
    core = (score >= float(threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        core, connectivity=8
    )

    regularized = alpha.astype(np.float32, copy=True)
    shape_scale = np.zeros_like(alpha, dtype=np.float32)
    size_gain = np.zeros_like(alpha, dtype=np.float32)
    elongation_map = np.zeros_like(alpha, dtype=np.float32)
    minor_extent = np.zeros_like(alpha, dtype=np.float32)
    major_extent = np.zeros_like(alpha, dtype=np.float32)
    pixel_view = _smoothstep(0.08, 0.35, view_evidence)
    view_override = pixel_view.copy()
    appearance_override = pixel_view.copy()
    background_limit = max(float(threshold) * 0.5, 0.05)
    margin = max(int(round(4.0 / max(normalization, 1e-6))), 2)

    for component_id in range(1, count):
        area_pixels = int(stats[component_id, cv2.CC_STAT_AREA])
        if area_pixels < 1:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        local_labels = labels[y : y + box_height, x : x + box_width]
        component = local_labels == component_id

        contour_result = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = contour_result[-2]
        if contours:
            points = np.concatenate(contours, axis=0)
            (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(points)
            # minAreaRect measures centre-to-centre extent; add one pixel to
            # recover occupied support for one- and two-pixel components.
            minor_px = max(min(rect_width, rect_height) + 1.0, 1.0)
            major_px = max(max(rect_width, rect_height) + 1.0, minor_px)
        else:
            minor_px = float(max(min(box_width, box_height), 1))
            major_px = float(max(box_width, box_height, 1))

        minor_normalized = minor_px * normalization
        major_normalized = major_px * normalization
        aspect_ratio = major_normalized / max(minor_normalized, 1e-6)
        minor_scale = float(
            _smoothstep(small_width, large_width, minor_normalized)
        )
        major_scale = float(
            _smoothstep(small_width, large_width, major_normalized)
        )
        geometric_scale = max(minor_scale, major_scale)
        elongation = float(_smoothstep(aspect_low, aspect_high, aspect_ratio))

        component_score = float(
            np.percentile(score[y : y + box_height, x : x + box_width][component], 85)
        )
        score_gate = float(
            _smoothstep(
                appearance_score_low,
                appearance_score_high,
                component_score,
            )
        )
        component_luminance = float(
            np.percentile(
                luminance[y : y + box_height, x : x + box_width][component],
                85,
            )
        )
        x0 = max(x - margin, 0)
        y0 = max(y - margin, 0)
        x1 = min(x + box_width + margin, width)
        y1 = min(y + box_height + margin, height)
        surrounding_score = score[y0:y1, x0:x1]
        surrounding_luminance = luminance[y0:y1, x0:x1]
        background = surrounding_luminance[surrounding_score < background_limit]
        background_luminance = (
            float(np.median(background))
            if background.size
            else float(np.median(surrounding_luminance))
        )
        contrast = max(component_luminance - background_luminance, 0.0)
        contrast_gate = float(
            _smoothstep(
                appearance_contrast_low,
                appearance_contrast_high,
                contrast,
            )
        )
        component_view = float(
            np.percentile(
                view_evidence[y : y + box_height, x : x + box_width][component],
                85,
            )
        )
        component_view_override = float(_smoothstep(0.08, 0.35, component_view))
        reflection_override = max(
            component_view_override, score_gate * contrast_gate
        )

        # Long thin components do not become "small" merely because their
        # thickness is small. Bright/view-dependent strips retain the normal
        # geometric scale; dark stable seams are explicitly attenuated.
        component_scale = geometric_scale * (
            (1.0 - elongation) + elongation * reflection_override
        )
        gain = small_gain + (large_gain - small_gain) * component_scale
        gain += (large_gain - gain) * reflection_override
        structure_gate = elongation * (1.0 - reflection_override)
        gain = (
            (1.0 - structure_gate) * gain
            + structure_gate * elongated_structure_gain
        )

        local_alpha = alpha[y : y + box_height, x : x + box_width]
        component_alpha = float(np.percentile(local_alpha[component], 70))
        candidate = gain * (
            (1.0 - component_scale) * component_alpha
            + component_scale * local_alpha[component]
        )
        local_regularized = regularized[y : y + box_height, x : x + box_width]
        local_regularized[component] = np.clip(candidate, 0.0, 1.0)
        regularized[y : y + box_height, x : x + box_width] = local_regularized

        for target, value in (
            (shape_scale, component_scale),
            (size_gain, gain),
            (elongation_map, elongation),
            (minor_extent, min(minor_normalized / large_width, 1.0)),
            (major_extent, min(major_normalized / large_width, 1.0)),
            (view_override, component_view_override),
            (appearance_override, reflection_override),
        ):
            local_target = target[y : y + box_height, x : x + box_width]
            if target is view_override or target is appearance_override:
                local_target[component] = np.maximum(
                    local_target[component], value
                )
            else:
                local_target[component] = value
            target[y : y + box_height, x : x + box_width] = local_target

    return {
        "regularized": regularized,
        "size_scale": shape_scale,
        "size_gain": size_gain,
        "view_override": view_override,
        "appearance_override": appearance_override,
        "elongation": elongation_map,
        "minor_extent": minor_extent,
        "major_extent": major_extent,
    }


def compute_shape_adaptive_feather_weights(
    original: np.ndarray,
    score_map: np.ndarray,
    *,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    steepness: float = 8.0,
    data_factor: int = 4,
    reference_short_side: float = 3072.0,
    small_width: float = 6.0,
    large_width: float = 8.0,
    small_gain: float = 0.6,
    large_gain: float = 1.15,
    elongated_structure_gain: float = 0.2,
    aspect_low: float = 2.5,
    aspect_high: float = 4.0,
    appearance_score_low: float = 0.30,
    appearance_score_high: float = 0.55,
    appearance_contrast_low: float = 0.025,
    appearance_contrast_high: float = 0.12,
    stable_detail_delight: float = 0.12,
    stable_chroma_delight: float = 0.55,
) -> Dict[str, np.ndarray]:
    """Build native-resolution, shape-aware V3 blend weights.

    ``data_factor`` is accepted for pipeline compatibility but deliberately
    does not control component measurement. Components are classified before
    training downsampling so 2--8 px highlights are not averaged away.
    """
    del data_factor
    score = np.clip(score_map.astype(np.float32), 0.0, 1.0)
    if original.shape[:2] != score.shape:
        raise ValueError("original and score_map dimensions must match")
    if large_width <= small_width:
        raise ValueError("large_width must be greater than small_width")
    if reference_short_side <= 0:
        raise ValueError("reference_short_side must be positive")

    cross = (
        np.zeros_like(score)
        if cross_view_map is None
        else np.clip(cross_view_map.astype(np.float32), 0.0, 1.0)
    )
    confidence = (
        np.zeros_like(score)
        if cross_view_confidence is None
        else np.clip(cross_view_confidence.astype(np.float32), 0.0, 1.0)
    )
    if cross.shape != score.shape or confidence.shape != score.shape:
        raise ValueError("cross-view maps must match score_map dimensions")

    view_evidence = cross * (0.25 + 0.75 * confidence)
    alpha = _zero_floor_sigmoid(
        score, threshold=threshold, steepness=steepness
    )
    maps = _native_shape_regularize(
        original,
        score,
        alpha,
        view_evidence,
        threshold=threshold,
        reference_short_side=reference_short_side,
        small_width=small_width,
        large_width=large_width,
        small_gain=small_gain,
        large_gain=large_gain,
        elongated_structure_gain=elongated_structure_gain,
        aspect_low=aspect_low,
        aspect_high=aspect_high,
        appearance_score_low=appearance_score_low,
        appearance_score_high=appearance_score_high,
        appearance_contrast_low=appearance_contrast_low,
        appearance_contrast_high=appearance_contrast_high,
    )

    normalization = reference_short_side / max(min(score.shape), 1)
    radius = max(int(round(1.0 / max(normalization, 1e-6))), 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    regularized = cv2.dilate(maps["regularized"], kernel, iterations=1)
    feather_sigma = max(0.8 / max(normalization, 1e-6), 0.5)
    regularized = cv2.GaussianBlur(
        regularized,
        (0, 0),
        sigmaX=feather_sigma,
        sigmaY=feather_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )

    luminance = (
        0.299 * original[..., 0]
        + 0.587 * original[..., 1]
        + 0.114 * original[..., 2]
    ).astype(np.float32)
    guide_sigma = max(1.5 / max(normalization, 1e-6), 0.5)
    guide_radius = max(int(round(6.0 / max(normalization, 1e-6))), 2)
    coarse_guide = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=guide_sigma,
        sigmaY=guide_sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    base_weight = np.clip(
        _guided_filter(
            coarse_guide,
            regularized,
            radius=guide_radius,
            epsilon=0.006,
        ),
        0.0,
        1.0,
    ).astype(np.float32)

    appearance_override = np.clip(
        cv2.GaussianBlur(
            maps["appearance_override"],
            (0, 0),
            sigmaX=guide_sigma,
            sigmaY=guide_sigma,
            borderType=cv2.BORDER_REPLICATE,
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    detail_fraction = np.clip(
        stable_detail_delight
        + (1.0 - stable_detail_delight) * appearance_override,
        0.0,
        1.0,
    )
    chroma_fraction = np.clip(
        stable_chroma_delight
        + (1.0 - stable_chroma_delight) * appearance_override,
        0.0,
        1.0,
    )
    diagnostics = {
        key: np.clip(values, 0.0, 1.0).astype(np.float32)
        for key, values in maps.items()
        if key != "regularized"
    }
    diagnostics.update(
        {
            "base_weight": base_weight,
            "detail_weight": (base_weight * detail_fraction).astype(np.float32),
            "chroma_weight": (base_weight * chroma_fraction).astype(np.float32),
            "size_gain": np.clip(
                maps["size_gain"] / max(large_gain, 1e-6), 0.0, 1.0
            ).astype(np.float32),
        }
    )
    return diagnostics


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

@register("delight_only", needs_scoremap=False)
def blend_delight_only(
    original: np.ndarray,
    delighted: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    **params,
) -> np.ndarray:
    """Use delighted pixels everywhere (ignore the original image)."""
    return np.clip(delighted, 0.0, 1.0).astype(np.float32)


@register("soft")
def blend_soft(
    original: np.ndarray,
    delighted: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.3,
    steepness: float = 8.0,
    **params,
) -> np.ndarray:
    """Continuous blend via sigmoid: ``w = σ(steepness · (score − threshold))``.

    ``threshold`` is the score level that maps to weight 0.5; higher
    values blend in delighted pixels more conservatively. ``steepness``
    controls the sharpness of the transition band.
    """
    score = _require_scoremap(score_map, "soft")
    weight = 1.0 / (1.0 + np.exp(-steepness * (score - threshold)))
    return _blend_with_weight(original, delighted, weight.astype(np.float32))


@register("hard")
def blend_hard(
    original: np.ndarray,
    delighted: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.3,
    **params,
) -> np.ndarray:
    """Binary cutoff: delighted where ``score >= threshold``, else original."""
    score = _require_scoremap(score_map, "hard")
    weight = (score >= threshold).astype(np.float32)
    return _blend_with_weight(original, delighted, weight)


@register(
    "adaptive_feather",
    provides_diagnostics=True,
    needs_v3_context=True,
    version=ADAPTIVE_FEATHER_VERSION,
)
def blend_adaptive_feather(
    original: np.ndarray,
    delighted: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.3,
    steepness: float = 8.0,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    data_factor: int = 4,
    return_diagnostics: bool = False,
    detail_sigma: float = 2.0,
    **params,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Scale-adaptive, edge-feathered and perceptual V3 blending.

    Large coherent regions retain (or slightly strengthen) normal delighting.
    Small stable components receive a lower, internally consistent weight;
    reliable per-observation cross-view residuals can override this size prior.
    Broad luminance, chroma and fine luminance detail use separate weights so
    a tiny embossed logo does not alternate between original and delighted
    colours at adjacent pixels.
    """
    score = _require_scoremap(score_map, "adaptive_feather")
    if original.shape != delighted.shape:
        raise ValueError("original and delighted images must have matching dimensions")
    weights = compute_adaptive_feather_weights(
        original,
        score,
        cross_view_map=cross_view_map,
        cross_view_confidence=cross_view_confidence,
        threshold=threshold,
        steepness=steepness,
        data_factor=data_factor,
        **params,
    )

    original_lab = cv2.cvtColor(
        original.astype(np.float32, copy=False), cv2.COLOR_RGB2LAB
    )
    delighted_lab = cv2.cvtColor(
        delighted.astype(np.float32, copy=False), cv2.COLOR_RGB2LAB
    )
    image_scale = min(original.shape[:2]) / 3072.0
    sigma = max(float(detail_sigma) * image_scale, 0.5)

    original_l = original_lab[..., 0]
    delighted_l = delighted_lab[..., 0]
    original_base = cv2.GaussianBlur(
        original_l, (0, 0), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    delighted_base = cv2.GaussianBlur(
        delighted_l, (0, 0), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    original_detail = original_l - original_base
    delighted_detail = delighted_l - delighted_base

    base_weight = weights["base_weight"]
    detail_weight = weights["detail_weight"]
    chroma_weight = weights["chroma_weight"]
    output_l = (
        (1.0 - base_weight) * original_base
        + base_weight * delighted_base
        + (1.0 - detail_weight) * original_detail
        + detail_weight * delighted_detail
    )
    output_ab = (
        (1.0 - chroma_weight[..., None]) * original_lab[..., 1:]
        + chroma_weight[..., None] * delighted_lab[..., 1:]
    )
    output_lab = np.concatenate(
        [np.clip(output_l, 0.0, 100.0)[..., None], output_ab], axis=2
    ).astype(np.float32, copy=False)
    blended = np.clip(
        cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0
    ).astype(np.float32, copy=False)
    if return_diagnostics:
        return blended, weights
    return blended


@register(
    "weak_hysteresis_feather",
    provides_diagnostics=True,
    needs_v3_context=True,
    needs_detector_cues=True,
    version=WEAK_HYSTERESIS_FEATHER_VERSION,
)
def blend_weak_hysteresis_feather(
    original: np.ndarray,
    delighted: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.3,
    steepness: float = 8.0,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    detector_cues: Optional[Dict[str, np.ndarray]] = None,
    data_factor: int = 4,
    return_diagnostics: bool = False,
    detail_sigma: float = 2.0,
    **params,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Experimental V5 weak-reflection recovery with continuous feathering."""
    score = _require_scoremap(score_map, "weak_hysteresis_feather")
    if original.shape != delighted.shape:
        raise ValueError("original and delighted images must have matching dimensions")
    weights = compute_weak_hysteresis_feather_weights(
        original,
        score,
        delighted=delighted,
        cross_view_map=cross_view_map,
        cross_view_confidence=cross_view_confidence,
        detector_cues=detector_cues,
        threshold=threshold,
        steepness=steepness,
        data_factor=data_factor,
        **params,
    )

    original_lab = cv2.cvtColor(original.astype(np.float32), cv2.COLOR_RGB2LAB)
    delighted_lab = cv2.cvtColor(delighted.astype(np.float32), cv2.COLOR_RGB2LAB)
    image_scale = min(original.shape[:2]) / 3072.0
    sigma = max(float(detail_sigma) * image_scale, 0.5)
    original_l = original_lab[..., 0]
    delighted_l = delighted_lab[..., 0]
    original_base = cv2.GaussianBlur(
        original_l,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    delighted_base = cv2.GaussianBlur(
        delighted_l,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    original_detail = original_l - original_base
    delighted_detail = delighted_l - delighted_base

    base_weight = weights["base_weight"]
    detail_weight = weights["detail_weight"]
    chroma_weight = weights["chroma_weight"]
    output_l = (
        (1.0 - base_weight) * original_base
        + base_weight * delighted_base
        + (1.0 - detail_weight) * original_detail
        + detail_weight * delighted_detail
    )
    output_ab = (
        (1.0 - chroma_weight[..., None]) * original_lab[..., 1:]
        + chroma_weight[..., None] * delighted_lab[..., 1:]
    )
    output_lab = np.concatenate(
        [np.clip(output_l, 0.0, 100.0)[..., None], output_ab], axis=2
    ).astype(np.float32)
    blended = np.clip(
        cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0
    ).astype(np.float32)
    if return_diagnostics:
        return blended, weights
    return blended


@register(
    "weak_hysteresis_outer_feather",
    provides_diagnostics=True,
    needs_v3_context=True,
    needs_detector_cues=True,
    version=WEAK_HYSTERESIS_OUTER_FEATHER_VERSION,
)
def blend_weak_hysteresis_outer_feather(
    original: np.ndarray,
    delighted: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.3,
    steepness: float = 8.0,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    detector_cues: Optional[Dict[str, np.ndarray]] = None,
    data_factor: int = 4,
    return_diagnostics: bool = False,
    detail_sigma: float = 2.0,
    **params,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """V5.1: place the V5 transition in a supported outer reflection shell."""
    resolved = dict(WEAK_HYSTERESIS_OUTER_FEATHER_DEFAULT_PARAMETERS)
    resolved.update(params)
    return blend_weak_hysteresis_feather(
        original,
        delighted,
        score_map,
        threshold=threshold,
        steepness=steepness,
        cross_view_map=cross_view_map,
        cross_view_confidence=cross_view_confidence,
        detector_cues=detector_cues,
        data_factor=data_factor,
        return_diagnostics=return_diagnostics,
        detail_sigma=detail_sigma,
        **resolved,
    )


@register(
    "evidence_matte",
    provides_diagnostics=True,
    needs_v3_context=True,
    needs_detector_cues=True,
    version=EVIDENCE_MATTE_VERSION,
)
def blend_evidence_matte(
    original: np.ndarray,
    delighted: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.3,
    steepness: float = 8.0,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    detector_cues: Optional[Dict[str, np.ndarray]] = None,
    data_factor: int = 4,
    return_diagnostics: bool = False,
    detail_sigma: float = 2.0,
    **params,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """V6 evidence-domain reflection matte with no spatial boundary shift."""
    score = _require_scoremap(score_map, "evidence_matte")
    if original.shape != delighted.shape:
        raise ValueError("original and delighted images must have matching dimensions")
    resolved = dict(EVIDENCE_MATTE_DEFAULT_PARAMETERS)
    resolved.update(params)
    weights = compute_evidence_matte_weights(
        original,
        score,
        delighted=delighted,
        cross_view_map=cross_view_map,
        cross_view_confidence=cross_view_confidence,
        detector_cues=detector_cues,
        threshold=threshold,
        steepness=steepness,
        data_factor=data_factor,
        include_diagnostics=return_diagnostics,
        **resolved,
    )

    original_lab = cv2.cvtColor(
        original.astype(np.float32, copy=False), cv2.COLOR_RGB2LAB
    )
    delighted_lab = cv2.cvtColor(
        delighted.astype(np.float32, copy=False), cv2.COLOR_RGB2LAB
    )
    image_scale = min(original.shape[:2]) / 3072.0
    sigma = max(float(detail_sigma) * image_scale, 0.5)
    original_l = original_lab[..., 0]
    delighted_l = delighted_lab[..., 0]
    original_base = cv2.GaussianBlur(
        original_l,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    delighted_base = cv2.GaussianBlur(
        delighted_l,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    original_detail = original_l - original_base
    delighted_detail = delighted_l - delighted_base
    base_weight = weights["base_weight"]
    detail_weight = weights["detail_weight"]
    chroma_weight = weights["chroma_weight"]
    output_l = (
        (1.0 - base_weight) * original_base
        + base_weight * delighted_base
        + (1.0 - detail_weight) * original_detail
        + detail_weight * delighted_detail
    )
    output_ab = (
        (1.0 - chroma_weight[..., None]) * original_lab[..., 1:]
        + chroma_weight[..., None] * delighted_lab[..., 1:]
    )
    output_lab = np.concatenate(
        [np.clip(output_l, 0.0, 100.0)[..., None], output_ab], axis=2
    ).astype(np.float32, copy=False)
    blended = np.clip(
        cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0
    ).astype(np.float32, copy=False)
    if return_diagnostics:
        return blended, weights
    return blended


def blend_shape_adaptive_feather(
    original: np.ndarray,
    delighted: np.ndarray,
    score_map: Optional[np.ndarray] = None,
    *,
    threshold: float = 0.3,
    steepness: float = 8.0,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    data_factor: int = 4,
    return_diagnostics: bool = False,
    detail_sigma: float = 2.0,
    **params,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Inactive historical native-scale blending prototype.

    Kept as unregistered code so the experiment remains recoverable, but it
    is intentionally unavailable through ``--blend-mode`` after restoring
    the ``size_view_perceptual_v1`` validation state.
    """
    score = _require_scoremap(score_map, "shape_adaptive_feather")
    if original.shape != delighted.shape:
        raise ValueError("original and delighted images must have matching dimensions")
    weights = compute_shape_adaptive_feather_weights(
        original,
        score,
        cross_view_map=cross_view_map,
        cross_view_confidence=cross_view_confidence,
        threshold=threshold,
        steepness=steepness,
        data_factor=data_factor,
        **params,
    )

    original_lab = cv2.cvtColor(original.astype(np.float32), cv2.COLOR_RGB2LAB)
    delighted_lab = cv2.cvtColor(delighted.astype(np.float32), cv2.COLOR_RGB2LAB)
    image_scale = min(original.shape[:2]) / 3072.0
    sigma = max(float(detail_sigma) * image_scale, 0.5)

    original_l = original_lab[..., 0]
    delighted_l = delighted_lab[..., 0]
    original_base = cv2.GaussianBlur(
        original_l,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    delighted_base = cv2.GaussianBlur(
        delighted_l,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    original_detail = original_l - original_base
    delighted_detail = delighted_l - delighted_base

    base_weight = weights["base_weight"]
    detail_weight = weights["detail_weight"]
    chroma_weight = weights["chroma_weight"]
    output_l = (
        (1.0 - base_weight) * original_base
        + base_weight * delighted_base
        + (1.0 - detail_weight) * original_detail
        + detail_weight * delighted_detail
    )
    output_ab = (
        (1.0 - chroma_weight[..., None]) * original_lab[..., 1:]
        + chroma_weight[..., None] * delighted_lab[..., 1:]
    )
    output_lab = np.concatenate(
        [np.clip(output_l, 0.0, 100.0)[..., None], output_ab], axis=2
    ).astype(np.float32)
    blended = np.clip(
        cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0
    ).astype(np.float32)
    if return_diagnostics:
        return blended, weights
    return blended

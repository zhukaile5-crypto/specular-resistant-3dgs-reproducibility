"""Structure-aware V3 specular detector.

V3 refines, rather than replaces, the V2 detector.  It keeps V2's signed
multi-scale appearance score and adds three conservative cues:

1. paired positive/negative edges and coherent fine texture are treated as
   structural evidence;
2. per-observation cross-view residuals provide view-specific support, while
   a separate confidence map may suppress stable structural responses; and
3. an exposure-aligned original/delighted residual can rescue true highlights
   that would otherwise look like ordinary edges.

The public API mirrors :mod:`stabledelight.utils.specular_detector`, but lives
in a separate module so the validated V2 implementation remains unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from stabledelight.utils.specular_detector import (
    _box_blur,
    _rgb_to_luminance,
    _robust_normalize,
    _to_float_rgb,
    compute_specular_score as compute_v2_specular_score,
)

DETECTOR_VERSION = "structure_observation_v3"


def _odd_kernel(value: int) -> int:
    value = max(int(value), 1)
    return value if value % 2 else value + 1


def _guided_base(
    luminance: np.ndarray,
    *,
    kernel_size: int = 15,
    epsilon: float = 0.01,
    mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Edge-preserving guided-filter base layer for a luminance image."""
    kernel_size = _odd_kernel(kernel_size)
    if mean is None:
        mean, correlation = _box_blur(
            np.stack((luminance, luminance * luminance), axis=0),
            kernel_size,
        )
    else:
        correlation = _box_blur(luminance * luminance, kernel_size)
    variance = np.maximum(correlation - mean * mean, 0.0)
    a = variance / (variance + max(float(epsilon), 1e-6))
    b = mean - a * mean
    mean_a, mean_b = _box_blur(
        np.stack((a, b), axis=0),
        kernel_size,
    )
    return (mean_a * luminance + mean_b).astype(np.float32, copy=False)


def compute_structure_penalty(
    image: np.ndarray | Image.Image,
    *,
    structure_kernel: int = 15,
    pair_kernel: int = 9,
    tensor_kernel: int = 15,
    return_cues: bool = False,
    _shared_context: Optional[dict] = None,
) -> np.ndarray | Tuple[np.ndarray, dict]:
    """Detect narrow, paired and directionally coherent bright structures.

    A grout boundary or ribbed cup produces adjacent positive and negative
    high-pass lobes.  A broad reflection is predominantly positive and has
    much lower directional coherence.  The returned map is therefore a
    *penalty confidence*, not a specular probability.
    """
    rgb = _to_float_rgb(image)
    shared = _shared_context if _shared_context is not None else {}
    luminance = shared.get("luminance")
    if luminance is None:
        luminance = _rgb_to_luminance(rgb)
        shared["luminance"] = luminance
    structure_kernel = _odd_kernel(structure_kernel)
    mean = shared.get(("luminance_blur", structure_kernel))
    base = _guided_base(
        luminance,
        kernel_size=structure_kernel,
        mean=mean,
    )
    residual = luminance - base

    positive = _robust_normalize(np.maximum(residual, 0.0), percentile=99.5)
    negative = _robust_normalize(np.maximum(-residual, 0.0), percentile=99.5)
    nearby_negative = _robust_normalize(
        _box_blur(negative, _odd_kernel(pair_kernel)), percentile=99.0
    )
    paired = np.sqrt(np.clip(positive * nearby_negative, 0.0, 1.0))

    grad_y, grad_x = np.gradient(luminance)
    tensor_kernel = _odd_kernel(tensor_kernel)
    jxx, jyy, jxy = _box_blur(
        np.stack(
            (grad_x * grad_x, grad_y * grad_y, grad_x * grad_y),
            axis=0,
        ),
        tensor_kernel,
    )
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (
        jxx + jyy + 1e-6
    )
    edge_strength = _robust_normalize(
        np.hypot(grad_x, grad_y), percentile=99.0
    )
    paired_blur, edge_blur = _box_blur(
        np.stack((paired, edge_strength), axis=0),
        _odd_kernel(pair_kernel),
    )
    paired_neighbourhood = _robust_normalize(
        paired_blur,
        percentile=99.0,
    )
    edge_neighbourhood = _robust_normalize(
        edge_blur,
        percentile=99.0,
    )

    # Repeated high-frequency material texture has both signed residual
    # energy and a coherent local orientation.  Requiring all three factors
    # avoids treating sensor noise or a smooth highlight as structure.
    penalty = np.cbrt(
        np.clip(
            np.maximum(paired, paired_neighbourhood)
            * coherence
            * np.maximum(edge_strength, edge_neighbourhood),
            0.0,
            1.0,
        )
    ).astype(np.float32, copy=False)

    if return_cues:
        return penalty, {
            "guided_base": base,
            "positive_structure": positive,
            "negative_structure": negative,
            "paired_structure": paired.astype(np.float32, copy=False),
            "paired_structure_neighbourhood": paired_neighbourhood,
            "structure_coherence": coherence.astype(np.float32, copy=False),
            "edge_strength": edge_strength,
        }
    return penalty


def _robust_align_delighted(
    original_luminance: np.ndarray,
    delighted_luminance: np.ndarray,
    *,
    max_samples: int = 200_000,
) -> np.ndarray:
    """Affine-align delighted luminance without fitting removed highlights."""
    flat_original = original_luminance.reshape(-1)
    flat_delighted = delighted_luminance.reshape(-1)
    step = max(1, int(np.ceil(flat_original.size / max_samples)))
    y = flat_original[::step].astype(np.float64)
    x = flat_delighted[::step].astype(np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    gain, offset = 1.0, 0.0
    for _ in range(5):
        if int(keep.sum()) < 100:
            break
        design = np.column_stack((x[keep], np.ones(int(keep.sum()))))
        gain, offset = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (gain * x + offset)
        centre = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - centre)))
        if mad <= 1e-6:
            break
        # Reject both model changes and shadows from the global fit.
        keep = np.abs(residual - centre) <= 3.0 * 1.4826 * mad
    gain = float(np.clip(gain, 0.5, 2.0))
    offset = float(np.clip(offset, -0.25, 0.25))
    return np.clip(gain * delighted_luminance + offset, 0.0, 1.0).astype(
        np.float32
    )


def compute_delight_residual_map(
    original: np.ndarray | Image.Image,
    delighted: np.ndarray | Image.Image,
    *,
    kernel_ratio: float = 0.02,
    min_kernel: int = 31,
    _original_luminance: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return broad positive luminance removed by delighting in [0, 1]."""
    original_rgb = _to_float_rgb(original)
    delighted_rgb = _to_float_rgb(delighted)
    if original_rgb.shape[:2] != delighted_rgb.shape[:2]:
        raise ValueError(
            "original and delighted images must have matching dimensions"
        )
    original_lum = _original_luminance
    if original_lum is None:
        original_lum = _rgb_to_luminance(original_rgb)
    delighted_lum = _rgb_to_luminance(delighted_rgb)
    aligned = _robust_align_delighted(original_lum, delighted_lum)
    kernel = max(min_kernel, int(round(min(original_lum.shape) * kernel_ratio)))
    kernel = _odd_kernel(kernel)
    removed = np.maximum(original_lum - aligned, 0.0)
    broad_removed = np.maximum(_box_blur(removed, kernel) - 0.01, 0.0)
    return _robust_normalize(broad_removed, percentile=99.0)


def compute_specular_score(
    image: np.ndarray | Image.Image,
    *,
    cross_view_map: Optional[np.ndarray] = None,
    cross_view_confidence: Optional[np.ndarray] = None,
    delighted_image: Optional[np.ndarray | Image.Image] = None,
    cross_view_weight: float = 1.0,
    structure_penalty_weight: float = 0.72,
    stable_structure_weight: float = 0.35,
    structure_rescue_weight: float = 0.95,
    delight_weight: float = 0.85,
    return_cues: bool = False,
) -> np.ndarray | Tuple[np.ndarray, dict]:
    """Compute the V3 structure-aware, threshold-independent score map."""
    rgb = _to_float_rgb(image)
    shared = {}
    single, v2_cues = compute_v2_specular_score(
        rgb,
        cross_view_map=None,
        return_cues=True,
        _shared_context=shared,
    )
    structure, structure_cues = compute_structure_penalty(
        rgb,
        return_cues=True,
        _shared_context=shared,
    )

    zeros = np.zeros(single.shape, dtype=np.float32)
    cross = zeros
    if cross_view_map is not None:
        cross = np.clip(
            cross_view_map.astype(np.float32, copy=False), 0.0, 1.0
        )
        if cross.shape != single.shape:
            raise ValueError(
                f"cross_view_map shape {cross.shape} does not match {single.shape}"
            )

    stable = zeros
    if cross_view_confidence is not None:
        stable = np.clip(
            cross_view_confidence.astype(np.float32, copy=False), 0.0, 1.0
        )
        if stable.shape != single.shape:
            raise ValueError(
                "cross_view_confidence shape does not match the input image"
            )

    delight = zeros
    if delighted_image is not None:
        delight = compute_delight_residual_map(
            rgb,
            delighted_image,
            _original_luminance=shared["luminance"],
        )

    appearance = np.maximum(
        v2_cues["luminance_excess"], v2_cues["chromaticity"]
    )
    cross_support = cross * (0.20 + 0.80 * appearance)
    delight_support = delight * (0.20 + 0.80 * appearance)
    rescue = np.maximum(cross_support, delight_support)

    # Only suppress an edge when it lacks independent view-specific or
    # delight-removal evidence.  Stable cross-view evidence makes the veto
    # stronger, but never acts alone on a smooth region.
    effective_structure = structure * (
        1.0 - np.clip(structure_rescue_weight, 0.0, 1.0) * rescue
    )
    refined = single * (
        1.0 - np.clip(structure_penalty_weight, 0.0, 1.0) * effective_structure
    )
    refined *= 1.0 - (
        np.clip(stable_structure_weight, 0.0, 1.0)
        * stable
        * effective_structure
    )

    support = np.maximum(
        max(float(cross_view_weight), 0.0) * cross_support,
        max(float(delight_weight), 0.0) * delight_support,
    )
    support_alpha = max(float(cross_view_weight), float(delight_weight), 0.0)
    support_alpha = support_alpha / (1.0 + support_alpha)
    score = refined + support_alpha * np.maximum(support - refined, 0.0)
    score = np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)

    if return_cues:
        cues = dict(v2_cues)
        cues.update(structure_cues)
        cues.update(
            {
                "v2_single": single,
                "structure_penalty": structure,
                "cross_view": cross,
                "cross_view_confidence": stable,
                "delight_residual": delight,
                "cross_support": cross_support,
                "delight_support": delight_support,
            }
        )
        return score, cues
    return score


def compute_specular_probability(
    image: np.ndarray | Image.Image,
    *,
    soft_threshold: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """Map the V3 score through the same sigmoid used by V2."""
    score = compute_specular_score(image, **kwargs)
    return (1.0 / (1.0 + np.exp(-8.0 * (score - soft_threshold)))).astype(
        np.float32
    )

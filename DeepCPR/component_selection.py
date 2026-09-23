# -*- coding: utf-8 -*-
"""Component-number selection for chromatographic overlap candidates.

The resolver in this module is deliberately independent of TIC peak counting
and SNR. Post-ITTFA overlap only triggers model comparison. Component number
is selected from local rank evidence, masked matrix reconstruction,
perturbation stability, and an optional FRR confirmation callback supplied by
the main resolver. FRR is used as a confirmation layer rather than as the
sole component-number criterion.
"""

from dataclasses import asdict, dataclass
from itertools import combinations, product

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


@dataclass
class ComponentSelectionConfig:
    active_profile_threshold: float = 3e-3
    rank_row_fraction: float = 0.02
    apex_window: int = 2
    shape_weight: float = 0.01
    max_iterations: int = 25
    convergence_tolerance: float = 1e-5
    cv_folds: int = 4
    cv_repeats: int = 1
    frr_candidate_limit: int = 3
    paired_bootstrap_runs: int = 400
    paired_confidence_level: float = 0.95
    stability_runs: int = 5
    stability_apex_sd: float = 1.5
    stability_spectral_cosine: float = 0.95
    stability_area_cv: float = 0.25
    max_candidates_per_count: int = 2
    random_seed: int = 20260910
    # Prefer the smaller supported model when independent evidence conflicts.
    prefer_simpler_on_conflict: bool = True
    # FRR is retained for the final fixed-count arbitration in DeepCPR.py.
    # Candidate component-number selection uses the independent shape, CV,
    # residual and stability evidence by default so it does not repeat the
    # expensive FRR grid search for competing subsets.
    enable_candidate_frr_confirmation: bool = False
    enable_frr_confirmation: bool = True
    frr_r2_margin: float = 1e-3
    record_prediction_confidence_proxy: bool = True
    enable_selective_ion_confirmation: bool = True
    selective_ion_contrast: float = 0.15
    # A disputed extra profile must improve paired masked validation beyond
    # the improvement obtained when the simpler-model residual is randomized.
    # This is a deterministic, local overfit test rather than a calibrated
    # probability of component existence.
    enable_residual_null_test: bool = True
    residual_null_runs: int = 12
    residual_null_percentile: float = 0.95
    residual_null_min_improvement: float = 1e-4
    enable_peak_shape_confirmation: bool = True
    peak_shape_bootstrap_runs: int = 8
    peak_shape_confidence_level: float = 0.90
    peak_shape_min_area_fraction: float = 0.05
    peak_shape_min_separation_scans: float = 1.0
    peak_shape_topology_window_scans: int = 2
    peak_shape_min_shoulder_fraction: float = 0.03
    peak_shape_min_topology_support_rate: float = 0.50
    peak_shape_min_observed_widths: float = 1.0
    peak_shape_context_scans: int = 12
    resolved_profile_max_boundary_fraction: float = 0.50


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / den)


def _r2(observed, fitted):
    observed = np.asarray(observed, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    variance = float(np.var(observed, ddof=1))
    if variance <= 1e-15:
        return 0.0
    return float(1.0 - np.var(fitted - observed, ddof=1) / variance)


def _active_profile_indices(profiles, threshold):
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2:
        return []
    return [
        index for index in range(profiles.shape[1])
        if np.max(np.maximum(profiles[:, index], 0.0)) > threshold
    ]


def prediction_confidence_proxy(profiles):
    """Describe model-profile support without claiming calibrated probability.

    The proxy is computed within one segment, so it is insensitive to the
    absolute intensity scale.  It combines relative peak height, relative
    integrated area, apex separation relative to profile width, and profile
    independence.  These descriptors are recorded as evidence only; they do
    not impose a fixed component-count cutoff.
    """
    profiles = np.maximum(np.asarray(profiles, dtype=float), 0.0)
    if profiles.ndim != 2 or profiles.shape[1] == 0:
        return []
    heights = np.max(profiles, axis=0)
    areas = np.trapz(profiles, axis=0)
    positive_heights = heights[heights > 0]
    positive_areas = areas[areas > 0]
    max_height = max(float(np.max(positive_heights)), 1e-12) if positive_heights.size else 1e-12
    total_area = max(float(np.sum(positive_areas)), 1e-12) if positive_areas.size else 1e-12
    apices = np.argmax(profiles, axis=0)
    widths = []
    for index, height in enumerate(heights):
        if height <= 1e-12:
            widths.append(float(profiles.shape[0]))
            continue
        support = np.where(profiles[:, index] >= 0.5 * height)[0]
        widths.append(float(max(1, support[-1] - support[0] + 1)) if support.size else 1.0)

    def percentile(value, values):
        if values.size <= 1:
            return 1.0 if value > 0 else 0.0
        return float(np.mean(values <= value))

    output = []
    for index in range(profiles.shape[1]):
        other = [j for j in range(profiles.shape[1]) if j != index and heights[j] > 1e-12]
        if other:
            correlations = []
            current = profiles[:, index]
            for j in other:
                a = current - np.mean(current)
                b = profiles[:, j] - np.mean(profiles[:, j])
                den = float(np.linalg.norm(a) * np.linalg.norm(b))
                correlations.append(float(np.dot(a, b) / den) if den > 1e-15 else 0.0)
            max_correlation = max(correlations)
            independence = float(np.clip(1.0 - max_correlation, 0.0, 1.0))
            nearest_j = min(
                other,
                key=lambda j: abs(int(apices[index]) - int(apices[j])),
            )
            nearest_distance = abs(
                int(apices[index]) - int(apices[nearest_j])
            )
            separation = float(np.clip(
                nearest_distance / max(widths[index], widths[nearest_j], 1.0),
                0.0,
                1.0,
            ))
        else:
            max_correlation = None
            independence = 1.0
            nearest_distance = None
            separation = 1.0
        relative_height = float(heights[index] / max_height)
        relative_area = float(max(areas[index], 0.0) / total_area)
        strength = 0.5 * percentile(heights[index], positive_heights) + 0.5 * percentile(
            areas[index], positive_areas
        )
        score = float(np.clip(
            0.4 * strength + 0.3 * separation + 0.3 * independence, 0.0, 1.0
        ))
        output.append({
            "profile_index": int(index),
            "relative_height": relative_height,
            "relative_area": relative_area,
            "apex_scan": int(apices[index]),
            "half_height_width_scans": float(widths[index]),
            "nearest_apex_distance_scans": nearest_distance,
            "apex_separation_score": separation,
            "maximum_profile_correlation": max_correlation,
            "profile_independence_score": independence,
            "strength_percentile_score": float(strength),
            "confidence_proxy_score": score,
        })
    return output


def _prediction_profile_independence_test(proxy, candidate_indices,
                                          simpler_indices):
    """Check whether the added predicted profile is distinct within a segment.

    This is a relative, within-segment test. It does not assign an absolute
    confidence threshold to the neural-network output. The added profile must
    be at least as independent and as apex-separated as the median active
    candidate, and must exceed the median on at least one of those measures.
    """
    candidate_indices = tuple(int(value) for value in candidate_indices)
    simpler_indices = tuple(int(value) for value in simpler_indices)
    added = [value for value in candidate_indices
             if value not in simpler_indices]
    by_index = {
        int(item["profile_index"]): item for item in (proxy or [])
    }
    if len(added) != 1 or any(index not in by_index
                              for index in candidate_indices):
        return {
            "supported": False,
            "reason": "prediction_profile_evidence_unavailable",
            "added_component_index": int(added[0]) if len(added) == 1 else None,
        }

    added_index = int(added[0])
    independence = np.asarray([
        by_index[index]["profile_independence_score"]
        for index in candidate_indices
    ], dtype=float)
    separation = np.asarray([
        by_index[index]["apex_separation_score"]
        for index in candidate_indices
    ], dtype=float)
    added_independence = float(
        by_index[added_index]["profile_independence_score"]
    )
    added_separation = float(
        by_index[added_index]["apex_separation_score"]
    )
    median_independence = float(np.median(independence))
    median_separation = float(np.median(separation))
    tolerance = 1e-6
    not_below_typical = bool(
        added_independence >= median_independence - tolerance
        and added_separation >= median_separation - tolerance
    )
    exceeds_typical = bool(
        added_independence > median_independence + tolerance
        or added_separation > median_separation + tolerance
    )
    supported = bool(not_below_typical and exceeds_typical)
    return {
        "supported": supported,
        "reason": (
            "added_prediction_profile_is_independent_within_segment"
            if supported else
            "added_prediction_profile_is_not_independent_within_segment"
        ),
        "added_component_index": added_index,
        "profile_independence_score": added_independence,
        "median_candidate_independence_score": median_independence,
        "apex_separation_score": added_separation,
        "median_candidate_apex_separation_score": median_separation,
        "relative_rule": (
            "not_below_candidate_median_on_both_and_above_median_on_one"
        ),
    }


def _selective_ion_evidence(observed, spectra, anchors, events, config,
                            candidate_indices=None):
    """Estimate whether a disputed pair has informative mass channels.

    This is a descriptive selective ion check, not an identification score.
    It searches channels with appreciable signal and reports the largest
    relative spectral contrast between the two candidate spectra. The
    corresponding observed ion traces are retained as EIC diagnostics.
    """
    X = np.maximum(np.asarray(observed, dtype=float), 0.0)
    S = np.maximum(np.asarray(spectra, dtype=float), 0.0)
    A = np.maximum(np.asarray(anchors, dtype=float), 0.0)
    if X.ndim != 2 or S.ndim != 2 or A.ndim != 2:
        return []
    total = np.sum(X, axis=0)
    active = total > max(float(np.max(total)) * 0.01, 1e-12)
    output = []
    index_map = (
        {int(global_index): local_index
         for local_index, global_index in enumerate(candidate_indices)}
        if candidate_indices is not None else
        {index: index for index in range(S.shape[0])}
    )
    for event in events:
        global_i, global_j = int(event["component_i"]), int(event["component_j"])
        if global_i not in index_map or global_j not in index_map:
            continue
        i, j = index_map[global_i], index_map[global_j]
        channels = np.where(active)[0]
        if channels.size == 0:
            continue
        pair_signal = S[i, channels] + S[j, channels]
        pair_threshold = max(
            float(np.max(pair_signal)) * 0.05,
            float(np.max(S)) * 0.01,
            1e-12,
        )
        informative = pair_signal >= pair_threshold
        channels = channels[informative]
        if channels.size == 0:
            continue
        contrast = np.abs(S[i, channels] - S[j, channels]) / np.maximum(
            S[i, channels] + S[j, channels], 1e-12
        )
        order = np.argsort(contrast)[::-1]
        top = channels[order[: min(5, order.size)]]
        top_contrast = float(contrast[order[0]]) if order.size else 0.0
        apex_i = int(np.argmax(A[:, i])) if i < A.shape[1] else None
        apex_j = int(np.argmax(A[:, j])) if j < A.shape[1] else None
        trace_separation = None
        eic_shape_supported = False
        eic_shape_score = None
        eic_valley_ratio = None
        eic_apex_ratio = None
        if apex_i is not None and apex_j is not None and apex_i != apex_j:
            values = []
            shape_scores = []
            left, right = sorted((apex_i, apex_j))
            for channel in top:
                trace = X[:, int(channel)]
                scale = max(float(np.max(trace)), 1e-12)
                values.append(abs(float(trace[apex_i] - trace[apex_j])) / scale)
                if right - left >= 2:
                    first = float(trace[apex_i])
                    second = float(trace[apex_j])
                    valley = float(np.min(trace[left:right + 1]))
                    peak_scale = max(float(np.max(trace)), 1e-12)
                    apex_ratio = min(first, second) / peak_scale
                    valley_ratio = valley / max(min(first, second), 1e-12)
                    # Require both predicted apex locations to carry signal
                    # and a measurable dip between them. This is deliberately
                    # conservative because a single noisy EIC point must not
                    # create evidence for a second compound.
                    score = float(np.clip(
                        apex_ratio * max(0.0, 1.0 - valley_ratio), 0.0, 1.0
                    ))
                    shape_scores.append(score)
                    if apex_ratio >= 0.20 and valley_ratio <= 0.90:
                        eic_shape_supported = True
            trace_separation = float(max(values)) if values else 0.0
            if shape_scores:
                best_shape = int(np.argmax(shape_scores))
                eic_shape_score = float(shape_scores[best_shape])
                best_trace = X[:, int(top[best_shape])]
                first = float(best_trace[apex_i])
                second = float(best_trace[apex_j])
                eic_apex_ratio = float(
                    min(first, second) / max(float(np.max(best_trace)), 1e-12)
                )
                left, right = sorted((apex_i, apex_j))
                eic_valley_ratio = float(
                    np.min(best_trace[left:right + 1])
                    / max(min(first, second), 1e-12)
                )
            else:
                best_trace = X[:, int(top[0])]
                first = float(best_trace[apex_i])
                second = float(best_trace[apex_j])
                eic_apex_ratio = float(
                    min(first, second) / max(float(np.max(best_trace)), 1e-12)
                )
            # Adjacent scans cannot contain an interior valley. In that case
            # retain a weaker two-apex criterion only when both locations are
            # strong and their ion response differs substantially. This is
            # needed for chromatographic peaks sampled at coarse scan spacing.
            if (not eic_shape_supported and right - left == 1
                    and eic_apex_ratio is not None
                    and eic_apex_ratio >= 0.50
                    and trace_separation is not None
                    and trace_separation >= 0.25):
                eic_shape_supported = True
        output.append({
            "component_i": i,
            "component_j": j,
            "top_mz_channel_indices": [int(value) for value in top],
            "maximum_relative_spectral_contrast": top_contrast,
            "apex_eic_separation": trace_separation,
            "eic_shape_score": eic_shape_score,
            "eic_apex_ratio": eic_apex_ratio,
            "eic_valley_ratio": eic_valley_ratio,
            "eic_two_apex_supported": bool(eic_shape_supported),
            "selective_ion_supported": bool(
                top_contrast >= float(config.selective_ion_contrast)
                and eic_shape_supported
            ),
        })
    return output


def _normalise_profiles(profiles):
    profiles = np.maximum(np.asarray(profiles, dtype=float), 0.0).copy()
    for index in range(profiles.shape[1]):
        maximum = float(np.max(profiles[:, index]))
        if maximum > 1e-15:
            profiles[:, index] /= maximum
    return profiles


def _isotonic_increasing(values):
    """Least-squares projection onto a nondecreasing sequence."""
    values = np.asarray(values, dtype=float)
    block_values = []
    block_weights = []
    block_lengths = []
    for value in values:
        block_values.append(float(value))
        block_weights.append(1.0)
        block_lengths.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            weight = block_weights[-2] + block_weights[-1]
            merged = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / weight
            length = block_lengths[-2] + block_lengths[-1]
            block_values[-2:] = [merged]
            block_weights[-2:] = [weight]
            block_lengths[-2:] = [length]
    return np.concatenate([
        np.full(length, value, dtype=float)
        for value, length in zip(block_values, block_lengths)
    ])


def _project_unimodal(values, anchor_apex, apex_window):
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    lower = max(0, int(anchor_apex) - int(apex_window))
    upper = min(values.size - 1, int(anchor_apex) + int(apex_window))
    best = None
    best_error = np.inf
    for apex in range(lower, upper + 1):
        left = _isotonic_increasing(values[:apex + 1])
        right = _isotonic_increasing(values[apex:][::-1])[::-1]
        apex_value = max(float(left[-1]), float(right[0]))
        left[-1] = apex_value
        right[0] = apex_value
        projected = np.concatenate((left[:-1], right))
        error = float(np.sum((projected - values) ** 2))
        if error < best_error:
            best = projected
            best_error = error
    return np.maximum(best, 0.0)


def _shift_zero_fill(values, shift):
    values = np.asarray(values, dtype=float)
    shifted = np.zeros_like(values)
    if shift == 0:
        shifted[:] = values
    elif shift > 0:
        shifted[shift:] = values[:-shift]
    else:
        shifted[:shift] = values[-shift:]
    return shifted


def _fit_weighted_mcr(observed, anchor, config, weights=None, rng=None,
                      perturb=False):
    """Fit nonnegative C and S using weighted multiplicative updates."""
    observed = np.maximum(np.asarray(observed, dtype=float), 0.0)
    anchor = _normalise_profiles(anchor)
    if observed.ndim != 2 or anchor.ndim != 2:
        raise ValueError("observed and anchor must be two-dimensional")
    if observed.shape[0] != anchor.shape[0] or anchor.shape[1] == 0:
        raise ValueError("anchor dimensions do not match the observed matrix")

    mass_active = np.sum(observed, axis=0) > 0
    if not np.any(mass_active):
        raise ValueError("the observed matrix has no active mass channels")
    X = observed[:, mass_active]
    scale = max(float(np.max(X)), 1e-12)
    Xn = X / scale

    if weights is None:
        W = np.ones_like(Xn)
    else:
        weights = np.asarray(weights, dtype=float)
        W = weights[:, mass_active]
        if W.shape != Xn.shape:
            raise ValueError("weights do not match the observed matrix")

    C0 = anchor.copy()
    if perturb:
        if rng is None:
            rng = np.random.default_rng(config.random_seed)
        perturbed = np.zeros_like(C0)
        for component in range(C0.shape[1]):
            shift = int(rng.integers(-1, 2))
            amplitude = float(rng.uniform(0.95, 1.05))
            perturbed[:, component] = _shift_zero_fill(
                C0[:, component], shift
            ) * amplitude
        C0 = _normalise_profiles(perturbed)

    anchor_apices = np.argmax(C0, axis=0).astype(int)
    C = np.maximum(C0, 1e-10)
    S = np.full((C.shape[1], Xn.shape[1]), 1e-10, dtype=float)
    groups = {}
    for mass_index in range(Xn.shape[1]):
        observed_rows = W[:, mass_index] > 0
        if np.sum(observed_rows) < C.shape[1]:
            continue
        positive_weights = W[observed_rows, mass_index]
        normalised_weights = positive_weights / np.max(positive_weights)
        key = (
            np.packbits(observed_rows).tobytes(),
            normalised_weights.tobytes(),
        )
        groups.setdefault(key, []).append(mass_index)
    for mass_indices in groups.values():
        observed_rows = W[:, mass_indices[0]] > 0
        positive_weights = W[observed_rows, mass_indices[0]]
        normalised_weights = positive_weights / np.max(positive_weights)
        root_weights = np.sqrt(normalised_weights)
        weighted_C = C[observed_rows] * root_weights[:, None]
        weighted_X = Xn[observed_rows][:, mass_indices] * root_weights[:, None]
        S[:, mass_indices] = np.maximum(
            np.linalg.lstsq(weighted_C, weighted_X, rcond=None)[0],
            1e-10,
        )
    epsilon = 1e-12
    previous_loss = np.inf

    for iteration in range(int(config.max_iterations)):
        reconstruction = C @ S
        S *= (C.T @ (W * Xn) + epsilon) / (
            C.T @ (W * reconstruction) + epsilon
        )

        reconstruction = C @ S
        curvature = float(np.mean(np.diag(S @ S.T)))
        penalty = float(config.shape_weight) * max(curvature, epsilon)
        C *= ((W * Xn) @ S.T + penalty * C0 + epsilon) / (
            (W * reconstruction) @ S.T + penalty * C + epsilon
        )

        for component in range(C.shape[1]):
            C[:, component] = _project_unimodal(
                C[:, component], anchor_apices[component], config.apex_window
            )

        # Keep the retention-time order encoded by the model anchors.  Equal
        # apices are allowed because they are precisely what triggers later
        # component-number comparison; crossing apices are not allowed.
        previous_apex = -1
        for component in np.argsort(anchor_apices):
            current_apex = int(np.argmax(C[:, component]))
            ordered_apex = max(current_apex, previous_apex)
            C[:, component] = _project_unimodal(
                C[:, component], ordered_apex, 0
            )
            previous_apex = int(np.argmax(C[:, component]))

        for component in range(C.shape[1]):
            maximum = float(np.max(C[:, component]))
            if maximum > epsilon:
                C[:, component] /= maximum
                S[component, :] *= maximum

        if iteration % 5 == 0 or iteration == config.max_iterations - 1:
            residual = W * (Xn - C @ S)
            loss = float(np.sum(residual ** 2) / max(np.sum(W * Xn ** 2), epsilon))
            loss += float(config.shape_weight) * float(
                np.sum((C - C0) ** 2) / max(np.sum(C0 ** 2), epsilon)
            )
            if np.isfinite(previous_loss):
                relative_change = abs(previous_loss - loss) / max(abs(previous_loss), epsilon)
                if relative_change < config.convergence_tolerance:
                    break
            previous_loss = loss

    spectra = np.zeros((C.shape[1], observed.shape[1]), dtype=float)
    spectra[:, mass_active] = S * scale
    return C, spectra, C @ spectra


def _fixed_profile_score(observed, profiles):
    profiles = _normalise_profiles(profiles)
    spectra = np.maximum(np.linalg.lstsq(profiles, observed, rcond=None)[0], 0.0)
    fitted = profiles @ spectra
    return _r2(observed, fitted)


def _rank_ratios(matrix, row_fraction):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 4:
        return []
    row_signal = np.max(np.abs(matrix), axis=1)
    threshold = max(float(row_fraction) * float(np.max(row_signal)), 1e-12)
    rows = row_signal > threshold
    if np.sum(rows) < 3:
        return []
    active = matrix[rows]
    columns = np.linalg.norm(active, axis=0) > 0
    active = active[:, columns]
    if active.shape[1] < 2:
        return []
    centred = active - np.mean(active, axis=0, keepdims=True)
    singular = np.linalg.svd(centred, full_matrices=False, compute_uv=False)
    if singular.size < 2 or singular[0] <= 1e-15:
        return []
    return [float(value / singular[0]) for value in singular[1:]]


def observed_rank_evidence(observed, row_fraction=0.02):
    """Return lightweight rank evidence for one baseline-corrected segment."""
    ratios = _rank_ratios(observed, row_fraction)
    return {
        "singular_value_ratios": ratios,
        "sigma2_over_sigma1": ratios[0] if ratios else None,
    }


def _build_clusters(events, active_indices):
    active_set = set(active_indices)
    adjacency = {index: set() for index in active_indices}
    for event in events:
        first = int(event["component_i"])
        second = int(event["component_j"])
        if first in active_set and second in active_set:
            adjacency[first].add(second)
            adjacency[second].add(first)

    clusters = []
    visited = set()
    for start in active_indices:
        if start in visited or not adjacency[start]:
            continue
        stack = [start]
        cluster = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.add(current)
            stack.extend(adjacency[current] - visited)
        clusters.append(tuple(sorted(cluster)))
    return clusters


def _cluster_rank_evidence(observed, anchors, active_indices, clusters, config):
    normalised = _normalise_profiles(anchors[:, active_indices])
    spectra = np.maximum(np.linalg.lstsq(normalised, observed, rcond=None)[0], 0.0)
    index_to_position = {index: pos for pos, index in enumerate(active_indices)}
    evidence = []
    for cluster in clusters:
        fixed = [index for index in active_indices if index not in cluster]
        residual = observed.copy()
        if fixed:
            positions = [index_to_position[index] for index in fixed]
            residual -= normalised[:, positions] @ spectra[positions, :]
        cluster_positions = [index_to_position[index] for index in cluster]
        local_profile = np.max(normalised[:, cluster_positions], axis=1)
        local_rows = local_profile > (
            config.active_profile_threshold * max(float(np.max(local_profile)), 1e-12)
        )
        local_residual = residual[local_rows] if np.sum(local_rows) >= 4 else residual
        ratios = _rank_ratios(local_residual, config.rank_row_fraction)
        ratio21 = ratios[0] if ratios else None
        evidence.append({
            "members": list(cluster),
            "singular_value_ratios": ratios,
            "sigma2_over_sigma1": ratio21,
            # This ratio is descriptive evidence.  It is deliberately not
            # converted to a component decision by a fixed threshold.
            "rank_class": "continuous_evidence" if ratio21 is not None else "indeterminate",
        })
    return evidence


def _candidate_subsets(active_indices, clusters):
    disputed = set(index for cluster in clusters for index in cluster)
    fixed = tuple(index for index in active_indices if index not in disputed)
    cluster_choices = []
    for cluster in clusters:
        choices = []
        # A collapsed group can consist entirely of model overpredictions
        # around a separate, undisputed profile. Include the empty choice so
        # the data can compare that possibility instead of forcing at least
        # one member of every disputed group into the final model.
        for count in range(0, len(cluster) + 1):
            choices.extend(combinations(cluster, count))
        cluster_choices.append(choices)
    if not cluster_choices:
        return [tuple(active_indices)]
    candidates = {
        tuple(sorted(fixed + tuple(index for choice in choices for index in choice)))
        for choices in product(*cluster_choices)
    }
    # An empty chromatographic model is undefined. When every active profile
    # belongs to a disputed group, retain all non-empty subsets only.
    candidates = {indices for indices in candidates if indices}
    return sorted(candidates, key=lambda item: (len(item), item))


def _shortlist_candidates(observed, anchors, candidates, config):
    scored = []
    for indices in candidates:
        scored.append((indices, _fixed_profile_score(observed, anchors[:, indices])))
    full_count = max(len(indices) for indices, _ in scored)
    keep = set()
    for count in sorted({len(indices) for indices, _ in scored}):
        group = sorted(
            (item for item in scored if len(item[0]) == count),
            key=lambda item: item[1], reverse=True,
        )
        limit = config.max_candidates_per_count
        if count >= full_count - 1:
            limit = len(group)
        keep.update(indices for indices, _ in group[:limit])
    return [(indices, score) for indices, score in scored if indices in keep]


def _frr_candidate_targets(evaluated, config):
    """Select a small, representative set of candidates for FRR.

    FRR is substantially more expensive than the masked MCR and validation
    stages.  Evaluating every shortlisted rotation therefore repeats nearly
    identical boundary searches without adding useful model-selection
    evidence.  The target set keeps the best candidate at each component
    count, the overall validation winner, and candidates with independent
    peak-shape evidence.  The configured limit is a cap on actual FRR calls,
    not on the number of candidates retained for later diagnostics.
    """
    if not evaluated:
        return set()
    limit = max(1, int(config.frr_candidate_limit))
    by_count = {}
    for item in evaluated:
        count = int(item["component_count"])
        current = by_count.get(count)
        if current is None or (
                float(item["cv_normalised_sse_mean"]),
                tuple(item["indices"]),
        ) < (
                float(current["cv_normalised_sse_mean"]),
                tuple(current["indices"]),
        ):
            by_count[count] = item

    ranked = sorted(
        evaluated,
        key=lambda item: (
            float(item["cv_normalised_sse_mean"]),
            -int(item["component_count"]),
            tuple(item["indices"]),
        ),
    )
    evidence_ranked = sorted(
        [
            item for item in evaluated
            if bool(
                (item.get("resolved_profile_shape_test") or {}).get(
                    "supported"
                )
                or (item.get("independent_peak_shape_test") or {}).get(
                    "supported"
                )
            )
        ],
        key=lambda item: (
            float(item["cv_normalised_sse_mean"]),
            -int(item["component_count"]),
            tuple(item["indices"]),
        ),
    )

    # First reserve the overall winner, the simplest and most complex models,
    # and then the best candidate for each component count.  This preserves
    # the FRR comparisons used by the downstream parsimony and confirmation
    # rules while avoiding calls for near-duplicate rotations.
    ordered = []
    ordered.extend(ranked[:1])
    ordered.extend(sorted(by_count.values(), key=lambda item: (
        int(item["component_count"]),
        float(item["cv_normalised_sse_mean"]),
        tuple(item["indices"]),
    ))[:1])
    ordered.extend(sorted(by_count.values(), key=lambda item: (
        -int(item["component_count"]),
        float(item["cv_normalised_sse_mean"]),
        tuple(item["indices"]),
    ))[:1])
    ordered.extend(sorted(by_count.values(), key=lambda item: (
        int(item["component_count"]),
        float(item["cv_normalised_sse_mean"]),
        tuple(item["indices"]),
    )))
    ordered.extend(evidence_ranked)
    ordered.extend(ranked)

    targets = set()
    for item in ordered:
        key = tuple(item["indices"])
        if key in targets:
            continue
        targets.add(key)
        if len(targets) >= limit:
            break
    return targets


def _build_cv_splits(observed, config):
    """Build common repeated masks so candidate errors are paired."""
    active_columns = np.where(np.sum(observed, axis=0) > 0)[0]
    if active_columns.size == 0:
        raise ValueError("the observed matrix has no active mass channels")
    folds = max(2, int(config.cv_folds))
    repeats = max(1, int(config.cv_repeats))
    rng = np.random.default_rng(int(config.random_seed))
    splits = []
    for repeat in range(repeats):
        row_groups = rng.permutation(observed.shape[0]) % folds
        column_groups = rng.permutation(active_columns.size) % folds
        for fold in range(folds):
            held_active = (
                row_groups[:, None] + column_groups[None, :]
            ) % folds == fold
            held = np.zeros_like(observed, dtype=bool)
            held[:, active_columns] = held_active
            splits.append({
                "repeat": repeat,
                "fold": fold,
                "held": held,
            })
    return splits


def _masked_cv(observed, anchor, config, splits):
    errors = []
    split_labels = []
    for split in splits:
        held = split["held"]
        weights = np.ones_like(observed, dtype=float)
        weights[held] = 0.0
        _, _, fitted = _fit_weighted_mcr(
            observed, anchor, config, weights=weights
        )
        denominator = max(float(np.sum(observed[held] ** 2)), 1e-12)
        errors.append(float(np.sum((observed[held] - fitted[held]) ** 2) / denominator))
        split_labels.append({
            "repeat": int(split["repeat"]),
            "fold": int(split["fold"]),
        })
    mean = float(np.mean(errors))
    standard_error = float(np.std(errors, ddof=1) / np.sqrt(len(errors))) if len(errors) > 1 else 0.0
    return mean, standard_error, errors, split_labels


def _fit_fixed_profile_spectra(X, C, train):
    """Fit spectra, grouping channels that share the same training rows."""
    spectra = np.zeros((C.shape[1], X.shape[1]), dtype=float)
    groups = {}
    for channel in range(X.shape[1]):
        rows = train[:, channel]
        if np.sum(rows) < C.shape[1]:
            continue
        key = np.packbits(rows).tobytes()
        groups.setdefault(key, []).append(channel)

    for channels in groups.values():
        rows = train[:, channels[0]]
        right_hand_sides = X[rows][:, channels]
        spectra[:, channels] = np.maximum(
            np.linalg.lstsq(C[rows], right_hand_sides, rcond=None)[0],
            0.0,
        )
    return spectra


def _fixed_profile_cv_errors(observed, profiles, splits):
    """Return paired validation errors for fixed chromatographic profiles.

    The spectra are fitted on the unmasked scans for each split and evaluated
    on the held scans.  This lightweight score is used only inside the
    residual null test, so it does not replace the full MCR cross validation
    used for model selection.
    """
    X = np.maximum(np.asarray(observed, dtype=float), 0.0)
    C = _normalise_profiles(profiles)
    if X.ndim != 2 or C.ndim != 2 or X.shape[0] != C.shape[0] or C.shape[1] == 0:
        return []
    errors = []
    for split in splits:
        held = np.asarray(split["held"], dtype=bool)
        train = ~held
        spectra = _fit_fixed_profile_spectra(X, C, train)
        fitted = C @ spectra
        denominator = max(float(np.sum(X[held] ** 2)), 1e-12)
        errors.append(float(np.sum((X[held] - fitted[held]) ** 2) / denominator))
    return errors


def _residual_null_test(observed, simpler_profiles, complex_profiles, splits,
                        config, seed_offset=0):
    """Test whether an extra profile improves validation beyond a null model.

    The null model is the simpler fixed-profile reconstruction plus residuals
    resampled independently by mass channel.  The observed and null paired
    validation improvements use the same masks.  A candidate is supported only
    when its observed improvement is positive, exceeds the configured minimum,
    and is above the requested upper quantile of the null improvements.
    """
    if not config.enable_residual_null_test:
        return {
            "enabled": False,
            "supported": False,
            "reason": "disabled",
        }
    X = np.maximum(np.asarray(observed, dtype=float), 0.0)
    simple = _normalise_profiles(simpler_profiles)
    complex_ = _normalise_profiles(complex_profiles)
    if (X.ndim != 2 or simple.ndim != 2 or complex_.ndim != 2
            or X.shape[0] != simple.shape[0]
            or X.shape[0] != complex_.shape[0]
            or complex_.shape[1] <= simple.shape[1]):
        return {
            "enabled": True,
            "supported": False,
            "reason": "invalid_profile_dimensions",
        }

    simple_spectra = np.maximum(
        np.linalg.lstsq(simple, X, rcond=None)[0], 0.0
    )
    simple_fit = simple @ simple_spectra
    residual = X - simple_fit
    simple_errors = _fixed_profile_cv_errors(X, simple, splits)
    complex_errors = _fixed_profile_cv_errors(X, complex_, splits)
    if (not simple_errors or len(simple_errors) != len(complex_errors)
            or not np.all(np.isfinite(simple_errors + complex_errors))):
        return {
            "enabled": True,
            "supported": False,
            "reason": "observed_cv_failed",
        }
    observed_differences = np.asarray(simple_errors) - np.asarray(complex_errors)
    observed_improvement = float(np.mean(observed_differences))

    runs = max(8, int(config.residual_null_runs))
    percentile = min(max(float(config.residual_null_percentile), 0.5), 0.999)
    rng = np.random.default_rng(int(config.random_seed) + int(seed_offset))
    null_improvements = []
    for _ in range(runs):
        null_residual = np.zeros_like(residual)
        for channel in range(residual.shape[1]):
            null_residual[:, channel] = residual[
                rng.permutation(residual.shape[0]), channel
            ]
        null_observed = np.maximum(simple_fit + null_residual, 0.0)
        null_simple = _fixed_profile_cv_errors(
            null_observed, simple, splits
        )
        null_complex = _fixed_profile_cv_errors(
            null_observed, complex_, splits
        )
        if (len(null_simple) == len(simple_errors)
                and np.all(np.isfinite(null_simple + null_complex))):
            null_improvements.append(float(
                np.mean(np.asarray(null_simple) - np.asarray(null_complex))
            ))
    if not null_improvements:
        return {
            "enabled": True,
            "supported": False,
            "reason": "null_cv_failed",
            "observed_improvement": observed_improvement,
        }
    null_cutoff = float(np.quantile(null_improvements, percentile))
    supported = bool(
        observed_improvement >= float(config.residual_null_min_improvement)
        and observed_improvement > null_cutoff
    )
    return {
        "enabled": True,
        "supported": supported,
        "reason": "observed_improvement_above_null" if supported
        else "improvement_not_above_null",
        "observed_improvement": observed_improvement,
        "null_percentile": percentile,
        "null_cutoff": null_cutoff,
        "null_mean": float(np.mean(null_improvements)),
        "null_sd": float(np.std(null_improvements, ddof=1))
        if len(null_improvements) > 1 else 0.0,
        "null_runs": len(null_improvements),
    }


def _trace_peak_at_anchor(trace, anchor_apex, config):
    """Test for a complete local maximum near one predicted apex."""
    y = _smooth_topology_trace(np.maximum(
        np.asarray(trace, dtype=float).reshape(-1), 0.0
    ))
    if y.size < 5 or not np.any(y > 0.0):
        return {
            "supported": False,
            "reason": "insufficient_residual_trace",
        }
    radius = max(1, int(config.apex_window))
    expected = int(np.clip(int(anchor_apex), 1, y.size - 2))
    start = max(1, expected - radius)
    end = min(y.size - 2, expected + radius)
    apex = max(range(start, end + 1), key=lambda scan: y[scan])
    local_maximum = bool(
        y[apex] >= y[apex - 1] and y[apex] >= y[apex + 1]
    )
    height = max(float(y[apex]), 1e-12)
    left_minimum = float(np.min(y[:apex + 1]))
    right_minimum = float(np.min(y[apex:]))
    prominence_fraction = float(
        min(height - left_minimum, height - right_minimum) / height
    )
    left_boundary_fraction = float(y[0] / height)
    right_boundary_fraction = float(y[-1] / height)
    complete = bool(
        left_boundary_fraction
        <= float(config.resolved_profile_max_boundary_fraction)
        and right_boundary_fraction
        <= float(config.resolved_profile_max_boundary_fraction)
    )
    supported = bool(
        local_maximum
        and complete
        and prominence_fraction
        >= float(config.peak_shape_min_shoulder_fraction)
    )
    return {
        "supported": supported,
        "reason": (
            "residual_peak_matches_added_component" if supported
            else "no_complete_residual_peak_at_added_component"
        ),
        "expected_apex_scan": expected,
        "observed_apex_scan": int(apex),
        "apex_distance_scans": int(abs(apex - expected)),
        "prominence_fraction": prominence_fraction,
        "left_boundary_fraction": left_boundary_fraction,
        "right_boundary_fraction": right_boundary_fraction,
        "complete_peak": complete,
    }


def _residual_component_peak_test(observed, simpler_model, complex_model,
                                  anchors, simpler_indices, complex_indices,
                                  config, seed_offset=0):
    """Seek a reproducible peak for an added component in model residuals.

    The common-ion consensus can hide a low-abundance impurity. This test
    removes the simpler MCR reconstruction, selects mass channels that are
    characteristic of the added component in the complex model, and asks
    whether their positive residual EIC has a complete apex at the model
    predicted retention position.
    """
    X = np.maximum(np.asarray(observed, dtype=float), 0.0)
    simple_C, simple_S = simpler_model
    complex_C, complex_S = complex_model
    simple_indices = tuple(int(value) for value in simpler_indices)
    full_indices = tuple(int(value) for value in complex_indices)
    added = [value for value in full_indices if value not in simple_indices]
    if (X.ndim != 2 or len(added) != 1
            or simple_C.shape[0] != X.shape[0]
            or complex_C.shape[0] != X.shape[0]):
        return {
            "enabled": True,
            "supported": False,
            "reason": "residual_component_test_requires_one_added_component",
        }
    added_global = int(added[0])
    added_local = full_indices.index(added_global)
    residual = np.maximum(X - simple_C @ simple_S, 0.0)
    extra_spectrum = np.maximum(complex_S[added_local], 0.0)
    competing = (
        np.max(np.maximum(simple_S, 0.0), axis=0)
        if simple_S.shape[0] else np.zeros_like(extra_spectrum)
    )
    specificity = extra_spectrum / np.maximum(
        extra_spectrum + competing, 1e-12
    )
    residual_area = np.trapz(residual, axis=0)
    channel_score = extra_spectrum * specificity * np.sqrt(
        np.maximum(residual_area, 0.0)
    )
    active = np.where(
        (extra_spectrum >= 0.01 * max(float(np.max(extra_spectrum)), 1e-12))
        & (residual_area > 0.0)
    )[0]
    if active.size < 3:
        return {
            "enabled": True,
            "supported": False,
            "reason": "insufficient_component_specific_residual_ions",
            "added_component_index": added_global,
        }
    if active.size > 32:
        active = active[np.argsort(channel_score[active])[-32:]]
    heights = np.max(residual[:, active], axis=0)
    valid = heights > 0.0
    active = active[valid]
    heights = heights[valid]
    if active.size < 3:
        return {
            "enabled": True,
            "supported": False,
            "reason": "insufficient_component_specific_residual_ions",
            "added_component_index": added_global,
        }
    traces = residual[:, active] / np.maximum(heights, 1e-12)[None, :]
    weights = np.sqrt(np.maximum(channel_score[active], 0.0))
    if not np.any(weights > 0.0):
        weights = np.ones(active.size, dtype=float)
    consensus = _weighted_consensus_trace(traces, weights)
    anchor_apex = int(np.argmax(anchors[:, added_global]))
    observed_topology = _trace_peak_at_anchor(
        consensus, anchor_apex, config
    )
    rng = np.random.default_rng(int(config.random_seed) + int(seed_offset))
    runs = max(8, int(config.peak_shape_bootstrap_runs))
    bootstrap_support = []
    for _ in range(runs):
        sampled = rng.integers(0, active.size, size=active.size)
        trace = _weighted_consensus_trace(
            traces[:, sampled], weights[sampled]
        )
        bootstrap_support.append(bool(
            _trace_peak_at_anchor(trace, anchor_apex, config)["supported"]
        ))
    support_rate = float(np.mean(bootstrap_support))
    supported = bool(
        observed_topology["supported"]
        and support_rate
        >= float(config.peak_shape_min_topology_support_rate)
    )
    return {
        "enabled": True,
        "supported": supported,
        "reason": (
            "reproducible_component_specific_residual_peak" if supported
            else "no_reproducible_component_specific_residual_peak"
        ),
        "added_component_index": added_global,
        "selected_ion_count": int(active.size),
        "selected_mz_channel_indices": [int(value) for value in active],
        "observed_topology": observed_topology,
        "bootstrap_support_rate": support_rate,
        "minimum_support_rate": float(
            config.peak_shape_min_topology_support_rate
        ),
        "bootstrap_runs": runs,
    }


def _normalised_ion_traces(observed):
    """Return informative, baseline-corrected ion traces and their weights."""
    X = np.maximum(np.asarray(observed, dtype=float), 0.0)
    if X.ndim != 2 or X.shape[0] < 6 or X.shape[1] == 0:
        return None, None
    # Remove residual sloping background using robust endpoint levels.  This
    # copy is used only for component-number evidence; the observed matrix
    # used by MCR and quantitative integration is not altered.
    edge_count = max(2, min(4, X.shape[0] // 5))
    left_level = np.median(X[:edge_count], axis=0)
    right_level = np.median(X[-edge_count:], axis=0)
    position = np.linspace(0.0, 1.0, X.shape[0])[:, None]
    local_baseline = (
        left_level[None, :] * (1.0 - position)
        + right_level[None, :] * position
    )
    X = np.maximum(X - local_baseline, 0.0)
    heights = np.max(X, axis=0)
    areas = np.trapz(X, axis=0)
    if not np.any(heights > 0) or not np.any(areas > 0):
        return None, None
    active = (
        (heights >= 0.01 * max(float(np.max(heights)), 1e-12))
        & (areas >= 0.005 * max(float(np.max(areas)), 1e-12))
    )
    indices = np.where(active)[0]
    if indices.size < 3:
        indices = np.argsort(areas)[-min(3, X.shape[1]):]
    # Limit the nonlinear bootstrap to the most informative channels. This
    # also prevents many nearly empty channels from dominating the consensus.
    if indices.size > 64:
        indices = indices[np.argsort(areas[indices])[-64:]]
    traces = X[:, indices] / np.maximum(heights[indices], 1e-12)[None, :]
    weights = np.sqrt(
        np.maximum(areas[indices], 0.0)
        / max(float(np.max(areas[indices])), 1e-12)
    )
    return traces, weights


def _weighted_consensus_trace(traces, weights):
    traces = np.asarray(traces, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if traces.ndim != 2 or traces.shape[1] != weights.size:
        return None
    denominator = max(float(np.sum(weights)), 1e-12)
    trace = np.sum(traces * weights[None, :], axis=1) / denominator
    maximum = float(np.max(trace))
    return trace / maximum if maximum > 1e-12 else None


def _resolved_profile_shape_test(chromatograms, config):
    """Check that every resolved profile is a complete chromatographic peak.

    Independent structure in the observed ion traces establishes that an
    additional peak may exist, but it does not establish that a particular
    MCR rotation represents that peak. A retained resolved profile must have
    an interior apex and must fall below half height at both DeepCS segment
    boundaries. This is the same half-peak principle used by the historical
    ``dishalfremove`` step, applied before model-selection evidence can protect
    an invalid rotation.
    """
    C = np.maximum(np.asarray(chromatograms, dtype=float), 0.0)
    if C.ndim != 2 or C.shape[0] < 3 or C.shape[1] == 0:
        return {
            "supported": False,
            "reason": "invalid_resolved_profile_matrix",
            "components": [],
        }
    maximum_boundary_fraction = float(
        config.resolved_profile_max_boundary_fraction
    )
    details = []
    for component in range(C.shape[1]):
        profile = C[:, component]
        maximum = float(np.max(profile))
        if maximum <= 1e-12:
            apex = None
            left_fraction = None
            right_fraction = None
            interior_apex = False
            complete = False
        else:
            apex = int(np.argmax(profile))
            left_fraction = float(profile[0] / maximum)
            right_fraction = float(profile[-1] / maximum)
            interior_apex = bool(0 < apex < profile.size - 1)
            complete = bool(
                interior_apex
                and left_fraction <= maximum_boundary_fraction
                and right_fraction <= maximum_boundary_fraction
            )
        details.append({
            "component_index": int(component),
            "apex_scan": apex,
            "left_boundary_fraction": left_fraction,
            "right_boundary_fraction": right_fraction,
            "interior_apex": interior_apex,
            "complete_peak": complete,
        })
    supported = bool(all(item["complete_peak"] for item in details))
    return {
        "supported": supported,
        "reason": (
            "all_resolved_profiles_are_complete_peaks" if supported
            else "resolved_profile_is_incomplete_at_segment_boundary"
        ),
        "maximum_boundary_fraction": maximum_boundary_fraction,
        "components": details,
    }


def _smooth_topology_trace(trace):
    """Smooth scan-scale fluctuations without changing peak locations."""
    trace = np.asarray(trace, dtype=float).reshape(-1)
    if trace.size < 5:
        return trace.copy()
    window = min(5, trace.size if trace.size % 2 else trace.size - 1)
    if window < 5:
        return trace.copy()
    return savgol_filter(trace, window_length=window, polyorder=2, mode="interp")


def _observed_peak_topology(trace, complex_fit, config):
    """Match each fitted component to an observed apex or shoulder.

    A second fitted basis function is not by itself a second chromatographic
    peak.  Each fitted centre must coincide with either a local apex or a
    reproducible change in slope.  A trailing shoulder has a temporarily less
    negative slope followed by renewed descent; a leading shoulder has the
    corresponding temporarily smaller positive slope followed by renewed
    ascent.
    """
    y = _smooth_topology_trace(trace)
    centres = np.asarray(complex_fit.get("centres", []), dtype=float)
    if y.size < 5 or centres.size == 0 or not np.all(np.isfinite(centres)):
        return {
            "supported": False,
            "reason": "insufficient_topology_data",
            "components": [],
        }
    derivative = np.diff(y)
    global_apex = int(np.argmax(y))
    search_radius = max(1, int(config.peak_shape_topology_window_scans))
    min_shoulder = float(config.peak_shape_min_shoulder_fraction)
    min_widths = float(config.peak_shape_min_observed_widths)
    sigma_left = np.asarray(complex_fit.get("sigma_left", []), dtype=float)
    sigma_right = np.asarray(complex_fit.get("sigma_right", []), dtype=float)
    widths_available = bool(
        sigma_left.size == centres.size
        and sigma_right.size == centres.size
        and np.all(sigma_left > 0.0)
        and np.all(sigma_right > 0.0)
    )
    details = []
    for component, centre in enumerate(centres):
        centre_scan = int(np.clip(round(float(centre)), 1, y.size - 2))
        search_start = max(1, centre_scan - search_radius)
        search_end = min(y.size - 2, centre_scan + search_radius)
        apex_scans = [
            scan for scan in range(search_start, search_end + 1)
            if y[scan] >= y[scan - 1] and y[scan] >= y[scan + 1]
        ]
        apex_scan = min(
            apex_scans,
            key=lambda scan: abs(scan - centre),
        ) if apex_scans else None

        left_boundary = (
            int(np.floor(0.5 * (centres[component - 1] + centre)))
            if component > 0 else 0
        )
        right_boundary = (
            int(np.ceil(0.5 * (centre + centres[component + 1])))
            if component + 1 < centres.size else y.size - 1
        )
        slope_start = max(left_boundary + 1, centre_scan - search_radius)
        slope_end = min(right_boundary - 1, centre_scan + search_radius)
        shoulder_strength = -np.inf
        shoulder_scan = None
        shoulder_side = "leading" if centre < global_apex else "trailing"
        for scan in range(slope_start, slope_end + 1):
            before = derivative[max(left_boundary, scan - 4):scan]
            after = derivative[scan + 1:min(right_boundary, scan + 5)]
            if before.size == 0 or after.size == 0:
                continue
            if shoulder_side == "leading":
                # A leading shoulder interrupts an otherwise positive slope.
                strength = min(
                    float(np.max(before) - derivative[scan]),
                    float(np.max(after) - derivative[scan]),
                )
            else:
                # A trailing shoulder interrupts an otherwise negative slope.
                strength = min(
                    float(derivative[scan] - np.min(before)),
                    float(derivative[scan] - np.min(after)),
                )
            if strength > shoulder_strength:
                shoulder_strength = strength
                shoulder_scan = int(scan)
        if not np.isfinite(shoulder_strength):
            shoulder_strength = 0.0
        shoulder_supported = bool(shoulder_strength >= min_shoulder)
        if widths_available:
            observed_left_widths = float(centre / sigma_left[component])
            observed_right_widths = float(
                (y.size - 1 - centre) / sigma_right[component]
            )
            complete = bool(
                observed_left_widths >= min_widths
                and observed_right_widths >= min_widths
            )
        else:
            observed_left_widths = None
            observed_right_widths = None
            complete = True
        component_supported = bool(
            complete and (apex_scan is not None or shoulder_supported)
        )
        details.append({
            "component_index": int(component),
            "fitted_centre_scan": float(centre),
            "local_apex_supported": bool(apex_scan is not None),
            "local_apex_scan": None if apex_scan is None else int(apex_scan),
            "shoulder_supported": shoulder_supported,
            "shoulder_side": shoulder_side,
            "shoulder_scan": shoulder_scan,
            "shoulder_strength_fraction": float(shoulder_strength),
            "observed_left_widths": observed_left_widths,
            "observed_right_widths": observed_right_widths,
            "peak_shape_complete": complete,
            "supported": component_supported,
        })
    supported = bool(details and all(item["supported"] for item in details))
    return {
        "supported": supported,
        "reason": "all_components_match_observed_topology" if supported
        else "fitted_component_without_observed_apex_or_shoulder",
        "global_apex_scan": global_apex,
        "minimum_shoulder_fraction": min_shoulder,
        "minimum_observed_widths": min_widths,
        "components": details,
    }


def _asymmetric_gaussian_sum(scans, parameters, component_count):
    fitted = np.full_like(scans, float(parameters[-1]), dtype=float)
    for component in range(component_count):
        offset = 4 * component
        centre, sigma_left, sigma_right, amplitude = parameters[offset:offset + 4]
        sigma = np.where(scans <= centre, sigma_left, sigma_right)
        fitted += amplitude * np.exp(
            -0.5 * ((scans - centre) / np.maximum(sigma, 1e-6)) ** 2
        )
    return fitted


def _asymmetric_peak_fit(trace, anchors, centre_limits=None):
    """Fit a sum of asymmetric Gaussian peaks to one consensus trace."""
    y = np.maximum(np.asarray(trace, dtype=float).reshape(-1), 0.0)
    A = np.maximum(np.asarray(anchors, dtype=float), 0.0)
    if (A.ndim != 2 or A.shape[0] != y.size or A.shape[1] == 0
            or y.size <= 4 * A.shape[1] + 1 or np.max(y) <= 1e-12):
        return None
    order = np.argsort(np.argmax(A, axis=0))
    A = A[:, order]
    apices = np.argmax(A, axis=0).astype(float)
    if apices.size > 1 and np.any(np.diff(apices) < 1.0):
        return None
    scans = np.arange(y.size, dtype=float)
    count = A.shape[1]
    centre_lower = np.zeros(count, dtype=float)
    centre_upper = np.full(count, float(y.size - 1))
    if centre_limits is not None:
        domain_start, domain_end = map(float, centre_limits)
        domain_start = float(np.clip(domain_start, 0.0, y.size - 1.0))
        domain_end = float(np.clip(domain_end, domain_start, y.size - 1.0))
        centre_lower[:] = domain_start
        centre_upper[:] = domain_end
    if count > 1:
        midpoints = 0.5 * (apices[:-1] + apices[1:])
        centre_upper[:-1] = np.minimum(centre_upper[:-1], midpoints)
        centre_lower[1:] = np.maximum(centre_lower[1:], midpoints)

    initial = []
    lower = []
    upper = []
    y_max = max(float(np.max(y)), 1e-12)
    for component in range(count):
        anchor = A[:, component]
        height = float(np.max(anchor))
        support = np.where(anchor >= 0.5 * height)[0] if height > 0 else np.array([])
        fwhm = float(max(2, support[-1] - support[0] + 1)) if support.size else 4.0
        sigma = float(np.clip(fwhm / 2.355, 0.75, max(1.0, y.size / 3.0)))
        initial.extend([
            float(apices[component]), sigma, sigma,
            max(float(y[int(apices[component])] - np.min(y)), 0.1 * y_max),
        ])
        lower.extend([
            float(centre_lower[component]), 0.5, 0.5, 0.0,
        ])
        upper.extend([
            float(centre_upper[component]), max(2.0, y.size / 2.0),
            max(2.0, y.size / 2.0), 2.0 * y_max,
        ])
    initial.append(max(0.0, float(np.min(y))))
    lower.append(0.0)
    upper.append(y_max)
    initial = np.asarray(initial, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    initial = np.minimum(np.maximum(initial, lower + 1e-6), upper - 1e-6)
    try:
        result = least_squares(
            lambda parameters: (
                _asymmetric_gaussian_sum(scans, parameters, count) - y
            ),
            initial,
            bounds=(lower, upper),
            max_nfev=500,
        )
    except Exception:
        return None
    if not result.success or not np.all(np.isfinite(result.x)):
        return None
    fitted = _asymmetric_gaussian_sum(scans, result.x, count)
    rss = max(float(np.sum((y - fitted) ** 2)), 1e-15)
    parameter_count = 4 * count + 1
    bic = float(y.size * np.log(rss / y.size) + parameter_count * np.log(y.size))
    centres = np.asarray([result.x[4 * i] for i in range(count)])
    left_widths = np.asarray([result.x[4 * i + 1] for i in range(count)])
    right_widths = np.asarray([result.x[4 * i + 2] for i in range(count)])
    amplitudes = np.asarray([result.x[4 * i + 3] for i in range(count)])
    areas = amplitudes * (left_widths + right_widths)
    area_fractions = areas / max(float(np.sum(areas)), 1e-12)
    centre_margin = np.minimum(centres - centre_lower, centre_upper - centres)
    width_upper = max(2.0, y.size / 2.0)
    boundary_hit = bool(
        np.any(centre_margin <= 0.25)
        or np.any(left_widths <= 0.55)
        or np.any(right_widths <= 0.55)
        or np.any(left_widths >= width_upper - 0.1)
        or np.any(right_widths >= width_upper - 0.1)
    )
    return {
        "component_count": int(count),
        "bic": bic,
        "rss": rss,
        "centres": centres.tolist(),
        "sigma_left": left_widths.tolist(),
        "sigma_right": right_widths.tolist(),
        "area_fractions": area_fractions.tolist(),
        "minimum_centre_separation": (
            float(np.min(np.diff(centres))) if count > 1 else None
        ),
        "boundary_hit": boundary_hit,
    }


def _independent_peak_shape_test(observed, simpler_anchors, complex_anchors,
                                 config, seed_offset=0,
                                 centre_limits=None):
    """Test whether a common-ion chromatogram contains an extra peak shape."""
    if not config.enable_peak_shape_confirmation:
        return {"enabled": False, "supported": False, "reason": "disabled"}
    traces, weights = _normalised_ion_traces(observed)
    if traces is None or traces.shape[1] < 3:
        return {
            "enabled": True,
            "supported": False,
            "reason": "insufficient_informative_ions",
        }
    consensus = _weighted_consensus_trace(traces, weights)
    simple_fit = _asymmetric_peak_fit(
        consensus, simpler_anchors, centre_limits=centre_limits
    )
    complex_fit = _asymmetric_peak_fit(
        consensus, complex_anchors, centre_limits=centre_limits
    )
    if simple_fit is None or complex_fit is None:
        return {
            "enabled": True,
            "supported": False,
            "reason": "asymmetric_peak_fit_failed",
        }
    observed_bic_gain = float(simple_fit["bic"] - complex_fit["bic"])
    observed_topology = _observed_peak_topology(
        consensus, complex_fit, config
    )
    rng = np.random.default_rng(int(config.random_seed) + int(seed_offset))
    runs = max(8, int(config.peak_shape_bootstrap_runs))
    bic_gains = []
    minimum_fractions = []
    separations = []
    boundary_hits = []
    topology_support = []
    for _ in range(runs):
        sampled = rng.integers(0, traces.shape[1], size=traces.shape[1])
        bootstrap_trace = _weighted_consensus_trace(
            traces[:, sampled], weights[sampled]
        )
        bootstrap_simple = _asymmetric_peak_fit(
            bootstrap_trace, simpler_anchors, centre_limits=centre_limits
        )
        bootstrap_complex = _asymmetric_peak_fit(
            bootstrap_trace, complex_anchors, centre_limits=centre_limits
        )
        if bootstrap_simple is None or bootstrap_complex is None:
            continue
        bic_gains.append(float(
            bootstrap_simple["bic"] - bootstrap_complex["bic"]
        ))
        minimum_fractions.append(float(
            np.min(bootstrap_complex["area_fractions"])
        ))
        if bootstrap_complex["minimum_centre_separation"] is not None:
            separations.append(float(
                bootstrap_complex["minimum_centre_separation"]
            ))
        boundary_hits.append(bool(bootstrap_complex["boundary_hit"]))
        topology_support.append(bool(
            _observed_peak_topology(
                bootstrap_trace, bootstrap_complex, config
            )["supported"]
        ))
    if len(bic_gains) < max(5, runs // 2):
        return {
            "enabled": True,
            "supported": False,
            "reason": "peak_shape_bootstrap_failed",
            "observed_bic_gain": observed_bic_gain,
            "observed_topology": observed_topology,
            "simple_fit": simple_fit,
            "complex_fit": complex_fit,
        }
    alpha = 1.0 - min(max(float(config.peak_shape_confidence_level), 0.5), 0.999)
    bic_lower = float(np.quantile(bic_gains, alpha))
    fraction_lower = float(np.quantile(minimum_fractions, alpha))
    separation_lower = float(np.quantile(separations, alpha)) if separations else 0.0
    boundary_rate = float(np.mean(boundary_hits)) if boundary_hits else 1.0
    topology_support_rate = float(np.mean(topology_support)) if topology_support else 0.0
    # A 50 percent bootstrap recurrence is deliberately distinct from the
    # BIC confidence level.  Compounds with selective spectra may be absent
    # from many resampled ion sets even though the observed consensus has two
    # clear peaks.  The observed topology remains mandatory, while this rate
    # verifies that the feature is not carried by only an exceptional draw.
    minimum_topology_rate = float(
        config.peak_shape_min_topology_support_rate
    )
    supported = bool(
        observed_bic_gain > 0.0
        and bic_lower > 0.0
        and fraction_lower >= float(config.peak_shape_min_area_fraction)
        and separation_lower >= float(config.peak_shape_min_separation_scans)
        and boundary_rate < 0.5
        and observed_topology["supported"]
        and topology_support_rate >= minimum_topology_rate
    )
    return {
        "enabled": True,
        "supported": supported,
        "reason": "independent_peak_shape_supported" if supported
        else "no_reproducible_independent_peak_shape",
        "informative_ion_count": int(traces.shape[1]),
        "observed_bic_gain": observed_bic_gain,
        "bootstrap_bic_gain_lower": bic_lower,
        "bootstrap_bic_gain_median": float(np.median(bic_gains)),
        "bootstrap_minimum_area_fraction_lower": fraction_lower,
        "bootstrap_minimum_separation_lower": separation_lower,
        "bootstrap_boundary_hit_rate": boundary_rate,
        "observed_topology": observed_topology,
        "bootstrap_topology_support_rate": topology_support_rate,
        "minimum_topology_support_rate": minimum_topology_rate,
        "bootstrap_runs": len(bic_gains),
        "simple_fit": simple_fit,
        "complex_fit": complex_fit,
    }


def _paired_comparison(candidate_errors, reference_errors, config, seed_offset=0):
    """Compare two candidates using paired masked-validation errors."""
    candidate_errors = np.asarray(candidate_errors, dtype=float)
    reference_errors = np.asarray(reference_errors, dtype=float)
    if candidate_errors.shape != reference_errors.shape or candidate_errors.size == 0:
        raise ValueError("paired validation errors must have equal nonzero length")
    differences = candidate_errors - reference_errors
    rng = np.random.default_rng(int(config.random_seed) + int(seed_offset))
    runs = max(200, int(config.paired_bootstrap_runs))
    sampled = rng.integers(0, differences.size, size=(runs, differences.size))
    bootstrap_means = np.mean(differences[sampled], axis=1)
    alpha = 1.0 - float(config.paired_confidence_level)
    alpha = min(max(alpha, 1e-6), 0.5)
    lower, upper = np.quantile(
        bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    return {
        "mean_error_difference": float(np.mean(differences)),
        "standard_error_difference": (
            float(np.std(differences, ddof=1) / np.sqrt(differences.size))
            if differences.size > 1 else 0.0
        ),
        "confidence_interval": [float(lower), float(upper)],
        "probability_candidate_better": float(np.mean(bootstrap_means < 0.0)),
        "probability_reference_better": float(np.mean(bootstrap_means > 0.0)),
        "paired_differences": differences.tolist(),
    }


def _decisive_multicomponent_candidate(evaluated, config):
    """Return a stable 3+ component model decisively favoured by validation.

    Independent peak-shape fitting can be conservative for strongly
    coeluting compounds because the additional chromatographic apex may be a
    shoulder rather than a separately fitted maximum.  That failure should
    not force a smaller model when the complete model is reproducible and its
    masked-validation error is significantly lower than every simpler model.

    The rule is intentionally unavailable to one-versus-two component
    decisions, where an extra profile can readily absorb baseline or peak
    asymmetry.  No fixed R2 or error-difference cutoff is used.  Evidence is
    assessed from the paired bootstrap confidence interval for the validation
    errors, together with residual randomisation, perturbation stability and
    complete resolved profiles.  A separately resolved residual apex is not
    mandatory because that is precisely the evidence that can disappear for
    strongly coeluting, spectrally similar compounds.
    """
    candidates = []
    for item in evaluated:
        if int(item["component_count"]) < 3:
            continue
        stability = item.get("stability") or {}
        resolved_shape = item.get("resolved_profile_shape_test") or {}
        residual_null = item.get("residual_null_test") or {}
        if not (
                stability.get("all_components_stable")
                and resolved_shape.get("supported")
                and residual_null.get("supported")):
            continue

        simpler = [
            other for other in evaluated
            if int(other["component_count"]) < int(item["component_count"])
        ]
        if not simpler:
            continue

        comparisons = []
        decisive = True
        for reference in simpler:
            comparison = _paired_comparison(
                item["cv_fold_errors"], reference["cv_fold_errors"], config,
                seed_offset=(
                    60000
                    + int(item["candidate_number"]) * 100
                    + int(reference["candidate_number"])
                ),
            )
            comparison.update({
                "candidate_indices": list(item["indices"]),
                "reference_indices": list(reference["indices"]),
            })
            comparisons.append(comparison)
            # Error is candidate minus reference.  An upper confidence bound
            # below zero shows that the complex candidate is consistently
            # better across the held-out masks.
            if comparison["confidence_interval"][1] >= 0.0:
                decisive = False

        if decisive:
            candidates.append((item, comparisons))

    if not candidates:
        return None, []
    return min(
        candidates,
        key=lambda pair: (
            -int(pair[0]["component_count"]),
            float(pair[0]["cv_normalised_sse_mean"]),
            tuple(pair[0]["indices"]),
        ),
    )


def _residual_peak_corroborated(item, simpler, config):
    """Require independent evidence before the residual path restores a model.

    The component-specific residual EIC is retained as useful diagnostic
    evidence, but it is not a hard completeness test. A weak component near a
    segment edge can leave a broad positive residual even when the candidate
    is independently supported by prediction geometry, masked validation,
    residual randomisation, perturbation stability, and FRR.
    """
    residual_peak = item.get("residual_component_peak_test") or {}
    prediction_shape = (
        residual_peak.get("prediction_profile_independence_test") or {}
    )
    residual_null = item.get("residual_null_test") or {}
    stability = item.get("stability") or {}
    if not (
            prediction_shape.get("supported")
            and residual_null.get("supported")
            and stability.get("all_components_stable")
            and item["cv_normalised_sse_mean"]
            < simpler["cv_normalised_sse_mean"]):
        return False
    if not config.enable_frr_confirmation:
        return False
    if item.get("frr_r2") is None or simpler.get("frr_r2") is None:
        return False
    return bool(
        item["frr_r2"] - simpler["frr_r2"]
        >= float(config.frr_r2_margin)
    )


def _component_count_elbow_test(item, evaluated):
    """Detect a supported CV elbow without an absolute improvement cutoff.

    This protects against deleting an entire disputed cluster when doing so
    causes clear underfitting. The gain obtained by adding the candidate to
    its direct simpler subset must exceed the remaining gain available from
    the next component count. A reproducible component-specific residual peak,
    residual randomisation, and complete resolved profiles remain mandatory,
    so a deterministic peak-shape or baseline improvement cannot create this
    protection by itself.
    """
    indices = tuple(item["indices"])
    count = int(item["component_count"])
    residual_null = item.get("residual_null_test") or {}
    residual_peak = item.get("residual_component_peak_test") or {}
    resolved_shape = item.get("resolved_profile_shape_test") or {}
    simpler_indices = tuple(residual_null.get("simpler_indices", ()))
    by_indices = {
        tuple(candidate["indices"]): candidate for candidate in evaluated
    }
    simpler = by_indices.get(simpler_indices)
    next_candidates = [
        candidate for candidate in evaluated
        if int(candidate["component_count"]) == count + 1
        and set(indices).issubset(set(candidate["indices"]))
    ]
    same_count = [
        candidate for candidate in evaluated
        if int(candidate["component_count"]) == count
    ]
    if not (
            simpler is not None
            and next_candidates
            and residual_null.get("supported")
            and residual_peak.get(
                "residual_peak_supported_before_prediction_check"
            )
            and resolved_shape.get("supported")):
        return {
            "supported": False,
            "reason": "component_count_elbow_prerequisites_not_met",
            "simpler_indices": list(simpler_indices),
            "residual_peak_supported_before_prediction_check": bool(
                residual_peak.get(
                    "residual_peak_supported_before_prediction_check"
                )
            ),
        }

    current_error = float(item["cv_normalised_sse_mean"])
    best_same_error = min(
        float(candidate["cv_normalised_sse_mean"])
        for candidate in same_count
    )
    tolerance = 1e-12
    if current_error > best_same_error + tolerance:
        return {
            "supported": False,
            "reason": "candidate_is_not_best_at_component_count",
            "simpler_indices": list(simpler_indices),
        }

    simpler_error = float(simpler["cv_normalised_sse_mean"])
    next_best = min(
        next_candidates,
        key=lambda candidate: candidate["cv_normalised_sse_mean"],
    )
    next_error = float(next_best["cv_normalised_sse_mean"])
    gain_to_candidate = float(simpler_error - current_error)
    remaining_gain = float(max(current_error - next_error, 0.0))
    supported = bool(
        gain_to_candidate > tolerance
        and gain_to_candidate > remaining_gain + tolerance
    )
    return {
        "supported": supported,
        "reason": (
            "masked_validation_elbow_supports_component_count"
            if supported else
            "no_masked_validation_elbow_at_component_count"
        ),
        "simpler_indices": list(simpler_indices),
        "next_indices": list(next_best["indices"]),
        "simpler_cv_error": simpler_error,
        "candidate_cv_error": current_error,
        "next_cv_error": next_error,
        "gain_to_candidate": gain_to_candidate,
        "remaining_gain_to_next_count": remaining_gain,
        "relative_rule": "gain_to_candidate_exceeds_remaining_gain",
        "residual_peak_supported_before_prediction_check": True,
    }


def _candidate_has_independent_shape(item, simpler, config):
    direct_shape = item.get("independent_peak_shape_test") or {}
    count_elbow = item.get("component_count_elbow_test") or {}
    return bool(
        direct_shape.get("supported")
        or _residual_peak_corroborated(item, simpler, config)
        or count_elbow.get("supported")
    )


def _corroborated_shape_candidates(evaluated, config):
    """Return complex models with reproducible shape and independent support."""
    by_indices = {
        tuple(item["indices"]): item for item in evaluated
    }
    supported = []
    for item in evaluated:
        resolved_shape = item.get("resolved_profile_shape_test") or {}
        if not resolved_shape.get("supported"):
            continue
        shape = item.get("independent_peak_shape_test") or {}
        simpler_indices = tuple(shape.get("simpler_indices", ()))
        simpler = by_indices.get(simpler_indices)
        if simpler is None:
            continue
        if not _candidate_has_independent_shape(item, simpler, config):
            continue
        residual = item.get("residual_null_test") or {}
        cv_support = bool(
            item["cv_normalised_sse_mean"]
            < simpler["cv_normalised_sse_mean"]
        )
        frr_support = bool(
            item.get("frr_r2") is not None
            and simpler.get("frr_r2") is not None
            and item["frr_r2"] - simpler["frr_r2"]
            >= float(config.frr_r2_margin)
        )
        if residual.get("supported") or cv_support or frr_support:
            supported.append(item)
    return supported


def _shape_gated_candidates(pool, evaluated, config):
    """Reject unsupported complex models when a simpler model is available.

    Cross validation can favour an over-split model because it absorbs
    deterministic peak-shape mismatch or baseline structure.  Once a simpler
    candidate exists, a complex candidate therefore needs explicit,
    reproducible independent peak-shape evidence.  If no simpler candidate
    was generated, the complex candidate is retained as the least
    assumptive available model and the limitation remains visible in JSON.
    """
    all_candidates = list(evaluated)
    output = []
    for item in pool:
        if int(item["component_count"]) <= 1:
            output.append(item)
            continue
        simpler_exists = any(
            int(other["component_count"]) < int(item["component_count"])
            for other in all_candidates
        )
        residual_peak = item.get("residual_component_peak_test") or {}
        simpler_indices = tuple(residual_peak.get("simpler_indices", ()))
        simpler = next(
            (candidate for candidate in all_candidates
             if tuple(candidate["indices"]) == simpler_indices),
            None,
        )
        shape_supported = bool(
            simpler is not None
            and _candidate_has_independent_shape(item, simpler, config)
        )
        resolved_shape_supported = bool(
            item.get("resolved_profile_shape_test")
            and item["resolved_profile_shape_test"].get("supported")
        )
        if not simpler_exists or (
                shape_supported and resolved_shape_supported):
            output.append(item)
    return output


def _stability(observed, anchor, config, seed_offset):
    rng = np.random.default_rng(int(config.random_seed) + int(seed_offset))
    chromatograms = []
    spectra = []
    areas = []
    for _ in range(int(config.stability_runs)):
        # Continuous channel reweighting perturbs the spectral evidence without
        # dropping an entire mass channel from the fitted spectrum.
        channel_weights = rng.uniform(0.5, 1.5, size=observed.shape[1])
        weights = np.broadcast_to(
            channel_weights[None, :], observed.shape
        ).copy()
        C, S, _ = _fit_weighted_mcr(
            observed,
            anchor,
            config,
            weights=weights,
            rng=rng,
            perturb=True,
        )
        chromatograms.append(C)
        spectra.append(S)
        areas.append(np.trapz(C, axis=0) * np.sum(S, axis=1))
    chromatograms = np.asarray(chromatograms)
    spectra = np.asarray(spectra)
    areas = np.asarray(areas)
    apex_sd = np.std(np.argmax(chromatograms, axis=1), axis=0, ddof=1)
    area_mean = np.mean(areas, axis=0)
    area_cv = np.std(areas, axis=0, ddof=1) / np.maximum(np.abs(area_mean), 1e-12)
    reference_spectra = np.median(spectra, axis=0)
    spectral_cosines = []
    for component in range(anchor.shape[1]):
        spectral_cosines.append(float(np.median([
            _cosine_similarity(run[component], reference_spectra[component])
            for run in spectra
        ])))
    component_stable = [
        bool(
            apex_sd[index] <= config.stability_apex_sd
            and spectral_cosines[index] >= config.stability_spectral_cosine
            and area_cv[index] <= config.stability_area_cv
        )
        for index in range(anchor.shape[1])
    ]
    return {
        "apex_sd": apex_sd.tolist(),
        "median_spectral_cosine": spectral_cosines,
        "area_cv": area_cv.tolist(),
        "component_stable": component_stable,
        "all_components_stable": bool(all(component_stable)),
    }


def select_component_model(observed, anchors, events, config=None,
                           frr_evaluator=None, prediction_profiles=None,
                           peak_shape_observed=None,
                           peak_shape_anchors=None,
                           peak_shape_context=None):
    """Compare component subsets and return one deterministic selected model."""
    config = config or ComponentSelectionConfig()
    observed = np.maximum(np.asarray(observed, dtype=float), 0.0)
    anchors = np.maximum(np.asarray(anchors, dtype=float), 0.0)
    if peak_shape_observed is None:
        peak_shape_observed = observed
    else:
        peak_shape_observed = np.maximum(
            np.asarray(peak_shape_observed, dtype=float), 0.0
        )
    if peak_shape_anchors is None:
        peak_shape_anchors = anchors
    else:
        peak_shape_anchors = np.maximum(
            np.asarray(peak_shape_anchors, dtype=float), 0.0
        )
    if (peak_shape_observed.ndim != 2 or peak_shape_anchors.ndim != 2
            or peak_shape_observed.shape[0] != peak_shape_anchors.shape[0]
            or peak_shape_anchors.shape[1] != anchors.shape[1]):
        raise ValueError(
            "peak-shape observed data and anchors have incompatible dimensions"
        )
    active_indices = _active_profile_indices(
        anchors, config.active_profile_threshold
    )
    prediction_proxy = (
        prediction_confidence_proxy(prediction_profiles)
        if config.record_prediction_confidence_proxy
        and prediction_profiles is not None else []
    )
    clusters = _build_clusters(events, active_indices)
    if not clusters:
        return None

    rank_evidence = _cluster_rank_evidence(
        observed, anchors, active_indices, clusters, config
    )
    all_candidates = _candidate_subsets(active_indices, clusters)
    shortlist = _shortlist_candidates(
        observed, anchors, all_candidates, config
    )
    cv_splits = _build_cv_splits(observed, config)

    evaluated = []
    fitted_models = {}
    for candidate_number, (indices, fixed_r2) in enumerate(shortlist):
        anchor = anchors[:, indices]
        C, S, fitted = _fit_weighted_mcr(observed, anchor, config)
        cv_mean, cv_se, cv_errors, cv_split_labels = _masked_cv(
            observed, anchor, config, cv_splits
        )
        fitted_models[indices] = (C, S)
        evaluated.append({
            "indices": list(indices),
            "component_count": len(indices),
            "fixed_profile_r2": float(fixed_r2),
            "constrained_mcr_r2": _r2(observed, fitted),
            "cv_normalised_sse_mean": cv_mean,
            "cv_normalised_sse_se": cv_se,
            "cv_fold_errors": cv_errors,
            "cv_split_labels": cv_split_labels,
            "candidate_number": candidate_number,
            "resolved_profile_shape_test": _resolved_profile_shape_test(
                C, config
            ),
            "stability": None,
            "frr_r2": None,
            "frr_used": False,
            "frr_error": None,
            "prediction_proxy_mean": (
                float(np.mean([
                    prediction_proxy[index]["confidence_proxy_score"]
                    for index in indices
                    if index < len(prediction_proxy)
                ]))
                if prediction_proxy else None
            ),
            "prediction_proxy_min": (
                float(np.min([
                    prediction_proxy[index]["confidence_proxy_score"]
                    for index in indices
                    if index < len(prediction_proxy)
                ]))
                if prediction_proxy else None
            ),
        })

    # Add candidate-specific selective ion evidence and explicit leave-one-out
    # comparisons. These records make it possible to distinguish a weak
    # overprediction from a candidate supported by informative mass channels.
    for item in evaluated:
        candidate_indices = tuple(item["indices"])
        candidate_C, candidate_S = fitted_models[candidate_indices]
        item["selective_ion_evidence"] = _selective_ion_evidence(
            observed,
            candidate_S,
            anchors[:, candidate_indices],
            events,
            config,
            candidate_indices=candidate_indices,
        )
        item["selective_ion_supported"] = bool(
            any(evidence["selective_ion_supported"]
                for evidence in item["selective_ion_evidence"])
        )
    full_candidates = [
        item for item in evaluated
        if item["component_count"] == max(
            candidate["component_count"] for candidate in evaluated
        )
    ]
    full_reference = min(
        full_candidates,
        key=lambda item: item["cv_normalised_sse_mean"],
    ) if full_candidates else None
    leave_one_out = []
    if full_reference is not None and full_reference["component_count"] > 1:
        for item in evaluated:
            if item is full_reference or item["component_count"] != (
                    full_reference["component_count"] - 1):
                continue
            comparison = _paired_comparison(
                item["cv_fold_errors"],
                full_reference["cv_fold_errors"],
                config,
                seed_offset=40000 + int(item["candidate_number"]),
            )
            leave_one_out.append({
                "removed_indices": [
                    index for index in full_reference["indices"]
                    if index not in item["indices"]
                ],
                "remaining_indices": list(item["indices"]),
                "full_indices": list(full_reference["indices"]),
                "cv_error_difference_simpler_minus_full": float(
                    item["cv_normalised_sse_mean"]
                    - full_reference["cv_normalised_sse_mean"]
                ),
                "paired_cv": comparison,
            })

    # Evaluate whether each additional profile improves masked validation
    # beyond what can be obtained by resampling the simpler-model residual.
    # This is intentionally done before the stability/parsimony decision so a
    # genuine but mildly distorted coelution is not discarded solely because
    # one ITTFA profile is unstable.
    for item in evaluated:
        item["residual_null_test"] = None
        item["independent_peak_shape_test"] = None
        item["residual_component_peak_test"] = None
        if item["component_count"] <= 1:
            continue
        simpler = [
            other for other in evaluated
            if other["component_count"] == item["component_count"] - 1
            and set(other["indices"]).issubset(set(item["indices"]))
        ]
        if not simpler:
            continue
        simpler_item = min(
            simpler,
            key=lambda candidate: (
                candidate["cv_normalised_sse_mean"],
                tuple(candidate["indices"]),
            ),
        )
        item["residual_null_test"] = _residual_null_test(
            observed,
            anchors[:, tuple(simpler_item["indices"])],
            anchors[:, tuple(item["indices"])],
            cv_splits,
            config,
            seed_offset=60000 + int(item["candidate_number"]),
        )
        item["residual_null_test"]["simpler_indices"] = list(
            simpler_item["indices"]
        )
        centre_limits = None
        if peak_shape_context is not None:
            context_offset = int(peak_shape_context.get("segment_offset", 0))
            context_count = int(
                peak_shape_context.get("segment_scan_count", anchors.shape[0])
            )
            centre_limits = (
                context_offset,
                context_offset + max(context_count - 1, 0),
            )
        item["independent_peak_shape_test"] = _independent_peak_shape_test(
            peak_shape_observed,
            peak_shape_anchors[:, tuple(simpler_item["indices"])],
            peak_shape_anchors[:, tuple(item["indices"])],
            config,
            seed_offset=70000 + int(item["candidate_number"]),
            centre_limits=centre_limits,
        )
        item["independent_peak_shape_test"]["simpler_indices"] = list(
            simpler_item["indices"]
        )
        item["residual_component_peak_test"] = (
            _residual_component_peak_test(
                observed,
                fitted_models[tuple(simpler_item["indices"])],
                fitted_models[tuple(item["indices"])],
                anchors,
                tuple(simpler_item["indices"]),
                tuple(item["indices"]),
                config,
                seed_offset=80000 + int(item["candidate_number"]),
            )
        )
        prediction_independence = _prediction_profile_independence_test(
            prediction_proxy,
            tuple(item["indices"]),
            tuple(simpler_item["indices"]),
        )
        residual_peak = item["residual_component_peak_test"]
        residual_peak["residual_peak_supported_before_prediction_check"] = bool(
            residual_peak.get("supported")
        )
        residual_peak["prediction_profile_independence_test"] = (
            prediction_independence
        )
        residual_peak["supported"] = bool(
            residual_peak.get("supported")
            and prediction_independence.get("supported")
        )
        if (residual_peak["residual_peak_supported_before_prediction_check"]
                and not residual_peak["supported"]):
            residual_peak["reason"] = (
                "residual_peak_lacks_independent_prediction_profile"
            )
        item["residual_component_peak_test"]["simpler_indices"] = list(
            simpler_item["indices"]
        )
        if peak_shape_context is not None:
            item["independent_peak_shape_test"]["context"] = dict(
                peak_shape_context
            )

    # FRR is deliberately evaluated only for the already shortlisted overlap
    # candidates. The callback is implemented by DeepCPR.py so this module
    # remains usable in isolation and does not create an import cycle.
    if (config.enable_candidate_frr_confirmation
            and config.enable_frr_confirmation
            and frr_evaluator is not None):
        frr_targets = _frr_candidate_targets(evaluated, config)
        for item in evaluated:
            candidate_key = tuple(item["indices"])
            if candidate_key not in frr_targets:
                item["frr_error"] = "not_evaluated_for_speed"
                continue
            try:
                frr_result = frr_evaluator(
                    fitted_models[candidate_key][0],
                    candidate_key,
                )
                frr_r2 = (
                    None if frr_result is None else frr_result.get("r2")
                )
                if frr_r2 is not None and np.isfinite(float(frr_r2)):
                    item["frr_r2"] = float(frr_r2)
                    item["frr_used"] = bool(frr_result.get("used", True))
                    if frr_result.get("error"):
                        item["frr_error"] = str(frr_result["error"])
            except Exception as exc:
                item["frr_error"] = "{}: {}".format(
                    type(exc).__name__, str(exc)
                )
    elif frr_evaluator is not None:
        for item in evaluated:
            item["frr_error"] = "disabled_for_component_selection"

    best = min(evaluated, key=lambda item: item["cv_normalised_sse_mean"])
    paired_comparisons = []
    competitive_pool = []
    for item in evaluated:
        comparison = _paired_comparison(
            item["cv_fold_errors"],
            best["cv_fold_errors"],
            config,
            seed_offset=10000 + item["candidate_number"],
        )
        comparison.update({
            "candidate_indices": list(item["indices"]),
            "reference_indices": list(best["indices"]),
        })
        paired_comparisons.append(comparison)
        # A candidate remains competitive if paired validation cannot show
        # that it is worse than the lowest-mean candidate.
        if comparison["confidence_interval"][0] <= 0.0:
            competitive_pool.append(item)

    # Stability is evaluated for every candidate, not only for candidates
    # already favoured by cross validation.  Otherwise an unstable complex
    # candidate can exclude a simpler candidate before stability is checked.
    for item in evaluated:
        indices = tuple(item["indices"])
        item["stability"] = _stability(
            observed,
            anchors[:, indices],
            config,
            seed_offset=item["candidate_number"] * 100,
        )

    for item in evaluated:
        item["component_count_elbow_test"] = _component_count_elbow_test(
            item, evaluated
        )

    stable_pool = [
        item for item in evaluated
        if item["stability"] is not None
        and item["stability"]["all_components_stable"]
    ]

    # A more complex model may have the lowest CV error simply because an
    # additional profile absorbs deterministic shape mismatch.  Among stable
    # candidates, retain the simpler model unless paired validation provides
    # evidence that it is worse than the best stable model.  If no candidate
    # is stable, the automatic policy deliberately chooses the smallest model
    # because unresolved instability is evidence against retaining an extra
    # component, not a reason to expose a manual review state.
    stable_comparisons = []
    if stable_pool:
        best_stable = min(
            stable_pool,
            key=lambda item: (
                item["cv_normalised_sse_mean"],
                tuple(item["indices"]),
            ),
        )
        stable_pool_for_decision = []
        for item in stable_pool:
            comparison = _paired_comparison(
                item["cv_fold_errors"],
                best_stable["cv_fold_errors"],
                config,
                seed_offset=20000 + item["candidate_number"],
            )
            comparison.update({
                "candidate_indices": list(item["indices"]),
                "reference_indices": list(best_stable["indices"]),
            })
            stable_comparisons.append(comparison)
            if comparison["confidence_interval"][0] <= 0.0:
                stable_pool_for_decision.append(item)
        decision_pool = stable_pool_for_decision or [best_stable]
        # Promote a complex candidate only when its observed validation gain
        # exceeds the residual null distribution. This overrides parsimony
        # for reproducibly informative extra profiles, while leaving the
        # legacy tie behaviour unchanged for ordinary candidates.
        decisive_candidate, decisive_validation = (
            _decisive_multicomponent_candidate(evaluated, config)
        )
        shape_supported = _corroborated_shape_candidates(evaluated, config)
        if decisive_candidate is not None:
            selected = decisive_candidate
            selection_reason = (
                "decisive_masked_validation_supports_stable_multicomponent_model"
            )
        elif shape_supported:
            selected = min(
                shape_supported,
                key=lambda item: (
                    -item["component_count"],
                    item["cv_normalised_sse_mean"],
                    tuple(item["indices"]),
                ),
            )
            selection_reason = (
                "corroborated_component_evidence_supports_additional_component"
            )
        else:
            gated_pool = _shape_gated_candidates(
                decision_pool, evaluated, config
            )
            if not gated_pool:
                # A simpler candidate may be unstable while every complex
                # candidate lacks shape support.  Prefer the simplest
                # evaluated model rather than allowing CV alone to restore an
                # unsupported extra component.
                gated_pool = list(evaluated)
            selected = min(
                gated_pool,
                key=lambda item: (
                    item["component_count"],
                    item["cv_normalised_sse_mean"],
                    tuple(item["indices"]),
                ),
            )
            selection_reason = (
                "smallest_shape_supported_or_simplest_model"
            )
    else:
        decision_pool = list(evaluated)
        decisive_candidate, decisive_validation = (
            _decisive_multicomponent_candidate(evaluated, config)
        )
        shape_supported = _corroborated_shape_candidates(evaluated, config)
        if decisive_candidate is not None:
            selected = decisive_candidate
            selection_reason = (
                "decisive_masked_validation_supports_stable_multicomponent_model"
            )
        elif shape_supported:
            selected = min(
                shape_supported,
                key=lambda item: (
                    -item["component_count"],
                    item["cv_normalised_sse_mean"],
                    tuple(item["indices"]),
                ),
            )
            selection_reason = (
                "corroborated_component_evidence_supports_additional_component"
            )
        elif config.prefer_simpler_on_conflict:
            gated_pool = _shape_gated_candidates(
                decision_pool, evaluated, config
            )
            if not gated_pool:
                gated_pool = list(evaluated)
            selected = min(
                gated_pool,
                key=lambda item: (
                    item["component_count"],
                    item["cv_normalised_sse_mean"],
                    tuple(item["indices"]),
                ),
            )
            selection_reason = (
                "no candidate stable; smallest_shape_supported_or_simplest_model"
            )
        else:
            selected = best
            selection_reason = (
                "no candidate stable; lowest cross validation error selected"
            )
    # A complex candidate that is unstable after ITTFA can still be retained
    # when FRR gives a reproducible full-data improvement and paired masked CV
    # does not show that candidate to be worse. This specifically protects
    # genuine high-similarity coelutions whose third profile is distorted by
    # ITTFA. FRR never creates a candidate and never overrides a clear CV
    # disadvantage.
    frr_confirmation = []
    if (config.enable_candidate_frr_confirmation
            and config.enable_frr_confirmation
            and frr_evaluator is not None
            and selected.get("frr_r2") is not None):
        for item in evaluated:
            if item is selected or item.get("frr_r2") is None:
                continue
            if int(item["component_count"]) <= int(selected["component_count"]):
                continue
            comparison = _paired_comparison(
                item["cv_fold_errors"], selected["cv_fold_errors"], config,
                seed_offset=30000 + int(item["candidate_number"]),
            )
            frr_gap = float(item["frr_r2"] - selected["frr_r2"])
            # The paired error is complex minus simple.  A lower confidence
            # limit at or below zero means the complex candidate has not been
            # shown worse by masked CV.  If the whole interval is positive,
            # the complex candidate is significantly worse and is rejected.
            noninferior_cv = comparison["confidence_interval"][0] <= 0.0
            residual_peak = item.get("residual_component_peak_test") or {}
            simpler_indices = tuple(
                residual_peak.get("simpler_indices", ())
            )
            residual_reference = next(
                (candidate for candidate in evaluated
                 if tuple(candidate["indices"]) == simpler_indices),
                None,
            )
            independent_shape = bool(
                residual_reference is not None
                and _candidate_has_independent_shape(
                    item, residual_reference, config
                )
            )
            resolved_shape = bool(
                item.get("resolved_profile_shape_test")
                and item["resolved_profile_shape_test"].get("supported")
            )
            supported = bool(
                frr_gap >= float(config.frr_r2_margin)
                and noninferior_cv
                and independent_shape
                and resolved_shape
            )
            frr_confirmation.append({
                "candidate_indices": list(item["indices"]),
                "reference_indices": list(selected["indices"]),
                "candidate_frr_r2": float(item["frr_r2"]),
                "reference_frr_r2": float(selected["frr_r2"]),
                "frr_r2_gap": frr_gap,
                "paired_cv": comparison,
                "independent_peak_shape_supported": independent_shape,
                "residual_component_peak_supported": bool(
                    item.get("residual_component_peak_test")
                    and item["residual_component_peak_test"].get("supported")
                ),
                "resolved_profile_shape_supported": resolved_shape,
                "supported": supported,
            })
        supported = [item for item in frr_confirmation if item["supported"]]
        if supported:
            chosen_indices = max(
                supported,
                key=lambda item: (item["frr_r2_gap"],
                                  len(item["candidate_indices"])),
            )["candidate_indices"]
            selected = next(
                item for item in evaluated
                if list(item["indices"]) == list(chosen_indices)
            )
            selection_reason = (
                "frr_confirmation_supports_more_components_without_cv_penalty"
            )

    selective_ion_confirmation = []
    if config.enable_selective_ion_confirmation:
        for item in evaluated:
            # A two component candidate can be produced by a single peak
            # overprediction and its apparent selective ion contrast is easily
            # inflated by low abundance channels.  Use selective ion evidence
            # to restore only a three component candidate; for one-to-two
            # decisions it remains a diagnostic record.
            if (item is selected
                    or item["component_count"] <= selected["component_count"]
                    or item["component_count"] < 3):
                continue
            if not item.get("selective_ion_supported"):
                continue
            residual_peak = item.get("residual_component_peak_test") or {}
            simpler_indices = tuple(
                residual_peak.get("simpler_indices", ())
            )
            residual_reference = next(
                (candidate for candidate in evaluated
                 if tuple(candidate["indices"]) == simpler_indices),
                None,
            )
            independent_shape = bool(
                residual_reference is not None
                and _candidate_has_independent_shape(
                    item, residual_reference, config
                )
            )
            resolved_shape = bool(
                item.get("resolved_profile_shape_test")
                and item["resolved_profile_shape_test"].get("supported")
            )
            # Ion contrast can identify useful mass channels, but it does not
            # prove that a separate chromatographic peak exists.  In
            # particular, baseline and low abundance ions can make an
            # over-split single peak look spectrally different.  Therefore
            # selective-ion evidence may corroborate, but never replace, the
            # independent peak-shape requirement.
            if not independent_shape or not resolved_shape:
                continue
            comparison = _paired_comparison(
                item["cv_fold_errors"], selected["cv_fold_errors"], config,
                seed_offset=50000 + int(item["candidate_number"]),
            )
            noninferior_cv = comparison["confidence_interval"][0] <= 0.0
            if noninferior_cv:
                selective_ion_confirmation.append({
                    "candidate_indices": list(item["indices"]),
                    "reference_indices": list(selected["indices"]),
                    "paired_cv": comparison,
                    "selective_ion_evidence": item["selective_ion_evidence"],
                    "independent_peak_shape_supported": independent_shape,
                    "resolved_profile_shape_supported": resolved_shape,
                    "supported": True,
                })
        if selective_ion_confirmation:
            chosen_indices = max(
                selective_ion_confirmation,
                key=lambda item: len(item["candidate_indices"]),
            )["candidate_indices"]
            selected = next(
                item for item in evaluated
                if list(item["indices"]) == list(chosen_indices)
            )
            selection_reason = (
                "selective_ion_evidence_supports_more_components_without_cv_penalty"
            )

    selected_indices = tuple(selected["indices"])
    selected_C, selected_S = fitted_models[selected_indices]

    selected_is_stable = bool(
        selected["stability"] is not None
        and selected["stability"]["all_components_stable"]
    )
    other_stable_competitors = [
        item for item in stable_pool if item is not selected
    ]
    if selected_is_stable and not other_stable_competitors:
        evidence_grade = "strong"
    elif selected_is_stable:
        evidence_grade = "moderate"
    else:
        evidence_grade = "weak"

    if selected["component_count"] < len(active_indices):
        action = "reduce_component_count"
    else:
        action = "retain_component_count"
    record = {
        "method": "paired_repeated_cv_stability_v2",
        "config": asdict(config),
        "active_indices": active_indices,
        "clusters": [list(cluster) for cluster in clusters],
        "rank_evidence": rank_evidence,
        "all_candidate_count": len(all_candidates),
        "evaluated_candidate_count": len(evaluated),
        "candidates": evaluated,
        "paired_comparisons_to_best": paired_comparisons,
        "best_mean_cv_indices": list(best["indices"]),
        "competitive_indices": [list(item["indices"]) for item in competitive_pool],
        "stable_indices": [list(item["indices"]) for item in stable_pool],
        "stable_comparisons_to_best_stable": stable_comparisons,
        "decisive_multicomponent_validation": decisive_validation,
        "frr_confirmation": frr_confirmation,
        "selected_indices": list(selected_indices),
        "selected_component_count": len(selected_indices),
        "selection_action": action,
        "evidence_grade": evidence_grade,
        # Retained for readers of version 1 JSON reports.  This is an
        # evidence grade, not a calibrated probability of correctness.
        "selection_confidence": evidence_grade,
        "selection_reason": selection_reason,
        "prediction_confidence_proxy": prediction_proxy,
        "leave_one_out_comparisons": leave_one_out,
        "selective_ion_confirmation": selective_ion_confirmation,
    }
    return {
        "chromatograms": selected_C,
        "spectra": selected_S,
        "reconstruction": selected_C @ selected_S,
        "record": record,
    }


def fit_component_subset(observed, anchors, indices, config=None):
    """Fit one consensus-selected subset to the observed GC MS matrix."""
    config = config or ComponentSelectionConfig()
    observed = np.maximum(np.asarray(observed, dtype=float), 0.0)
    anchors = np.maximum(np.asarray(anchors, dtype=float), 0.0)
    indices = tuple(int(index) for index in indices)
    if not indices:
        raise ValueError("a component subset cannot be empty")
    if min(indices) < 0 or max(indices) >= anchors.shape[1]:
        raise ValueError("component subset index is outside the anchor matrix")
    chromatograms, spectra, reconstruction = _fit_weighted_mcr(
        observed, anchors[:, indices], config
    )
    return {
        "chromatograms": chromatograms,
        "spectra": spectra,
        "reconstruction": reconstruction,
        "indices": list(indices),
        "r2": _r2(observed, reconstruction),
    }


def _fit_two_group_rank_model(values):
    """Fit a two-normal mixture to log singular-value ratios.

    The lower-mean group is interpreted only as the lower-rank population in
    the current batch.  The returned posterior is not a probability that the
    chemical truth is one component.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    if np.sum(valid) < 4 or np.ptp(np.log(values[valid])) <= 1e-12:
        return {
            "status": "insufficient_distribution",
            "posteriors_lower_rank": [0.5 if item else None for item in valid],
            "log_means": None,
            "log_standard_deviations": None,
            "weights": None,
        }

    transformed = np.log(values[valid])
    means = np.quantile(transformed, [0.25, 0.75]).astype(float)
    total_variance = max(float(np.var(transformed)), 1e-8)
    variances = np.full(2, total_variance / 2.0, dtype=float)
    weights = np.full(2, 0.5, dtype=float)

    for _ in range(200):
        previous = means.copy()
        densities = []
        for group in range(2):
            variance = max(float(variances[group]), total_variance * 1e-6)
            density = np.exp(
                -0.5 * (transformed - means[group]) ** 2 / variance
            ) / np.sqrt(2.0 * np.pi * variance)
            densities.append(weights[group] * density)
        responsibilities = np.column_stack(densities)
        responsibilities /= np.maximum(
            np.sum(responsibilities, axis=1, keepdims=True), 1e-300
        )
        effective = np.maximum(np.sum(responsibilities, axis=0), 1e-12)
        weights = effective / transformed.size
        means = np.sum(responsibilities * transformed[:, None], axis=0) / effective
        variances = np.sum(
            responsibilities * (transformed[:, None] - means) ** 2,
            axis=0,
        ) / effective
        variances = np.maximum(variances, total_variance * 1e-6)
        if np.max(np.abs(means - previous)) <= 1e-9:
            break

    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]
    densities = []
    for group in range(2):
        density = np.exp(
            -0.5 * (transformed - means[group]) ** 2 / variances[group]
        ) / np.sqrt(2.0 * np.pi * variances[group])
        densities.append(weights[group] * density)
    weighted_densities = np.column_stack(densities)
    mixture_density = np.maximum(
        np.sum(weighted_densities, axis=1), 1e-300
    )
    responsibilities = weighted_densities / mixture_density[:, None]

    one_mean = float(np.mean(transformed))
    one_variance = max(float(np.var(transformed)), 1e-12)
    one_density = np.exp(
        -0.5 * (transformed - one_mean) ** 2 / one_variance
    ) / np.sqrt(2.0 * np.pi * one_variance)
    one_log_likelihood = float(np.sum(np.log(np.maximum(one_density, 1e-300))))
    two_log_likelihood = float(np.sum(np.log(mixture_density)))
    one_bic = 2.0 * np.log(transformed.size) - 2.0 * one_log_likelihood
    two_bic = 5.0 * np.log(transformed.size) - 2.0 * two_log_likelihood
    if two_bic >= one_bic:
        return {
            "status": "no_two_population_support",
            "posteriors_lower_rank": [0.5 if item else None for item in valid],
            "log_means": means.tolist(),
            "log_standard_deviations": np.sqrt(variances).tolist(),
            "weights": weights.tolist(),
            "bic_one_population": float(one_bic),
            "bic_two_populations": float(two_bic),
            "equal_membership_sigma_ratio": None,
        }

    grid = np.linspace(float(means[0]), float(means[1]), 10000)
    grid_density = []
    for group in range(2):
        density = np.exp(
            -0.5 * (grid - means[group]) ** 2 / variances[group]
        ) / np.sqrt(2.0 * np.pi * variances[group])
        grid_density.append(weights[group] * density)
    grid_density = np.asarray(grid_density)
    boundary_index = int(np.argmin(np.abs(grid_density[0] - grid_density[1])))

    posterior = []
    position = 0
    for is_valid in valid:
        if is_valid:
            posterior.append(float(responsibilities[position, 0]))
            position += 1
        else:
            posterior.append(None)
    return {
        "status": "fitted",
        "posteriors_lower_rank": posterior,
        "log_means": means.tolist(),
        "log_standard_deviations": np.sqrt(variances).tolist(),
        "weights": weights.tolist(),
        "bic_one_population": float(one_bic),
        "bic_two_populations": float(two_bic),
        "equal_membership_sigma_ratio": float(np.exp(grid[boundary_index])),
    }


def _rank_model_membership(values, model):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values > 0.0)
    if model.get("status") != "fitted":
        return [0.5 if item else None for item in valid]
    means = np.asarray(model["log_means"], dtype=float)
    deviations = np.asarray(model["log_standard_deviations"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    transformed = np.log(values[valid])
    densities = []
    for group in range(2):
        variance = max(float(deviations[group] ** 2), 1e-15)
        density = np.exp(
            -0.5 * (transformed - means[group]) ** 2 / variance
        ) / np.sqrt(2.0 * np.pi * variance)
        densities.append(weights[group] * density)
    densities = np.column_stack(densities)
    responsibilities = densities / np.maximum(
        np.sum(densities, axis=1, keepdims=True), 1e-300
    )
    posterior = []
    position = 0
    for is_valid in valid:
        if is_valid:
            posterior.append(float(responsibilities[position, 0]))
            position += 1
        else:
            posterior.append(None)
    return posterior


def _retention_groups(records):
    """Group records from different injections by overlapping RT intervals."""
    ordered = sorted(
        range(len(records)),
        key=lambda index: (
            float(records[index]["rt_start"]),
            float(records[index]["rt_end"]),
        ),
    )
    groups = []
    for index in ordered:
        start = float(records[index]["rt_start"])
        end = float(records[index]["rt_end"])
        matching = []
        for group_index, group in enumerate(groups):
            # Do not join two segments from the same injection.  Positive RT
            # overlap across injections is sufficient and has no tuned width.
            files = {str(records[item].get("file", "")) for item in group}
            if str(records[index].get("file", "")) in files:
                continue
            overlaps = [
                max(
                    0.0,
                    min(end, float(records[item]["rt_end"]))
                    - max(start, float(records[item]["rt_start"])),
                )
                for item in group
            ]
            if max(overlaps, default=0.0) > 0.0:
                matching.append((max(overlaps), group_index))
        if not matching:
            groups.append([index])
            continue
        # A broad interval can overlap two adjacent groups.  Assign it to the
        # group with the largest actual overlap instead of bridging and merging
        # chemically separate regions.
        primary = max(matching, key=lambda item: item[0])[1]
        groups[primary].append(index)
    return groups


def _best_candidate_for_count(record, count):
    candidates = [
        item for item in record.get("candidates", [])
        if int(item["component_count"]) == int(count)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["cv_normalised_sse_mean"])


def _record_rank_ratio(record):
    ratios = [
        item.get("sigma2_over_sigma1")
        for item in record.get("rank_evidence", [])
    ]
    ratios = [float(item) for item in ratios if item is not None and item > 0]
    return float(np.median(ratios)) if ratios else None


def _aggregate_paired_difference(difference_sets, random_seed, runs=4000,
                                 confidence_level=0.95):
    """Hierarchical bootstrap over injections and their paired CV splits."""
    arrays = [np.asarray(item, dtype=float) for item in difference_sets if len(item)]
    if not arrays:
        return None
    rng = np.random.default_rng(int(random_seed))
    bootstrap = np.empty(max(500, int(runs)), dtype=float)
    for run in range(bootstrap.size):
        record_indices = rng.integers(0, len(arrays), size=len(arrays))
        record_means = []
        for record_index in record_indices:
            values = arrays[record_index]
            sampled = values[rng.integers(0, values.size, size=values.size)]
            record_means.append(float(np.mean(sampled)))
        bootstrap[run] = float(np.mean(record_means))
    alpha = min(max(1.0 - float(confidence_level), 1e-6), 0.5)
    return {
        "mean_simpler_minus_full_error": float(
            np.mean([np.mean(item) for item in arrays])
        ),
        "confidence_interval": np.quantile(
            bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0]
        ).tolist(),
        "probability_simpler_better": float(np.mean(bootstrap < 0.0)),
        "probability_full_better": float(np.mean(bootstrap > 0.0)),
        "injection_count": len(arrays),
    }


def build_batch_consensus(records, random_seed=20260910,
                          concentration_map=None):
    """Build cross-injection decisions without a fixed rank cutoff.

    Records must contain ``rt_start`` and ``rt_end``.  A reduction is marked
    applicable when the local single-segment selector supports a reduction.
    Repeated lower-rank evidence across injections strengthens the decision but
    is not required, because contaminants and sporadic noise can occur in only
    one injection. Candidate increases above the legacy output are never
    authorised.
    """
    input_records = [dict(item) for item in records]
    concentration_map = concentration_map or {}
    normalised_concentrations = {}
    for name, concentration in concentration_map.items():
        key = str(name).lower()
        normalised_concentrations[key] = float(concentration)
        normalised_concentrations[key.rsplit(".", 1)[0]] = float(concentration)

    def record_concentration(record):
        file_name = str(record.get("file", "")).lower()
        return normalised_concentrations.get(
            file_name,
            normalised_concentrations.get(file_name.rsplit(".", 1)[0]),
        )
    reference_records = [
        item for item in input_records
        if item.get("record_type") == "batch_rank_reference"
    ]
    records = [
        item for item in input_records if item.get("candidates")
    ]
    reference_ratios = [
        item.get("sigma2_over_sigma1") for item in reference_records
    ]
    reference_source = "disputed_segment_references"
    if not any(item is not None for item in reference_ratios):
        reference_ratios = [_record_rank_ratio(item) for item in records]
        reference_source = "disputed_records_fallback"
    rank_model = _fit_two_group_rank_model([
        np.nan if item is None else item for item in reference_ratios
    ])
    reference_posteriors = _rank_model_membership([
        np.nan if item is None else item for item in reference_ratios
    ], rank_model)
    for record, posterior in zip(reference_records, reference_posteriors):
        record["lower_rank_posterior"] = posterior
    decision_ratios = [_record_rank_ratio(item) for item in records]
    decision_posteriors = _rank_model_membership([
        np.nan if item is None else item for item in decision_ratios
    ], rank_model)
    for record, posterior in zip(
            records, decision_posteriors):
        record["lower_rank_posterior"] = posterior

    groups = _retention_groups(records)
    maximum_injection_count = max(
        (
            len({str(records[index].get("file", "")) for index in group})
            for group in groups
        ),
        default=1,
    )
    consensus_groups = []
    decisions = []
    for group_number, member_indices in enumerate(groups):
        members = [records[index] for index in member_indices]
        group_start = min(float(item["rt_start"]) for item in members)
        group_end = max(float(item["rt_end"]) for item in members)
        reference_by_file = {}
        for reference in reference_records:
            overlap = max(
                0.0,
                min(group_end, float(reference["rt_end"]))
                - max(group_start, float(reference["rt_start"])),
            )
            if overlap <= 0.0:
                continue
            file_name = str(reference.get("file", ""))
            previous = reference_by_file.get(file_name)
            if previous is None or overlap > previous[0]:
                reference_by_file[file_name] = (overlap, reference)
        group_references = [item[1] for item in reference_by_file.values()]
        evidence_members = group_references or members
        posteriors = [
            item["lower_rank_posterior"] for item in evidence_members
            if item.get("lower_rank_posterior") is not None
        ]
        mean_posterior = float(np.mean(posteriors)) if posteriors else 0.5
        lower_rank_population = mean_posterior > (1.0 - mean_posterior)
        unanimous_lower_rank = bool(posteriors) and all(
            posterior > (1.0 - posterior) for posterior in posteriors
        )
        unanimous_higher_rank = bool(posteriors) and all(
            posterior < (1.0 - posterior) for posterior in posteriors
        )
        concentration_evidence = sorted(
            (
                record_concentration(item),
                float(item["sigma2_over_sigma1"]),
                float(item["lower_rank_posterior"]),
            )
            for item in evidence_members
            if record_concentration(item) is not None
            and item.get("sigma2_over_sigma1") is not None
            and item.get("lower_rank_posterior") is not None
        )
        concentration_evidence.reverse()
        dilution_pattern_support = bool(
            len(concentration_evidence) >= 3
            and concentration_evidence[0][2]
            > (1.0 - concentration_evidence[0][2])
            and sum(
                posterior > (1.0 - posterior)
                for _, _, posterior in concentration_evidence
            ) > len(concentration_evidence) / 2.0
            and all(
                concentration_evidence[index][1]
                <= concentration_evidence[index + 1][1]
                for index in range(len(concentration_evidence) - 1)
            )
        )
        injection_count = len({
            str(item.get("file", "")) for item in evidence_members
        })

        paired_sets = []
        for member in members:
            counts = sorted({
                int(item["component_count"])
                for item in member.get("candidates", [])
            })
            if len(counts) < 2:
                continue
            simpler = _best_candidate_for_count(member, counts[0])
            full = _best_candidate_for_count(member, counts[-1])
            if simpler is None or full is None:
                continue
            if len(simpler.get("cv_fold_errors", [])) != len(
                    full.get("cv_fold_errors", [])):
                continue
            paired_sets.append(
                np.asarray(simpler["cv_fold_errors"], dtype=float)
                - np.asarray(full["cv_fold_errors"], dtype=float)
            )
        aggregate_cv = _aggregate_paired_difference(
            paired_sets, int(random_seed) + group_number
        )

        group_decisions = []
        for member in members:
            counts = sorted({
                int(item["component_count"])
                for item in member.get("candidates", [])
            })
            active_count = len(member.get("active_indices", []))
            legacy_count = int(member.get("legacy_component_count", active_count))
            single_selected_count = member.get("selected_component_count")
            local_reduction = bool(
                member.get("selection_action") == "reduce_component_count"
                or (
                    single_selected_count is not None
                    and int(single_selected_count) < legacy_count
                )
            )
            cross_injection_support = bool(
                injection_count >= 2
                and (unanimous_lower_rank or dilution_pattern_support)
            )
            # Cross concentration recurrence strengthens a local reduction
            # recommendation but is no longer a prerequisite.  A contaminant,
            # solvent peak, or sporadic noise feature may occur in one
            # injection only; its absence elsewhere must not force retention.
            allow_reduction = local_reduction
            target_count = (
                int(single_selected_count)
                if allow_reduction and single_selected_count is not None else
                counts[0]
                if allow_reduction and counts else
                active_count
            )
            # A shadow recommendation may never create a component that the
            # legacy path removed.  Such increases require separate evidence.
            target_count = min(int(target_count), legacy_count)
            selected_indices = tuple(member.get("selected_indices") or ())
            candidate = next(
                (
                    item for item in member.get("candidates", [])
                    if tuple(item.get("indices", ())) == selected_indices
                    and int(item.get("component_count", -1)) == target_count
                ),
                None,
            )
            if candidate is None:
                candidate = _best_candidate_for_count(member, target_count)
            applicable = bool(
                allow_reduction
                and candidate is not None
                and target_count < legacy_count
            )
            selection_authorised = candidate is not None
            decision = {
                "file": member.get("file"),
                "segment_index": member.get("segment_index"),
                "segment_number": member.get("segment_number"),
                "sigma2_over_sigma1": _record_rank_ratio(member),
                "lower_rank_posterior": member.get("lower_rank_posterior"),
                "single_segment_selected_count": member.get(
                    "selected_component_count"
                ),
                "single_segment_evidence_grade": member.get(
                    "evidence_grade", member.get("selection_confidence")
                ),
                "legacy_component_count": legacy_count,
                "selected_component_count": (
                    int(candidate["component_count"]) if candidate is not None
                    else legacy_count
                ),
                "selected_indices": (
                    list(candidate["indices"]) if candidate is not None else None
                ),
                # An application-ready consensus locks both outcomes. A
                # reviewed retain decision must bypass the legacy overlap
                # deletion path just as a reviewed reduction must be applied.
                "authorise_selection": selection_authorised,
                "apply": applicable,
                "reason": (
                    "repeated_lower_rank_population"
                    if applicable and cross_injection_support and unanimous_lower_rank else
                    "lower_rank_signal_increases_with_dilution"
                    if applicable and cross_injection_support and dilution_pattern_support else
                    "single_segment_local_evidence"
                    if applicable else
                    "retain_without_local_reduction_evidence"
                ),
                "local_reduction_evidence": local_reduction,
                "cross_injection_support": cross_injection_support,
            }
            decisions.append(decision)
            group_decisions.append(decision)

        consistency = abs(2.0 * mean_posterior - 1.0)
        clarity = float(np.mean([
            abs(2.0 * item - 1.0) for item in posteriors
        ])) if posteriors else 0.0
        consensus_groups.append({
            "group_number": group_number,
            "rt_start": min(float(item["rt_start"]) for item in members),
            "rt_end": max(float(item["rt_end"]) for item in members),
            "injection_count": injection_count,
            "files": sorted({
                str(item.get("file", "")) for item in evidence_members
            }),
            "rank_reference_members": [
                {
                    "file": item.get("file"),
                    "segment_index": item.get("segment_index"),
                    "sigma2_over_sigma1": item.get("sigma2_over_sigma1"),
                    "lower_rank_posterior": item.get("lower_rank_posterior"),
                }
                for item in group_references
            ],
            "mean_lower_rank_posterior": mean_posterior,
            "rank_population": (
                "lower_rank_consistent" if unanimous_lower_rank else
                "lower_rank_dilution_pattern" if dilution_pattern_support else
                "higher_rank_consistent" if unanimous_higher_rank else
                "mixed_rank_membership"
            ),
            "mean_rank_population": (
                "lower_rank" if lower_rank_population else "higher_rank"
            ),
            "unanimous_lower_rank": unanimous_lower_rank,
            "dilution_pattern_support": dilution_pattern_support,
            "concentration_rank_evidence": [
                {
                    "concentration": concentration,
                    "sigma2_over_sigma1": ratio,
                    "lower_rank_posterior": posterior,
                }
                for concentration, ratio, posterior in concentration_evidence
            ],
            "aggregate_paired_cv": aggregate_cv,
            "evidence_score": float(
                consistency * clarity * injection_count / maximum_injection_count
            ),
            "decisions": group_decisions,
        })

    scores = np.asarray(
        [item["evidence_score"] for item in consensus_groups], dtype=float
    )
    if scores.size >= 3 and np.ptp(scores) > 0:
        lower_grade, upper_grade = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
        for item in consensus_groups:
            if item["evidence_score"] >= upper_grade:
                item["evidence_grade"] = "strong"
            elif item["evidence_score"] >= lower_grade:
                item["evidence_grade"] = "moderate"
            else:
                item["evidence_grade"] = "weak"
    else:
        for item in consensus_groups:
            item["evidence_grade"] = "weak"

    return {
        "method": "adaptive_rank_mixture_cross_injection_consensus_v1",
        "rank_model": rank_model,
        "rank_reference_count": len(reference_ratios),
        "rank_reference_source": reference_source,
        "decision_record_count": len(records),
        "groups": consensus_groups,
        "decisions": decisions,
        "limitations": [
            "lower_rank_posterior_is_batch_membership_not_chemical_truth_probability",
            "strictly_proportional_spectra_with_complete_coelution_are_not_identifiable",
            "cross_injection_recurrence_supports_but_is_not_required_for_local_reduction",
            "component_count_increases_are_not_authorised",
        ],
    }

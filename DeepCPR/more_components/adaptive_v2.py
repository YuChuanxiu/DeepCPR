"""Improved residual-guided adaptive resolution for DeepCPR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from ..model_runtime import load_model_auto, tensorflow_available
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable


_DEEPCPR = None
_TF = None


def _load_deepcpr():
    """Return the original ``DeepCPR.DeepCPR`` module lazily."""
    global _DEEPCPR
    if _DEEPCPR is None:
        from .. import DeepCPR as _mod

        _DEEPCPR = _mod
    return _DEEPCPR


def _load_tf():
    global _TF
    if _TF is None:
        import tensorflow as _tf

        _TF = _tf
    return _TF


def _cosine(a, b) -> float:
    """Cosine similarity between two flattened arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


@dataclass
class SegmentResolution:
    """Result of resolving one local segment (one residual round)."""

    chromatograms: np.ndarray
    spectra: np.ndarray
    reconstruction: np.ndarray
    r2: float
    wayname: str = "non_frr"
    predicted_profiles: Optional[np.ndarray] = None

    @property
    def n_components(self) -> int:
        return int(self.chromatograms.shape[1])


@dataclass
class AdaptiveResult:
    """Final result of one local-segment adaptive resolution."""

    chromatograms: np.ndarray
    spectra: np.ndarray
    reconstruction: np.ndarray
    residual: np.ndarray
    signal: np.ndarray
    iterations: int
    history: list = field(default_factory=list)

    @property
    def n_components(self) -> int:
        return int(self.chromatograms.shape[1])

    @property
    def r2(self) -> float:
        denominator = float(np.sum(self.signal**2))
        if denominator <= 1e-12:
            return 1.0
        return float(
            np.clip(1.0 - np.sum(self.residual**2) / denominator, -np.inf, 1.0)
        )


def predict_segment_profiles(chrom_seg, model, model_points: int = 128) -> np.ndarray:
    """Predict at most five initial chromatographic profiles.

    ``chrom_seg`` is the *unnormalized* current data/residual.  A normalized
    copy is centre-padded to ``128 x 800`` only for inference.  The returned
    ``C_0`` is cropped back to the segment length and is therefore in the same
    normalized-profile units used by the original ``DeepCPR`` function.
    """

    segment = np.asarray(chrom_seg, dtype=np.float32)
    if segment.ndim != 2:
        raise ValueError(f"expected (scans, m/z), got {segment.shape}")
    n_scan, n_mz = segment.shape
    if n_mz != 800:
        raise ValueError(f"DeepCPR expects 800 m/z channels, got {n_mz}")
    if n_scan > model_points:
        raise ValueError(
            f"a local segment has {n_scan} scans; maximum is {model_points}"
        )

    scale = float(np.max(segment))
    normalized = segment / scale if scale > 0 else segment.copy()

    offset = (model_points - n_scan) // 2
    batch = np.zeros((1, model_points, 1, n_mz), dtype=np.float32)
    batch[0, offset : offset + n_scan, 0, :] = normalized

    try:
        output = model.predict(batch, verbose=0)
    except TypeError:
        output = model.predict(batch)
    output = np.asarray(output)

    if output.ndim == 4:
        output = output[0, :, 0, :]
    elif output.ndim == 3:
        output = output[0]
    output = output[offset : offset + n_scan, :]
    C_0 = np.asarray(output, dtype=np.float32)
    if C_0.shape[0] != n_scan:
        raise ValueError(
            f"predictor returned {C_0.shape[0]} scan points; expected {n_scan}"
        )
    C_0[C_0 < 0] = 0

    return C_0


def _resolve_helpers():
    mod = _load_deepcpr()
    return {
        "peak_preprocess": mod.peak_preprocess,
        "peak_trans": mod.peak_trans,
        "peak_halfremove": mod.peak_halfremove,
        "testt": mod.testt,
        "tail_fix": mod.tail_fix,
        "com_find": mod.com_find,
        "peak_find": mod.peak_find,
        "ITTFA_PRO": mod.ITTFA_PRO,
        "peak_oddremove": mod.peak_oddremove,
        "peak_rightlap": mod.peak_rightlap,
        "peak_dislap": mod.peak_dislap,
        "dishalfremove": mod.dishalfremove,
        "dynamic_FRR": mod.dynamic_FRR,
        "fnnls": mod.fnnls,
        "fnnls_batch": mod.fnnls_batch,
    }


def _nnls_spectra(C, X):
    fnnls_batch = _resolve_helpers()["fnnls_batch"]
    return fnnls_batch(
        np.dot(C.T, C), np.dot(C.T, X), tole="None"
    ).astype(np.float32, copy=False)


ITTFA_COLLAPSE_COSINE = 0.98


def _safe_ittfa_pro(chrom_seg, C_in, COM, collapse_cosine=None):
    """Run ITTFA_PRO, falling back to direct NNLS on pathological inputs.

    ``ITTFA_PRO`` contains Gaussian tail-refinement branches that assume a
    non-empty interval between the apex and peak edges.  Those branches can
    raise ``ValueError`` on highly irregular residual profiles during the
    adaptive loop.  A direct NNLS fallback keeps the loop robust without
    changing the original ``DeepCPR.py`` implementation.
    """

    collapse_cosine = (
        ITTFA_COLLAPSE_COSINE if collapse_cosine is None else collapse_cosine
    )
    C_in = np.asarray(C_in, dtype=np.float32)
    try:
        C_ed, St = _resolve_helpers()["ITTFA_PRO"](chrom_seg, C_in, COM)
    except Exception:
        # Irregular residual profiles can leave an empty Gaussian-tail interval.
        C_ed = C_in
        St = _nnls_spectra(C_ed, chrom_seg)
        return C_ed, St

    C_ed = np.asarray(C_ed, dtype=np.float32)
    C_out = C_ed.copy()
    collapsed = False
    for i in range(COM):
        for j in range(i + 1, COM):
            # Preserve distinct initial profiles if ITTFA collapses them.
            if _cosine(C_ed[:, i], C_ed[:, j]) >= collapse_cosine:
                C_out[:, i] = C_in[:, i]
                C_out[:, j] = C_in[:, j]
                collapsed = True

    if collapsed:
        St = _regularized_nnls(C_out, chrom_seg)
    return C_out, St


def _r2(Xhat, X) -> float:
    denominator = np.var(X, ddof=1)
    if denominator <= 1e-12:
        return 1.0
    return float(1.0 - np.var(Xhat - X, ddof=1) / denominator)


def resolve_one_segment(
    chrom_seg, C_0, debug_hook=None, refine_fn=None
) -> Optional[SegmentResolution]:
    """Resolve one local segment using the original DeepCPR post-processing.

    This function mirrors the per-segment body of ``DeepCPR()`` (from the
    initial predicted profiles ``C_0`` to the final ``C_ed``, ``St``,
    reconstruction and R2).  ``chrom_seg`` is the unnormalized data/residual
    and ``C_0`` is the normalized initial-profile matrix.
    """

    h = _resolve_helpers()
    refine = refine_fn or _safe_ittfa_pro

    def emit(name: str, C_mat) -> None:
        if debug_hook is not None:
            debug_hook(name, np.asarray(C_mat, dtype=np.float32))

    C = h["peak_preprocess"](C_0)
    emit("peak_preprocess", C)
    if np.all(C == 0):
        return None

    zeros = lambda: np.zeros_like(C)

    C_in = np.copy(h["peak_trans"](C, zeros()))
    emit("peak_trans", C_in)

    C_out, halfdo = h["peak_halfremove"](C_in)
    if halfdo == "Done":
        if np.all(C_out == 0):
            return None
        C_in = np.copy(h["peak_trans"](C_out, zeros()))
    else:
        C_in = C_out
    emit("peak_halfremove", C_in)

    C_out, testdo, lapdo, lapnum = h["testt"](C_in)
    if testdo == "Done":
        if np.all(C_out == 0):
            return None
        C_in = np.copy(h["peak_trans"](C_out, zeros()))
    else:
        C_in = C_out
    emit("testt", C_in)

    C_in = h["tail_fix"](C_in)
    emit("tail_fix", C_in)

    COM = h["com_find"](C_in)
    if COM == 0:
        return None
    C_ed_it, St_it = refine(chrom_seg, C_in, COM)
    emit("ittfa_1", C_ed_it)

    C_out, odddo = h["peak_oddremove"](C_in, C_ed_it)
    if odddo == "Done":
        if np.all(C_out == 0):
            return None
        C_in = np.copy(h["peak_trans"](C_out, zeros()))
        COM = h["com_find"](C_in)
        if COM == 0:
            return None
        C_ed_it, St_it = refine(chrom_seg, C_in, COM)
        C_out, disrightlapdo, disrightlapnum = h["peak_rightlap"](C_in, C_ed_it)
        if disrightlapdo == "Done":
            if np.all(C_out == 0):
                return None
            C_in = np.copy(h["peak_trans"](C_out, zeros()))
        else:
            C_in = C_out
    else:
        C_out, disrightlapdo, disrightlapnum = h["peak_rightlap"](C_in, C_ed_it)
        if disrightlapdo == "Done":
            if np.all(C_out == 0):
                return None
            C_in = np.copy(h["peak_trans"](C_out, zeros()))
        else:
            C_in = C_out

    emit("after_oddremove_rightlap", C_in)

    C_in_0 = np.copy(C_in)
    C_pre_0 = np.copy(C_in)

    COM = h["com_find"](C_in)
    if COM > 1:
        C_ed_it, St_it = refine(chrom_seg, C_in, COM)
        C_out, dislapdo, dislapnum = h["peak_dislap"](C_in, C_ed_it)
        dislapnum = list(set(dislapnum))
        C_out, dishalfdo, dishalfnum = h["dishalfremove"](C_in, C_ed_it, threshold=0.05)
        dishalfnum = list(set(dishalfnum))
    else:
        dislapdo = "None"
        dishalfdo = "None"

    # Compare direct NNLS, ITTFA, and eligible FRR candidates by R2.
    chrom_resol = None
    C_ed = None
    R2 = None
    St = None
    C_p = None
    wayname = "non_frr"

    def assign(cr, c, r2, s, cp):
        nonlocal chrom_resol, C_ed, R2, St, C_p
        chrom_resol, C_ed, R2, St, C_p = cr, c, r2, s, cp

    # Branch 1: overlapping profiles were also flagged as half peaks.
    if lapdo == "Done" and (dislapdo == "Done" or dishalfdo == "Done"):
        for k in dislapnum:
            C_in_0[:, k] = 0
            if k in lapnum:
                C_pre_0[:, k] = 0
        for k in dishalfnum:
            C_in_0[:, k] = 0
            if k in lapnum:
                C_pre_0[:, k] = 0

        # Always retain at least the strongest candidate profile.
        if h["com_find"](C_pre_0) == 0:
            reserve = int(np.argmax([np.max(C_in[:, q]) for q in range(COM)]))
            C_pre_0 = zeros()
            C_pre_0[:, reserve] = C_in[:, reserve]

        C_pre = np.copy(h["peak_trans"](np.copy(C_pre_0), zeros()))
        emit("candidate_pre", C_pre)
        St_pre = _nnls_spectra(C_pre, chrom_seg)
        chrom_resol_pre = np.dot(C_pre, St_pre)
        R2_pre = _r2(chrom_resol_pre, chrom_seg)

        if h["com_find"](C_in_0) == 0:
            reserve = int(np.argmax([np.max(C_in[:, q]) for q in range(COM)]))
            C_in_0 = zeros()
            C_in_0[:, reserve] = C_in[:, reserve]

        COM = h["com_find"](C_in_0)
        C_in_it = h["peak_trans"](C_in_0, zeros())
        C_ed_it, St_it = refine(chrom_seg, C_in_it, COM)
        emit("candidate_ittfa_branch", C_ed_it)
        chrom_resol_it = np.dot(C_ed_it, St_it)
        R2_it = _r2(chrom_resol_it, chrom_seg)

        COM = h["com_find"](C_pre)
        peak_st, peak_ed = h["peak_find"](C_pre)
        if COM > 1:
            if R2_it <= 0.99 and COM < 4:
                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = h["dynamic_FRR"](
                    chrom_seg, C_pre, peak_st, peak_ed, COM
                )
                emit("candidate_frr", C_ed_frr)
                if R2_frr > R2_it:
                    assign(chrom_resol_frr, C_ed_frr, R2_frr, St_frr, C_pre)
                    wayname = "frr"
                else:
                    assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_in_it)
            elif COM > 3 and R2_it < 0.7 and R2_pre < 0.9:
                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = h["dynamic_FRR"](
                    chrom_seg, C_pre, peak_st, peak_ed, COM
                )
                emit("candidate_frr", C_ed_frr)
                if R2_frr > R2_it:
                    assign(chrom_resol_frr, C_ed_frr, R2_frr, St_frr, C_pre)
                    wayname = "frr"
                else:
                    assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_in_it)
            else:
                assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_in_it)

            if R2_pre > R2:
                assign(chrom_resol_pre, C_pre, R2_pre, St_pre, C_pre)
                wayname = "non_frr"
        elif COM == 1:
            assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_in_it)
            if R2_pre > R2_it:
                assign(chrom_resol_pre, C_pre, R2_pre, St_pre, C_pre)
                wayname = "non_frr"

    # Branch 2: no overlap cleanup was triggered.
    elif (lapdo == "Done" and dislapdo == "None" and dishalfdo == "None") or (
        lapdo == "None" and dislapdo == "None" and dishalfdo == "None"
    ):
        C_pre = np.copy(C_pre_0)
        emit("candidate_pre", C_pre)
        St_pre = _nnls_spectra(C_pre, chrom_seg)
        chrom_resol_pre = np.dot(C_pre, St_pre)
        R2_pre = _r2(chrom_resol_pre, chrom_seg)

        COM = h["com_find"](C_pre)
        C_ed_it, St_it = refine(chrom_seg, C_pre, COM)
        emit("candidate_ittfa_branch", C_ed_it)
        chrom_resol_it = np.dot(C_ed_it, St_it)
        R2_it = _r2(chrom_resol_it, chrom_seg)

        COM = h["com_find"](C_pre)
        peak_st, peak_ed = h["peak_find"](C_pre)
        if COM > 1:
            if R2_it <= 0.99 and COM < 4:
                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = h["dynamic_FRR"](
                    chrom_seg, C_pre, peak_st, peak_ed, COM
                )
                emit("candidate_frr", C_ed_frr)
                if R2_frr > R2_it:
                    assign(chrom_resol_frr, C_ed_frr, R2_frr, St_frr, C_pre)
                    wayname = "frr"
                else:
                    assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_pre)
            elif COM > 3 and R2_it < 0.7 and R2_pre < 0.9:
                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = h["dynamic_FRR"](
                    chrom_seg, C_pre, peak_st, peak_ed, COM
                )
                emit("candidate_frr", C_ed_frr)
                if R2_frr > R2_it:
                    assign(chrom_resol_frr, C_ed_frr, R2_frr, St_frr, C_pre)
                    wayname = "frr"
                else:
                    assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_pre)
            else:
                assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_pre)

            if R2_pre > R2:
                assign(chrom_resol_pre, C_pre, R2_pre, St_pre, C_pre)
                wayname = "non_frr"
        elif COM == 1:
            assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_pre)
            if R2_pre > R2_it:
                assign(chrom_resol_pre, C_pre, R2_pre, St_pre, C_pre)
                wayname = "non_frr"

    # Branch 3: overlap cleanup triggered without a coincident-peak flag.
    elif lapdo == "None" and (dislapdo == "Done" or dishalfdo == "Done"):
        for k in dislapnum:
            C_in_0[:, k] = 0
        for k in dishalfnum:
            C_in_0[:, k] = 0

        if h["com_find"](C_in_0) == 0:
            reserve = int(np.argmax([np.max(C_in[:, q]) for q in range(COM)]))
            C_in_0 = zeros()
            C_in_0[:, reserve] = C_in[:, reserve]

        COM = h["com_find"](C_in_0)
        C_in_it_dis = h["peak_trans"](C_in_0, zeros())
        C_ed_it_dis, St_it_dis = refine(chrom_seg, C_in_it_dis, COM)
        chrom_resol_it_dis = np.dot(C_ed_it_dis, St_it_dis)
        R2_it_dis = _r2(chrom_resol_it_dis, chrom_seg)

        COM = h["com_find"](C_pre_0)
        C_in_it = C_pre_0
        C_ed_it, St_it = refine(chrom_seg, C_in_it, COM)
        emit("candidate_ittfa_branch", C_ed_it)
        chrom_resol_it = np.dot(C_ed_it, St_it)
        R2_it = _r2(chrom_resol_it, chrom_seg)
        if R2_it < R2_it_dis:
            chrom_resol = chrom_resol_it_dis
            C_ed_it = C_ed_it_dis
            St_it = St_it_dis
            R2_it = R2_it_dis
            C_in_it = C_in_it_dis

        C_pre = np.copy(C_pre_0)
        emit("candidate_pre", C_pre)
        St_pre = _nnls_spectra(C_pre, chrom_seg)
        chrom_resol_pre = np.dot(C_pre, St_pre)
        R2_pre = _r2(chrom_resol_pre, chrom_seg)

        COM = h["com_find"](C_pre)
        peak_st, peak_ed = h["peak_find"](C_pre)
        if COM > 1:
            if R2_it <= 0.99 and COM < 4:
                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = h["dynamic_FRR"](
                    chrom_seg, C_pre, peak_st, peak_ed, COM
                )
                emit("candidate_frr", C_ed_frr)
                if R2_frr > R2_it:
                    assign(chrom_resol_frr, C_ed_frr, R2_frr, St_frr, C_pre)
                    wayname = "frr"
                else:
                    assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_in_it)
            else:
                assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_in_it)

            if R2_pre > R2:
                assign(chrom_resol_pre, C_pre, R2_pre, St_pre, C_pre)
                wayname = "non_frr"
        elif COM == 1:
            assign(chrom_resol_it, C_ed_it, R2_it, St_it, C_in_it)
            if R2_pre > R2_it:
                assign(chrom_resol_pre, C_pre, R2_pre, St_pre, C_pre)
                wayname = "non_frr"

    # Defensive fallback for an unassigned branch result.
    if C_ed is None:
        if np.all(C_pre_0 == 0):
            return None
        C_pre = np.copy(C_pre_0)
        St_pre = _nnls_spectra(C_pre, chrom_seg)
        chrom_resol_pre = np.dot(C_pre, St_pre)
        R2_pre = _r2(chrom_resol_pre, chrom_seg)
        assign(chrom_resol_pre, C_pre, R2_pre, St_pre, C_pre)
        wayname = "non_frr"

    emit("final", C_ed)
    return SegmentResolution(
        chromatograms=np.asarray(C_ed, dtype=np.float32),
        spectra=np.asarray(St, dtype=np.float32),
        reconstruction=np.asarray(chrom_resol, dtype=np.float32),
        r2=float(R2),
        wayname=wayname,
        predicted_profiles=np.asarray(C_p, dtype=np.float32) if C_p is not None else None,
    )


def _regularized_nnls(C, X, reg: float = 1e-4):
    """Stable non-negative least-squares spectrum estimate.

    The hand-written ``fnnls`` in ``DeepCPR.py`` can produce an unbounded,
    nearly constant spectrum for a profile that is almost collinear with a
    constant/offset structure.  This ridge-regularized estimate is bounded and
    is used for the final joint refinement.
    """

    C = np.asarray(C, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if C.shape[1] == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)
    # Ridge regularization stabilizes nearly collinear profile columns.
    gram = C.T @ C + reg * np.eye(C.shape[1], dtype=np.float64)
    try:
        S = np.linalg.solve(gram, C.T @ X)
    except np.linalg.LinAlgError:
        S = np.linalg.lstsq(gram, C.T @ X, rcond=None)[0]
    S = np.maximum(S, 0.0)
    return S.astype(np.float32)


def _spectrum_sanity_filter(
    C,
    S,
    *,
    max_nonzero_ratio: float = 0.5,
    max_cv_flat: float = 0.5,
    max_scale_ratio: float = 100.0,
):
    """Drop components whose spectra are pathological (huge or near-constant)."""

    C = np.asarray(C, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    if C.shape[1] == 0:
        return C, S

    norms = np.linalg.norm(S, axis=1)
    nonzero_norms = norms[norms > 0]
    median_norm = float(np.median(nonzero_norms)) if nonzero_norms.size else 1.0

    keep = np.ones(C.shape[1], dtype=bool)
    for j in range(C.shape[1]):
        s = S[j, :]
        nonzero_ratio = float(np.count_nonzero(s > 1e-8) / max(s.size, 1))
        mean = float(np.mean(s))
        std = float(np.std(s))
        cv = std / (mean + 1e-12)

        if median_norm > 0 and norms[j] > max_scale_ratio * median_norm:
            keep[j] = False
        elif nonzero_ratio > max_nonzero_ratio and cv < max_cv_flat:
            keep[j] = False

    return C[:, keep], S[keep]


def _taper_flat_tail(
    C,
    *,
    tail_window: int = 12,
    flat_tol: float = 0.02,
    tail_level: float = 0.05,
    decay_window: int = 8,
):
    """Taper flat, elevated profile tails back to zero.

    ``ITTFA_PRO`` can leave a nearly horizontal tail when the Gaussian sigma
    degenerates.  This post-processing detects a flat elevated head/tail and
    multiplies it by a cosine window that decays to zero at the boundary.
    """

    C = np.array(np.asarray(C, dtype=np.float32), copy=True)
    n, k = C.shape
    if n == 0 or k == 0:
        return C

    for j in range(k):
        col = C[:, j]
        m = float(np.max(col))
        if m <= 1e-12:
            continue

        tw = min(tail_window, n)
        tail = col[-tw:]
        if float(np.std(tail)) / m < flat_tol and float(np.mean(tail)) / m > tail_level:
            dw = min(decay_window, n)
            window = 0.5 * (1.0 + np.cos(np.pi * np.arange(dw) / max(dw - 1, 1)))
            C[-dw:, j] = col[-dw:] * window.astype(np.float32)

        head = col[:tw]
        if float(np.std(head)) / m < flat_tol and float(np.mean(head)) / m > tail_level:
            dw = min(decay_window, n)
            window = 0.5 * (1.0 - np.cos(np.pi * np.arange(dw) / max(dw - 1, 1)))
            C[:dw, j] = col[:dw] * window.astype(np.float32)

    return C


def _remove_embedded_profiles(C, S, min_selectivity: float = 0.4):
    """Drop profiles that have no selectivity region (fully embedded peaks).

    A fully embedded/co-eluted spurious component never dominates the mixture
    at any retention-time scan.  ``selectivity_j`` is the maximum, over scans,
    of ``C[j, t] / sum_i C[i, t]``.  Profiles below ``min_selectivity`` are
    removed.
    """

    C = np.asarray(C, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    if C.shape[1] == 0:
        return C, S

    denominator = np.sum(C, axis=1, keepdims=True) + 1e-12
    keep = []
    for j in range(C.shape[1]):
        selectivity = float(np.max(C[:, j] / denominator[:, 0]))
        if selectivity >= min_selectivity:
            keep.append(j)
    if not keep:
        return C, S
    return C[:, keep], S[keep]


def _profile_shape_filter(
    C,
    *,
    max_support: int = 80,
    max_fwhm_factor: float = 2.5,
):
    """Drop broad, baseline-like profiles before the final refinement."""

    C = np.asarray(C, dtype=np.float32)
    n, k = C.shape
    if k == 0:
        return C

    fwhms = []
    supports = []
    for j in range(k):
        col = C[:, j]
        m = float(np.max(col))
        if m <= 1e-12:
            fwhms.append(0)
            supports.append(0)
            continue
        above = np.where(col >= 0.5 * m)[0]
        fwhm = int(above[-1] - above[0] + 1) if above.size else 0
        support = int(np.count_nonzero(col > 0.01 * m))
        fwhms.append(fwhm)
        supports.append(support)

    positive_fwhms = [f for f in fwhms if f > 0]
    median_fwhm = float(np.median(positive_fwhms)) if positive_fwhms else 1.0

    keep = []
    for j in range(k):
        col = C[:, j]
        apex = int(np.argmax(col))
        reject = False
        if supports[j] > max_support:
            reject = True
        if fwhms[j] > max_fwhm_factor * median_fwhm and fwhms[j] > 0:
            reject = True
        if (apex == 0 or apex == n - 1) and fwhms[j] > 1.5 * median_fwhm:
            reject = True
        if not reject:
            keep.append(j)

    if not keep:
        return C
    return C[:, keep]


def _drop_zero_columns(C):
    """Remove all-zero profile columns."""

    C = np.asarray(C, dtype=np.float32)
    if C.shape[1] == 0:
        return C
    keep = [j for j in range(C.shape[1]) if np.any(C[:, j] != 0)]
    if not keep:
        return np.zeros((C.shape[0], 0), dtype=np.float32)
    return C[:, keep]


def _final_cleanup_profiles(X, C_all, r2_margin: float = 0.005):
    """Reuse the original DeepCPR shape-cleanup functions on the merged set.

    Unlike ``resolve_one_segment``, this does not rerun the five-output /
    FRR branch selection; it only applies the generic per-column cleanup
    functions that work for an arbitrary number of components.
    """

    C = _drop_zero_columns(C_all)
    if C.shape[1] == 0:
        return C

    def r2_of(Cmat):
        S = _regularized_nnls(Cmat, X)
        rec = Cmat @ S
        ss_res = float(np.sum((X - rec) ** 2))
        ss_tot = float(np.sum(X**2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    before = r2_of(C)

    h = _resolve_helpers()
    C_tail = h["tail_fix"](C)
    C_tail = _drop_zero_columns(C_tail)
    if C_tail.shape[1] == 0:
        return C

    C_ed_ref, _ = _safe_ittfa_pro(X, C_tail, int(C_tail.shape[1]))
    C_out, _ = h["peak_oddremove"](C_tail, C_ed_ref)
    C_out, _, _ = h["peak_rightlap"](C_out, C_ed_ref)
    C_out, _, _ = h["peak_dislap"](C_out, C_ed_ref)
    C_out, _, _ = h["dishalfremove"](C_out, C_ed_ref, threshold=0.05)
    C_out = _drop_zero_columns(C_out)
    if C_out.shape[1] == 0:
        return C

    after = r2_of(C_out)
    return C_out if after >= before - r2_margin else C


def _residual_has_peak(
    residual,
    *,
    min_prominence: float = 2.0,
):
    """Return True when the residual TIC still contains a peak-like bump.

    This uses a pure-NumPy robust z-score: the highest TIC point must stand at
    least ``min_prominence`` median-absolute-deviation units above the median.
    A flat baseline/noise-only residual has a small score and is rejected.
    """

    tic = np.maximum(np.asarray(residual, dtype=np.float64).sum(axis=1), 0.0)
    if float(np.max(tic)) <= 1e-12:
        return False
    med = float(np.median(tic))
    mad = float(np.median(np.abs(tic - med))) + 1e-12
    strength = (float(np.max(tic)) - med) / mad
    return strength >= min_prominence


def _deduplicate_profiles(
    profile_blocks,
    spectrum_blocks=None,
    *,
    apex_tolerance: int = 2,
    profile_cosine: float = 0.95,
    spectral_cosine: float = 0.90,
):
    """Merge round-wise profiles and remove cross-round duplicates."""

    C_all = np.concatenate(
        [np.asarray(b, dtype=np.float32) for b in profile_blocks], axis=1
    )
    S_all = None
    if spectrum_blocks is not None:
        S_all = np.concatenate(
            [np.asarray(s, dtype=np.float32) for s in spectrum_blocks], axis=0
        )

    strengths = np.asarray(
        [
            float(np.max(C_all[:, j]))
            * (
                float(np.linalg.norm(S_all[j, :]))
                if S_all is not None
                else 1.0
            )
            for j in range(C_all.shape[1])
        ],
        dtype=np.float64,
    )

    keep = []
    for j in range(C_all.shape[1]):
        candidate = C_all[:, j]
        apex = int(np.argmax(candidate))
        duplicate_index = None
        for ki, k in enumerate(keep):
            existing = C_all[:, k]
            if abs(apex - int(np.argmax(existing))) <= apex_tolerance:
                pc = _cosine(candidate, existing)
                sc = _cosine(S_all[j], S_all[k]) if S_all is not None else 0.0
                if pc >= profile_cosine or sc >= spectral_cosine:
                    duplicate_index = ki
                    break
        if duplicate_index is None:
            keep.append(j)
        elif strengths[j] > strengths[keep[duplicate_index]]:
            keep[duplicate_index] = j

    if not keep:
        keep = []
    C_kept = C_all[:, keep] if keep else np.zeros((C_all.shape[0], 0), dtype=np.float32)
    S_kept = S_all[keep] if (S_all is not None and keep) else None
    return C_kept, S_kept, keep


def _final_ittfa_refine(original, C_all):
    """Jointly re-estimate spectra of all retained profiles against X0."""

    COM = int(C_all.shape[1])
    if COM == 0:
        return np.asarray(C_all, dtype=np.float32), np.zeros(
            (0, original.shape[1]), dtype=np.float32
        )
    C_final, _ = _safe_ittfa_pro(original, C_all, COM)
    S_final = _regularized_nnls(C_final, original)
    return np.asarray(C_final, dtype=np.float32), np.asarray(S_final, dtype=np.float32)


def _nnls_solve(A, B):
    """Solve NNLS column-wise: minimize ||A x - B[:,j]||^2 for x >= 0."""

    from scipy.optimize import nnls

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    out = np.zeros((A.shape[1], B.shape[1]), dtype=np.float64)
    for j in range(B.shape[1]):
        out[:, j], _ = nnls(A, B[:, j])
    return out


def _mcr_als_refine(
    X,
    C0,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    normalize_C: bool = True,
    use_unimodality: bool = True,
):
    """MCR-ALS joint refinement initialized with chromatograms.

    Alternate non-negative least-squares updates of chromatograms and spectra:
        S = NNLS(C, X)
        C = NNLS(S^T, X^T)
    Each iteration applies unimodality to C and transfers column scaling to S
    to prevent scale drift and multimodal chromatographic profiles.
    """

    X = np.maximum(np.asarray(X, dtype=np.float64), 0.0)
    C = np.maximum(np.asarray(C0, dtype=np.float64), 0.0)
    if C.shape[1] == 0:
        return C.astype(np.float32), np.zeros((0, X.shape[1]), dtype=np.float32)

    unimod = _load_deepcpr().unimod
    S = _nnls_solve(C, X)
    prev_residual = float("inf")

    for _ in range(max_iter):
        C = _nnls_solve(S.T, X.T).T

        if use_unimodality:
            C = unimod(C, 1.1, 2)

        S = _nnls_solve(C, X)

        if normalize_C:
            norms = np.linalg.norm(C, axis=0)
            norms[norms == 0] = 1.0
            C = C / norms[None, :]
            S = S * norms[:, None]

        reconstruction = C @ S
        residual = float(np.linalg.norm(X - reconstruction))
        if prev_residual != float("inf") and prev_residual - residual < tol * max(
            prev_residual, 1e-12
        ):
            break
        prev_residual = residual

    return C.astype(np.float32), S.astype(np.float32)


def adaptive_resolve_segment_v2(
    original,
    model,
    *,
    max_iterations: int = 4,
    residual_tolerance: float = 0.03,
    min_improvement: float = 0.01,
    max_components: int = 32,
    min_component_fraction: float = 0.002,
    enable_peak_stop: bool = True,
    min_residual_peak_prominence: float = 2.0,
    dedup_apex_tolerance: int = 2,
    dedup_profile_cosine: float = 0.95,
    dedup_spectral_cosine: float = 0.90,
    max_nonzero_ratio: float = 0.5,
    max_cv_flat: float = 0.5,
    max_spectrum_scale: float = 100.0,
    tail_window: int = 12,
    tail_level: float = 0.05,
    enable_flat_tail_taper: bool = False,
    final_cleanup: bool = False,
    final_cleanup_r2_margin: float = 0.005,
    remove_embedded_peaks: bool = True,
    min_selectivity: float = 0.4,
    max_support: int = 80,
    max_fwhm_factor: float = 2.5,
    final_refine_method: str = "ittfa",
    resolve_fn: Optional[Callable] = None,
    final_refine_fn: Optional[Callable] = None,
) -> AdaptiveResult:
    """Resolve a local GC-MS segment with a data-dependent component count.

    Every round calls ``predict_segment_profiles`` and the full
    ``resolve_one_segment`` pipeline, then computes the new residual from the
    final reconstruction of that round.  The relative residual size is judged
    *before* normalizing for the next model call.
    """

    X = np.maximum(np.asarray(original, dtype=np.float32), 0.0)
    if X.ndim != 2 or X.shape[1] != 800 or X.shape[0] > 128:
        raise ValueError(
            f"original must be (scans<=128, 800), got {X.shape}"
        )

    original_norm = float(np.linalg.norm(X))
    if original_norm <= 1e-12:
        empty_C = np.zeros((X.shape[0], 0), dtype=np.float32)
        empty_S = np.zeros((0, X.shape[1]), dtype=np.float32)
        return AdaptiveResult(
            empty_C, empty_S, np.zeros_like(X), X.copy(), X, 0, []
        )

    resolve_fn = resolve_fn or resolve_one_segment
    if final_refine_fn is None:
        final_refine_fn = (
            _mcr_als_refine if final_refine_method == "mcr_als" else _final_ittfa_refine
        )

    residual = X.copy()
    profile_blocks = []
    spectrum_blocks = []
    history = []

    # Repeatedly resolve the residual until signal or quality criteria stop.
    for iteration in range(max_iterations):
        before = float(np.linalg.norm(residual))
        residual_ratio_before = before / original_norm
        if residual_ratio_before <= residual_tolerance:
            break

        if enable_peak_stop and not _residual_has_peak(
            residual,
            min_prominence=min_residual_peak_prominence,
        ):
            history.append(
                {
                    "round": iteration + 1,
                    "status": "no_peak_like_residual",
                    "residual_ratio_before": residual_ratio_before,
                }
            )
            break

        C_0 = predict_segment_profiles(residual, model)
        segment = resolve_fn(residual, C_0)
        if (
            segment is None
            or segment.chromatograms.size == 0
            or segment.chromatograms.shape[1] == 0
        ):
            history.append(
                {
                    "round": iteration + 1,
                    "status": "no_valid_component",
                    "residual_ratio_before": residual_ratio_before,
                }
            )
            break

        current_components = sum(b.shape[1] for b in profile_blocks)
        if current_components + segment.chromatograms.shape[1] > max_components:
            break

        new_residual = np.maximum(residual - segment.reconstruction, 0.0).astype(
            np.float32
        )
        after = float(np.linalg.norm(new_residual))
        improvement = (before - after) / before if before > 0 else 0.0

        if improvement < min_improvement or segment.r2 <= 0:
            history.append(
                {
                    "round": iteration + 1,
                    "status": "rejected_round",
                    "n_components": int(segment.chromatograms.shape[1]),
                    "r2": float(segment.r2),
                    "wayname": segment.wayname,
                    "residual_ratio_before": residual_ratio_before,
                    "residual_ratio_after": after / original_norm,
                    "improvement": improvement,
                }
            )
            break

        profile_blocks.append(segment.chromatograms)
        spectrum_blocks.append(segment.spectra)
        residual = new_residual
        history.append(
            {
                "round": iteration + 1,
                "status": "accepted",
                "n_components": int(segment.chromatograms.shape[1]),
                "r2": float(segment.r2),
                "wayname": segment.wayname,
                "residual_ratio_before": residual_ratio_before,
                "residual_ratio_after": after / original_norm,
                "improvement": improvement,
            }
        )

        if after / original_norm <= residual_tolerance:
            break

    if profile_blocks:
        # Merge, filter, and jointly refine all accepted round-wise profiles.
        C_all, _, _ = _deduplicate_profiles(
            profile_blocks,
            spectrum_blocks,
            apex_tolerance=dedup_apex_tolerance,
            profile_cosine=dedup_profile_cosine,
            spectral_cosine=dedup_spectral_cosine,
        )
        C_all = _profile_shape_filter(
            C_all,
            max_support=max_support,
            max_fwhm_factor=max_fwhm_factor,
        )
        if final_cleanup:
            C_all = _final_cleanup_profiles(
                X, C_all, r2_margin=final_cleanup_r2_margin
            )
        C_final, S_final = final_refine_fn(X, C_all)
        C_final, S_final = _spectrum_sanity_filter(
            C_final,
            S_final,
            max_nonzero_ratio=max_nonzero_ratio,
            max_cv_flat=max_cv_flat,
            max_scale_ratio=max_spectrum_scale,
        )
        if enable_flat_tail_taper:
            C_final = _taper_flat_tail(
                C_final,
                tail_window=tail_window,
                tail_level=tail_level,
            )
        if remove_embedded_peaks:
            C_final, S_final = _remove_embedded_profiles(
                C_final, S_final, min_selectivity=min_selectivity
            )
        if C_final.shape[1] > 0:
            contributions = np.array(
                [
                    float(
                        np.linalg.norm(C_final[:, j])
                        * np.linalg.norm(S_final[j, :])
                    )
                    for j in range(C_final.shape[1])
                ]
            )
            keep = contributions / original_norm >= min_component_fraction
            C_final = C_final[:, keep]
            S_final = S_final[keep]
        reconstruction = np.asarray(C_final @ S_final, dtype=np.float32)
        residual = np.maximum(X - reconstruction, 0.0).astype(np.float32)
    else:
        C_final = np.zeros((X.shape[0], 0), dtype=np.float32)
        S_final = np.zeros((0, X.shape[1]), dtype=np.float32)
        reconstruction = np.zeros_like(X)
        residual = X.copy()

    return AdaptiveResult(
        chromatograms=C_final,
        spectra=S_final,
        reconstruction=reconstruction,
        residual=residual,
        signal=X,
        iterations=len(history),
        history=history,
    )


def DeepCPRAdaptiveV2(
    work_path,
    modelpath,
    filename,
    figure_savepath=None,
    dist=5,
    thres=5,
    generate_image=False,
    max_iterations=4,
    residual_tolerance=0.03,
    min_improvement=0.01,
    max_components=32,
    dedup_apex_tolerance=2,
    dedup_profile_cosine=0.95,
    dedup_spectral_cosine=0.90,
    enable_flat_tail_taper=False,
    tail_window=12,
    tail_level=0.05,
    final_cleanup=False,
    final_cleanup_r2_margin=0.005,
    remove_embedded_peaks=True,
    min_selectivity=0.4,
    final_refine_method="ittfa",
):
    """Improved adaptive extension of ``DeepCPR`` (see module docstring)."""

    mod = _load_deepcpr()

    mat, RT, Xtest, ind_st_DeepSeg, ind_en_DeepSeg = mod.data_process(
        work_path, filename, dist, thres
    )
    TICsum_origin = np.asarray(
        [np.sum(Xtest[i, :]) for i in range(Xtest.shape[0])]
    )
    mz_min = math.floor(mod.judge_min(min(mat["mz"])))
    mz_max = math.ceil(max(mat["mz"]))

    restored_model = load_model_auto(
        modelpath,
        custom_objects={"R_squared": mod.R_squared, "PW_loss": mod.PW_loss},
    )

    peak_excel_single = []
    peak_excel_seg = []
    ms_single = []
    gap = (np.max(RT) - np.min(RT)) / max(Xtest.shape[0] - 1, 1)

    for i in tqdm(range(len(ind_st_DeepSeg)), desc="adaptive v2 resolving processing"):
        start = int(ind_st_DeepSeg[i])
        end = int(ind_en_DeepSeg[i])
        length = end - start
        if length <= 0:
            continue
        if length > 128:
            raise ValueError(
                f"Local segment {i} contains {length} scans; the trained model accepts at most 128."
            )

        test = np.zeros((length, 800), dtype=np.float32)
        test[:, mz_min:mz_max] = Xtest[start:end, :]
        test_br = mod.airPLS_correct_matrix(test, lambda_=500, itermax=15)
        if not np.any(test_br):
            continue

        result = adaptive_resolve_segment_v2(
            test_br,
            restored_model,
            max_iterations=max_iterations,
            residual_tolerance=residual_tolerance,
            min_improvement=min_improvement,
            max_components=max_components,
            dedup_apex_tolerance=dedup_apex_tolerance,
            dedup_profile_cosine=dedup_profile_cosine,
            dedup_spectral_cosine=dedup_spectral_cosine,
            enable_flat_tail_taper=enable_flat_tail_taper,
            tail_window=tail_window,
            tail_level=tail_level,
            final_cleanup=final_cleanup,
            final_cleanup_r2_margin=final_cleanup_r2_margin,
            remove_embedded_peaks=remove_embedded_peaks,
            min_selectivity=min_selectivity,
            final_refine_method=final_refine_method,
        )
        COM = result.n_components
        if COM == 0:
            continue

        C_ed = result.chromatograms
        St = result.spectra
        chrom_resol = result.reconstruction
        R2 = result.r2
        tic_single = C_ed * np.sum(St, axis=1, keepdims=True).T
        tic = np.sum(chrom_resol, axis=1)
        peak_area = float(np.trapz(tic))

        rt_st = np.min(RT) + start * gap
        rt_ed = np.min(RT) + end * gap
        peak_excel_seg.append(
            {
                "num": i + 1,
                "RT_st": round(float(rt_st), 5),
                "RT_ed": round(float(rt_ed), 5),
                "R2": round(float(R2), 5),
                "PA": peak_area,
                "COM": COM,
                "iterations": result.iterations,
            }
        )

        left = TICsum_origin[max(0, start - 2) : start]
        right = TICsum_origin[end : min(Xtest.shape[0], end + 2)]
        noise_scale = (
            float(np.std(np.concatenate((left, right))))
            if (left.size + right.size)
            else 1.0
        )
        noise_scale = max(noise_scale, 1e-12)

        for r in range(COM):
            apex = int(np.argmax(C_ed[:, r]))
            rt = np.min(RT) + (start + apex) * gap
            component_tic = tic_single[:, r]
            snr = round(float(np.max(component_tic) / noise_scale), 2)
            ms_single.append({"ms": St[r, :]})
            peak_excel_single.append(
                {
                    "#number": len(peak_excel_single) + 1,
                    "rt": round(float(rt), 5),
                    "peak area": float(np.trapz(component_tic)),
                    "COM": COM,
                    "R2": float(R2),
                    "SNR": snr,
                    "adaptive_iteration_count": result.iterations,
                }
            )

        if generate_image and figure_savepath is not None:
            import os
            import matplotlib.pyplot as plt

            os.makedirs(figure_savepath, exist_ok=True)
            plt.figure(clear=True, figsize=(10, 7))
            plt.subplot(2, 2, 1)
            plt.plot(test_br)
            plt.title("original GC-MS data")
            plt.subplot(2, 2, 2)
            plt.plot(chrom_resol)
            plt.title(f"adaptive reconstruction (K={COM})")
            plt.subplot(2, 2, 3)
            plt.plot(C_ed)
            plt.title("resolved chromatographic profiles")
            plt.subplot(2, 2, 4)
            plt.plot(tic_single)
            plt.title("resolved TIC components")
            plt.tight_layout()
            plt.savefig(os.path.join(figure_savepath, f"num={i}.tif"))
            plt.close()

    return peak_excel_single, peak_excel_seg, ms_single


def data_resolution_v2(
    dataset_path,
    DeepCS_path,
    DeepCPR_path,
    save_path,
    generate_image,
    adaptive_kwargs=None,
):
    """Store the improved adaptive resolution results (same layout as ``data_resolution``)."""

    import gc
    import os

    import pandas as pd

    mod = _load_deepcpr()

    px_savepath1 = save_path + "/single"
    px_savepath2 = save_path + "/seg"
    ms_savepath = save_path + "/ms"
    for path in (px_savepath1, px_savepath2, ms_savepath):
        os.makedirs(path, exist_ok=True)

    files = os.listdir(dataset_path)
    for file in files:
        if tensorflow_available():
            _load_tf().keras.backend.clear_session()

        filename = os.path.join(dataset_path, file)
        file_pre = file.split(".")[0]

        if generate_image is True:
            figure_savepath = save_path + "/figure/" + file_pre
            os.makedirs(figure_savepath, exist_ok=True)
        else:
            figure_savepath = None

        print("file loading:", file)
        peak_excel_single, peak_excel_seg, ms_single = DeepCPRAdaptiveV2(
            DeepCS_path,
            DeepCPR_path,
            filename,
            figure_savepath,
            dist=5,
            thres=5,
            generate_image=generate_image,
            **(adaptive_kwargs or {}),
        )

        pe_single = pd.DataFrame(peak_excel_single)
        pe_single.to_csv(px_savepath1 + "/" + file_pre + ".csv", index=False)
        pe_seg = pd.DataFrame(peak_excel_seg)
        pe_seg.to_csv(px_savepath2 + "/" + file_pre + ".csv", index=False)

        ms_dir = ms_savepath + "/" + file_pre
        os.makedirs(ms_dir, exist_ok=True)
        for i in range(len(ms_single)):
            ms_single_com = ms_single[i]["ms"]
            RT = peak_excel_single[i]["rt"]
            mz_values = np.arange(1, ms_single_com.shape[0] + 1)
            mod.save_as_msp(
                ms_dir + "/" + str(i) + ".msp",
                RT,
                mz_values,
                ms_single_com,
            )

        del peak_excel_single, peak_excel_seg, ms_single, pe_single, pe_seg
        gc.collect()
        print("\n")


__all__ = [
    "SegmentResolution",
    "AdaptiveResult",
    "predict_segment_profiles",
    "resolve_one_segment",
    "adaptive_resolve_segment_v2",
    "DeepCPRAdaptiveV2",
    "data_resolution_v2",
]

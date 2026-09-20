import os
import warnings

# Keep TensorFlow's routine device-initialization messages out of application
# logs while preserving warnings and errors. Users can override this setting.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from scipy.spatial.distance import cosine
from scipy.optimize import OptimizeWarning, curve_fit
import numpy as np
from numpy import float32
try:
    import tensorflow as tf
except Exception:  # TensorFlow is optional when ONNX models are used.
    tf = None
import matplotlib.pyplot as plt
import math
from numpy import hstack
from .NetCDF import netcdf_reader
from .model_runtime import load_model_auto
from .DeepCS import Chromseg
from .component_selection import (
    ComponentSelectionConfig,
    _resolved_profile_shape_test,
    select_component_model,
)
from scipy import integrate
from scipy.signal import find_peaks
import pandas as pd
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from itertools import groupby
import copy
from sklearn.metrics import explained_variance_score
import tensorly as tl
import itertools
import pywt
import gc
from tqdm import tqdm

try:
    from joblib import Parallel, delayed, parallel_config
except Exception:  # Fall back to the strict serial executor without joblib.
    Parallel = None
    delayed = None
    parallel_config = None

# Models are immutable during one resolution batch.  Reusing the loaded
# object avoids TensorFlow session teardown and model deserialization for
# every concentration file processed by app.py.
_MODEL_CACHE = {}

_SUPPORTED_IMAGE_FORMATS = {"png", "svg"}


def _fit_gaussian(x, y, p0, maxfev):
    """Fit a Gaussian while hiding only the non-actionable covariance warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        return curve_fit(gaussian, x, y, p0=p0, maxfev=maxfev)


def _normalize_image_formats(image_formats):
    """Return unique supported figure formats in the requested order."""
    if isinstance(image_formats, str):
        image_formats = [image_formats]
    normalized_formats = []
    for image_format in image_formats or []:
        normalized = str(image_format).strip().lower().lstrip(".")
        if normalized not in _SUPPORTED_IMAGE_FORMATS:
            supported = ", ".join(sorted(_SUPPORTED_IMAGE_FORMATS))
            raise ValueError(
                f"Unsupported image format {image_format!r}. "
                f"Choose from: {supported}."
            )
        if normalized not in normalized_formats:
            normalized_formats.append(normalized)
    if not normalized_formats:
        raise ValueError("Select at least one resolution-figure format.")
    return tuple(normalized_formats)


def _load_cached_model(modelpath):
    path = os.path.abspath(os.fspath(modelpath))
    try:
        signature = (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        signature = (path, None, None)
    model = _MODEL_CACHE.get(signature)
    if model is None:
        model = load_model_auto(
            path,
            custom_objects={"R_squared": R_squared, "PW_loss": PW_loss},
        )
        _MODEL_CACHE.clear()
        _MODEL_CACHE[signature] = model
    return model

try:
    import numba as _numba
    from numba import njit as _njit
    from numba import prange as _prange
except Exception:  # Fall back to the pure-Python solvers when Numba is absent.
    _numba = None
    _njit = None
    _prange = range


def data_restore(num, dis_st, predict, ind_st_DeepSeg, ind_en_DeepSeg, mz_min, mz_max, chrom_divid):
    """
    Extraction of GC-MS data segment and their predictive chromatography
    """
    chrom_origin = chrom_divid[num]
    C_0 = np.zeros((ind_en_DeepSeg[num]-ind_st_DeepSeg[num], 5), dtype=float32)
    C_0[:, :] = predict[num][int(dis_st):int(dis_st+ind_en_DeepSeg[num]-ind_st_DeepSeg[num]), :]
    C_0[C_0 < 0] = 0
    return chrom_origin, C_0


def peak_find(CC):
    """
    Analyze the locations of peaks in chromatograms.
    """
    C = np.copy(CC)

    threshold = 3e-3
    fun = lambda x: x[1]-x[0]
    peak_st = []
    peak_ed = []
    for r in range(C.shape[1]):
        st = []
        ed = []
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            st.append(min(l1))
            ed.append(max(l1))
            st_copy = st.copy()
            ed_copy = ed.copy()

        if len(st) > 1:
            long_idx = np.argmax([np.max(ed[ii]-st[ii]) for ii in range(len(st))])
            st_idx = [gg for gg in range(len(st))]
            del st_idx[long_idx]
            for jj in range(len(st_idx)):
                st.remove(st_copy[st_idx[jj]])
                ed.remove(ed_copy[st_idx[jj]])

        if len(st) == 1:
            peak_st.append(st[0])
            peak_ed.append(ed[0])

    return peak_st, peak_ed


def com_find(CC):
    """
    obtaining the number of components in chromatograms.
    """
    C = np.copy(CC)

    threshold = 3e-3
    fun = lambda x: x[1]-x[0]
    peak_st = []
    peak_ed = []
    for r in range(C.shape[1]):
        st = []
        ed = []
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            st.append(min(l1))
            ed.append(max(l1))
            st_copy = st.copy()
            ed_copy = ed.copy()

        if len(st) > 1:
            long_idx = np.argmax([np.max(ed[ii]-st[ii]) for ii in range(len(st))])
            st_idx = [gg for gg in range(len(st))]
            del st_idx[long_idx]
            for jj in range(len(st_idx)):
                st.remove(st_copy[st_idx[jj]])
                ed.remove(ed_copy[st_idx[jj]])

        if len(st) == 1:
            peak_st.append(st[0])
            peak_ed.append(ed[0])

    COM = len(peak_st)
    return COM


def peak_preprocess(CC, enable_gaussian_tail_correction=True):
    """
    Pre-processing of predicted chromatographic profiles.
    """
    # low intensity noise filtering
    C = np.copy(CC)

    threshold = 3e-3
    for r in range(C.shape[1]):
        if max(C[:, r]) < 0.04:
            C[:, r] = 0

    # chromatographic filling
    for r in range(C.shape[1]):
        peak_st = []
        peak_ed = []
        fun = lambda x: x[1]-x[0]
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            peak_st.append(min(l1))
            peak_ed.append(max(l1))

        if len(peak_st)>1:
            for m in range(1, len(peak_st)):
                for n in range(len(peak_ed)-1):
                    if peak_st[m] == peak_ed[n]:
                        C[peak_st[m], r] = (C[peak_st[m]-1, r] + C[peak_st[m]+1, r])/2

    # filtering peaks with width less 5
    for r in range(C.shape[1]):
        peak_st = []
        peak_ed = []
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            peak_st.append(min(l1))
            peak_ed.append(max(l1))

        if len(peak_st) > 0:
            for ps in range(len(peak_st)):
                if peak_ed[ps]-peak_st[ps]+2 < 5:
                    C[peak_st[ps]:peak_ed[ps]+1, r] = 0
        if max(C[:, r]) < 0.04:
            C[:, r] = 0

    # preserving peak with the highest intensity in each m/z channel
    fun = lambda x: x[1]-x[0]
    for r in range(C.shape[1]):
        peak_st = []
        peak_ed = []
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            peak_st.append(min(l1))
            peak_ed.append(max(l1))

        if len(peak_st) > 1:
            intensity = []
            for ps in range(len(peak_st)):
                intensity.append(max(C[peak_st[ps]:peak_ed[ps], r]))
            p = intensity.index(max(intensity))
            if peak_st[p] != 0:
                C[0:peak_st[p], r] = 0
            if peak_ed[p] != (C.shape[0]-1):
                C[peak_ed[p]:, r] = 0

    if not enable_gaussian_tail_correction:
        return C

    # refine the shape of gaussian peak
    for r in range(C.shape[1]):
        peaks, _ = find_peaks(C[:, r], height=0.003)
        if len(peaks) > 1:
            inten_max = peaks[np.argmax([C[i, r] for i in peaks])]
            peak_st, peak_ed = peak_find(C[:, r:r+1])
            gaussleftfit = 'no'
            gaussrightfit = 'no'
            for z in range(inten_max, peak_st[0]+1, -1):
                if C[z, r] > 0.003 and C[z-1, r] > C[z, r]:
                    gaussleftfit = 'yes'
                    lp = z
                    break
            for y in range(inten_max, peak_ed[0]-1, 1):
                if C[y, r] > 0.003 and C[y+1, r] > C[y, r]:
                    gaussrightfit = 'yes'
                    rp = y
                    break

            if gaussleftfit == 'yes' and gaussrightfit == 'yes':
                peak_x = np.arange(lp+1, rp)
                peak_x = [float(i) for i in peak_x]
                peak_y = C[lp+1:rp, r]
                peak_y = [float(i) for i in peak_y]
                if len(peak_x) > 2 and np.abs(inten_max-lp) > 2 and np.abs(inten_max-rp) > 2:
                    if C[lp, r] < 0.5*np.max(C[:, r]):
                        s = 2*np.abs(getnearpos(C[lp:inten_max, r], 0.5*np.max(C[:, r])) - inten_max)
                    elif C[rp, r] < 0.5*np.max(C[:, r]):
                        s = 2*np.abs(getnearpos(C[inten_max:rp, r], 0.5*np.max(C[:, r])) - inten_max)
                    else:
                        if np.argmin([C[lp, r], C[rp, r]]) == 0:
                            s = np.max(C[:, r])*np.abs(inten_max-lp)/(np.max(C[:, r])-C[lp, r])
                        if np.argmin([C[lp, r], C[rp, r]]) == 1:
                            s = np.max(C[:, r])*np.abs(rp-inten_max)/(np.max(C[:, r])-C[rp, r])

                    popt, pcov = _fit_gaussian(
                        peak_x, peak_y,
                        p0=[np.max(C[:, r]), inten_max, s], maxfev=500000)
                    for j in range(rp, C.shape[0],1):
                        C[j, r] = gaussian(j, *popt)
                        if gaussian(j, *popt) < 0.003:
                            C[j:C.shape[0], r] = 0
                            break
                    for j in range(lp, -1, -1):
                        C[j, r] = gaussian(j, *popt)
                        if gaussian(j, *popt) < 0.003:
                            C[0:j, r] = 0
                            break

            if gaussleftfit == 'no' and gaussrightfit == 'yes':
                peak_x = np.arange(peak_st[0], rp)
                peak_x = [float(i) for i in peak_x]
                peak_y = C[peak_st[0]:rp, r]
                peak_y = [float(i) for i in peak_y]
                if len(peak_x) > 2 and np.abs(inten_max-rp) > 2:
                    if inten_max == peak_st[0] and peak_st[0] > 0:
                        s = 2*np.abs(getnearpos(C[peak_st[0]-1:inten_max, r], 0.5*np.max(C[:, r])) - inten_max)
                    elif inten_max == peak_st[0] and peak_st[0] == 0:
                        s = 2*np.abs(getnearpos(C[peak_st[0]:inten_max+1, r], 0.5*np.max(C[:, r])) - inten_max)
                    else:
                        s = 2*np.abs(getnearpos(C[peak_st[0]:inten_max, r], 0.5*np.max(C[:, r])) - inten_max)
                    popt, pcov = _fit_gaussian(
                        peak_x, peak_y,
                        p0=[np.max(C[:, r]), inten_max, s], maxfev=500000)
                    for j in range(rp, C.shape[0],1):
                        C[j, r] = gaussian(j, *popt)
                        if gaussian(j, *popt) < 0.003:
                            C[j:C.shape[0], r] = 0
                            break

            if gaussleftfit == 'yes' and gaussrightfit == 'no':
                peak_x = np.arange(lp+1, peak_ed[0])
                peak_x = [float(i) for i in peak_x]
                peak_y = C[lp+1:peak_ed[0], r]
                peak_y = [float(i) for i in peak_y]
                if len(peak_x) > 2 and np.abs(inten_max-lp) > 2:
                    if inten_max == peak_st[0] and peak_ed[0] < C.shape[0]:
                        s = 2*np.abs(getnearpos(C[inten_max:peak_ed[0]+1, r], 0.5*np.max(C[:, r])) - inten_max)
                    elif inten_max == peak_st[0] and peak_ed[0] == C.shape[0]:
                        s = 2*np.abs(getnearpos(C[inten_max-1:peak_ed[0], r], 0.5*np.max(C[:, r])) - inten_max)
                    else:
                        s = 2*np.abs(getnearpos(C[inten_max:peak_ed[0], r], 0.5*np.max(C[:, r])) - inten_max)
                    popt, pcov = _fit_gaussian(
                        peak_x, peak_y,
                        p0=[np.max(C[:, r]), inten_max, s], maxfev=500000)
                    for j in range(lp, -1, -1):
                        C[j, r] = gaussian(j, *popt)
                        if gaussian(j, *popt) < 0.003:
                            C[0:j, r] = 0
                            break
    return C


def cal_deriv(x, y):
    diff_x = []
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))
    deriv.insert(0, slopes[0])
    deriv.append(slopes[-1])
    return deriv


def tail_fix(CC, enable_gaussian_tail_correction=True):
    """
    Refining the shapes of predicted chromatographic profiles to Gaussian-like.
    """
    C = np.copy(CC)
    if not enable_gaussian_tail_correction:
        return C
    COM = com_find(C)
    peak_st, peak_ed = peak_find(C)
    for r in range(COM):
        xrange = [i for i in range(peak_st[r], peak_ed[r]+1)]
        yrange = [C[:, r][i] for i in range(peak_st[r], peak_ed[r]+1)]
        deriv = cal_deriv(xrange, yrange)
        x_max = np.argmax(deriv)
        x_min = np.argmin(deriv)

        peaks, _ = find_peaks(C[:, r], height=0.003)
        inten_max = peaks[np.argmax([C[i, r] for i in peaks])]
        leftfit = 'no'
        rightfit = 'no'
        for z in range(x_max, 0, -1):
            if deriv[z-1] > deriv[z]:
                leftfit = 'yes'
                lp = z + min(xrange)
                break
        for y in range(x_min, len(deriv)-1, 1):
            if deriv[y+1] < deriv[y]:
                rightfit = 'yes'
                rp = y + min(xrange)
                break

        if leftfit == 'yes' and rightfit == 'yes':
            peak_x = np.arange(lp + 1, rp)
            peak_x = [float(i) for i in peak_x]
            peak_y = C[lp + 1:rp, r]
            peak_y = [float(i) for i in peak_y]
            if len(peak_x) > 2 and np.abs(rp - lp) > 2:
                if C[lp, r] < 0.5 * np.max(C[:, r]):
                    s = 2 * np.abs(getnearpos(C[lp:inten_max, r], 0.5 * np.max(C[:, r])) - inten_max)
                elif C[rp, r] < 0.5 * np.max(C[:, r]):
                    s = 2 * np.abs(getnearpos(C[inten_max:rp, r], 0.5 * np.max(C[:, r])) - inten_max)
                else:
                    if np.argmin([C[lp, r], C[rp, r]]) == 0:
                        s = np.max(C[:, r]) * np.abs(inten_max - lp) / (np.max(C[:, r]) - C[lp, r])
                    if np.argmin([C[lp, r], C[rp, r]]) == 1:
                        s = np.max(C[:, r]) * np.abs(rp - inten_max) / (np.max(C[:, r]) - C[rp, r])

                popt, pcov = _fit_gaussian(
                    peak_x, peak_y,
                    p0=[np.max(C[:, r]), inten_max, s], maxfev=500000)
                for j in range(rp, C.shape[0], 1):
                    C[j, r] = gaussian(j, *popt)
                    if gaussian(j, *popt) < 0.05:
                        C[j:C.shape[0], r] = 0
                        break
                for j in range(lp, -1, -1):
                    C[j, r] = gaussian(j, *popt)
                    if gaussian(j, *popt) < 0.05:
                        C[0:j, r] = 0
                        break

        if leftfit == 'no' and rightfit == 'yes':
            peak_x = np.arange(peak_st[r], rp)
            peak_x = [float(i) for i in peak_x]
            peak_y = C[peak_st[r]:rp, r]
            peak_y = [float(i) for i in peak_y]
            if len(peak_x) > 2:
                s = 2 * np.abs(getnearpos(C[peak_st[r]:rp, r], 0.5 * np.max(C[:, r])) - inten_max)
                popt, pcov = _fit_gaussian(
                    peak_x, peak_y,
                    p0=[np.max(C[:, r]), inten_max, s], maxfev=500000)
                for j in range(rp, C.shape[0], 1):
                    C[j, r] = gaussian(j, *popt)
                    if gaussian(j, *popt) < 0.05:
                        C[j:C.shape[0], r] = 0
                        break

        if leftfit == 'yes' and rightfit == 'no':
            peak_x = np.arange(lp, peak_ed[r])
            peak_x = [float(i) for i in peak_x]
            peak_y = C[lp:peak_ed[r], r]
            peak_y = [float(i) for i in peak_y]
            if len(peak_x) > 2:
                s = 2*np.abs(getnearpos(C[lp:peak_ed[r], r], 0.5*np.max(C[:,r])) - inten_max)
                popt, pcov = _fit_gaussian(
                    peak_x, peak_y,
                    p0=[np.max(C[:, r]), inten_max, s], maxfev=500000)
                for j in range(lp, -1, -1):
                    C[j, r] = gaussian(j, *popt)
                    if gaussian(j, *popt) < 0.05:
                        C[0:j, r] = 0
                        break

    return C


def peak_halfremove(CC):
    """
    1. remove half peak and add mark.
    """
    C = np.copy(CC)
    threshold = 3e-3
    halfdo = 'None'
    COM = com_find(C)
    if COM > 1:
        fun = lambda x: x[1]-x[0]
        remove_r = []
        for r in range(COM):
            st = []
            ed = []
            for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
                l1 = [j for i, j in g]
                st.append(min(l1))
                ed.append(max(l1))
                st_copy = st.copy()
                ed_copy = ed.copy()

            if len(st) > 1:
                long_idx = np.argmax([ed[ii]-st[ii] for ii in range(len(st))])
                st_idx = [gg for gg in range(len(st))]
                del st_idx[long_idx]
                for jj in range(len(st_idx)):
                    st.remove(st_copy[st_idx[jj]])
                    ed.remove(ed_copy[st_idx[jj]])

            if min(st) == 0:
                if C[st, r] > (max(C[st[0]:ed[0], r])/3) and np.max(C[st[0]:ed[0], r]) < (np.max(C)*0.25):
                    remove_r.append(r)
                    halfdo = 'Done'
            if max(ed) == (C.shape[0]-1):
                if C[ed, r] > (max(C[st[0]:ed[0], r])/3) and np.max(C[st[0]:ed[0], r]) < (np.max(C)*0.25):
                    remove_r.append(r)
                    halfdo = 'Done'

        if len(remove_r) == COM:
            intenmax_idx = np.argmax([max(C[:, j]) for j in range(COM)])
            for i in range(COM):
                if i != intenmax_idx:
                    C[:, i] = 0
        if len(remove_r) < COM:
            for i in range(len(remove_r)):
                C[:, remove_r[i]] = 0

    return C, halfdo


def dishalfremove(CC, C_ed, threshold=3e-3, protected_indices=None):
    """
    2. remove half peak and add mark.
    """
    C = np.copy(CC)

    dishalfdo = 'None'
    dishalfnum = []
    protected_indices = set(protected_indices or [])
    COM = com_find(C_ed)
    if COM > 1:
        fun = lambda x: x[1]-x[0]
        for r in range(COM):
            if r in protected_indices:
                continue
            st = []
            ed = []
            for k, g in groupby(enumerate(np.where(C_ed[:, r] > threshold)[0]), fun):
                l1 = [j for i, j in g]
                st.append(min(l1))
                ed.append(max(l1))
                st_copy = st.copy()
                ed_copy = ed.copy()

            if len(st) > 1:
                long_idx = np.argmax([np.max(ed[ii]-st[ii]) for ii in range(len(st))])
                st_idx = [gg for gg in range(len(st))]
                del st_idx[long_idx]
                for jj in range(len(st_idx)):
                    st.remove(st_copy[st_idx[jj]])
                    ed.remove(ed_copy[st_idx[jj]])

            if len(st) > 0:
                if min(st) == 0 and C_ed[st, r] > (max(C_ed[st[0]:ed[0], r])/2):
                    C[:, r] = 0
                    dishalfdo = 'Done'
                    dishalfnum.append(r)
                if max(ed) == (C.shape[0]-1) and C_ed[ed, r] > (max(C_ed[st[0]:ed[0], r])/2):
                    C[:, r] = 0
                    dishalfdo = 'Done'
                    dishalfnum.append(r)

    return C, dishalfdo, dishalfnum


def _hard_boundary_half_indices(pre_ittfa_C, post_ittfa_C,
                                candidate_indices, active_threshold=3e-3):
    """Return boundary-half failures that invalidate the complete model.

    A post ITTFA boundary half profile is a hard failure when the same
    boundary was not already truncated in the input. If every active profile
    in a multi-component candidate is marked by ``dishalfremove``, the whole
    decomposition is also invalid: retaining two opposing half profiles just
    because their combined R2 is larger does not yield valid chromatographic
    components. A sole pre-truncated profile remains eligible.
    """
    pre = np.asarray(pre_ittfa_C)
    post = np.asarray(post_ittfa_C)
    failures = []
    if pre.ndim != 2 or post.ndim != 2:
        return failures
    candidates = sorted(set(int(value) for value in (candidate_indices or [])))
    active = [
        index for index in range(pre.shape[1])
        if float(np.max(np.maximum(pre[:, index], 0))) > active_threshold
    ]
    if len(active) > 1 and set(active).issubset(set(candidates)):
        return active
    for index in candidates:
        if index < 0 or index >= pre.shape[1] or index >= post.shape[1]:
            continue
        pre_profile = np.maximum(np.asarray(pre[:, index], dtype=float), 0)
        post_profile = np.maximum(np.asarray(post[:, index], dtype=float), 0)
        pre_maximum = float(np.max(pre_profile))
        post_maximum = float(np.max(post_profile))
        if pre_maximum <= 0 or post_maximum <= 0:
            continue
        pre_left_half = pre_profile[0] > 0.5 * pre_maximum
        pre_right_half = pre_profile[-1] > 0.5 * pre_maximum
        post_left_half = post_profile[0] > 0.5 * post_maximum
        post_right_half = post_profile[-1] > 0.5 * post_maximum
        if ((post_left_half and not pre_left_half)
                or (post_right_half and not pre_right_half)):
            failures.append(index)
    return failures


def testt(CC):
    """
    refine chromatogram by detecting and removing coincide peaks
    """
    C = np.copy(CC)
    testdo = 'None'
    lapdo = 'None'
    lapnum = []
    removenum = []
    COM = com_find(C)
    if COM > 1:
        peak_st, peak_ed = peak_find(C)
        for m in range(COM-1):
            for n in range(m+1, COM):
                C_concat = np.zeros((C.shape[0], 2))
                C_concat[:, 0] = C[:, m]
                C_concat[:, 1] = C[:, n]
                C_overlap = C_concat.min(axis=1)
                intenmin_idx = [m, n][np.argmin([max(C[:, m]), max(C[:, n])])]
                intenmax_idx = [m, n][np.argmax([max(C[:, m]), max(C[:, n])])]
                area_min = np.trapz(C_overlap[min(peak_st):max(peak_ed)])
                area_max = np.trapz(C[min(peak_st):max(peak_ed), intenmin_idx])

                if area_max < 0.05:
                    ol_degree = 0
                else:
                    ol_degree = area_min / area_max

                if np.abs(np.argmax(C[:, m]) - np.argmax(C[:, n])) < 4 and ol_degree > 0.6:
                    C[:, intenmin_idx] = 0
                    testdo = 'Done'
                    removenum.append(intenmin_idx)

                if np.abs(np.argmax(C[:, m]) - np.argmax(C[:, n])) > 3 and ol_degree > 0.95:
                    if np.max(C[intenmin_idx]) == 0:
                        C[:, intenmin_idx] = 0
                        testdo = 'Done'
                        removenum.append(intenmin_idx)
                    elif np.max(C[intenmax_idx])/np.max(C[intenmin_idx]) > 8:
                        C[:, intenmin_idx] = 0
                        testdo = 'Done'
                        removenum.append(intenmin_idx)

                if np.abs(np.argmax(C[:, m]) - np.argmax(C[:, n])) > 3 and ol_degree > 0.75:
                    lapdo = 'Done'
                    lapnum.append(intenmin_idx)

    lapnum_2 = copy.deepcopy(list(set(lapnum)))
    lapnum_1 = copy.deepcopy(list(set(lapnum)))
    for b in range(len(lapnum_2)):
        if lapnum_2[b] in removenum:
            lapnum_1.remove(lapnum_2[b])

    if len(lapnum_1) > 0:
        lapdo = 'Done'
    else:
        lapdo = 'None'

    return C, testdo, lapdo, lapnum_1


def peak_filter(C, C_ed, St, COM):
    """
    Remove components with spectra or chromatogram is none.
    """
    for i in range(COM):
        if np.all(St[i, :] == 0) or np.all(C_ed[:, i] == 0):
            C[:, i] = 0
    return C


def peak_trans(C, C_trans):
    """
    rearrange peak in chromatographic list.
    """
    signone = []
    for m in range(C.shape[1]):
        if np.all(C[:, m] == 0) == False:
            signone.append(m)
    if len(signone) == 0:
        return C_trans
    lenth = len(signone)-1
    if signone[lenth] != lenth:
        for r in range(lenth+1):
            C_trans[:, r] = C[:, signone[r]]
    else:
        C_trans = C

    return C_trans


def _compact_profiles_with_indices(C):
    """Return nonzero profile columns and their original column indices."""
    C = np.asarray(C)
    active = [i for i in range(C.shape[1]) if not np.all(C[:, i] == 0)]
    if not active:
        return np.zeros((C.shape[0], 0), dtype=C.dtype), []
    # ``np.asarray(copy=...)`` is unavailable in the NumPy version used by
    # the TensorFlow 2.10 environment.  ``np.array`` preserves explicit-copy
    # semantics across the supported versions.
    return np.array(C[:, active], copy=True), active


def _reconstruction_r2(observed, reconstructed):
    """Score a reconstruction with the variance definition used by DeepCPR."""
    if reconstructed is None:
        return float('-inf')
    denominator = np.var(observed, ddof=1)
    if denominator == 0:
        return float('-inf')
    return float(1 - np.var(reconstructed - observed, ddof=1) / denominator)


def _nnls_reconstruct(chrom_seg, C):
    """Fit nonnegative spectra for fixed chromatographic profiles and score R2."""
    C_compact, active = _compact_profiles_with_indices(C)
    if C_compact.shape[1] == 0:
        return None, None, float('-inf'), active

    CtC = np.dot(C_compact.T, C_compact)
    St = fnnls_batch(
        CtC, np.dot(C_compact.T, chrom_seg), tole='None'
    ).astype(np.float32, copy=False)
    reconstruction = np.dot(C_compact, St)
    r2 = _reconstruction_r2(chrom_seg, reconstruction)
    return reconstruction, St, float(r2), active


def _profile_cosine(a, b):
    """Cosine similarity that returns zero for a zero or invalid profile."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _align_component_anchors(reference_profiles, anchor_profiles):
    """Align non-Gaussian anchors to the current profile column order."""
    reference, _ = _compact_profiles_with_indices(reference_profiles)
    anchors, _ = _compact_profiles_with_indices(anchor_profiles)
    if reference.shape[1] == 0:
        return reference
    if anchors.shape[1] < reference.shape[1]:
        return reference.copy()

    row_scale = max(reference.shape[0] - 1, 1)
    costs = np.zeros((reference.shape[1], anchors.shape[1]), dtype=float)
    for ref_index in range(reference.shape[1]):
        for anchor_index in range(anchors.shape[1]):
            apex_distance = abs(
                int(np.argmax(reference[:, ref_index]))
                - int(np.argmax(anchors[:, anchor_index]))
            ) / row_scale
            shape_distance = 1.0 - _profile_cosine(
                reference[:, ref_index], anchors[:, anchor_index]
            )
            costs[ref_index, anchor_index] = apex_distance + shape_distance

    best_assignment = None
    best_cost = np.inf
    for assignment in itertools.permutations(
            range(anchors.shape[1]), reference.shape[1]):
        cost = sum(
            costs[ref_index, anchor_index]
            for ref_index, anchor_index in enumerate(assignment)
        )
        if cost < best_cost:
            best_assignment = assignment
            best_cost = cost
    return anchors[:, list(best_assignment)].copy()


def _fit_spectra_on_scan_rows(chrom_seg, C, rows):
    """Fit nonnegative spectra using only the selected chromatographic scans."""
    C_compact, active = _compact_profiles_with_indices(C)
    rows = np.asarray(rows, dtype=bool)
    if C_compact.shape[1] == 0 or rows.size != chrom_seg.shape[0] or np.sum(rows) < 3:
        return None, None, active

    C_rows = C_compact[rows]
    X_rows = np.asarray(chrom_seg)[rows]
    CtC = np.dot(C_rows.T, C_rows)
    St = fnnls_batch(
        CtC, np.dot(C_rows.T, X_rows), tole='None'
    ).astype(np.float32, copy=False)
    return C_compact, St, active


def _cross_validated_reconstruction_r2(chrom_seg, C):
    """Estimate reconstruction R2 on scans not used for spectral fitting.

    The chromatographic profiles are fixed and spectra are fitted on one
    subset of scans, then evaluated on the complementary subset.  This is an
    evidence score for overlap arbitration, not the final reported R2.
    """
    X = np.asarray(chrom_seg)
    if X.ndim != 2 or X.shape[0] < 8:
        return None

    scores = []
    scan_index = np.arange(X.shape[0])
    for test_rows in (scan_index % 2 == 0, scan_index % 2 == 1):
        train_rows = ~test_rows
        C_compact, St, _ = _fit_spectra_on_scan_rows(X, C, train_rows)
        if C_compact is None or St is None:
            return None
        scores.append(_reconstruction_r2(X[test_rows], np.dot(C_compact[test_rows], St)))
    if not scores or not np.all(np.isfinite(scores)):
        return None
    return float(np.mean(scores))


def _apex_spectrum_evidence(chrom_seg, raw_chrom_seg, pre_ittfa_C, event,
                            similarity_threshold=0.99):
    """Compare spectra at the two pre ITTFA profile apex scans.

    The baseline corrected spectrum is the primary measure because it is the
    same signal used by the resolver.  The raw spectrum is retained as an
    independent confirmation when available.  These are scan slices, not the
    spectra fitted by NNLS.
    """
    result = {
        'apex_i_local': None,
        'apex_j_local': None,
        'apex_baseline_corrected_cosine': None,
        'apex_raw_cosine': None,
        'apex_spectrum_high_similarity': False,
    }
    C = np.asarray(pre_ittfa_C)
    X = np.asarray(chrom_seg)
    if C.ndim != 2 or X.ndim != 2 or X.shape[0] != C.shape[0]:
        return result
    m = int(event['component_i'])
    n = int(event['component_j'])
    if m >= C.shape[1] or n >= C.shape[1]:
        return result
    apex_i = int(np.argmax(C[:, m]))
    apex_j = int(np.argmax(C[:, n]))
    result['apex_i_local'] = apex_i
    result['apex_j_local'] = apex_j
    result['apex_baseline_corrected_cosine'] = _profile_cosine(X[apex_i], X[apex_j])

    raw_cosine = None
    if raw_chrom_seg is not None:
        raw = np.asarray(raw_chrom_seg)
        if raw.ndim == 2 and raw.shape == X.shape:
            raw_cosine = _profile_cosine(raw[apex_i], raw[apex_j])
    result['apex_raw_cosine'] = raw_cosine

    corrected_high = (
        result['apex_baseline_corrected_cosine'] is not None
        and result['apex_baseline_corrected_cosine'] >= similarity_threshold
    )
    raw_high = raw_cosine is None or raw_cosine >= similarity_threshold
    # Signal quality is retained as descriptive evidence only.  It is not used
    # to decide whether two profiles represent one or multiple components.
    result['apex_spectrum_high_similarity'] = bool(corrected_high and raw_high)
    return result


def _overlap_evidence(chrom_seg, pre_ittfa_C, event,
                      raw_chrom_seg=None,
                      cv_margin=0.002,
                      weak_height_ratio=0.25,
                      high_spectral_similarity=0.99,
                      apex_similarity_threshold=0.99):
    """Classify an ITTFA overlap candidate without treating R2 as ground truth.

    ``ittfa_collapse`` is reserved for cases with clearly distinct pre ITTFA
    profiles, very similar fitted spectra, and a reproducible two profile
    reconstruction on held out scans.  ``model_overprediction`` is reserved
    for candidates whose weaker profile does not improve held out
    reconstruction and is weak relative to its partner.  Every candidate is
    ultimately assigned a deterministic retain or delete action by
    ``resolve_overlap_candidates``; there is no silent unresolved retain path.
    """
    C, _ = _compact_profiles_with_indices(pre_ittfa_C)
    m = int(event['component_i'])
    n = int(event['component_j'])
    result = {
        'evidence_class': 'r2_fallback',
        'pre_profile_apex_separation': None,
        'pre_profile_cosine': None,
        'pre_profile_area_ratio': None,
        'pre_spectral_cosine': None,
        'cv_r2_keep': None,
        'cv_r2_delete': None,
        'cv_r2_keep_minus_delete': None,
        'evidence_reason': 'fallback_branch_score_required',
    }
    result.update(_apex_spectrum_evidence(
        chrom_seg, raw_chrom_seg, pre_ittfa_C, event,
        similarity_threshold=apex_similarity_threshold,
    ))
    if m >= C.shape[1] or n >= C.shape[1]:
        return result

    pre_apex_separation = abs(int(np.argmax(C[:, m])) - int(np.argmax(C[:, n])))
    pre_profile_cosine = _profile_cosine(C[:, m], C[:, n])
    area_m = float(np.trapz(np.maximum(C[:, m], 0)))
    area_n = float(np.trapz(np.maximum(C[:, n], 0)))
    pre_area_ratio = min(area_m, area_n) / max(area_m, area_n, 1e-12)
    result.update({
        'pre_profile_apex_separation': int(pre_apex_separation),
        'pre_profile_cosine': float(pre_profile_cosine),
        'pre_profile_area_ratio': float(pre_area_ratio),
    })

    keep_cv = _cross_validated_reconstruction_r2(chrom_seg, C)
    delete = C.copy()
    weak = int(event['weak_index'])
    if weak < delete.shape[1]:
        delete[:, weak] = 0
    delete_cv = _cross_validated_reconstruction_r2(chrom_seg, delete)
    cv_gap = None
    if keep_cv is not None and delete_cv is not None:
        cv_gap = float(keep_cv - delete_cv)
        result.update({
            'cv_r2_keep': float(keep_cv),
            'cv_r2_delete': float(delete_cv),
            'cv_r2_keep_minus_delete': cv_gap,
        })

    _, St, _, _ = _nnls_reconstruct(chrom_seg, C)
    pre_spectral_cosine = None
    if St is not None and m < St.shape[0] and n < St.shape[0]:
        pre_spectral_cosine = _profile_cosine(St[m], St[n])
        result['pre_spectral_cosine'] = float(pre_spectral_cosine)

    pre_profiles_distinct = pre_apex_separation >= 3 and pre_profile_cosine <= 0.8
    event_weak_ratio = float(event.get('weak_height_ratio', 0.0))

    if (
        pre_profiles_distinct
        and pre_spectral_cosine is not None
        and pre_spectral_cosine >= high_spectral_similarity
        and cv_gap is not None
        and cv_gap > cv_margin
    ):
        result['evidence_class'] = 'ittfa_collapse'
        result['evidence_reason'] = 'distinct_pre_profiles_and_reproducible_high_similarity_spectra'
    elif (
        cv_gap is not None
        and cv_gap < -cv_margin
        and event_weak_ratio <= weak_height_ratio
    ):
        result['evidence_class'] = 'model_overprediction'
        result['evidence_reason'] = 'weak_profile_does_not_improve_held_out_reconstruction'
    elif (
        cv_gap is not None
        and abs(cv_gap) <= cv_margin
        and event_weak_ratio < weak_height_ratio
    ):
        result['evidence_class'] = 'model_overprediction'
        result['evidence_reason'] = 'weak_profile_has_no_reproducible_reconstruction_gain'

    return result


def detect_overlap_events(pre_ittfa_C, post_ittfa_C,
                          apex_tolerance=1, cosine_threshold=0.92):
    """Identify complete post ITTFA overlap without deleting profiles."""
    pre = np.asarray(pre_ittfa_C)
    post = np.asarray(post_ittfa_C)
    component_count = min(post.shape[1], pre.shape[1])
    events = []
    for m in range(component_count - 1):
        for n in range(m + 1, component_count):
            post_cosine = _profile_cosine(post[:, m], post[:, n])
            post_apex_distance = abs(int(np.argmax(post[:, m])) - int(np.argmax(post[:, n])))
            if post_apex_distance > apex_tolerance or post_cosine <= cosine_threshold:
                continue

            pre_cosine = _profile_cosine(pre[:, m], pre[:, n])
            pre_apex_distance = abs(int(np.argmax(pre[:, m])) - int(np.argmax(pre[:, n])))
            height_m = float(np.max(pre[:, m]))
            height_n = float(np.max(pre[:, n]))
            strong = m if height_m >= height_n else n
            weak = n if strong == m else m
            strong_height = max(height_m, height_n)
            weak_ratio = 0.0 if strong_height == 0 else min(height_m, height_n) / strong_height
            events.append({
                'component_i': int(m),
                'component_j': int(n),
                'strong_index': int(strong),
                'weak_index': int(weak),
                'pre_apex_distance': int(pre_apex_distance),
                'post_apex_distance': int(post_apex_distance),
                'pre_chrom_cosine': float(pre_cosine),
                'post_chrom_cosine': float(post_cosine),
                'weak_height_ratio': float(weak_ratio),
            })
    return events


def detect_component_selection_events(pre_ittfa_C, post_ittfa_C,
                                      boundary_threshold=0.05):
    """Collect rightlap, dislap, and boundary-half triggers without deletion."""
    pre = np.asarray(pre_ittfa_C)
    post = np.asarray(post_ittfa_C)
    component_count = min(pre.shape[1], post.shape[1])
    event_map = {}

    def add_pair(first, second, reason):
        first, second = sorted((int(first), int(second)))
        key = (first, second)
        if key not in event_map:
            height_first = float(np.max(pre[:, first]))
            height_second = float(np.max(pre[:, second]))
            strong = first if height_first >= height_second else second
            weak = second if strong == first else first
            event_map[key] = {
                'component_i': first,
                'component_j': second,
                'strong_index': int(strong),
                'weak_index': int(weak),
                'pre_apex_distance': abs(
                    int(np.argmax(pre[:, first])) - int(np.argmax(pre[:, second]))
                ),
                'post_apex_distance': abs(
                    int(np.argmax(post[:, first])) - int(np.argmax(post[:, second]))
                ),
                'pre_chrom_cosine': _profile_cosine(
                    pre[:, first], pre[:, second]
                ),
                'post_chrom_cosine': _profile_cosine(
                    post[:, first], post[:, second]
                ),
                'weak_height_ratio': min(height_first, height_second) /
                    max(height_first, height_second, 1e-12),
                'trigger_reasons': [],
            }
        if reason not in event_map[key]['trigger_reasons']:
            event_map[key]['trigger_reasons'].append(reason)

    for first in range(component_count - 1):
        for second in range(first + 1, component_count):
            similarity = _profile_cosine(post[:, first], post[:, second])
            apex_distance = abs(
                int(np.argmax(post[:, first])) - int(np.argmax(post[:, second]))
            )
            if apex_distance <= 1 and similarity > 0.92:
                add_pair(first, second, 'peak_rightlap')
            if similarity > 0.92 or (apex_distance < 3 and similarity > 0.8):
                add_pair(first, second, 'peak_dislap')

    # A boundary half profile is paired with its nearest-apex neighbour so the
    # selector can compare retaining and removing either member of that local
    # ambiguity.  A sole component is never removed by this detector.
    if component_count > 1:
        apices = [int(np.argmax(post[:, index])) for index in range(component_count)]
        for index in range(component_count):
            profile = post[:, index]
            maximum = float(np.max(profile))
            if maximum <= 0:
                continue
            active = np.where(profile > boundary_threshold)[0]
            if active.size == 0:
                continue
            left_half = active[0] == 0 and profile[0] > 0.5 * maximum
            right_half = (
                active[-1] == profile.size - 1
                and profile[-1] > 0.5 * maximum
            )
            if not (left_half or right_half):
                continue
            neighbours = [other for other in range(component_count) if other != index]
            neighbour = min(
                neighbours,
                key=lambda other: (abs(apices[index] - apices[other]), other),
            )
            add_pair(index, neighbour, 'dishalfremove')

    return [event_map[key] for key in sorted(event_map)]


def resolve_overlap_candidates(chrom_seg, pre_ittfa_C, events,
                               raw_chrom_seg=None,
                               r2_margin=1e-3, weak_height_ratio=0.2,
                               evidence_cv_margin=0.002,
                               evidence_weak_height_ratio=0.25,
                               evidence_high_spectral_similarity=0.99,
                               evidence_apex_similarity_threshold=0.99,
                               high_similarity_r2_margin=0.002):
    """Evaluate keep and delete branches for post ITTFA overlap events.

    The pre ITTFA profiles are used as the retention-shape candidate. ITTFA is
    not rerun on the keep branch because that would reproduce the collapse being
    diagnosed. FRR is attempted only for genuinely ambiguous direct NNLS cases.
    """
    # Keep one immutable profile set for evidence scoring.  ``working`` stores
    # cumulative deletions, but must not change the keep/delete comparison for
    # a later event that shares one of the same profiles.
    base_profiles = np.asarray(pre_ittfa_C, dtype=np.float32).copy()
    working = base_profiles.copy()
    protected_pairs = []
    decisions = []

    for event in events:
        m = int(event['component_i'])
        n = int(event['component_j'])
        weak = int(event['weak_index'])
        if m >= base_profiles.shape[1] or n >= base_profiles.shape[1]:
            continue
        if np.all(base_profiles[:, m] == 0) or np.all(base_profiles[:, n] == 0):
            continue

        keep_reconstruction, keep_St, keep_r2, _ = _nnls_reconstruct(chrom_seg, base_profiles)
        delete_candidate = base_profiles.copy()
        delete_candidate[:, weak] = 0
        delete_reconstruction, delete_St, delete_r2, _ = _nnls_reconstruct(
            chrom_seg, delete_candidate
        )

        keep_frr_r2 = None
        delete_frr_r2 = None
        keep_frr = None
        delete_frr = None
        direct_gap = keep_r2 - delete_r2

        # Full rank resolution is deliberately restricted to ambiguous cases.
        # This keeps the normal workflow tractable while allowing FRR to arbitrate
        # cases in which fixed-profile NNLS cannot decide.
        if abs(direct_gap) <= 0.01:
            for label, candidate in (('keep', base_profiles), ('delete', delete_candidate)):
                candidate_compact, _ = _compact_profiles_with_indices(candidate)
                candidate_count = candidate_compact.shape[1]
                if candidate_count < 2 or candidate_count > 3:
                    continue
                try:
                    peak_st, peak_ed = peak_find(candidate_compact)
                    if len(peak_st) != candidate_count:
                        continue
                    frr_reconstruction, _, _, frr_St = dynamic_FRR(
                        chrom_seg, candidate_compact, peak_st, peak_ed,
                        candidate_count,
                    )
                    frr_r2 = _reconstruction_r2(chrom_seg, frr_reconstruction)
                    if frr_reconstruction is None or frr_St is None or not np.isfinite(frr_r2):
                        continue
                    if label == 'keep':
                        keep_frr_r2 = float(frr_r2)
                        keep_frr = (candidate_compact, frr_St)
                    else:
                        delete_frr_r2 = float(frr_r2)
                        delete_frr = (candidate_compact, frr_St)
                except Exception:
                    # FRR is an arbitration aid. A failed boundary search must
                    # not prevent the deterministic NNLS comparison.
                    continue

        best_keep_r2 = max([x for x in (keep_r2, keep_frr_r2) if x is not None])
        best_delete_r2 = max([x for x in (delete_r2, delete_frr_r2) if x is not None])
        best_gap = best_keep_r2 - best_delete_r2
        weak_ratio = float(event['weak_height_ratio'])

        evidence = _overlap_evidence(
            chrom_seg,
            pre_ittfa_C,
            event,
            raw_chrom_seg=raw_chrom_seg,
            cv_margin=evidence_cv_margin,
            weak_height_ratio=evidence_weak_height_ratio,
            high_spectral_similarity=evidence_high_spectral_similarity,
            apex_similarity_threshold=evidence_apex_similarity_threshold,
        )

        apex_similarity_high = bool(evidence.get('apex_spectrum_high_similarity', False))
        cv_gap = evidence.get('cv_r2_keep_minus_delete')
        pre_area_ratio = evidence.get('pre_profile_area_ratio')

        # Direct apex spectra are used as an additional identity cue. Retention
        # requires a reproducible reconstruction gain and balanced profile
        # support. No TIC peak count or TIC shape calculation is used here.
        if apex_similarity_high:
            reproducible_two_profile_support = (
                best_gap > high_similarity_r2_margin
                and cv_gap is not None
                and cv_gap > evidence_cv_margin
                and (pre_area_ratio is None or pre_area_ratio >= 0.2)
            )
            if reproducible_two_profile_support:
                decision = 'retain_high_similarity_r2_support'
                reason = 'high_apex_spectral_similarity_with_reproducible_r2_support'
                evidence['evidence_class'] = 'high_similarity_true_coelution_support'
                evidence['evidence_reason'] = reason
                protected_pairs.append((m, n))
            else:
                decision = 'delete_high_similarity_without_reproducible_r2_support'
                reason = 'high_apex_spectral_similarity_without_reproducible_r2_support'
                evidence['evidence_class'] = 'model_overprediction'
                evidence['evidence_reason'] = reason
                working[:, weak] = 0
        elif evidence['evidence_class'] == 'ittfa_collapse':
            decision = 'retain_ittfa_collapse'
            reason = evidence['evidence_reason']
            protected_pairs.append((m, n))
        elif evidence['evidence_class'] == 'model_overprediction':
            decision = 'delete_model_overprediction'
            reason = evidence['evidence_reason']
            working[:, weak] = 0
        elif best_gap > r2_margin:
            decision = 'retain_r2_fallback'
            reason = 'fallback_keep_branch_improves_reconstruction'
            evidence['evidence_class'] = 'r2_fallback_retain'
            evidence['evidence_reason'] = reason
            protected_pairs.append((m, n))
        elif best_gap < -r2_margin:
            decision = 'delete_r2_fallback'
            reason = 'fallback_delete_branch_improves_reconstruction'
            evidence['evidence_class'] = 'r2_fallback_delete'
            evidence['evidence_reason'] = reason
            working[:, weak] = 0
        else:
            # A tie must still produce an explicit action.  Prefer deletion
            # only when the weaker profile has poor independent support; this
            # avoids silently protecting an embedded model artefact while
            # retaining balanced profiles when the evidence is symmetric.
            pre_area_ratio = evidence.get('pre_profile_area_ratio')
            weak_support_is_poor = (
                weak_ratio < max(weak_height_ratio, 0.25)
                or (pre_area_ratio is not None and pre_area_ratio < 0.5)
            )
            if weak_support_is_poor:
                decision = 'delete_r2_tie'
                reason = 'tie_break_delete_weak_or_unbalanced_profile'
                evidence['evidence_class'] = 'r2_tie_delete'
                evidence['evidence_reason'] = reason
                working[:, weak] = 0
            else:
                decision = 'retain_r2_tie'
                reason = 'tie_break_retain_balanced_profiles'
                evidence['evidence_class'] = 'r2_tie_retain'
                evidence['evidence_reason'] = reason
                protected_pairs.append((m, n))

        event_result = dict(event)
        event_result.update({
            'r2_keep_pre': float(keep_r2),
            'r2_delete_pre': float(delete_r2),
            'r2_keep_frr': keep_frr_r2,
            'r2_delete_frr': delete_frr_r2,
            'r2_keep_best': float(best_keep_r2),
            'r2_delete_best': float(best_delete_r2),
            'r2_keep_minus_delete': float(best_gap),
            **evidence,
            'decision': decision,
            'decision_reason': reason,
            'frr_keep_used': keep_frr is not None,
            'frr_delete_used': delete_frr is not None,
        })
        decisions.append(event_result)

    # Compact after all decisions and remap protected pairs to the compacted
    # column indices used by later deletion functions.
    compact, active = _compact_profiles_with_indices(working)
    index_map = {old: new for new, old in enumerate(active)}
    remapped_pairs = []
    for m, n in protected_pairs:
        if m in index_map and n in index_map:
            remapped_pairs.append((index_map[m], index_map[n]))

    if compact.shape[1] == 0 and pre_ittfa_C.shape[1] > 0:
        reserve = int(np.argmax([np.max(pre_ittfa_C[:, h]) for h in range(pre_ittfa_C.shape[1])]))
        compact = pre_ittfa_C[:, reserve:reserve + 1].copy()
        remapped_pairs = []
    return compact, remapped_pairs, decisions


def peak_oddremove(CC, C_ed, protected_indices=None):
    """
    Remove predicted profiles with iterated results that do not have a peak-shaped pattern, which possibly influenced by baseline.

    At least one component is retained.  When every unprotected profile is
    boundary truncated, the profile with the smallest relative boundary
    intensity is kept; integrated fitted response breaks any tie.  This
    prevents a valid DeepCS segment from disappearing solely because all
    first pass ITTFA profiles fail the same boundary test.
    """
    C = np.copy(CC)
    COM = com_find(C_ed)
    odddo = 'None'
    protected_indices = set(protected_indices or [])
    if COM > 1:
        removal_indices = []
        boundary_scores = {}
        fitted_areas = {}
        for m in range(COM):
            if m in protected_indices:
                continue
            peak_st, peak_ed = peak_find(C_ed[:, m].reshape((C_ed.shape[0], 1)))
            if not peak_st or not peak_ed:
                continue
            peak_start = int(peak_st[0])
            peak_end = int(peak_ed[0])
            profile_max = max(float(np.max(C_ed[:, m])), 1e-12)
            boundary_scores[m] = max(
                float(C_ed[peak_start, m]),
                float(C_ed[peak_end, m]),
            ) / profile_max
            fitted_areas[m] = float(np.trapz(np.maximum(C_ed[:, m], 0)))
            if C_ed[peak_start, m] > 0.1 or C_ed[peak_end, m] > 0.1:
                removal_indices.append(m)

        active_indices = [
            m for m in range(min(COM, C.shape[1]))
            if np.max(np.abs(C[:, m])) > 3e-3
        ]
        retained_indices = set(active_indices) - set(removal_indices)
        if removal_indices and not retained_indices:
            reserve = min(
                removal_indices,
                key=lambda m: (
                    boundary_scores.get(m, np.inf),
                    -fitted_areas.get(m, 0.0),
                ),
            )
            removal_indices.remove(reserve)

        for m in removal_indices:
            C[:, m] = 0
            odddo = 'Done'
    return C, odddo


def _is_ittfa_profile_collapse(pre_ittfa_C, post_ittfa_C, m, n,
                               min_pre_apex_separation=3,
                               max_pre_similarity=0.8):
    """Return True when ITTFA collapsed two initially distinct profiles.

    The post ITTFA overlap is handled by the caller. This guard only checks
    whether the corresponding pre ITTFA profiles were sufficiently distinct
    that automatic deletion would be unsafe.
    """
    if pre_ittfa_C is None:
        return False
    pre = np.asarray(pre_ittfa_C)
    post = np.asarray(post_ittfa_C)
    if pre.ndim != 2 or post.ndim != 2 or m >= pre.shape[1] or n >= pre.shape[1]:
        return False
    if m >= post.shape[1] or n >= post.shape[1]:
        return False
    if np.allclose(pre[:, m], 0) or np.allclose(pre[:, n], 0):
        return False
    pre_apex_separation = abs(int(np.argmax(pre[:, m])) - int(np.argmax(pre[:, n])))
    pre_similarity = 1 - cosine(pre[:, m], pre[:, n])
    if not np.isfinite(pre_similarity):
        return False
    return (
        pre_apex_separation >= min_pre_apex_separation
        and pre_similarity <= max_pre_similarity
    )


def peak_rightlap(CC, C_ed, pre_ittfa_C=None,
                  min_pre_apex_separation=3,
                  max_pre_similarity=0.8,
                  return_events=False):
    """Record complete post ITTFA overlaps without deleting a component.

    A post ITTFA overlap is not, by itself, evidence that one compound is
    absent.  The pre ITTFA profiles and the original GC MS segment are needed
    to compare a keep branch with a delete branch.  This function therefore
    only records candidate events.  The legacy return shape is retained unless
    ``return_events`` is requested.
    """
    C = np.copy(CC)
    disrightlapdo = 'None'
    disrightlapnum = []
    COM = com_find(C_ed)
    overlap_events = []
    if COM > 1 and pre_ittfa_C is not None:
        overlap_events = detect_overlap_events(
            pre_ittfa_C, C_ed, apex_tolerance=1, cosine_threshold=0.92
        )
    elif COM > 1:
        # Compatibility path for external callers that do not provide the
        # corresponding pre ITTFA profiles.
        for m in range(COM - 1):
            for n in range(m + 1, COM):
                post_apex_distance = int(abs(np.argmax(C_ed[:, m]) - np.argmax(C_ed[:, n])))
                post_cosine = _profile_cosine(C_ed[:, m], C_ed[:, n])
                if post_apex_distance <= 1 and post_cosine > 0.92:
                    strong = m if np.max(C[:, m]) >= np.max(C[:, n]) else n
                    weak = n if strong == m else m
                    overlap_events.append({
                        'component_i': int(m),
                        'component_j': int(n),
                        'strong_index': int(strong),
                        'weak_index': int(weak),
                        'pre_apex_distance': None,
                        'post_apex_distance': post_apex_distance,
                        'pre_chrom_cosine': None,
                        'post_chrom_cosine': float(post_cosine),
                        'weak_height_ratio': float(min(np.max(C[:, m]), np.max(C[:, n])) /
                                                   max(np.max(C[:, m]), np.max(C[:, n]), 1e-12)),
                    })
    # Complete overlap is resolved by ``resolve_overlap_candidates``.  No
    # profile is removed at this detection stage.
    if return_events:
        return C, disrightlapdo, disrightlapnum, overlap_events
    return C, disrightlapdo, disrightlapnum


def peak_dislap(CC, C_ed, pre_ittfa_C=None,
                min_pre_apex_separation=3,
                max_pre_similarity=0.8,
                protected_pairs=None):
    """
    Remove excessive overlapping chromatograms after iteration.

    The same ITTFA-collapse protection used by peak_rightlap is applied here
    so that a protected pair is not deleted by the later overlap check.
    """
    C = np.copy(CC)
    dislapdo = 'None'
    dislapnum = []
    protected_pairs = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in (protected_pairs or [])
        if len(pair) == 2
    }
    COM = com_find(C_ed)
    if COM > 1:
        for m in range(COM-1):
            for n in range(m+1, COM):
                if tuple(sorted((m, n))) in protected_pairs:
                    continue
                S_c = 1 - cosine((C_ed[:, m]), (C_ed[:, n]))
                if _is_ittfa_profile_collapse(
                        pre_ittfa_C, C_ed, m, n,
                        min_pre_apex_separation=min_pre_apex_separation,
                        max_pre_similarity=max_pre_similarity):
                    continue
                if S_c > 0.92:
                    if max(C[:, m]) >= max(C[:, n]):
                        C[:, n] = 0
                        dislapdo = "Done"
                        dislapnum.append(n)
                    else:
                        C[:, m] = 0
                        dislapdo = "Done"
                        dislapnum.append(m)

                elif np.abs(np.argmax(C_ed[:, m]) - np.argmax(C_ed[:, n])) < 3 and S_c > 0.8:
                    if max(C[:, m]) >= max(C[:, n]):
                        C[:, n] = 0
                        dislapdo = "Done"
                        dislapnum.append(n)
                    else:
                        C[:, m] = 0
                        dislapdo = "Done"
                        dislapnum.append(m)

    return C, dislapdo, dislapnum


def _unimod_python(c, rmod, cmod, imax=None):
    ns = c.shape[1]
    if imax is None:
        imax = np.argmax(c, axis=0)
    for j in range(0, ns):
        rmax = c[imax[j], j]
        k = imax[j]
        while k > 0:
            k = k-1
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 0  # 1e-30
                    if cmod == 1:
                        c[k, j] = c[k+1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k+1, j])/2
                            c[k+1, j] = c[k, j]
                            k = k+2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
        rmax = c[imax[j], j]
        k = imax[j]

        while k < c.shape[0]-1:
            k = k+1
            if k == 53:
                k = 53
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 1e-30
                    if cmod == 1:
                        c[k, j] = c[k-1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k-1, j])/2
                            c[k-1, j] = c[k, j]
                            k = k-2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
    return c


if _njit is not None:
    @_njit(cache=True)
    def _unimod_core(c, rmod, cmod, imax):
        """Numba-compiled scalar implementation of ``_unimod_python``."""
        ns = c.shape[1]
        for j in range(0, ns):
            rmax = c[imax[j], j]
            k = imax[j]
            while k > 0:
                k = k - 1
                if c[k, j] <= rmax:
                    rmax = c[k, j]
                else:
                    rmax2 = rmax * rmod
                    if c[k, j] > rmax2:
                        if cmod == 0:
                            c[k, j] = 0
                        if cmod == 1:
                            c[k, j] = c[k + 1, j]
                        if cmod == 2:
                            if rmax > 0:
                                c[k, j] = (c[k, j] + c[k + 1, j]) / 2
                                c[k + 1, j] = c[k, j]
                                k = k + 2
                            else:
                                c[k, j] = 0
                        rmax = c[k, j]
            rmax = c[imax[j], j]
            k = imax[j]
            while k < c.shape[0] - 1:
                k = k + 1
                if k == 53:
                    k = 53
                if c[k, j] <= rmax:
                    rmax = c[k, j]
                else:
                    rmax2 = rmax * rmod
                    if c[k, j] > rmax2:
                        if cmod == 0:
                            c[k, j] = 1e-30
                        if cmod == 1:
                            c[k, j] = c[k - 1, j]
                        if cmod == 2:
                            if rmax > 0:
                                c[k, j] = (c[k, j] + c[k - 1, j]) / 2
                                c[k - 1, j] = c[k, j]
                                k = k - 2
                            else:
                                c[k, j] = 0
                        rmax = c[k, j]
        return c

    def unimod(c, rmod, cmod, imax=None):
        """Use the Numba unimodality kernel when it is available."""
        if imax is None:
            imax = np.argmax(c, axis=0)
        return _unimod_core(c, rmod, cmod, imax)
else:
    def unimod(c, rmod, cmod, imax=None):
        """Pure-Python fallback used when Numba is unavailable."""
        return _unimod_python(c, rmod, cmod, imax)


def _fnnls_python(x, y, tole):
    xtx = np.dot(x, x.T)
    xty = np.dot(x, y.T)
    if tole == 'None':
        tol = 10*np.spacing(1)*np.linalg.norm(xtx)*max(xtx.shape)
    mn = xtx.shape
    P = np.zeros(mn[1])
    Z = np.array(range(1, mn[1]+1), dtype='int64')
    xx = np.zeros(mn[1])
    ZZ = Z-1
    w = xty-np.dot(xtx, xx)
    iter = 0
    itmax = 30*mn[1]
    z = np.zeros(mn[1])
    while np.any(Z) and np.any(w[ZZ] > tol):
        t = ZZ[np.argmax(w[ZZ])]
        P[t] = t+1
        Z[t] = 0
        PP = np.nonzero(P)[0]
        ZZ = np.nonzero(Z)[0]
        nzz = np.shape(ZZ)
        if len(PP) == 1:
            z[PP] = xty[PP]/xtx[PP, PP]
        elif len(PP) > 1:
            try:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]))
            except np.linalg.LinAlgError:
                small = 1e-6*np.identity(xtx[np.ix_(PP, PP)].shape[0])
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]+small))
        z[ZZ] = np.zeros(nzz)
        while np.any(z[PP] <= tol) and iter < itmax:
            iter += 1
            qq = np.nonzero((tuple(z <= tol) and tuple(P != 0)))
            epsilon = 1e-10
            divider = xx[qq] - z[qq]
            divider[abs(divider) < epsilon] = epsilon
            alpha = np.min(xx[qq] / divider)
            xx = xx + alpha*(z - xx)
            ij = np.nonzero(tuple(np.abs(xx) < tol) and tuple(P != 0))
            Z[ij[0]] = ij[0]+1
            P[ij[0]] = np.zeros(max(np.shape(ij[0])))
            PP = np.nonzero(P)[0]
            ZZ = np.nonzero(Z)[0]
            nzz = np.shape(ZZ)
            if len(PP) == 1:
                z[PP] = xty[PP]/xtx[PP, PP]
            elif len(PP) > 1:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]))
            z[ZZ] = np.zeros(nzz)
        xx = np.copy(z)
        xx[xx < 0] = 0
        w = xty - np.dot(xtx, xx)
    return {'xx': xx, 'w': w}


if _njit is not None:
    @_njit(cache=True)
    def _submatrix(A, idx):
        """Extract ``A[np.ix_(idx, idx)]`` without using ``np.ix_``."""
        n = idx.shape[0]
        out = np.empty((n, n), dtype=A.dtype)
        for i in range(n):
            for j in range(n):
                out[i, j] = A[idx[i], idx[j]]
        return out

    @_njit(cache=True)
    def _fnnls_core(xtx, xty, tol):
        """Numba NNLS kernel, numerically equivalent to ``_fnnls_python``.

        All inputs are converted to float64 to avoid mixed-dtype limitations
        in Numba. Relative floating-point differences are about 1e-7.
        """
        xtx = xtx.astype(np.float64)
        xty = xty.astype(np.float64)
        mn1 = xtx.shape[0]
        P = np.zeros(mn1)
        Z = np.arange(1, mn1 + 1, dtype=np.int64)
        xx = np.zeros(mn1)
        ZZ = Z - 1
        w = xty - np.dot(xtx, xx)
        it = 0
        itmax = 30 * mn1
        z = np.zeros(mn1)
        while np.any(Z) and np.any(w[ZZ] > tol):
            t = ZZ[np.argmax(w[ZZ])]
            P[t] = t + 1
            Z[t] = 0
            PP = np.nonzero(P)[0]
            ZZ = np.nonzero(Z)[0]
            nzz = ZZ.shape[0]
            if PP.shape[0] == 1:
                z[PP] = xty[PP] / xtx[PP[0], PP[0]]
            elif PP.shape[0] > 1:
                A = _submatrix(xtx, PP)
                if np.linalg.det(A) == 0:
                    small = 1e-6 * np.identity(A.shape[0])
                    z[PP] = np.dot(xty[PP], np.linalg.inv(A + small))
                else:
                    z[PP] = np.dot(xty[PP], np.linalg.inv(A))
            z[ZZ] = np.zeros(nzz)
            while np.any(z[PP] <= tol) and it < itmax:
                it += 1
                qq = np.nonzero(P != 0)[0]   # == nonzero(tuple(z<=tol) and tuple(P!=0))
                epsilon = 1e-10
                divider = xx[qq] - z[qq]
                divider[np.abs(divider) < epsilon] = epsilon
                alpha = np.min(xx[qq] / divider)
                xx = xx + alpha * (z - xx)
                ij = np.nonzero(P != 0)[0]   # == nonzero(tuple(abs(xx)<tol) and tuple(P!=0))
                Z[ij] = ij + 1
                P[ij] = np.zeros(ij.shape[0])
                PP = np.nonzero(P)[0]
                ZZ = np.nonzero(Z)[0]
                nzz = ZZ.shape[0]
                if PP.shape[0] == 1:
                    z[PP] = xty[PP] / xtx[PP[0], PP[0]]
                elif PP.shape[0] > 1:
                    A = _submatrix(xtx, PP)
                    if np.linalg.det(A) == 0:
                        small = 1e-6 * np.identity(A.shape[0])
                        z[PP] = np.dot(xty[PP], np.linalg.inv(A + small))
                    else:
                        z[PP] = np.dot(xty[PP], np.linalg.inv(A))
                z[ZZ] = np.zeros(nzz)
            xx = z.copy()
            xx[xx < 0] = 0
            w = xty - np.dot(xtx, xx)
        return xx, w

    def fnnls(x, y, tole):
        """Solve NNLS with the Numba kernel."""
        xtx = np.dot(x, x.T)
        xty = np.dot(x, y.T)
        if tole == 'None':
            tol = 10 * np.spacing(1) * np.linalg.norm(xtx) * max(xtx.shape)
        xx, w = _fnnls_core(xtx, xty, tol)
        return {'xx': xx, 'w': w}

    @_njit(cache=True, parallel=True)
    def _fnnls_batch_core(xtx, xty, tol):
        """Solve independent NNLS right-hand sides in parallel."""
        component_count, channel_count = xty.shape
        solutions = np.empty(
            (component_count, channel_count), dtype=np.float64
        )
        for channel in _prange(channel_count):
            solution, _ = _fnnls_core(xtx, xty[:, channel], tol)
            solutions[:, channel] = solution
        return solutions

    def fnnls_batch(x, y, tole='None'):
        """Fit all m/z channels with the same NNLS design matrix."""
        matrix = np.asarray(x)
        right_hand_sides = np.asarray(y)
        if right_hand_sides.ndim == 1:
            return fnnls(matrix, right_hand_sides, tole)['xx'][:, None]
        xtx = np.dot(matrix, matrix.T)
        xty = np.dot(matrix, right_hand_sides)
        if tole == 'None':
            tol = 10 * np.spacing(1) * np.linalg.norm(xtx) * max(xtx.shape)
        else:
            tol = float(tole)
        return _fnnls_batch_core(
            np.asarray(xtx, dtype=np.float64),
            np.asarray(xty, dtype=np.float64),
            float(tol),
        )
else:
    def fnnls(x, y, tole):
        """Pure-Python NNLS fallback used when Numba is unavailable."""
        return _fnnls_python(x, y, tole)

    def fnnls_batch(x, y, tole='None'):
        """Fallback that applies the scalar solver to each m/z channel."""
        right_hand_sides = np.asarray(y)
        if right_hand_sides.ndim == 1:
            return fnnls(x, right_hand_sides, tole)['xx'][:, None]
        solutions = np.zeros_like(right_hand_sides, dtype=np.float64)
        for channel in range(right_hand_sides.shape[1]):
            solutions[:, channel] = fnnls(
                x, right_hand_sides[:, channel], tole
            )['xx']
        return solutions


def ITTFA_PRO(X, C_0, COM, enable_gaussian_tail_correction=True):
    """
    Using ITTFA and initial chromatographic estimation to resolve mass spectra and refine chromatograms.
    """
    u, s, v = tl.truncated_svd(X, COM)
    T = np.dot(u, np.diag(s))

    C_ed = np.zeros((C_0.shape[0], COM), dtype=float32)

    for r in range(COM):
        C = C_0[:, r].reshape((C_0[:, r].shape[0], 1))
        l = 0
        while l < 30:
            C_st = C
            C = np.dot(np.dot(np.dot(T, np.linalg.pinv(np.dot(T.T, T))), T.T), C)
            C[C < 0] = 0
            C = unimod(C, 1.1, 2)

            if np.linalg.norm(C) != 0:
                C = C/np.linalg.norm(C)
            normc = np.linalg.norm(C-C_st)

            l += 1
            C = C.reshape((C_0.shape[0]))
            C_ed[:, r] = C
            C = C.reshape((C_0.shape[0], 1))

            if normc < 1e-6 or l == 30:
                peaks, _ = find_peaks(C[:, 0], height=0.003)
                if enable_gaussian_tail_correction and len(peaks) > 0:
                    inten_max = peaks[np.argmax([C[i] for i in peaks])]
                    peak_st, peak_ed = peak_find(C)
                    gaussleftfit = 'no'
                    gaussrightfit = 'no'

                    for z in range(inten_max, peak_st[0]+1, -1):
                        if C[z] > 0.003 and C[z-1] > C[z]:
                            gaussleftfit = 'yes'
                            lp = z
                            break
                    for y in range(inten_max, peak_ed[0]-1, 1):
                        if C[y] > 0.003 and C[y+1] > C[y]:
                            gaussrightfit = 'yes'
                            rp = y
                            break

                    if gaussleftfit == 'yes' and gaussrightfit == 'yes':
                        peak_x = np.arange(lp+1, rp)
                        peak_y = C[lp+1:rp, 0]
                        if len(peak_x) > 2 and np.abs(inten_max-lp) > 2 and np.abs(inten_max-rp) > 2:
                            if C[lp, 0] < 0.5*np.max(C):
                                s = 2*np.abs(getnearpos(C[lp:inten_max, 0], 0.5*np.max(C)) - inten_max)
                            elif C[rp, 0] < 0.5*np.max(C):
                                s = 2*np.abs(getnearpos(C[inten_max:rp, 0], 0.5*np.max(C)) - inten_max)
                            else:
                                if np.argmin([C[lp, 0], C[rp, 0]]) == 0:
                                    s = np.max(C)*np.abs(inten_max-lp)/(np.max(C)-C[lp, 0])
                                if np.argmin([C[lp, 0], C[rp, 0]]) == 1:
                                    s = np.max(C)*np.abs(rp-inten_max)/(np.max(C)-C[rp, 0])

                            popt, pcov = _fit_gaussian(
                                peak_x, peak_y,
                                p0=[np.max(C), inten_max, s],
                                maxfev=10000000)
                            for j in range(rp, C.shape[0], 1):
                                C[j, 0] = gaussian(j, *popt)
                                if gaussian(j, *popt) < 0.001:
                                    C[j:C.shape[0], 0] = 0
                                    break
                            for j in range(lp, -1, -1):
                                C[j, 0] = gaussian(j, *popt)
                                if gaussian(j, *popt) < 0.001:
                                    C[0:j, 0] = 0
                                    break

                    if gaussleftfit == 'no' and gaussrightfit == 'yes':
                        peak_x = np.arange(peak_st[0], rp)
                        peak_y = C[peak_st[0]:rp, 0]
                        if len(peak_x) > 2 and np.abs(inten_max-rp) > 2:
                            if np.abs(peak_st[0] - inten_max) > 1:
                                s = 2*np.abs(getnearpos(C[peak_st[0]:inten_max, 0], 0.5*np.max(C)) - inten_max)
                                popt, pcov = _fit_gaussian(
                                    peak_x, peak_y,
                                    p0=[np.max(C), inten_max, s],
                                    maxfev=10000000)
                                for j in range(rp, C.shape[0], 1):
                                    C[j, 0] = gaussian(j, *popt)
                                    if gaussian(j, *popt) < 0.001:
                                        C[j:C.shape[0], 0] = 0

                    if gaussleftfit == 'yes' and gaussrightfit == 'no':
                        peak_x = np.arange(lp+1, peak_ed[0])
                        peak_y = C[lp+1:peak_ed[0], 0]
                        if len(peak_x) > 2 and np.abs(inten_max-lp) > 2:
                            s = 2*np.abs(getnearpos(C[inten_max:peak_ed[0], 0], 0.5*np.max(C)) - inten_max)
                            popt, pcov = _fit_gaussian(
                                peak_x, peak_y,
                                p0=[np.max(C), inten_max, s],
                                maxfev=10000000)
                            for j in range(lp, -1, -1):
                                C[j, 0] = gaussian(j, *popt)
                                if gaussian(j, *popt) < 0.001:
                                    C[0:j, 0] = 0
                                    break

                C = C.reshape((C_0.shape[0]))
                C_ed[:, r] = C
                break

    chrom_origin = X
    CtC = np.dot(C_ed.T, C_ed)
    St = fnnls_batch(CtC, np.dot(C_ed.T, chrom_origin), tole='None')

    return C_ed, St


def ITTFA(X, C_0, COM):
    u, s, v = tl.truncated_svd(X, COM)
    T = np.dot(u, np.diag(s))

    C_ed = np.zeros((C_0.shape[0], COM), dtype=float32)

    for r in range(COM):
        C = C_0[:, r].reshape((C_0[:, r].shape[0], 1))
        l = 0
        # norm_set = []
        while l < 30:
            C_st = C
            C = np.dot(np.dot(np.dot(T, np.linalg.pinv(np.dot(T.T, T))), T.T), C)
            C[C < 0] = 0
            C = unimod(C, 1.1, 2)
            if np.linalg.norm(C) != 0:
                C = C/np.linalg.norm(C)
            normc = np.linalg.norm(C-C_st)
            l += 1
            C = C.reshape((C_0.shape[0]))
            C_ed[:, r] = C
            C = C.reshape((C_0.shape[0], 1))
            if normc < 1e-6 or l == 30:
                C = C.reshape((C_0.shape[0]))
                C_ed[:, r] = C
                break

    chrom_origin = X
    CtC = np.dot(C_ed.T, C_ed)
    St = fnnls_batch(CtC, np.dot(C_ed.T, chrom_origin), tole='None')

    return C_ed, St


def judge_max(n):
    n1 = str(float(n))
    n2 = n1.split('.')
    if n2[1] == '0':
        return int(n+1)
    else:
        return n


def judge_min(n):
    n1 = str(float(n))
    n2 = n1.split('.')
    if n2[1] == '0':
        return int(n-1)
    else:
        return n


def data_process(work_path, filename, dist, thres):
    """
    Read the CDF file and segment the data according to the peaks.
    """
    ncr = netcdf_reader(filename, bmmap=False)
    sie = hstack((ncr.f.variables['scan_index'].data, np.array([len(ncr.f.variables['intensity_values'].data)], dtype=int)))
    mat = ncr.mat(1, len(sie)-2, 1)
    RT = mat['rt']
    Xtest = mat['d']
    model_size = 128

    y_DeepSeg, ind_st_DeepSeg, ind_en_DeepSeg = Chromseg(
        work_path, mat, model_size, distance=dist, threshold=thres
    )
    return mat, RT, Xtest, ind_st_DeepSeg, ind_en_DeepSeg


def gaussian(x, *param):
    """
    Gaussian formula
    """
    with np.errstate(divide="ignore", invalid="ignore",
                     over="ignore", under="ignore"):
        return param[0] * np.exp(
            -np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.))
        )


def getnearpos(array,value):
    """
    Obtain the nearest location to  the given values in an array.
    """
    idx = (np.abs(array-value)).argmin()
    return idx


def R_squared(y_true, y_pred):
    """
    Custom metrics of DeepCPR model.
    """
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def FR(x, s, o, z, com, x_svd=None):
    """
    Subfunction of Full rank resolution.
    """
    xs = x[s, :]
    xs[xs < 0] = 0
    xz = x[z, :]
    # xo = x[o, :]
    xc = np.vstack((xs, xz))
    mc = np.vstack((xs, np.zeros(xz.shape)))

    u, s0, v = tl.truncated_svd(xc, com)
    t = np.dot(u, np.diag(s0))
    r = np.dot(np.dot(np.linalg.pinv(np.dot(t.T, t)), t.T), np.sum(mc, 1))
    # Reuse the invariant truncated SVD throughout the grid search.
    if x_svd is None:
        u1, s1, v1 = tl.truncated_svd(x, com)
    else:
        u1, s1, v1 = x_svd
    t1 = np.dot(u1, np.diag(s1))
    c = np.dot(t1, r)

    c1, ind = contrain_FR(c, s, o)
    c1[c1 < 0] = 0
    spec = x[s[ind], :]

    if c1[s[ind]] == 0:
        pu = 1e-6
    else:
        pu = c1[s[ind]]

    cc = c1/pu

    res_x = np.dot(np.array(cc, ndmin=2).T, np.array(spec, ndmin=2))
    # left_x = x - res_x
    spec = spec.reshape(1, spec.shape[0])
    return cc, spec, res_x


def _cached_original_fr(x, s, o, z, com, x_svd, cache):
    """Reuse deterministic FR results for repeated original-matrix partitions."""
    if cache is None:
        return FR(x, s, o, z, com, x_svd=x_svd)
    key = (tuple(s), tuple(o), tuple(z), int(com))
    result = cache.get(key)
    if result is None:
        result = FR(x, s, o, z, com, x_svd=x_svd)
        cache[key] = tuple(np.array(value, copy=True) for value in result)
    return tuple(np.array(value, copy=True) for value in result)


def contrain_FR(c, s, o):
    """
    Subfunction of Full rank resolution.
    """
    ind_s = np.argmax(np.abs(c[s]))
    if c[s][ind_s] < 0:
        c = -c

    if s[0] < o[0]:
        if c[s[-2]] < c[s[-1]]:
            ind1 = s[-1]
            ind2 = o[np.argmax(c[o])]
        else:
            ind1 = s[np.argmax(c[s])]
            ind2 = o[0]
    else:
        if c[s[1]] < c[s[0]]:
            ind1 = o[np.argmax(c[o])]
            ind2 = s[0]
        else:
            ind1 = o[-1]
            ind2 = s[np.argmax(c[s])]

    for i, indd in enumerate(np.arange(ind1, 0, -1)):
        if c[indd-1] >= c[indd]:
            c[0:indd] = 0
            break
        if c[indd-1] < 0:
            c[0:indd] = 0
            break

    for i, indd in enumerate(np.arange(ind2, len(c)-1, 1)):
        if c[indd+1] >= c[indd]:
            c[indd+1:len(c)] = 0
            break
        if c[indd+1] < 0:
            c[indd+1:len(c)] = 0
            break
    return c, ind_s


def full_rank_resolution(x, CC, peak_st, peak_ed, COM, x_svd=None,
                         original_fr_cache=None):
    """
    Full rank resolution for GC-MS resolution.
    The initial estimations are the start and end locations for every compounds.
    x_svd: optional precomputed tl.truncated_svd(x, COM); reused by FR for the
    original data matrix x across all grid-search combinations (bit-identical).
    """
    C = np.copy(CC)

    if COM == 2:
        p1 = peak_ed[0]
        p2 = peak_st[1]
        if p1 == p2:
            p2 = p2+1
        s = list(range(0, min(int(p1), int(p2))))
        o = list(range(min(int(p1), int(p2)), max(int(p1), int(p2))))
        z = list(range(max(int(p1), int(p2)), x.shape[0]))

        if len(s) < 3:
            re_x = None
            re_chrom = None
            R2 = 0
            S = None
        else:
            cc1, ss1, xx1 = _cached_original_fr(
                x, s, o, z, COM, x_svd, original_fr_cache
            )
            xx2 = x-xx1
            CC2 = C[:, 1].reshape(C.shape[0], 1)
            cc2, ss2 = ITTFA(xx2, CC2, 1)
            xx2 = np.dot(cc2, ss2)
            re_x = xx1+xx2

            re_chrom = np.zeros((x.shape[0], COM))
            re_chrom[:, 0] = np.sum(xx1, 1)
            re_chrom[:, 1] = np.sum(xx2, 1)

            R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            # xx = [xx1, xx2]
            S = np.concatenate((ss1, ss2), 0)

    if COM == 3:
        p1 = peak_ed[0]
        p2 = peak_st[1]
        if p1 == p2:
            p2 = p2+1
        s1 = list(range(0, min(int(p1), int(p2))))
        o1 = list(range(min(int(p1), int(p2)), max(int(p1), int(p2))))
        z1 = list(range(max(int(p1), int(p2)), x.shape[0]))

        p3 = peak_ed[1]
        p4 = peak_st[2]
        if p3 == p4:
            p4 = p4+1
        s3 = list(range(int(max(int(p3), int(p4))), x.shape[0]))
        o3 = list(range(min(int(p3), int(p4)), max(int(p3), int(p4))))
        z3 = list(range(0, min(int(p3), int(p4))))

        if len(s1) < 3 or len(s3) < 3:
            re_x = None
            re_chrom = None
            R2 = 0
            S = None
        else:
            cc1, ss1, xx1 = _cached_original_fr(
                x, s1, o1, z1, COM, x_svd, original_fr_cache
            )
            cc3, ss3, xx3 = _cached_original_fr(
                x, s3, o3, z3, COM, x_svd, original_fr_cache
            )

            xx2 = x-xx1-xx3
            CC2 = C[:, 1].reshape(C.shape[0], 1)
            cc2, ss2 = ITTFA(xx2, CC2, 1)
            xx2 = np.dot(cc2, ss2)

            re_x = xx1+xx2+xx3

            re_chrom = np.zeros((x.shape[0], COM))
            re_chrom[:, 0] = np.sum(xx1, 1)
            re_chrom[:, 1] = np.sum(xx2, 1)
            re_chrom[:, 2] = np.sum(xx3, 1)

            R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            # xx = [xx1, xx2, xx3]
            S = np.concatenate((ss1, ss2, ss3), 0)

    if COM == 4:
        p1 = peak_ed[0]
        p2 = peak_st[1]
        if p1 == p2:
            p2 = p2+1
        s1 = list(range(0, min(int(p1), int(p2))))
        o1 = list(range(min(int(p1), int(p2)), max(int(p1), int(p2))))
        z1 = list(range(max(int(p1), int(p2)), x.shape[0]))

        p3 = peak_ed[1]
        p4 = peak_st[2]
        if p3 == p4:
            p4 = p4+1
        s2 = list(range(min(int(p1), int(p2)), min(int(p3), int(p4))))
        o2 = list(range(min(int(p3), int(p4)), max(int(p3), int(p4))))
        z2 = list(range(max(int(p3), int(p4)), x.shape[0]))

        p5 = peak_ed[2]
        p6 = peak_st[3]
        if p5 == p6:
            p6 = p6+1
        s4 = list(range(int(max(int(p5), int(p6))), x.shape[0]))
        o4 = list(range(min(int(p5), int(p6)), max(int(p5), int(p6))))
        z4 = list(range(0, min(int(p5), int(p6))))

        if len(s1) < 3 or len(s2) < 3 or len(s4) < 3:
            re_x = None
            re_chrom = None
            R2 = 0
            S = None
        else:
            cc1, ss1, xx1 = _cached_original_fr(
                x, s1, o1, z1, COM, x_svd, original_fr_cache
            )
            xx_3 = x-xx1

            cc2, ss2, xx2 = FR(xx_3, s2, o2, z2, int(COM-1))
            cc4, ss4, xx4 = FR(xx_3, s4, o4, z4, int(COM-1))

            xx3 = x-xx1-xx2-xx4
            CC3 = C[:, 2].reshape(C.shape[0], 1)
            cc3, ss3 = ITTFA(xx3, CC3, 1)

            xx3 = np.dot(cc3, ss3)

            re_x = xx1+xx2+xx3+xx4

            re_chrom = np.zeros((x.shape[0], COM))
            re_chrom[:, 0] = np.sum(xx1, 1)
            re_chrom[:, 1] = np.sum(xx2, 1)
            re_chrom[:, 2] = np.sum(xx3, 1)
            re_chrom[:, 3] = np.sum(xx4, 1)

            R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            # xx = [xx1, xx2, xx3, xx4]
            S = np.concatenate((ss1, ss2, ss3, ss4), 0)

    if COM == 5:
        p1 = peak_ed[0]
        p2 = peak_st[1]
        if p1 == p2:
            p2 = p2+1
        s1 = list(range(0, min(int(p1), int(p2))))
        o1 = list(range(min(int(p1), int(p2)), max(int(p1), int(p2))))
        z1 = list(range(max(int(p1), int(p2)), x.shape[0]))

        p7 = peak_ed[3]
        p8 = peak_st[4]
        if p7 == p8:
            p8 = p8+1
        s5 = list(range(max(int(p7), int(p8)), x.shape[0]))
        o5 = list(range(min(int(p7), int(p8)), max(int(p7), int(p8))))
        z5 = list(range(0, min(int(p7), int(p8))))

        p3 = peak_ed[1]
        p4 = peak_st[2]
        if p3 == p4:
            p4 = p4+1
        s2 = list(range(min(int(p1), int(p2)), min(int(p3), int(p4))))
        o2 = list(range(min(int(p3), int(p4)), max(int(p3), int(p4))))
        z2 = list(range(max(int(p3), int(p4)), x.shape[0]))

        p5 = peak_ed[2]
        p6 = peak_st[3]
        if p5 == p6:
            p6 = p6+1
        s4 = list(range(max(int(p5), int(p6)), max(int(p7), int(p8))))
        o4 = list(range(min(int(p5), int(p6)), max(int(p5), int(p6))))
        z4 = list(range(0, min(int(p5), int(p6))))

        if len(s1) < 3 or len(s2) < 3 or len(s4) < 3 or len(s5) < 3:
            re_x = None
            re_chrom = None
            R2 = 0
            S = None
        else:
            cc1, ss1, xx1 = _cached_original_fr(
                x, s1, o1, z1, COM, x_svd, original_fr_cache
            )
            cc5, ss5, xx5 = _cached_original_fr(
                x, s5, o5, z5, COM, x_svd, original_fr_cache
            )

            xx_3 = x-xx1-xx5
            cc2, ss2, xx2 = FR(xx_3, s2, o2, z2, int(COM-2))
            cc4, ss4, xx4 = FR(xx_3, s4, o4, z4, int(COM-2))

            xx3 = x-xx1-xx2-xx4-xx5

            CC3 = C[:, 2].reshape(C.shape[0], 1)
            cc3, ss3 = ITTFA(xx3, CC3, 1)

            xx3 = np.dot(cc3, ss3)

            re_x = xx1+xx2+xx3+xx4+xx5

            re_chrom = np.zeros((x.shape[0], COM))
            re_chrom[:, 0] = np.sum(xx1, 1)
            re_chrom[:, 1] = np.sum(xx2, 1)
            re_chrom[:, 2] = np.sum(xx3, 1)
            re_chrom[:, 3] = np.sum(xx4, 1)
            re_chrom[:, 4] = np.sum(xx5, 1)

            R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            # xx = [xx1, xx2, xx3, xx4, xx5]
            S = np.concatenate((ss1, ss2, ss3, ss4, ss5), 0)

    return re_x, re_chrom, R2, S


def dynamic_FRR(x, CC, peak_st, peak_ed, COM):
    """
    fully full rank resolution according to the retention time information from the predicted chromatographic profiles.
    """
    C = np.copy(CC)
    # Compute the invariant truncated SVD once for every grid candidate.
    x_svd = tl.truncated_svd(x, COM)
    original_fr_cache = {}
    ar = [i for i in range(-2, 3)]

    if COM == 2:
        ar = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        metrcis = []
        best_result = None
        arlist = list(itertools.product(ar, ar))

        for j in range(len(arlist)):
            peak_st_1 = np.add(peak_st[1:2], arlist[j][0:1]).tolist()
            peak_ed_1 = np.add(peak_ed[0:1], arlist[j][1:2]).tolist()
            peak_st_dy = peak_st[0:1] + peak_st_1
            peak_ed_dy = peak_ed_1 + peak_ed[-1:]
            for g in range(1, len(peak_st_dy)):
                if peak_st_dy[g] < 2:
                    peak_st_dy[g] = 2
                if peak_st_dy[g] > C.shape[0]:
                    peak_st_dy[g] = C.shape[0]-1
            for g in range(0, (len(peak_ed_dy)-1)):
                if peak_ed_dy[g] > (C.shape[0]-3):
                    peak_ed_dy[g] = C.shape[0]-3
                if peak_ed_dy[g] < 2:
                    peak_ed_dy[g] = 2

            re_x, re_chrom, R2, S = full_rank_resolution(
                x, C, peak_st_dy, peak_ed_dy, COM, x_svd=x_svd,
                original_fr_cache=original_fr_cache,
            )
            metrcis.append((R2, j))
            if len(metrcis) > 1:
                metrcis.remove(metrcis[np.argmin([metrcis[i][0] for i in range(len(metrcis))])])
            if metrcis[0][1] == j:
                best_result = (re_x, re_chrom, R2, S)
            if metrcis[0][0] > 0.99:
                break

        return best_result

    if COM == 3:
        ar = [-5, -3, 0, 3, 5]
        metrcis = []
        best_result = None
        arlist = list(itertools.product(ar, ar, ar, ar))

        for j in range(len(arlist)):
            peak_st_1 = np.add(peak_st[1:3], arlist[j][0:2]).tolist()
            peak_ed_1 = np.add(peak_ed[0:2], arlist[j][2:4]).tolist()
            peak_st_dy = peak_st[0:1] + peak_st_1
            peak_ed_dy = peak_ed_1 + peak_ed[-1:]
            for g in range(1, len(peak_st_dy)):
                if peak_st_dy[g] < 2:
                    peak_st_dy[g] = 2
                if peak_st_dy[g] > C.shape[0]:
                    peak_st_dy[g] = C.shape[0]-1
            for g in range(0, (len(peak_ed_dy)-1)):
                if peak_ed_dy[g] > (C.shape[0]-3):
                    peak_ed_dy[g] = C.shape[0]-3
                if peak_ed_dy[g] < 2:
                    peak_ed_dy[g] = 2
            re_x, re_chrom, R2, S = full_rank_resolution(
                x, C, peak_st_dy, peak_ed_dy, COM, x_svd=x_svd,
                original_fr_cache=original_fr_cache,
            )
            metrcis.append((R2, j))
            if len(metrcis) > 1:
                metrcis.remove(metrcis[np.argmin([metrcis[i][0] for i in range(len(metrcis))])])
            if metrcis[0][1] == j:
                best_result = (re_x, re_chrom, R2, S)
            if metrcis[0][0] > 0.99:
                break

        return best_result

    if COM == 4:
        metrcis = []
        best_result = None
        arlist = list(itertools.product(ar, ar, ar, ar, ar, ar))

        for j in range(len(arlist)):
            peak_st_1 = np.add(peak_st[1:4], arlist[j][0:3]).tolist()
            peak_ed_1 = np.add(peak_ed[0:3], arlist[j][3:6]).tolist()
            peak_st_dy = peak_st[0:1] + peak_st_1
            peak_ed_dy = peak_ed_1 + peak_ed[-1:]
            for g in range(1, len(peak_st_dy)):
                if peak_st_dy[g] < 2:
                    peak_st_dy[g] = 2
                if peak_st_dy[g] > C.shape[0]:
                    peak_st_dy[g] = C.shape[0]-1
            for g in range(0, (len(peak_ed_dy)-1)):
                if peak_ed_dy[g] > (C.shape[0]-3):
                    peak_ed_dy[g] = C.shape[0]-3
                if peak_ed_dy[g] < 2:
                    peak_ed_dy[g] = 2
            re_x, re_chrom, R2, S = full_rank_resolution(
                x, C, peak_st_dy, peak_ed_dy, COM, x_svd=x_svd,
                original_fr_cache=original_fr_cache,
            )
            metrcis.append((R2, j))
            if len(metrcis) > 1:
                metrcis.remove(metrcis[np.argmin([metrcis[i][0] for i in range(len(metrcis))])])
            if metrcis[0][1] == j:
                best_result = (re_x, re_chrom, R2, S)
            if metrcis[0][0] > 0.99:
                break

        return best_result

    if COM == 5:
        metrcis = []
        best_result = None
        arlist = list(itertools.product(ar, ar, ar, ar, ar, ar, ar, ar))

        for j in range(len(arlist)):
            peak_st_1 = np.add(peak_st[1:5], arlist[j][0:4]).tolist()
            peak_ed_1 = np.add(peak_ed[0:4], arlist[j][4:8]).tolist()
            peak_st_dy = peak_st[0:1] + peak_st_1
            peak_ed_dy = peak_ed_1 + peak_ed[-1:]
            for g in range(1, len(peak_st_dy)):
                if peak_st_dy[g] < 2:
                    peak_st_dy[g] = 2
                if peak_st_dy[g] > C.shape[0]:
                    peak_st_dy[g] = C.shape[0]-1
            for g in range(0, (len(peak_ed_dy)-1)):
                if peak_ed_dy[g] > (C.shape[0]-3):
                    peak_ed_dy[g] = C.shape[0]-3
                if peak_ed_dy[g] < 2:
                    peak_ed_dy[g] = 2
            re_x, re_chrom, R2, S = full_rank_resolution(
                x, C, peak_st_dy, peak_ed_dy, COM, x_svd=x_svd,
                original_fr_cache=original_fr_cache,
            )
            metrcis.append((R2, j))
            if len(metrcis) > 1:
                metrcis.remove(metrcis[np.argmin([metrcis[i][0] for i in range(len(metrcis))])])
            if metrcis[0][1] == j:
                best_result = (re_x, re_chrom, R2, S)
            if metrcis[0][0] > 0.99:
                break

        return best_result

    raise ValueError("dynamic_FRR supports component counts from 2 to 5")


def _whittaker_penalty(m, lambda_, differences=1):
    """Precompute the constant penalty matrix used by :func:`WhittakerSmooth`.

    It only depends on the signal length ``m`` (plus ``lambda_`` and the
    difference order), not on the data or the weights, so it can be built once
    per segment and reused for every m/z column and every airPLS iteration.
    The operations mirror the original ``lambda_*E.T*E`` exactly, so the
    resulting matrix is bit-for-bit identical.
    """
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:]-E[:-1]
    return (lambda_ * E.T) * E


def WhittakerSmooth(x, w, lambda_, differences=1, penalty=None):
    """
    Subfunction of airPLS.
    """
    m = x.size
    if differences == 1 and m > 1:
        # For porder=1, A = W + lambda_*E.T@E is tridiagonal. LAPACK's
        # banded solver avoids sparse assembly and SuperLU. Its different
        # rounding path changes results by only about 1e-15 versus spsolve.
        main = w + 2.0 * lambda_
        main[0] = w[0] + lambda_
        main[-1] = w[-1] + lambda_
        off = np.full(m - 1, -lambda_)
        ab = np.zeros((3, m))
        ab[1, :] = main
        ab[0, 1:] = off      # Upper diagonal A[j-1, j] = -lambda_
        ab[2, :-1] = off     # Lower diagonal A[j+1, j] = -lambda_
        z = solve_banded((1, 1), ab, w * x)
        # Match spsolve by returning a one-dimensional result.
        return z
    # Retain the general sparse path for higher-order differences or m <= 1.
    X = np.matrix(x)
    if penalty is None:
        E = eye(m, format='csc')
        for i in range(differences):
            E = E[1:]-E[:-1]
        penalty = (lambda_ * E.T) * E
    W = diags(w, 0, shape=(m, m), format='csc')
    A = csc_matrix(W+penalty)
    B = csc_matrix((w * x).reshape((m, 1)))
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=500, porder=1, itermax=15, penalty=None):
    """
    Function for baseline removing.
    """
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax+1):
        z = WhittakerSmooth(x, w, lambda_, porder, penalty)
        d = x-z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001*(abs(x)).sum() or i == itermax):
            break
        w[d >= 0] = 0
        w[d < 0] = np.exp(i*np.abs(d[d < 0])/dssn)
        w[0] = np.exp(i*(d[d < 0]).max()/dssn)
        w[-1] = w[0]
    return z


if _njit is not None:
    @_njit(cache=True)
    def _solve_airpls_tridiagonal(rhs, weights, lambda_):
        """Solve the first-order Whittaker system with the Thomas algorithm."""
        size = rhs.shape[0]
        cprime = np.empty(size, dtype=np.float64)
        dprime = np.empty(size, dtype=np.float64)
        result = np.empty(size, dtype=np.float64)
        off = -lambda_

        diagonal = weights[0] + lambda_
        cprime[0] = off / diagonal if size > 1 else 0.0
        dprime[0] = rhs[0] / diagonal
        for row in range(1, size):
            diagonal = weights[row] + (lambda_ if row == size - 1 else 2.0 * lambda_)
            denominator = diagonal - off * cprime[row - 1]
            cprime[row] = off / denominator if row < size - 1 else 0.0
            dprime[row] = (rhs[row] - off * dprime[row - 1]) / denominator

        result[size - 1] = dprime[size - 1]
        for row in range(size - 2, -1, -1):
            result[row] = dprime[row] - cprime[row] * result[row + 1]
        return result


    @_njit(cache=True, parallel=True)
    def _airpls_correct_matrix_core(values, lambda_, itermax):
        """Apply independent airPLS corrections to all m/z channels."""
        scan_count, channel_count = values.shape
        corrected = np.zeros((scan_count, channel_count), dtype=np.float64)
        for channel in _prange(channel_count):
            signal = values[:, channel]
            if not np.any(signal):
                continue
            absolute_sum = np.sum(np.abs(signal))
            weights = np.ones(scan_count, dtype=np.float64)
            baseline = np.zeros(scan_count, dtype=np.float64)
            residual = np.zeros(scan_count, dtype=np.float64)
            rhs = np.empty(scan_count, dtype=np.float64)

            for iteration in range(1, itermax + 1):
                for row in range(scan_count):
                    rhs[row] = weights[row] * signal[row]
                baseline = _solve_airpls_tridiagonal(rhs, weights, lambda_)

                negative_sum = 0.0
                maximum_negative = -np.inf
                for row in range(scan_count):
                    residual[row] = signal[row] - baseline[row]
                    if residual[row] < 0.0:
                        negative_sum += residual[row]
                        if residual[row] > maximum_negative:
                            maximum_negative = residual[row]
                negative_sum = abs(negative_sum)
                if negative_sum < 0.001 * absolute_sum or iteration == itermax:
                    break

                for row in range(scan_count):
                    if residual[row] >= 0.0:
                        weights[row] = 0.0
                    else:
                        weights[row] = np.exp(
                            iteration * abs(residual[row]) / negative_sum
                        )
                weights[0] = np.exp(iteration * maximum_negative / negative_sum)
                weights[scan_count - 1] = weights[0]

            for row in range(scan_count):
                value = signal[row] - baseline[row]
                corrected[row, channel] = value if value > 0.0 else 0.0
        return corrected


def airPLS_correct_matrix(values, lambda_=500, itermax=15):
    """Baseline-correct all m/z channels with the fastest available solver."""
    input_dtype = np.asarray(values).dtype
    matrix = np.asarray(values, dtype=np.float64)
    if _njit is not None and matrix.shape[0] > 1:
        corrected = _airpls_correct_matrix_core(
            np.ascontiguousarray(matrix), float(lambda_), int(itermax)
        )
    else:
        corrected = np.zeros_like(matrix)
        penalty = _whittaker_penalty(matrix.shape[0], lambda_, 1)
        for channel in range(matrix.shape[1]):
            if np.any(matrix[:, channel]):
                corrected[:, channel] = matrix[:, channel] - airPLS(
                    matrix[:, channel],
                    lambda_=lambda_,
                    porder=1,
                    itermax=itermax,
                    penalty=penalty,
                )
        corrected[corrected < 0] = 0
    return corrected.astype(input_dtype, copy=False)


def PW_loss(y_true, y_pred):
    """
    Custom loss of DeepCPR model.
    """
    weight_high = tf.cast(50, dtype=float32)
    weight_low = tf.cast(10, dtype=float32)
    threshold = tf.cast(0.001, dtype=float32)

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mask = tf.greater(y_true, threshold)

    a = tf.square(tf.subtract(y_true, y_pred))
    loss = tf.reduce_mean(tf.where(mask, weight_high*a, weight_low*a))

    return loss


def MAE(x1, x2):
    """
    Function for calculation of Mean Average Error.
    """
    return np.sum(np.abs([x1[i]-x2[i] for i in range(len(x1))])) / len(x1)


def save_as_msp(filename, RT, mz_values, intensity_values):
    """
    Save resolved spectra as msp format files.
    """
    with open(filename, 'w') as file:
        file.write("Name: Unknown Compound\n")
        file.write(f"RT: {str(RT)}\n")
        file.write(f"Num Peaks: {len(mz_values)}\n")
        for mz, intensity in zip(mz_values, intensity_values):
            file.write(f"{mz} {intensity}\n")


def _deepcpr_impl(work_path, modelpath, filename, figure_savepath, dist, thres,
                  generate_image,
                  component_selection_config=None,
                  parallel_segments=True, image_formats=("png",)):
    """
    Main function of DeepCPR resolution.
    work_path: the path of GC-MS data segment model, which should lead to the model itself, like (.h5).
    modelpath: path of the chromatographic profile prediction model (.h5/.keras or .onnx).
    filename: path of data waiting for resolution.
    figure_savepath: path to store pictures.
    dist, thres: some parameters for data segment model. (default 3, 15)
    generate_image: whether to generate resolution figures.
    image_formats: one or more resolution-figure formats (PNG and/or SVG).
    """

    image_formats = (
        _normalize_image_formats(image_formats) if generate_image else ("png",)
    )

    if component_selection_config is None:
        component_selection_config = ComponentSelectionConfig()
    elif isinstance(component_selection_config, dict):
        component_selection_config = ComponentSelectionConfig(
            **component_selection_config
        )

    mat, RT, Xtest, ind_st_DeepSeg, ind_en_DeepSeg = data_process(
        work_path, filename, dist, thres
    )
    TICsum_origin = np.sum(Xtest, axis=1)

    mz_min = math.floor(judge_min(min(mat['mz'])))
    mz_max = math.ceil(max(mat['mz']))
    if mz_min < 0:
        Xtest = Xtest[:, 1:]
        mz_min = 0
        mz_max = Xtest.shape[1]
    if mz_max > 800:
        raise ValueError('m/z axis of this file exceeds the '
                         '800-bin input window of the DeepCPR model')

    # data segment, baseline correction, chromatographic profile prediction
    x_seg_pre = np.zeros((len(ind_st_DeepSeg), 128, 800), dtype=np.float32)
    chrom_divid = []
    for i in range(len(ind_st_DeepSeg)):
        test = np.zeros((ind_en_DeepSeg[i]-ind_st_DeepSeg[i], 800),
                        dtype=float32)
        test[:, mz_min:mz_max] = Xtest[int(ind_st_DeepSeg[i]):int(ind_en_DeepSeg[i]), :]

        test_br = airPLS_correct_matrix(test, lambda_=500, itermax=15)
        chrom_divid.append(test_br)
        test_pre = test_br/np.max(test_br)
        dis_st = math.ceil((128-(int(ind_en_DeepSeg[i])-int(ind_st_DeepSeg[i])))/2)

        x_seg = np.zeros((128, 800), dtype=float32)
        x_seg[int(dis_st):int(dis_st+ind_en_DeepSeg[i]-ind_st_DeepSeg[i]), :] = test_pre
        x_seg_pre[i] = x_seg
    X = x_seg_pre.reshape(x_seg_pre.shape[0], x_seg_pre.shape[1], 1, x_seg_pre.shape[2])

    restored_model = _load_cached_model(modelpath)
    prechrom = restored_model.predict(X, verbose=0)

    predict = np.asarray(prechrom, dtype=float32).reshape(
        prechrom.shape[0], 128, 5
    )

    def _resolve_segment_job(i):
        """Resolve one independent chromatographic segment."""
        def _run():
            peak_excel_single = []
            peak_excel_seg = []
            ms_single = []
            tol_com = [0]
            # Preserve the original ``continue`` exits without changing
            # any of the established segment-resolution branches.
            for _segment_guard in range(1):
                num = i
                wayname = 'non_frr'
                dis_st = math.ceil((128-(int(ind_en_DeepSeg[i])-int(ind_st_DeepSeg[i])))/2)
                def _resolve_ittfa(segment, profiles, components):
                    return ITTFA_PRO(
                        segment, profiles, components,
                        enable_gaussian_tail_correction=True,
                    )

                noise_width = 2
                if i > 0:
                    if (int(ind_st_DeepSeg[i]) - int(ind_en_DeepSeg[i-1])) > noise_width:
                        noise_windows = TICsum_origin[int(ind_st_DeepSeg[i]-noise_width):int(ind_st_DeepSeg[i])]
                    else:
                        noise_windows = TICsum_origin[int(ind_en_DeepSeg[i]):int(ind_en_DeepSeg[i]+noise_width)]
                if i == 0:
                    if int(ind_st_DeepSeg[i]) > noise_width:
                        noise_windows = TICsum_origin[int(ind_st_DeepSeg[i]-noise_width):int(ind_st_DeepSeg[i])]
                    else:
                        noise_windows = TICsum_origin[int(ind_en_DeepSeg[i]):int(ind_en_DeepSeg[i]+noise_width)]

                chrom_seg, C_0 = data_restore(num, dis_st, predict, ind_st_DeepSeg, ind_en_DeepSeg, mz_min, mz_max, chrom_divid)
                raw_chrom_seg = np.zeros_like(chrom_seg)
                raw_chrom_seg[:, mz_min:mz_max] = Xtest[int(ind_st_DeepSeg[i]):int(ind_en_DeepSeg[i]), :]

                # The uncorrected component-selection anchor is built lazily below,
                # only after an overlap event has been detected.  Most segments do
                # not enter component selection and do not need this duplicate
                # preprocessing path.
                C_component_anchor = None

                C = peak_preprocess(
                    C_0,
                    enable_gaussian_tail_correction=True,
                )

                if np.all(C == 0):
                    continue
                else:
                    C1 = np.zeros_like(C)
                    C1 = peak_trans(C, C1)
                    C_in = np.copy(C1)

                    C_out, halfdo = peak_halfremove(C_in)
                    if halfdo == 'Done':
                        C2 = np.zeros_like(C)
                        C2 = peak_trans(C_out, C2)
                        C_in = np.copy(C2)
                    else:
                        C_in = C_out

                    C_out, testdo, lapdo, lapnum = testt(C_in)
                    if testdo == 'Done':
                        C4 = np.zeros_like(C)
                        C4 = peak_trans(C_out, C4)
                        C_in = np.copy(C4)
                    else:
                        C_in = C_out

                    C_in = tail_fix(
                        C_in,
                        enable_gaussian_tail_correction=True,
                    )

                    COM = com_find(C_in)
                    C_ed_it, St_it = _resolve_ittfa(chrom_seg, C_in, COM)
                    overlap_events = detect_overlap_events(C_in, C_ed_it)
                    overlap_protected_indices = sorted({
                        int(index)
                        for event in overlap_events
                        for index in (event['component_i'], event['component_j'])
                    })
                    C_out, odddo = peak_oddremove(
                        C_in, C_ed_it,
                        protected_indices=overlap_protected_indices,
                    )
                    if np.all(C_out == 0):
                        continue

                    if odddo == 'Done':
                        C5 = np.zeros_like(C)
                        C5 = peak_trans(C_out, C5)
                        C_in = np.copy(C5)

                        COM = com_find(C_in)
                        C_ed_it, St_it = _resolve_ittfa(chrom_seg, C_in, COM)
                        overlap_events = detect_overlap_events(C_in, C_ed_it)
                    else:
                        overlap_events = detect_overlap_events(C_in, C_ed_it)

                    component_selection_events = detect_component_selection_events(
                        C_in, C_ed_it
                    )

                    if component_selection_events:
                        # Build the uncorrected anchor only for disputed segments.
                        # It is used for candidate positions and identities, not as
                        # the ordinary Gaussian-corrected resolution input.
                        C_component_anchor = peak_preprocess(
                            C_0,
                            enable_gaussian_tail_correction=False,
                        )
                        anchor_buffer = np.zeros_like(C_component_anchor)
                        C_component_anchor = peak_trans(
                            C_component_anchor, anchor_buffer
                        )
                        anchor_out, anchor_halfdo = peak_halfremove(
                            C_component_anchor
                        )
                        if anchor_halfdo == 'Done':
                            anchor_buffer = np.zeros_like(C_component_anchor)
                            C_component_anchor = peak_trans(
                                anchor_out, anchor_buffer
                            )
                        else:
                            C_component_anchor = anchor_out
                        anchor_out, anchor_testdo, _, _ = testt(
                            C_component_anchor
                        )
                        if anchor_testdo == 'Done':
                            anchor_buffer = np.zeros_like(C_component_anchor)
                            C_component_anchor = peak_trans(
                                anchor_out, anchor_buffer
                            )
                        else:
                            C_component_anchor = anchor_out

                        C_component_anchor = _align_component_anchors(
                            C_in, C_component_anchor
                        )

                    C_out, disrightlapdo, disrightlapnum, rightlap_events = peak_rightlap(
                        C_in, C_ed_it, pre_ittfa_C=C_in, return_events=True
                    )
                    # ``peak_rightlap`` now only records complete overlap.  Resolve
                    # each event against the original GC MS segment before any later
                    # deletion rule is allowed to act.
                    overlap_events = rightlap_events or overlap_events
                    protected_pairs = []
                    component_selection_result = None
                    component_selection_applied = False
                    # Keep the Gaussian-corrected pre-ITTFA profiles that correspond
                    # to the component-selection indices.  Once the component count
                    # is selected, the manuscript workflow still compares this fixed
                    # component set with FRR; FRR must not be allowed to change the
                    # component count.
                    component_selection_pre_profiles = np.copy(C_in)
                    if component_selection_events:
                        try:
                            def _component_selection_frr(candidate_C,
                                                         candidate_indices):
                                """Run FRR only for a shortlisted disputed candidate."""
                                # Use the original pre ITTFA anchor profiles for the
                                # FRR boundary search. The MCR-fitted profiles are
                                # useful for cross validation, but using them as FRR
                                # input would silently change the chromatographic
                                # shapes that the legacy FRR branch evaluates.
                                candidate_C = np.asarray(
                                    C_component_anchor[:, list(candidate_indices)],
                                    dtype=np.float32,
                                )
                                compact_C, _ = _compact_profiles_with_indices(candidate_C)
                                component_count = int(compact_C.shape[1])
                                # The existing dynamic FRR implementation is defined
                                # for the two and three component arbitration cases.
                                # Larger candidates remain governed by masked CV and
                                # stability rather than triggering a combinatorial FRR
                                # search.
                                if component_count < 2 or component_count > 3:
                                    return {
                                        'used': False,
                                        'r2': None,
                                        'error': 'FRR confirmation limited to 2 or 3 components',
                                    }
                                peak_st, peak_ed = peak_find(compact_C)
                                if (len(peak_st) != component_count
                                        or len(peak_ed) != component_count):
                                    return {
                                        'used': False,
                                        'r2': None,
                                        'error': 'FRR peak boundaries were not found for all components',
                                    }
                                try:
                                    frr_reconstruction, _, frr_r2, _ = dynamic_FRR(
                                        chrom_seg,
                                        compact_C,
                                        peak_st,
                                        peak_ed,
                                        component_count,
                                    )
                                    if frr_reconstruction is None or not np.isfinite(frr_r2):
                                        return {
                                            'used': False,
                                            'r2': None,
                                            'error': 'FRR returned a non finite R2',
                                        }
                                    return {
                                        'used': True,
                                        'r2': float(frr_r2),
                                        'component_count': component_count,
                                    }
                                except Exception as exc:
                                    return {
                                        'used': False,
                                        'r2': None,
                                        'error': '{}: {}'.format(
                                            type(exc).__name__, str(exc)
                                        ),
                                    }

                            # Use additional scans on both sides only for the
                            # chromatographic-shape existence test.  The ordinary MCR
                            # fit, FRR comparison, and quantitative integration keep
                            # using the original DeepCS segment.
                            shape_context_scans = max(
                                0,
                                int(component_selection_config.peak_shape_context_scans),
                            )
                            segment_scan_start = int(ind_st_DeepSeg[i])
                            segment_scan_end = int(ind_en_DeepSeg[i])
                            shape_scan_start = max(
                                0, segment_scan_start - shape_context_scans
                            )
                            shape_scan_end = min(
                                Xtest.shape[0],
                                segment_scan_end + shape_context_scans,
                            )
                            shape_observed = np.zeros(
                                (shape_scan_end - shape_scan_start, chrom_seg.shape[1]),
                                dtype=float32,
                            )
                            shape_observed[:, mz_min:mz_max] = Xtest[
                                shape_scan_start:shape_scan_end, :
                            ]
                            shape_anchors = np.zeros(
                                (shape_scan_end - shape_scan_start,
                                 C_component_anchor.shape[1]),
                                dtype=float32,
                            )
                            shape_anchor_offset = segment_scan_start - shape_scan_start
                            shape_anchors[
                                shape_anchor_offset:
                                shape_anchor_offset + C_component_anchor.shape[0], :
                            ] = C_component_anchor
                            shape_context = {
                                'scan_start': int(shape_scan_start),
                                'scan_end_exclusive': int(shape_scan_end),
                                'segment_offset': int(shape_anchor_offset),
                                'segment_scan_count': int(C_component_anchor.shape[0]),
                                'left_context_scans': int(
                                    segment_scan_start - shape_scan_start
                                ),
                                'right_context_scans': int(
                                    shape_scan_end - segment_scan_end
                                ),
                            }

                            component_selection_result = select_component_model(
                                chrom_seg,
                                C_component_anchor,
                                component_selection_events,
                                config=component_selection_config,
                                frr_evaluator=_component_selection_frr,
                                prediction_profiles=C_0,
                                peak_shape_observed=shape_observed,
                                peak_shape_anchors=shape_anchors,
                                peak_shape_context=shape_context,
                            )
                        except Exception as exc:
                            component_selection_result = {
                                'record': {
                                    'method': 'paired_repeated_cv_stability_v2',
                                    'selection_status': 'failed',
                                    'selection_error': '{}: {}'.format(
                                        type(exc).__name__, str(exc)
                                    ),
                                }
                            }

                        if (component_selection_result is not None
                                and 'chromatograms' in component_selection_result):
                            selected_C = np.asarray(
                                component_selection_result['chromatograms'],
                                dtype=float32,
                            )
                            selected_St = np.asarray(
                                component_selection_result['spectra'],
                                dtype=float32,
                            )
                            C_in = np.copy(selected_C)
                            C_ed_it = np.copy(selected_C)
                            St_it = np.copy(selected_St)
                            protected_pairs = list(itertools.combinations(
                                range(selected_C.shape[1]), 2
                            ))
                            component_selection_applied = True
                            disrightlapdo = 'ComponentSelection'

                    if overlap_events and not component_selection_applied:
                        resolved_C, protected_pairs, _ = resolve_overlap_candidates(
                            chrom_seg, C_in, overlap_events,
                            raw_chrom_seg=raw_chrom_seg,
                        )
                        C_in = resolved_C
                        if C_in.shape[1] > 0:
                            _, St_candidate, _, _ = _nnls_reconstruct(chrom_seg, C_in)
                            if St_candidate is not None:
                                # The keep branch is deliberately represented by the
                                # pre ITTFA profiles plus NNLS spectra.  Re-running
                                # ITTFA here would recreate the collapse being tested.
                                C_ed_it = np.copy(C_in)
                                St_it = St_candidate
                        disrightlapdo = 'Candidate'
                    overlap_keep_active = bool(protected_pairs)

                    C_in_0 = np.copy(C_in)
                    C_pre_0 = np.copy(C_in)

                    COM = com_find(C_in)
                    if COM > 1:
                        if component_selection_applied:
                            C_ed_it = np.copy(component_selection_result['chromatograms'])
                            St_it = np.copy(component_selection_result['spectra'])
                        else:
                            C_ed_it, St_it = _resolve_ittfa(chrom_seg, C_in, COM)
                        C_out, dislapdo, dislapnum = peak_dislap(
                            C_in, C_ed_it, pre_ittfa_C=C_in,
                            protected_pairs=protected_pairs,
                        )
                        dislapnum = list(set(dislapnum))
                        protected_indices = sorted({
                            int(index)
                            for pair in protected_pairs
                            for index in pair
                        })
                        C_out, dishalfdo, dishalfnum = dishalfremove(
                            C_in, C_ed_it, threshold=0.05,
                            protected_indices=protected_indices,
                        )
                        dishalfnum = list(set(dishalfnum))
                    else:
                        dislapdo = 'None'
                        dishalfdo = 'None'
                        dislapnum = []
                        dishalfnum = []

                    hard_boundary_failure_indices = _hard_boundary_half_indices(
                        C_in, C_ed_it, dishalfnum
                    )

                    if lapdo == 'Done':
                        if dislapdo == 'Done' or dishalfdo == 'Done':
                            if len(dislapnum) != 0:
                                for k in dislapnum:
                                    C_in_0[:, k] = 0
                                    if k in lapnum:
                                        C_pre_0[:, k] = 0

                            if len(dishalfnum) != 0:
                                for k in dishalfnum:
                                    C_in_0[:, k] = 0
                                    if k in lapnum:
                                        C_pre_0[:, k] = 0

                            COM_C_pre_0 = com_find(C_pre_0)
                            if COM_C_pre_0 == 0:
                                reserve = np.argmax([np.max(C_in[:, h]) for h in range(COM)])
                                C_in_pre = np.zeros_like(C)
                                C_in_pre[:, reserve] = C_in[:, reserve]
                                C_pre_0 = C_in_pre

                            C_aa = np.copy(C_pre_0)
                            C_bb = np.zeros_like(C)
                            C_bb = peak_trans(C_aa, C_bb)
                            C_pre = np.copy(C_bb)

                            CpCp = np.dot(C_pre.T, C_pre)
                            St_pre = fnnls_batch(
                                CpCp, np.dot(C_pre.T, chrom_seg), tole='None'
                            )

                            chrom_resol_pre = np.dot(C_pre, St_pre)
                            R2_pre = 1 - np.var(chrom_resol_pre - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                            # ITTFA
                            COM_C_in = com_find(C_in_0)
                            if COM_C_in == 0:
                                reserve = np.argmax([np.max(C_in[:, h]) for h in range(COM)])
                                C_in_it = np.zeros_like(C)
                                C_in_it[:, reserve] = C_in[:, reserve]
                                C_in_0 = C_in_it

                            COM = com_find(C_in_0)
                            C_in_aa = np.zeros_like(C)
                            C_in_it = peak_trans(C_in_0, C_in_aa)
                            if overlap_keep_active:
                                _, St_it, _, _ = _nnls_reconstruct(chrom_seg, C_in_it)
                                C_ed_it, _ = _compact_profiles_with_indices(C_in_it)
                            else:
                                C_ed_it, St_it = _resolve_ittfa(chrom_seg, C_in_it, COM)

                            chrom_resol_it = np.dot(C_ed_it, St_it)

                            R2_it = 1 - np.var(chrom_resol_it - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                            # ``dishalfremove`` identifies a boundary half profile in
                            # the complete candidate set. Such a profile is a shape
                            # failure, so the complete model must not be restored
                            # solely because its R2 or FRR is larger than the
                            # deletion branch. The reduced candidate remains subject
                            # to the usual ITTFA reconstruction check.
                            full_model_shape_valid = (
                                len(hard_boundary_failure_indices) == 0
                            )

                            # full rank resolutiuon
                            COM = com_find(C_pre)
                            peak_st, peak_ed = peak_find(C_pre)
                            if COM > 1:
                                if (full_model_shape_valid and not component_selection_applied
                                        and R2_it <= 0.99 and COM < 4):
                                    chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(
                                        chrom_seg, C_pre, peak_st, peak_ed, COM,
                                    )
                                    if R2_frr > R2_it:
                                        chrom_resol = chrom_resol_frr
                                        C_ed = C_ed_frr
                                        R2 = R2_frr
                                        St = St_frr
                                        C_p = C_pre
                                        wayname = 'frr'
                                    else:
                                        chrom_resol = chrom_resol_it
                                        C_ed = C_ed_it
                                        R2 = R2_it
                                        St = St_it
                                        C_p = C_in_it

                                elif (not component_selection_applied) and COM > 3 and R2_it < 0.7 and R2_pre < 0.9:
                                    chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(
                                        chrom_seg, C_pre, peak_st, peak_ed, COM,
                                    )
                                    if R2_frr > R2_it:
                                        chrom_resol = chrom_resol_frr
                                        C_ed = C_ed_frr
                                        R2 = R2_frr
                                        St = St_frr
                                        C_p = C_pre
                                        wayname = 'frr'
                                    else:
                                        chrom_resol = chrom_resol_it
                                        C_ed = C_ed_it
                                        R2 = R2_it
                                        St = St_it
                                        C_p = C_in_it

                                else:
                                    chrom_resol = chrom_resol_it
                                    C_ed = C_ed_it
                                    R2 = R2_it
                                    St = St_it
                                    C_p = C_in_it

                                if full_model_shape_valid and R2_pre > R2:
                                    chrom_resol = chrom_resol_pre
                                    C_ed = C_pre
                                    R2 = R2_pre
                                    St = St_pre
                                    C_p = C_pre
                                    wayname = 'non_frr'

                            if COM == 1:
                                chrom_resol = chrom_resol_it
                                C_ed = C_ed_it
                                R2 = R2_it
                                St = St_it
                                C_p = C_in_it
                                if full_model_shape_valid and R2_pre > R2_it:
                                    chrom_resol = chrom_resol_pre
                                    C_ed = C_pre
                                    R2 = R2_pre
                                    St = St_pre
                                    C_p = C_pre

                    if (lapdo == 'Done' and dislapdo == 'None' and dishalfdo == 'None') or (lapdo == 'None' and dislapdo == 'None' and dishalfdo == 'None'):
                        C_pre = np.copy(C_pre_0)
                        CpCp = np.dot(C_pre.T, C_pre)
                        St_pre = fnnls_batch(
                            CpCp, np.dot(C_pre.T, chrom_seg), tole='None'
                        )

                        chrom_resol_pre = np.dot(C_pre, St_pre)
                        R2_pre = 1 - np.var(chrom_resol_pre - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                        # ITTFA
                        COM = com_find(C_pre)
                        if overlap_keep_active:
                            _, St_it, _, _ = _nnls_reconstruct(chrom_seg, C_pre)
                            C_ed_it, _ = _compact_profiles_with_indices(C_pre)
                        else:
                            C_ed_it, St_it = _resolve_ittfa(chrom_seg, C_pre, COM)
                        chrom_resol_it = np.dot(C_ed_it, St_it)
                        R2_it = 1 - np.var(chrom_resol_it - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                        # full rank resolution
                        COM = com_find(C_pre)
                        peak_st, peak_ed = peak_find(C_pre)
                        if COM > 1:
                            if (not component_selection_applied) and R2_it <= 0.99 and COM < 4:
                                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(
                                    chrom_seg, C_pre, peak_st, peak_ed, COM,
                                )
                                if R2_frr > R2_it:
                                    chrom_resol = chrom_resol_frr
                                    C_ed = C_ed_frr
                                    R2 = R2_frr
                                    St = St_frr
                                    C_p = C_pre
                                    wayname = 'frr'
                                else:
                                    chrom_resol = chrom_resol_it
                                    C_ed = C_ed_it
                                    R2 = R2_it
                                    St = St_it
                                    C_p = C_pre

                            elif (not component_selection_applied) and COM > 3 and R2_it < 0.7 and R2_pre < 0.9:
                                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(
                                    chrom_seg, C_pre, peak_st, peak_ed, COM,
                                )
                                if R2_frr > R2_it:
                                    chrom_resol = chrom_resol_frr
                                    C_ed = C_ed_frr
                                    R2 = R2_frr
                                    St = St_frr
                                    C_p = C_pre
                                    wayname = 'frr'
                                else:
                                    chrom_resol = chrom_resol_it
                                    C_ed = C_ed_it
                                    R2 = R2_it
                                    St = St_it
                                    C_p = C_pre

                            else:
                                chrom_resol = chrom_resol_it
                                C_ed = C_ed_it
                                R2 = R2_it
                                St = St_it
                                C_p = C_pre

                            if R2_pre > R2:
                                chrom_resol = chrom_resol_pre
                                C_ed = C_pre
                                R2 = R2_pre
                                St = St_pre
                                C_p = C_pre
                                wayname = 'non_frr'

                        if COM == 1:
                            chrom_resol = chrom_resol_it
                            C_ed = C_ed_it
                            R2 = R2_it
                            St = St_it
                            C_p = C_pre
                            if R2_pre > R2_it:
                                chrom_resol = chrom_resol_pre
                                C_ed = C_pre
                                R2 = R2_pre
                                St = St_pre
                                C_p = C_pre

                    if lapdo == 'None':
                        if dislapdo == 'Done' or dishalfdo == 'Done':
                            # ITTFA,
                            if len(dislapnum) != 0:
                                for k in dislapnum:
                                    C_in_0[:, k] = 0
                            if len(dishalfnum) != 0:
                                for k in dishalfnum:
                                    C_in_0[:, k] = 0

                            COM_C_in = com_find(C_in_0)

                            if COM_C_in == 0:
                                reserve = np.argmax([np.max(C_in[:, h]) for h in range(COM)])
                                C_in_it = np.zeros_like(C)
                                C_in_it[:, reserve] = C_in[:, reserve]
                                C_in_0 = C_in_it

                            COM = com_find(C_in_0)
                            C_in_aa = np.zeros_like(C)
                            C_in_it_dis = peak_trans(C_in_0, C_in_aa)

                            if overlap_keep_active:
                                _, St_it_dis, _, _ = _nnls_reconstruct(chrom_seg, C_in_it_dis)
                                C_ed_it_dis, _ = _compact_profiles_with_indices(C_in_it_dis)
                            else:
                                C_ed_it_dis, St_it_dis = _resolve_ittfa(chrom_seg, C_in_it_dis, COM)

                            chrom_resol_it_dis = np.dot(C_ed_it_dis, St_it_dis)

                            R2_it_dis = 1 - np.var(chrom_resol_it_dis - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                            # ITTFA_1
                            COM = com_find(C_pre_0)
                            C_in_it = C_pre_0
                            if overlap_keep_active:
                                _, St_it, _, _ = _nnls_reconstruct(chrom_seg, C_in_it)
                                C_ed_it, _ = _compact_profiles_with_indices(C_in_it)
                            else:
                                C_ed_it, St_it = _resolve_ittfa(chrom_seg, C_in_it, COM)
                            chrom_resol_it = np.dot(C_ed_it, St_it)
                            R2_it = 1 - np.var(chrom_resol_it - chrom_seg, ddof=1) / np.var(chrom_seg, ddof=1)

                            # A boundary-half profile is an explicit shape failure,
                            # not merely a lower-ranked model. In this branch
                            # C_pre_0 still represents the complete pre-deletion
                            # model, so allowing its higher R2 or FRR result to win
                            # would restore the profile rejected by dishalfremove.
                            # Keep the deletion branch eligible, but remove the full
                            # model from the competition for this segment.
                            full_model_shape_valid = (
                                len(hard_boundary_failure_indices) == 0
                            )

                            if R2_it < R2_it_dis:
                                chrom_resol = chrom_resol_it_dis
                                C_ed_it = C_ed_it_dis
                                St_it = St_it_dis
                                R2_it = R2_it_dis
                                C_in_it = C_in_it_dis
                            elif not full_model_shape_valid:
                                chrom_resol = chrom_resol_it_dis
                                C_ed_it = C_ed_it_dis
                                St_it = St_it_dis
                                R2_it = R2_it_dis
                                C_in_it = C_in_it_dis

                            # initial resolution
                            C_pre = np.copy(C_pre_0)
                            CpCp = np.dot(C_pre.T, C_pre)
                            St_pre = fnnls_batch(
                                CpCp, np.dot(C_pre.T, chrom_seg), tole='None'
                            )

                            chrom_resol_pre = np.dot(C_pre, St_pre)
                            R2_pre = 1 - np.var(chrom_resol_pre - chrom_seg, ddof=1) / np.var(chrom_seg, ddof=1)

                            # full rank resolution
                            COM = com_find(C_pre)
                            peak_st, peak_ed = peak_find(C_pre)
                            if COM > 1:
                                if (full_model_shape_valid and not component_selection_applied
                                        and R2_it <= 0.99 and COM < 4):
                                    chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(
                                        chrom_seg, C_pre, peak_st, peak_ed, COM,
                                    )
                                    if R2_frr > R2_it:
                                        chrom_resol = chrom_resol_frr
                                        C_ed = C_ed_frr
                                        R2 = R2_frr
                                        St = St_frr
                                        C_p = C_pre
                                        wayname = 'frr'
                                    else:
                                        chrom_resol = chrom_resol_it
                                        C_ed = C_ed_it
                                        R2 = R2_it
                                        St = St_it
                                        C_p = C_in_it

                                else:
                                    chrom_resol = chrom_resol_it
                                    C_ed = C_ed_it
                                    R2 = R2_it
                                    St = St_it
                                    C_p = C_in_it

                                if full_model_shape_valid and R2_pre > R2:
                                    chrom_resol = chrom_resol_pre
                                    C_ed = C_pre
                                    R2 = R2_pre
                                    St = St_pre
                                    C_p = C_pre
                                    wayname = 'non_frr'

                            if COM == 1:
                                chrom_resol = chrom_resol_it
                                C_ed = C_ed_it
                                R2 = R2_it
                                St = St_it
                                C_p = C_in_it
                                if full_model_shape_valid and R2_pre > R2_it:
                                    chrom_resol = chrom_resol_pre
                                    C_ed = C_pre
                                    R2 = R2_pre
                                    St = St_pre
                                    C_p = C_pre

                    post_selection_frr = None
                    if component_selection_applied:
                        C_ed = np.copy(component_selection_result['chromatograms'])
                        St = np.copy(component_selection_result['spectra'])
                        chrom_resol = np.dot(C_ed, St)
                        selected_r2 = _reconstruction_r2(chrom_seg, chrom_resol)
                        R2 = selected_r2
                        C_p = np.copy(C_ed)
                        R2_pre = selected_r2
                        R2_it = selected_r2
                        wayname = 'component_selection'

                        # Component selection has already decided the number of
                        # components.  Apply the historical final R2 rule only to
                        # the fixed selected model: when its R2 is below 0.99, run
                        # FRR and retain the reconstruction with the larger R2.
                        # This deliberately cannot restore or delete a component.
                        selected_indices = tuple(
                            int(index) for index in component_selection_result.get(
                                'record', {}
                            ).get('selected_indices', ())
                        )
                        component_selection_r2 = float(selected_r2)
                        frr_source = None
                        if (selected_indices
                                and component_selection_pre_profiles.ndim == 2
                                and max(selected_indices)
                                < component_selection_pre_profiles.shape[1]):
                            frr_source = component_selection_pre_profiles[
                                :, list(selected_indices)
                            ]
                        if frr_source is None:
                            frr_source = np.copy(C_ed)
                        frr_source, _ = _compact_profiles_with_indices(frr_source)
                        selected_count = int(com_find(frr_source))

                        # The component selector determines only the component
                        # number.  Once that number is fixed, restore the historical
                        # resolution order by running standard ITTFA from the
                        # selected pre-ITTFA profiles.  The MCR fit above remains
                        # available as a diagnostic R2, but it is not used as the
                        # final chromatographic profile.
                        if 0 < selected_count < 6:
                            try:
                                ittfa_C, ittfa_St = _resolve_ittfa(
                                    chrom_seg, frr_source, selected_count
                                )
                                ittfa_reconstruction = np.dot(ittfa_C, ittfa_St)
                                ittfa_r2 = _reconstruction_r2(
                                    chrom_seg, ittfa_reconstruction
                                )
                                if (ittfa_C is not None and ittfa_St is not None
                                        and np.isfinite(float(ittfa_r2))):
                                    C_ed = np.copy(ittfa_C)
                                    St = np.copy(ittfa_St)
                                    chrom_resol = np.copy(ittfa_reconstruction)
                                    selected_r2 = float(ittfa_r2)
                                    R2 = float(ittfa_r2)
                                    C_p = np.copy(frr_source)
                                    wayname = 'ittfa'
                            except Exception:
                                # Keep the component-selection MCR result as a
                                # fallback if the fixed-count ITTFA refit fails.
                                pass
                        pre_reconstruction, pre_St, pre_r2, _ = _nnls_reconstruct(
                            chrom_seg, frr_source
                        )
                        post_selection_frr = {
                            'attempted': False,
                            'selected_component_count': selected_count,
                            'r2_pre_ittfa': float(pre_r2),
                            'r2_before': float(selected_r2),
                            'r2_component_selection': component_selection_r2,
                            'r2_frr_trigger': float(min(
                                selected_r2, pre_r2
                            )),
                            'r2_frr': None,
                            'used': False,
                            'final_path': 'component_selection',
                            'reason': 'r2_at_or_above_threshold',
                        }
                        frr_reconstruction = None
                        frr_C = None
                        frr_St = None
                        frr_r2 = None
                        # Preserve the manuscript rule after the component count has
                        # been fixed.  Component selection can produce an MCR R2
                        # just above 0.99 even when the direct pre-ITTFA NNLS fit is
                        # below the threshold.  In that case FRR must still be
                        # evaluated so the final profile is not silently left at the
                        # component-selection rotation.
                        frr_trigger_r2 = min(float(selected_r2), float(pre_r2))
                        if frr_trigger_r2 < 0.99 and 1 < selected_count < 4:
                            post_selection_frr['attempted'] = True
                            peak_st_frr, peak_ed_frr = peak_find(frr_source)
                            if (len(peak_st_frr) == selected_count
                                    and len(peak_ed_frr) == selected_count):
                                try:
                                    (frr_reconstruction, frr_C, frr_r2, frr_St) = (
                                        dynamic_FRR(
                                            chrom_seg,
                                            frr_source,
                                            peak_st_frr,
                                            peak_ed_frr,
                                            selected_count,
                                        )
                                    )
                                    if frr_r2 is not None and np.isfinite(float(frr_r2)):
                                        post_selection_frr['r2_frr'] = float(frr_r2)
                                    else:
                                        frr_r2 = None
                                        post_selection_frr['reason'] = (
                                            'frr_returned_non_finite_r2'
                                        )
                                except Exception as exc:
                                    post_selection_frr['reason'] = (
                                        'frr_failed: {}: {}'.format(
                                            type(exc).__name__, str(exc)
                                        )
                                    )
                            else:
                                post_selection_frr['reason'] = (
                                    'frr_peak_boundaries_not_found'
                                )
                        elif frr_trigger_r2 < 0.99:
                            post_selection_frr['reason'] = (
                                'frr_supported_only_for_two_or_three_components'
                            )

                        # Reproduce the manuscript ordering exactly after the
                        # component count has been fixed.  First compare FRR with the
                        # selected ITTFA/MCR result, then compare the non-ITTFA NNLS
                        # result against the current best.  None of these comparisons
                        # is allowed to change the selected component count.
                        best_r2 = float(selected_r2)
                        if np.isfinite(float(pre_r2)) and float(pre_r2) > best_r2:
                            chrom_resol = pre_reconstruction
                            C_ed = np.copy(frr_source)
                            R2 = float(pre_r2)
                            St = pre_St
                            C_p = np.copy(frr_source)
                            wayname = 'non_frr'
                            best_r2 = float(pre_r2)
                            post_selection_frr['final_path'] = 'non_frr'
                        if (frr_r2 is not None and frr_reconstruction is not None
                                and frr_C is not None and frr_St is not None
                                and float(frr_r2) > best_r2):
                            chrom_resol = frr_reconstruction
                            C_ed = frr_C
                            R2 = float(frr_r2)
                            St = frr_St
                            C_p = np.copy(frr_source)
                            wayname = 'frr'
                            best_r2 = float(frr_r2)
                            post_selection_frr['used'] = True
                            post_selection_frr['final_path'] = 'frr'
                            post_selection_frr['reason'] = (
                                'frr_or_non_ittfa_result_higher_than_selected_model'
                            )
                        elif post_selection_frr['final_path'] == 'component_selection':
                            if float(pre_r2) > float(selected_r2):
                                post_selection_frr['reason'] = (
                                    'non_ittfa_r2_higher_than_selected_model'
                                )
                            elif frr_r2 is not None:
                                post_selection_frr['reason'] = (
                                    'selected_model_r2_higher_than_pre_ittfa_and_frr'
                                )
                    # A deletion followed by a new ITTFA rotation can create a
                    # boundary half profile even when every profile in the first
                    # ITTFA result was acceptable.  In apply mode, validate the
                    # actually selected output after all R2 and FRR arbitration and
                    # repeat the reduced ITTFA fit until no further hard shape
                    # failure is found.  Overlap protection intentionally does not
                    # apply here: it prevents similarity-only deletion, not rejection
                    # of an individual profile that is not a complete peak.
                    post_resolution_shape_cleanup = []
                    cleanup_limit = max(int(com_find(C_ed)) - 1, 0)
                    for cleanup_pass in range(cleanup_limit):
                        resolved_compact, _ = _compact_profiles_with_indices(C_ed)
                        source_compact, _ = _compact_profiles_with_indices(C_p)
                        if (resolved_compact.shape[1] <= 1
                                or source_compact.shape[1]
                                != resolved_compact.shape[1]):
                            break
                        shape_evidence = _resolved_profile_shape_test(
                            resolved_compact, component_selection_config
                        )
                        invalid_indices = [
                            int(item['component_index'])
                            for item in shape_evidence.get('components', [])
                            if not item.get('complete_peak', False)
                        ]
                        cleanup_record = {
                            'pass': int(cleanup_pass + 1),
                            'component_count_before': int(
                                resolved_compact.shape[1]
                            ),
                            'shape_evidence': shape_evidence,
                            'removed_indices': invalid_indices,
                        }
                        if not invalid_indices:
                            cleanup_record['component_count_after'] = int(
                                resolved_compact.shape[1]
                            )
                            post_resolution_shape_cleanup.append(cleanup_record)
                            break

                        keep_indices = [
                            index for index in range(source_compact.shape[1])
                            if index not in set(invalid_indices)
                        ]
                        if not keep_indices:
                            # A GC MS segment must retain at least one component.
                            # Prefer the profile with the smallest boundary fraction
                            # rather than restoring the model with the largest R2.
                            components = shape_evidence.get('components', [])
                            reserve = min(
                                range(len(components)),
                                key=lambda index: max(
                                    float(components[index].get(
                                        'left_boundary_fraction') or 0.0
                                    ),
                                    float(components[index].get(
                                        'right_boundary_fraction') or 0.0
                                    ),
                                ),
                            )
                            keep_indices = [int(reserve)]
                        C_p = np.asarray(
                            source_compact[:, keep_indices], dtype=float32
                        )
                        C_ed, St = _resolve_ittfa(
                            chrom_seg, C_p, C_p.shape[1]
                        )
                        chrom_resol = np.dot(C_ed, St)
                        R2 = _reconstruction_r2(chrom_seg, chrom_resol)
                        R2_pre = R2
                        R2_it = R2
                        wayname = 'post_resolution_shape_cleanup'
                        cleanup_record['component_count_after'] = int(C_p.shape[1])
                        cleanup_record['r2_after'] = float(R2)
                        post_resolution_shape_cleanup.append(cleanup_record)

                    COM = com_find(C_ed)

                    if wayname == 'frr':
                        tic_single = C_ed
                    else:
                        # Each component TIC is its chromatographic profile scaled by
                        # the sum of its nonnegative mass spectrum.  This is exactly
                        # the row-wise sum of the outer product, without a Python
                        # loop over scans or an intermediate matrix per component.
                        tic_single = np.asarray(C_ed) * np.sum(
                            np.asarray(St), axis=1
                        )[None, :]

                    if generate_image is True:
                        C_tic = np.copy(tic_single)
                        C_tic = C_tic / np.max(C_tic)
                        figure = plt.figure(clear=True)
                        figure.suptitle('num=' + str(i))
                        plt.subplot(2, 2, 1)
                        plt.plot(chrom_seg)
                        plt.title('original GCMS data')
                        plt.subplot(2, 2, 2)
                        plt.plot(chrom_resol)
                        plt.title('resolved GCMS data')
                        plt.subplot(2, 2, 3)
                        plt.plot(C_p)
                        plt.title('predictive chromatographic profile')
                        plt.subplot(2, 2, 4)
                        plt.plot(C_tic)
                        plt.title('resolved chromatogram')
                        plt.tight_layout()
                        # Save the two requested output formats directly.  Writing a
                        # TIFF and converting it to PNG after the run adds a full
                        # extra image-processing pass without improving the figures.
                        for image_format in image_formats:
                            figure.savefig(os.path.join(
                                figure_savepath, f'num={i}.{image_format}'
                            ))
                        plt.close()
                        plt.cla()
                        plt.clf()

                    tic = np.sum(chrom_resol, axis=1).astype(float32, copy=False)
                    peak_area = integrate.trapz(tic)

                    gap = (np.max(RT) - np.min(RT)) / (Xtest.shape[0]-1)
                    rt_st = np.min(RT) + ind_st_DeepSeg[i]*gap
                    rt_ed = np.min(RT) + ind_en_DeepSeg[i]*gap

                    tol_com.append(COM)
                    sum_com = np.sum(tol_com) - tol_com[-1]

                    COM = com_find(C_ed)
                    for r in range(COM):
                        num_com = sum_com + (r+1)
                        tic_single_area = tic_single[:, r].flatten()
                        peak_area_single = integrate.trapz(tic_single_area)
                        rt = np.min(RT) + (ind_st_DeepSeg[i] + np.argmax(C_ed[:, r]))*gap
                        St_single = St[r, :]

                        signal_windows = TICsum_origin[int(ind_st_DeepSeg[i]+np.argmax(C_ed[:, r]) - 3):int(ind_st_DeepSeg[i] + np.argmax(C_ed[:, r]) + 3)]
                        scalemax = 50
                        scales = np.arange(1, scalemax)
                        coefficients, frequencies = pywt.cwt(signal_windows, scales, 'mexh')
                        max_coefficient_index = np.unravel_index(np.argmax(np.abs(coefficients)), coefficients.shape)
                        max_scale = scales[max_coefficient_index[0]]
                        while max_scale == max(scales):
                            scalemax += 50
                            scales = np.arange(1, scalemax)
                            coefficients, frequencies = pywt.cwt(signal_windows, scales, 'mexh')
                            max_coefficient_index = np.unravel_index(np.argmax(np.abs(coefficients)), coefficients.shape)
                            max_scale = scales[max_coefficient_index[0]]

                        noise_coefficients, noise_frequencies = pywt.cwt(noise_windows, 1, 'mexh')
                        noise_intensity = np.percentile(np.abs(noise_coefficients), 95)
                        snr = round(np.max(np.abs(coefficients))/noise_intensity, 2)

                        ms_single.append({'ms': St_single})
                        peak_excel_single.append({'#number': num_com, 'rt': round(rt,5), 'peak area': peak_area_single, 'COM': COM, 'R2': R2, 'SNR': snr})

                    peak_excel_seg.append({'num': i+1, 'RT_st': round(rt_st, 5), 'RT_ed': round(rt_ed, 5), 'R2': round(R2, 5), 'PA': peak_area})

            return {
                "segment_index": int(i),
                "peak_excel_single": peak_excel_single,
                "peak_excel_seg": peak_excel_seg,
                "ms_single": ms_single,
            }

        return _run()

    segment_indices = list(range(len(ind_st_DeepSeg)))
    use_parallel = bool(
        parallel_segments and len(segment_indices) > 1 and Parallel is not None
    )
    if use_parallel:
        worker_count = min(8, len(segment_indices))
        with parallel_config(
                backend="loky", n_jobs=worker_count,
                inner_max_num_threads=1):
            segment_results = Parallel()(delayed(_resolve_segment_job)(
                index
            ) for index in segment_indices)
    else:
        segment_results = [
            _resolve_segment_job(index)
            for index in tqdm(
                segment_indices, desc="serial resolving processing"
            )
        ]

    peak_excel_single = []
    peak_excel_seg = []
    ms_single = []
    for result in sorted(
            segment_results, key=lambda item: item["segment_index"]):
        for peak_record in result["peak_excel_single"]:
            peak_record["#number"] = len(peak_excel_single) + 1
            peak_excel_single.append(peak_record)
        peak_excel_seg.extend(result["peak_excel_seg"])
        ms_single.extend(result["ms_single"])

    return peak_excel_single, peak_excel_seg, ms_single

def DeepCPR(work_path, modelpath, filename, figure_savepath, dist, thres,
            generate_image,
            image_formats=("png",)):
    """Resolve chromatographic segments with the production executor."""
    return _deepcpr_impl(
        work_path, modelpath, filename, figure_savepath, dist, thres,
        generate_image,
        parallel_segments=True,
        image_formats=image_formats,
    )


def DeepCPRSerial(work_path, modelpath, filename, figure_savepath, dist, thres,
                  generate_image,
                  image_formats=("png",)):
    """Run the same resolver serially as a strict comparison baseline."""
    return _deepcpr_impl(
        work_path, modelpath, filename, figure_savepath, dist, thres,
        generate_image,
        parallel_segments=False,
        image_formats=image_formats,
    )


def data_resolution(dataset_path, DeepCS_path, DeepCPR_path, save_path,
                    generate_image, adaptive=False, adaptive_kwargs=None,
                    image_formats=("png",)):
    """
    Function that stores the resolution result of DeepCPR.
    """

    image_formats = (
        _normalize_image_formats(image_formats) if generate_image else ("png",)
    )

    px_savepath1 = save_path + '/single'
    px_savepath2 = save_path + '/seg'
    ms_savepath = save_path + '/ms'

    if not os.path.exists(px_savepath1):
        os.makedirs(px_savepath1)
    if not os.path.exists(px_savepath2):
        os.makedirs(px_savepath2)
    if not os.path.exists(ms_savepath):
        os.makedirs(ms_savepath)

    files = os.listdir(dataset_path)
    for file in files:
        filename = os.path.join(dataset_path, file)
        file_pre = file.split('.')[0]

        if generate_image is True:
            figure_savepath = save_path + '/figure/' + file_pre
            if not os.path.exists(figure_savepath):
                os.makedirs(figure_savepath)
        else:
            figure_savepath = None

        print('file loading:', file, flush=True)

        if adaptive:
            peak_excel_single, peak_excel_seg, ms_single = DeepCPRAdaptive(
                DeepCS_path,
                DeepCPR_path,
                filename,
                figure_savepath,
                dist=3,
                thres=15,
                generate_image=generate_image,
                image_formats=image_formats,
                **(adaptive_kwargs or {}),
            )
        else:
            peak_excel_single, peak_excel_seg, ms_single = DeepCPR(
                DeepCS_path,
                DeepCPR_path,
                filename,
                figure_savepath,
                dist=3,
                thres=15,
                generate_image=generate_image,
                image_formats=image_formats,
            )
        pe_single = pd.DataFrame(peak_excel_single)
        pe_single.to_csv(px_savepath1 + '/' + file_pre + '.csv', index=False)
        pe_seg = pd.DataFrame(peak_excel_seg)
        pe_seg.to_csv(px_savepath2 + '/' + file_pre + '.csv', index=False)

        if not os.path.exists(ms_savepath + '/' + file_pre):
            os.makedirs(ms_savepath + '/' + file_pre)
        for i in range(len(ms_single)):
            ms_single_com = ms_single[i]['ms']
            RT = peak_excel_single[i]['rt']
            mz_values = np.arange(1, ms_single_com.shape[0]+1)
            intensity_values = ms_single_com
            save_as_msp(ms_savepath + '/' + file_pre + '/' + str(i) + '.msp', RT, mz_values, intensity_values)

        del peak_excel_single, peak_excel_seg, ms_single, pe_single, pe_seg
        gc.collect()
        print('', flush=True)

def DeepCPRAdaptive(work_path, modelpath, filename, figure_savepath=None,
                    dist=5, thres=5, generate_image=False,
                    max_iterations=4, max_components=32,
                    min_component_fraction=0.002,
                    residual_tolerance=0.05,
                    min_improvement=0.01,
                    enable_peak_stop=True,
                    min_residual_peak_prominence=2.0,
                    enable_flat_tail_taper=True,
                    tail_window=12,
                    tail_level=0.05,
                     max_support=80,
                     max_fwhm_factor=2.5,
                     remove_embedded_peaks=False,
                     image_formats=("png",)):
    """Adaptive extension of :func:`DeepCPR` for more than five components.

    The original network still predicts at most five profiles per forward
    pass.  This function repeatedly applies it to the positive residual of a
    local segment, estimates spectra with ``ITTFA_PRO``, and jointly refits all
    retained profiles.  Consequently, the number of returned components is
    data-dependent rather than fixed at five.

    The return value has the same three objects as ``DeepCPR``:
    ``peak_excel_single``, ``peak_excel_seg`` and ``ms_single``.  Existing
    ``data_resolution`` users can call this function in a separate wrapper
    while retaining the original five-output implementation unchanged.
    """
    image_formats = (
        _normalize_image_formats(image_formats) if generate_image else ("png",)
    )
    # The adaptive resolver now lives under ``more_components``.
    from .more_components.adaptive_v2 import adaptive_resolve_segment_v2

    mat, RT, Xtest, ind_st_DeepSeg, ind_en_DeepSeg = data_process(
        work_path, filename, dist, thres
    )
    TICsum_origin = np.asarray([np.sum(Xtest[i, :]) for i in range(Xtest.shape[0])])
    mz_min = math.floor(judge_min(min(mat['mz'])))
    mz_max = math.ceil(max(mat['mz']))
    if mz_min < 0:
        Xtest = Xtest[:, 1:]
        mz_min = 0
        mz_max = Xtest.shape[1]
    if mz_max > 800:
        raise ValueError('m/z axis of this file exceeds the '
                         '800-bin input window of the DeepCPR model')

    restored_model = _load_cached_model(modelpath)

    peak_excel_single = []
    peak_excel_seg = []
    ms_single = []
    gap = (np.max(RT) - np.min(RT)) / max(Xtest.shape[0] - 1, 1)

    for i in tqdm(range(len(ind_st_DeepSeg)), desc='adaptive resolving processing'):
        start = int(ind_st_DeepSeg[i])
        end = int(ind_en_DeepSeg[i])
        length = end - start
        if length <= 0:
            continue
        if length > 128:
            raise ValueError(
                f'Local segment {i} contains {length} scans; the trained model accepts at most 128.'
            )

        test = np.zeros((length, 800), dtype=float32)
        test[:, mz_min:mz_max] = Xtest[start:end, :]
        test_br = airPLS_correct_matrix(test, lambda_=500, itermax=15)
        if not np.any(test_br):
            continue

        result = adaptive_resolve_segment_v2(
            test_br,
            restored_model,
            max_iterations=max_iterations,
            max_components=max_components,
            min_component_fraction=min_component_fraction,
            residual_tolerance=residual_tolerance,
            min_improvement=min_improvement,
            enable_peak_stop=enable_peak_stop,
            min_residual_peak_prominence=min_residual_peak_prominence,
            enable_flat_tail_taper=enable_flat_tail_taper,
            tail_window=tail_window,
            tail_level=tail_level,
            max_support=max_support,
            max_fwhm_factor=max_fwhm_factor,
            remove_embedded_peaks=remove_embedded_peaks,
        )
        COM = result.n_components
        if COM == 0:
            continue

        C_ed = result.chromatograms
        St = result.spectra
        chrom_resol = result.reconstruction
        R2 = result.r2
        # TIC contribution of each resolved component.
        tic_single = C_ed * np.sum(St, axis=1, keepdims=True).T
        tic = np.sum(chrom_resol, axis=1)
        peak_area = float(np.trapz(tic))

        rt_st = np.min(RT) + start * gap
        rt_ed = np.min(RT) + end * gap
        peak_excel_seg.append({
            'num': i + 1,
            'RT_st': round(float(rt_st), 5),
            'RT_ed': round(float(rt_ed), 5),
            'R2': round(float(R2), 5),
            'PA': peak_area,
            'COM': COM,
            'iterations': result.iterations,
        })

        # Use the neighboring TIC values as a simple, robust noise reference.
        left = TICsum_origin[max(0, start - 2):start]
        right = TICsum_origin[end:min(Xtest.shape[0], end + 2)]
        noise_scale = float(np.std(np.concatenate((left, right)))) if (left.size + right.size) else 1.0
        noise_scale = max(noise_scale, 1e-12)

        for r in range(COM):
            apex = int(np.argmax(C_ed[:, r]))
            rt = np.min(RT) + (start + apex) * gap
            component_tic = tic_single[:, r]
            snr = round(float(np.max(component_tic) / noise_scale), 2)
            ms_single.append({'ms': St[r, :]})
            peak_excel_single.append({
                '#number': len(peak_excel_single) + 1,
                'rt': round(float(rt), 5),
                'peak area': float(np.trapz(component_tic)),
                'COM': COM,
                'R2': float(R2),
                'SNR': snr,
                'adaptive_iteration_count': result.iterations,
            })

        if generate_image and figure_savepath is not None:
            os.makedirs(figure_savepath, exist_ok=True)
            figure = plt.figure(clear=True, figsize=(10, 7))
            plt.subplot(2, 2, 1)
            plt.plot(test_br)
            plt.title('original GC-MS data')
            plt.subplot(2, 2, 2)
            plt.plot(chrom_resol)
            plt.title(f'adaptive reconstruction (K={COM})')
            plt.subplot(2, 2, 3)
            plt.plot(C_ed)
            plt.title('resolved chromatographic profiles')
            plt.subplot(2, 2, 4)
            plt.plot(tic_single)
            plt.title('resolved TIC components')
            plt.tight_layout()
            for image_format in image_formats:
                figure.savefig(os.path.join(
                    figure_savepath, f'num={i}.{image_format}'
                ))
            plt.close()

    return peak_excel_single, peak_excel_seg, ms_single

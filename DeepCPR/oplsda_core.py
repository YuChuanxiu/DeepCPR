"""Core OPLS DA and VIP4t calculations for the reproducibility analysis.

This module is used by q2_nested_cv.py. It deliberately has no independent
analysis entry point, so the public analysis can be reproduced through one
script and one documented command.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPS = np.finfo(float).eps


@dataclass
class OPLSModel:
    mean: np.ndarray
    std: np.ndarray
    wo: np.ndarray
    po: np.ndarray
    w: np.ndarray
    c: float
    vip4t: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predictive_component(x: np.ndarray, y: np.ndarray):
    denom = float(y @ y)
    if denom <= EPS:
        raise ValueError("Cannot fit OPLS with a constant response.")
    w = (x.T @ y) / denom
    norm = float(np.linalg.norm(w))
    if norm <= EPS:
        raise ValueError("Predictive OPLS weight has zero norm.")
    w = w / norm
    t = x @ w
    t_denom = float(t @ t)
    if t_denom <= EPS:
        raise ValueError("Predictive OPLS score has zero variance.")
    c = float((y @ t) / t_denom)
    p = (x.T @ t) / t_denom
    return w, t, c, p


def fit_opls(
    x: np.ndarray,
    y: np.ndarray,
    n_orthogonal: int,
    scaling_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> OPLSModel:
    """Fit OPLS DA after calculating scaling from the supplied calibration rows."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if scaling_override is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=1)
    else:
        mean, std = scaling_override
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
    std = std.copy()
    std[~np.isfinite(std) | (std == 0.0)] = 1.0
    x_work = (x - mean) / std

    wo_list = []
    po_list = []
    to_list = []
    co_list = []
    for _ in range(n_orthogonal):
        w_i, _t_i, _c_i, p_i = _predictive_component(x_work, y)
        wo_i = p_i - (float(w_i @ p_i) / float(w_i @ w_i)) * w_i
        norm = float(np.linalg.norm(wo_i))
        if norm <= EPS:
            break
        wo_i = wo_i / norm
        to_i = x_work @ wo_i
        denom = float(to_i @ to_i)
        if denom <= EPS:
            break
        po_i = (x_work.T @ to_i) / denom
        co_i = float((y @ to_i) / denom)
        wo_list.append(wo_i)
        po_list.append(po_i)
        to_list.append(to_i)
        co_list.append(co_i)
        x_work = x_work - np.outer(to_i, po_i)

    w, t, c, p = _predictive_component(x_work, y)
    wo = np.column_stack(wo_list) if wo_list else np.empty((x.shape[1], 0))
    po = np.column_stack(po_list) if po_list else np.empty((x.shape[1], 0))
    vip4t = calculate_vip4t(p, t, c, po, to_list, co_list)
    return OPLSModel(mean, std, wo, po, w, c, vip4t)


def calculate_vip4t(
    p: np.ndarray,
    t: np.ndarray,
    c: float,
    po: np.ndarray,
    to_list: list[np.ndarray],
    co_list: list[float],
) -> np.ndarray:
    """Calculate total VIP4t from predictive and orthogonal terms."""
    n_variables = p.size
    sxp = float(np.sum(np.outer(t, p) ** 2))
    syp = float(np.sum((t * c) ** 2))
    sxo = np.array(
        [float(np.sum(np.outer(to_i, po[:, i]) ** 2)) for i, to_i in enumerate(to_list)]
    )
    syo = np.array(
        [float(np.sum((to_i * co_list[i]) ** 2)) for i, to_i in enumerate(to_list)]
    )
    ssx = sxp + float(sxo.sum())
    ssy = syp + float(syo.sum())
    if ssx <= EPS or ssy <= EPS:
        return np.full(n_variables, np.nan)

    p_norm = float(np.linalg.norm(p))
    form_2 = ((p / p_norm) ** 2) * sxp / ssx
    form_4 = ((p / p_norm) ** 2) * syp / ssy
    form_1 = np.zeros(n_variables)
    form_3 = np.zeros(n_variables)
    for i in range(po.shape[1]):
        po_norm = float(np.linalg.norm(po[:, i]))
        if po_norm > EPS:
            form_1 += ((po[:, i] / po_norm) ** 2) * sxo[i] / ssx
            form_3 += ((po[:, i] / po_norm) ** 2) * syo[i] / ssy
    return np.sqrt(n_variables / 2.0 * (form_1 + form_2 + form_3 + form_4))


def predict_opls(model: OPLSModel, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_work = (np.asarray(x, dtype=float) - model.mean) / model.std
    for i in range(model.wo.shape[1]):
        to_i = x_work @ model.wo[:, i]
        x_work = x_work - np.outer(to_i, model.po[:, i])
    score = x_work @ model.w
    y_hat = score * model.c
    label = np.where(y_hat >= 0.0, 1.0, -1.0)
    return y_hat, label


def stratified_splits(y: np.ndarray, n_splits: int, rng: np.random.Generator):
    """Yield shuffled stratified calibration and validation index sets."""
    per_class = []
    for value in np.unique(y):
        indices = np.flatnonzero(y == value).copy()
        rng.shuffle(indices)
        per_class.append(np.array_split(indices, n_splits))
    all_indices = np.arange(y.size)
    for fold in range(n_splits):
        test = np.sort(np.concatenate([parts[fold] for parts in per_class]))
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        yield train, test


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    """Return sensitivity, specificity, accuracy and balanced accuracy."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    tp = int(np.sum((y == -1) & (pred == -1)))
    fn = int(np.sum((y == -1) & (pred == 1)))
    tn = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 1) & (pred == -1)))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / y.size
    balanced = (sensitivity + specificity) / 2.0
    return {
        "n": int(y.size),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
    }


def summarize_distribution(frame: pd.DataFrame, column: str):
    values = frame[column].to_numpy(dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "p2_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
        "min": float(values.min()),
        "max": float(values.max()),
    }

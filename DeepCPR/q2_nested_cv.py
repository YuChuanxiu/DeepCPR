"""Q2 based leakage controlled OPLS DA reanalysis for the clinical Table S11 matrix.

The script reuses the OPLS implementation in oplsda_core.py, but selects the
number of orthogonal components from Q2 calculated only within each outer
calibration sample. The outer validation sample is used only for prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oplsda_core import (  # noqa: E402
    EPS,
    fit_opls,
    metrics,
    predict_opls,
    sha256,
    stratified_splits,
    summarize_distribution,
)


def q2_for_components(
    x: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_orthogonal: int,
    scaling_override: tuple[np.ndarray, np.ndarray],
) -> float:
    """Calculate Q2 inside an outer calibration sample."""
    press = 0.0
    for train, test in splits:
        model = fit_opls(
            x[train], y[train], n_orthogonal, scaling_override=scaling_override
        )
        score, _pred = predict_opls(model, x[test])
        press += float(np.sum((y[test] - score) ** 2))
    denominator = float(np.sum(y**2))
    if denominator <= EPS:
        return float("nan")
    return float(1.0 - press / denominator)


def select_components_by_q2(
    x: np.ndarray,
    y: np.ndarray,
    candidates: list[int],
    rng: np.random.Generator,
    scaling_override: tuple[np.ndarray, np.ndarray],
    inner_folds: int = 10,
    q2_threshold: float = 0.01,
) -> tuple[int, dict[int, float], dict[int, float]]:
    """Select the largest component count supported by incremental Q2.

    The original implementation uses Q2 as a stopping criterion rather than
    selecting the numerical maximum. Here a component is retained when its
    incremental Q2 is at least 0.01. Selection is performed only in the outer
    calibration samples.
    """
    splits = list(stratified_splits(y, inner_folds, rng))
    q2_values = {
        int(k): q2_for_components(
            x, y, splits, int(k), scaling_override=scaling_override
        )
        for k in candidates
    }
    increments: dict[int, float] = {}
    previous = q2_values[candidates[0]]
    increments[int(candidates[0])] = float(previous)
    selected = int(candidates[0])
    for k in candidates[1:]:
        current = q2_values[int(k)]
        increment = float(current - previous)
        increments[int(k)] = increment
        if np.isfinite(increment) and increment >= q2_threshold:
            selected = int(k)
            previous = current
        else:
            break
    return selected, q2_values, increments


def repeated_q2_cv(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    repeats: int,
    seed: int,
    max_components: int,
    inner_folds: int,
    q2_threshold: float,
):
    candidates = list(range(0, min(max_components, x.shape[1] - 1) + 1))
    prediction_rows: list[dict] = []
    repeat_metrics: list[dict] = []
    component_rows: list[dict] = []
    vip_hits = np.zeros(x.shape[1], dtype=int)
    vip_sum = np.zeros(x.shape[1], dtype=float)
    model_count = 0
    selected_counts = {k: 0 for k in candidates}

    for repeat in range(repeats):
        outer_rng = np.random.default_rng(seed + repeat * 1009)
        outer_splits = list(stratified_splits(y, 10, outer_rng))
        pred = np.empty(y.size)
        score = np.empty(y.size)

        for fold, (train, test) in enumerate(outer_splits):
            inner_rng = np.random.default_rng(seed + repeat * 1009 + fold + 1)
            outer_mean = x[train].mean(axis=0)
            outer_std = x[train].std(axis=0, ddof=1)
            chosen, q2_values, q2_increments = select_components_by_q2(
                x[train],
                y[train],
                candidates,
                inner_rng,
                scaling_override=(outer_mean, outer_std),
                inner_folds=inner_folds,
                q2_threshold=q2_threshold,
            )
            selected_counts[chosen] += 1
            component_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "selected_orthogonal_components": chosen,
                    "q2_by_components": json.dumps(q2_values, sort_keys=True),
                    "incremental_q2_by_components": json.dumps(
                        q2_increments, sort_keys=True
                    ),
                }
            )

            model = fit_opls(x[train], y[train], chosen)
            score[test], pred[test] = predict_opls(model, x[test])
            vip_hits += np.nan_to_num(model.vip4t >= 1.0, nan=False).astype(int)
            vip_sum += np.nan_to_num(model.vip4t, nan=0.0)
            model_count += 1

            for idx in test:
                prediction_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": fold,
                        "sample": int(idx),
                        "true_label": int(y[idx]),
                        "score": float(score[idx]),
                        "predicted_label": int(pred[idx]),
                        "selected_orthogonal_components": chosen,
                    }
                )

        repeat_metrics.append({"repeat": repeat, **metrics(y, pred)})

    metric_frame = pd.DataFrame(repeat_metrics)
    vip_frame = pd.DataFrame(
        {
            "retention_time_min": feature_names,
            "mean_vip4t": vip_sum / model_count,
            "vip4t_ge_1_frequency_percent": vip_hits / model_count * 100.0,
            "calibration_analyses": model_count,
        }
    ).sort_values("mean_vip4t", ascending=False)
    return (
        pd.DataFrame(prediction_rows),
        metric_frame,
        pd.DataFrame(component_rows),
        vip_frame,
        selected_counts,
    )


def write_boxplot(metrics_frame: pd.DataFrame, output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Accuracy", "Sensitivity", "Specificity"]
    # Match the established Figure 4a palette: gray, red, and blue.
    colors = ["#666666", "#ef2024", "#1749a6"]
    values = [
        metrics_frame["accuracy"].to_numpy() * 100,
        metrics_frame["sensitivity"].to_numpy() * 100,
        metrics_frame["specificity"].to_numpy() * 100,
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    boxplot = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        widths=0.55,
        boxprops={"facecolor": "none", "linewidth": 1.1},
        medianprops={"linewidth": 1.3},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 2.8, "markeredgecolor": "none"},
    )
    for i, color in enumerate(colors):
        boxplot["boxes"][i].set_facecolor(color)
        boxplot["boxes"][i].set_edgecolor(color)
        boxplot["boxes"][i].set_alpha(0.5)
        boxplot["medians"][i].set_color(color)
        boxplot["medians"][i].set_alpha(1.0)
        boxplot["whiskers"][2 * i].set_color(color)
        boxplot["whiskers"][2 * i + 1].set_color(color)
        boxplot["caps"][2 * i].set_color(color)
        boxplot["caps"][2 * i + 1].set_color(color)
        boxplot["fliers"][i].set_markerfacecolor(color)
        boxplot["fliers"][i].set_alpha(0.5)

    rng = np.random.default_rng(17)
    for i, (vals, color) in enumerate(zip(values, colors), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(i + jitter, vals, s=10, color=color, alpha=0.42, linewidths=0)
        ax.scatter(i, vals.mean(), s=58, color="black", edgecolor="white", linewidth=0.7, zorder=5)
    ax.set_ylabel("Percentage")
    ax.set_ylim(90, 101)
    ax.set_yticks([90, 95, 100])
    ax.grid(axis="y", color="#d7dde5", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext, dpi in (("svg", None), ("png", 600), ("pdf", None), ("tiff", 600)):
        kwargs = {} if dpi is None else {"dpi": dpi}
        fig.savefig(output_dir / f"Figure4a_Q2_repeated_metrics.{ext}", **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-components", type=int, default=9)
    parser.add_argument("--inner-folds", type=int, default=10)
    parser.add_argument("--q2-threshold", type=float, default=0.01)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.input, index_col=0)
    x = table.to_numpy(dtype=float)
    if x.shape != (136, 53):
        raise ValueError(f"Expected a 136 by 53 matrix, found {x.shape}.")
    if not table.index.is_unique:
        raise ValueError("Sample identifiers in PlasmaTable.csv must be unique.")
    if not np.isfinite(x).all():
        raise ValueError("PlasmaTable.csv contains missing or nonfinite peak areas.")
    first_group = table.index[:61].astype(str).str.startswith("n").all()
    second_group = table.index[61:].astype(str).str.startswith("x").all()
    if not (first_group and second_group):
        raise ValueError(
            "Unexpected sample order. The first 61 rows must be healthy control "
            "samples identified by n, followed by 75 semen abnormality samples "
            "identified by x."
        )
    y = np.r_[np.ones(61), -np.ones(75)]

    predictions, repeat_metrics, component_rows, vip_frame, selected_counts = repeated_q2_cv(
        x,
        y,
        list(table.columns.astype(str)),
        args.repeats,
        args.seed,
        args.max_components,
        args.inner_folds,
        args.q2_threshold,
    )
    predictions.to_csv(args.output_dir / "q2_nested_cv_predictions.csv", index=False)
    repeat_metrics.to_csv(args.output_dir / "q2_repeat_metrics.csv", index=False)
    component_rows.to_csv(args.output_dir / "q2_component_selection.csv", index=False)
    vip_frame.to_csv(args.output_dir / "q2_vip_stability.csv", index=False)
    write_boxplot(repeat_metrics, args.output_dir)

    summary = {
        "input": args.input.name,
        "input_sha256": sha256(args.input),
        "script_sha256": sha256(Path(__file__)),
        "helper_script_sha256": sha256(HERE / "oplsda_core.py"),
        "dataset": {"samples": 136, "variables": 53, "healthy_controls": 61, "semen_abnormalities": 75},
        "validation": {
            "repeats": args.repeats,
            "outer_folds": 10,
            "inner_folds_for_q2": args.inner_folds,
            "component_candidates": list(range(args.max_components + 1)),
            "q2_threshold_for_each_added_component": args.q2_threshold,
            "selection_description": "Within each outer calibration sample, the scaling parameters are calculated from all outer calibration samples. Q2 is then calculated by inner folds using those fixed scaling parameters. Successive orthogonal components are retained while incremental Q2 is at least the threshold. The outer validation samples are used only for prediction.",
            "selected_component_counts": selected_counts,
        },
        "accuracy": summarize_distribution(repeat_metrics, "accuracy"),
        "sensitivity": summarize_distribution(repeat_metrics, "sensitivity"),
        "specificity": summarize_distribution(repeat_metrics, "specificity"),
        "balanced_accuracy": summarize_distribution(repeat_metrics, "balanced_accuracy"),
        "vip": {
            "description": "For each analytical variable, VIP4t was averaged over the calibration analyses from the repeated sample divisions.",
            "calibration_analyses": int(args.repeats * 10),
            "variables_with_mean_vip4t_ge_1": int(np.sum(vip_frame["mean_vip4t"] >= 1.0)),
        },
        "limitations": [
            "The 136 original chromatograms were processed separately by DeepCPR before Table S11 was assembled. This analysis starts from that fixed matrix and therefore does not repeat peak extraction or target retention time matching within each cross validation division.",
            "The percentile interval describes variation among repeated sample divisions and is not an external clinical validation interval.",
        ],
    }
    (args.output_dir / "q2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

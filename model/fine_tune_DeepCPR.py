"""Leakage-aware supervised fine-tuning entry point for DeepCPR.

Each dataset directory must contain matched ``data<ID>.npy`` and
``label<ID>.npy`` files. One pair represents one sample:

* input:  (128, 800) or (1, 128, 800)
* label:  (128, 5) or (1, 128, 5)

The input is a globally normalized GC-MS segment. Label channels contain the
known component chromatographic profiles, packed from channel 0 and ordered by
increasing apex scan. Experimental CDF files or MCR/ITTFA outputs are not valid
supervised labels by themselves.

Example:

    python model/fine_tune_deepcpr.py \
      --train-dir data/real_shape_hybrid/train \
      --validation-dir data/real_shape_hybrid/validation \
      --retention-dir data/original_gaussian/validation \
      --output model/DeepCPR_real_shape_finetuned.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT = (128, 1, 800)
EXPECTED_LABEL = (128, 1, 5)
FILE_PATTERN = re.compile(r"^(data|label)(.+)\.npy$", re.IGNORECASE)


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def discover_pairs(directory: Path) -> list[tuple[str, Path, Path]]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    data_files: dict[str, Path] = {}
    label_files: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        kind, sample_id = match.groups()
        target = data_files if kind.lower() == "data" else label_files
        if sample_id in target:
            raise ValueError(f"Duplicate {kind} sample id {sample_id!r} in {directory}")
        target[sample_id] = path

    missing_labels = sorted(set(data_files) - set(label_files), key=natural_key)
    missing_data = sorted(set(label_files) - set(data_files), key=natural_key)
    if missing_labels or missing_data:
        raise ValueError(
            f"Unmatched files in {directory}: missing labels={missing_labels[:5]}, "
            f"missing data={missing_data[:5]}"
        )
    if not data_files:
        raise FileNotFoundError(f"No matched data<ID>.npy/label<ID>.npy files in {directory}")

    return [
        (sample_id, data_files[sample_id], label_files[sample_id])
        for sample_id in sorted(data_files, key=natural_key)
    ]


def limit_pairs(
    pairs: Sequence[tuple[str, Path, Path]], maximum: int
) -> list[tuple[str, Path, Path]]:
    if maximum < 0:
        raise ValueError("Sample limits must be zero or positive.")
    return list(pairs if maximum == 0 else pairs[:maximum])


def _standardize_array(array: np.ndarray, expected: tuple[int, int, int], name: str) -> np.ndarray:
    value = np.asarray(array)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    flat_shape = (expected[0], expected[2])
    if value.shape == flat_shape:
        value = value[:, np.newaxis, :]
    if value.shape != expected:
        raise ValueError(
            f"{name} has shape {array.shape}; expected {flat_shape}, "
            f"(1, {flat_shape[0]}, {flat_shape[1]}), or {expected}."
        )
    value = value.astype(np.float32, copy=False)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return value


def load_pair(data_path: Path, label_path: Path, threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    x = _standardize_array(np.load(data_path, allow_pickle=False), EXPECTED_INPUT, str(data_path))
    y = _standardize_array(np.load(label_path, allow_pickle=False), EXPECTED_LABEL, str(label_path))

    tolerance = 1e-5
    if float(x.min()) < -tolerance or float(x.max()) > 1.0 + tolerance:
        raise ValueError(f"Input must be normalized to [0, 1]: {data_path}")
    if float(y.min()) < -tolerance or float(y.max()) > 1.0 + tolerance:
        raise ValueError(f"Labels must be normalized to [0, 1]: {label_path}")
    if float(x.max()) <= threshold or float(y.max()) <= threshold:
        raise ValueError(f"Empty input or label pair: {data_path.name}, {label_path.name}")

    profiles = y[:, 0, :]
    active = np.flatnonzero(np.max(profiles, axis=0) > threshold)
    expected_active = np.arange(active.size)
    if not np.array_equal(active, expected_active):
        raise ValueError(
            f"Active label channels must be contiguous from channel 0 in {label_path}; "
            f"found {active.tolist()}."
        )
    apex_scans = np.array([np.argmax(profiles[:, index]) for index in active], dtype=int)
    if apex_scans.size > 1 and np.any(np.diff(apex_scans) < 0):
        raise ValueError(
            f"Active label channels must be ordered by increasing apex scan in {label_path}; "
            f"found {apex_scans.tolist()}."
        )
    return x, y, int(active.size)


class PairSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        pairs: Sequence[tuple[str, Path, Path]],
        batch_size: int,
        threshold: float,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.pairs = list(pairs)
        self.batch_size = batch_size
        self.threshold = threshold
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.pairs) / self.batch_size)

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        selected = self.indices[start:start + self.batch_size]
        inputs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for index in selected:
            _, data_path, label_path = self.pairs[int(index)]
            x, y, _ = load_pair(data_path, label_path, self.threshold)
            inputs.append(x)
            labels.append(y)
        return np.stack(inputs), np.stack(labels)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


def preflight(
    name: str,
    pairs: Sequence[tuple[str, Path, Path]],
    threshold: float,
    maximum: int,
) -> dict[str, object]:
    if maximum < 0:
        raise ValueError("--preflight-samples must be zero or positive.")
    checked = list(pairs if maximum == 0 else pairs[:maximum])
    component_counts: Counter[int] = Counter()
    for _, data_path, label_path in checked:
        _, _, count = load_pair(data_path, label_path, threshold)
        component_counts[count] += 1
    report = {
        "name": name,
        "samples": len(pairs),
        "preflight_checked": len(checked),
        "component_count_distribution_checked": dict(sorted(component_counts.items())),
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def make_loss(high_weight: float, low_weight: float, threshold: float):
    def PW_loss(y_true, y_pred):
        y_true_cast = tf.cast(y_true, tf.float32)
        y_pred_cast = tf.cast(y_pred, tf.float32)
        squared_error = tf.square(y_true_cast - y_pred_cast)
        weights = tf.where(
            y_true_cast > tf.cast(threshold, tf.float32),
            tf.cast(high_weight, tf.float32),
            tf.cast(low_weight, tf.float32),
        )
        return tf.reduce_mean(weights * squared_error)

    PW_loss.__name__ = "PW_loss"
    return PW_loss


def R_squared(y_true, y_pred):
    y_true_cast = tf.cast(y_true, tf.float32)
    y_pred_cast = tf.cast(y_pred, tf.float32)
    residual = tf.reduce_sum(tf.square(y_true_cast - y_pred_cast))
    total = tf.reduce_sum(tf.square(y_true_cast - tf.reduce_mean(y_true_cast)))
    return 1.0 - tf.math.divide_no_nan(residual, total)


def configure_trainable_layers(
    model: tf.keras.Model, freeze_before: str, train_batchnorm: bool
) -> dict[str, object]:
    if freeze_before.lower() == "none":
        start_index = 0
    else:
        matches = [index for index, layer in enumerate(model.layers) if layer.name == freeze_before]
        if not matches:
            raise ValueError(
                f"Layer {freeze_before!r} was not found. Use --freeze-before none for full "
                "fine-tuning or choose a name from model.layers."
            )
        start_index = matches[0]

    for index, layer in enumerate(model.layers):
        layer.trainable = index >= start_index
        if isinstance(layer, tf.keras.layers.BatchNormalization) and not train_batchnorm:
            layer.trainable = False

    trainable_parameters = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    frozen_parameters = int(sum(np.prod(weight.shape) for weight in model.non_trainable_weights))
    report = {
        "freeze_before": freeze_before,
        "freeze_before_index": start_index,
        "train_batchnorm": train_batchnorm,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if trainable_parameters == 0:
        raise ValueError("No trainable parameters remain after applying the freeze policy.")
    return report


def compile_model(model: tf.keras.Model, args: argparse.Namespace) -> None:
    optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
    }
    if args.clipnorm > 0:
        optimizer_kwargs["clipnorm"] = args.clipnorm
    optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
    model.compile(
        optimizer=optimizer,
        loss=make_loss(args.high_weight, args.low_weight, args.peak_threshold),
        metrics=[R_squared],
    )


def to_python_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in metrics.items()}


def evaluate(model: tf.keras.Model, sequence: PairSequence) -> dict[str, float]:
    return to_python_metrics(model.evaluate(sequence, verbose=0, return_dict=True))


def write_history(path: Path, history: dict[str, list[float]]) -> None:
    keys = list(history)
    rows = zip(*(history[key] for key in keys))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for epoch, row in enumerate(rows, start=1):
            writer.writerow([epoch, *row])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the 128x800 -> 128x5 DeepCPR model on exact paired labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=ROOT / "example" / "DeepCPR.h5")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument(
        "--retention-dir",
        type=Path,
        help="Optional original synthetic validation set used to quantify forgetting.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "DeepCPR_finetuned.h5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--high-weight", type=float, default=50.0)
    parser.add_argument("--low-weight", type=float, default=10.0)
    parser.add_argument("--peak-threshold", type=float, default=0.001)
    parser.add_argument("--freeze-before", default="conv2d_40")
    parser.add_argument("--train-batchnorm", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--preflight-samples",
        type=int,
        default=64,
        help="Samples checked before training; zero checks all samples.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0, help="Zero uses all samples.")
    parser.add_argument("--max-validation-samples", type=int, default=0, help="Zero uses all samples.")
    parser.add_argument("--max-retention-samples", type=int, default=0, help="Zero uses all samples.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one in-memory gradient update and exit without saving a model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check_only and args.smoke_test:
        raise ValueError("Choose either --check-only or --smoke-test, not both.")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive.")
    if args.learning_rate <= 0 or args.high_weight <= 0 or args.low_weight <= 0:
        raise ValueError("Learning rate and loss weights must be positive.")

    model_path = args.model.resolve()
    output_path = args.output.resolve()
    train_dir = args.train_dir.resolve()
    validation_dir = args.validation_dir.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if train_dir == validation_dir:
        raise ValueError("Training and validation directories must be different.")
    if output_path == model_path:
        raise ValueError("Output must not overwrite the pretrained model.")
    if output_path.exists() and not args.overwrite and not (args.check_only or args.smoke_test):
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")

    tf.keras.utils.set_random_seed(args.seed)
    train_pairs = limit_pairs(discover_pairs(train_dir), args.max_train_samples)
    validation_pairs = limit_pairs(
        discover_pairs(validation_dir), args.max_validation_samples
    )
    retention_pairs = None
    if args.retention_dir:
        retention_dir = args.retention_dir.resolve()
        if retention_dir in {train_dir, validation_dir}:
            raise ValueError("Retention directory must differ from training and validation directories.")
        retention_pairs = limit_pairs(
            discover_pairs(retention_dir), args.max_retention_samples
        )

    reports = [preflight(
        "train", train_pairs, args.peak_threshold, args.preflight_samples
    )]
    reports.append(preflight(
        "validation", validation_pairs, args.peak_threshold, args.preflight_samples
    ))
    if retention_pairs is not None:
        reports.append(preflight(
            "retention", retention_pairs, args.peak_threshold, args.preflight_samples
        ))

    model = tf.keras.models.load_model(str(model_path), compile=False)
    if tuple(model.input_shape[1:]) != EXPECTED_INPUT:
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if tuple(model.output_shape[1:]) != EXPECTED_LABEL:
        raise ValueError(f"Unexpected model output shape: {model.output_shape}")
    trainable_report = configure_trainable_layers(
        model, args.freeze_before, args.train_batchnorm
    )
    compile_model(model, args)

    train_sequence = PairSequence(
        train_pairs, args.batch_size, args.peak_threshold, True, args.seed
    )
    validation_sequence = PairSequence(
        validation_pairs, args.batch_size, args.peak_threshold, False, args.seed
    )
    retention_sequence = None
    if retention_pairs is not None:
        retention_sequence = PairSequence(
            retention_pairs, args.batch_size, args.peak_threshold, False, args.seed
        )

    first_x, first_y = train_sequence[0]
    if args.check_only:
        metrics = to_python_metrics(model.test_on_batch(first_x, first_y, return_dict=True))
        print(json.dumps({"check_only_metrics": metrics}, indent=2), flush=True)
        return

    if args.smoke_test:
        tracked_weight = model.trainable_weights[0]
        before = tracked_weight.numpy().copy()
        metrics = to_python_metrics(model.train_on_batch(first_x, first_y, return_dict=True))
        max_update = float(np.max(np.abs(tracked_weight.numpy() - before)))
        if max_update <= 0:
            raise RuntimeError("Smoke test completed without changing the first trainable weight.")
        print(json.dumps({
            "smoke_test_metrics": metrics,
            "first_trainable_weight": tracked_weight.name,
            "max_abs_weight_update": max_update,
        }, indent=2), flush=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_path.with_suffix(".history.csv")
    metadata_path = output_path.with_suffix(".run.json")
    baseline_metrics = {"validation": evaluate(model, validation_sequence)}
    if retention_sequence is not None:
        baseline_metrics["retention"] = evaluate(model, retention_sequence)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, args.patience // 2),
            min_lr=max(args.learning_rate * 0.01, 1e-9),
            verbose=1,
        ),
    ]
    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(str(output_path), include_optimizer=True)
    write_history(history_path, history.history)

    final_metrics = {"validation": evaluate(model, validation_sequence)}
    if retention_sequence is not None:
        final_metrics["retention"] = evaluate(model, retention_sequence)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf.__version__,
        "pretrained_model": str(model_path),
        "output_model": str(output_path),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "datasets": reports,
        "trainable_policy": trainable_report,
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "epochs_completed": len(history.history.get("loss", [])),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Saved fine-tuned model: {output_path}")
    print(f"Saved training history: {history_path}")
    print(f"Saved run metadata: {metadata_path}")


if __name__ == "__main__":
    main()

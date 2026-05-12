#!/usr/bin/env python3
"""Layer-wise linear probing for all-layer MERT features.

The expected feature files must be extracted with:

    python3 src/extract_mert_features.py --save-all-layers ...

MERT hidden states usually contain 13 entries: embedding output at index 0 plus
12 Transformer layers at indices 1..12. This script probes Transformer layers
1..12 by default. If a feature file contains exactly 12 saved layers, they are
treated as Transformer layers 1..12.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from train_classifier import (
    TrackExample,
    assert_group_disjoint,
    build_cv_splits,
    build_gtzan_artist_groups,
    load_grouped_features,
)


DEFAULT_WESTERN_FEATURES = "datasets/features/mert_gtzan_all_layers/features.npz"
DEFAULT_EASTERN_FEATURES = "datasets/features/mert_eastern_all_layers/features.npz"
DEFAULT_GTZAN_ARTIST_INDEX = "datasets/gtzan_meta/index.txt"
DEFAULT_OUTPUT_DIR = "outputs/layerwise_probe"


@dataclass(frozen=True)
class ProbeDataset:
    name: str
    examples: list[TrackExample]
    class_names: list[str]
    labels: np.ndarray
    groups: np.ndarray | None
    group_summary: dict[str, object] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one linear probe per MERT layer on Western and Eastern datasets."
    )
    parser.add_argument(
        "--western-features",
        type=Path,
        default=Path(DEFAULT_WESTERN_FEATURES),
        help="All-layer Western MERT features.npz.",
    )
    parser.add_argument(
        "--eastern-features",
        type=Path,
        default=Path(DEFAULT_EASTERN_FEATURES),
        help="All-layer Eastern MERT features.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV, JSON, and plots.",
    )
    parser.add_argument(
        "--gtzan-artist-index",
        type=Path,
        default=Path(DEFAULT_GTZAN_ARTIST_INDEX),
        help=(
            "GTZAN index.txt for artist-filtered Western splits. The same grouping "
            "helper also falls back to source groups for non-GTZAN files."
        ),
    )
    parser.add_argument(
        "--no-group-split",
        action="store_true",
        help="Use random stratified folds instead of artist/source group folds.",
    )
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="Also probe hidden-state index 0 when the file stores embedding + 12 layers.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Class weighting for LogisticRegression.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0, help="Inverse L2 regularization.")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_probe_dataset(
    name: str,
    features_path: Path,
    gtzan_artist_index: Path | None,
    use_group_split: bool,
) -> ProbeDataset:
    examples, class_names = load_grouped_features(features_path)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    groups = None
    group_summary = None
    if use_group_split:
        if gtzan_artist_index is None:
            raise ValueError("--gtzan-artist-index is required for group splits")
        groups, group_summary = build_gtzan_artist_groups(examples, gtzan_artist_index)
    validate_all_layer_examples(name, examples)
    return ProbeDataset(
        name=name,
        examples=examples,
        class_names=class_names,
        labels=labels,
        groups=groups,
        group_summary=group_summary,
    )


def validate_all_layer_examples(name: str, examples: list[TrackExample]) -> None:
    if not examples:
        raise ValueError(f"{name}: no examples found")
    shape = examples[0].features.shape
    if len(shape) != 3:
        raise ValueError(
            f"{name}: expected grouped all-layer features shaped "
            f"[chunks, layers, dim], got {shape}. Re-extract with --save-all-layers."
        )
    if shape[-1] <= 1:
        raise ValueError(f"{name}: unexpected hidden dimension in feature shape {shape}")


def transformer_layer_map(
    num_saved_layers: int,
    include_embedding: bool,
) -> list[tuple[int, int]]:
    if num_saved_layers == 13:
        layer_pairs = [(layer_number, layer_number) for layer_number in range(1, 13)]
        if include_embedding:
            return [(0, 0), *layer_pairs]
        return layer_pairs
    if num_saved_layers == 12:
        return [(layer_number, layer_number - 1) for layer_number in range(1, 13)]
    raise ValueError(
        "Expected 12 Transformer layers or embedding+12 layers, "
        f"got {num_saved_layers} saved layers"
    )


def layer_feature_matrix(
    examples: list[TrackExample],
    saved_layer_index: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for example in examples:
        # [chunks, layers, dim] -> [dim]. Multiple chunks are averaged per track.
        rows.append(example.features[:, saved_layer_index, :].mean(axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    class_weight: str,
    c_value: float,
    max_iter: int,
    seed: int,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices])
    validation_features = scaler.transform(features[validation_indices])
    test_features = scaler.transform(features[test_indices])

    model = LogisticRegression(
        C=c_value,
        class_weight=None if class_weight == "none" else class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(train_features, labels[train_indices])

    train_predictions = model.predict(train_features)
    validation_predictions = model.predict(validation_features)
    test_predictions = model.predict(test_features)

    output: dict[str, float] = {}
    for split_name, split_labels, split_predictions in [
        ("train", labels[train_indices], train_predictions),
        ("validation", labels[validation_indices], validation_predictions),
        ("test", labels[test_indices], test_predictions),
    ]:
        for metric_name, value in metric_dict(split_labels, split_predictions).items():
            output[f"{split_name}_{metric_name}"] = value
    return output


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["mert_layer"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (dataset_name, layer_number), layer_rows in sorted(grouped.items()):
        test_acc = np.asarray([row["test_accuracy"] for row in layer_rows], dtype=float)
        test_f1 = np.asarray([row["test_macro_f1"] for row in layer_rows], dtype=float)
        val_f1 = np.asarray(
            [row["validation_macro_f1"] for row in layer_rows], dtype=float
        )
        summary_rows.append(
            {
                "dataset": dataset_name,
                "mert_layer": layer_number,
                "test_accuracy_mean": float(test_acc.mean()),
                "test_accuracy_std": float(test_acc.std(ddof=0)),
                "test_macro_f1_mean": float(test_f1.mean()),
                "test_macro_f1_std": float(test_f1.std(ddof=0)),
                "validation_macro_f1_mean": float(val_f1.mean()),
                "validation_macro_f1_std": float(val_f1.std(ddof=0)),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(
    summary_rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 5))
    for dataset_name in ["western", "eastern"]:
        dataset_rows = [
            row for row in summary_rows if str(row["dataset"]) == dataset_name
        ]
        if not dataset_rows:
            continue
        dataset_rows = sorted(dataset_rows, key=lambda row: int(row["mert_layer"]))
        layers = [int(row["mert_layer"]) for row in dataset_rows]
        means = [float(row[f"{metric}_mean"]) for row in dataset_rows]
        stds = [float(row[f"{metric}_std"]) for row in dataset_rows]
        plt.errorbar(
            layers,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=dataset_name,
        )
    plt.xlabel("MERT Transformer Layer")
    plt.ylabel(ylabel)
    plt.xticks(sorted({int(row["mert_layer"]) for row in summary_rows}))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def best_layer(summary_rows: list[dict[str, object]], dataset_name: str) -> dict[str, object]:
    rows = [row for row in summary_rows if str(row["dataset"]) == dataset_name]
    if not rows:
        return {}
    return max(rows, key=lambda row: float(row["test_macro_f1_mean"]))


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def run_dataset(
    dataset: ProbeDataset,
    layer_pairs: list[tuple[int, int]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    splits = build_cv_splits(
        dataset.labels,
        folds=args.folds,
        seed=args.seed,
        groups=dataset.groups,
    )
    if dataset.groups is not None:
        assert_group_disjoint(splits, dataset.groups)

    rows: list[dict[str, object]] = []
    for mert_layer, saved_layer_index in layer_pairs:
        features = layer_feature_matrix(dataset.examples, saved_layer_index)
        for fold_id, (train_indices, validation_indices, test_indices) in enumerate(
            splits, start=1
        ):
            metrics = train_probe(
                features=features,
                labels=dataset.labels,
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                class_weight=args.class_weight,
                c_value=args.c,
                max_iter=args.max_iter,
                seed=args.seed + fold_id + mert_layer,
            )
            rows.append(
                {
                    "dataset": dataset.name,
                    "mert_layer": mert_layer,
                    "saved_layer_index": saved_layer_index,
                    "fold": fold_id,
                    "num_train": int(len(train_indices)),
                    "num_validation": int(len(validation_indices)),
                    "num_test": int(len(test_indices)),
                    **metrics,
                }
            )
            print(
                f"{dataset.name} layer={mert_layer} fold={fold_id} "
                f"test_acc={metrics['test_accuracy']:.4f} "
                f"test_macro_f1={metrics['test_macro_f1']:.4f}"
            )
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_group_split = not args.no_group_split
    gtzan_artist_index = args.gtzan_artist_index if use_group_split else None
    western = load_probe_dataset(
        name="western",
        features_path=args.western_features,
        gtzan_artist_index=gtzan_artist_index,
        use_group_split=use_group_split,
    )
    eastern = load_probe_dataset(
        name="eastern",
        features_path=args.eastern_features,
        gtzan_artist_index=gtzan_artist_index,
        use_group_split=use_group_split,
    )

    western_num_layers = western.examples[0].features.shape[-2]
    eastern_num_layers = eastern.examples[0].features.shape[-2]
    if western_num_layers != eastern_num_layers:
        raise ValueError(
            "Western and Eastern all-layer files store different layer counts: "
            f"{western_num_layers} vs {eastern_num_layers}"
        )
    layer_pairs = transformer_layer_map(
        western_num_layers,
        include_embedding=args.include_embedding,
    )

    print(
        f"Loaded Western: {len(western.examples)} tracks, "
        f"{len(western.class_names)} classes"
    )
    print(
        f"Loaded Eastern: {len(eastern.examples)} tracks, "
        f"{len(eastern.class_names)} classes"
    )
    print(
        "Probing layers: "
        + ", ".join(str(layer_number) for layer_number, _ in layer_pairs)
    )
    print(f"Class weight: {args.class_weight}")
    print("Split strategy: group-aware" if use_group_split else "Split strategy: random")

    metric_rows = []
    metric_rows.extend(run_dataset(western, layer_pairs, args))
    metric_rows.extend(run_dataset(eastern, layer_pairs, args))
    summary_rows = summarize_rows(metric_rows)

    write_csv(args.output_dir / "layerwise_fold_metrics.csv", metric_rows)
    write_csv(args.output_dir / "layerwise_summary.csv", summary_rows)

    plot_curve(
        summary_rows,
        metric="test_macro_f1",
        ylabel="Test Macro F1",
        output_path=args.output_dir / "macro_f1_curve.png",
        dpi=args.dpi,
    )
    plot_curve(
        summary_rows,
        metric="test_accuracy",
        ylabel="Test Accuracy",
        output_path=args.output_dir / "accuracy_curve.png",
        dpi=args.dpi,
    )

    summary = {
        "args": serializable_args(args),
        "western_features": str(args.western_features),
        "eastern_features": str(args.eastern_features),
        "num_saved_layers": western_num_layers,
        "probed_layers": [
            {"mert_layer": layer_number, "saved_layer_index": saved_layer_index}
            for layer_number, saved_layer_index in layer_pairs
        ],
        "western_classes": western.class_names,
        "eastern_classes": eastern.class_names,
        "western_group_split": western.group_summary,
        "eastern_group_split": eastern.group_summary,
        "best_western_layer": best_layer(summary_rows, "western"),
        "best_eastern_layer": best_layer(summary_rows, "eastern"),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved layer-wise probing outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

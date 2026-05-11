#!/usr/bin/env python3
"""Visualize Experiment 2 mixed 18-class baseline results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = "datasets/features/mert_mixed_18/features.npz"
DEFAULT_RESULTS_DIR = "outputs/mlp_mert_18_artist_filtered_weighted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 18-class mixed Western+Eastern baseline results."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(DEFAULT_FEATURES),
        help="Path to mixed 18-class features.npz.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(DEFAULT_RESULTS_DIR),
        help="Training output directory containing cv_predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory. Defaults to RESULTS_DIR/figures.",
    )
    parser.add_argument(
        "--embedding-method",
        choices=["tsne", "pca"],
        default="tsne",
        help="2D projection method for raw MERT features.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train with --save-cv-predictions first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def savefig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def load_feature_metadata(features_path: Path) -> dict[str, Any]:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {features_path}")
    data = np.load(features_path, allow_pickle=True)
    class_names = [str(name) for name in data["class_names"].tolist()]
    if "class_regions" in data.files:
        class_regions = [str(region) for region in data["class_regions"].tolist()]
    else:
        class_regions = ["unknown"] * len(class_names)
    return {
        "data": data,
        "class_names": class_names,
        "class_regions": class_regions,
    }


def get_test_predictions(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise ValueError("cv_predictions.csv contains no rows with split == 'test'")
    true_labels = np.asarray(
        [int(row["true_label"]) for row in test_rows], dtype=np.int64
    )
    pred_labels = np.asarray(
        [int(row["pred_label"]) for row in test_rows], dtype=np.int64
    )
    return true_labels, pred_labels


def plot_confusion_matrices(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    dpi: int,
) -> None:
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    for name, values, fmt, colorbar_label in [
        ("confusion_matrix_counts", matrix, "d", "Number of tracks"),
        ("confusion_matrix_normalized", normalized, ".2f", "Row-normalized proportion"),
    ]:
        plt.figure(figsize=(12, 10))
        image = plt.imshow(values, aspect="auto", cmap="Blues")
        plt.colorbar(image, label=colorbar_label)
        plt.xticks(labels, class_names, rotation=45, ha="right")
        plt.yticks(labels, class_names)
        plt.xlabel("Predicted genre")
        plt.ylabel("True genre")
        plt.title(name.replace("_", " ").title())

        if len(class_names) <= 20:
            for row in range(len(class_names)):
                for col in range(len(class_names)):
                    value = values[row, col]
                    if matrix[row, col] == 0:
                        continue
                    text = format(value, fmt)
                    plt.text(
                        col,
                        row,
                        text,
                        ha="center",
                        va="center",
                        color="white" if normalized[row, col] > 0.45 else "black",
                        fontsize=7,
                    )
        savefig(output_dir / f"{name}.png", dpi)


def plot_per_class_f1(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str],
    class_regions: list[str],
    output_dir: Path,
    dpi: int,
) -> None:
    report = classification_report(
        true_labels,
        pred_labels,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    f1_scores = np.asarray([report[name]["f1-score"] for name in class_names])
    colors = [
        "#4c78a8" if region == "western" else "#f58518" for region in class_regions
    ]
    order = np.argsort(f1_scores)

    plt.figure(figsize=(9, max(5, 0.38 * len(class_names))))
    plt.barh(
        np.arange(len(class_names)), f1_scores[order], color=[colors[i] for i in order]
    )
    plt.yticks(np.arange(len(class_names)), [class_names[i] for i in order])
    plt.xlabel("F1-score")
    plt.xlim(0, 1)
    plt.title("Per-class F1 on cross-validated test folds")
    savefig(output_dir / "per_class_f1.png", dpi)

    with (output_dir / "classification_report_test.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)


def plot_region_confusion(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_regions: list[str],
    output_dir: Path,
    dpi: int,
) -> None:
    region_names = ["western", "eastern"]
    region_to_id = {name: index for index, name in enumerate(region_names)}
    true_regions = np.asarray(
        [region_to_id[class_regions[label]] for label in true_labels]
    )
    pred_regions = np.asarray(
        [region_to_id[class_regions[label]] for label in pred_labels]
    )
    matrix = confusion_matrix(
        true_regions, pred_regions, labels=np.arange(len(region_names))
    )
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(5.5, 4.8))
    image = plt.imshow(normalized, cmap="Purples", vmin=0, vmax=1)
    plt.colorbar(image, label="Row-normalized proportion")
    plt.xticks(np.arange(len(region_names)), region_names)
    plt.yticks(np.arange(len(region_names)), region_names)
    plt.xlabel("Predicted region")
    plt.ylabel("True region")
    plt.title("Western/Eastern Region Confusion")
    for row in range(len(region_names)):
        for col in range(len(region_names)):
            plt.text(
                col,
                row,
                f"{normalized[row, col]:.2f}\n({matrix[row, col]})",
                ha="center",
                va="center",
                color="white" if normalized[row, col] > 0.5 else "black",
            )
    savefig(output_dir / "region_confusion_matrix.png", dpi)


def group_track_features(
    data: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    if "track_ids" in data.files:
        track_ids = data["track_ids"].astype(np.int64)
    else:
        track_ids = np.arange(len(labels), dtype=np.int64)
    if "segment_indices" in data.files:
        segment_indices = data["segment_indices"].astype(np.int64)
    else:
        segment_indices = np.zeros(len(labels), dtype=np.int64)

    grouped_features: list[np.ndarray] = []
    grouped_labels: list[int] = []
    grouped_track_ids: list[int] = []
    for track_id in np.unique(track_ids):
        row_indices = np.where(track_ids == track_id)[0]
        row_indices = row_indices[np.argsort(segment_indices[row_indices])]
        grouped_features.append(features[row_indices].mean(axis=0).reshape(-1))
        grouped_labels.append(int(labels[row_indices[0]]))
        grouped_track_ids.append(int(track_id))
    return (
        np.stack(grouped_features, axis=0).astype(np.float32),
        np.asarray(grouped_labels, dtype=np.int64),
        np.asarray(grouped_track_ids, dtype=np.int64),
    )


def reduce_to_2d(features: np.ndarray, method: str, seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    if method == "pca":
        _, _, vh = np.linalg.svd(scaled, full_matrices=False)
        return (scaled @ vh[:2].T).astype(np.float32)
    perplexity = min(30, max(5, (len(features) - 1) // 3))
    return (
        TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        .fit_transform(scaled)
        .astype(np.float32)
    )


def scatter_by_labels(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    title: str,
    path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab20")
    for label_id, label_name in enumerate(label_names):
        mask = labels == label_id
        if not mask.any():
            continue
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=13,
            alpha=0.72,
            color=cmap(label_id % 20),
            label=label_name,
            linewidths=0,
        )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    savefig(path, dpi)


def scatter_by_region(
    coords: np.ndarray,
    labels: np.ndarray,
    class_regions: list[str],
    title: str,
    path: Path,
    dpi: int,
) -> None:
    region_names = ["western", "eastern"]
    colors = {"western": "#4c78a8", "eastern": "#f58518"}
    plt.figure(figsize=(8, 6.5))
    for region in region_names:
        region_labels = [
            idx for idx, value in enumerate(class_regions) if value == region
        ]
        mask = np.isin(labels, region_labels)
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=14,
            alpha=0.72,
            color=colors[region],
            label=region,
            linewidths=0,
        )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.legend(frameon=False)
    savefig(path, dpi)


def plot_feature_embedding(
    features_path: Path,
    output_dir: Path,
    class_names: list[str],
    class_regions: list[str],
    method: str,
    seed: int,
    dpi: int,
) -> None:
    data = np.load(features_path, allow_pickle=True)
    track_features, track_labels, _ = group_track_features(data)
    coords = reduce_to_2d(track_features, method=method, seed=seed)
    np.save(output_dir / f"raw_mert_{method}_coords.npy", coords)
    scatter_by_labels(
        coords,
        track_labels,
        class_names,
        f"Raw MERT features colored by 18 genres ({method})",
        output_dir / f"raw_mert_{method}_by_genre.png",
        dpi,
    )
    scatter_by_region(
        coords,
        track_labels,
        class_regions,
        f"Raw MERT features colored by region ({method})",
        output_dir / f"raw_mert_{method}_by_region.png",
        dpi,
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_feature_metadata(args.features)
    class_names = metadata["class_names"]
    class_regions = metadata["class_regions"]

    prediction_rows = read_csv(args.results_dir / "cv_predictions.csv")
    true_labels, pred_labels = get_test_predictions(prediction_rows)

    plot_confusion_matrices(true_labels, pred_labels, class_names, output_dir, args.dpi)
    plot_per_class_f1(
        true_labels, pred_labels, class_names, class_regions, output_dir, args.dpi
    )
    plot_region_confusion(true_labels, pred_labels, class_regions, output_dir, args.dpi)
    plot_feature_embedding(
        args.features,
        output_dir,
        class_names,
        class_regions,
        args.embedding_method,
        args.seed,
        args.dpi,
    )
    print(f"Saved Experiment 2 figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

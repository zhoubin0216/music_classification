#!/usr/bin/env python3
"""Visualize Experiment 1 Western-bias probing outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_DIR = "outputs/experiment1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create heatmaps and embedding plots for Experiment 1."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help="Directory produced by experiment1_bias_probe.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures. Defaults to INPUT_DIR/figures.",
    )
    parser.add_argument(
        "--embedding-method",
        choices=["tsne", "pca"],
        default="tsne",
        help="2D projection method for feature scatter plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def savefig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_projection_heatmap(input_dir: Path, output_dir: Path, dpi: int) -> None:
    rows = read_csv(input_dir / "eastern_to_western_projection.csv")
    eastern_names = [row["eastern_genre"] for row in rows]
    proportion_keys = [key for key in rows[0] if key.startswith("proportion_")]
    western_names = [key.removeprefix("proportion_") for key in proportion_keys]
    matrix = np.asarray(
        [[float(row[key]) for key in proportion_keys] for row in rows],
        dtype=np.float32,
    )

    fig_width = max(9.0, 0.7 * len(western_names))
    fig_height = max(4.5, 0.45 * len(eastern_names))
    plt.figure(figsize=(fig_width, fig_height))
    image = plt.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    plt.colorbar(image, label="Projection proportion")
    plt.xticks(np.arange(len(western_names)), western_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(eastern_names)), eastern_names)
    plt.xlabel("Western predicted genre")
    plt.ylabel("Eastern source genre")
    plt.title("Eastern genres projected into Western 10-class label space")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if value >= 0.15:
                plt.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.55 else "black",
                    fontsize=8,
                )

    savefig(output_dir / "projection_heatmap.png", dpi)


def plot_confidence_by_genre(input_dir: Path, output_dir: Path, dpi: int) -> None:
    summary = load_json(input_dir / "confidence_summary.json")
    genre_items = list(summary["by_eastern_genre"].items())
    genre_names = [name for name, _ in genre_items]
    confidence = np.asarray([item["mean_confidence"] for _, item in genre_items])
    confidence_std = np.asarray([item["std_confidence"] for _, item in genre_items])
    entropy = np.asarray([item["mean_entropy"] for _, item in genre_items])
    entropy_std = np.asarray([item["std_entropy"] for _, item in genre_items])

    y = np.arange(len(genre_names))
    plt.figure(figsize=(8.5, max(4.5, 0.45 * len(genre_names))))
    plt.barh(y, confidence, xerr=confidence_std, color="#4c78a8", alpha=0.85)
    plt.yticks(y, genre_names)
    plt.xlabel("Mean max softmax probability")
    plt.ylabel("Eastern genre")
    plt.xlim(0, 1)
    plt.title("Western model confidence on Eastern genres")
    savefig(output_dir / "confidence_by_eastern_genre.png", dpi)

    plt.figure(figsize=(8.5, max(4.5, 0.45 * len(genre_names))))
    plt.barh(y, entropy, xerr=entropy_std, color="#f58518", alpha=0.85)
    plt.axvline(
        float(summary["max_entropy_for_10_classes"]),
        color="black",
        linestyle="--",
        linewidth=1,
        label="Max entropy for 10 classes",
    )
    plt.yticks(y, genre_names)
    plt.xlabel("Mean predictive entropy")
    plt.ylabel("Eastern genre")
    plt.title("Prediction uncertainty on Eastern genres")
    plt.legend()
    savefig(output_dir / "entropy_by_eastern_genre.png", dpi)


def plot_confidence_histogram(input_dir: Path, output_dir: Path, dpi: int) -> None:
    rows = read_csv(input_dir / "eastern_closed_set_predictions.csv")
    confidence = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float32)
    entropy = np.asarray([float(row["entropy"]) for row in rows], dtype=np.float32)

    plt.figure(figsize=(7.5, 4.5))
    plt.hist(confidence, bins=30, color="#4c78a8", edgecolor="white")
    plt.xlabel("Max softmax probability")
    plt.ylabel("Number of tracks")
    plt.title("Closed-set confidence distribution on Eastern tracks")
    savefig(output_dir / "confidence_histogram.png", dpi)

    plt.figure(figsize=(7.5, 4.5))
    plt.hist(entropy, bins=30, color="#f58518", edgecolor="white")
    plt.xlabel("Predictive entropy")
    plt.ylabel("Number of tracks")
    plt.title("Closed-set entropy distribution on Eastern tracks")
    savefig(output_dir / "entropy_histogram.png", dpi)


def reduce_to_2d(features: np.ndarray, method: str, seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    if method == "pca":
        _, _, vh = np.linalg.svd(scaled, full_matrices=False)
        return (scaled @ vh[:2].T).astype(np.float32)

    perplexity = min(30, max(5, (len(features) - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(scaled).astype(np.float32)


def scatter_by_label(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    title: str,
    path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8.5, 6.5))
    cmap = plt.get_cmap("tab20")
    for label_id, label_name in enumerate(label_names):
        mask = labels == label_id
        if not mask.any():
            continue
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=14,
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


def plot_feature_spaces(input_dir: Path, output_dir: Path, method: str, seed: int, dpi: int) -> None:
    data = np.load(input_dir / "eastern_bias_probe_features.npz", allow_pickle=True)
    eastern_labels = data["eastern_labels"].astype(np.int64)
    western_predictions = data["predicted_western_labels"].astype(np.int64)
    eastern_names = [str(name) for name in data["eastern_class_names"].tolist()]
    western_names = [str(name) for name in data["western_class_names"].tolist()]

    spaces = {
        "raw_mert": data["raw_mert_features"],
        "mlp_pooled": data["mlp_pooled_features"],
        "mlp_penultimate": data["mlp_penultimate_features"],
    }
    for space_name, features in spaces.items():
        coords = reduce_to_2d(features.astype(np.float32), method=method, seed=seed)
        np.save(output_dir / f"{space_name}_{method}_coords.npy", coords)
        scatter_by_label(
            coords,
            eastern_labels,
            eastern_names,
            f"{space_name}: colored by Eastern genre",
            output_dir / f"{space_name}_{method}_by_eastern_genre.png",
            dpi,
        )
        scatter_by_label(
            coords,
            western_predictions,
            western_names,
            f"{space_name}: colored by Western projection",
            output_dir / f"{space_name}_{method}_by_western_projection.png",
            dpi,
        )


def plot_clustering_summary(input_dir: Path, output_dir: Path, dpi: int) -> None:
    summary = load_json(input_dir / "clustering_summary.json")
    spaces = list(summary["spaces"].keys())
    ari = [summary["spaces"][name]["adjusted_rand_index"] for name in spaces]
    nmi = [summary["spaces"][name]["normalized_mutual_info"] for name in spaces]

    x = np.arange(len(spaces))
    width = 0.36
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width / 2, ari, width, label="ARI", color="#4c78a8")
    plt.bar(x + width / 2, nmi, width, label="NMI", color="#54a24b")
    plt.xticks(x, spaces, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("KMeans alignment with Eastern genre labels")
    plt.legend()
    savefig(output_dir / "clustering_metrics.png", dpi)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_projection_heatmap(input_dir, output_dir, args.dpi)
    plot_confidence_by_genre(input_dir, output_dir, args.dpi)
    plot_confidence_histogram(input_dir, output_dir, args.dpi)
    plot_feature_spaces(input_dir, output_dir, args.embedding_method, args.seed, args.dpi)
    plot_clustering_summary(input_dir, output_dir, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

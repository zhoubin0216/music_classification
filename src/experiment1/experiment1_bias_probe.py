#!/usr/bin/env python3
"""Experiment 1: probe Western-genre bias on Eastern music.

This script loads a Western-trained 10-class MLP checkpoint, applies it to
Eastern MERT features without further training, and exports:

  - per-track closed-set Western predictions for Eastern tracks
  - Eastern-genre -> Western-prediction distribution tables
  - confidence and entropy summaries
  - raw MERT track features and MLP-mapped features for clustering/UMAP

The prediction labels should be interpreted as Western closed-set projections,
not as Eastern genre accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from train_classifier import (
    MLPGenreClassifier,
    Standardizer,
    apply_standardizer,
    collate_tracks,
    load_grouped_features,
)


DEFAULT_EASTERN_FEATURES = "datasets/features/mert_eastern/features.npz"
DEFAULT_WESTERN_MODEL = "outputs/mlp_mert_gtzan/final_model.pt"
DEFAULT_OUTPUT_DIR = "outputs/experiment1_western_bias_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a Western-trained MLP to Eastern MERT features."
    )
    parser.add_argument(
        "--eastern-features",
        type=Path,
        default=Path(DEFAULT_EASTERN_FEATURES),
        help="Path to Eastern features.npz.",
    )
    parser.add_argument(
        "--western-model",
        type=Path,
        default=Path(DEFAULT_WESTERN_MODEL),
        help="Path to Western-trained final_model.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for experiment outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=None,
        help="Number of KMeans clusters. Defaults to the number of Eastern genres.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:0, mps, or cpu.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint


def get_model_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    return {
        "hidden_dim": int(args.get("hidden_dim", 64)),
        "aggregation_hidden_dim": int(args.get("aggregation_hidden_dim", 16)),
        "dropout": float(args.get("dropout", 0.0)),
    }


def build_model(checkpoint: dict[str, Any], device: torch.device) -> MLPGenreClassifier:
    class_names = [str(name) for name in checkpoint["class_names"]]
    input_shape = tuple(int(value) for value in checkpoint["input_shape"])
    model_args = get_model_args(checkpoint)
    model = MLPGenreClassifier(
        input_shape=input_shape,
        num_classes=len(class_names),
        hidden_dim=model_args["hidden_dim"],
        aggregation_hidden_dim=model_args["aggregation_hidden_dim"],
        dropout=model_args["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    return model


def validate_feature_shape(examples: list[Any], checkpoint: dict[str, Any]) -> None:
    expected_input_shape = tuple(int(value) for value in checkpoint["input_shape"])
    actual_input_shape = tuple(examples[0].features.shape[1:])
    if actual_input_shape != expected_input_shape:
        raise ValueError(
            "Eastern features do not match the Western model input shape. "
            f"Expected per-chunk shape {expected_input_shape}, got {actual_input_shape}. "
            "Use the same MERT layer/chunk/all-layers settings for both datasets."
        )


def make_standardized_examples(examples: list[Any], checkpoint: dict[str, Any]) -> list[Any]:
    standardizer = Standardizer(
        mean=np.asarray(checkpoint["standardizer_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["standardizer_std"], dtype=np.float32),
    )
    return apply_standardizer(examples, standardizer)


def raw_track_feature(features: np.ndarray) -> np.ndarray:
    """Return a 2D clustering feature for raw MERT features from one track."""
    chunk_mean = features.mean(axis=0)
    return chunk_mean.reshape(-1).astype(np.float32)


@torch.inference_mode()
def forward_features(
    model: MLPGenreClassifier,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.layer_aggregation is not None:
        features = model.layer_aggregation(features)

    pooled, chunk_weights = model.chunk_aggregation(features, mask)
    penultimate = model.classifier[:-1](pooled)
    logits = model.classifier[-1](penultimate)
    probabilities = torch.softmax(logits, dim=1)
    return logits, probabilities, pooled, penultimate, chunk_weights


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    western_class_names: list[str],
) -> None:
    fieldnames = [
        "track_index",
        "track_id",
        "path",
        "eastern_label",
        "eastern_genre",
        "predicted_western_label",
        "predicted_western_genre",
        "confidence",
        "entropy",
        *[f"prob_{name}" for name in western_class_names],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_projection_distribution(
    path: Path,
    eastern_labels: np.ndarray,
    eastern_class_names: list[str],
    predicted_labels: np.ndarray,
    western_class_names: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "eastern_genre",
                "num_tracks",
                *[f"count_{name}" for name in western_class_names],
                *[f"proportion_{name}" for name in western_class_names],
            ]
        )
        for eastern_id, eastern_name in enumerate(eastern_class_names):
            mask = eastern_labels == eastern_id
            counts = np.bincount(
                predicted_labels[mask],
                minlength=len(western_class_names),
            )
            total = int(mask.sum())
            proportions = counts / max(total, 1)
            writer.writerow(
                [
                    eastern_name,
                    total,
                    *counts.astype(int).tolist(),
                    *[float(value) for value in proportions],
                ]
            )


def summarize_confidence(
    eastern_labels: np.ndarray,
    eastern_class_names: list[str],
    confidence: np.ndarray,
    entropies: np.ndarray,
    predicted_labels: np.ndarray,
    western_class_names: list[str],
) -> dict[str, Any]:
    by_eastern_genre: dict[str, Any] = {}
    for eastern_id, eastern_name in enumerate(eastern_class_names):
        mask = eastern_labels == eastern_id
        if not mask.any():
            continue
        counts = np.bincount(
            predicted_labels[mask],
            minlength=len(western_class_names),
        )
        top_western_id = int(counts.argmax())
        by_eastern_genre[eastern_name] = {
            "num_tracks": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()),
            "std_confidence": float(confidence[mask].std()),
            "mean_entropy": float(entropies[mask].mean()),
            "std_entropy": float(entropies[mask].std()),
            "top_projected_western_genre": western_class_names[top_western_id],
            "top_projected_count": int(counts[top_western_id]),
            "top_projected_proportion": float(counts[top_western_id] / mask.sum()),
        }

    return {
        "num_tracks": int(len(eastern_labels)),
        "mean_confidence": float(confidence.mean()),
        "std_confidence": float(confidence.std()),
        "mean_entropy": float(entropies.mean()),
        "std_entropy": float(entropies.std()),
        "max_entropy_for_10_classes": float(math.log(len(western_class_names))),
        "by_eastern_genre": by_eastern_genre,
    }


def run_kmeans(
    features: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    scaled_features = StandardScaler().fit_transform(features)
    cluster_ids = KMeans(
        n_clusters=num_clusters,
        n_init=20,
        random_state=seed,
    ).fit_predict(scaled_features)
    metrics = {
        "adjusted_rand_index": float(adjusted_rand_score(labels, cluster_ids)),
        "normalized_mutual_info": float(normalized_mutual_info_score(labels, cluster_ids)),
    }
    return cluster_ids.astype(np.int64), metrics


def write_cluster_assignments(
    path: Path,
    rows: list[dict[str, Any]],
    cluster_outputs: dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "track_index",
        "track_id",
        "path",
        "eastern_label",
        "eastern_genre",
        "predicted_western_label",
        "predicted_western_genre",
        *[f"cluster_{name}" for name in cluster_outputs],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(rows):
            writer.writerow(
                {
                    "track_index": row["track_index"],
                    "track_id": row["track_id"],
                    "path": row["path"],
                    "eastern_label": row["eastern_label"],
                    "eastern_genre": row["eastern_genre"],
                    "predicted_western_label": row["predicted_western_label"],
                    "predicted_western_genre": row["predicted_western_genre"],
                    **{
                        f"cluster_{name}": int(cluster_ids[row_index])
                        for name, cluster_ids in cluster_outputs.items()
                    },
                }
            )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = load_checkpoint(args.western_model, device)
    western_class_names = [str(name) for name in checkpoint["class_names"]]
    model = build_model(checkpoint, device)

    examples, eastern_class_names = load_grouped_features(args.eastern_features)
    validate_feature_shape(examples, checkpoint)
    scaled_examples = make_standardized_examples(examples, checkpoint)

    print(f"Loaded Western model: {args.western_model}")
    print(f"Western classes ({len(western_class_names)}): {', '.join(western_class_names)}")
    print(f"Loaded Eastern tracks: {len(examples)} from {args.eastern_features}")
    print(f"Eastern classes ({len(eastern_class_names)}): {', '.join(eastern_class_names)}")
    print(f"Using device: {device}")

    rows: list[dict[str, Any]] = []
    eastern_labels: list[int] = []
    predicted_labels: list[int] = []
    confidence_values: list[float] = []
    entropy_values: list[float] = []
    all_probabilities: list[np.ndarray] = []
    raw_features: list[np.ndarray] = []
    pooled_features: list[np.ndarray] = []
    penultimate_features: list[np.ndarray] = []
    chunk_weight_rows: list[np.ndarray] = []
    chunk_mask_rows: list[np.ndarray] = []

    for start in range(0, len(examples), args.batch_size):
        stop = min(start + args.batch_size, len(examples))
        batch_items = [
            (
                torch.from_numpy(scaled_examples[index].features).float(),
                torch.tensor(scaled_examples[index].label, dtype=torch.long),
                torch.tensor(index, dtype=torch.long),
            )
            for index in range(start, stop)
        ]
        features, labels, mask, indices = collate_tracks(batch_items)
        features = features.to(device)
        mask = mask.to(device)

        _, probabilities, pooled, penultimate, chunk_weights = forward_features(
            model,
            features,
            mask,
        )

        probabilities_np = probabilities.cpu().numpy()
        pooled_np = pooled.cpu().numpy().astype(np.float32)
        penultimate_np = penultimate.cpu().numpy().astype(np.float32)
        chunk_weights_np = chunk_weights.cpu().numpy().astype(np.float32)
        chunk_mask_np = mask.cpu().numpy()
        batch_predicted = probabilities_np.argmax(axis=1)
        batch_confidence = probabilities_np.max(axis=1)
        batch_entropy = entropy(probabilities_np)

        for row_index, example_index in enumerate(indices.numpy()):
            example = examples[int(example_index)]
            eastern_label = int(labels[row_index].item())
            predicted_label = int(batch_predicted[row_index])
            probabilities_row = probabilities_np[row_index]

            rows.append(
                {
                    "track_index": int(example_index),
                    "track_id": int(example.track_id),
                    "path": example.path,
                    "eastern_label": eastern_label,
                    "eastern_genre": eastern_class_names[eastern_label],
                    "predicted_western_label": predicted_label,
                    "predicted_western_genre": western_class_names[predicted_label],
                    "confidence": float(batch_confidence[row_index]),
                    "entropy": float(batch_entropy[row_index]),
                    **{
                        f"prob_{name}": float(probabilities_row[class_id])
                        for class_id, name in enumerate(western_class_names)
                    },
                }
            )
            eastern_labels.append(eastern_label)
            predicted_labels.append(predicted_label)
            confidence_values.append(float(batch_confidence[row_index]))
            entropy_values.append(float(batch_entropy[row_index]))
            all_probabilities.append(probabilities_row.astype(np.float32))
            raw_features.append(raw_track_feature(example.features))
            pooled_features.append(pooled_np[row_index])
            penultimate_features.append(penultimate_np[row_index])
            chunk_weight_rows.append(chunk_weights_np[row_index])
            chunk_mask_rows.append(chunk_mask_np[row_index])

    eastern_labels_np = np.asarray(eastern_labels, dtype=np.int64)
    predicted_labels_np = np.asarray(predicted_labels, dtype=np.int64)
    confidence_np = np.asarray(confidence_values, dtype=np.float32)
    entropy_np = np.asarray(entropy_values, dtype=np.float32)

    write_predictions(
        args.output_dir / "eastern_closed_set_predictions.csv",
        rows,
        western_class_names,
    )
    write_projection_distribution(
        args.output_dir / "eastern_to_western_projection.csv",
        eastern_labels_np,
        eastern_class_names,
        predicted_labels_np,
        western_class_names,
    )

    summary = summarize_confidence(
        eastern_labels_np,
        eastern_class_names,
        confidence_np,
        entropy_np,
        predicted_labels_np,
        western_class_names,
    )
    summary.update(
        {
            "western_model": str(args.western_model),
            "eastern_features": str(args.eastern_features),
            "output_dir": str(args.output_dir),
            "western_class_names": western_class_names,
            "eastern_class_names": eastern_class_names,
        }
    )
    with (args.output_dir / "confidence_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    raw_feature_array = np.stack(raw_features, axis=0)
    pooled_feature_array = np.stack(pooled_features, axis=0)
    penultimate_feature_array = np.stack(penultimate_features, axis=0)
    probability_array = np.stack(all_probabilities, axis=0)

    num_clusters = args.num_clusters or len(eastern_class_names)
    cluster_spaces = {
        "raw_mert": raw_feature_array,
        "mlp_pooled": pooled_feature_array,
        "mlp_penultimate": penultimate_feature_array,
    }
    cluster_outputs: dict[str, np.ndarray] = {}
    clustering_summary: dict[str, Any] = {
        "num_clusters": int(num_clusters),
        "label_reference": "eastern_genre",
        "spaces": {},
    }
    for space_name, feature_array in cluster_spaces.items():
        cluster_ids, cluster_metrics = run_kmeans(
            feature_array,
            eastern_labels_np,
            num_clusters=num_clusters,
        )
        cluster_outputs[space_name] = cluster_ids
        clustering_summary["spaces"][space_name] = cluster_metrics

    write_cluster_assignments(
        args.output_dir / "cluster_assignments.csv",
        rows,
        cluster_outputs,
    )
    with (args.output_dir / "clustering_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(clustering_summary, handle, indent=2)

    np.savez_compressed(
        args.output_dir / "eastern_bias_probe_features.npz",
        raw_mert_features=raw_feature_array,
        mlp_pooled_features=pooled_feature_array,
        mlp_penultimate_features=penultimate_feature_array,
        probabilities=probability_array,
        eastern_labels=eastern_labels_np,
        predicted_western_labels=predicted_labels_np,
        confidence=confidence_np,
        entropy=entropy_np,
        paths=np.asarray([row["path"] for row in rows]),
        track_ids=np.asarray([row["track_id"] for row in rows], dtype=np.int64),
        eastern_class_names=np.asarray(eastern_class_names),
        western_class_names=np.asarray(western_class_names),
        raw_mert_clusters=cluster_outputs["raw_mert"],
        mlp_pooled_clusters=cluster_outputs["mlp_pooled"],
        mlp_penultimate_clusters=cluster_outputs["mlp_penultimate"],
        chunk_weights=np.asarray(chunk_weight_rows, dtype=object),
        chunk_masks=np.asarray(chunk_mask_rows, dtype=object),
    )

    print(f"Saved predictions to: {args.output_dir / 'eastern_closed_set_predictions.csv'}")
    print(f"Saved projection table to: {args.output_dir / 'eastern_to_western_projection.csv'}")
    print(f"Saved clustering summary to: {args.output_dir / 'clustering_summary.json'}")
    print(f"Saved feature exports to: {args.output_dir / 'eastern_bias_probe_features.npz'}")
    print(f"Saved confidence summary to: {args.output_dir / 'confidence_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare mixed Western+Eastern MERT features for 18-class classification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_WESTERN_FEATURES = "datasets/features/mert_gtzan/features.npz"
DEFAULT_EASTERN_FEATURES = "datasets/features/mert_eastern/features.npz"
DEFAULT_OUTPUT_DIR = "datasets/features/mert_mixed_18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Western and Eastern MERT features into one 18-class feature file."
    )
    parser.add_argument(
        "--western-features",
        type=Path,
        default=Path(DEFAULT_WESTERN_FEATURES),
        help="Path to Western features.npz.",
    )
    parser.add_argument(
        "--eastern-features",
        type=Path,
        default=Path(DEFAULT_EASTERN_FEATURES),
        help="Path to Eastern features.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for merged features.npz and metadata files.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def as_string_array(values: np.ndarray) -> np.ndarray:
    return values.astype(str)


def get_required(data: dict[str, Any], key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {path}")
    return data[key]


def get_track_ids(data: dict[str, Any], num_rows: int) -> np.ndarray:
    if "track_ids" in data:
        return data["track_ids"].astype(np.int64)
    return np.arange(num_rows, dtype=np.int64)


def get_segment_indices(data: dict[str, Any], num_rows: int) -> np.ndarray:
    if "segment_indices" in data:
        return data["segment_indices"].astype(np.int64)
    return np.zeros(num_rows, dtype=np.int64)


def offset_track_ids(track_ids: np.ndarray, offset: int) -> np.ndarray:
    unique_track_ids = np.unique(track_ids)
    mapping = {int(track_id): offset + index for index, track_id in enumerate(unique_track_ids)}
    return np.asarray([mapping[int(track_id)] for track_id in track_ids], dtype=np.int64)


def validate_compatible_features(
    western_features: np.ndarray,
    eastern_features: np.ndarray,
) -> None:
    if western_features.ndim != eastern_features.ndim:
        raise ValueError(
            "Western and Eastern features must have the same number of dimensions: "
            f"{western_features.shape} vs {eastern_features.shape}"
        )
    if western_features.shape[1:] != eastern_features.shape[1:]:
        raise ValueError(
            "Western and Eastern per-row feature shapes do not match: "
            f"{western_features.shape[1:]} vs {eastern_features.shape[1:]}"
        )


def write_metadata(
    path: Path,
    paths: np.ndarray,
    labels: np.ndarray,
    original_labels: np.ndarray,
    track_ids: np.ndarray,
    segment_indices: np.ndarray,
    row_regions: np.ndarray,
    class_names: np.ndarray,
) -> None:
    fieldnames = [
        "feature_index",
        "path",
        "label",
        "class_name",
        "original_label",
        "track_id",
        "segment_index",
        "region",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(len(labels)):
            label = int(labels[row_index])
            writer.writerow(
                {
                    "feature_index": row_index,
                    "path": str(paths[row_index]),
                    "label": label,
                    "class_name": str(class_names[label]),
                    "original_label": int(original_labels[row_index]),
                    "track_id": int(track_ids[row_index]),
                    "segment_index": int(segment_indices[row_index]),
                    "region": str(row_regions[row_index]),
                }
            )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    western = load_npz(args.western_features)
    eastern = load_npz(args.eastern_features)

    western_features = get_required(western, "features", args.western_features).astype(np.float32)
    eastern_features = get_required(eastern, "features", args.eastern_features).astype(np.float32)
    validate_compatible_features(western_features, eastern_features)

    western_labels = get_required(western, "labels", args.western_features).astype(np.int64)
    eastern_labels = get_required(eastern, "labels", args.eastern_features).astype(np.int64)
    western_class_names = as_string_array(get_required(western, "class_names", args.western_features))
    eastern_class_names = as_string_array(get_required(eastern, "class_names", args.eastern_features))

    eastern_label_offset = len(western_class_names)
    mixed_labels = np.concatenate([western_labels, eastern_labels + eastern_label_offset])
    original_labels = np.concatenate([western_labels, eastern_labels])
    class_names = np.concatenate([western_class_names, eastern_class_names])
    class_regions = np.asarray(
        ["western"] * len(western_class_names) + ["eastern"] * len(eastern_class_names)
    )

    western_paths = as_string_array(
        western["paths"] if "paths" in western else np.asarray([""] * len(western_labels))
    )
    eastern_paths = as_string_array(
        eastern["paths"] if "paths" in eastern else np.asarray([""] * len(eastern_labels))
    )
    paths = np.concatenate([western_paths, eastern_paths])

    western_track_ids = get_track_ids(western, len(western_labels))
    eastern_track_ids = get_track_ids(eastern, len(eastern_labels))
    western_track_ids = offset_track_ids(western_track_ids, 0)
    eastern_track_ids = offset_track_ids(eastern_track_ids, int(np.unique(western_track_ids).size))
    track_ids = np.concatenate([western_track_ids, eastern_track_ids])

    segment_indices = np.concatenate(
        [
            get_segment_indices(western, len(western_labels)),
            get_segment_indices(eastern, len(eastern_labels)),
        ]
    )
    row_regions = np.asarray(["western"] * len(western_labels) + ["eastern"] * len(eastern_labels))

    mixed_features = np.concatenate([western_features, eastern_features], axis=0)
    np.savez_compressed(
        args.output_dir / "features.npz",
        features=mixed_features,
        labels=mixed_labels.astype(np.int64),
        original_labels=original_labels.astype(np.int64),
        paths=paths,
        track_ids=track_ids.astype(np.int64),
        segment_indices=segment_indices.astype(np.int64),
        row_regions=row_regions,
        class_names=class_names,
        class_regions=class_regions,
        western_class_names=western_class_names,
        eastern_class_names=eastern_class_names,
        western_features_path=np.asarray(str(args.western_features)),
        eastern_features_path=np.asarray(str(args.eastern_features)),
    )
    write_metadata(
        args.output_dir / "metadata.csv",
        paths=paths,
        labels=mixed_labels,
        original_labels=original_labels,
        track_ids=track_ids,
        segment_indices=segment_indices,
        row_regions=row_regions,
        class_names=class_names,
    )

    config = {
        "western_features": str(args.western_features),
        "eastern_features": str(args.eastern_features),
        "output_dir": str(args.output_dir),
        "feature_shape": list(mixed_features.shape),
        "num_western_rows": int(len(western_labels)),
        "num_eastern_rows": int(len(eastern_labels)),
        "num_western_tracks": int(np.unique(western_track_ids).size),
        "num_eastern_tracks": int(np.unique(eastern_track_ids).size),
        "num_classes": int(len(class_names)),
        "class_names": class_names.tolist(),
        "class_regions": class_regions.tolist(),
    }
    with (args.output_dir / "merge_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Saved mixed features to: {args.output_dir / 'features.npz'}")
    print(f"Feature shape: {mixed_features.shape}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

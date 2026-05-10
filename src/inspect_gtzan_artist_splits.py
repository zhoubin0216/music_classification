#!/usr/bin/env python3
"""Inspect group-disjoint splits for extracted MERT features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from train_classifier import (
    assert_group_disjoint,
    build_cv_splits,
    build_gtzan_artist_groups,
    load_grouped_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check artist/source group-filtered split sizes and leakage."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("datasets/features/mert_gtzan/features.npz"),
    )
    parser.add_argument(
        "--gtzan-artist-index",
        type=Path,
        default=Path("datasets/gtzan_meta/index.txt"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def class_counts(labels: np.ndarray, class_names: list[str], indices: np.ndarray) -> str:
    counts = np.bincount(labels[indices], minlength=len(class_names))
    return ", ".join(
        f"{class_name}:{int(count)}"
        for class_name, count in zip(class_names, counts)
        if count > 0
    )


def main() -> int:
    args = parse_args()
    examples, class_names = load_grouped_features(args.features)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    groups, group_summary = build_gtzan_artist_groups(examples, args.gtzan_artist_index)
    splits = build_cv_splits(labels, folds=args.folds, seed=args.seed, groups=groups)
    assert_group_disjoint(splits, groups)

    print(f"Loaded tracks: {len(examples)}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Total groups: {group_summary['num_groups']}")
    print(f"GTZAN artist groups: {group_summary['num_artist_groups']}")
    print(f"Fallback source groups: {group_summary['num_fallback_groups']}")
    print(f"Missing artist entries: {group_summary['num_missing_artist_entries']}")
    for fold_id, (train_indices, validation_indices, test_indices) in enumerate(splits, start=1):
        print()
        print(
            f"fold={fold_id} sizes: "
            f"train={len(train_indices)} val={len(validation_indices)} test={len(test_indices)}"
        )
        print(
            f"fold={fold_id} groups: "
            f"train={len(set(groups[train_indices]))} "
            f"val={len(set(groups[validation_indices]))} "
            f"test={len(set(groups[test_indices]))}"
        )
        print(f"fold={fold_id} train counts: {class_counts(labels, class_names, train_indices)}")
        print(f"fold={fold_id} val counts:   {class_counts(labels, class_names, validation_indices)}")
        print(f"fold={fold_id} test counts:  {class_counts(labels, class_names, test_indices)}")
    print()
    print("No group overlap found across train/validation/test within any fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

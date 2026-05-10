#!/usr/bin/env python3
"""Train an MLP genre classifier on frozen MERT features.

The expected input is the ``features.npz`` written by extract_mert_features.py.
For the intended 10-second setup, first run feature extraction with:

    python3 src/extract_mert_features.py --chunk-seconds 10 --save-chunks \
        --output-dir datasets/features/mert_gtzan_10s

This script groups chunk-level feature rows by track_id, then performs 5-fold
cross validation. For each fold, 3 folds are used for training, 1 for
validation, and 1 for testing, giving an overall 6:2:2 split. By default, the
fold models are used only for validation/testing estimates; the saved model is
trained once at the end on all tracks for the median best epoch from CV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_FEATURES_PATH = "datasets/features/chunked_mert_gtzan/features.npz"
DEFAULT_OUTPUT_DIR = "outputs/mlp_chunked_mert_gtzan"


@dataclass(frozen=True)
class TrackExample:
    features: np.ndarray
    label: int
    path: str
    track_id: int


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray


class TrackFeatureDataset(Dataset):
    def __init__(self, examples: list[TrackExample], indices: Iterable[int]) -> None:
        self.examples = examples
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example_index = self.indices[item]
        example = self.examples[example_index]
        features = torch.from_numpy(example.features).float()
        label = torch.tensor(example.label, dtype=torch.long)
        index = torch.tensor(example_index, dtype=torch.long)
        return features, label, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP classifier with weighted aggregation on MERT features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(DEFAULT_FEATURES_PATH),
        help="Path to features.npz from extract_mert_features.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory where models, metrics, and predictions will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--aggregation-hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gtzan-artist-index",
        type=Path,
        default=None,
        help=(
            "Optional GTZAN index.txt with 'filename ::: artist ::: title'. "
            "When set, CV uses artist-disjoint StratifiedGroupKFold."
        ),
    )
    parser.add_argument(
        "--save-fold-artifacts",
        action="store_true",
        help="Save per-fold models, reports, histories, and confusion matrices.",
    )
    parser.add_argument(
        "--save-cv-predictions",
        action="store_true",
        help="Save per-track train/validation/test predictions from all CV folds.",
    )
    parser.add_argument(
        "--train-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train and save final_model.pt on all tracks after cross-validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Feature arrays are small, so 0 is usually enough.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:0, mps, or cpu.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_grouped_features(features_path: Path) -> tuple[list[TrackExample], list[str]]:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {features_path}")

    data = np.load(features_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    paths = (
        data["paths"].astype(str)
        if "paths" in data.files
        else np.asarray([""] * len(labels))
    )
    class_names = (
        [str(name) for name in data["class_names"].tolist()]
        if "class_names" in data.files
        else [str(label) for label in sorted(np.unique(labels))]
    )

    if features.ndim < 2:
        raise ValueError(
            f"Expected feature array with at least 2 dims, got {features.shape}"
        )
    if len(features) != len(labels):
        raise ValueError(
            f"features and labels length mismatch: {len(features)} vs {len(labels)}"
        )

    if "track_ids" in data.files:
        track_ids = data["track_ids"].astype(np.int64)
    else:
        track_ids = np.arange(len(labels), dtype=np.int64)

    if "segment_indices" in data.files:
        segment_indices = data["segment_indices"].astype(np.int64)
    else:
        segment_indices = np.zeros(len(labels), dtype=np.int64)

    grouped_rows: dict[int, list[int]] = {}
    track_order: list[int] = []
    for row_index, track_id in enumerate(track_ids):
        track_id_int = int(track_id)
        if track_id_int not in grouped_rows:
            grouped_rows[track_id_int] = []
            track_order.append(track_id_int)
        grouped_rows[track_id_int].append(row_index)

    examples: list[TrackExample] = []
    for track_id in track_order:
        row_indices = np.asarray(grouped_rows[track_id], dtype=np.int64)
        row_indices = row_indices[np.argsort(segment_indices[row_indices])]
        unique_labels = np.unique(labels[row_indices])
        if len(unique_labels) != 1:
            raise ValueError(
                f"Track {track_id} has inconsistent labels: {unique_labels}"
            )

        examples.append(
            TrackExample(
                features=features[row_indices],
                label=int(unique_labels[0]),
                path=str(paths[row_indices[0]]),
                track_id=track_id,
            )
        )

    return examples, class_names


def fit_standardizer(
    examples: list[TrackExample], train_indices: np.ndarray
) -> Standardizer:
    train_frames = np.concatenate(
        [
            examples[index].features.reshape(-1, examples[index].features.shape[-1])
            for index in train_indices
        ],
        axis=0,
    )
    mean = train_frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_frames.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return Standardizer(mean=mean, std=std)


def apply_standardizer(
    examples: list[TrackExample], standardizer: Standardizer
) -> list[TrackExample]:
    scaled: list[TrackExample] = []
    for example in examples:
        features = ((example.features - standardizer.mean) / standardizer.std).astype(
            np.float32
        )
        scaled.append(
            TrackExample(
                features=features,
                label=example.label,
                path=example.path,
                track_id=example.track_id,
            )
        )
    return scaled


def collate_tracks(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels, indices = zip(*batch)
    max_chunks = max(feature.shape[0] for feature in features)
    trailing_shape = features[0].shape[1:]

    padded = torch.zeros(
        (len(features), max_chunks, *trailing_shape), dtype=torch.float32
    )
    mask = torch.zeros((len(features), max_chunks), dtype=torch.bool)

    for row, feature in enumerate(features):
        num_chunks = feature.shape[0]
        padded[row, :num_chunks] = feature
        mask[row, :num_chunks] = True

    return padded, torch.stack(labels), mask, torch.stack(indices)


class LayerWeightedAggregation(nn.Module):
    """Learn a weighted average across MERT hidden-state layers."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, chunks, layers, dim]
        layer_weights = torch.softmax(self.layer_logits, dim=0)
        return (features * layer_weights.view(1, 1, -1, 1)).sum(dim=2)


class ChunkWeightedAggregation(nn.Module):
    """Learn attention weights across 10-second chunks from the same track."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features: [batch, chunks, dim], mask: [batch, chunks]
        scores = self.scorer(features).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class MLPGenreClassifier(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        hidden_dim: int,
        aggregation_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if len(input_shape) == 1:
            self.layer_aggregation = None
            input_dim = input_shape[0]
        elif len(input_shape) == 2:
            self.layer_aggregation = LayerWeightedAggregation(num_layers=input_shape[0])
            input_dim = input_shape[1]
        else:
            raise ValueError(
                "Expected per-chunk features shaped [dim] or [layers, dim], "
                f"got {input_shape}"
            )

        self.chunk_aggregation = ChunkWeightedAggregation(
            input_dim=input_dim,
            hidden_dim=aggregation_hidden_dim,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, num_classes)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, num_classes), num_classes),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_aggregation is not None:
            features = self.layer_aggregation(features)
        pooled, chunk_weights = self.chunk_aggregation(features, mask)
        logits = self.classifier(pooled)
        return logits, chunk_weights


def make_loader(
    examples: list[TrackExample],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TrackFeatureDataset(examples, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tracks,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []

    for features, labels, mask, indices in loader:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits, _ = model(features, mask)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())
        all_indices.append(indices.numpy())

    average_loss = total_loss / max(total_examples, 1)
    return (
        average_loss,
        np.concatenate(all_labels),
        np.concatenate(all_predictions),
        np.concatenate(all_indices),
    )


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


def save_confusion_matrix(
    path: Path,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(class_names)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label/pred_label", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])


def write_predictions(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    fieldnames = [
        "fold",
        "split",
        "track_index",
        "track_id",
        "path",
        "true_label",
        "true_name",
        "pred_label",
        "pred_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_gtzan_artist_index(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        raise FileNotFoundError(f"GTZAN artist index does not exist: {index_path}")

    filename_to_artist: dict[str, str] = {}
    with index_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(":::")]
            if len(parts) < 3:
                continue
            filename, artist = parts[0], parts[1]
            if filename and artist:
                filename_to_artist[filename] = artist
    if not filename_to_artist:
        raise RuntimeError(f"No filename/artist entries found in: {index_path}")
    return filename_to_artist


def filename_from_path(path: str) -> str:
    return Path(path).name


def build_gtzan_artist_groups(
    examples: list[TrackExample],
    index_path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    filename_to_artist = parse_gtzan_artist_index(index_path)
    groups: list[str] = []
    missing_filenames: list[str] = []

    for example in examples:
        filename = filename_from_path(example.path)
        artist = filename_to_artist.get(filename)
        if artist is None:
            missing_filenames.append(filename)
            artist = f"__unknown_artist__::{filename}"
        groups.append(artist)

    groups_array = np.asarray(groups)
    summary = {
        "index_path": str(index_path),
        "num_examples": len(examples),
        "num_artist_groups": int(np.unique(groups_array).size),
        "num_missing_artist_entries": len(missing_filenames),
        "missing_artist_filenames": missing_filenames,
    }
    return groups_array, summary


def build_cv_splits(
    labels: np.ndarray,
    folds: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if groups is None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        heldout_folds = [
            test_index for _, test_index in splitter.split(np.zeros(len(labels)), labels)
        ]
    else:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        heldout_folds = [
            test_index
            for _, test_index in splitter.split(
                np.zeros(len(labels)),
                labels,
                groups=groups,
            )
        ]

    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(labels))
    for fold_id in range(folds):
        test_indices = np.sort(heldout_folds[fold_id])
        validation_indices = np.sort(heldout_folds[(fold_id + 1) % folds])
        excluded = np.concatenate([test_indices, validation_indices])
        train_indices = np.setdiff1d(all_indices, excluded, assume_unique=False)
        splits.append((train_indices, validation_indices, test_indices))
    return splits


def assert_group_disjoint(
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    groups: np.ndarray,
) -> None:
    for fold_id, (train_indices, validation_indices, test_indices) in enumerate(splits, start=1):
        train_groups = set(groups[train_indices].tolist())
        validation_groups = set(groups[validation_indices].tolist())
        test_groups = set(groups[test_indices].tolist())
        if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
            raise RuntimeError(f"Artist-group leakage detected in fold {fold_id}")


def train_fold(
    fold_id: int,
    examples: list[TrackExample],
    class_names: list[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    fold_dir = args.output_dir / f"fold_{fold_id}"
    if args.save_fold_artifacts:
        fold_dir.mkdir(parents=True, exist_ok=True)

    standardizer = fit_standardizer(examples, train_indices)
    scaled_examples = apply_standardizer(examples, standardizer)

    input_shape = tuple(scaled_examples[0].features.shape[1:])
    model = MLPGenreClassifier(
        input_shape=input_shape,
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        aggregation_hidden_dim=args.aggregation_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = make_loader(
        scaled_examples,
        train_indices,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = make_loader(
        scaled_examples,
        validation_indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        scaled_examples,
        test_indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_labels, train_predictions, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        validation_loss, validation_labels, validation_predictions, _ = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        train_metrics = metric_dict(train_labels, train_predictions)
        validation_metrics = metric_dict(validation_labels, validation_predictions)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"fold={fold_id} epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={validation_loss:.4f} val_f1={validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"fold={fold_id} early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold_id} did not produce a best model state")

    model.load_state_dict(best_state)
    train_loss, train_labels, train_predictions, train_track_indices = run_epoch(
        model,
        train_loader,
        criterion,
        device,
    )
    (
        validation_loss,
        validation_labels,
        validation_predictions,
        validation_track_indices,
    ) = run_epoch(
        model,
        validation_loader,
        criterion,
        device,
    )
    test_loss, test_labels, test_predictions, test_track_indices = run_epoch(
        model,
        test_loader,
        criterion,
        device,
    )

    split_outputs = {
        "train": (train_loss, train_labels, train_predictions, train_track_indices),
        "validation": (
            validation_loss,
            validation_labels,
            validation_predictions,
            validation_track_indices,
        ),
        "test": (test_loss, test_labels, test_predictions, test_track_indices),
    }

    fold_metrics: dict[str, object] = {
        "fold": fold_id,
        "best_epoch": best_epoch,
        "input_shape": list(input_shape),
        "num_train": int(len(train_indices)),
        "num_validation": int(len(validation_indices)),
        "num_test": int(len(test_indices)),
    }

    prediction_rows: list[dict[str, object]] = []
    for split_name, (loss, labels, predictions, track_indices) in split_outputs.items():
        split_metrics = metric_dict(labels, predictions)
        split_metrics["loss"] = float(loss)
        fold_metrics[split_name] = split_metrics

        for track_index, true_label, pred_label in zip(
            track_indices, labels, predictions
        ):
            example = examples[int(track_index)]
            prediction_rows.append(
                {
                    "fold": fold_id,
                    "split": split_name,
                    "track_index": int(track_index),
                    "track_id": example.track_id,
                    "path": example.path,
                    "true_label": int(true_label),
                    "true_name": class_names[int(true_label)],
                    "pred_label": int(pred_label),
                    "pred_name": class_names[int(pred_label)],
                }
            )

    if args.save_fold_artifacts:
        save_confusion_matrix(
            fold_dir / "test_confusion_matrix.csv",
            test_labels,
            test_predictions,
            class_names,
        )
        with (fold_dir / "classification_report.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                classification_report(
                    test_labels,
                    test_predictions,
                    labels=np.arange(len(class_names)),
                    target_names=class_names,
                    zero_division=0,
                    output_dict=True,
                ),
                handle,
                indent=2,
            )
        with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(fold_metrics, handle, indent=2)
        with (fold_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        torch.save(
            {
                "model_state_dict": best_state,
                "class_names": class_names,
                "input_shape": input_shape,
                "standardizer_mean": standardizer.mean,
                "standardizer_std": standardizer.std,
                "args": vars(args),
            },
            fold_dir / "best_model.pt",
        )

    print(
        f"fold={fold_id} done "
        f"test_acc={fold_metrics['test']['accuracy']:.4f} "
        f"test_macro_f1={fold_metrics['test']['macro_f1']:.4f}"
    )
    return fold_metrics, prediction_rows


def train_final_model(
    examples: list[TrackExample],
    class_names: list[str],
    final_epochs: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    all_indices = np.arange(len(examples))
    standardizer = fit_standardizer(examples, all_indices)
    scaled_examples = apply_standardizer(examples, standardizer)
    input_shape = tuple(scaled_examples[0].features.shape[1:])

    model = MLPGenreClassifier(
        input_shape=input_shape,
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        aggregation_hidden_dim=args.aggregation_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader = make_loader(
        scaled_examples,
        all_indices,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, final_epochs + 1):
        train_loss, train_labels, train_predictions, _ = run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        train_metrics = metric_dict(train_labels, train_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
            }
        )
        print(
            f"final epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['macro_f1']:.4f}"
        )

    final_model_path = args.output_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "input_shape": input_shape,
            "standardizer_mean": standardizer.mean,
            "standardizer_std": standardizer.std,
            "feature_path": str(args.features),
            "final_epochs": final_epochs,
            "args": vars(args),
        },
        final_model_path,
    )
    with (args.output_dir / "final_training_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(history, handle, indent=2)

    return {
        "model_path": str(final_model_path),
        "final_epochs": final_epochs,
        "input_shape": list(input_shape),
        "num_train": len(examples),
        "last_train_accuracy": history[-1]["train_accuracy"] if history else None,
        "last_train_macro_f1": history[-1]["train_macro_f1"] if history else None,
    }


def summarize(
    fold_metrics: list[dict[str, object]], include_folds: bool = False
) -> dict[str, object]:
    test_accuracies = np.asarray(
        [fold["test"]["accuracy"] for fold in fold_metrics], dtype=np.float64
    )
    test_macro_f1 = np.asarray(
        [fold["test"]["macro_f1"] for fold in fold_metrics], dtype=np.float64
    )
    validation_macro_f1 = np.asarray(
        [fold["validation"]["macro_f1"] for fold in fold_metrics],
        dtype=np.float64,
    )
    best_epochs = np.asarray(
        [fold["best_epoch"] for fold in fold_metrics], dtype=np.int64
    )

    summary: dict[str, object] = {
        "num_folds": len(fold_metrics),
        "test_accuracy_mean": float(test_accuracies.mean()),
        "test_accuracy_std": float(test_accuracies.std(ddof=0)),
        "test_macro_f1_mean": float(test_macro_f1.mean()),
        "test_macro_f1_std": float(test_macro_f1.std(ddof=0)),
        "validation_macro_f1_mean": float(validation_macro_f1.mean()),
        "validation_macro_f1_std": float(validation_macro_f1.std(ddof=0)),
        "median_best_epoch": int(np.median(best_epochs)),
    }
    if include_folds:
        summary["folds"] = fold_metrics
    return summary


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    examples, class_names = load_grouped_features(args.features)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    chunk_counts = np.asarray(
        [example.features.shape[0] for example in examples], dtype=np.int64
    )

    print(f"Loaded {len(examples)} tracks from {args.features}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(
        "Chunks per track: "
        f"min={chunk_counts.min()}, median={np.median(chunk_counts):.1f}, max={chunk_counts.max()}"
    )
    print(f"Using device: {device}")

    if args.folds != 5:
        print(
            f"Warning: --folds {args.folds} changes the requested 5-fold 6:2:2 protocol."
        )

    groups: np.ndarray | None = None
    group_summary: dict[str, object] | None = None
    if args.gtzan_artist_index is not None:
        groups, group_summary = build_gtzan_artist_groups(examples, args.gtzan_artist_index)
        print(
            "Using artist-filtered GTZAN splits: "
            f"{group_summary['num_artist_groups']} artist groups, "
            f"{group_summary['num_missing_artist_entries']} missing index entries"
        )

    splits = build_cv_splits(labels, folds=args.folds, seed=args.seed, groups=groups)
    if groups is not None:
        assert_group_disjoint(splits, groups)

    all_fold_metrics: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []

    for fold_id, (train_indices, validation_indices, test_indices) in enumerate(
        splits, start=1
    ):
        print(
            f"fold={fold_id} split sizes: "
            f"train={len(train_indices)} validation={len(validation_indices)} test={len(test_indices)}"
        )
        fold_metrics, prediction_rows = train_fold(
            fold_id=fold_id,
            examples=examples,
            class_names=class_names,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            args=args,
            device=device,
        )
        all_fold_metrics.append(fold_metrics)
        all_prediction_rows.extend(prediction_rows)

    summary = summarize(all_fold_metrics, include_folds=args.save_fold_artifacts)
    if group_summary is not None:
        summary["group_split"] = group_summary
        summary["split_strategy"] = "stratified_group_kfold_by_artist"
    else:
        summary["split_strategy"] = "stratified_kfold_random_tracks"

    if args.train_final_model:
        final_epochs = max(1, int(summary["median_best_epoch"]))
        summary["final_model"] = train_final_model(
            examples=examples,
            class_names=class_names,
            final_epochs=final_epochs,
            args=args,
            device=device,
        )

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if args.save_cv_predictions:
        write_predictions(args.output_dir / "cv_predictions.csv", all_prediction_rows)

    print(
        "Cross-validation summary: "
        f"test_acc={summary['test_accuracy_mean']:.4f}+/-{summary['test_accuracy_std']:.4f}, "
        f"test_macro_f1={summary['test_macro_f1_mean']:.4f}+/-{summary['test_macro_f1_std']:.4f}"
    )
    print(f"Saved results to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

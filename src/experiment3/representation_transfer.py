#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from train_classifier import (
    LayerWeightedAggregation,
    ChunkWeightedAggregation,
    MLPGenreClassifier,
)
import types
import pathlib


fake_module = types.ModuleType("pathlib._local")

fake_module.PosixPath = pathlib.PosixPath
fake_module.WindowsPath = pathlib.WindowsPath

sys.modules["pathlib._local"] = fake_module

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.model_selection import StratifiedGroupKFold

from torch import nn
from torch.utils.data import DataLoader, Dataset



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class TrackExample:
    features: np.ndarray
    label: int
    path: str
    track_id: int
    group_id: str


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray


def infer_group_id(path: str):
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def load_grouped_features(features_path: Path):
    data = np.load(features_path, allow_pickle=True)

    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    paths = data["paths"].astype(str)
    class_names = [str(x) for x in data["class_names"]]

    if "track_ids" in data.files:
        track_ids = data["track_ids"].astype(np.int64)
    else:
        track_ids = np.arange(len(labels))

    if "segment_indices" in data.files:
        segment_indices = data["segment_indices"].astype(np.int64)
    else:
        segment_indices = np.zeros(len(labels), dtype=np.int64)

    grouped = {}

    for idx, track_id in enumerate(track_ids):
        grouped.setdefault(int(track_id), []).append(idx)

    examples = []

    for track_id, row_indices in grouped.items():
        row_indices = np.asarray(row_indices)
        row_indices = row_indices[np.argsort(segment_indices[row_indices])]

        examples.append(
            TrackExample(
                features=features[row_indices],
                label=int(labels[row_indices[0]]),
                path=str(paths[row_indices[0]]),
                track_id=int(track_id),
                group_id=infer_group_id(str(paths[row_indices[0]])),
            )
        )

    return examples, class_names


class TrackDataset(Dataset):
    def __init__(self, examples, indices):
        self.examples = examples
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex = self.examples[self.indices[idx]]
        return (
            torch.from_numpy(ex.features).float(),
            torch.tensor(ex.label),
            torch.tensor(self.indices[idx]),
        )


def collate_tracks(batch):
    features, labels, indices = zip(*batch)

    max_chunks = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[-1]

    padded = torch.zeros(len(features), max_chunks, feature_dim)
    mask = torch.zeros(len(features), max_chunks, dtype=torch.bool)

    for i, feat in enumerate(features):
        padded[i, : feat.shape[0]] = feat
        mask[i, : feat.shape[0]] = True

    return padded, torch.stack(labels), mask, torch.stack(indices)


def fit_standardizer(examples, train_indices):
    train_features = np.concatenate(
        [examples[i].features.reshape(-1, examples[i].features.shape[-1]) for i in train_indices],
        axis=0,
    )

    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).astype(np.float32)

    std = np.maximum(std, 1e-6)

    return Standardizer(mean, std)


def apply_standardizer(examples, standardizer):
    scaled = []

    for ex in examples:
        features = ((ex.features - standardizer.mean) / standardizer.std).astype(np.float32)

        scaled.append(
            TrackExample(
                features=features,
                label=ex.label,
                path=ex.path,
                track_id=ex.track_id,
                group_id=ex.group_id,
            )
        )

    return scaled


class ChunkAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, mask):
        scores = self.scorer(x).squeeze(-1)

        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1)

        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)

        return pooled, weights


class EasternBaselineClassifier(nn.Module):

    def __init__(
        self,
        input_shape,
        num_classes,
        hidden_dim=64,
        aggregation_hidden_dim=16,
        dropout=0.5,
    ):
        super().__init__()

        if len(input_shape) == 1:
            self.layer_aggregation = None
            input_dim = input_shape[0]

        elif len(input_shape) == 2:
            self.layer_aggregation = LayerWeightedAggregation(
                num_layers=input_shape[0]
            )
            input_dim = input_shape[1]

        else:
            raise ValueError(
                f"Unexpected input shape: {input_shape}"
            )

        self.chunk_aggregation = ChunkWeightedAggregation(
            input_dim=input_dim,
            hidden_dim=aggregation_hidden_dim,
        )

        self.backbone = nn.Sequential(

            nn.LayerNorm(input_dim),

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                max(hidden_dim // 2, num_classes),
            ),

            nn.ReLU(),
            nn.Dropout(dropout),
        )

        latent_dim = max(hidden_dim // 2, num_classes)

        self.classifier = nn.Linear(
            latent_dim,
            num_classes,
        )

    def forward(self, features, mask):

        if self.layer_aggregation is not None:
            features = self.layer_aggregation(features)

        pooled, chunk_weights = self.chunk_aggregation(
            features,
            mask,
        )

        latent = self.backbone(pooled)

        logits = self.classifier(latent)

        return logits, latent, pooled, chunk_weights


class TransferClassifier(nn.Module):

    def __init__(
        self,
        western_checkpoint,
        num_classes=8,
        freeze_backbone=True,
    ):
        super().__init__()

        checkpoint = torch.load(
            western_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        hidden_dim = checkpoint["args"]["hidden_dim"]
        aggregation_hidden_dim = checkpoint["args"]["aggregation_hidden_dim"]
        dropout = checkpoint["args"]["dropout"]

        input_shape = tuple(checkpoint["input_shape"])

        western_model = MLPGenreClassifier(
            input_shape=input_shape,
            num_classes=len(checkpoint["class_names"]),
            hidden_dim=hidden_dim,
            aggregation_hidden_dim=aggregation_hidden_dim,
            dropout=dropout,
        )

        western_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.layer_aggregation = western_model.layer_aggregation

        self.chunk_aggregation = western_model.chunk_aggregation

        self.backbone = western_model.classifier[:-1]

        latent_dim = max(
            hidden_dim // 2,
            len(checkpoint["class_names"]),
        )

        self.classifier = nn.Linear(
            latent_dim,
            num_classes,
        )

        if freeze_backbone:

            if self.layer_aggregation is not None:
                for p in self.layer_aggregation.parameters():
                    p.requires_grad = False

            for p in self.chunk_aggregation.parameters():
                p.requires_grad = False

            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, features, mask):

        if self.layer_aggregation is not None:
            features = self.layer_aggregation(features)

        pooled, chunk_weights = self.chunk_aggregation(
            features,
            mask,
        )

        latent = self.backbone(pooled)

        logits = self.classifier(latent)

        return logits, latent, pooled, chunk_weights


def metric_dict(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None

    model.train(is_training)

    total_loss = 0

    all_labels = []
    all_predictions = []

    latent_features = []
    pooled_features = []
    chunk_weights_all = []

    for features, labels, mask, _ in loader:
        features = features.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        if is_training:
            optimizer.zero_grad()

        logits, latent, pooled, chunk_weights = model(features, mask)

        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)

        total_loss += float(loss.item()) * labels.size(0)

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

        latent_features.append(latent.detach().cpu().numpy())
        pooled_features.append(pooled.detach().cpu().numpy())
        chunk_weights_all.append(chunk_weights.detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)
    predictions_np = np.concatenate(all_predictions)

    metrics = metric_dict(labels_np, predictions_np)

    metrics["loss"] = total_loss / len(labels_np)

    return (
        metrics,
        labels_np,
        predictions_np,
        np.concatenate(latent_features),
        np.concatenate(pooled_features),
        chunk_weights_all,
    )


def build_splits(examples, folds=5):
    labels = np.asarray([x.label for x in examples])
    groups = np.asarray([x.group_id for x in examples])

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    heldout = []

    for _, test_idx in splitter.split(np.zeros(len(labels)), labels, groups):
        heldout.append(test_idx)

    splits = []

    all_indices = np.arange(len(labels))

    for fold_id in range(folds):
        test_idx = heldout[fold_id]
        val_idx = heldout[(fold_id + 1) % folds]

        excluded = np.concatenate([test_idx, val_idx])

        train_idx = np.setdiff1d(all_indices, excluded)

        splits.append((train_idx, val_idx, test_idx))

    return splits


def train_experiment(
    experiment_name,
    model,
    examples,
    class_names,
    splits,
    output_dir,
    device,
    epochs=50,
):
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    all_fold_results = []

    for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits):
        fold_dir = experiment_dir / f"fold_{fold_id+1}"
        fold_dir.mkdir(exist_ok=True)

        standardizer = fit_standardizer(examples, train_idx)
        scaled_examples = apply_standardizer(examples, standardizer)

        train_loader = DataLoader(
            TrackDataset(scaled_examples, train_idx),
            batch_size=32,
            shuffle=True,
            collate_fn=collate_tracks,
        )

        val_loader = DataLoader(
            TrackDataset(scaled_examples, val_idx),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_tracks,
        )

        test_loader = DataLoader(
            TrackDataset(scaled_examples, test_idx),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_tracks,
        )

        model_fold = copy.deepcopy(model).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model_fold.parameters()),
            lr=1e-3,
        )

        best_state = None
        best_f1 = -1

        history = []

        for epoch in range(epochs):
            train_metrics, *_ = run_epoch(
                model_fold,
                train_loader,
                criterion,
                device,
                optimizer,
            )

            val_outputs = run_epoch(
                model_fold,
                val_loader,
                criterion,
                device,
            )

            val_metrics = val_outputs[0]

            history.append(
                {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "validation": val_metrics,
                }
            )

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                best_state = copy.deepcopy(model_fold.state_dict())

        model_fold.load_state_dict(best_state)

        test_outputs = run_epoch(
            model_fold,
            test_loader,
            criterion,
            device,
        )

        (
            test_metrics,
            test_labels,
            test_predictions,
            latent_features,
            pooled_features,
            chunk_weights,
        ) = test_outputs

        all_fold_results.append(test_metrics)

        torch.save(
            {
                "model_state_dict": model_fold.state_dict(),
                "class_names": class_names,
                "standardizer_mean": standardizer.mean,
                "standardizer_std": standardizer.std,
            },
            fold_dir / "best_model.pt",
        )

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        with open(fold_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        report = classification_report(
            test_labels,
            test_predictions,
            target_names=class_names,
            output_dict=True,
        )

        with open(fold_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        matrix = confusion_matrix(test_labels, test_predictions)

        np.savetxt(
            fold_dir / "confusion_matrix.csv",
            matrix,
            delimiter=",",
            fmt="%d",
        )

        np.savez_compressed(
            fold_dir / "analysis_features.npz",
            latent_features=latent_features,
            pooled_features=pooled_features,
            chunk_weights=np.asarray(chunk_weights, dtype=object),
            labels=test_labels,
            predictions=test_predictions,
            class_names=np.asarray(class_names),
        )

        with open(fold_dir / "predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "path",
                "true_label",
                "predicted_label",
            ])

            for idx, pred in zip(test_idx, test_predictions):
                writer.writerow(
                    [
                        examples[idx].path,
                        class_names[examples[idx].label],
                        class_names[pred],
                    ]
                )

    summary = {
        "accuracy_mean": float(np.mean([x["accuracy"] for x in all_fold_results])),
        "accuracy_std": float(np.std([x["accuracy"] for x in all_fold_results])),
        "macro_f1_mean": float(np.mean([x["macro_f1"] for x in all_fold_results])),
        "macro_f1_std": float(np.std([x["macro_f1"] for x in all_fold_results])),
    }

    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eastern-features", type=Path, required=True)
    parser.add_argument("--western-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment3"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    set_seed(42)

    device = torch.device(args.device)

    examples, class_names = load_grouped_features(args.eastern_features)

    splits = build_splits(examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = []

    for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits):
        for split_name, indices in zip(
            ["train", "validation", "test"],
            [train_idx, val_idx, test_idx],
        ):
            for idx in indices:
                ex = examples[idx]

                split_rows.append(
                    {
                        "fold": fold_id + 1,
                        "split": split_name,
                        "track_id": ex.track_id,
                        "group_id": ex.group_id,
                        "path": ex.path,
                        "label": ex.label,
                        "genre": class_names[ex.label],
                    }
                )

    with open(args.output_dir / "shared_splits.json", "w") as f:
        json.dump(split_rows, f, indent=2)

    transfer_model = TransferClassifier(
        args.western_checkpoint,
        num_classes=len(class_names),
        freeze_backbone=True,
    )

    train_experiment(
        experiment_name="western_transfer",
        model=transfer_model,
        examples=examples,
        class_names=class_names,
        splits=splits,
        output_dir=args.output_dir,
        device=device,
    )

    input_shape = tuple(examples[0].features.shape[1:])

    baseline_model = EasternBaselineClassifier(
        input_shape=input_shape,
        num_classes=len(class_names),
    )

    train_experiment(
        experiment_name="eastern_baseline",
        model=baseline_model,
        examples=examples,
        class_names=class_names,
        splits=splits,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
# ```
#
# ```bash
# python src/train_eastern_transfer.py \
#   --eastern-features datasets/features/mert_eastern/features.npz \
#   --western-checkpoint outputs/mlp_mert_gtzan_artist_filtered/final_model.pt \
#   --output-dir outputs/cross_cultural_transfer
# ```
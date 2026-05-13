#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Labels
# =========================================================

TRADITIONAL_CLASSES = {
    "classical",
    "carnatic",
    "enka",
    "guqinguzheng",
    "jingju",
    "sitar",
}

VOCAL_CLASSES = {
    "blues",
    "country",
    "disco",
    "hiphop",
    "metal",
    "pop",
    "reggae",
    "rock",
    "carnatic",
    "cpop",
    "enka",
    "jingju",
    "jpop",
    "kpop",
}

HARD_CONFUSION_PAIRS = {

    ("rock", "metal"),
    ("enka", "sitar"),
    ("enka", "cpop"),
    ("sitar", "enka"),
    ("jpop", "cpop"),
}


# =========================================================
# Utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Dataset
# =========================================================

@dataclass
class TrackExample:
    features: np.ndarray
    label: int
    track_id: int
    path: str


class TrackDataset(Dataset):

    def __init__(
        self,
        examples,
        class_names,
    ):
        self.examples = examples
        self.class_names = class_names

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        ex = self.examples[idx]

        genre_label = ex.label

        class_name = self.class_names[genre_label]

        traditional_label = int(
            class_name in TRADITIONAL_CLASSES
        )

        vocal_label = int(
            class_name in VOCAL_CLASSES
        )

        return {
            "features": torch.tensor(
                ex.features,
                dtype=torch.float32,
            ),
            "genre_label": torch.tensor(
                genre_label,
                dtype=torch.long,
            ),
            "traditional_label": torch.tensor(
                traditional_label,
                dtype=torch.long,
            ),
            "vocal_label": torch.tensor(
                vocal_label,
                dtype=torch.long,
            ),
            "track_id": ex.track_id,
            "path": ex.path,
        }


# =========================================================
# Collate
# =========================================================

def collate_fn(batch):

    lengths = [x["features"].shape[0] for x in batch]

    max_len = max(lengths)

    feature_dim = batch[0]["features"].shape[-1]

    padded = torch.zeros(
        len(batch),
        max_len,
        feature_dim,
    )

    mask = torch.zeros(
        len(batch),
        max_len,
        dtype=torch.bool,
    )

    for i, item in enumerate(batch):

        length = item["features"].shape[0]

        padded[i, :length] = item["features"]

        mask[i, :length] = True

    return {
        "features": padded,
        "mask": mask,
        "genre_label": torch.stack(
            [x["genre_label"] for x in batch]
        ),
        "traditional_label": torch.stack(
            [x["traditional_label"] for x in batch]
        ),
        "vocal_label": torch.stack(
            [x["vocal_label"] for x in batch]
        ),
        "track_ids": [x["track_id"] for x in batch],
        "paths": [x["path"] for x in batch],
    }


# =========================================================
# Pooling
# =========================================================

class ChunkAttentionPooling(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        use_mean_std=True,
    ):
        super().__init__()

        self.use_mean_std = use_mean_std

        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, mask):

        scores = self.scorer(x).squeeze(-1)

        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)

        mean = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        if not self.use_mean_std:
            return mean, weights

        variance = torch.sum(
            weights.unsqueeze(-1) * (x - mean.unsqueeze(1)) ** 2,
            dim=1,
        )

        std = torch.sqrt(variance + 1e-6)

        pooled = torch.cat([mean, std], dim=-1)

        return pooled, weights


# =========================================================
# Adapter
# =========================================================

class BottleneckAdapter(nn.Module):

    def __init__(
        self,
        dim,
        bottleneck_dim=32,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, dim),
        )

        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):

        return x + self.gate * self.net(x)


# =========================================================
# Model
# =========================================================

class MultiTaskGenreClassifier(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        dropout=0.3,
        num_genres=18,
        use_mean_std_pooling=True,
        use_adapter=True,
        adapter_dim=32,
        use_multitask=True,
    ):
        super().__init__()

        self.use_multitask = use_multitask

        self.pooling = ChunkAttentionPooling(
            input_dim=input_dim,
            use_mean_std=use_mean_std_pooling,
        )

        pooled_dim = (
            input_dim * 2
            if use_mean_std_pooling
            else input_dim
        )

        self.use_adapter = use_adapter

        if use_adapter:
            self.adapter = BottleneckAdapter(
                pooled_dim,
                adapter_dim,
            )

        self.backbone = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        latent_dim = hidden_dim // 2

        self.genre_head = nn.Linear(
            latent_dim,
            num_genres,
        )

        if use_multitask:

            self.traditional_head = nn.Linear(
                latent_dim,
                2,
            )

            self.vocal_head = nn.Linear(
                latent_dim,
                2,
            )

    def forward(self, features, mask):

        pooled, attention_weights = self.pooling(
            features,
            mask,
        )

        if self.use_adapter:
            pooled = self.adapter(pooled)

        latent = self.backbone(pooled)

        genre_logits = self.genre_head(latent)

        outputs = {
            "genre_logits": genre_logits,
            "latent": latent,
            "pooled": pooled,
            "attention_weights": attention_weights,
        }

        if self.use_multitask:

            outputs["traditional_logits"] = self.traditional_head(latent)

            outputs["vocal_logits"] = self.vocal_head(latent)

        return outputs


# =========================================================
# Load NPZ
# =========================================================

def load_examples(npz_path):

    data = np.load(npz_path, allow_pickle=True)

    features = data["features"]
    labels = data["labels"]
    track_ids = data["track_ids"]

    if "paths" in data:
        paths = data["paths"]
    else:
        paths = np.asarray([""] * len(labels))

    class_names = data["class_names"].astype(str).tolist()

    examples = []

    for i in range(len(labels)):

        examples.append(
            TrackExample(
                features=features[i],
                label=int(labels[i]),
                track_id=int(track_ids[i]),
                path=str(paths[i]),
            )
        )

    return examples, class_names

# =========================================================
# Standardization
# =========================================================

def fit_standardizer(examples):

    frames = np.concatenate(
        [
            x.features.reshape(
                -1,
                x.features.shape[-1],
            )
            for x in examples
        ],
        axis=0,
    )

    mean = frames.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std = frames.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    std = np.maximum(std, 1e-6)

    return mean, std


def apply_standardizer(
    examples,
    mean,
    std,
):

    normalized = []

    for ex in examples:

        features = (
            (ex.features - mean)
            / std
        ).astype(np.float32)

        normalized.append(
            TrackExample(
                features=features,
                label=ex.label,
                track_id=ex.track_id,
                path=ex.path,
            )
        )

    return normalized


# =========================================================
# Train
# =========================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    multitask_weight,
    enable_hard_mining=False,
    hard_mining_weight=2.0,
):

    model.train()

    total_loss = 0.0

    all_labels = []
    all_preds = []

    for batch in loader:

        features = batch["features"].to(device)
        mask = batch["mask"].to(device)

        genre_labels = batch["genre_label"].to(device)
        traditional_labels = batch["traditional_label"].to(device)
        vocal_labels = batch["vocal_label"].to(device)

        optimizer.zero_grad()

        outputs = model(features, mask)

        per_sample_loss = F.cross_entropy(
            outputs["genre_logits"],
            genre_labels,
            reduction="none",
        )

        if enable_hard_mining:

            weights = torch.ones_like(
                per_sample_loss,
                device=device,
            )

            probs = torch.softmax(
                outputs["genre_logits"],
                dim=1,
            )

            for i in range(len(genre_labels)):

                true_idx = genre_labels[i].item()

                true_name = LABEL_TO_NAME[
                    true_idx
                ]

                for other_idx, other_name in LABEL_TO_NAME.items():

                    if other_idx == true_idx:
                        continue

                    if (
                            true_name,
                            other_name,
                    ) in HARD_CONFUSION_PAIRS:
                        confusion_prob = probs[
                            i,
                            other_idx
                        ]

                        weights[i] += (
                                confusion_prob
                                * (hard_mining_weight - 1.0)
                        )

            per_sample_loss = (
                    per_sample_loss * weights
            )

        genre_loss = per_sample_loss.mean()

        loss = genre_loss

        if model.use_multitask:

            traditional_loss = F.cross_entropy(
                outputs["traditional_logits"],
                traditional_labels,
            )

            vocal_loss = F.cross_entropy(
                outputs["vocal_logits"],
                vocal_labels,
            )

            loss = (
                genre_loss
                + multitask_weight * traditional_loss
                + multitask_weight * vocal_loss
            )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = outputs["genre_logits"].argmax(dim=1)

        all_labels.extend(
            genre_labels.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

    return {
        "loss": total_loss / len(loader),
        "f1": f1_score(
            all_labels,
            all_preds,
            average="macro",
        ),
        "accuracy": accuracy_score(
            all_labels,
            all_preds,
        ),
    }


# =========================================================
# Evaluate
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    class_names,
):

    model.eval()

    all_labels = []
    all_preds = []

    latent_features = []
    pooled_features = []

    for batch in loader:

        features = batch["features"].to(device)
        mask = batch["mask"].to(device)

        labels = batch["genre_label"].to(device)

        outputs = model(features, mask)

        preds = outputs["genre_logits"].argmax(dim=1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        latent_features.append(
            outputs["latent"].cpu().numpy()
        )

        pooled_features.append(
            outputs["pooled"].cpu().numpy()
        )

    latent_features = np.concatenate(latent_features)

    pooled_features = np.concatenate(pooled_features)

    return {
        "accuracy": accuracy_score(
            all_labels,
            all_preds,
        ),
        "macro_f1": f1_score(
            all_labels,
            all_preds,
            average="macro",
        ),
        "report": classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            all_labels,
            all_preds,
        ),
        "labels": np.asarray(all_labels),
        "predictions": np.asarray(all_preds),
        "latent_features": latent_features,
        "pooled_features": pooled_features,
    }


# =========================================================
# Split Reader
# =========================================================

def load_split_csv(csv_path):

    split_map = {}

    with open(csv_path, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:

            path = str(row["path"]).strip()

            split = str(row["split"]).strip()

            split_map[path] = split

    print(f"Loaded split definitions: {len(split_map)}")

    return split_map


# =========================================================
# Main
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--multitask-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--disable-multitask",
        action="store_true",
    )

    parser.add_argument(
        "--enable-hard-mining",
        action="store_true",
    )

    parser.add_argument(
        "--hard-mining-weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--disable-adapter",
        action="store_true",
    )

    parser.add_argument(
        "--disable-mean-std-pooling",
        action="store_true",
    )

    return parser.parse_args()


# =========================================================
# Run
# =========================================================

def main():
    args = parse_args()

    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples, class_names = load_examples(
        args.features
    )

    global LABEL_TO_NAME

    LABEL_TO_NAME = {
        i: name
        for i, name in enumerate(class_names)
    }

    split_map = load_split_csv(
        args.split_csv
    )

    train_examples = []
    val_examples = []
    test_examples = []

    missing_paths = []

    for ex in examples:

        path_key = str(ex.path).strip()

        split = split_map.get(path_key)

        if split is None:
            missing_paths.append(path_key)

            continue

        if split == "train":

            train_examples.append(ex)

        elif split == "validation":

            val_examples.append(ex)

        elif split == "test":

            test_examples.append(ex)

    if len(missing_paths) > 0:
        print("\nMissing paths in split csv:")
        print(missing_paths[:10])

        raise RuntimeError(
            f"{len(missing_paths)} samples missing split assignment"
        )

    print("\n===== Dataset Split =====")

    print(f"Train samples: {len(train_examples)}")
    print(f"Validation samples: {len(val_examples)}")
    print(f"Test samples: {len(test_examples)}")

    mean, std = fit_standardizer(
        train_examples
    )

    train_examples = apply_standardizer(
        train_examples,
        mean,
        std,
    )

    val_examples = apply_standardizer(
        val_examples,
        mean,
        std,
    )

    test_examples = apply_standardizer(
        test_examples,
        mean,
        std,
    )

    train_dataset = TrackDataset(
        train_examples,
        class_names,
    )

    val_dataset = TrackDataset(
        val_examples,
        class_names,
    )

    test_dataset = TrackDataset(
        test_examples,
        class_names,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = MultiTaskGenreClassifier(
        input_dim=train_examples[0].features.shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_genres=len(class_names),
        use_mean_std_pooling=not args.disable_mean_std_pooling,
        use_adapter=not args.disable_adapter,
        use_multitask=not args.disable_multitask,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    best_f1 = -1
    best_state = None

    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(args.epochs):

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.multitask_weight,
            enable_hard_mining=args.enable_hard_mining,
            hard_mining_weight=args.hard_mining_weight,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            class_names,
        )

        history.append({
            "epoch": epoch + 1,
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        })

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:

            best_f1 = val_metrics["macro_f1"]

            best_epoch = epoch + 1

            patience_counter = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:

            patience_counter += 1

        if patience_counter >= args.patience:
            print(
                f"\nEarly stopping triggered "
                f"at epoch {epoch + 1}. "
                f"Best epoch: {best_epoch} "
                f"(Val F1 = {best_f1:.4f})"
            )

            break

    model.load_state_dict(best_state)

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        class_names,
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": class_names,
            "args": vars(args),
        },
        args.output_dir / "best_model.pt",
    )

    np.savez_compressed(
        args.output_dir / "analysis_features.npz",
        latent_features=test_metrics["latent_features"],
        pooled_features=test_metrics["pooled_features"],
        labels=test_metrics["labels"],
        predictions=test_metrics["predictions"],
        class_names=np.asarray(class_names),
    )

    with (args.output_dir / "classification_report.json").open("w") as f:
        json.dump(test_metrics["report"], f, indent=2)

    np.save(
        args.output_dir / "confusion_matrix.npy",
        test_metrics["confusion_matrix"],
    )

    with (args.output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "multitask_enabled": not args.disable_multitask,
        "hard_mining_enabled": args.enable_hard_mining,
        "hard_mining_weight": args.hard_mining_weight,
    }

    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Final =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()



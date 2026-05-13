#!/usr/bin/env python3

from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================

EXPERIMENT_DIR = Path(
    "outputs/multitask/full_model"
)

CONFUSION_MATRIX_PATH = (
    EXPERIMENT_DIR / "confusion_matrix.npy"
)

REPORT_PATH = (
    EXPERIMENT_DIR / "classification_report.json"
)

OUTPUT_PATH = (
    EXPERIMENT_DIR / "confusion_matrix.png"
)


# =========================================================
# Load Data
# =========================================================

cm = np.load(CONFUSION_MATRIX_PATH)

with open(REPORT_PATH, "r") as f:
    report = json.load(f)

class_names = [
    k for k in report.keys()
    if k not in [
        "accuracy",
        "macro avg",
        "weighted avg",
    ]
]


# =========================================================
# Normalize
# =========================================================

cm = cm.astype(np.float32)

row_sums = cm.sum(axis=1, keepdims=True)

cm_normalized = cm / np.maximum(row_sums, 1e-8)


# =========================================================
# Plot
# =========================================================

fig, ax = plt.subplots(figsize=(14, 12))

im = ax.imshow(cm_normalized)

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))

ax.set_xticklabels(
    class_names,
    rotation=45,
    ha="right",
    fontsize=10,
)

ax.set_yticklabels(
    class_names,
    fontsize=10,
)

ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)

ax.set_title(
    "Normalized Confusion Matrix",
    fontsize=16,
)

# =========================================================
# Cell Values
# =========================================================

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        value = cm_normalized[i, j]

        if value < 0.01:
            continue

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

fig.colorbar(im)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
)

print(f"\nSaved to: {OUTPUT_PATH}")
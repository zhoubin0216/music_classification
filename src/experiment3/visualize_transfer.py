#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(
    # "outputs/cross_cultural_transfer/western_transfer/fold_1"
    "outputs/cross_cultural_transfer/eastern_baseline/fold_1"
)

# TITLE_PREFIX = "Western Transfer"
TITLE_PREFIX = "Eastern Baseline"

# =========================================================
# Load
# =========================================================

analysis = np.load(
    EXPERIMENT_DIR / "analysis_features.npz",
    allow_pickle=True,
)

latent_features = analysis["latent_features"]
pooled_features = analysis["pooled_features"]

labels = analysis["labels"]
predictions = analysis["predictions"]

class_names = analysis["class_names"]


# =========================================================
# Confusion Matrix
# =========================================================

cm = confusion_matrix(labels, predictions)

cm_normalized = cm.astype(np.float32) / cm.sum(
    axis=1,
    keepdims=True,
)

plt.figure(figsize=(10, 8))

plt.imshow(cm_normalized, interpolation="nearest")

plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45,
)

plt.yticks(
    np.arange(len(class_names)),
    class_names,
)

for i in range(len(class_names)):
    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title(f"{TITLE_PREFIX} - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    EXPERIMENT_DIR / "confusion_matrix_heatmap.png",
    dpi=300,
)

plt.close()


# =========================================================
# Tsen - Latent Features
# =========================================================

scaler = StandardScaler()

latent_scaled = scaler.fit_transform(latent_features)

reducer = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
)

embedding = reducer.fit_transform(latent_scaled)

plt.figure(figsize=(10, 8))

for class_idx, class_name in enumerate(class_names):

    mask = labels == class_idx

    plt.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        label=class_name,
        alpha=0.7,
        s=25,
    )

plt.legend()

plt.title(f"{TITLE_PREFIX} - Latent Tsen")

plt.xlabel("Tsen-1")
plt.ylabel("Tsen-2")

plt.tight_layout()

plt.savefig(
    EXPERIMENT_DIR / "latent_Tsen.png",
    dpi=300,
)

plt.close()


# =========================================================
# Tsen - Pooled Features
# =========================================================

pooled_scaled = scaler.fit_transform(pooled_features)

embedding_pooled = reducer.fit_transform(pooled_scaled)

plt.figure(figsize=(10, 8))

for class_idx, class_name in enumerate(class_names):

    mask = labels == class_idx

    plt.scatter(
        embedding_pooled[mask, 0],
        embedding_pooled[mask, 1],
        label=class_name,
        alpha=0.7,
        s=25,
    )

plt.legend()

plt.title(f"{TITLE_PREFIX} - Pooled Tsen")

plt.xlabel("Tsen-1")
plt.ylabel("Tsen-2")

plt.tight_layout()

plt.savefig(
    EXPERIMENT_DIR / "pooled_Tsen.png",
    dpi=300,
)

plt.close()


print("Done.")
print(f"Saved to: {EXPERIMENT_DIR}")

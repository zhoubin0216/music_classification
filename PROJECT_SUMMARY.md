# Music Classification Project Summary

## 1. Project Goal

This project investigates cross-cultural music genre classification with pretrained deep audio representations.

The core research question is whether a modern music representation model, MERT, can support genre recognition across Western and Eastern music traditions, and whether a classifier trained on Western music exhibits projection bias when applied to Eastern music.

The current pipeline is:

```text
audio -> MERT feature extraction -> MLP classifier -> evaluation / visualization
```

MERT is used as a frozen feature extractor. We do not pretrain MERT from scratch.

## 2. Data

### Western Dataset

Western music comes from GTZAN-style genre folders.

Classes:

```text
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
```

Feature extraction result:

```text
datasets/features/mert_gtzan/features.npz
```

Summary:

```text
Input files: 1000
Successful files: 999
Failed files: 1
Feature shape: (999, 768)
Feature type: 30s track-level MERT final-layer embedding
```

### Eastern Dataset

Eastern music contains 8 genres:

```text
carnatic, cpop, enka, guqinguzheng, jingju, jpop, kpop, sitar
```

Feature extraction result:

```text
datasets/features/mert_eastern/features.npz
```

Summary:

```text
Input files: 2111
Successful files: 2111
Failed files: 0
Feature shape: (2111, 768)
Feature type: 30s track-level MERT final-layer embedding
```

### Mixed 18-Class Dataset

Western and Eastern features were merged into one 18-class dataset:

```text
datasets/features/mert_mixed_18/features.npz
```

Summary:

```text
Western rows/tracks: 999
Eastern rows/tracks: 2111
Total rows/tracks: 3110
Feature shape: (3110, 768)
Number of classes: 18
```

The merged dataset also stores class region metadata:

```text
western: first 10 classes
eastern: last 8 classes
```

## 3. Main Code Files

### Feature Extraction

```text
src/extract_mert_features.py
```

Purpose:

- Loads audio from folder-structured datasets.
- Uses Hugging Face MERT, default `m-a-p/MERT-v1-95M`.
- Extracts hidden states.
- Applies mean pooling over time.
- Saves features, labels, paths, class names, and metadata.

Important behavior:

- Default output is one feature vector per track: `(768,)`.
- Optional `--save-chunks` can save one feature row per chunk.
- Optional `--save-all-layers` can save one pooled vector per MERT layer.

### MLP Classifier Training

```text
src/train_classifier.py
```

Purpose:

- Loads `features.npz`.
- Groups rows by `track_id` if chunk-level features are used.
- Trains an MLP genre classifier.
- Supports 5-fold cross-validation.
- Uses a 6:2:2 protocol per fold:

```text
3 folds train
1 fold validation
1 fold test
```

Model components:

- Optional MERT layer weighted aggregation.
- Optional chunk weighted aggregation.
- MLP classifier head.

Training defaults currently use regularization:

```text
learning rate: 3e-4
hidden dim: 64
aggregation hidden dim: 16
dropout: 0.5
weight decay: 1e-3
patience: 5
label smoothing: 0.05
```

The final model is trained on all tracks using the median best epoch from cross-validation.

### Experiment 1: Western Bias Probe

```text
src/experiment1/experiment1_bias_probe.py
src/experiment1/visualize_experiment1.py
```

Purpose:

- Loads the Western-trained 10-class MLP.
- Applies it directly to Eastern MERT features.
- Treats predictions as Western closed-set projections, not as Eastern classification accuracy.
- Exports prediction distributions, confidence statistics, clustering features, and visualizations.

### Experiment 2: Mixed 18-Class Baseline

```text
src/experiment2/prepare_mixed_features.py
src/experiment2/visualize_experiment2.py
```

Purpose:

- Merges Western and Eastern feature files into an 18-class feature file.
- Trains an 18-class MLP baseline with `train_classifier.py`.
- Visualizes confusion matrix, per-class F1, region confusion, and raw MERT t-SNE/PCA.

## 4. Western-Only Baseline

The Western-only baseline trains a 10-class MLP on GTZAN MERT features.

Output:

```text
outputs/mlp_mert_gtzan_artist_filtered/
```

Cross-validation result:

```text
Test accuracy mean: 0.6249
Test accuracy std: 0.0571
Test macro F1 mean: 0.5886
Test macro F1 std: 0.0531
```

Final model:

```text
outputs/mlp_mert_gtzan_artist_filtered/final_model.pt
```

Final model training:

```text
Training tracks: 999
```

Interpretation:

- Artist-filtered evaluation is the current Western-only baseline.
- The previous random-split score around 0.8 was likely inflated by GTZAN artist leakage.
- Train performance can still be higher than test performance, so regularization and early stopping remain useful.

## 5. Experiment 1: Bias Validation

### Goal

Experiment 1 tests how a classifier trained only on Western genres projects Eastern music into Western genre labels.

Pipeline:

```text
Eastern music -> MERT -> Western-trained 10-class MLP -> Western label projection
```

Important interpretation:

This is not Eastern genre classification accuracy. The Eastern labels are outside the Western 10-class label space. The result should be interpreted as closed-set projection behavior and potential Western label bias.

### Outputs

Directory:

```text
outputs/experiment1/
```

Important files:

```text
eastern_closed_set_predictions.csv
eastern_to_western_projection.csv
confidence_summary.json
clustering_summary.json
eastern_bias_probe_features.npz
figures/
```

### Projection Results

Top Western projection by Eastern genre:

| Eastern genre | Tracks | Top Western projection | Proportion |
|---|---:|---|---:|
| carnatic | 458 | hiphop | 0.349 |
| cpop | 100 | pop | 0.820 |
| enka | 117 | pop | 0.487 |
| guqinguzheng | 434 | classical | 0.687 |
| jingju | 390 | classical | 0.844 |
| jpop | 204 | pop | 0.539 |
| kpop | 298 | pop | 0.876 |
| sitar | 110 | pop | 0.436 |

Overall prediction confidence:

```text
Mean confidence: 0.5683
Confidence std: 0.2228
Mean entropy: 1.3267
Entropy std: 0.5331
Max entropy for 10 classes: 2.3026
```

Notable observations:

- C-pop, J-pop, and K-pop are strongly projected toward `pop`.
- Jingju and guqinguzheng are strongly projected toward `classical`.
- Carnatic is spread across several Western labels, especially `hiphop`, `jazz`, and `pop`.
- Sitar has relatively low confidence and high entropy, indicating greater uncertainty.

### Feature-Space Analysis

Three feature spaces were exported:

```text
raw_mert_features
mlp_pooled_features
mlp_penultimate_features
```

Meaning:

- `raw_mert_features`: original MERT embeddings.
- `mlp_pooled_features`: representation after standardization and aggregation before classifier MLP.
- `mlp_penultimate_features`: hidden representation immediately before the final Western classifier layer.

Because the current model uses one 30s feature per track with `input_shape = [768]`, `raw_mert` and `mlp_pooled` are almost identical in visualizations.

KMeans clustering against Eastern labels:

| Space | ARI | NMI |
|---|---:|---:|
| raw_mert | 0.4819 | 0.5897 |
| mlp_pooled | 0.4819 | 0.5897 |
| mlp_penultimate | 0.3858 | 0.4341 |

Interpretation:

- Raw MERT already separates several Eastern genres, especially Carnatic and Jingju.
- The Western-trained MLP reshapes the representation space, but the reshaped penultimate space aligns less strongly with true Eastern labels by ARI/NMI.
- This supports the idea that the Western classifier projects Eastern music along Western genre-discriminative axes rather than genuinely learning Eastern genre semantics.

### Visualizations

Generated figures include:

```text
projection_heatmap.png
confidence_by_eastern_genre.png
entropy_by_eastern_genre.png
confidence_histogram.png
entropy_histogram.png
raw_mert_tsne_by_eastern_genre.png
raw_mert_tsne_by_western_projection.png
mlp_pooled_tsne_by_eastern_genre.png
mlp_pooled_tsne_by_western_projection.png
mlp_penultimate_tsne_by_eastern_genre.png
mlp_penultimate_tsne_by_western_projection.png
clustering_metrics.png
```

Main visualization interpretation:

- Raw MERT shows clear grouping for some Eastern genres.
- MLP pooled looks nearly identical to raw MERT due to the current single-vector input setup.
- MLP penultimate appears more spatially reshaped, but this does not necessarily mean better Eastern genre understanding.
- If penultimate separation is more aligned with Western projection labels than Eastern labels, it indicates Western-label bias.

## 6. Experiment 2: Mixed 18-Class Baseline

### Goal

Experiment 2 trains a classifier on a mixed dataset with both Western and Eastern genres.

Pipeline:

```text
Western + Eastern music -> MERT -> 18-class MLP classifier
```

Classes:

```text
Western:
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Eastern:
carnatic, cpop, enka, guqinguzheng, jingju, jpop, kpop, sitar
```

### Feature Preparation

The mixed feature file was prepared with:

```text
src/experiment2/prepare_mixed_features.py
```

Output:

```text
datasets/features/mert_mixed_18/features.npz
datasets/features/mert_mixed_18/metadata.csv
datasets/features/mert_mixed_18/merge_config.json
```

Mixed feature summary:

```text
Feature shape: (3110, 768)
Western tracks: 999
Eastern tracks: 2111
Number of classes: 18
```

### Training

The 18-class model was trained with:

```text
src/train_classifier.py
```

Output:

```text
outputs/mlp_mert_18/
```

Cross-validation result:

```text
Test accuracy mean: 0.8759
Test accuracy std: 0.0120
Test macro F1 mean: 0.8066
Test macro F1 std: 0.0226
Validation macro F1 mean: 0.8182
Median best epoch: 45
```

Final model:

```text
outputs/mlp_mert_18/final_model.pt
```

Final model training:

```text
Training tracks: 3110
Final epochs: 45
Final train accuracy: 0.8341
Final train macro F1: 0.7529
```

### Per-Class Performance

Selected F1-scores from the 18-class cross-validated test folds:

| Class | F1 |
|---|---:|
| jingju | 0.9974 |
| carnatic | 0.9881 |
| guqinguzheng | 0.9705 |
| kpop | 0.9216 |
| classical | 0.9154 |
| jazz | 0.8945 |
| jpop | 0.8449 |
| metal | 0.8169 |
| hiphop | 0.8018 |
| sitar | 0.8000 |
| cpop | 0.7940 |
| blues | 0.7647 |
| disco | 0.7619 |
| reggae | 0.7500 |
| enka | 0.7196 |
| country | 0.6629 |
| pop | 0.6420 |
| rock | 0.4795 |

Interpretation:

- Several Eastern genres are classified very well, especially Jingju, Carnatic, and guqinguzheng.
- Western classes such as rock, pop, and country are more difficult.
- The high overall accuracy is partly influenced by class imbalance: some Eastern classes have more samples than Western classes.
- Macro F1 is therefore more informative than accuracy for comparing all 18 classes fairly.

### Visualizations

Generated figures include:

```text
confusion_matrix_counts.png
confusion_matrix_normalized.png
per_class_f1.png
region_confusion_matrix.png
raw_mert_tsne_by_genre.png
raw_mert_tsne_by_region.png
```

These figures support:

- Class-level confusion analysis.
- Western-vs-Eastern region confusion analysis.
- Feature-space separation in raw MERT embeddings.

## 7. Important Technical Decisions

### MERT is Frozen

MERT is not trained or fine-tuned. It is used only as a pretrained feature extractor.

Reason:

- The project dataset is too small for MERT pretraining.
- Fine-tuning MERT may overfit and is more expensive.
- Frozen features make the experiments easier to interpret.

### Mean Pooling

For each MERT hidden-state sequence:

```text
(time_steps, 768) -> mean over time -> (768,)
```

This greatly reduces storage and gives one clip-level embedding per 30s track.

### Chunking Experiments

Chunked features were considered:

```text
30s track -> 3 x 10s chunks
```

However, chunking did not necessarily improve performance. For GTZAN-style 30s clips, a single 30s pooled embedding can be more stable than several short segment embeddings plus learned aggregation.

### Layer Aggregation

MERT supports multiple hidden layers. The code can optionally save all layers and learn a weighted layer aggregation.

This was discussed as a possible ablation, but the main experiments currently use final-layer 768-dimensional features.

### Dataset Leakage Avoidance

When chunk features are used, the training script groups chunks by `track_id` so that all chunks from the same track stay in the same train/validation/test split.

This avoids inflated results caused by the same song appearing in both training and test splits.

## 8. Environment and Dependency Notes

The project uses Python with:

```text
numpy
torch
torchaudio
transformers==4.38.0
scikit-learn
matplotlib
soundfile
tqdm
```

PyTorch CUDA version must match the server driver. On the server, `nvidia-smi` showed CUDA 12.4 support, so the compatible PyTorch wheel should use `cu124`.

Example:

```bash
uv pip uninstall torch torchaudio torchvision
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 9. Git and File Management Notes

Large data and generated results should not be committed.

Ignored paths include:

```text
datasets/
outputs/
__pycache__/
*.pyc
audios/
```

The local `src/MERT/` directory is not needed for the current pipeline because the code uses Hugging Face MERT. It is only useful for MERT pretraining, fairseq training, or checkpoint conversion.

## 10. Current Conclusions

1. Frozen MERT embeddings provide a strong baseline for Western genre classification.
2. A Western-trained classifier projects Eastern music into familiar Western labels, often with high confidence.
3. Projection patterns are musically interpretable:
   - Pop-related Eastern genres tend to map to `pop`.
   - Jingju and guqinguzheng tend to map to `classical`.
   - Carnatic and sitar are more ambiguous.
4. Raw MERT already captures meaningful Eastern genre structure.
5. The Western MLP penultimate space is reshaped by Western classification objectives and does not necessarily align better with true Eastern labels.
6. The 18-class mixed classifier achieves strong overall performance, but macro F1 and per-class F1 are necessary because the mixed dataset is imbalanced.
7. Some Eastern genres are easier to classify than several Western genres in the mixed setting, likely because their acoustic and stylistic signatures are more distinct in MERT space.

## 11. Suggested Next Steps

1. Add a simple baseline:

```text
MERT -> Logistic Regression
MERT -> Linear SVM
```

This will show whether the MLP is actually needed.

2. Compare feature settings:

```text
30s pooled final-layer MERT
10s chunked MERT + mean aggregation
10s chunked MERT + learned weighted aggregation
all-layer MERT + learned layer aggregation
```

3. Improve experiment 2 reporting:

- Report Western macro F1 separately.
- Report Eastern macro F1 separately.
- Report Western-to-Eastern and Eastern-to-Western confusion rates.

4. Add UMAP if desired.

t-SNE is useful for local structure, but UMAP may provide a more stable global visualization.

5. Review labels and class balance.

Some classes have far more tracks than others. `train_classifier.py --class-weight balanced` enables class-weighted cross entropy for fairer 18-class training.

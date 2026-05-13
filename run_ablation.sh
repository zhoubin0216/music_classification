#!/bin/bash

FEATURES="datasets/features/mert_mixed_18/features.npz"
SPLIT="datasets/features/mert_mixed_18/balanced_track_split.csv"

# =========================================================
# Baseline
# =========================================================

#python src/experiment4/representation_multitask.py \
#  --features $FEATURES \
#  --split-csv $SPLIT \
#  --output-dir outputs/multitask/baseline \
#  --disable-multitask \
#  --disable-adapter \
#  --disable-mean-std-pooling \
#  --patience 5

# =========================================================
# Multi-task Only
# =========================================================

#python src/experiment4/representation_multitask.py \
#  --features $FEATURES \
#  --split-csv $SPLIT \
#  --output-dir outputs/multitask/multitask_only \
#  --disable-adapter \
#  --disable-mean-std-pooling \
#  --patience 5

# =========================================================
# Hard Mining Only
# =========================================================

#python src/experiment4/representation_multitask.py \
#  --features $FEATURES \
#  --split-csv $SPLIT \
#  --output-dir outputs/multitask/hard_mining_only_3 \
#  --disable-multitask \
#  --disable-adapter \
#  --disable-mean-std-pooling \
#  --enable-hard-mining \
#  --hard-mining-weight 2.0 \
#  --patience 5

# =========================================================
# Full Model
# =========================================================

python src/experiment4/representation_multitask.py \
  --features $FEATURES \
  --split-csv $SPLIT \
  --output-dir outputs/multitask/full_model \
  --disable-adapter \
  --disable-mean-std-pooling \
  --enable-hard-mining \
  --hard-mining-weight 2.0 \
  --patience 5
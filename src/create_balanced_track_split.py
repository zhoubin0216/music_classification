#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# Track ID Extraction
# =========================================================

def extract_track_id(path_str: str):

    path = Path(path_str)

    stem = path.stem

    # eastern sliced:
    # carnatic_001_003.wav -> carnatic_001
    stem = re.sub(r'_[0-9]{3}$', '', stem)

    return stem


# =========================================================
# Main
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--metadata-csv',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-csv',
        type=Path,
        required=True,
    )

    parser.add_argument(
        '--output-summary',
        type=Path,
        default=None,
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    return parser.parse_args()


# =========================================================
# Split Logic
# =========================================================

def main():

    args = parse_args()

    df = pd.read_csv(args.metadata_csv)

    # -----------------------------------------------------
    # Build Track Table
    # -----------------------------------------------------

    track_records = []

    grouped = df.groupby('path')

    # First recover track_id
    df['group_track_id'] = df['path'].apply(extract_track_id)

    track_grouped = df.groupby('group_track_id')

    for track_id, group in track_grouped:

        class_name = group.iloc[0]['class_name']

        track_records.append({
            'track_id': track_id,
            'class_name': class_name,
        })

    track_df = pd.DataFrame(track_records)

    # -----------------------------------------------------
    # Stratified Track Split
    # -----------------------------------------------------

    train_tracks, temp_tracks = train_test_split(
        track_df,
        test_size=0.4,
        stratify=track_df['class_name'],
        random_state=args.seed,
    )

    val_tracks, test_tracks = train_test_split(
        temp_tracks,
        test_size=0.5,
        stratify=temp_tracks['class_name'],
        random_state=args.seed,
    )

    train_track_set = set(train_tracks['track_id'])
    val_track_set = set(val_tracks['track_id'])
    test_track_set = set(test_tracks['track_id'])

    # -----------------------------------------------------
    # Assign Split
    # -----------------------------------------------------

    splits = []

    for _, row in df.iterrows():

        track_id = row['group_track_id']

        if track_id in train_track_set:
            split = 'train'

        elif track_id in val_track_set:
            split = 'validation'

        elif track_id in test_track_set:
            split = 'test'

        else:
            raise RuntimeError(f'Unknown track: {track_id}')

        splits.append(split)

    df['split'] = splits

    # -----------------------------------------------------
    # Output CSV
    # -----------------------------------------------------

    output_columns = [
        'feature_index',
        'path',
        'label',
        'class_name',
        'group_track_id',
        'split',
    ]

    output_df = df[output_columns].rename(columns={
        'group_track_id': 'track_id'
    })

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        args.output_csv,
        index=False,
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    summary = {
        'num_total_samples': int(len(df)),
        'num_total_tracks': int(track_df.shape[0]),
        'num_train_tracks': int(train_tracks.shape[0]),
        'num_val_tracks': int(val_tracks.shape[0]),
        'num_test_tracks': int(test_tracks.shape[0]),
        'train_distribution': train_tracks['class_name'].value_counts().to_dict(),
        'validation_distribution': val_tracks['class_name'].value_counts().to_dict(),
        'test_distribution': test_tracks['class_name'].value_counts().to_dict(),
    }

    if args.output_summary is not None:

        with open(args.output_summary, 'w') as f:
            json.dump(summary, f, indent=2)

    print('\n===== Split Complete =====')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


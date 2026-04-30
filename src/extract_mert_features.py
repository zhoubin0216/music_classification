#!/usr/bin/env python3
"""Extract frozen MERT embeddings for a folder-structured audio dataset.

Expected dataset layout:

    data_root/
      blues/*.wav
      classical/*.wav
      ...

The script writes:
  - features.npz: feature matrix, numeric labels, class names, source paths
  - metadata.csv: one row per successfully processed audio file
  - extraction_config.json: exact extraction settings

Use --save-chunks with --chunk-seconds 10 for models that aggregate multiple
segment embeddings per track.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2FeatureExtractor

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        desc = kwargs.get("desc")
        if desc:
            print(desc)
        return iterable

    tqdm.write = print  # type: ignore[attr-defined]


DEFAULT_DATA_ROOT = "/home/dt2119/dt2119/music_classification/datasets/audios/western_data/genres_original"
DEFAULT_OUTPUT_DIR = (
    "/home/dt2119/dt2119/music_classification/datasets/features/mert_gtzan"
)
DEFAULT_MODEL_NAME = "m-a-p/MERT-v1-95M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MERT embeddings from GTZAN-style genre folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(DEFAULT_DATA_ROOT),
        help="Root directory containing one subfolder per genre.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory where features and metadata will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Hidden-state layer to save. Use -1 for the final layer.",
    )
    parser.add_argument(
        "--save-all-layers",
        action="store_true",
        help="Save one time-pooled vector per hidden layer instead of one selected layer.",
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help=(
            "Save one feature row per audio chunk. Without this flag, chunk "
            "features are averaged into one track-level embedding."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help=(
            "Chunk length before MERT inference. Use 0 or a negative value to process "
            "each file as one chunk."
        ),
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=1.0,
        help=(
            "Drop trailing chunks shorter than this value. This avoids MERT "
            "convolution errors from tiny resampling remainders."
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".wav"],
        help="Audio file extensions to include.",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Stop immediately if an audio file cannot be processed.",
    )
    return parser.parse_args()


def discover_audio_files(
    data_root: Path, extensions: Iterable[str], max_files_per_class: int | None
) -> tuple[list[tuple[Path, str, int]], list[str]]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    normalized_extensions = {ext.lower() for ext in extensions}
    class_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    if not class_names:
        raise RuntimeError(f"No class subdirectories found under: {data_root}")

    items: list[tuple[Path, str, int]] = []
    for label_id, class_dir in enumerate(class_dirs):
        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in normalized_extensions
        )
        if max_files_per_class is not None:
            files = files[:max_files_per_class]
        items.extend((path, class_dir.name, label_id) for path in files)

    if not items:
        raise RuntimeError(
            f"No audio files with extensions {sorted(normalized_extensions)} "
            f"found under: {data_root}"
        )

    return items, class_names


def load_mono_audio(path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2:
        raise RuntimeError(
            f"Expected waveform with shape [channels, samples], got {waveform.shape}"
        )
    waveform = waveform.float().mean(dim=0)
    return waveform, sample_rate


def iter_chunks(
    waveform: torch.Tensor,
    chunk_samples: int,
    min_chunk_samples: int,
) -> Iterable[torch.Tensor]:
    if chunk_samples <= 0 or waveform.numel() <= chunk_samples:
        if waveform.numel() >= min_chunk_samples:
            yield waveform
        return

    for start in range(0, waveform.numel(), chunk_samples):
        chunk = waveform[start : start + chunk_samples]
        if chunk.numel() >= min_chunk_samples:
            yield chunk


def move_inputs_to_device(
    inputs: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


@torch.inference_mode()
def extract_embedding(
    path: Path,
    model: torch.nn.Module,
    processor: Wav2Vec2FeatureExtractor,
    resampler_cache: dict[tuple[int, int], torchaudio.transforms.Resample],
    device: torch.device,
    layer: int,
    save_all_layers: bool,
    save_chunks: bool,
    chunk_seconds: float,
    min_chunk_seconds: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    waveform, source_sample_rate = load_mono_audio(path)
    target_sample_rate = int(processor.sampling_rate)

    if source_sample_rate != target_sample_rate:
        key = (source_sample_rate, target_sample_rate)
        if key not in resampler_cache:
            resampler_cache[key] = torchaudio.transforms.Resample(
                orig_freq=source_sample_rate,
                new_freq=target_sample_rate,
            )
        waveform = resampler_cache[key](waveform)

    chunk_samples = int(round(chunk_seconds * target_sample_rate))
    if chunk_seconds <= 0:
        chunk_samples = 0
    min_chunk_samples = max(1, int(round(min_chunk_seconds * target_sample_rate)))

    weighted_sum: torch.Tensor | None = None
    chunk_embeddings: list[torch.Tensor] = []
    total_samples = 0
    num_chunks = 0

    for chunk in iter_chunks(waveform, chunk_samples, min_chunk_samples):
        if chunk.numel() == 0:
            continue

        inputs = processor(
            chunk.cpu().numpy(),
            sampling_rate=target_sample_rate,
            return_tensors="pt",
        )
        inputs = move_inputs_to_device(inputs, device)
        outputs = model(**inputs, output_hidden_states=True)

        if save_all_layers:
            # [layers, batch, frames, dim] -> [layers, dim]
            hidden = torch.stack(outputs.hidden_states, dim=0).mean(dim=2).squeeze(1)
        else:
            hidden = outputs.hidden_states[layer].mean(dim=1).squeeze(0)

        hidden = hidden.detach().cpu()
        weight = int(chunk.numel())
        if save_chunks:
            chunk_embeddings.append(hidden)
        else:
            weighted_sum = (
                hidden * weight
                if weighted_sum is None
                else weighted_sum + hidden * weight
            )
        total_samples += weight
        num_chunks += 1

    if (
        total_samples == 0
        or (save_chunks and not chunk_embeddings)
        or (not save_chunks and weighted_sum is None)
    ):
        raise RuntimeError("Audio file produced no non-empty chunks")

    if save_chunks:
        embedding = torch.stack(chunk_embeddings, dim=0).numpy().astype(np.float32)
    else:
        embedding = (weighted_sum / total_samples).numpy().astype(np.float32)
    stats = {
        "duration_seconds": float(waveform.numel() / target_sample_rate),
        "processed_seconds": float(total_samples / target_sample_rate),
        "skipped_tail_seconds": float(
            (waveform.numel() - total_samples) / target_sample_rate
        ),
        "source_sample_rate": int(source_sample_rate),
        "target_sample_rate": int(target_sample_rate),
        "num_chunks": int(num_chunks),
    }
    return embedding, stats


def write_metadata(metadata_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "feature_index",
        "track_id",
        "segment_index",
        "path",
        "label",
        "label_id",
        "duration_seconds",
        "processed_seconds",
        "skipped_tail_seconds",
        "source_sample_rate",
        "target_sample_rate",
        "num_chunks",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items, class_names = discover_audio_files(
        args.data_root,
        args.extensions,
        args.max_files_per_class,
    )
    print(f"Found {len(items)} files across {len(class_names)} classes:")
    print(", ".join(class_names))

    print(f"Loading MERT model: {args.model_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval().to(device)

    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    track_ids: list[int] = []
    segment_indices: list[int] = []
    metadata_rows: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    successful_tracks = 0
    resampler_cache: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

    for path, label, label_id in tqdm(items, desc="Extracting MERT features"):
        try:
            embedding, stats = extract_embedding(
                path=path,
                model=model,
                processor=processor,
                resampler_cache=resampler_cache,
                device=device,
                layer=args.layer,
                save_all_layers=args.save_all_layers,
                save_chunks=args.save_chunks,
                chunk_seconds=args.chunk_seconds,
                min_chunk_seconds=args.min_chunk_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch extraction running by default.
            if args.fail_on_error:
                raise
            failed.append({"path": str(path), "error": repr(exc)})
            tqdm.write(f"Skipping {path}: {exc}")
            continue

        track_id = successful_tracks
        successful_tracks += 1

        if args.save_chunks:
            segment_embeddings = embedding
        else:
            segment_embeddings = embedding[None, ...]

        for segment_index, segment_embedding in enumerate(segment_embeddings):
            feature_index = len(features)
            features.append(segment_embedding)
            labels.append(label_id)
            paths.append(str(path))
            track_ids.append(track_id)
            segment_indices.append(segment_index)
            metadata_rows.append(
                {
                    "feature_index": feature_index,
                    "track_id": track_id,
                    "segment_index": segment_index,
                    "path": str(path),
                    "label": label,
                    "label_id": label_id,
                    **stats,
                }
            )

    if not features:
        raise RuntimeError("No features were extracted successfully.")

    feature_array = np.stack(features, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)

    np.savez_compressed(
        args.output_dir / "features.npz",
        features=feature_array,
        labels=label_array,
        paths=np.asarray(paths),
        track_ids=np.asarray(track_ids, dtype=np.int64),
        segment_indices=np.asarray(segment_indices, dtype=np.int64),
        class_names=np.asarray(class_names),
        model_name=np.asarray(args.model_name),
        layer=np.asarray("all" if args.save_all_layers else args.layer),
        sample_rate=np.asarray(int(processor.sampling_rate)),
        save_chunks=np.asarray(args.save_chunks),
    )
    write_metadata(args.output_dir / "metadata.csv", metadata_rows)

    config = {
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "device": str(device),
        "layer": "all" if args.save_all_layers else args.layer,
        "save_all_layers": args.save_all_layers,
        "save_chunks": args.save_chunks,
        "chunk_seconds": args.chunk_seconds,
        "min_chunk_seconds": args.min_chunk_seconds,
        "extensions": args.extensions,
        "max_files_per_class": args.max_files_per_class,
        "num_input_files": len(items),
        "num_successful_tracks": successful_tracks,
        "num_feature_rows": len(features),
        "num_failed_files": len(failed),
        "feature_shape": list(feature_array.shape),
        "class_names": class_names,
    }
    with (args.output_dir / "extraction_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)

    if failed:
        with (args.output_dir / "failed_files.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(failed, handle, indent=2)
        print(f"Finished with {len(failed)} failed files. See failed_files.json.")

    print(f"Saved features: {args.output_dir / 'features.npz'}")
    print(f"Feature shape: {feature_array.shape}")
    print(f"Saved metadata: {args.output_dir / 'metadata.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)

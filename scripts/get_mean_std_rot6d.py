#!/usr/bin/env python3
"""Compute SignSpark rot6d normalization stats from raw training poses."""

from __future__ import annotations

import argparse
import csv
import gzip
import pickle
import sys
from pathlib import Path
from typing import Iterable

import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mGPT.data.humanml.dataset_t2m import bad_how2sign_ids
from mGPT.data.humanml.load_data import load_csl_sample, load_h2s_sample, load_phoenix_sample
from mGPT.data.humanml.pose_rep import ROT6D_NFEATS


DATASET_ALIASES = {
    "h2s": "how2sign",
    "how2sign": "how2sign",
    "csl": "csl",
    "csl-daily": "csl",
    "csldaily": "csl",
    "phoenix": "phoenix",
    "phoenix14t": "phoenix",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute empirical rot6d mean/std from raw train pose files."
    )
    parser.add_argument(
        "--data-root",
        default=REPO_ROOT / "data",
        type=Path,
        help="Root data directory containing How2Sign, CSL-Daily, and Phoenix_2014T.",
    )
    parser.add_argument(
        "--datasets",
        default="how2sign,csl,phoenix",
        help="Comma-separated datasets to use: how2sign,csl,phoenix.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split used for statistics. Default: train.",
    )
    parser.add_argument(
        "--out-mean",
        default=REPO_ROOT / "data" / "CSL-Daily" / "rot6d_mean.pt",
        type=Path,
        help="Output path for the 246-D rot6d mean tensor.",
    )
    parser.add_argument(
        "--out-std",
        default=REPO_ROOT / "data" / "CSL-Daily" / "rot6d_std.pt",
        type=Path,
        help="Output path for the 246-D rot6d std tensor.",
    )
    parser.add_argument(
        "--eps",
        default=1e-6,
        type=float,
        help="Lower bound applied to output std values.",
    )
    parser.add_argument(
        "--max-clips",
        default=None,
        type=int,
        help="Optional debug limit on number of clips per dataset.",
    )
    parser.add_argument(
        "--log-every",
        default=500,
        type=int,
        help="Print progress every N processed clips.",
    )
    parser.add_argument(
        "--verbose-skips",
        action="store_true",
        help="Print every skipped clip and exception.",
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def normalize_datasets(raw: str) -> list[str]:
    datasets = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in DATASET_ALIASES:
            raise ValueError(f"Unknown dataset '{item}'. Use how2sign,csl,phoenix.")
        dataset = DATASET_ALIASES[key]
        if dataset not in datasets:
            datasets.append(dataset)
    if not datasets:
        raise ValueError("No datasets selected.")
    return datasets


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def iter_how2sign_annotations(data_root: Path, split: str) -> Iterable[dict]:
    csv_path = (
        data_root
        / "How2Sign"
        / split
        / "re_aligned"
        / f"how2sign_realigned_{split}_preprocessed_fps.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["SENTENCE_NAME"]
            if name in bad_how2sign_ids:
                continue
            start = float(row.get("START_REALIGNED", 0.0))
            end = float(row.get("END_REALIGNED", 0.0))
            if end - start >= 30:
                continue
            yield {
                "name": name,
                "fps": float(row["fps"]),
                "text": row.get("SENTENCE", ""),
                "src": "how2sign",
            }


def iter_csl_annotations(data_root: Path, split: str) -> Iterable[dict]:
    ann_path = data_root / "CSL-Daily" / f"csl_clean.{split}"
    for ann in load_pickle(ann_path):
        ann = dict(ann)
        ann["src"] = "csl"
        yield ann


def iter_phoenix_annotations(data_root: Path, split: str) -> Iterable[dict]:
    ann_split = "dev" if split == "val" else split
    ann_path = data_root / "Phoenix_2014T" / f"phoenix14t.{ann_split}"
    for ann in load_pickle(ann_path):
        ann = dict(ann)
        ann["src"] = "phoenix"
        yield ann


def load_rot6d_clip(dataset: str, ann: dict, data_root: Path, split: str):
    if dataset == "how2sign":
        pose_dir = data_root / "How2Sign" / split / "poses"
        clip, _, name, _ = load_h2s_sample(ann, str(pose_dir), pose_rep="rot6d")
    elif dataset == "csl":
        clip, _, name, _ = load_csl_sample(ann, str(data_root / "CSL-Daily"), pose_rep="rot6d")
    elif dataset == "phoenix":
        # Phoenix annotation names already include the split prefix, e.g.
        # "train/11August_2010_Wednesday_tagesschau-1".
        clip, _, name, _ = load_phoenix_sample(
            ann, str(data_root / "Phoenix_2014T"), pose_rep="rot6d"
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return clip, name


def update_running_stats(
    count: int,
    mean: torch.Tensor,
    m2: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    batch = batch.double().reshape(-1, ROT6D_NFEATS)
    batch_count = batch.shape[0]
    if batch_count == 0:
        return count, mean, m2

    batch_mean = batch.mean(dim=0)
    batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
    if count == 0:
        return batch_count, batch_mean, batch_m2

    total = count + batch_count
    delta = batch_mean - mean
    mean = mean + delta * (batch_count / total)
    m2 = m2 + batch_m2 + delta.pow(2) * count * batch_count / total
    return total, mean, m2


def annotations_for_dataset(dataset: str, data_root: Path, split: str) -> Iterable[dict]:
    if dataset == "how2sign":
        return iter_how2sign_annotations(data_root, split)
    if dataset == "csl":
        return iter_csl_annotations(data_root, split)
    if dataset == "phoenix":
        return iter_phoenix_annotations(data_root, split)
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    args = parse_args()
    datasets = normalize_datasets(args.datasets)
    data_root = args.data_root.resolve()

    frame_count = 0
    clip_count = 0
    skipped = 0
    running_mean = torch.zeros(ROT6D_NFEATS, dtype=torch.float64)
    running_m2 = torch.zeros(ROT6D_NFEATS, dtype=torch.float64)

    for dataset in datasets:
        dataset_clips = 0
        dataset_frames_before = frame_count
        annotations = list(annotations_for_dataset(dataset, data_root, args.split))
        progress_total = args.max_clips if args.max_clips is not None else len(annotations)
        progress_total = min(progress_total, len(annotations))
        progress = tqdm(
            annotations,
            total=progress_total,
            desc=f"{dataset}:{args.split}",
            unit="clip",
            dynamic_ncols=True,
            disable=args.disable_progress,
        )

        for ann in progress:
            if args.max_clips is not None and dataset_clips >= args.max_clips:
                break
            try:
                clip, name = load_rot6d_clip(dataset, ann, data_root, args.split)
            except Exception as exc:
                skipped += 1
                if args.verbose_skips:
                    tqdm.write(f"skip {dataset}:{ann.get('name', '<unknown>')} ({exc})")
                progress.set_postfix(
                    clips=clip_count, frames=frame_count, skipped=skipped, refresh=False
                )
                continue
            if clip is None:
                skipped += 1
                progress.set_postfix(
                    clips=clip_count, frames=frame_count, skipped=skipped, refresh=False
                )
                continue

            clip_tensor = torch.as_tensor(clip, dtype=torch.float32)
            if clip_tensor.ndim != 2 or clip_tensor.shape[1] != ROT6D_NFEATS:
                raise ValueError(
                    f"{dataset}:{name} has shape {tuple(clip_tensor.shape)}, "
                    f"expected (*, {ROT6D_NFEATS})."
                )

            frame_count, running_mean, running_m2 = update_running_stats(
                frame_count, running_mean, running_m2, clip_tensor
            )
            clip_count += 1
            dataset_clips += 1
            progress.set_postfix(
                clips=clip_count, frames=frame_count, skipped=skipped, refresh=False
            )
            if args.disable_progress and args.log_every > 0 and clip_count % args.log_every == 0:
                print(
                    f"processed clips={clip_count}, frames={frame_count}, skipped={skipped}",
                    flush=True,
                )

        progress.close()
        print(
            f"done {dataset}: clips={dataset_clips}, "
            f"frames={frame_count - dataset_frames_before}",
            flush=True,
        )

    if frame_count == 0:
        raise RuntimeError("No frames were loaded; cannot compute rot6d stats.")

    rot6d_mean = running_mean.float()
    rot6d_std = torch.sqrt((running_m2 / frame_count).clamp_min(args.eps * args.eps)).float()

    args.out_mean.parent.mkdir(parents=True, exist_ok=True)
    args.out_std.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rot6d_mean, args.out_mean)
    torch.save(rot6d_std, args.out_std)

    print(f"saved mean: {args.out_mean} {tuple(rot6d_mean.shape)}")
    print(f"saved std:  {args.out_std} {tuple(rot6d_std.shape)}")
    print(f"total clips={clip_count}, frames={frame_count}, skipped={skipped}")


if __name__ == "__main__":
    main()

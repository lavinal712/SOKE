import argparse
import csv
import gzip
import json
import math
import os
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from mGPT.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d


SMPLX_KEYS = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute rot6d mean/std for SignSparK SMPL-X body/hand features."
    )
    parser.add_argument("--dataset-name", default="how2sign_csl_phoenix")
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default="./data/How2Sign")
    parser.add_argument("--csl-root", default="./data/CSL-Daily")
    parser.add_argument("--phoenix-root", default="./data/Phoenix_2014T")
    parser.add_argument("--out-dir", default="./data/stats")
    parser.add_argument("--mean-name", default="rot6d_mean.pt")
    parser.add_argument("--std-name", default="rot6d_std.pt")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--min-motion-len", type=int, default=40)
    parser.add_argument("--max-motion-len", type=int, default=400)
    parser.add_argument("--unit-len", type=int, default=4)
    parser.add_argument("--std-eps", type=float, default=1e-6)
    parser.add_argument(
        "--body-joints",
        default="8,11,12,13,14,15,16,17,18,19,20",
        help="0-based SMPL-X body_pose joints used for rot6d body features.",
    )
    parser.add_argument(
        "--no-length-adjust",
        action="store_true",
        help="Skip the same length resampling/cropping used by the H2S dataset.",
    )
    return parser.parse_args()


def frame_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r"_(\d+)_3D\.pkl$", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\.pkl$", name)
    return int(match.group(1)) if match else name


def sample_paths(paths, count):
    count = max(int(count), 1)
    step = float(len(paths)) / count
    return [paths[int(math.floor(i * step))] for i in range(count)]


def adjust_length(features, min_len, max_len, unit_len):
    length = len(features)
    if length < min_len:
        ids = np.linspace(0, length - 1, num=min_len, dtype=int)
        return features[ids]
    if length > max_len:
        ids = np.linspace(0, length - 1, num=max_len, dtype=int)
        return features[ids]

    keep_len = (length // unit_len) * unit_len
    if keep_len <= 0:
        keep_len = length
    start = (length - keep_len) // 2
    return features[start:start + keep_len]


def read_smplx_sequence(frame_paths):
    poses = np.empty((len(frame_paths), 179), dtype=np.float32)
    for idx, frame_path in enumerate(frame_paths):
        with open(frame_path, "rb") as f:
            frame = pickle.load(f)
        poses[idx] = np.concatenate([frame[key] for key in SMPLX_KEYS], axis=0)
    return poses


def to_rot6d_features(smplx_poses, body_joints):
    body = smplx_poses[:, 3:66].reshape(len(smplx_poses), 21, 3)
    left_hand = smplx_poses[:, 66:111].reshape(len(smplx_poses), 15, 3)
    right_hand = smplx_poses[:, 111:156].reshape(len(smplx_poses), 15, 3)
    axis_angle = np.concatenate([body[:, body_joints], left_hand, right_hand], axis=1)

    with torch.no_grad():
        rotations = torch.from_numpy(axis_angle).float()
        rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(rotations))
        rot6d = rot6d.reshape(len(smplx_poses), -1)
    return rot6d.cpu().numpy().astype(np.float64, copy=False)


def h2s_items(root, split):
    csv_path = Path(root) / split / "re_aligned" / f"how2sign_realigned_{split}_preprocessed_fps.csv"
    if not csv_path.exists():
        return []

    items = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            duration = float(row["END_REALIGNED"]) - float(row["START_REALIGNED"])
            if duration >= 30:
                continue
            items.append({
                "src": "how2sign",
                "name": row["SENTENCE_NAME"],
                "fps": float(row["fps"]),
                "root": str(Path(root) / split / "poses"),
            })
    return items


def csl_items(root, split):
    ann_path = Path(root) / ("csl_clean.train" if split == "train" else f"csl_clean.{split}")
    if not ann_path.exists():
        return []
    with gzip.open(ann_path, "rb") as f:
        anns = pickle.load(f)
    return [{"src": "csl", "name": ann["name"], "root": root} for ann in anns]


def phoenix_items(root, split):
    ann_name = "phoenix14t.dev" if split == "val" else f"phoenix14t.{split}"
    ann_path = Path(root) / ann_name
    if not ann_path.exists():
        return []
    with gzip.open(ann_path, "rb") as f:
        anns = pickle.load(f)
    return [{"src": "phoenix", "name": ann["name"], "root": root} for ann in anns]


def collect_items(args):
    dataset_name = args.dataset_name.lower()
    items = []
    if "how2sign" in dataset_name:
        items.extend(h2s_items(args.root, args.split))
    if "csl" in dataset_name:
        items.extend(csl_items(args.csl_root, args.split))
    if "phoenix" in dataset_name:
        items.extend(phoenix_items(args.phoenix_root, args.split))
    return items


def item_frame_paths(item):
    if item["src"] == "how2sign":
        pose_dir = Path(item["root"]) / item["name"]
        paths = sorted((str(p) for p in pose_dir.glob("*.pkl")), key=frame_sort_key)
        if paths and item["fps"] > 24:
            paths = sample_paths(paths, int(24 * len(paths) / item["fps"]))
        return paths

    if item["src"] == "csl":
        pose_dir = Path(item["root"]) / "poses" / item["name"]
    else:
        pose_dir = Path(item["root"]) / item["name"]
    return sorted((str(p) for p in pose_dir.glob("*.pkl")), key=frame_sort_key)


def process_item(item, body_joints, args):
    sample_id = f"{item['src']}:{item['name']}"
    try:
        frame_paths = item_frame_paths(item)
        if len(frame_paths) < 4:
            return None, f"{sample_id} has fewer than 4 frames"

        smplx = read_smplx_sequence(frame_paths)
        rot6d = to_rot6d_features(smplx, body_joints)
        if not args.no_length_adjust:
            rot6d = adjust_length(rot6d, args.min_motion_len, args.max_motion_len, args.unit_len)

        if len(rot6d) == 0:
            return None, f"{sample_id} produced no frames"
        return (len(rot6d), rot6d.sum(axis=0), np.square(rot6d).sum(axis=0)), None
    except Exception as exc:
        return None, f"{sample_id} failed: {exc}"


def main():
    args = parse_args()
    body_joints = [int(x) for x in args.body_joints.split(",") if x.strip()]
    if not body_joints or min(body_joints) < 0 or max(body_joints) >= 21:
        raise ValueError("--body-joints must contain SMPL-X body_pose joint ids in [0, 20].")

    items = collect_items(args)
    if not items:
        raise RuntimeError("No samples found. Check dataset roots, split, and dataset name.")

    nfeats = (len(body_joints) + 15 + 15) * 6
    total_count = 0
    total_sum = np.zeros(nfeats, dtype=np.float64)
    total_sumsq = np.zeros(nfeats, dtype=np.float64)
    errors = []

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, total=None: x

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = [executor.submit(process_item, item, body_joints, args) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures)):
            stats, error = future.result()
            if error is not None:
                errors.append(error)
                continue
            count, sum_vec, sumsq_vec = stats
            total_count += count
            total_sum += sum_vec
            total_sumsq += sumsq_vec

    if total_count == 0:
        raise RuntimeError(f"No valid frames found. First errors: {errors[:5]}")

    mean = total_sum / total_count
    var = np.maximum(total_sumsq / total_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), args.std_eps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_path = out_dir / args.mean_name
    std_path = out_dir / args.std_name
    torch.save(torch.from_numpy(mean).float(), mean_path)
    torch.save(torch.from_numpy(std).float(), std_path)

    meta = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_samples": len(items),
        "num_failed_samples": len(errors),
        "num_frames": int(total_count),
        "nfeats": nfeats,
        "body_joints": body_joints,
        "mean_path": str(mean_path),
        "std_path": str(std_path),
    }
    with open(out_dir / "rot6d_stats_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    if errors:
        print("First skipped samples:")
        for error in errors[:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()

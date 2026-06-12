import numpy as np
import torch

from mGPT.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


# SMPL-X body_pose uses 21 joints excluding the global root.  This selection
# keeps the manual upper-body chain used for signing: spine3 plus neck/head,
# shoulders, elbows, wrists, and collars.
UPPER_BODY_BODY_POSE_IDXS = (8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
SOKE_BODY_POSE_IDXS = tuple(range(11, 21))
ROT6D_NFEATS = (len(UPPER_BODY_BODY_POSE_IDXS) + 15 + 15) * 6
AXIS_ANGLE_NFEATS = 133


def normalize_pose_rep(pose_rep):
    pose_rep = str(pose_rep or "axis_angle").lower()
    if pose_rep in {"axis_angle", "axis-angle", "aa"}:
        return "axis_angle"
    if pose_rep in {"rot6d", "rotation6d", "6d", "6drot"}:
        return "rot6d"
    raise ValueError(f"Unsupported pose representation: {pose_rep}")


def smplx_axis_angle_to_soke_features(clip_poses):
    # Original SOKE feature layout: drop global/lower-body rotations and shape,
    # keep upper body, both hands, jaw, and expression.
    features = clip_poses[:, (3 + 3 * 11):]
    return np.concatenate([features[:, :-20], features[:, -10:]], axis=1)


def _axis_angle_to_rot6d_np(axis_angle):
    shape = axis_angle.shape[:-1]
    tensor = torch.as_tensor(axis_angle, dtype=torch.float32).reshape(-1, 3)
    rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(tensor))
    return rot6d.reshape(*shape, 6).cpu().numpy()


def smplx_axis_angle_to_rot6d_features(clip_poses):
    body_pose = clip_poses[:, 3:66].reshape(len(clip_poses), 21, 3)
    body_pose = body_pose[:, UPPER_BODY_BODY_POSE_IDXS]
    left_hand = clip_poses[:, 66:111].reshape(len(clip_poses), 15, 3)
    right_hand = clip_poses[:, 111:156].reshape(len(clip_poses), 15, 3)

    body_6d = _axis_angle_to_rot6d_np(body_pose).reshape(len(clip_poses), -1)
    left_6d = _axis_angle_to_rot6d_np(left_hand).reshape(len(clip_poses), -1)
    right_6d = _axis_angle_to_rot6d_np(right_hand).reshape(len(clip_poses), -1)
    return np.concatenate([body_6d, left_6d, right_6d], axis=1)


def smplx_axis_angle_to_features(clip_poses, pose_rep="axis_angle"):
    pose_rep = normalize_pose_rep(pose_rep)
    if pose_rep == "rot6d":
        return smplx_axis_angle_to_rot6d_features(clip_poses)
    return smplx_axis_angle_to_soke_features(clip_poses)


def _rot6d_to_axis_angle(rot6d):
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot6d.reshape(-1, 6))).reshape(
        *rot6d.shape[:-1], 3
    )


def rot6d_features_to_smplx_axis_angle(features):
    flat = features.reshape(-1, ROT6D_NFEATS)
    device = flat.device
    dtype = flat.dtype

    body_end = len(UPPER_BODY_BODY_POSE_IDXS) * 6
    left_end = body_end + 15 * 6

    body_aa = _rot6d_to_axis_angle(
        flat[:, :body_end].reshape(-1, len(UPPER_BODY_BODY_POSE_IDXS), 6)
    )
    left_aa = _rot6d_to_axis_angle(flat[:, body_end:left_end].reshape(-1, 15, 6))
    right_aa = _rot6d_to_axis_angle(flat[:, left_end:].reshape(-1, 15, 6))

    root_pose = torch.zeros(flat.shape[0], 3, device=device, dtype=dtype)
    body_pose = torch.zeros(flat.shape[0], 63, device=device, dtype=dtype)
    for src_idx, dst_idx in enumerate(UPPER_BODY_BODY_POSE_IDXS):
        body_pose[:, dst_idx * 3:(dst_idx + 1) * 3] = body_aa[:, src_idx]

    jaw_pose = torch.zeros(flat.shape[0], 3, device=device, dtype=dtype)
    expr = torch.zeros(flat.shape[0], 10, device=device, dtype=dtype)
    return root_pose, body_pose, left_aa.reshape(flat.shape[0], -1), right_aa.reshape(flat.shape[0], -1), jaw_pose, expr



def rot6d_features_to_soke_axis_angle_features(features):
    _, body_pose, left_hand, right_hand, jaw_pose, expr = rot6d_features_to_smplx_axis_angle(features)
    body_pose = body_pose.reshape(-1, 21, 3)[:, SOKE_BODY_POSE_IDXS].reshape(-1, 30)
    soke = torch.cat([body_pose, left_hand, right_hand, jaw_pose, expr], dim=-1)
    return soke.reshape(*features.shape[:-1], AXIS_ANGLE_NFEATS)

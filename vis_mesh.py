import copy
import json
import os, pickle; os.environ["PYOPENGL_PLATFORM"] = "egl"
from pathlib import Path
import time
import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.utils.logger import create_logger
from mGPT.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
from mGPT.data.humanml.pose_rep import (
    AXIS_ANGLE_NFEATS,
    ROT6D_NFEATS,
    rot6d_features_to_smplx_axis_angle,
)
import mGPT.render.matplot.plot_3d_global as plot_3d
import pyrender, trimesh
from mGPT.utils.human_models import smpl_x
try:
    from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips, clips_array
except ImportError:
    from moviepy import ImageSequenceClip, VideoFileClip, concatenate_videoclips, clips_array
# from moviepy.video.fx.all import crop
import matplotlib.pyplot as plt
import pandas as pd
import random; random.seed(0)
from PIL import Image
import seaborn as sns; sns.set_style('darkgrid')
import math
import random; random.seed(0)


keys = ['smplx_root_pose', 
        'smplx_body_pose', 
        'smplx_lhand_pose', 
        'smplx_rhand_pose', 
        'smplx_jaw_pose', 
        'smplx_shape', 
        'smplx_expr'
    ]

def _load_axis_angle_stats(mean_path, std_path, device):
    mean = torch.load(mean_path, map_location='cpu', weights_only=True).float()
    std = torch.load(std_path, map_location='cpu', weights_only=True).float()
    mean = mean[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3 + 3 * 11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    if mean.numel() != AXIS_ANGLE_NFEATS or std.numel() != AXIS_ANGLE_NFEATS:
        raise ValueError(
            f'Axis-angle stats must be {AXIS_ANGLE_NFEATS}D after SOKE slicing, '
            f'got mean={mean.numel()} std={std.numel()}.'
        )
    return mean.to(device), std.to(device)


def _load_feature_stats(mean_path, std_path, nfeats, device, name):
    mean = torch.load(mean_path, map_location='cpu', weights_only=True).float().reshape(-1)
    std = torch.load(std_path, map_location='cpu', weights_only=True).float().reshape(-1)
    if mean.numel() != nfeats or std.numel() != nfeats:
        raise ValueError(
            f'{name} stats must be {nfeats}D, got mean={mean.numel()} std={std.numel()}.'
        )
    return mean.to(device), std.to(device)


def load_training_stats(cfg, device):
    h2s_cfg = cfg.DATASET.H2S
    axis_mean, axis_std = _load_axis_angle_stats(
        h2s_cfg.MEAN_PATH,
        h2s_cfg.STD_PATH,
        device,
    )
    stats = {
        AXIS_ANGLE_NFEATS: (axis_mean, axis_std, False),
    }

    pose_rep = str(h2s_cfg.get('POSE_REP', 'axis_angle')).lower()
    if pose_rep in {'rot6d', 'rotation6d', '6d', '6drot'}:
        rot6d_mean_path = h2s_cfg.get('ROT6D_MEAN_PATH', None)
        rot6d_std_path = h2s_cfg.get('ROT6D_STD_PATH', None)
        if rot6d_mean_path and rot6d_std_path:
            rot6d_mean, rot6d_std = _load_feature_stats(
                rot6d_mean_path,
                rot6d_std_path,
                ROT6D_NFEATS,
                device,
                'rot6d',
            )
            stats[ROT6D_NFEATS] = (rot6d_mean, rot6d_std, True)

    print(f'visualization axis-angle stats: {h2s_cfg.MEAN_PATH}, {h2s_cfg.STD_PATH}')
    if ROT6D_NFEATS in stats:
        print(
            'visualization rot6d stats: '
            f'{h2s_cfg.ROT6D_MEAN_PATH}, {h2s_cfg.ROT6D_STD_PATH}'
        )
    return stats


def stats_for_features(features, stats):
    nfeats = features.shape[-1]
    if nfeats not in stats:
        raise ValueError(
            f'No visualization mean/std for {nfeats}D features. '
            f'Known feature dims: {sorted(stats.keys())}.'
        )
    return stats[nfeats]


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        bbox = None

    return bbox


def process_bbox(bbox, img_width, img_height):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = img_width / img_height
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    bbox = bbox.astype(np.float32)
    return bbox


def render_mesh(img, mesh, face, cam_trans, only_mesh=False):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    return_mesh = mesh
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    # material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(0.11, 0.53, 0.8, 0.5))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    # focal, princpt = cam_param['focal'], cam_param['princpt']
    # camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    # scene.add(camera)

    # focal_length = 5000
    # # camera = pyrender.PerspectiveCamera(yfov=np.pi / 55.0)
    # camera = pyrender.camera.IntrinsicsCamera(
    #     fx=focal_length, fy=focal_length,
    #     cx=camera_center[0], cy=camera_center[1])
    # # print(camera.get_projection_matrix(210,260))
    # scene.add(camera, pose=camera_pose)

    camera_center = [1.0*img.shape[1]/2, 1.0*img.shape[0]/2]
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_trans
    camera_pose[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

    focal_length = 5000
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 55.0)
    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    # print(camera.get_projection_matrix(210,260))
    scene.add(camera, pose=camera_pose)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    # light_pose = np.eye(4)
    # light_pose[:3, 3] = np.array([0, -1, 1])
    # scene.add(light, pose=light_pose)
    # light_pose[:3, 3] = np.array([0, 1, 1])
    # scene.add(light, pose=light_pose)
    # light_pose[:3, 3] = np.array([1, 1, 2])
    # scene.add(light, pose=light_pose)

    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
    light_pose = np.eye(4)
    light_pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    img = np.array(img, dtype=np.uint8)
    return img, return_mesh


def get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr):
    batch_size = root_pose.shape[0]
    zero_pose = torch.zeros((1, 3), dtype=body_pose.dtype, device=body_pose.device).repeat(batch_size, 1)
    smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).to(body_pose.device)
    output = smplx_layer(
        betas=shape,
        body_pose=body_pose,
        global_orient=root_pose,
        right_hand_pose=rhand_pose,
        left_hand_pose=lhand_pose,
        jaw_pose=jaw_pose,
        leye_pose=zero_pose,
        reye_pose=zero_pose,
        expression=expr,
    )
    return output.vertices, output.joints

def feats2joints(features, mean, std, rot6d=False):
    # smpl2joints and drop lowerbody
    features = features * std + mean

    shape_param = torch.tensor([[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
                            0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]]).to(features)
    B, T = features.shape[:2]
    shape_param = shape_param.repeat(B, T, 1).view(B*T, -1)

    if rot6d:
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, expr = rot6d_features_to_smplx_axis_angle(features)
    else:
        zero_pose = torch.zeros(*features.shape[:-1], 36).to(features)
        features = torch.cat([zero_pose, features], dim=-1).view(B*T, -1)  # 133 + 36 = 169
        root_pose = features[..., 0:3]
        body_pose = features[..., 3:66]
        lhand_pose = features[..., 66:111]
        rhand_pose = features[..., 111:156]
        jaw_pose = features[..., 156:159]
        expr = features[..., 159:169]

    vertices, joints = get_coord(root_pose=root_pose, body_pose=body_pose,
                                 lhand_pose=lhand_pose, rhand_pose=rhand_pose,
                                 jaw_pose=jaw_pose, shape=shape_param,
                                 expr=expr)
    return vertices, joints


def sample(input,count):
    ss=float(len(input))/count
    return [ input[int(math.floor(i*ss))] for i in range(count)]


def main(save_mesh=False):
    # parse options
    cfg = parse_args(phase="demo")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.USE_GPUS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_stats = load_training_stats(cfg, device)
    axis_mean, axis_std, _ = training_stats[AXIS_ANGLE_NFEATS]
    
    dataset = cfg.DEMO_DATASET
    # visualize parameters
    fps = 18
    h, w = 512, 512
    focal = [5000, 5000]
    princpt = [h/2, w/2]
    bbox = process_bbox([0,0,h,w], h, w)
    focal = [focal[0] / w * bbox[2], focal[1] / h * bbox[3]]
    princpt = [princpt[0] / w * bbox[2] + bbox[0], princpt[1] / h * bbox[3] + bbox[1]]
    cam_trans = np.array([-2.6177440e-03, 0.1, -13], dtype=np.float32)
    save_dir = f'visualize/compare_{dataset}_v4'
    save_mesh_dir = f'visualize/compare_{dataset}_v4'
    os.makedirs(save_dir, exist_ok=True)
    if save_mesh:
        os.makedirs(save_mesh_dir, exist_ok=True)

    if dataset == 'csl':
        baseline = 'results/mgpt/baseline'
        raw_vid_dir = 'data/CSL-Daily/csl-daily'
    elif dataset == 'how2sign':
        baseline = 'results/mgpt/baseline'
        raw_vid_dir = 'data/How2Sign/test/raw_videos'
    elif dataset == 'phoenix':
        baseline = 'data/Phoenix_2014T'
        raw_vid_dir = '/aiarena/group/gmgroup/hongyq/data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    ours = 'results/signspark/SignSparK_v4'
    split = 'test'

    scores_ours = {}
    for rank in range(8):
        score_path = os.path.join(ours, f'{split}_rank_{rank}/test_scores.json')
        if os.path.exists(score_path):
            with open(score_path, 'r') as f:
                scores = json.load(f)
                scores_ours.update(scores)
    
    names = []
    for k,v in scores_ours.items():
        if dataset in list(v.keys())[0]:
            names.append(k)
    random.shuffle(names)
    print('tot num: ', len(names))

    start, end = 0, 20
    for i in range(len(names)):
        if i < start:
            continue
        if i > end:
            break

        n = names[i]
        print(i, n,)
        # for r in range(6):
        #     dir = os.path.join(baseline, f'{split}_rank_{r}')
        #     if os.path.exists(dir) and f"{n.split('/')[-1]}.pkl" in os.listdir(dir):
        #         with open(os.path.join(dir, f"{n.split('/')[-1]}.pkl"), 'rb') as f:
        #             res_base = pickle.load(f)
        #         break
        dir = os.path.join(baseline, f"{split}/{n.split('/')[-1]}")
        clip_poses = np.zeros([len(os.listdir(dir)), 179])
        for idx, file in enumerate(sorted(os.listdir(dir))):
            with open(os.path.join(dir, file), 'rb') as f:
                poses = pickle.load(f)

            pose = np.concatenate([poses[key] for key in keys], 0)
            clip_poses[idx] = pose
        clip_poses = clip_poses[:,(3+3*11):]
        # remove shape
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1) #179-36-10=133
        clip_poses = (clip_poses - axis_mean.detach().cpu().numpy()) / (axis_std.detach().cpu().numpy()+1e-10)
        res_base = {
            'feats_rst': clip_poses.astype(np.float32),
        }
        for r in range(8):
            dir = os.path.join(ours, f'{split}_rank_{r}')
            if os.path.exists(dir) and f"{n.split('/')[-1]}.pkl" in os.listdir(dir):
                with open(os.path.join(dir, f"{n.split('/')[-1]}.pkl"), 'rb') as f:
                    res_ours = pickle.load(f)
                break
        
        feats_ref = res_ours['feats_ref']
        feats_rst_ours = res_ours['feats_rst']
        feats_rst_base = res_base['feats_rst']
        text = res_ours['text']
        print(text)

        ref_mean, ref_std, ref_rot6d = stats_for_features(feats_ref, training_stats)
        rst_mean, rst_std, rst_rot6d = stats_for_features(feats_rst_ours, training_stats)
        base_mean, base_std, base_rot6d = stats_for_features(feats_rst_base, training_stats)
        vertices_ref = feats2joints(torch.from_numpy(feats_ref).to(device).unsqueeze(0), mean=ref_mean, std=ref_std, rot6d=ref_rot6d)[0].cpu().numpy()
        vertices_rst_ours = feats2joints(torch.from_numpy(feats_rst_ours).to(device).unsqueeze(0), mean=rst_mean, std=rst_std, rot6d=rst_rot6d)[0].cpu().numpy()
        vertices_rst_base = feats2joints(torch.from_numpy(feats_rst_base).to(device).unsqueeze(0), base_mean, base_std, rot6d=base_rot6d)[0].cpu().numpy()

        frames = []
        rst_len = feats_rst_ours.shape[0]
        rst_len_base = feats_rst_base.shape[0]

        # read raw video
        gt_frames = []
        if dataset == 'how2sign':
            gt_vid_path = os.path.join(raw_vid_dir, n+'.mp4')
            clip = VideoFileClip(gt_vid_path)
            raw_w, raw_h = clip.size
            clip = clip.crop(width=720, height=720, x_center=raw_w//2, y_center=raw_h//2)
            clip = clip.resize(width=512, height=512)
            for frame in clip.iter_frames():
                frame = Image.fromarray(frame, 'RGB')
                gt_frames.append(frame)
            csv = pd.read_csv('data/How2Sign/test/re_aligned/how2sign_realigned_test_preprocessed_fps.csv')
            raw_fps = csv[csv['SENTENCE_NAME']==n]['fps'].item()
            if raw_fps > 25:
                gt_frames = sample(gt_frames, count=int(25*len(gt_frames)/raw_fps))
        else:
            gt_vid_path = os.path.join(raw_vid_dir, n)
            frame_lst = os.listdir(gt_vid_path)
            frame_lst = sorted(frame_lst)
            for fname in frame_lst:
                img = Image.open(os.path.join(gt_vid_path, fname)).resize((w, h)).convert('RGB')
                gt_frames.append(img)
        ref_len = len(gt_frames)

        print(ref_len, rst_len, rst_len_base)
        if save_mesh and save_mesh_dir:
            cur_mesh_dir = os.path.join(save_mesh_dir, n.split('/')[-1], 'mesh')
            os.makedirs(cur_mesh_dir, exist_ok=True)
        for f_idx in tqdm(range(0, max(ref_len, rst_len))):
            img_gt = gt_frames[min(f_idx, ref_len-1)]
            if save_mesh:
                img_gt.save(os.path.join(cur_mesh_dir, f'{f_idx:03d}.png'))
            img_gt_mesh = np.zeros((h, w, 3), dtype=np.int8)
            img_gt_mesh, mesh_gt = render_mesh(img=img_gt_mesh, mesh=vertices_ref[min(f_idx, vertices_ref.shape[0]-1)], face=smpl_x.face, cam_trans=cam_trans)

            img_pred_base = np.zeros((h, w, 3), dtype=np.int8)
            img_pred_base, mesh_base = render_mesh(img=img_pred_base, mesh=vertices_rst_base[min(f_idx, rst_len_base-1)], face=smpl_x.face, cam_trans=cam_trans)
            
            img_pred = np.zeros((h, w, 3), dtype=np.int8)
            img_pred, mesh_ours = render_mesh(img=img_pred, mesh=vertices_rst_ours[min(f_idx, rst_len-1)], face=smpl_x.face, cam_trans=cam_trans)
            
            frames.append(np.concatenate([img_gt, img_pred_base, img_pred], axis=1))

            if save_mesh:
                mesh_gt.export(os.path.join(cur_mesh_dir, f'{f_idx:03d}_gt.ply'))
                mesh_base.export(os.path.join(cur_mesh_dir, f'{f_idx:03d}_base.ply'))
                mesh_ours.export(os.path.join(cur_mesh_dir, f'{f_idx:03d}_ours.ply'))
        
        save_path = os.path.join(save_dir, f"{i:04d}_{n.split('/')[-1]}.mp4")
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(save_path, fps=fps)


if __name__ == "__main__":
    main(save_mesh=False)
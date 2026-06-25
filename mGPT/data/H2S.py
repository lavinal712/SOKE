import numpy as np
import torch
import os 
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, H2SMotionDatasetVQ, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T
from .humanml.load_data import AXIS_ANGLE_NFEATS, ROT6D_BODY_JOINTS, ROT6D_NFEATS
from .utils import humanml3d_collate
from mGPT.utils.human_models import get_coord
from mGPT.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix


class H2SDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'humanml3d'
        self.name = "humanml3d"
        self.njoints = 22

        self.hparams.dataset_name = cfg.DATASET.H2S.DATASET_NAME
        self.hparams.csl_root = cfg.DATASET.H2S.CSL_ROOT
        self.hparams.phoenix_root = cfg.DATASET.H2S.get('PHOENIX_ROOT', None)
        self.hparams.pred_data_dir = cfg.DATASET.H2S.get('pred_data_dir', False)
        self.hparams.pose_rep = str(cfg.DATASET.H2S.get('POSE_REP', 'axis_angle')).lower()
        self.hparams.handfix = bool(cfg.DATASET.H2S.get('HANDFIX', False))
        self.hparams.BODYONLY = bool(cfg.DATASET.H2S.get('BODYONLY', False))
        if self.hparams.BODYONLY:
            if not self.hparams.handfix:
                raise ValueError("BODYONLY=True requires DATASET.H2S.HANDFIX=True.")
            if self.hparams.pose_rep != 'rot6d':
                raise ValueError("BODYONLY=True is only supported with POSE_REP=rot6d.")
        
        # Path to the dataset
        data_root = cfg.DATASET.H2S.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "re_aligned")
        self.hparams.motion_dir = pjoin(data_root, "poses")
        
        # Mean and std of the dataset
        # if self.hparams.dataset_name == 'how2sign':
        #     mean_path = pjoin(data_root, "h2s_mean.pt")
        #     std_path = pjoin(data_root, "h2s_std.pt")
        # elif self.hparams.dataset_name == 'csl':
        #     mean_path = pjoin(self.hparams.csl_root, "csl_mean.pt")
        #     std_path = pjoin(self.hparams.csl_root, "csl_std.pt")
        # elif self.hparams.dataset_name == 'how2sign_csl':
        #     mean_path = pjoin(self.hparams.csl_root, "h2s_csl_mean.pt")
        #     std_path = pjoin(self.hparams.csl_root, "h2s_csl_std.pt")
        mean_path = cfg.DATASET.H2S.MEAN_PATH
        std_path = cfg.DATASET.H2S.STD_PATH
        axis_mean = torch.load(mean_path).float()
        axis_std = torch.load(std_path).float()
        axis_mean = axis_mean[(3 + 3 * 11):]
        axis_mean = torch.cat([axis_mean[:-20], axis_mean[-10:]], dim=0)
        axis_std = axis_std[(3 + 3 * 11):]
        axis_std = torch.cat([axis_std[:-20], axis_std[-10:]], dim=0)
        self.hparams.save_mean = axis_mean
        self.hparams.save_std = axis_std

        if self.hparams.pose_rep == 'rot6d':
            mean_path = cfg.DATASET.H2S.ROT6D_MEAN_PATH
            std_path = cfg.DATASET.H2S.ROT6D_STD_PATH
            if not os.path.exists(mean_path) or not os.path.exists(std_path):
                raise FileNotFoundError(f"Missing rot6d mean/std: {mean_path}, {std_path}")
            print('rot6d mean path', mean_path, 'std_path: ', std_path)
            self.hparams.mean = torch.load(mean_path).float().reshape(-1)
            self.hparams.std = torch.load(std_path).float().reshape(-1)
            if self.hparams.mean.numel() != ROT6D_NFEATS or self.hparams.std.numel() != ROT6D_NFEATS:
                raise ValueError(
                    f"rot6d mean/std must have {ROT6D_NFEATS} dims, "
                    f"got {self.hparams.mean.numel()} and {self.hparams.std.numel()}."
                )
            self.nfeats = len(ROT6D_BODY_JOINTS) * 6 if self.hparams.BODYONLY else ROT6D_NFEATS
        else:
            print('mean path', mean_path, 'std_path: ', std_path)
            self.hparams.pose_rep = 'axis_angle'
            self.hparams.mean = axis_mean
            self.hparams.std = axis_std
            self.nfeats = AXIS_ANGLE_NFEATS
        
        # Mean and std for fair evaluation
        # dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "Comp_v6_KLD01", "meta")
        # self.hparams.mean_eval = np.load(pjoin(dis_data_root_eval, "mean.npy"))
        # self.hparams.std_eval = np.load(pjoin(dis_data_root_eval, "std.npy"))
        self.hparams.mean_eval = self.hparams.mean
        self.hparams.std_eval = self.hparams.std
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.H2S.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.H2S.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.H2S.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.H2S.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # Dataset switch
        self.DatasetEval = H2SMotionDatasetVQ if cfg.TRAIN.STAGE in ["vae"] else Text2MotionDatasetEval

        if cfg.TRAIN.STAGE in ["vae"]:
            # if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
            self.hparams.win_size = 64
            self.Dataset = H2SMotionDatasetVQ
            # else:
                # self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.H2S.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset

        # Get additional info of the dataset
        # self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
        # self.nfeats = 133  #self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats
        

    def _rot6d_to_axis_angle_features(self, features):
        out_shape = features.shape[:-1]
        body_end = len(ROT6D_BODY_JOINTS) * 6
        nfeats = features.shape[-1]
        if nfeats not in (body_end, ROT6D_NFEATS):
            raise ValueError(f"Expected {body_end} or {ROT6D_NFEATS} rot6d dims, got {nfeats}.")

        flat = features.reshape(-1, nfeats)
        left_end = body_end + 15 * 6

        body_aa = matrix_to_axis_angle(
            rotation_6d_to_matrix(flat[:, :body_end].reshape(-1, len(ROT6D_BODY_JOINTS), 6))
        ).reshape(flat.shape[0], len(ROT6D_BODY_JOINTS), 3)
        if nfeats == ROT6D_NFEATS:
            left_aa = matrix_to_axis_angle(
                rotation_6d_to_matrix(flat[:, body_end:left_end].reshape(-1, 15, 6))
            ).reshape(flat.shape[0], -1)
            right_aa = matrix_to_axis_angle(
                rotation_6d_to_matrix(flat[:, left_end:].reshape(-1, 15, 6))
            ).reshape(flat.shape[0], -1)
        else:
            left_aa = flat.new_zeros(flat.shape[0], 45)
            right_aa = flat.new_zeros(flat.shape[0], 45)

        body_pose = flat.new_zeros(flat.shape[0], 21, 3)
        for src_idx, dst_idx in enumerate(ROT6D_BODY_JOINTS):
            body_pose[:, dst_idx] = body_aa[:, src_idx]
        body_pose = body_pose[:, 11:21].reshape(flat.shape[0], -1)
        jaw_expr = flat.new_zeros(flat.shape[0], 13)
        features = torch.cat([body_pose, left_aa, right_aa, jaw_expr], dim=-1)
        return features.reshape(*out_shape, AXIS_ANGLE_NFEATS)

    def feats2joints(self, features):
        #smpl2joints and drop lowerbody
        if self.hparams.pose_rep == 'rot6d':
            body_end = len(ROT6D_BODY_JOINTS) * 6
            if self.hparams.BODYONLY and features.shape[-1] == body_end:
                mean = self.hparams.mean[:body_end].to(features)
                std = self.hparams.std[:body_end].to(features)
                features = features * std + mean
            else:
                features = self.denormalize(features)
            features = self._rot6d_to_axis_angle_features(features)
        else:
            features = self.denormalize(features)
        # return recover_from_ric(features, self.njoints)

        zero_pose = torch.zeros(*features.shape[:-1], 36).to(features)
        shape_param = torch.tensor([[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172, 
                              0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]]).to(features)
        B, T = features.shape[:2]
        shape_param = shape_param.repeat(B, T, 1).view(B*T, -1)
        # print(features.shape, shape_param.shape)
        features = torch.cat([zero_pose, features], dim=-1).view(B*T, -1)  #133+36=169
        vertices, joints = get_coord(root_pose=features[..., 0:3], body_pose=features[..., 3:66], 
                                     lhand_pose=features[..., 66:111], rhand_pose=features[..., 111:156], 
                                     jaw_pose=features[..., 156:159], shape=shape_param, 
                                     expr=features[..., 159:169])
        return vertices, joints

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        features = features * std + mean
        return features

    def features_for_prediction_save(self, features):
        if self.hparams.pose_rep != 'rot6d':
            return features
        body_end = len(ROT6D_BODY_JOINTS) * 6
        if self.hparams.BODYONLY and features.shape[-1] == body_end:
            mean = self.hparams.mean[:body_end].to(features)
            std = self.hparams.std[:body_end].to(features)
            features = features * std + mean
        else:
            features = self.denormalize(features)
        features = self._rot6d_to_axis_angle_features(features)
        mean = self.hparams.save_mean.to(features)
        std = self.hparams.save_std.to(features)
        return (features - mean) / (std + 1e-10)

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        if self.hparams.pose_rep == 'rot6d':
            return self.features_for_prediction_save(features)
        ori_mean = self.hparams.mean.to(features)
        ori_std = self.hparams.std.to(features)
        eval_mean = self.hparams.mean_eval.to(features)
        eval_std = self.hparams.std_eval.to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list

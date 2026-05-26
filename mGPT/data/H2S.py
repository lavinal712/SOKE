import numpy as np
import torch
import os 
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, H2SMotionDatasetVQ, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T
from .utils import humanml3d_collate
from .humanml.pose_rep import (
    AXIS_ANGLE_NFEATS,
    ROT6D_NFEATS,
    normalize_pose_rep,
    rot6d_features_to_smplx_axis_angle,
    rot6d_features_to_soke_axis_angle_features,
)
from mGPT.utils.human_models import get_coord


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
        self.hparams.pose_rep = normalize_pose_rep(cfg.DATASET.H2S.get("POSE_REP", "axis_angle"))
        self.nfeats = ROT6D_NFEATS if self.hparams.pose_rep == "rot6d" else AXIS_ANGLE_NFEATS

        if self.hparams.pose_rep == "rot6d":
            mean_path = cfg.DATASET.H2S.get("ROT6D_MEAN_PATH", None)
            std_path = cfg.DATASET.H2S.get("ROT6D_STD_PATH", None)
            print('rot6d mean path', mean_path, 'std_path: ', std_path)
            self.hparams.mean = self._load_feature_stat(mean_path, self.nfeats, 0.0, "rot6d mean")
            self.hparams.std = self._load_feature_stat(std_path, self.nfeats, 1.0, "rot6d std")
            self.hparams.save_mean, self.hparams.save_std = self._load_axis_angle_stats(
                cfg.DATASET.H2S.MEAN_PATH, cfg.DATASET.H2S.STD_PATH
            )
        else:
            mean_path = cfg.DATASET.H2S.MEAN_PATH
            std_path = cfg.DATASET.H2S.STD_PATH
            print('mean path', mean_path, 'std_path: ', std_path)
            self.hparams.mean, self.hparams.std = self._load_axis_angle_stats(mean_path, std_path)
            self.hparams.save_mean = self.hparams.mean
            self.hparams.save_std = self.hparams.std
        
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
        cfg.DATASET.NFEATS = self.nfeats
        

    def _load_axis_angle_stats(self, mean_path, std_path):
        mean = torch.load(mean_path).float()
        std = torch.load(std_path).float()
        mean = mean[(3+3*11):]
        mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
        std = std[(3+3*11):]
        std = torch.cat([std[:-20], std[-10:]], dim=0)
        return mean, std

    def _load_feature_stat(self, path, nfeats, fill_value, name):
        if path and os.path.exists(path):
            stat = torch.load(path).float().reshape(-1)
            if stat.numel() != nfeats:
                raise ValueError(f"{name} has {stat.numel()} dims, expected {nfeats}.")
            return stat
        print(f"{name} not provided; using constant {fill_value} stats for {nfeats} dims.")
        return torch.full((nfeats,), fill_value, dtype=torch.float32)

    def features_for_prediction_save(self, features):
        if self.hparams.pose_rep != "rot6d":
            return features
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        raw_rot6d = features * std + mean
        raw_axis_angle = rot6d_features_to_soke_axis_angle_features(raw_rot6d)
        save_mean = self.hparams.save_mean.to(raw_axis_angle)
        save_std = self.hparams.save_std.to(raw_axis_angle)
        return (raw_axis_angle - save_mean) / (save_std + 1e-10)

    def feats2joints(self, features):
        #smpl2joints and drop lowerbody
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        features = features * std + mean
        # return recover_from_ric(features, self.njoints)

        shape_param = torch.tensor([[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172, 
                              0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]]).to(features)
        B, T = features.shape[:2]
        shape_param = shape_param.repeat(B, T, 1).view(B*T, -1)

        if self.hparams.pose_rep == "rot6d":
            root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, expr = rot6d_features_to_smplx_axis_angle(features)
        else:
            zero_pose = torch.zeros(*features.shape[:-1], 36).to(features)
            features = torch.cat([zero_pose, features], dim=-1).view(B*T, -1)  #133+36=169
            root_pose = features[..., 0:3]
            body_pose = features[..., 3:66]
            lhand_pose = features[..., 66:111]
            rhand_pose = features[..., 111:156]
            jaw_pose = features[..., 156:159]
            expr = features[..., 159:169]

        vertices, joints = get_coord(root_pose=root_pose, body_pose=body_pose,
                                     lhand_pose=lhand_pose, rhand_pose=rhand_pose,
                                     jaw_pose=jaw_pose, shape=shape_param, expr=expr)
        return vertices, joints

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
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

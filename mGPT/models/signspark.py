import random

import torch

from mGPT.archs.signspark_unet import SignSpark_UNet
from mGPT.config import instantiate_from_config
from mGPT.models.base import BaseModel
from mGPT.losses.signspark import SignSparkLosses


class SignSpark(BaseModel):
    def __init__(
        self,
        cfg,
        datamodule,
        fm=None,
        motion_vae=None,
        stage="flow_matching",
        debug=True,
        condition="text",
        task="t2m",
        metrics_dict=["TM2TMetrics"],
        nfeats=None,
        text_dim=512,
        hidden_dim=512,
        dim_mults=(1, 2, 4, 8),
        text_encoder_name="M-CLIP/XLM-Roberta-Large-Vit-B-32",
        freeze_text_encoder=True,
        keyframe_strategy="uniform_segments",
        keyframe_ratio=0.12,
        keyframe_num=None,
        min_keyframes=3,
        keyframe_segment_size=16,
        include_endpoints=True,
        sampling_steps=16,
        guidance_scale=1.5,
        inference_mode="kf2p",
        cfg_text_dropout=0.1,
        cfg_keyframe_dropout=0.1,
        cfg_no_keyframe_dropout=0.0,
        cfg_guidance_dropout=0.1,
        **kwargs,
    ):
        self.save_hyperparameters(ignore="datamodule", logger=False)
        self.datamodule = datamodule
        super().__init__()

        self.nfeats = nfeats if nfeats is not None else datamodule.nfeats
        self.text_encoder_name = text_encoder_name
        self.freeze_text_encoder = freeze_text_encoder

        if motion_vae is not None:
            self.vae = instantiate_from_config(motion_vae)
            if "flow_matching" in self.hparams.stage or self.hparams.stage == "signspark":
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False

        self.text_model, self.text_tokenizer = self.build_text_encoder(text_encoder_name)
        if freeze_text_encoder:
            self.text_model.eval()
            for p in self.text_model.parameters():
                p.requires_grad = False

        if fm is not None:
            self.fm = instantiate_from_config(fm)
        else:
            self.fm = SignSpark_UNet(
                input_dim=self.nfeats,
                text_dim=text_dim,
                cond_dim=hidden_dim,
                dim=hidden_dim,
                dim_mults=dim_mults,
                cfg_text_dropout=cfg_text_dropout,
                cfg_scale=guidance_scale,
                keyframe_strategy=keyframe_strategy,
            )

        self._losses = torch.nn.ModuleDict({
            split: SignSparkLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        self.feats2joints = datamodule.feats2joints

    def build_text_encoder(self, model_name):
        try:
            from multilingual_clip import pt_multilingual_clip
            import transformers
        except ImportError as exc:
            raise ImportError(
                "SignSpark requires `multilingual-clip` and `transformers` "
                "for the Multilingual-CLIP text encoder."
            ) from exc

        text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        return text_model, tokenizer

    def encode_text(self, texts, device=None, dtype=None):

        def _text_model_forward(texts, tokenizer, text_model):
            txt_tok = tokenizer(texts, padding=True, return_tensors='pt')
            txt_tok = {k: v.to(text_model.device) for k, v in txt_tok.items()}
            text_emb = text_model.transformer(**txt_tok)[0]
            attention_mask = txt_tok['attention_mask']
            text_emb = (text_emb * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1)[:, None]
            text_emb = text_model.LinearTransformation(text_emb)
            return text_emb

        if torch.is_tensor(texts):
            text_emb = texts
        elif self.freeze_text_encoder:
            self.text_model.eval()
            with torch.no_grad():
                # text_emb = self.text_model.forward(texts, self.text_tokenizer)
                text_emb = _text_model_forward(texts, self.text_tokenizer, self.text_model)
        else:
            # text_emb = self.text_model.forward(texts, self.text_tokenizer)
            text_emb = _text_model_forward(texts, self.text_tokenizer, self.text_model)

        if device is not None:
            text_emb = text_emb.to(device=device)
        if dtype is not None:
            text_emb = text_emb.to(dtype=dtype)
        return text_emb

    def build_keyframe_mask(self, lengths, max_len, device):
        if not torch.is_tensor(lengths):
            lengths_tensor = torch.tensor(lengths, device=device)
        else:
            lengths_tensor = lengths.to(device=device)
        lengths_tensor = lengths_tensor.long()

        mask = torch.zeros(len(lengths_tensor), max_len, 1, device=device)
        for i, cur_len in enumerate(lengths_tensor.tolist()):
            cur_len = max(int(cur_len), 1)
            if self.hparams.keyframe_num is not None:
                num_keyframes = int(self.hparams.keyframe_num)
            elif self.hparams.keyframe_ratio is not None:
                num_keyframes = int(round(cur_len * float(self.hparams.keyframe_ratio)))
            else:
                num_keyframes = int(round(cur_len / max(int(self.hparams.keyframe_segment_size), 1)))

            num_keyframes = max(num_keyframes, int(self.hparams.min_keyframes))
            num_keyframes = min(num_keyframes, cur_len)

            if self.hparams.keyframe_strategy == "random":
                ids = torch.randperm(cur_len, device=device)[:num_keyframes]
                ids = ids.sort().values
            else:
                ids = torch.linspace(0, cur_len - 1, num_keyframes, device=device).round().long()

            if self.hparams.include_endpoints and cur_len > 1:
                endpoints = torch.tensor([0, cur_len - 1], device=device, dtype=torch.long)
                ids = torch.unique(torch.cat([ids, endpoints])).clamp(0, cur_len - 1)

            mask[i, ids, 0] = 1.0

        return mask

    def build_length_mask(self, lengths, max_len, device):
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=device)
        lengths = lengths.to(device=device, dtype=torch.long)
        ids = torch.arange(max_len, device=device)
        return (ids[None, :] < lengths[:, None]).unsqueeze(-1).float()

    def maybe_drop_keyframes(self, keyframe_mask, lengths):
        if not self.training:
            return keyframe_mask

        # Sparse keyframe augmentation for the conditional branch only.
        if self.hparams.cfg_keyframe_dropout > 0:
            keep = torch.rand_like(keyframe_mask) >= float(self.hparams.cfg_keyframe_dropout)
            keyframe_mask = keyframe_mask * keep.to(keyframe_mask.dtype)

        for i, cur_len in enumerate(lengths):
            cur_len = int(cur_len)
            if keyframe_mask[i, :cur_len].sum() == 0:
                keyframe_mask[i, 0, 0] = 1.0
        return keyframe_mask

    def build_guidance_dropout(self, b_size, device):
        if not self.training:
            return None
        dropout = float(self.hparams.cfg_guidance_dropout)
        if dropout <= 0:
            return None
        return torch.rand(b_size, device=device) < dropout

    def use_no_keyframes(self):
        mode = str(self.hparams.inference_mode).lower()
        if hasattr(self.hparams, "cfg") and self.hparams.cfg.TEST.get("INFERENCE_MODE", None):
            mode = str(self.hparams.cfg.TEST.INFERENCE_MODE).lower()
        return mode in ["text2pose", "text_to_pose", "t2p", "text_only", "no_keyframe", "no_keyframes"]

    def make_flow_inputs(self, feats_ref, lengths, keyframe_mask=None):
        b_size, nframes, _ = feats_ref.shape
        device = feats_ref.device

        if keyframe_mask is None:
            keyframe_mask = self.build_keyframe_mask(lengths, nframes, device)
        elif keyframe_mask.dim() == 2:
            keyframe_mask = keyframe_mask.unsqueeze(-1)
        keyframe_mask = keyframe_mask.to(device=device, dtype=feats_ref.dtype)
        keyframe_mask = self.maybe_drop_keyframes(keyframe_mask, lengths)

        length_mask = self.build_length_mask(lengths, nframes, device).to(feats_ref.dtype)
        keyframe_mask = keyframe_mask * length_mask

        noise = torch.randn_like(feats_ref) * length_mask
        t = torch.rand(b_size, device=device, dtype=feats_ref.dtype).clamp(0.0, 1.0 - 1e-4)
        t_view = t[:, None, None]

        x_t = t_view * feats_ref + (1.0 - t_view) * noise
        control = keyframe_mask * feats_ref + (1.0 - keyframe_mask) * x_t
        control = control * length_mask

        return control, t, keyframe_mask, noise

    def fm_forward(
        self,
        text_emb,
        motion,
        t,
        lengths,
        tasks=None,
        keyframe_mask=None,
        uncond=False,
    ):
        if uncond and hasattr(self.fm, "null_text"):
            outputs = self.fm(
                text_emb,
                motion,
                t,
                lengths,
                tasks,
                keyframe_mask=keyframe_mask,
                uncond=True,
            )
        else:
            outputs = self.fm(
                text_emb,
                motion,
                t,
                lengths,
                tasks,
                keyframe_mask=keyframe_mask,
            )
        return outputs["pred_velocity"] if isinstance(outputs, dict) else outputs

    def predict_velocity(
        self,
        text_emb,
        motion,
        t,
        lengths,
        tasks=None,
        keyframe_mask=None,
    ):
        velocity = self.fm_forward(
            text_emb,
            motion,
            t,
            lengths,
            tasks,
            keyframe_mask=keyframe_mask,
            uncond=False,
        )

        guidance_scale = float(self.hparams.guidance_scale)
        if self.training or guidance_scale == 1.0 or not hasattr(self.fm, "null_text"):
            return velocity

        velocity_uncond = self.fm_forward(
            text_emb,
            motion,
            t,
            lengths,
            tasks,
            keyframe_mask=keyframe_mask,
            uncond=True,
        )
        return velocity_uncond + guidance_scale * (velocity - velocity_uncond)

    def sample_cfm(self, text_emb, feats_ref, lengths, tasks=None, batch=None, keyframe_mask=None):
        b_size, nframes, _ = feats_ref.shape
        device = feats_ref.device
        dtype = feats_ref.dtype

        if self.use_no_keyframes():
            keyframe_mask = torch.zeros(b_size, nframes, 1, device=device, dtype=dtype)
        elif keyframe_mask is None:
            keyframe_mask = self.build_keyframe_mask(lengths, nframes, device)
        elif keyframe_mask.dim() == 2:
            keyframe_mask = keyframe_mask.unsqueeze(-1)
        keyframe_mask = keyframe_mask.to(device=device, dtype=dtype)

        length_mask = self.build_length_mask(lengths, nframes, device).to(dtype)
        keyframe_mask = keyframe_mask * length_mask

        sample = torch.randn_like(feats_ref) * length_mask
        sample = keyframe_mask * feats_ref + (1.0 - keyframe_mask) * sample

        sampling_steps = max(int(self.hparams.sampling_steps), 1)
        time_grid = torch.linspace(0.0, 1.0, sampling_steps + 1, device=device, dtype=dtype)
        for step in range(sampling_steps):
            t = torch.full((b_size,), time_grid[step].item(), device=device, dtype=dtype)
            dt = time_grid[step + 1] - time_grid[step]

            sample = keyframe_mask * feats_ref + (1.0 - keyframe_mask) * sample
            sample = sample * length_mask
            velocity = self.predict_velocity(
                text_emb,
                sample,
                t,
                lengths,
                tasks,
                keyframe_mask=keyframe_mask,
            )
            sample = sample + dt * velocity
            sample = keyframe_mask * feats_ref + (1.0 - keyframe_mask) * sample
            sample = sample * length_mask

        return sample, keyframe_mask

    def forward(self, batch, task="t2m"):
        return self.val_t2m_forward(batch)

    def train_fm_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch.get("tasks", None)
        all_captions = batch.get("all_captions", None)

        if self.hparams.condition == "caption" and all_captions is not None:
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]

        text_emb = self.encode_text(texts, device=feats_ref.device, dtype=feats_ref.dtype)
        control, t, keyframe_mask, noise = self.make_flow_inputs(
            feats_ref,
            lengths,
            keyframe_mask=batch.get("keyframe_mask", None),
        )
        drop_mask = self.build_guidance_dropout(feats_ref.shape[0], feats_ref.device)

        fm_outputs = self.fm(
            text_emb,
            control,
            t,
            lengths,
            tasks,
            keyframe_mask=keyframe_mask,
            drop_mask=drop_mask,
        )
        pred_velocity = fm_outputs["pred_velocity"] if isinstance(fm_outputs, dict) else fm_outputs
        remaining = (1.0 - t[:, None, None]).clamp_min(1e-4)
        m_rst = control + remaining * pred_velocity

        return {
            "m_ref": feats_ref,
            "m_rst": m_rst,
            "pred_velocity": pred_velocity,
            "control": control,
            "noise": noise,
            "t": t,
            "length": lengths,
            "keyframe_mask": keyframe_mask,
        }

    @torch.no_grad()
    def val_t2m_forward(self, batch, vis=False):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch.get("tasks", None)

        text_emb = self.encode_text(texts, device=feats_ref.device, dtype=feats_ref.dtype)
        feats_rst, keyframe_mask = self.sample_cfm(
            text_emb,
            feats_ref,
            lengths,
            tasks=tasks,
            batch=batch,
            keyframe_mask=batch.get("keyframe_mask", None),
        )

        vertices_ref, joints_ref = self.feats2joints(feats_ref)
        vertices_rst, joints_rst = self.feats2joints(feats_rst)

        feats_ref_eval = self.datamodule.renorm4t2m(feats_ref)
        feats_rst_eval = self.datamodule.renorm4t2m(feats_rst)

        return {
            "m_ref": feats_ref_eval,
            "m_rst": feats_rst_eval,
            "m_ref_model": feats_ref,
            "m_rst_model": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "length": lengths,
            "lengths_rst": lengths,
            "keyframe_mask": keyframe_mask,
        }

    def val_m2t_forward(self, batch):
        pass

    def val_m2m_forward(self, batch, task="pred"):
        pass

    def allsplit_step(self, split: str, batch, batch_idx):
        loss = None
        lengths = batch["length"]
        src = batch["src"]
        name = batch["name"]

        if self.hparams.stage in ["flow_matching", "signspark"] and split in ["train"]:
            rs_set = self.train_fm_forward(batch)
            loss = self._losses["losses_" + split].update(rs_set)

        if split in ["val", "test"]:
            if self.hparams.stage in ["flow_matching", "signspark"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                    getattr(self.metrics, "TM2TMetrics").update(
                        feats_rst=rs_set["m_rst"],
                        feats_ref=rs_set["m_ref"],
                        joints_rst=rs_set["joints_rst"],
                        joints_ref=rs_set["joints_ref"],
                        vertices_rst=rs_set["vertices_rst"],
                        vertices_ref=rs_set["vertices_ref"],
                        lengths=lengths,
                        lengths_rst=rs_set["lengths_rst"],
                        split=split,
                        src=src,
                        name=name,
                    )
                elif self.hparams.task == "m2t":
                    rs_set_m2t = self.val_m2t_forward(batch)
                    getattr(self.metrics, "M2TMetrics").update(
                        pred_texts=rs_set_m2t["t_pred"],
                        gt_texts=rs_set_m2t["t_ref"],
                        lengths=rs_set_m2t["length"],
                        src=src,
                    )

        if split in ["test"]:
            if self.hparams.stage in ["flow_matching", "signspark"]:
                feats_ref_save = self.datamodule.features_for_prediction_save(
                    rs_set.get("m_ref_model", rs_set["m_ref"])
                )
                feats_rst_save = self.datamodule.features_for_prediction_save(
                    rs_set.get("m_rst_model", rs_set["m_rst"])
                )
                return {
                    "name": name,
                    "feats_ref": feats_ref_save,
                    "feats_rst": feats_rst_save,
                    "lengths": batch["length"],
                    "lengths_rst": rs_set["lengths_rst"],
                    "text": batch["text"],
                }

        return loss

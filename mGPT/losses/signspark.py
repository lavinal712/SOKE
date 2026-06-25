import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLosses


def create_mask(length, max_len, device):
    if not torch.is_tensor(length):
        length = torch.tensor(length, device=device)
    length = length.to(device=device, dtype=torch.long)

    ids = torch.arange(max_len, device=device)
    mask = ids[None, :] < length[:, None]
    return mask.unsqueeze(-1).float()


class SignSparKLosses(BaseLosses):
    def __init__(self, cfg, stage, num_joints, **kwargs):
        self.stage = stage
        self.recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS
        self.body_dim = cfg.LOSS.get("BODY_DIM", None)
        self.hand_weight = cfg.LOSS.get("HAND_WEIGHT", 1.0)

        losses = []
        params = {}
        if stage in ["flow_matching", "signspark"]:
            losses = ["cfm_loss", "recon_loss", "vel_loss"]
            params["cfm_loss"] = cfg.LOSS.get(
                "LAMBDA_CFM", cfg.model.params.get("flow_loss_weight", 1.0)
            )
            params["recon_loss"] = cfg.LOSS.get(
                "LAMBDA_RECON", cfg.model.params.get("recons_loss_weight", 1.0)
            )
            params["vel_loss"] = cfg.LOSS.get(
                "LAMBDA_VELOCITY", cfg.model.params.get("velocity_loss_weight", 0.5)
            )

        # Loss values are masked manually below; BaseLosses handles buffers/logging.
        super().__init__(
            cfg,
            losses,
            params,
            {loss: nn.MSELoss for loss in losses},
            num_joints,
            **kwargs,
        )

    def motion_loss(self, pred, target, length=None, loss_type="l2"):
        if loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        elif loss_type == "l1_smooth":
            loss = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            raise NotImplementedError(f"Loss {loss_type} not implemented.")

        if self.body_dim is not None and self.hand_weight != 1.0:
            loss[..., self.body_dim:] *= self.hand_weight

        if length is not None:
            mask = create_mask(length, pred.shape[1], pred.device)
            loss = loss * mask
            return loss.sum() / mask.expand_as(loss).sum().clamp_min(1.0)

        return loss.mean()

    def update_loss(self, name, value):
        getattr(self, name).add_(value.detach())
        return self._params[name] * value

    def update(self, rs_set):
        """Update SignSparK CFM, reconstruction, and velocity losses."""
        if "outputs" in rs_set and isinstance(rs_set["outputs"], dict):
            rs_set = rs_set["outputs"]

        if self.stage not in ["flow_matching", "signspark"]:
            return torch.tensor(0.0, device=self.total.device)

        m_ref = rs_set["m_ref"]
        m_rst = rs_set["m_rst"]
        pred_velocity = rs_set["pred_velocity"]
        target_velocity = rs_set.get("target_velocity", None)
        if target_velocity is None:
            t = rs_set["t"]
            while t.dim() < m_ref.dim():
                t = t.unsqueeze(-1)
            target_velocity = (m_ref - rs_set["control"]) / (1.0 - t).clamp_min(1e-4)

        keyframe_mask = rs_set.get("keyframe_mask", None)
        if keyframe_mask is not None:
            if keyframe_mask.dim() == 2:
                keyframe_mask = keyframe_mask.unsqueeze(-1)
            keyframe_mask = keyframe_mask.to(device=m_ref.device, dtype=m_ref.dtype)
            target_velocity = target_velocity * (1.0 - keyframe_mask)

        lengths = rs_set.get("length", None)
        cfm_loss = self.motion_loss(pred_velocity, target_velocity, lengths, "l2")
        recon_loss = self.motion_loss(m_rst, m_ref, lengths, self.recons_loss)

        vel_rst = m_rst[:, 1:] - m_rst[:, :-1]
        vel_ref = m_ref[:, 1:] - m_ref[:, :-1]
        vel_lengths = None
        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=m_ref.device)
            vel_lengths = lengths.to(device=m_ref.device, dtype=torch.long).clamp_min(1) - 1
        vel_loss = self.motion_loss(vel_rst, vel_ref, vel_lengths, self.recons_loss)

        total = self.update_loss("cfm_loss", cfm_loss)
        total += self.update_loss("recon_loss", recon_loss)
        total += self.update_loss("vel_loss", vel_loss)
        self.total += total.detach()
        self.count += 1

        return total

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLosses


HAND_START_IDX = 66


def create_mask(length, max_len, device):
    if not torch.is_tensor(length):
        length = torch.tensor(length, device=device)
    length = length.to(device=device, dtype=torch.long)

    ids = torch.arange(max_len, device=device)
    mask = ids[None, :] < length[:, None]
    return mask.unsqueeze(-1).float()


class SignSparkLosses(BaseLosses):
    def __init__(self, cfg, stage, num_joints, **kwargs):
        self.stage = stage
        self.recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS

        losses = []
        params = {}
        if stage in ["flow_matching", "signspark"]:
            losses.append("cfm_loss")
            params["cfm_loss"] = cfg.LOSS.get(
                "LAMBDA_CFM",
                cfg.model.params.get("flow_loss_weight", 1.0),
            )

            losses.append("recon_loss")
            params["recon_loss"] = cfg.LOSS.get(
                "LAMBDA_RECON",
                cfg.model.params.get("recons_loss_weight", 1.0),
            )

            losses.append("vel_loss")
            params["vel_loss"] = cfg.LOSS.get(
                "LAMBDA_VELOCITY",
                cfg.model.params.get("velocity_loss_weight", 0.1),
            )

            losses.append("hand_acc_loss")
            params["hand_acc_loss"] = cfg.LOSS.get(
                "LAMBDA_HAND_ACC",
                cfg.model.params.get("hand_acc_loss_weight", 0.0),
            )

        # BaseLosses instantiates these, but SignSparK uses masked losses below.
        losses_func = {loss: nn.MSELoss for loss in losses}

        super().__init__(cfg, losses, params, losses_func, num_joints, **kwargs)

    def motion_loss(self, pred, target, length=None, loss_type="l2"):
        if loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        elif loss_type == "l1_smooth":
            loss = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            raise NotImplementedError(f"Loss {loss_type} not implemented.")

        if length is not None:
            mask = create_mask(length, pred.shape[1], pred.device)
            loss = loss * mask
            return loss.sum() / mask.expand_as(loss).sum().clamp_min(1.0)

        return loss.mean()

    def update_loss(self, loss_name, loss_value):
        getattr(self, loss_name).add_(loss_value.detach())
        return self._params[loss_name] * loss_value

    def update(self, rs_set):
        if "outputs" in rs_set and isinstance(rs_set["outputs"], dict):
            rs_set = rs_set["outputs"]

        if self.stage not in ["flow_matching", "signspark"]:
            return torch.tensor(0.0, device=self.total.device)

        m_ref = rs_set["m_ref"]
        pred_velocity = rs_set["pred_velocity"]
        control = rs_set["control"]
        noise = rs_set["noise"]
        keyframe_mask = rs_set.get("keyframe_mask", None)
        t = rs_set["t"]
        length = rs_set.get("length", None)

        while t.dim() < pred_velocity.dim():
            t = t.unsqueeze(-1)

        remaining = (1.0 - t).clamp_min(1e-4)
        target_velocity = m_ref - noise.to(device=m_ref.device, dtype=m_ref.dtype)
        if keyframe_mask is not None:
            if keyframe_mask.dim() == 2:
                keyframe_mask = keyframe_mask.unsqueeze(-1)
            keyframe_mask = keyframe_mask.to(device=m_ref.device, dtype=m_ref.dtype)
            target_velocity = target_velocity * (1.0 - keyframe_mask)
        cfm_loss = self.motion_loss(
            pred_velocity,
            target_velocity,
            length=length,
            loss_type="l2",
        )

        m_rst = control + remaining * pred_velocity
        recon_loss = self.motion_loss(
            m_rst,
            m_ref,
            length=length,
            loss_type=self.recons_loss,
        )

        vel_rst = m_rst[:, 1:] - m_rst[:, :-1]
        vel_ref = m_ref[:, 1:] - m_ref[:, :-1]
        vel_length = None
        if length is not None:
            if not torch.is_tensor(length):
                length = torch.tensor(length, device=m_ref.device)
            vel_length = length.to(device=m_ref.device, dtype=torch.long).clamp_min(1) - 1

        vel_loss = self.motion_loss(
            vel_rst,
            vel_ref,
            length=vel_length,
            loss_type=self.recons_loss,
        )

        hand_rst = m_rst[..., HAND_START_IDX:]
        hand_ref = m_ref[..., HAND_START_IDX:]
        hand_acc_rst = hand_rst[:, 2:] - 2.0 * hand_rst[:, 1:-1] + hand_rst[:, :-2]
        hand_acc_ref = hand_ref[:, 2:] - 2.0 * hand_ref[:, 1:-1] + hand_ref[:, :-2]
        hand_acc_length = None
        if length is not None:
            hand_acc_length = length.to(device=m_ref.device, dtype=torch.long).clamp_min(2) - 2

        hand_acc_loss = self.motion_loss(
            hand_acc_rst,
            hand_acc_ref,
            length=hand_acc_length,
            loss_type=self.recons_loss,
        )

        total = self.update_loss("cfm_loss", cfm_loss)
        total += self.update_loss("recon_loss", recon_loss)
        total += self.update_loss("vel_loss", vel_loss)
        total += self.update_loss("hand_acc_loss", hand_acc_loss)

        rs_set["m_rst"] = m_rst

        self.total += total.detach()
        self.count += 1

        return total

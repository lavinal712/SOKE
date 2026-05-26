import logging
import math
import os
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint

from .tools.embeddings import Timesteps, TimestepEmbedding
from mGPT.utils.temos_utils import lengths_to_mask


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self,
                 inp_channels,
                 out_channels,
                 kernel_size,
                 n_groups=8,
                 zero=False):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2),
            # adding the height dimension for group norm
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

        if zero:
            # zero init the convolution
            nn.init.zeros_(self.block[0].weight)
            nn.init.zeros_(self.block[0].bias)

    def forward(self, x):
        """
        Args:
            x: [n, c, l]
        """
        return self.block(x)


class Conv1dAdaGNBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2),
            # adding the height dimension for group norm
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
        )
        self.block2 = nn.Mish()

    def forward(self, x, c):
        """
        Args:
            x: [n, nfeat, l]
            c: [n, ncond, 1]
        """
        scale, shift = c.chunk(2, dim=1)
        x = self.block1(x)
        x = ada_shift_scale(x, shift, scale)
        x = self.block2(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads
                                       ), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)


def ada_shift_scale(x, shift, scale):
    return x * (1 + scale) + shift


class ResidualTemporalBlock(nn.Module):
    def __init__(self,
                 inp_channels,
                 out_channels,
                 embed_dim,
                 kernel_size=5,
                 adagn=False,
                 zero=False):
        super().__init__()
        self.adagn = adagn

        self.blocks = nn.ModuleList([
            # adagn only the first conv (following guided-diffusion)
            (Conv1dAdaGNBlock(inp_channels, out_channels, kernel_size) if adagn
             else Conv1dBlock(inp_channels, out_channels, kernel_size)),
            Conv1dBlock(out_channels, out_channels, kernel_size, zero=zero),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            # adagn = scale and shift
            nn.Linear(embed_dim, out_channels * 2 if adagn else out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        if adagn:
            # zero the linear layer in the time_mlp so that the default behaviour is identity
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        cond = self.time_mlp(t)
        if self.adagn:
            # using adagn
            out = self.blocks[0](x, cond)
        else:
            # using addition
            out = self.blocks[0](x) + cond
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
            self,
            input_dim,
            cond_dim,
            dim=256,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            adagn=False,
            zero=False,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        print('dims: ', dims, 'mults: ', dim_mults)
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            # SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    ResidualTemporalBlock(dim_in,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    ResidualTemporalBlock(dim_out,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                    if attention else nn.Identity(),
                    Downsample1d(dim_out) if not is_last else nn.Identity()
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim,
                                                mid_dim,
                                                embed_dim=time_dim,
                                                adagn=adagn,
                                                zero=zero)
        self.mid_attn = Residual(
            PreNorm(mid_dim,
                    LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim,
                                                mid_dim,
                                                embed_dim=time_dim,
                                                adagn=adagn,
                                                zero=zero)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # print(dim_out, dim_in)
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    ResidualTemporalBlock(dim_out * 2,
                                          dim_in,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    ResidualTemporalBlock(dim_in,
                                          dim_in,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                    if attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))

        # use the last dim_in to support the case where the mult doesn't start with 1.
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_in, dim_in, kernel_size=5),
            nn.Conv1d(dim_in, input_dim, 1),
        )

        if zero:
            # zero the convolution in the final conv
            nn.init.zeros_(self.final_conv[1].weight)
            nn.init.zeros_(self.final_conv[1].bias)

    def forward(self, x, cond):
        '''
            x : [ seqlen x batch x dim ]
            cons: [ batch x cond_dim]
        '''

        x = einops.rearrange(x, 's b d -> b d s')
        # print('x:', x.shape)

        c = self.time_mlp(cond)
        # print('c:', c.shape)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        # print('x:', x.shape)

        x = einops.rearrange(x, 'b d s -> s b d')
        return x


def cal_concat_multiple(in1, in2, multiple):
    """
    calculate the output channels of the concatenation of the two inputs while keeping the output channels a multiple of the given number
    """
    a = (in1 + in2) / multiple
    return int((1 - (a - math.floor(a))) * multiple + in1 + in2)


class TemporalUnetLarge(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        dim=256,
        dim_mults=(1, 2, 4, 8),
        out_mult=8,
        attention=False,
        adagn=False,
        zero=False,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        print('dims: ', dims, 'mults: ', dim_mults)
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            # SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    ResidualTemporalBlock(dim_in,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    ResidualTemporalBlock(dim_out,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                    if attention else nn.Identity(),
                    Downsample1d(dim_out) if not is_last else nn.Identity()
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim,
                                                mid_dim,
                                                embed_dim=time_dim,
                                                adagn=adagn,
                                                zero=zero)
        self.mid_attn = Residual(
            PreNorm(mid_dim,
                    LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim,
                                                mid_dim,
                                                embed_dim=time_dim,
                                                adagn=adagn,
                                                zero=zero)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # print(dim_out, dim_in)
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    ResidualTemporalBlock(dim_out * 2,
                                          dim_in,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    ResidualTemporalBlock(dim_in,
                                          dim_in,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                    if attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))

        # use the last dim_in to support the case where the mult doesn't start with 1.
        final_in = cal_concat_multiple(dim_in, input_dim, out_mult)
        # only temporary
        final_type = 4  # NOTE: flag.LARGE_OUT_TYPE 
        if final_type == 1:
            print('using final type 1')
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(dim_in + input_dim, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          out_mult * input_dim,
                          5,
                          padding=2,
                          groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim, input_dim, 1,
                          groups=input_dim),
            )
        elif final_type == 2:
            # more kernels of size 5
            print('using final type 2')
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(dim_in + input_dim, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          out_mult * input_dim,
                          5,
                          padding=2,
                          groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim,
                          input_dim,
                          5,
                          padding=2,
                          groups=input_dim),
            )
        elif final_type == 3:
            # all kernels of size 5
            print('using final type 3')
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(dim_in + input_dim, final_in, 5, padding=2),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          out_mult * input_dim,
                          5,
                          padding=2,
                          groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim,
                          input_dim,
                          5,
                          padding=2,
                          groups=input_dim),
            )
        else:
            raise NotImplementedError()

        if zero:
            # zero the convolution in the final conv
            nn.init.zeros_(self.final_conv[-1].weight)
            nn.init.zeros_(self.final_conv[-1].bias)

    def forward(self, x, cond):
        '''
            x : [ seqlen x batch x dim ]
            cons: [ batch x cond_dim]
        '''

        x = einops.rearrange(x, 's b d -> b d s')
        src = x
        # print('x:', x.shape)

        c = self.time_mlp(cond)
        # print('c:', c.shape)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)

        # [batch, last_dim + in_dim, seqlen]
        x = torch.concat([x, src], dim=1)
        # [batch, in_dim, seqlen]
        x = self.final_conv(x)
        # print('x:', x.shape)

        x = einops.rearrange(x, 'b d s -> s b d')
        return x


class SignSpark_UNet(nn.Module):
    def __init__(
        self,
        model_path: str = "",
        model_type: str = "unet",
        stage: str = "flow_matching",
        framerate: float = 20.0,
        input_dim: int = 263,
        cond_dim: int = 1024,
        dim: int = 256,
        dim_mults: Tuple[int] = (1, 2, 4, 8),
        out_mult: int = 8,
        attention: bool = False,
        adagn: bool = False,
        zero: bool = False,
        text_dim: int = 768,
        time_embed_dim: int = 256,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        cfg_dropout: float = 0.1,
        cfg_text_dropout: Optional[float] = None,
        cfg_scale: float = 1.5,
        keyframe_strategy: str = "random",
        **kwargs,
    ):
        super().__init__()

        self.model_path = model_path
        self.model_type = model_type
        self.stage = stage
        self.framerate = framerate
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.cfg_dropout = cfg_dropout if cfg_text_dropout is None else cfg_text_dropout
        self.cfg_scale = cfg_scale
        self.keyframe_strategy = keyframe_strategy
        self.downsample_factor = 2 ** max(len(dim_mults) - 1, 0)

        self.text_proj = nn.Linear(text_dim, cond_dim)
        self.null_text = nn.Parameter(torch.zeros(1, cond_dim))
        self.time_proj = Timesteps(num_channels=time_embed_dim,
                                  flip_sin_to_cos=flip_sin_to_cos,
                                  downscale_freq_shift=freq_shift)
        self.time_embedding = TimestepEmbedding(channel=time_embed_dim, time_embed_dim=cond_dim)
        self.motion_proj = nn.Linear(input_dim + 1, input_dim)

        if model_type == "unet":
            self.unet = TemporalUnet(
                input_dim=input_dim,
                cond_dim=cond_dim,
                dim=dim,
                dim_mults=dim_mults,
                attention=attention,
                adagn=adagn,
                zero=zero,
            )
        elif model_type == "unet_large":
            self.unet = TemporalUnetLarge(
                input_dim=input_dim,
                cond_dim=cond_dim,
                dim=dim,
                dim_mults=dim_mults,
                out_mult=out_mult,
                attention=attention,
                adagn=adagn,
                zero=zero,
            )
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")

    def _build_motion_mask(self, lengths, max_len, device):
        if lengths is None:
            return None
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=device)
        lengths = lengths.to(device=device, dtype=torch.long)
        ids = torch.arange(max_len, device=device)
        return (ids[None, :] < lengths[:, None]).unsqueeze(-1).float()

    def forward(
        self,
        texts: torch.Tensor,
        motion: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: List[int],
        tasks: dict = None,
        keyframe_mask: Optional[torch.Tensor] = None,
        uncond: bool = False,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            texts: Multilingual-CLIP sentence embeddings, [B, text_dim].
            motion: keyframe-conditioned control sequence C_t, [B, T, D].
            timesteps: CFM timestep t, [B] or broadcastable from [B, 1, 1].
            keyframe_mask: sparse keyframe indicator, [B, T] or [B, T, 1].
        """
        b_size, nframes, _ = motion.shape

        if keyframe_mask is None:
            keyframe_mask = torch.zeros(b_size, nframes, 1, device=motion.device, dtype=motion.dtype)
        elif keyframe_mask.dim() == 2:
            keyframe_mask = keyframe_mask.unsqueeze(-1)
        keyframe_mask = keyframe_mask.to(device=motion.device, dtype=motion.dtype)

        if uncond:
            motion = torch.zeros_like(motion)
            keyframe_mask = torch.zeros_like(keyframe_mask)
            text_cond = self.null_text.to(device=motion.device, dtype=motion.dtype).repeat(b_size, 1)
        else:
            text_cond = self.text_proj(texts.to(motion.dtype))
            if drop_mask is not None and drop_mask.any():
                drop_mask = drop_mask.to(device=motion.device, dtype=torch.bool)
                motion = motion.clone()
                keyframe_mask = keyframe_mask.clone()
                text_cond = text_cond.clone()
                motion[drop_mask] = 0.0
                keyframe_mask[drop_mask] = 0.0
                text_cond[drop_mask] = self.null_text.to(device=motion.device, dtype=motion.dtype)

        timesteps = timesteps.reshape(b_size).to(device=motion.device)
        time_cond = self.time_embedding(self.time_proj(timesteps))
        cond = text_cond + time_cond

        motion_in = torch.cat([motion, keyframe_mask], dim=-1)
        motion_in = self.motion_proj(motion_in)

        pad_len = (self.downsample_factor - nframes % self.downsample_factor) % self.downsample_factor
        if pad_len > 0:
            motion_in = F.pad(motion_in, (0, 0, 0, pad_len))

        motion_in = motion_in.permute(1, 0, 2)
        pred_velocity = self.unet(motion_in, cond).permute(1, 0, 2)

        if pad_len > 0:
            pred_velocity = pred_velocity[:, :nframes]

        motion_mask = self._build_motion_mask(lengths, nframes, motion.device)
        if motion_mask is not None:
            pred_velocity = pred_velocity * motion_mask.to(pred_velocity.dtype)

        return {"pred_velocity": pred_velocity}

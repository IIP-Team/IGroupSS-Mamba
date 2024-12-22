# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba
from models.csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

class Block_Group(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 group_type = None,
                 k_group = None,
                 dim=None,
                 dt_rank = "auto",
                 d_inner = None,
                 d_state = None,
                 d_model = None,
                 ssm_ratio = None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type
        d_inner = int(ssm_ratio * d_model)

        self.fc1 = nn.Linear(dim, 4, bias=True)
        self.fc2 = nn.Linear(4, dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # in proj ============================
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        self.act: nn.Module = act_layer()
        self.forward_conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner, kernel_size=1
        )
        self.conv2d = nn.Conv2d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias=True, kernel_size=(1, 1), **kwargs,
        )
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias = True, kernel_size=(1, 1, 1), ** kwargs,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


        # x proj ============================
        d_inner = int(ssm_ratio * (d_model // 4))
        dt_rank = math.ceil((d_model // 4) / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

    def scan(self, x, scan_type=None, group_type=None, route=None):
        if scan_type == 'Interval':
            x1 = x[:, 0::4, :, :]
            x2 = x[:, 1::4, :, :]
            x3 = x[:, 2::4, :, :]
            x4 = x[:, 3::4, :, :]
            xs1 = x1.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs2 = torch.transpose(x2, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = x3.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = torch.flip(xs3, dims=[-1])
            xs4 = torch.transpose(x4, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs4 = torch.flip(xs4, dims=[-1])
            xs = torch.stack([xs1, xs2, xs3, xs4], dim=1).view(self.B, 4, -1, self.L)
        return xs

    def Interval_Combine(self, vectors):
        num = len(vectors)
        B, H, W, L = vectors[0].shape
        # merged_vector = torch.empty(total_length, dtype=vectors[0].dtype)
        merged_vector = torch.zeros(B, H, W, num*L)
        for j in range(L):
            for i in range(num):
                merged_vector[:,:,:,j*num+i] = vectors[i][:,:,:,j]
        return merged_vector

    def forward(self, x: Tensor, route=None, SelectiveScan = SelectiveScanMamba):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)

        # forward con1d
        if self.group_type == 'Patch':
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)
            x = self.act(x)

        zz = x.mean(dim=2).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(zz))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        if len(x.size()) == 4:
            B, D, H, W = x.shape
            L = H * W
        elif len(x.size()) == 3:
            B, D, L = x.shape
        elif len(x.size()) == 5:
            B, D, T, H, W = x.shape
            L = T * H * W
        self.B = B
        self.L = L
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape

        # scan
        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type, route=route)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if self.seq:
            out_y = []
            for i in range(self.k_group):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if self.scan_type == 'Interval':
            x_mamba1 = rearrange(out_y[:, 0].view(B, -1, W, H), 'b c h w -> b h w c')
            x_mamba2 = rearrange(torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3), 'b c h w -> b h w c')
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            x_mamba3 = rearrange(inv_y[:, 0].view(B, -1, W, H), 'b c h w -> b h w c')
            x_mamba4 = rearrange(torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3), 'b c h w -> b h w c')
            y = self.Interval_Combine([x_mamba1, x_mamba2, x_mamba3, x_mamba4]).cuda()
            y = y * fc_out_2.unsqueeze(1).unsqueeze(1)
            y = self.out_norm(y)

        y = y * z
        out = self.dropout(self.out_proj(y))

        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class Block_SpeGroup(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 k_group = None,
                 dim=None,
                 dt_rank="auto",
                 d_state = None,
                 d_model = None,
                 d_model_spe = None,
                 ssm_ratio = None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.scan_type = scan_type
        d_inner = int(ssm_ratio * d_model)

        self.fc1 = nn.Linear(dim, 4, bias=True)
        self.fc2 = nn.Linear(4, dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # in proj ============================
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        self.act: nn.Module = act_layer()
        self.forward_conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner, kernel_size=1
        )
        self.conv2d = nn.Conv2d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias=True, kernel_size=(1, 1), **kwargs,
        )
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias = True, kernel_size=(1, 1, 1), ** kwargs,
        )

        # # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # x proj ============================
        d_inner = int(ssm_ratio * d_model_spe)
        dt_rank = math.ceil(d_model_spe / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

    def scan(self, x, scan_type=None, group_type=None, route=None):
        if scan_type == 'Interval':
            x1 = x[:, 0::4, :, :].permute(0, 2, 1, 3).contiguous()
            x2 = x[:, 1::4, :, :].permute(0, 2, 1, 3).contiguous()
            x3 = x[:, 2::4, :, :].permute(0, 2, 1, 3).contiguous()
            x4 = x[:, 3::4, :, :].permute(0, 2, 1, 3).contiguous()
            xs1 = x1.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs2 = torch.transpose(x2, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = x3.view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs3 = torch.flip(xs3, dims=[-1])
            xs4 = torch.transpose(x4, dim0=2, dim1=3).contiguous().view(self.B, -1, self.L).view(self.B, 1, -1, self.L)
            xs4 = torch.flip(xs4, dims=[-1])
            xs = torch.stack([xs1, xs2, xs3, xs4], dim=1).view(self.B, 4, -1, self.L)
        return xs

    def Interval_Combine(self, vectors):
        num = len(vectors)
        B, H, W, L = vectors[0].shape
        # merged_vector = torch.empty(total_length, dtype=vectors[0].dtype)
        merged_vector = torch.zeros(B, H, W, num*L)
        for j in range(L):
            for i in range(num):
                merged_vector[:,:,:,j*num+i] = vectors[i][:,:,:,j]
        return merged_vector

    def forward(self, x: Tensor, group_type=None, route=None, SelectiveScan = SelectiveScanMamba):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)

        # forward con1d
        if group_type == 'Patch':
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)
            x = self.act(x)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        zz = x.mean(dim=2).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(zz))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = x.size(1) // 4 * x.size(2)
        self.B = B
        self.L = L

        # scan
        xs = self.scan(x, scan_type=self.scan_type, group_type=group_type, route=route)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if self.seq:
            out_y = []
            for i in range(self.k_group):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if self.scan_type == 'Interval':
            x_mamba1 = out_y[:, 0]  #[64, 11, 88]
            x_mamba1 = x_mamba1.transpose(dim0=1, dim1=2).contiguous().view(B, W, -1, H).permute(0, 3, 1, 2)
            x_mamba2 = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            x_mamba2 = x_mamba2.transpose(dim0=1, dim1=2).contiguous().view(B, W, -1, H).permute(0, 3, 1, 2)
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            x_mamba3 = inv_y[:, 0]
            x_mamba3 = x_mamba3.transpose(dim0=1, dim1=2).contiguous().view(B, W, -1, H).permute(0, 3, 1, 2)
            x_mamba4 = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            x_mamba4 = x_mamba4.transpose(dim0=1, dim1=2).contiguous().view(B, W, -1, H).permute(0, 3, 1, 2)
            y = self.Interval_Combine([x_mamba1, x_mamba2, x_mamba3, x_mamba4]).cuda()
            y = y * fc_out_2.unsqueeze(1).unsqueeze(1)
            y = self.out_norm(y)

        y = y * z
        out = self.dropout(self.out_proj(y))

        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        return self.mlp(x)

class VisionMamba(nn.Module):
    def __init__(
            self,
            group_type=None,
            k_group=None,
            depth=None,
            embed_dim=None,
            embed_dims_spe=None,
            dt_rank: int = None,
            d_inner: int = None,
            d_state: int = None,
            ssm_ratio: int = None,
            num_classes: int = None,
            drop_rate=0.,
            drop_path_rate=0.1,
            fused_add_norm=False,
            residual_in_fp32=True,
            bimamba=True,
            # video
            fc_drop_rate=0.,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            Pos_Cls = False,
            scan_type=None,
            route =None,
            pos: str = None,
            cls: str = None,
            spa_downks=None,
            conv3D_channel: int = None,
            conv3D_kernel: int = None,
            dim_patch: int = None,
            dim_linear: int = None,
            **kwargs,
        ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.Pos_Cls = Pos_Cls
        self.scan_type = scan_type
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.k_group = k_group
        self.group_type = group_type
        self.route = route
        self.spa_downks = spa_downks
        self.depth = depth

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )

        self.embedding_spatial_spectral = nn.Sequential(nn.Linear(conv3D_channel, embed_dim))
        self.embedding_spatial = nn.Sequential(nn.Linear(conv3D_channel * dim_linear, embed_dim))
        self.embedding_spectral = nn.Sequential(nn.Linear(dim_patch * dim_patch, embed_dim))

        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1792 + 1, self.embed_dim))
        # self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 28, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.layers = nn.ModuleList([Block_Group(
                    scan_type=scan_type,
                    group_type=group_type,
                    k_group=k_group,
                    dim=embed_dim,
                    d_state=d_state,
                    d_model=embed_dim,
                    ssm_ratio=ssm_ratio,
                    bimamba=bimamba,
                    **kwargs,)
                for i in range(depth)])

        self.layers_spe = nn.ModuleList([Block_SpeGroup(
            scan_type=scan_type,
            k_group=k_group,
            dim=embed_dim,
            d_state=d_state,
            d_model=embed_dim,
            d_model_spe=embed_dims_spe[i],
            ssm_ratio=ssm_ratio,
            bimamba=bimamba,
            **kwargs, )
            for i in range(depth)])

        self.FFNs = nn.ModuleList([MLP_Block(
            in_features=embed_dim,
            hidden_features=embed_dim * 1,)
            for i in range(depth)])


    def get_num_layers(self):
        return len(self.layers)

    def scan(self, x, scan_type=None, group_type=None):
        x = rearrange(x, 'b c t h w -> b (c t) h w')
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.embedding_spatial(x)
        return x

    def Downsample(self, p):
        p = rearrange(p, 'b h w c -> b c h w')  # [64, 5, 5, 64]
        B, C, H, W = p.shape  # [64, 64, 6, 6]
        size_patch = ((H - self.spa_downks[0]) // self.spa_downks[1]) + 1
        x = torch.zeros(p.shape[0], C, size_patch, size_patch).cuda()  # [64, 3, 3, 64]
        for i in range(0, size_patch):
            for j in range(0, size_patch):
                temp = p[:, :, i * self.spa_downks[1]: i * self.spa_downks[1] + self.spa_downks[0],
                       j * self.spa_downks[1]: j * self.spa_downks[1] + self.spa_downks[0]]  # [64, 4, 4, 64]
                temp = temp.mean(dim=[2, 3])  # [64, 64]
                x[:, :, i, j] = temp
        x = rearrange(x, 'b c h w -> b h w c')  # [64, 5, 5, 64]
        # x = self.proj(x)  # 加入后性能有所降低
        return x

    def forward_features(self, x, inference_params=None):
        x = self.conv3d_features(x)
        #scan
        x = self.scan(x, scan_type=self.scan_type, group_type = self.group_type)
        x = self.pos_drop(x)

        # mamba impl
        for i in range(self.depth):
            x = x + self.drop_path(self.layers[i](self.norm(x)))
            x = x + self.drop_path(self.layers_spe[i](self.norm(x), group_type='Patch'))
            x = x + self.drop_path(self.FFNs[i](self.norm(x)))
            if i != self.depth-1:
                x = self.Downsample(x)

        return self.flatten(self.avgpool(x.permute(0, 3, 1, 2)))

    def forward(self, x, inference_params=None):
        feature = self.forward_features(x, inference_params)
        x = self.head(self.head_drop(feature))
        return x, feature

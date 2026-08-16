from typing import Tuple, List
from torch import Tensor
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from pytorch_wavelets import DWTForward
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from basicsr.utils.registry import ARCH_REGISTRY



######################
# Meta Architecture
######################
@ARCH_REGISTRY.register()
class HFDMMamba(nn.Module):
    def __init__(self,
                 scale: int = 4,
                 in_chans: int = 3,
                 num_layers: int = 6,
                 embedding_dim: int = 64,
                 img_range: float = 1.0,
                 use_shuffle: bool = False,
                 recursive: int = 2,
                 attn_drop_rate: float = 0,
                 d_state: int = 16):
        super().__init__()
        self.scale = scale
        self.num_in_channels = in_chans
        self.num_out_channels = in_chans
        self.img_range = img_range
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        
        
        # -- SHALLOW FEATURES --
        self.conv_1 = nn.Conv2d(self.num_in_channels, embedding_dim, kernel_size=3, padding=1)
        
        # -- DEEP FEATURES --
        self.body = nn.ModuleList(
            [ResGroup(in_ch=embedding_dim,
                       use_shuffle=use_shuffle,
                       recursive=recursive,
                       attn_drop_rate=attn_drop_rate,
                       d_state=d_state) for i in range(num_layers)]
        )
        
        # -- UPSCALE --
        self.norm = LayerNorm(embedding_dim, data_format='channels_first')
        self.conv_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embedding_dim, (scale**2) * self.num_out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        # -- SHALLOW FEATURES --
        x = self.conv_1(x)
        res = x
        
        # -- DEEP FEATURES --
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.norm(x)
                
        # -- HR IMAGE RECONSTRUCTION --
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x
    
    
    
#############################
# Components
#############################    
class ResGroup(nn.Module):
    def __init__(self,
                 in_ch: int,
                 recursive: int = 2,
                 use_shuffle: bool = False,
                 attn_drop_rate: float = 0,
                 d_state: int = 16):
        super().__init__()
        
        self.local_block = RME(in_ch=in_ch,
                               use_shuffle=use_shuffle,
                               recursive=recursive)
        self.global_block = SME(in_ch=in_ch,
                                d_state=d_state,
                                attn_drop_rate=attn_drop_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_block(x)
        x = self.local_block(x)
        return x



#############################
# Global Block
#############################
class HFM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super().__init__()
        self.K = reduction_ratio      
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.K, stride=self.K),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=self.K, mode='bilinear', align_corners=True),
        )    
    def forward(self, x):   
        downsampled = self.downsample(x)
        upsampled = self.upsample(downsampled)
        high_freq = x - upsampled
        return high_freq


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        # 1. 高频提取
        self.hfm = HFM(d_model, reduction_ratio=2)
        
        # 2. 抑制性门控层
        self.hf_norm = nn.LayerNorm(d_model)
        self.hf_gate = nn.Linear(d_model, self.d_inner)
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        nn.init.constant_(self.hf_gate.weight, 0)
        nn.init.constant_(self.hf_gate.bias, 0)

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
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, dt_bias: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
    
        x_hw = x.reshape(B, C, L)
        x_wh = x.transpose(2, 3).contiguous().reshape(B, C, L)
    
        dt_hw = dt_bias.transpose(1, 2).reshape(B, C, L)
        dt_wh = (
            dt_hw.reshape(B, C, H, W)
            .transpose(2, 3)
            .contiguous()
            .reshape(B, C, L)
        )
    
        # 保持原来的结果累加顺序：0、2、1、3
        directions = (
            (0, x_hw, dt_hw, False, False),
            (2, x_hw, dt_hw, True,  False),
            (1, x_wh, dt_wh, False, True),
            (3, x_wh, dt_wh, True,  True),
        )
    
        y = None
    
        for k, x_k, dt_k, reverse, transpose_back in directions:
            if reverse:
                x_k = torch.flip(x_k, dims=[-1])
                dt_k = torch.flip(dt_k, dims=[-1])
    
            x_dbl = torch.einsum(
                "b d l, c d -> b c l",
                x_k,
                self.x_proj_weight[k],
            )
    
            dts, Bs, Cs = torch.split(
                x_dbl,
                [self.dt_rank, self.d_state, self.d_state],
                dim=1,
            )
    
            dts = torch.einsum(
                "b r l, d r -> b d l",
                dts,
                self.dt_projs_weight[k],
            )
    
            dts = dts + dt_k
    
            start = k * self.d_inner
            end = (k + 1) * self.d_inner
    
            As = -torch.exp(
                self.A_logs[start:end].float()
            ).reshape(-1, self.d_state)
    
            Ds = self.Ds[start:end].float().reshape(-1)
    
            delta_bias = self.dt_projs_bias[k].float().reshape(-1)
    
            y_k = self.selective_scan(
                x_k.float(),
                dts.contiguous().float(),
                As,
                Bs.float(),
                Cs.float(),
                Ds,
                z=None,
                delta_bias=delta_bias,
                delta_softplus=True,
                return_last_state=False,
            )
    
            if reverse:
                y_k = torch.flip(y_k, dims=[-1])
    
            if transpose_back:
                y_k = (
                    y_k.reshape(B, C, W, H)
                    .transpose(2, 3)
                    .contiguous()
                    .reshape(B, C, L)
                )
    
            if y is None:
                y = y_k
            else:
                y = y + y_k
    
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        
        # 1. 高频特征提取
        hf_seq = self.hfm(x).flatten(2).transpose(1, 2)
        
        # 2. 原始输入投影
        x_flat = x.flatten(2).transpose(1, 2) # (B, L, d_model)
        xz = self.in_proj(x_flat)
        x_inner, z = xz.chunk(2, dim=-1) # (B, L, d_inner)
        
        dt_bias = self.hf_gate(self.hf_norm(hf_seq))

        # 4. 局部卷积支路
        x_conv = x_inner.transpose(1, 2).contiguous().view(B, -1, H, W)
        x_conv = self.act(self.conv2d(x_conv))
        
        # 5. 核心扫描 (传入调制后的特征)
        y = self.forward_core(x_conv, dt_bias)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, -1, H, W)
        return out

class SME(nn.Module):
    def __init__(self,
                in_ch: int,
                attn_drop_rate: float = 0,
                d_state: int = 16,
                expand: float = 2.,
                **kwargs):
        super().__init__()
        
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = SS2D(d_model=in_ch, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
    
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x
     
#############################
# Local Blocks
#############################
class RME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 recursive: int = 2,
                 use_shuffle: bool = False,):
        super().__init__()
        
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = MoEBlock(in_ch=in_ch, use_shuffle=use_shuffle, recursive=recursive)
        
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x



#################
# MoE Layer
#################
class MoEBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 use_shuffle: bool = False,
                 num_heads: int = 3,
                 recursive: int = 2):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive
        self.num_heads = num_heads
        self.head_dim = in_ch // num_heads
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2*in_ch, kernel_size=1, padding=0)
        )
        
        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
            nn.GELU())
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            # StripedConv2d(in_ch, kernel_size=5, depthwise=True),
            nn.GELU())
        
        self.k_proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, groups=num_heads),
            nn.Sigmoid()
        )
        
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        
    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x
        
        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.k_proj(res + x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        
        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)
        
        x = self.conv_2(x)
        k = self.calibrate(k)
        
        x = x * k
        x = self.proj(x)
        return x 
    

#################
# Utilities
#################
class StripedConv2d(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int,
                 depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, self.kernel_size), padding=(0, self.padding), groups=in_ch if depthwise else 1),
            nn.Conv2d(in_ch, in_ch, kernel_size=(self.kernel_size, 1), padding=(self.padding, 0), groups=in_ch if depthwise else 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
    
    
def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class GatedFFN(nn.Module):
    def __init__(self, 
                 in_ch,
                 mlp_ratio,
                 kernel_size,
                 act_layer,):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio
        
        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )
        
        self.gate = nn.Conv2d(mlp_ch // 2, mlp_ch // 2, 
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x
    
    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)
        
        gate = self.gate(gate)
        x = x * gate
        
        x = self.fn_2(x)
        return x
    
    
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

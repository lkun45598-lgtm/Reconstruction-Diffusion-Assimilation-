
from typing import Final, Optional, Type 
'''
Final: 
    表示一个常数不能被修改 TIMEOUT: Final[int] = 30
Optional:
    attn_mask: Optional[torch.Tensor] = None
Type:
    MyClass vs Type[MyClass] 示例 vs 类本身
'''
import torch
from torch import nn as nn
from torch.nn import functional as F
from .init import trunc_normal_
'''
.init 里面放置了很多用于初始化的函数， 防止Transformer类模型在训练初期不稳定
    way:
        在深度学习中，尤其是 Transformer 和 Vision Transformer (ViT) 架构中，是最核心、最常用的权重初始化方法之一。
        它的主要作用是：在模型刚建立时，给参数（权重）赋一个“合理的、不极端”的初始值。
        普通的正态分布4. 为什么不用普通的 nn.init.normal_？
        普通的正态分布初始化（无截断）有“长尾效应”，即有概率产生极端值。
        截断正态分布：就像是一个“被修剪过的盆栽”，它保留了正态分布在中心区域的统计特性，但强行去掉了边缘的不确定性。
to_2tuple:
    在处理图像任务时，很多参数在**对称（长宽相等）和非对称（长宽不等）**的情况下逻辑是一致的，
    但底层算子（如卷积、池化）通常要求明确的 (H, W)。
    在HW相等的情况下也防止作者偷懒。
'''
# hyper parameter for Attention， some GPU can use to accelerate Attn Speed.
USE_FUSED_ATTN = True

'''
score = QK = Weight of V. 这里决定了哪些区域（token/patch）是关注的， 那些区域要被mask掉不去关注。
vanila attn is BND 
BNN 中间N是token序号， 最后一个N是weight
BND 和V求完了之后的融合结果。
'''
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    '''
    place not care about; usually not used in diffsion secario.
    Args:
        scores: Tensor of shape (B, nhead, N, N)
        attn_mask: Optional Tensor of shape (B, nhead, N, N) inf value
    '''
    return scores if attn_mask is None else scores + attn_mask


# --------- Attn in timm.models.vision_transformer
class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = USE_FUSED_ATTN

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ''' 
        Args:
            x: BND shape tensor. 
            attn_mask: BND shape tensor with 0/inf filled. inf means the places that can't be seen by the chosen element.
        '''
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
# --------- Attn in Swin-Transformer
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads=4, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        '''
        goal:
            Window based multi-head self attention (W-MSA) module with relative position bias.
        init:
            dim (int): Transformer model space dimension.
            window_size (tuple[int]): The height and width of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to gen query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set; softmax(QKt/sqrt(d)) *v
            attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
            proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        input:
            x: input features with shape of (num_windows*B, N, C)
                - Note: x doesn't need to add abs pos encoding 
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None, used for shifted windows.
        '''
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, D = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


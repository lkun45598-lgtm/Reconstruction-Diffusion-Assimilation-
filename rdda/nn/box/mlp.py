import torch.nn as nn
from typing import Optional, Type, Union, Tuple
from .tool import to_2tuple
from functools import partial


class Mlp(nn.Module):
    ''' Mlp in timms model '''
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Optional[Type[nn.Module]] = None,
            bias: Union[bool, Tuple[bool, bool]] = True,
            drop: Union[float, Tuple[float, float]] = 0.,
            use_conv: bool = False,
            device=None,
            dtype=None,
    ):
        ''' fc act norm drop fc drop
        Args:
            hidden_feature: 一般是变大
            norm_layer: nn.GroupNorm,nn.LayerNorm
                - nn.GroupNorm(num_groups=8, num_channels=64) But这个用不了, 下面Norm的方式写死了
                - nn.LayerNorm(hidden_features) 必须同时给出，提高计算效率
            bias:  Conv2d里面怎么用bias-卷机操作里也有这个bias项
            use_conv: 是否用Conv2d代替Linear,参数更少
        Inputs:
            x: BCHW/BD depends on use_conv
        '''
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features, **dd) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1], **dd)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

''' 
放卷积相关的类以及Block

nn.Conv2d param:
    cin: 输入通道数
    cout: 输出通道数
    kernel_size: 3
    stride: 1
    padding: 1 

    If only change Channel, HW remains the same.
        3,1,1 or 1,1,0 or 5,1,2, 7,1,3
        如果不改变HW, stride=1 
'''

import torch 
import torch.nn as nn 
from .drop import DropBlock

KERNEL_SIZE = 3

class InitConv(nn.Module):
    def __init__(self, c_in, c_out):
        '''  
        goal:
            原始输入x0的通道数 到UNet顶层通道数， HW不变
        init:
            c_in: 输入通道数
            c_out: 输出通道数 3代表scale shift gate
        input:
            x0: Bc_inHW
        output:
            x: Bc_outHW
        '''
        super().__init__()
        self.init_conv = nn.Conv2d(c_in,c_out,[3,3],[1,1],[1,1])
    def forward(self,x0):
        return self.init_conv(x0)


class LastConv(nn.Module):
    def __init__(self, c_in, c_out):
        ''' 
        goal:
            最后一个卷积层，用于调整C到原始输入的通道数, 但是HW不变
        init:
            c_in: 输入通道数
            c_out: 输出通道数
        input:
            x: Bc_inHW
        output:
            x: Bc_outHW
        '''
        super().__init__()
        self.last_conv = nn.Conv2d(c_in,c_out,1)
    def forward(self,x):
        return self.last_conv(x)
    

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer 
    Paper: Squeeze-and-Excitation Networks
    Helpful for channel-channel interaction, proved useful in many CNN network.
    """
    def __init__(self, C, reduction=16):
        '''
        Args:
            C: 输入通道数
            reduction: 降维比例
        '''
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # CHW->C11 average pool
        self.fc = nn.Sequential(
            nn.Linear(C, max(C // reduction, 4), bias=False), # 降维
            nn.ReLU(inplace=True),  # inplace 操作可以节省内存，直接在原始数据上进行修改，而不创建新的副本来存储结果。
            nn.Linear(max(C // reduction, 4), C, bias=False), # 升维
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        
        y = self.avg_pool(x).view(b, c) # (B, C, 1, 1) -> (B, C)
        
        y = self.fc(y).view(b, c, 1, 1) # (B, C) -> (B, C, 1, 1)
        y = self.sigmoid(y) # 生成 0-1 之间的通道权重
        
        return x * y
    

class CConv2d(nn.Module):
    ''' Gircular Convolution 2d '''
    def __init__(self, in_channels, out_channels, kernel_size, strides=[1, 1], padding=0, dilation=1, groups=1, bias=True):
        super(CConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.strides = strides
        self.kernel_size = kernel_size
    def forward(self, x):
        in_width = x.shape[-1]
        if (in_width % self.strides[1] == 0):
            pad_along_width = max(self.kernel_size[1] - self.strides[1], 0)
        else:
            pad_along_width = max(self.kernel_size[1] - (in_width %self.strides[1]), 0)
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        if pad_left > 0 or pad_right > 0:
            left_pad = x[..., -pad_left:] if pad_left > 0 else torch.empty_like(x[..., :0])
            right_pad = x[..., :pad_right] if pad_right > 0 else torch.empty_like(x[..., :0])
            # print([left_pad.shape,right_pad.shape])
            x_padded = torch.cat([left_pad, x, right_pad], dim=-1)
        return self.conv(x_padded)
    

class CConvTranspose2d(nn.Module):
    ''' Gircular Convolution Transpose 2d '''
    def __init__(self, in_channels, out_channels, kernel_size, strides=[1,1], padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super(CConvTranspose2d, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=strides, padding=padding, output_padding=output_padding,dilation=dilation, groups=groups, bias=bias)
        self.kernel_size = kernel_size
        self.stride = strides
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
    
    def forward(self, x):
        pad_w = self.calculate_circular_padding(self.kernel_size[1], self.stride[1], self.dilation[1])
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_left > 0 or pad_right > 0:
            left_pad = x[..., -pad_left:] if pad_left > 0 else torch.empty_like(x[..., :0])
            right_pad = x[..., :pad_right] if pad_right > 0 else torch.empty_like(x[..., :0])
            # print([left_pad.shape,right_pad.shape])
            x_padded = torch.cat([left_pad, x, right_pad], dim=-1)
        output = self.conv_transpose(x_padded)
        output = self.trim_output(output, x.shape)
        return output
    
    def calculate_circular_padding(self, kernel_size, stride, dilation):
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        return max(0, effective_kernel_size - stride)
    
    def trim_output(self, output, input_shape):
        input_w = input_shape[3]
        output_w = (input_w - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
        current_w = output.shape[3]
        trim_w = max(0, current_w - output_w)
        trim_left = trim_w // 2
        trim_right = trim_w - trim_left
        if trim_left > 0 or trim_right > 0:
            w_start = trim_left
            w_end = current_w - trim_right if trim_right > 0 else current_w
            output = output[:, :, :, w_start:w_end]
        return output
    
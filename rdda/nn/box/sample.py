from torch import nn


class Upsample(nn.Module):
    def __init__(self, c_in, c_out, method='bilinear'):
        """上采样模块(HW)
        [B, c_in, H, W] -> [B, c_out, H*2, W*2]
        其中C的变化往往使用Conv2d, HW可以插值或者PixelShuffle
        
        method:
            - 'transposed': 转置卷积 (ConvTranspose)
                - 场景: 早期 GAN (如 DCGAN)。
                - 特点: 参数可学习，但容易产生"棋盘格伪影" (Checkerboard Artifacts)。
            
            - 'bilinear': 双线性插值 + 卷积 (Resize + Conv)
                - 场景: 语义分割 (U-Net, DeepLab), 特征金字塔 (FPN)。
                - 特点: 现代主流选择。先插值放大会让特征更平滑，避免了棋盘格效应。
            
            - 'nearest': 最近邻插值 + 卷积 (Nearest + Conv)
                - 场景: 追求极致速度，或处理分类掩码 (Mask) 时避免引入不存在的中间值。
                - 特点: 边缘尖锐，计算量最小。
                
            - 'pixelshuffle': 亚像素卷积 (PixelShuffle)
                - 场景: 超分辨率 (Super-Resolution, 如 ESPCN, SRGAN)。
                - 特点: 通过多通道重排实现放大，细节保留最好，但计算量主要在低分辨率层。
                - nn.PixelShuffle: 像素 shuffle 操作，将通道数放大 4 倍，再通过 1x1 卷积恢复通道数。
        """
        super().__init__()
        self.method = method
        
        if method == 'transposed':
            # 为了兼容旧 checkpoint，直接存储 ConvTranspose2d，不使用 .up 包装
            self.conv_transpose = nn.ConvTranspose2d(
                c_in, c_out, 
                kernel_size=4, 
                stride=2, 
                padding=1
            )
            
        elif method == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
            
        elif method == 'nearest':
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
            
        elif method == 'pixelshuffle':
            self.conv = nn.Conv2d(c_in, c_out * 4, kernel_size=3, stride=1, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
            
        else:
            raise ValueError(f"不支持的上采样方法: {method}，"
                           f"可选: 'transposed', 'bilinear', 'nearest', 'pixelshuffle'")
    
    def forward(self, x):
        if self.method == 'transposed':
            return self.conv_transpose(x)
        elif self.method in ['bilinear', 'nearest']:
            x = self.upsample(x)
            return self.conv(x)
        elif self.method == 'pixelshuffle':
            x = self.conv(x)
            return self.pixel_shuffle(x)


class Downsample(nn.Module):
    
    def __init__(self, c_in, c_out, method='conv'):
        """下采样模块(HW) [B, c_in, H, W] -> [B, c_out, H/2, W/2]
            - If Transformer-based, tensor must be processed to 'BCHW' outside.
        
        method:
            - 'conv': 步长卷积 (Strided Convolution)
                - 场景: 现代 CNN 主干 (ResNet, YOLO, ConvNeXt)。
                - 特点: 网络自己学习如何丢弃信息，通常优于固定池化。
                
            - 'maxpool': 最大池化 (Max Pooling)
                - 场景: 强调纹理、边缘特征 (VGG, Early Inception)。
                - 特点: 具有平移不变性，提取局部最显著特征。
                
            - 'avgpool': 平均池化 (Average Pooling)
                - 场景: 强调背景、平滑信息，或用于防混叠 (Anti-aliasing)。
                - 特点: 平滑特征图，融合局部信息。
                
            - 'pixelunshuffle': 像素反洗牌 (PixelUnshuffle / Space-to-Depth)
                - 场景: Vision Transformer (Swin), 可逆网络, MobileNet变体。
                - 特点: 无损下采样。将空间信息"折叠"到通道中，完全保留原始像素信息，不丢失数据。

            - 'patchmerge':
                - 场景: Swin Transformer 源码的下采样方法
                - 特点: 利用窗口注意力机制，有效处理长距离依赖。
        """
        super().__init__()
        self.method = method
        
        if method == 'conv':
            # 为了兼容旧 checkpoint，直接存储 Conv2d，不使用 .down 包装
            self.conv = nn.Conv2d(
                c_in, c_out, 
                kernel_size=3, 
                stride=2, 
                padding=1
            )
            
        elif method == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
            
        elif method == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
            
        elif method == 'pixelunshuffle':
            self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
            self.conv = nn.Conv2d(c_in * 4, c_out, kernel_size=3, stride=1, padding=1)

        elif method == 'patchmerge':
           # 这个方法组织像素的方式和nn.PixelUnshuffle完全一致，只是通道数变化方式改为MlP
           self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
           self.norm = nn.LayerNorm(c_in * 4)
           # 这里的 bias=False是因为刚LayerNorm归一化
           self.linear = nn.Linear(c_in * 4, c_out, bias=False)  
        else:
            raise ValueError(f"不支持下采样方法: {method}，"
                           f"可选: 'conv', 'maxpool', 'avgpool', 'pixelunshuffle', 'patchmerge'")
    
    def forward(self, x):
        assert x.ndim==4, 'Only BCHW tensor accepted'
        if self.method == 'conv':
            return self.conv(x)
        elif self.method in ['maxpool', 'avgpool']:
            x = self.pool(x)
            return self.conv(x)
        elif self.method == 'pixelunshuffle':
            x = self.pixel_unshuffle(x)
            return self.conv(x)
        elif self.method == 'patchmerge':
            x = self.pixel_unshuffle(x)
            # x: [B, C*4, H/2, W/2] -> [B, H/2, W/2, C*4]
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.linear(x)
            # x: [B, H/2, W/2, C_out] -> [B, C_out, H/2, W/2]
            return x.permute(0, 3, 1, 2)

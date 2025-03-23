import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class UNet(nn.Module):
    """
    Enhanced U-Net for cityscapes segmentation with ASPP (including global context)
    and improved upsampling.
    
    Args:
        in_channels (int): Number of channels in the input image.
        n_classes (int): Number of segmentation classes.
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 19) -> None:
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.aspp = ASPP(1024, 1024, use_global=True)
        self.up1 = Up(1024, 512, 256)
        self.up2 = Up(256, 256, 128)
        self.up3 = Up(128, 128, 64)
        self.up4 = Up(64, 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.aspp(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """
    Applies two consecutive convolutions, each followed by batch normalization,
    ReLU activation, and dropout for regularization.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (Optional[int]): Number of channels after the first convolution.
                                      Defaults to out_channels if not provided.
        dilation (int): Dilation rate for convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 mid_channels: Optional[int] = None, dilation: int = 1) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscales the input feature map using max pooling followed by a DoubleConv.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dilation (int): Dilation rate for the convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscales the input feature map, concatenates with the corresponding skip connection,
    and applies a DoubleConv.
    
    Args:
        prev_channels (int): Number of channels from the previous (deeper) layer.
        skip_channels (int): Number of channels from the skip connection.
        out_channels (int): Number of output channels after upsampling.
    """
    def __init__(self, prev_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(prev_channels, prev_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(prev_channels + skip_channels, out_channels, mid_channels=(prev_channels + skip_channels) // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Ensure x1 and x2 are spatially aligned
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module for multi-scale context aggregation.
    Optionally includes a global context branch.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for each branch before concatenation.
        use_global (bool): Whether to include a global average pooling branch.
    """
    def __init__(self, in_channels: int, out_channels: int, use_global: bool = False) -> None:
        super(ASPP, self).__init__()
        self.use_global = use_global
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        
        # If using global context, add an image pooling branch
        if self.use_global:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        
        # The number of branches concatenated:
        num_branches = 4 if self.use_global else 3
        self.bn = nn.BatchNorm2d(out_channels * num_branches)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(out_channels * num_branches, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        branches = [x1, x2, x3]
        
        if self.use_global:
            x_global = self.global_avg_pool(x)
            # Upsample global context to match spatial dimensions of the other branches
            x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=True)
            branches.append(x_global)
        
        x = torch.cat(branches, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x

class OutConv(nn.Module):
    """
    Final output convolution block.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (typically number of classes).
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

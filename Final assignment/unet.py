import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class UNet(nn.Module):
    """
    Widened U-Net for cityscapes segmentation with ASPP.
    New channel sizes: 96 -> 192 -> 384 -> 768 -> 1536
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 19) -> None:
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 96)    # Was 64
        self.down1 = Down(96, 192)               # Was 128
        self.down2 = Down(192, 384)              # Was 256
        self.down3 = Down(384, 768)              # Was 512
        self.down4 = Down(768, 1536)             # Was 1024
        self.aspp = ASPP(1536, 1536, use_global=True)  # Updated to match
        self.up1 = Up(1536, 768, 384)            # Was 1024, 512, 256
        self.up2 = Up(384, 384, 192)             # Was 256, 256, 128
        self.up3 = Up(192, 192, 96)              # Was 128, 128, 64
        self.up4 = Up(96, 96, 96)                # Was 64, 64, 64
        self.outc = OutConv(96, n_classes)

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

# No changes to DoubleConv
class DoubleConv(nn.Module):
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

# No changes to Down
class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

# Update Up to match new channel sizes
class Up(nn.Module):
    def __init__(self, prev_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(prev_channels, prev_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(prev_channels + skip_channels, out_channels, 
                              mid_channels=(prev_channels + skip_channels) // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Update ASPP to match widened bottleneck
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_global: bool = False) -> None:
        super(ASPP, self).__init__()
        self.use_global = use_global
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        
        if self.use_global:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        
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
            x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=True)
            branches.append(x_global)
        
        x = torch.cat(branches, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x

# Update OutConv to match final channel size
class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=34):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 96)
        self.down1 = Down(96, 192)
        self.down2 = Down(192, 384)
        self.down3 = Down(384, 768)
        self.down4 = Down(768, 1280)
        self.aspp = ASPP(1280, 1280)
        self.up1 = Up(1280, 768, 384)
        self.up2 = Up(384, 384, 192)
        self.up3 = Up(192, 192, 96)
        self.up4 = Up(96, 96, 96)
        self.outc = OutConv(96, n_classes)
    
    def forward(self, x):
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
    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Keep dropout for regularization
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, prev_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(prev_channels, prev_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(prev_channels + skip_channels, out_channels, (prev_channels + skip_channels) // 2)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    def forward(self, x):
        return self.conv(x)
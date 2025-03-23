import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=19, features_start=64, use_attention=True):
        """
        Enhanced U-Net with residual connections, dropout, attention gates, and deep supervision
        
        Args:
            in_channels: Number of input channels (e.g. 3 for RGB)
            n_classes: Number of output classes
            features_start: Number of features in first layer
            use_attention: Whether to use attention gates
        """
        super(EnhancedUNet, self).__init__()
        self.use_attention = use_attention
        
        # Encoder (downsampling path)
        self.encoder1 = ResBlock(in_channels, features_start)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = ResBlock(features_start, features_start * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = ResBlock(features_start * 2, features_start * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = ResBlock(features_start * 4, features_start * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge
        self.bridge = ASPP(features_start * 8, features_start * 16)
        
        # Decoder (upsampling path)
        if use_attention:
            self.attention1 = AttentionGate(features_start * 8, features_start * 16, features_start * 8)
            self.attention2 = AttentionGate(features_start * 4, features_start * 8, features_start * 4)
            self.attention3 = AttentionGate(features_start * 2, features_start * 4, features_start * 2)
            self.attention4 = AttentionGate(features_start, features_start * 2, features_start)
        
        self.decoder1 = UpBlock(features_start * 16, features_start * 8)
        self.decoder2 = UpBlock(features_start * 8, features_start * 4)
        self.decoder3 = UpBlock(features_start * 4, features_start * 2)
        self.decoder4 = UpBlock(features_start * 2, features_start)
        
        # Deep supervision outputs
        self.dsv1 = nn.Conv2d(features_start * 8, n_classes, kernel_size=1)
        self.dsv2 = nn.Conv2d(features_start * 4, n_classes, kernel_size=1)
        self.dsv3 = nn.Conv2d(features_start * 2, n_classes, kernel_size=1)
        self.dsv4 = nn.Conv2d(features_start, n_classes, kernel_size=1)
        
        # Final output
        self.final = nn.Conv2d(n_classes * 4, n_classes, kernel_size=1)
        
        # SE blocks for channel recalibration
        self.se1 = SEBlock(features_start * 8)
        self.se2 = SEBlock(features_start * 4)
        self.se3 = SEBlock(features_start * 2)
        self.se4 = SEBlock(features_start)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool4(enc4))
        
        # Decoder with skip connections and attention
        if self.use_attention:
            enc4 = self.attention1(enc4, bridge)
        
        dec1 = self.decoder1(bridge, enc4)
        dec1 = self.se1(dec1)
        
        if self.use_attention:
            enc3 = self.attention2(enc3, dec1)
            
        dec2 = self.decoder2(dec1, enc3)
        dec2 = self.se2(dec2)
        
        if self.use_attention:
            enc2 = self.attention3(enc2, dec2)
            
        dec3 = self.decoder3(dec2, enc2)
        dec3 = self.se3(dec3)
        
        if self.use_attention:
            enc1 = self.attention4(enc1, dec3)
            
        dec4 = self.decoder4(dec3, enc1)
        dec4 = self.se4(dec4)
        
        # Deep supervision
        dsv1 = self.dsv1(dec1)
        dsv2 = self.dsv2(dec2)
        dsv3 = self.dsv3(dec3)
        dsv4 = self.dsv4(dec4)
        
        # Upsample to match the input resolution
        size = dsv4.size()[2:]
        dsv1 = F.interpolate(dsv1, size=size, mode='bilinear', align_corners=False)
        dsv2 = F.interpolate(dsv2, size=size, mode='bilinear', align_corners=False)
        dsv3 = F.interpolate(dsv3, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate deep supervision outputs
        out = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        
        # Final convolution
        out = self.final(out)
        
        return out


class ResBlock(nn.Module):
    """Residual block with dropout for regularization"""
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(ResBlock, self).__init__()
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )
        
    def forward(self, x):
        residual = self.residual_conv(x)
        return residual + self.conv_block(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(UpBlock, self).__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        
        self.conv_block = ResBlock(out_channels * 2, out_channels, dropout_p)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle different sizes
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, 
                      diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        
        return self.conv_block(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module for capturing multi-scale context"""
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        rates = [1, 6, 12, 18]
        
        self.aspp = nn.ModuleList()
        for rate in rates:
            if rate == 1:
                conv = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)
            else:
                conv = nn.Conv2d(
                    in_channels, out_channels // 4, kernel_size=3, 
                    padding=rate, dilation=rate, bias=False
                )
            self.aspp.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ))
        
        # Global pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels + out_channels // 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        size = x.size()[2:]
        
        # Process each branch
        outputs = []
        for aspp_branch in self.aspp:
            outputs.append(aspp_branch(x))
        
        # Global average pooling branch
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        outputs.append(global_feat)
        
        # Concatenate branches
        x = torch.cat(outputs, dim=1)
        
        # Final projection
        return self.project(x)


class AttentionGate(nn.Module):
    """Attention Gate to focus on relevant features and suppress noise"""
    def __init__(self, x_channels, g_channels, out_channels):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        # g is the gating signal from the higher level
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Handle different sizes
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Multiply with the attention map
        return x * psi


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)